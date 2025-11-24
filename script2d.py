#!/usr/bin/env python3
"""
Command-line adaptation of rmhdpinn_2d.ipynb.

Usage:
    python script2d.py --data-dir data2d/2dshock --epochs 15000 --log-interval 50

Plots are saved to disk when --plot-interval is set.
"""
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from RMHDEquations2D import flux_x, flux_y, primitives_to_conserved  # noqa: E402
from jacobians import compute_AX, compute_AY, compute_M  # noqa: E402


# --- Globals that depend on loaded data ---
SNAPSHOTS: Dict[float, Dict[str, np.ndarray]] = {}
AVAILABLE_TIMES: np.ndarray = np.array([])
X_DOMAIN: Tuple[float, float] = (0.0, 1.0)
Y_DOMAIN: Tuple[float, float] = (0.0, 1.0)
T_DOMAIN: Tuple[float, float] = (0.0, 0.4)

# --- Constants ---
PRIMITIVE_LABELS = ["rho", "vx", "vy", "vz", "Bx", "By", "Bz", "p"]
RAW_FIELD_KEYS = ["rho", "u1", "u2", "u3", "b1", "b2", "b3", "p"]
DEVICE = torch.device("cpu")  # overwritten in main


def _select_device() -> torch.device:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_snapshots(data_dir: Path) -> Dict[float, Dict[str, np.ndarray]]:
    snapshots: Dict[float, Dict[str, np.ndarray]] = {}
    for path in sorted(data_dir.glob("data2D_t_*.pkl")):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        time_value = float(payload["time"])
        centers = np.asarray(payload["centers"], dtype=np.float64)
        fields = payload["cell_data"]
        primitives = np.stack([np.asarray(fields[key], dtype=np.float64) for key in RAW_FIELD_KEYS], axis=1)
        x_unique = np.unique(centers[:, 0])
        y_unique = np.unique(centers[:, 1])
        nx = x_unique.size
        ny = y_unique.size
        grid = np.empty((ny, nx, primitives.shape[1]), dtype=np.float64)
        ix = np.searchsorted(x_unique, centers[:, 0])
        iy = np.searchsorted(y_unique, centers[:, 1])
        grid[iy, ix] = primitives
        snapshots[time_value] = {
            "x": x_unique,
            "y": y_unique,
            "grid": grid,
            "centers": centers,
        }
    if not snapshots:
        raise RuntimeError(f"No snapshot files found in {data_dir}")
    return snapshots


def _resolve_time_key(t: Union[float, torch.Tensor, np.ndarray]) -> float:
    t_scalar = float(t.detach().cpu().item() if isinstance(t, torch.Tensor) else np.asarray(t).item())
    matches = np.isclose(AVAILABLE_TIMES, t_scalar, atol=1e-12)
    if not np.any(matches):
        raise ValueError(f"Requested time {t_scalar} not found. Available times: {AVAILABLE_TIMES.tolist()}")
    return float(AVAILABLE_TIMES[matches][0])


def _prepare_xy(xy: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, Tuple[int, ...]]:
    arr = xy.detach().cpu().numpy() if isinstance(xy, torch.Tensor) else np.asarray(xy, dtype=np.float64)
    arr = np.atleast_2d(arr).astype(np.float64)
    original_shape = tuple(arr.shape)
    flat = arr.reshape(-1, 2)
    return flat, original_shape


def _bilinear_lookup(snapshot: Dict[str, np.ndarray], xy: np.ndarray) -> np.ndarray:
    x_grid = snapshot["x"]
    y_grid = snapshot["y"]
    field = snapshot["grid"]
    nx = x_grid.size
    ny = y_grid.size
    x = np.clip(xy[:, 0], x_grid[0], x_grid[-1])
    y = np.clip(xy[:, 1], y_grid[0], y_grid[-1])
    ix1 = np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, nx - 2)
    iy1 = np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, ny - 2)
    ix2 = ix1 + 1
    iy2 = iy1 + 1
    x1 = x_grid[ix1]
    x2 = x_grid[ix2]
    y1 = y_grid[iy1]
    y2 = y_grid[iy2]
    tx = np.divide(x - x1, x2 - x1, out=np.zeros_like(x), where=(x2 - x1) != 0)
    ty = np.divide(y - y1, y2 - y1, out=np.zeros_like(y), where=(y2 - y1) != 0)
    c11 = field[iy1, ix1]
    c21 = field[iy1, ix2]
    c12 = field[iy2, ix1]
    c22 = field[iy2, ix2]
    interp = (
        (1 - tx)[:, None] * (1 - ty)[:, None] * c11
        + tx[:, None] * (1 - ty)[:, None] * c21
        + (1 - tx)[:, None] * ty[:, None] * c12
        + tx[:, None] * ty[:, None] * c22
    )
    return interp


def cond(
    t: Union[float, torch.Tensor, np.ndarray],
    xy: Union[np.ndarray, torch.Tensor],
    *,
    device: torch.device | None = None,
    as_tensor: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    xy_flat, original_shape = _prepare_xy(xy)
    if isinstance(t, torch.Tensor):
        t_arr = np.asarray(t.detach().cpu().numpy()).reshape(-1)
    else:
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    if t_arr.size == 1:
        t_arr = np.full(xy_flat.shape[0], float(t_arr[0]), dtype=np.float64)
    if t_arr.size != xy_flat.shape[0]:
        raise ValueError("Time array must be scalar or match the number of query points")
    outputs = np.zeros((xy_flat.shape[0], len(PRIMITIVE_LABELS)), dtype=np.float64)
    for time_value in np.unique(t_arr):
        time_key = _resolve_time_key(time_value)
        mask = np.isclose(t_arr, time_value)
        outputs[mask] = _bilinear_lookup(SNAPSHOTS[time_key], xy_flat[mask])
    if original_shape == ():
        reshaped = outputs.reshape(len(PRIMITIVE_LABELS),)
    else:
        target_shape = original_shape[:-1] if original_shape[-1] == 2 else original_shape
        reshaped = outputs.reshape(target_shape + (len(PRIMITIVE_LABELS),))
    if not as_tensor:
        return reshaped
    target_device = device or DEVICE
    return torch.as_tensor(reshaped, device=target_device, dtype=torch.get_default_dtype())


def plot_reference_profiles(times: Iterable[float] = (0.0, 0.2, 0.4), field: str = "rho", *, output_path: Path | None = None):
    field_idx = PRIMITIVE_LABELS.index(field)
    fig, axes = plt.subplots(1, len(tuple(times)), figsize=(5 * len(tuple(times)), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, times):
        time_key = _resolve_time_key(t)
        snapshot = SNAPSHOTS[time_key]
        data = snapshot["grid"][..., field_idx]
        pcm = ax.pcolormesh(snapshot["x"], snapshot["y"], data, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, shrink=0.8, label=field)
        ax.set_title(f"t={t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"Reference {field} snapshots")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return fig


def _partial_derivative(outputs: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
    if outputs.ndim == 0:
        raise ValueError("outputs must have at least one dimension")
    outputs = outputs.reshape(-1, outputs.shape[-1])
    grads = []
    for component in range(outputs.shape[1]):
        grad = torch.autograd.grad(outputs[:, component].sum(), coords, create_graph=True, retain_graph=True)[0]
        grads.append(grad[:, dim])
    return torch.stack(grads, dim=1)


def rmhd_residual(primitives: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if not coords.requires_grad:
        raise ValueError("coords must require gradients for residual computation.")
    conserved = primitives_to_conserved(primitives, gamma=5 / 3)
    flux_x_term = flux_x(primitives, gamma=5 / 3)
    flux_y_term = flux_y(primitives, gamma=5 / 3)
    dU_dt = _partial_derivative(conserved, coords, dim=2)
    dF_dx = _partial_derivative(flux_x_term, coords, dim=0)
    dG_dy = _partial_derivative(flux_y_term, coords, dim=1)
    return dU_dt + dF_dx + dG_dy


def jacobian_residual(primitives: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if not coords.requires_grad:
        raise ValueError("coords must require gradients for residual computation.")
    M = compute_M(primitives)
    AX = compute_AX(primitives)
    AY = compute_AY(primitives)
    dP_dx = _partial_derivative(primitives, coords, dim=0).unsqueeze(-1)
    dP_dy = _partial_derivative(primitives, coords, dim=1).unsqueeze(-1)
    dP_dt = _partial_derivative(primitives, coords, dim=2).unsqueeze(-1)
    return (M @ dP_dt + AX @ dP_dx + AY @ dP_dy).squeeze(-1)


class TrainableTanh(nn.Module):
    def __init__(self, init_gain: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.log_gain = nn.Parameter(torch.tensor(float(math.log(init_gain))))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = torch.exp(self.log_gain)
        return torch.tanh(gain * x + self.bias)


class PINN(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 8, width: int = 16, depth: int = 12, activation=TrainableTanh):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.activation = activation()
        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(width, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        for layer in self.layers:
            x = self.activation(layer(x))
        raw = self.output_layer(x)
        rho = torch.exp(raw[:, 0:1])
        velocities = torch.tanh(raw[:, 1:4])
        magnetic = raw[:, 4:7]
        pressure = torch.exp(raw[:, 7:8])
        return torch.cat([rho, velocities, magnetic, pressure], dim=1)


def sample_domain(n_points: int) -> torch.Tensor:
    x = torch.rand(n_points, 1, device=DEVICE) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]
    y = torch.rand(n_points, 1, device=DEVICE) * (Y_DOMAIN[1] - Y_DOMAIN[0]) + Y_DOMAIN[0]
    t = torch.rand(n_points, 1, device=DEVICE) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]
    coords = torch.cat([x, y, t], dim=1)
    coords.requires_grad_(True)
    return coords


def prepare_boundary_points(n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    per_side = n_points // 4
    remainder = n_points - 4 * per_side
    counts = [per_side] * 4
    counts[0] += remainder

    def rand_y(count):
        return torch.rand(count, 1, device=DEVICE) * (Y_DOMAIN[1] - Y_DOMAIN[0]) + Y_DOMAIN[0]

    def rand_x(count):
        return torch.rand(count, 1, device=DEVICE) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]

    x_left = torch.full((counts[0], 1), X_DOMAIN[0], device=DEVICE)
    y_left = rand_y(counts[0])

    x_right = torch.full((counts[1], 1), X_DOMAIN[1], device=DEVICE)
    y_right = rand_y(counts[1])

    x_bottom = rand_x(counts[2])
    y_bottom = torch.full((counts[2], 1), Y_DOMAIN[0], device=DEVICE)

    x_top = rand_x(counts[3])
    y_top = torch.full((counts[3], 1), Y_DOMAIN[1], device=DEVICE)

    x_all = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
    y_all = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
    return x_all, y_all


def log_history(history: Dict[str, list], key: str, value: torch.Tensor, window_size: int = 10):
    history.setdefault(key, []).append(value.detach().item())
    if len(history[key]) >= window_size:
        history[key][-1] = float(np.mean(history[key][-window_size:]))


def normalize_batch(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = targets
    rms = torch.sqrt(torch.mean(diff.pow(2), dim=0, keepdim=True))
    inv_scale = 1.0 / torch.clamp(rms, min=eps)
    return preds * inv_scale, targets * inv_scale, inv_scale


def plot_training_progress(
    history: Dict[str, list],
    model: PINN,
    epoch: int,
    epochs: int,
    *,
    field: str = "rho",
    output_path: Path | None = None,
):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

    ax_loss = fig.add_subplot(gs[0, 0])
    for name, series in history.items():
        if not series:
            continue
        steps = np.arange(1, len(series) + 1)
        values = np.clip(np.asarray(series, dtype=np.float64), 1e-16, None)
        ax_loss.plot(steps, values, label=name)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss value")
    ax_loss.legend(fontsize="small", ncol=2)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_title("Loss history")

    t_slice = 0.4
    x_vis = torch.linspace(X_DOMAIN[0], X_DOMAIN[1], 160, device=DEVICE)
    y_vis = torch.linspace(Y_DOMAIN[0], Y_DOMAIN[1], 160, device=DEVICE)
    X_mesh, Y_mesh = torch.meshgrid(x_vis, y_vis, indexing="ij")
    coords_vis = torch.stack([X_mesh.reshape(-1), Y_mesh.reshape(-1), torch.full_like(X_mesh.reshape(-1), t_slice)], dim=1)
    field_idx = PRIMITIVE_LABELS.index(field)
    with torch.no_grad():
        field_pred = model(coords_vis)[:, field_idx].reshape(x_vis.numel(), y_vis.numel()).cpu().numpy()
    ax_slice = fig.add_subplot(gs[0, 1])
    pcm = ax_slice.pcolormesh(x_vis.cpu().numpy(), y_vis.cpu().numpy(), field_pred.T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax_slice, shrink=0.8, label=field)
    ax_slice.set_title(f"{field} @ t={t_slice:.2f}")
    ax_slice.set_xlabel("x")
    ax_slice.set_ylabel("y")

    ax_line = fig.add_subplot(gs[1, :])
    y_mid = 0.5 * (Y_DOMAIN[0] + Y_DOMAIN[1])
    x_line = torch.linspace(X_DOMAIN[0], X_DOMAIN[1], 256, device=DEVICE).unsqueeze(-1)
    y_line = torch.full_like(x_line, y_mid)
    xy_line = torch.cat([x_line, y_line], dim=1)
    coords_line = torch.cat([xy_line, torch.full((x_line.shape[0], 1), t_slice, device=DEVICE)], dim=1)
    with torch.no_grad():
        pred_line = model(coords_line).cpu().numpy()
    ref_line = cond(t_slice, xy_line.cpu().numpy(), as_tensor=False)
    rho_idx = PRIMITIVE_LABELS.index("rho")
    x_vals = x_line.cpu().numpy().ravel()
    ax_line.plot(x_vals, pred_line[:, rho_idx], color="tab:blue", linestyle="-", label=r"$\\rho$ (PINN)")
    ax_line.plot(x_vals, ref_line[:, rho_idx], color="tab:orange", linestyle="--", label=r"$\\rho$ (data)")
    ax_line.set_xlabel("x (y = midline)")
    ax_line.set_ylabel("rho")
    ax_line.set_title(f"Lineout comparison at t={t_slice:.2f}")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(fontsize="small", loc="upper right")

    fig.suptitle(f"Epoch {epoch}/{epochs}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


class ScaledMuon(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group.get("lr", 1e-3)
            alpha = group.get("momentum", 0.9)
            role = group.get("role", "hidden")
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state.setdefault(p, {})
                if "momentum_buf" not in state:
                    state["momentum_buf"] = torch.zeros_like(p)
                m = state["momentum_buf"]
                m.mul_(alpha).add_(g, alpha=1.0 - alpha)
                g_tilde = (1.0 - alpha) * g + alpha * m
                if role == "first":
                    m_rows = p.shape[0]
                    row_norms = g_tilde.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    step = (m_rows**0.5) * g_tilde / row_norms
                    p.add_(step, alpha=-lr)
                elif role == "last":
                    n_cols = p.shape[1]
                    col_norms = g_tilde.norm(dim=0, keepdim=True).clamp_min(1e-12)
                    step = (n_cols**-0.5) * g_tilde / col_norms
                    p.add_(step, alpha=-lr)
                elif role == "hidden":
                    U, _, Vh = torch.linalg.svd(g_tilde, full_matrices=False)
                    signm = U @ Vh
                    p.add_(signm, alpha=-lr)
                elif role == "bias":
                    rms = (g_tilde.pow(2).mean()).sqrt().clamp_min(1e-12)
                    p.add_(g_tilde / rms, alpha=-lr)
                elif role == "scalar":
                    p.add_(g_tilde.sign(), alpha=-lr)
                else:
                    raise ValueError(f"Unknown role: {role}")


@dataclass
class TrainingConfig:
    epochs: int = 15000
    n_domain: int = 1024
    n_intermediate: int = 8092
    n_boundary: int = 512
    lambda_domain: float = 0.0001
    lambda_bdy: float = 10.0
    lambda_time: Dict[float, float] | None = None
    grad_clip: float | None = 0.4
    schedule_epoch: int = 1000
    lambda_domain_schedule: List[float] = None
    lrg: float = 1e-3
    lrg_decay: float = 0.5
    log_interval: int = 50
    plot_interval: int = 0
    plot_field: str = "rho"
    plot_dir: Path | None = None
    condition_times: List[float] = None


def build_optimizer(model: PINN, lrg: float) -> ScaledMuon:
    param_groups = []

    def add_group(param, role):
        param_groups.append({"params": [param], "role": role, "lr": lrg, "momentum": 0.9})

    add_group(model.layers[0].weight, "first")
    add_group(model.layers[0].bias, "bias")
    for layer in model.layers[1:]:
        add_group(layer.weight, "hidden")
        add_group(layer.bias, "bias")
    add_group(model.output_layer.weight, "last")
    add_group(model.output_layer.bias, "bias")
    for name, p in model.named_parameters():
        if "activation" in name and p.ndim == 0:
            add_group(p, "scalar")
    return ScaledMuon(param_groups)


def _format_loss_row(latest_losses: Dict[str, float]) -> str:
    return ", ".join(f"{name}: {value:.3e}" for name, value in latest_losses.items())


def train(model: PINN, optimizer: ScaledMuon, config: TrainingConfig):
    history = {"domain": [], "boundary": [], "total": []}
    lambda_time = config.lambda_time or {0.0: 5.0, 0.08: 20.0, 0.2: 20.0}
    for t in sorted(lambda_time):
        history[f"t={t:.3f}"] = []

    n_domain = config.n_domain
    n_intermediate = config.n_intermediate
    lambda_domain = config.lambda_domain
    lrg = config.lrg
    lambda_domain_idx = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        coords_domain = sample_domain(n_domain)
        preds_domain = model(coords_domain)
        residual = jacobian_residual(preds_domain, coords_domain)
        domain_loss = F.mse_loss(residual, torch.zeros_like(residual))
        total_loss = lambda_domain * domain_loss

        x_data = torch.rand(n_intermediate, 1, device=DEVICE) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]
        y_data = torch.rand(n_intermediate, 1, device=DEVICE) * (Y_DOMAIN[1] - Y_DOMAIN[0]) + Y_DOMAIN[0]
        xy_data = torch.cat([x_data, y_data], dim=1)

        data_losses: Dict[str, torch.Tensor] = {}
        for t_val, weight in lambda_time.items():
            t_tensor = torch.full((n_intermediate, 1), t_val, device=DEVICE)
            coords_data = torch.cat([xy_data, t_tensor], dim=1)
            preds_data = model(coords_data)
            targets_data = cond(t_val, xy_data, device=DEVICE)
            data_loss = F.mse_loss(preds_data, targets_data)
            data_losses[f"t={t_val:.3f}"] = data_loss
            total_loss = total_loss + weight * data_loss

        x_b, y_b = prepare_boundary_points(config.n_boundary)
        xy_boundary = torch.cat([x_b, y_b], dim=1)
        t_boundary = torch.rand(config.n_boundary, 1, device=DEVICE) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]
        coords_boundary = torch.cat([xy_boundary, t_boundary], dim=1)

        preds_boundary = model(coords_boundary)
        targets_boundary = cond(0.0, xy_boundary, device=DEVICE)
        boundary_loss = F.mse_loss(preds_boundary, targets_boundary)
        total_loss = total_loss + config.lambda_bdy * boundary_loss

        total_loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if config.schedule_epoch and epoch % config.schedule_epoch == 0:
            n_domain *= 2
            n_intermediate = max(1024, n_intermediate - 256)
            if config.lambda_domain_schedule and lambda_domain_idx < len(config.lambda_domain_schedule):
                lambda_domain = config.lambda_domain_schedule[lambda_domain_idx]
                lambda_domain_idx += 1
            lrg *= config.lrg_decay
            for group in optimizer.param_groups:
                group["lr"] = lrg

        log_history(history, "domain", domain_loss)
        for key, loss_value in data_losses.items():
            log_history(history, key, loss_value)
        log_history(history, "boundary", boundary_loss)
        log_history(history, "total", total_loss)

        if epoch == 1 or (config.log_interval and epoch % config.log_interval == 0):
            latest_losses = {
                "domain": history["domain"][-1],
                "boundary": history["boundary"][-1],
                **{k: history[k][-1] for k in data_losses},
                "total": history["total"][-1],
            }
            print(f"Epoch {epoch}/{config.epochs} - {_format_loss_row(latest_losses)}")

        if config.plot_interval and epoch % config.plot_interval == 0:
            output_path = None
            if config.plot_dir:
                output_path = config.plot_dir / f"training_epoch_{epoch:05d}.png"
            plot_training_progress(history, model, epoch, config.epochs, field=config.plot_field, output_path=output_path)

    return history


def plot_rho_times(model: PINN, times: Iterable[float] = (0.0, 0.2, 0.4), nx: int = 160, ny: int = 160, *, output_path: Path | None = None):
    model.eval()
    x = torch.linspace(X_DOMAIN[0], X_DOMAIN[1], nx, device=DEVICE)
    y = torch.linspace(Y_DOMAIN[0], Y_DOMAIN[1], ny, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, len(tuple(times)), figsize=(5 * len(tuple(times)), 4), sharex=True, sharey=True)
    if len(tuple(times)) == 1:
        axes = np.array([axes])

    with torch.no_grad():
        pcm = None
        for ax, t_val in zip(axes, times):
            T = torch.full_like(X, fill_value=t_val)
            coords = torch.stack([X, Y, T], dim=-1).reshape(-1, 3)
            rho = model(coords)[:, 0].reshape(nx, ny).cpu()
            pcm = ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), rho.numpy(), shading="auto", cmap="viridis")
            ax.set_title(f"t = {t_val:.2f}")
            ax.set_xlabel("x")
        axes[0].set_ylabel("y")

    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cax, label=r"$\\rho$")
    fig.suptitle("Density slices")
    fig.tight_layout(rect=[0, 0, 0.86, 0.98])
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 2D RMHD PINN (converted from rmhdpinn_2d.ipynb).")
    parser.add_argument("--data-dir", type=Path, default=Path("data2d/2dshock"), help="Directory containing data2D_t_*.pkl snapshots.")
    parser.add_argument("--epochs", type=int, default=15000, help="Number of training epochs.")
    parser.add_argument("--width", type=int, default=128, help="Network width.")
    parser.add_argument("--depth", type=int, default=64, help="Network depth.")
    parser.add_argument("--n-domain", type=int, default=1024, help="Number of domain residual samples.")
    parser.add_argument("--n-intermediate", type=int, default=8092, help="Number of interior data samples.")
    parser.add_argument("--n-boundary", type=int, default=512, help="Number of boundary samples.")
    parser.add_argument("--lambda-domain", type=float, default=0.0001, help="Residual loss weight.")
    parser.add_argument("--lambda-bdy", type=float, default=10.0, help="Boundary loss weight.")
    parser.add_argument("--condition-times", type=float, nargs="+", default=[0.0, 0.08, 0.2], help="Times with supervised data.")
    parser.add_argument("--lambda-time", type=float, nargs="+", help="Weights for each supervised time (same length as condition-times).")
    parser.add_argument("--grad-clip", type=float, default=0.4, help="Gradient clipping norm (set negative to disable).")
    parser.add_argument("--schedule-epoch", type=int, default=1000, help="Epoch interval for scheduler updates.")
    parser.add_argument(
        "--lambda-domain-schedule",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        help="Residual weight schedule values applied every schedule-epoch.",
    )
    parser.add_argument("--lrg", type=float, default=1e-3, help="Initial optimizer step size.")
    parser.add_argument("--lrg-decay", type=float, default=0.5, help="Decay factor applied on schedule.")
    parser.add_argument("--log-interval", type=int, default=50, help="How often to print losses.")
    parser.add_argument("--plot-interval", type=int, default=0, help="How often to save training plots (0 disables).")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory to write training plots.")
    parser.add_argument("--plot-field", type=str, default="rho", help="Field to visualize in training plots.")
    parser.add_argument("--save-model", type=Path, default=None, help="Optional path to save the trained model state_dict.")
    parser.add_argument("--t-max", type=float, default=0.4, help="Maximum time for sampling domain points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps). Defaults to auto-detect.")
    return parser.parse_args()


def main():
    args = parse_args()

    global DEVICE, T_DOMAIN, SNAPSHOTS, AVAILABLE_TIMES, X_DOMAIN, Y_DOMAIN

    DEVICE = torch.device(args.device) if args.device else _select_device()
    default_dtype = torch.float32 if DEVICE.type == "mps" else torch.float64
    torch.set_default_dtype(default_dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    T_DOMAIN = (0.0, args.t_max)

    if args.grad_clip is not None and args.grad_clip < 0:
        args.grad_clip = None

    lambda_time_values = args.lambda_time or [5.0, 20.0, 20.0]
    if len(lambda_time_values) != len(args.condition_times):
        raise ValueError("lambda-time must match condition-times length.")
    lambda_time = {float(t): float(w) for t, w in zip(args.condition_times, lambda_time_values)}

    print(f"Loading snapshots from {args.data_dir} ...")
    SNAPSHOTS = load_snapshots(args.data_dir)
    AVAILABLE_TIMES = np.array(sorted(SNAPSHOTS.keys()), dtype=np.float64)
    X_DOMAIN = (SNAPSHOTS[AVAILABLE_TIMES[0]]["x"][0], SNAPSHOTS[AVAILABLE_TIMES[0]]["x"][-1])
    Y_DOMAIN = (SNAPSHOTS[AVAILABLE_TIMES[0]]["y"][0], SNAPSHOTS[AVAILABLE_TIMES[0]]["y"][-1])

    model = PINN(input_dim=3, output_dim=len(PRIMITIVE_LABELS), width=args.width, depth=args.depth).to(DEVICE)
    optimizer = build_optimizer(model, args.lrg)

    config = TrainingConfig(
        epochs=args.epochs,
        n_domain=args.n_domain,
        n_intermediate=args.n_intermediate,
        n_boundary=args.n_boundary,
        lambda_domain=args.lambda_domain,
        lambda_bdy=args.lambda_bdy,
        lambda_time=lambda_time,
        grad_clip=args.grad_clip,
        schedule_epoch=args.schedule_epoch,
        lambda_domain_schedule=args.lambda_domain_schedule,
        lrg=args.lrg,
        lrg_decay=args.lrg_decay,
        log_interval=args.log_interval,
        plot_interval=args.plot_interval,
        plot_field=args.plot_field,
        plot_dir=args.plot_dir,
        condition_times=args.condition_times,
    )

    print(f"Training on {DEVICE} with dtype {torch.get_default_dtype()}.")
    history = train(model, optimizer, config)

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "history": history}, args.save_model)
        print(f"Saved model to {args.save_model}")

    if args.plot_dir:
        plot_rho_times(model, times=args.condition_times, output_path=args.plot_dir / "rho_slices.png")


if __name__ == "__main__":
    main()
