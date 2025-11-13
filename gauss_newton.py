from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call


@dataclass
class GaussNewtonStepResult:
    loss_before: float
    loss_after: float
    grad_norm: float
    step_norm: float
    damping: float
    attempts: int
    converged: bool


class GaussNewtonPINNOptimizer:
    """
    Stateless Gauss-Newton optimizer with a simple Levenberg-Marquardt style damping strategy.

    Parameters
    ----------
    model:
        Neural network to optimize. Parameters are updated in-place.
    damping:
        Initial damping factor added to the (approximate) Gauss-Newton system.
    damping_increase:
        Multiplicative factor applied to the damping coefficient after a failed step.
    damping_decrease:
        Multiplicative factor applied to the damping coefficient after a successful step.
    damping_cap:
        Upper bound on the damping value to avoid numerical overflow.
    max_attempts:
        Maximum number of damping adjustments attempted per GN step.
    vectorize_jacobian:
        Whether to request a vectorized Jacobian from autograd (forward-over-backward).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        damping: float = 1e-2,
        damping_increase: float = 5.0,
        damping_decrease: float = 0.3,
        damping_cap: float = 1e6,
        max_attempts: int = 6,
        vectorize_jacobian: bool = True,
        reg_weight: float = 0.0,
        jtj_diagonal_eps: float = 1e-9,
        use_direct_solve: bool = False,
        cg_max_iters: int | None = None,
        cg_tol: float = 1e-10,
        report_linear_solvers: bool = True,
    ) -> None:
        self.model = model
        self._params = [p for p in model.parameters() if p.requires_grad]
        if not self._params:
            raise ValueError("GaussNewtonPINNOptimizer requires at least one trainable parameter.")

        self._buffers = OrderedDict(model.named_buffers())
        self._param_names, self._param_shapes, self._param_numels = self._collect_param_specs()

        self.damping = damping
        self.damping_increase = damping_increase
        self.damping_decrease = damping_decrease
        self.damping_cap = damping_cap
        self.max_attempts = max_attempts
        self.vectorize_jacobian = vectorize_jacobian
        self.reg_weight = reg_weight
        self.jtj_diagonal_eps = jtj_diagonal_eps
        self.use_direct_solve = use_direct_solve
        self.cg_max_iters = cg_max_iters
        self.cg_tol = cg_tol
        self.report_linear_solvers = report_linear_solvers

    def _collect_param_specs(self) -> tuple[list[str], list[torch.Size], list[int]]:
        names: list[str] = []
        shapes: list[torch.Size] = []
        counts: list[int] = []
        for name, tensor in self.model.named_parameters():
            if tensor.requires_grad:
                names.append(name)
                shapes.append(tensor.shape)
                counts.append(tensor.numel())
        return names, shapes, counts

    def _vector_to_param_dict(self, vec: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if vec.ndim != 1:
            raise ValueError("Parameter vector must be 1-dimensional.")
        splits = torch.split(vec, self._param_numels)
        param_dict = OrderedDict()
        for name, shape, value in zip(self._param_names, self._param_shapes, splits):
            param_dict[name] = value.view(shape)
        return param_dict

    def _functional_model(self, vec: torch.Tensor) -> Callable:
        param_dict = self._vector_to_param_dict(vec)
        state_dict = OrderedDict(param_dict)
        if self._buffers:
            state_dict.update(self._buffers)

        def forward(*args, **kwargs):
            return functional_call(
                self.model,
                state_dict,
                args=args if args else (),
                kwargs=kwargs if kwargs else None,
            )

        return forward

    def _flatten_params(self) -> torch.Tensor:
        return parameters_to_vector(self._params).detach()

    def _assign_params(self, vec: torch.Tensor) -> None:
        vector_to_parameters(vec, self._params)

    def step(
        self,
        residual_builder: Callable[[Callable, torch.Tensor, bool], torch.Tensor],
    ) -> GaussNewtonStepResult:
        """
        Perform a single Gauss-Newton step.

        Parameters
        ----------
        residual_builder:
            Callable taking (model_forward, param_vector, create_graph) and returning
            the stacked residual vector with the desired weighting applied.
        """

        base_vec = self._flatten_params()
        vec_req = base_vec.clone().requires_grad_(True)

        def residual_with_graph(vec: torch.Tensor) -> torch.Tensor:
            model_forward = self._functional_model(vec)
            residual = residual_builder(model_forward, vec, True)
            if self.reg_weight > 0.0:
                residual = torch.cat(
                    (residual, torch.sqrt(torch.tensor(self.reg_weight, device=residual.device, dtype=residual.dtype)) * vec)
                )
            return residual

        residual_vec = residual_with_graph(vec_req)
        flat_residual = residual_vec.reshape(-1)
        loss_before = flat_residual.pow(2).sum()

        jacobian = torch.autograd.functional.jacobian(
            residual_with_graph,
            vec_req,
            vectorize=self.vectorize_jacobian,
        )
        jacobian = jacobian.reshape(flat_residual.numel(), -1)

        jt_residual = jacobian.transpose(0, 1) @ flat_residual
        jtj = jacobian.transpose(0, 1) @ jacobian
        diag = torch.diagonal(jtj)

        damping = float(self.damping)
        attempts = 0
        success = False
        best_vec = base_vec
        best_loss = loss_before

        def jacobian_vector_product(v: torch.Tensor) -> torch.Tensor:
            _, jvp = torch.autograd.functional.jvp(
                residual_with_graph,
                (vec_req,),
                (v.view_as(vec_req),),
                create_graph=False,
                strict=True,
            )
            return jvp.reshape(-1)

        def jacobian_transpose_vector_product(w: torch.Tensor) -> torch.Tensor:
            grad = torch.autograd.grad(
                flat_residual,
                vec_req,
                grad_outputs=w.reshape(flat_residual.shape),
                retain_graph=True,
                allow_unused=False,
            )[0]
            return grad.reshape(-1)

        def damped_matvec(v: torch.Tensor, damp: float) -> torch.Tensor:
            jv = jacobian_vector_product(v)
            jt_jv = jacobian_transpose_vector_product(jv)
            return jt_jv + damp * (diag + self.jtj_diagonal_eps) * v

        def conjugate_gradient(
            matvec_fn,
            rhs: torch.Tensor,
            *,
            damping_value: float,
            tol: float,
            max_iters: int | None,
        ) -> tuple[torch.Tensor, int, bool]:
            x = torch.zeros_like(rhs)
            r = rhs.clone() - matvec_fn(x, damping_value)
            p = r.clone()
            rs_old = torch.dot(r, r)
            cg_tol = tol * tol
            max_iters = max_iters or rhs.numel()
            if rs_old <= cg_tol:
                return x, 0, True

            converged = False
            for k in range(max_iters):
                Ap = matvec_fn(p, damping_value)
                denom = torch.dot(p, Ap)
                if denom.abs() < 1e-32:
                    break
                alpha = rs_old / denom
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = torch.dot(r, r)
                if rs_new <= cg_tol:
                    converged = True
                    rs_old = rs_new
                    break
                beta = rs_new / rs_old
                p = r + beta * p
                rs_old = rs_new

            return x, k + 1, converged

        while attempts < self.max_attempts and damping <= self.damping_cap:
            attempts += 1
            system = jtj.clone()
            system.diagonal().add_(damping * (diag + self.jtj_diagonal_eps))
            delta_direct = None
            try:
                if self.use_direct_solve:
                    delta_direct = torch.linalg.solve(system, -jt_residual)
            except (RuntimeError, torch.linalg.LinAlgError):
                delta_direct = None

            delta_cg, cg_iters, cg_converged = conjugate_gradient(
                damped_matvec,
                -jt_residual,
                damping_value=damping,
                tol=self.cg_tol,
                max_iters=self.cg_max_iters,
            )

            if self.report_linear_solvers:
                direct_norm = float('nan') if delta_direct is None else torch.linalg.norm(delta_direct).item()
                print(
                    f"[GN] damping={damping:.2e} | direct_step_norm={direct_norm:.3e} | "
                    f"cg_step_norm={torch.linalg.norm(delta_cg).item():.3e} "
                    f"(iters={cg_iters}, converged={cg_converged})"
                )

            delta = delta_direct if delta_direct is not None else delta_cg

            candidate_vec = base_vec + delta
            model_candidate = self._functional_model(candidate_vec)
            residual_candidate = residual_builder(model_candidate, candidate_vec, False)
            if self.reg_weight > 0.0:
                reg_term = torch.sqrt(
                    torch.tensor(self.reg_weight, device=residual_candidate.device, dtype=residual_candidate.dtype)
                ) * candidate_vec
                residual_candidate = torch.cat((residual_candidate, reg_term))

            cand_flat = residual_candidate.reshape(-1)
            loss_candidate = cand_flat.pow(2).sum()

            if torch.isfinite(loss_candidate) and loss_candidate < best_loss:
                best_loss = loss_candidate
                best_vec = candidate_vec
                success = True
                self.damping = max(self.damping * self.damping_decrease, 1e-12)
                grad_norm = torch.linalg.norm(jt_residual).item()
                step_norm = torch.linalg.norm(delta).item()
                break

            damping *= self.damping_increase

        if not success:
            grad_norm = torch.linalg.norm(jt_residual).item()
            step_norm = 0.0
            self.damping = min(damping, self.damping_cap)

        self._assign_params(best_vec.detach())

        return GaussNewtonStepResult(
            loss_before=loss_before.detach().item(),
            loss_after=best_loss.detach().item(),
            grad_norm=grad_norm,
            step_norm=step_norm,
            damping=float(self.damping),
            attempts=attempts,
            converged=success,
        )
