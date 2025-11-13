import torch

Gamma = 4 / 3


def _split_primitives(primitives):
    rho = primitives[..., 0]
    v_x = primitives[..., 1]
    v_y = primitives[..., 2]
    v_z = primitives[..., 3]
    B_x = primitives[..., 4]
    B_y = primitives[..., 5]
    B_z = primitives[..., 6]
    p = primitives[..., 7]

    v = torch.stack((v_x, v_y, v_z), dim=-1)
    B = torch.stack((B_x, B_y, B_z), dim=-1)
    return rho, v, B, p


def _lorentz_factor(v):
    one_minus_v2 = 1.0 - torch.sum(v * v, dim=-1)
    one_minus_v2 = torch.clamp(one_minus_v2, min=1e-12)
    return torch.rsqrt(one_minus_v2)


def _specific_enthalpy(rho, p, gamma=Gamma):
    return 1.0 + gamma * p / ((gamma - 1.0) * rho)


def _auxiliary_state(primitives, gamma=Gamma):
    rho, v, B, p = _split_primitives(primitives)

    W = _lorentz_factor(v)
    h = _specific_enthalpy(rho, p, gamma)
    rho_h = rho * h

    v_dot_B = torch.sum(v * B, dim=-1)
    B2 = torch.sum(B * B, dim=-1)

    inv_W = 1.0 / W
    b0 = W * v_dot_B
    b_vec = B * inv_W.unsqueeze(-1) + b0.unsqueeze(-1) * v
    b2 = B2 * inv_W**2 + v_dot_B**2

    p_tot = p + 0.5 * b2
    prefactor = (rho_h + b2) * W**2

    momentum = prefactor.unsqueeze(-1) * v - b0.unsqueeze(-1) * b_vec
    energy = prefactor - p_tot - b0**2
    D = rho * W

    return {
        "rho": rho,
        "v": v,
        "B": B,
        "p": p,
        "W": W,
        "D": D,
        "momentum": momentum,
        "energy": energy,
        "p_tot": p_tot,
        "b_vec": b_vec,
    }


def primitives_to_conserved(primitives, gamma=Gamma):
    state = _auxiliary_state(primitives, gamma)
    return torch.stack(
        (
            state["D"],
            state["momentum"][..., 0],
            state["momentum"][..., 1],
            state["momentum"][..., 2],
            state["B"][..., 0],
            state["B"][..., 1],
            state["B"][..., 2],
            state["energy"],
        ),
        dim=-1,
    )


def flux_x(primitives, gamma=Gamma):
    state = _auxiliary_state(primitives, gamma)
    v = state["v"]
    B = state["B"]
    S = state["momentum"]
    b = state["b_vec"]
    p_tot = state["p_tot"]

    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            state["D"] * v[..., 0],
            S[..., 0] * v[..., 0] + p_tot - b[..., 0] * b[..., 0],
            S[..., 1] * v[..., 0] - b[..., 0] * b[..., 1],
            S[..., 2] * v[..., 0] - b[..., 0] * b[..., 2],
            zero,
            B[..., 1] * v[..., 0] - B[..., 0] * v[..., 1],
            B[..., 2] * v[..., 0] - B[..., 0] * v[..., 2],
            S[..., 0],
        ),
        dim=-1,
    )


def flux_y(primitives, gamma=Gamma):
    state = _auxiliary_state(primitives, gamma)
    v = state["v"]
    B = state["B"]
    S = state["momentum"]
    b = state["b_vec"]
    p_tot = state["p_tot"]

    zero = torch.zeros_like(v[..., 1])
    return torch.stack(
        (
            state["D"] * v[..., 1],
            S[..., 0] * v[..., 1] - b[..., 1] * b[..., 0],
            S[..., 1] * v[..., 1] + p_tot - b[..., 1] * b[..., 1],
            S[..., 2] * v[..., 1] - b[..., 1] * b[..., 2],
            B[..., 0] * v[..., 1] - B[..., 1] * v[..., 0],
            zero,
            B[..., 2] * v[..., 1] - B[..., 1] * v[..., 2],
            S[..., 1],
        ),
        dim=-1,
    )


__all__ = ["Gamma", "primitives_to_conserved", "flux_x", "flux_y"]
