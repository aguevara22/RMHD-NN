import torch

# Default adiabatic index used across the notebooks
Gamma = 5 / 3


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


def _enthalpy_like(rho, p, gamma=Gamma):
    return rho + gamma * p / (gamma - 1.0)


def _shared_quantities(primitives, gamma):
    rho, v, B, p = _split_primitives(primitives)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

    lf = _lorentz_factor(v)
    he = _enthalpy_like(rho, p, gamma)
    B2 = torch.sum(B * B, dim=-1)
    Bv = torch.sum(B * v, dim=-1)

    lf2 = lf * lf
    lf3 = lf2 * lf
    lf4 = lf2 * lf2
    lf_inv2 = lf2.reciprocal()

    return {
        "primitives": primitives,
        "rho": rho,
        "v": v,
        "B": B,
        "p": p,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
        "lf": lf,
        "he": he,
        "B2": B2,
        "Bv": Bv,
        "lf2": lf2,
        "lf3": lf3,
        "lf4": lf4,
        "lf_inv2": lf_inv2,
        "gamma": gamma,
    }


def _allocate(shape_tensor):
    return torch.zeros(
        shape_tensor.shape[:-1] + (8, 8),
        dtype=shape_tensor.dtype,
        device=shape_tensor.device,
    )


def _compute_M_from_shared(shared):
    gamma = shared["gamma"]
    rho = shared["rho"]
    vx, vy, vz = shared["vx"], shared["vy"], shared["vz"]
    Bx, By, Bz = shared["Bx"], shared["By"], shared["Bz"]
    lf = shared["lf"]
    he = shared["he"]
    B2 = shared["B2"]
    Bv = shared["Bv"]
    lf2, lf3, lf4 = shared["lf2"], shared["lf3"], shared["lf4"]
    lf_inv2 = shared["lf_inv2"]

    M = _allocate(shared["primitives"])

    M[..., 0, 0] = lf
    M[..., 0, 1] = lf3 * vx * rho
    M[..., 0, 2] = lf3 * vy * rho
    M[..., 0, 3] = lf3 * vz * rho

    M[..., 1, 0] = lf2 * vx
    M[..., 1, 1] = B2 - Bx * Bx + he * lf2 + 2 * he * lf4 * vx * vx
    M[..., 1, 2] = -Bx * By + 2 * he * lf4 * vx * vy
    M[..., 1, 3] = -Bx * Bz + 2 * he * lf4 * vx * vz
    M[..., 1, 4] = -By * vy - Bz * vz
    M[..., 1, 5] = 2 * By * vx - Bx * vy
    M[..., 1, 6] = 2 * Bz * vx - Bx * vz
    M[..., 1, 7] = (gamma * lf2 * vx) / (gamma - 1.0)

    M[..., 2, 0] = lf2 * vy
    M[..., 2, 1] = -Bx * By + 2 * he * lf4 * vx * vy
    M[..., 2, 2] = B2 - By * By + he * lf2 + 2 * he * lf4 * vy * vy
    M[..., 2, 3] = -By * Bz + 2 * he * lf4 * vy * vz
    M[..., 2, 4] = -By * vx + 2 * Bx * vy
    M[..., 2, 5] = -Bv + By * vy
    M[..., 2, 6] = 2 * Bz * vy - By * vz
    M[..., 2, 7] = (gamma * lf2 * vy) / (gamma - 1.0)

    M[..., 3, 0] = lf2 * vz
    M[..., 3, 1] = -Bx * Bz + 2 * he * lf4 * vx * vz
    M[..., 3, 2] = -By * Bz + 2 * he * lf4 * vy * vz
    M[..., 3, 3] = Bx * Bx + By * By + he * lf2 + 2 * he * lf4 * vz * vz
    M[..., 3, 4] = -Bz * vx + 2 * Bx * vz
    M[..., 3, 5] = -Bz * vy + 2 * By * vz
    M[..., 3, 6] = -Bv + Bz * vz
    M[..., 3, 7] = (gamma * lf2 * vz) / (gamma - 1.0)

    M[..., 4, 4] = 1.0
    M[..., 5, 5] = 1.0
    M[..., 6, 6] = 1.0

    M[..., 7, 0] = lf2
    M[..., 7, 1] = -Bv * Bx + (B2 + 2 * he * lf4) * vx
    M[..., 7, 2] = -Bv * By + (B2 + 2 * he * lf4) * vy
    M[..., 7, 3] = -Bv * Bz + (B2 + 2 * he * lf4) * vz
    M[..., 7, 4] = Bx * (2 - lf_inv2) - Bv * vx
    M[..., 7, 5] = By * (2 - lf_inv2) - Bv * vy
    M[..., 7, 6] = Bz * (2 - lf_inv2) - Bv * vz
    M[..., 7, 7] = -1.0 + (gamma * lf2) / (gamma - 1.0)

    return M


def _compute_AX_from_shared(shared):
    gamma = shared["gamma"]
    rho = shared["rho"]
    vx, vy, vz = shared["vx"], shared["vy"], shared["vz"]
    Bx, By, Bz = shared["Bx"], shared["By"], shared["Bz"]
    he = shared["he"]
    B2 = shared["B2"]
    Bv = shared["Bv"]
    lf = shared["lf"]
    lf2, lf4 = shared["lf2"], shared["lf4"]
    lf_inv2 = shared["lf_inv2"]

    AX = _allocate(shared["primitives"])

    AX[..., 0, 0] = lf * vx
    AX[..., 0, 1] = lf * (1.0 + lf2 * vx * vx) * rho
    AX[..., 0, 2] = shared["lf3"] * vx * vy * rho
    AX[..., 0, 3] = shared["lf3"] * vx * vz * rho

    AX[..., 1, 0] = lf2 * vx * vx
    AX[..., 1, 1] = (
        B2 * vx
        + 2 * (-Bv * Bv + he) * lf4 * vx ** 3
        - Bx * (2 * Bv + Bx * vx)
        + 2 * lf2 * vx * (he + Bv * (-2 * Bv + By * vy + Bz * vz))
    )
    AX[..., 1, 2] = (
        -2 * Bv * By
        - 2 * Bv * By * lf2 * vx * vx
        + B2 * vy
        + By * By * vy
        - 2 * Bz * Bz * vy
        + 2 * (-Bv * Bv + he) * lf4 * vx * vx * vy
        + 3 * By * Bz * vz
    )
    AX[..., 1, 3] = (
        -2 * Bv * Bz
        - 2 * Bv * Bz * lf2 * vx * vx
        + 3 * By * Bz * vy
        + B2 * vz
        - 2 * By * By * vz
        + Bz * Bz * vz
        + 2 * (-Bv * Bv + he) * lf4 * vx * vx * vz
    )
    AX[..., 1, 4] = -Bx * lf_inv2 - Bv * vx * (3 + 2 * lf2 * vx * vx) + By * vx * vy + Bz * vx * vz
    AX[..., 1, 5] = -2 * Bv * (1 + lf2 * vx * vx) * vy + 3 * Bz * vy * vz + By * (2 - lf_inv2 + vy * vy - 2 * vz * vz)
    AX[..., 1, 6] = -2 * Bv * (1 + lf2 * vx * vx) * vz + 3 * By * vy * vz + Bz * (2 - lf_inv2 - 2 * vy * vy + vz * vz)
    AX[..., 1, 7] = 1.0 + (gamma * lf2 * vx * vx) / (gamma - 1.0)

    AX[..., 2, 0] = lf2 * vx * vy
    AX[..., 2, 1] = (
        -2 * Bv * By
        + B2 * vy
        - Bx * Bx * vy
        + 2 * (-Bv * Bv + he) * lf4 * vx * vx * vy
        + lf2 * (he - Bv * (Bv + 2 * Bx * vx)) * vy
    )
    AX[..., 2, 2] = (
        -By * By * vx
        + Bz * Bz * vx
        - 2 * Bv * By * lf2 * vx * vy
        - (Bv * Bv - he) * lf2 * vx * (1 + 2 * lf2 * vy * vy)
        - Bx * Bz * vz
    )
    AX[..., 2, 3] = (
        -2 * By * Bz * vx
        - Bx * Bz * vy
        - 2 * Bv * Bz * lf2 * vx * vy
        + 2 * Bx * By * vz
        + 2 * (-Bv * Bv + he) * lf4 * vx * vy * vz
    )
    AX[..., 2, 4] = -vy * (2 * Bv * lf2 * vx * vx + Bz * vz) + By * (-1 - vx * vx + vz * vz)
    AX[..., 2, 5] = -vx * (Bv + By * vy + 2 * Bv * lf2 * vy * vy + Bz * vz) + Bx * (-1 + vz * vz)
    AX[..., 2, 6] = 2 * Bz * vx * vy - (Bx * vy + 2 * vx * (By + Bv * lf2 * vy)) * vz
    AX[..., 2, 7] = (gamma * lf2 * vx * vy) / (gamma - 1.0)

    AX[..., 3, 0] = lf2 * vx * vz
    AX[..., 3, 1] = (
        -2 * Bv * Bz
        + B2 * vz
        - Bx * Bx * vz
        + 2 * (-Bv * Bv + he) * lf4 * vx * vx * vz
        + lf2 * (he - Bv * (Bv + 2 * Bx * vx)) * vz
    )
    AX[..., 3, 2] = (
        2 * vy * (Bx * Bz + (-Bv * Bv + he) * lf4 * vx * vz)
        - By * (Bx * vz + 2 * vx * (Bz + Bv * lf2 * vz))
    )
    AX[..., 3, 3] = (
        -Bv * Bx
        + B2 * vx
        - 2 * Bz * Bz * vx
        + Bx * Bz * vz
        + 2 * (-Bv * Bv + he) * lf4 * vx * vz * vz
        + lf2 * vx * (he - Bv * (Bv + 2 * Bz * vz))
    )
    AX[..., 3, 4] = Bz * (-1 - vx * vx + vy * vy) - (2 * Bv * lf2 * vx * vx + By * vy) * vz
    AX[..., 3, 5] = -2 * Bz * vx * vy + 2 * By * vx * vz - (Bx + 2 * Bv * lf2 * vx) * vy * vz
    AX[..., 3, 6] = -((Bx + 2 * Bv * lf2 * vx) * (1 + lf2 * vz * vz) * lf_inv2)
    AX[..., 3, 7] = (gamma * lf2 * vx * vz) / (gamma - 1.0)

    AX[..., 5, 1] = By
    AX[..., 5, 2] = -Bx
    AX[..., 5, 4] = -vy
    AX[..., 5, 5] = vx

    AX[..., 6, 1] = Bz
    AX[..., 6, 3] = -Bx
    AX[..., 6, 4] = -vz
    AX[..., 6, 6] = vx

    AX[..., 7, 0] = lf2 * vx
    AX[..., 7, 1] = B2 - Bx * Bx + he * lf2 + 2 * he * lf4 * vx * vx
    AX[..., 7, 2] = -Bx * By + 2 * he * lf4 * vx * vy
    AX[..., 7, 3] = -Bx * Bz + 2 * he * lf4 * vx * vz
    AX[..., 7, 4] = -By * vy - Bz * vz
    AX[..., 7, 5] = 2 * By * vx - Bx * vy
    AX[..., 7, 6] = 2 * Bz * vx - Bx * vz
    AX[..., 7, 7] = (gamma * lf2 * vx) / (gamma - 1.0)

    return AX


def _compute_AY_from_shared(shared):
    gamma = shared["gamma"]
    rho = shared["rho"]
    vx, vy, vz = shared["vx"], shared["vy"], shared["vz"]
    Bx, By, Bz = shared["Bx"], shared["By"], shared["Bz"]
    he = shared["he"]
    B2 = shared["B2"]
    Bv = shared["Bv"]
    lf = shared["lf"]
    lf2, lf4 = shared["lf2"], shared["lf4"]
    lf_inv2 = shared["lf_inv2"]

    AY = _allocate(shared["primitives"])

    AY[..., 0, 0] = lf * vy
    AY[..., 0, 1] = shared["lf3"] * vx * vy * rho
    AY[..., 0, 2] = lf * (1.0 + lf2 * vy * vy) * rho
    AY[..., 0, 3] = shared["lf3"] * vy * vz * rho

    AY[..., 1, 0] = lf2 * vx * vy
    AY[..., 1, 1] = (
        -Bx * Bx * vy
        + Bz * Bz * vy
        - 2 * Bv * Bx * lf2 * vx * vy
        - (Bv * Bv - he) * lf2 * (1 + 2 * lf2 * vx * vx) * vy
        - By * Bz * vz
    )
    AY[..., 1, 2] = (
        -2 * Bv * Bx
        + B2 * vx
        - By * By * vx
        + 2 * (-Bv * Bv + he) * lf4 * vx * vy * vy
        + lf2 * vx * (he - Bv * (Bv + 2 * By * vy))
    )
    AY[..., 1, 3] = (
        -By * Bz * vx
        - 2 * Bx * Bz * vy
        - 2 * Bv * Bz * lf2 * vx * vy
        + 2 * Bx * By * vz
        + 2 * (-Bv * Bv + he) * lf4 * vx * vy * vz
    )
    AY[..., 1, 4] = -vy * (Bv + Bx * vx + 2 * Bv * lf2 * vx * vx + Bz * vz) + By * (-1 + vz * vz)
    AY[..., 1, 5] = -vx * (2 * Bv * lf2 * vy * vy + Bz * vz) + Bx * (-1 - vy * vy + vz * vz)
    AY[..., 1, 6] = 2 * Bz * vx * vy - (By * vx + 2 * (Bx + Bv * lf2 * vx) * vy) * vz
    AY[..., 1, 7] = (gamma * lf2 * vx * vy) / (gamma - 1.0)

    AY[..., 2, 0] = lf2 * vy * vy
    AY[..., 2, 1] = (
        -2 * Bv * Bx
        + B2 * vx
        + Bx * Bx * vx
        - 2 * Bz * Bz * vx
        - 2 * Bv * Bx * lf2 * vy * vy
        + 2 * (-Bv * Bv + he) * lf4 * vy * vy * vx
        + 3 * Bx * Bz * vz
    )
    AY[..., 2, 2] = (
        B2 * vy
        + 2 * (-Bv * Bv + he) * lf4 * vy ** 3
        - By * (2 * Bv + By * vy)
        + 2 * lf2 * vy * (he + Bv * (-2 * Bv + Bx * vx + Bz * vz))
    )
    AY[..., 2, 3] = (
        -2 * Bv * Bz
        + 3 * Bx * Bz * vx
        - 2 * Bv * Bz * lf2 * vy * vy
        + B2 * vz
        - 2 * Bx * Bx * vz
        + Bz * Bz * vz
        + 2 * (-Bv * Bv + he) * lf4 * vy * vy * vz
    )
    AY[..., 2, 4] = -2 * Bv * vx * (1 + lf2 * vy * vy) + 3 * Bz * vx * vz + Bx * (2 - lf_inv2 + vx * vx - 2 * vz * vz)
    AY[..., 2, 5] = -By * lf_inv2 + Bx * vx * vy - Bv * vy * (3 + 2 * lf2 * vy * vy) + Bz * vy * vz
    AY[..., 2, 6] = 3 * Bx * vx * vz - 2 * Bv * (1 + lf2 * vy * vy) * vz + Bz * (2 - lf_inv2 - 2 * vx * vx + vz * vz)
    AY[..., 2, 7] = 1.0 + (gamma * lf2 * vy * vy) / (gamma - 1.0)

    AY[..., 3, 0] = lf2 * vy * vz
    AY[..., 3, 1] = (
        2 * vx * (By * Bz + (-Bv * Bv + he) * lf4 * vy * vz)
        - Bx * (By * vz + 2 * vy * (Bz + Bv * lf2 * vz))
    )
    AY[..., 3, 2] = (
        -2 * Bv * Bz
        + B2 * vz
        - By * By * vz
        + 2 * (-Bv * Bv + he) * lf4 * vy * vy * vz
        + lf2 * (he - Bv * (Bv + 2 * By * vy)) * vz
    )
    AY[..., 3, 3] = (
        -Bv * By
        + B2 * vy
        - 2 * Bz * Bz * vy
        + By * Bz * vz
        + 2 * (-Bv * Bv + he) * lf4 * vy * vz * vz
        + lf2 * vy * (he - Bv * (Bv + 2 * Bz * vz))
    )
    AY[..., 3, 4] = -2 * Bz * vx * vy + 2 * Bx * vy * vz - (By + 2 * Bv * lf2 * vy) * vx * vz
    AY[..., 3, 5] = Bz * (-1 + vx * vx - vy * vy) - (Bx * vx + 2 * Bv * lf2 * vy * vy) * vz
    AY[..., 3, 6] = -((By + 2 * Bv * lf2 * vy) * (1 + lf2 * vz * vz) * lf_inv2)
    AY[..., 3, 7] = (gamma * lf2 * vy * vz) / (gamma - 1.0)

    AY[..., 4, 1] = -By
    AY[..., 4, 2] = Bx
    AY[..., 4, 4] = vy
    AY[..., 4, 5] = -vx

    AY[..., 6, 2] = Bz
    AY[..., 6, 3] = -By
    AY[..., 6, 5] = -vz
    AY[..., 6, 6] = vy

    AY[..., 7, 0] = lf2 * vy
    AY[..., 7, 1] = -Bx * By + 2 * he * lf4 * vx * vy
    AY[..., 7, 2] = B2 - By * By + he * lf2 + 2 * he * lf4 * vy * vy
    AY[..., 7, 3] = -By * Bz + 2 * he * lf4 * vy * vz
    AY[..., 7, 4] = -By * vx + 2 * Bx * vy
    AY[..., 7, 5] = -Bx * vx - Bz * vz
    AY[..., 7, 6] = 2 * Bz * vy - By * vz
    AY[..., 7, 7] = (gamma * lf2 * vy) / (gamma - 1.0)

    return AY


def compute_M(primitives, gamma=Gamma):
    shared = _shared_quantities(primitives, gamma)
    return _compute_M_from_shared(shared)


def compute_AX(primitives, gamma=Gamma):
    shared = _shared_quantities(primitives, gamma)
    return _compute_AX_from_shared(shared)


def compute_AY(primitives, gamma=Gamma):
    shared = _shared_quantities(primitives, gamma)
    return _compute_AY_from_shared(shared)


def compute_jacobians(primitives, gamma=Gamma, which=None):
    """
    Compute multiple Jacobians (M, AX, AY) reusing shared quantities.
    Parameters
    ----------
    primitives : Tensor[..., 8]
    which : iterable or None
        If None, returns all three in a dict. Otherwise, compute the requested keys.
    """
    if which is None:
        which = ("M", "AX", "AY")
    shared = _shared_quantities(primitives, gamma)
    builders = {
        "M": _compute_M_from_shared,
        "AX": _compute_AX_from_shared,
        "AY": _compute_AY_from_shared,
    }
    results = {}
    for key in which:
        if key not in builders:
            raise ValueError(f"Unknown Jacobian key '{key}'.")
        results[key] = builders[key](shared)
    return results


__all__ = ["Gamma", "compute_M", "compute_AX", "compute_AY", "compute_jacobians"]
