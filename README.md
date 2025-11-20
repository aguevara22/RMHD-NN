# Reconstructing RMHD from Physically Informed Neural Networks

A research code (`rmhdpinn.ipynb`) that implements physics-informed neural networks (PINNs) for relativistic magnetohydrodynamics (RMHD). Instead of advancing the standard conservative form, the workflow relies on Jacobians of the primitive-variable system (`M`, `AX`, source terms) to measure how well a neural surrogate satisfies the PDEs. The notebook first trains a baseline PINN, then iteratively learns residual-correction networks using stored Jacobian operators.

## Table of Contents
- [RMHD](#RMHD)
- [Overview](#overview)
- [Physics Background](#physics-background)
- [Notebook Workflow](#notebook-workflow)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Running the Notebook](#running-the-notebook)
- [Extending the Framework](#extending-the-framework)

## RMHD

Relativistic magnetohydrodynamics (RMHD) describes a conducting fluid coupled to electromagnetic fields in a relativistic setting. It is a generalization of more standard MHD in plasma physics and a preamble to the full-fledged GR-MHD modelling of black hole accretion.

As in the full GRMHD setup, the governing equations follow from stress--energy conservation, current conservation, and Maxwell (We will further assume the ideal-MHD condition $F^{\mu\nu}u_\nu=0$.). These form a first order coupled PDE system. For their numerical implementation it is usually written in the conservative scheme:

$$ \partial_t U(P) + \partial_i J^i(P) = 0 $$

for functions of the primitives $P=(\rho_0,p_0,u^\mu,B^\mu)$. A numerical integrator would typically perform 

1) time step to update $U(P)$ (primitive function)
2) The numerical inversion $P=P(U)$, a trascendental function highly sensitive to the background
3) Evaluate the current divergence $\partial_i J^i(P)$ via finite differences or finite elements and repeat step 1.


Linearizing around a homogeneous background $(\rho_0,p_0,u^\mu_0,B^\mu_0)$ yields a first-order system

$$ M \partial_t \delta P  + A^i \partial_i \delta P = 0$$

where the Jacobians $A^i=\partial F^i/\partial U$ encode the characteristic structure; the eigenvalues of $A^i n_i$ give the wave speeds along direction $n_i$.

Alfvén waves emerge as the transverse, incompressible characteristic family. In RMHD their propagation speed is

$$v_A^2 = \frac{b^2}{b^2 + h}$$

with $b^2$ the magnetic-field energy density in the fluid frame and $h=\rho_0+p_0+\varepsilon_0$ the enthalpy. They propagate strictly along the magnetic field, are polarized perpendicular to both $n_i$ and $B_i$, and remain linearly degenerate, making them an essential diagnostic mode of any RMHD linearization.



## Overview of our approach

The goal is to approximate RMHD dynamics with a neural surrogate that respects the governing equations. The neural network (PINN) itself is the map $x^\mu \to P=(\rho_0,p_0,u^\mu,B^\mu)$.

A primary PINN fits available simulation data, and imposes the constraint $  \partial_t U(P) + \partial_i J^i(P) = 0 $ in the loss function. Schematically:

$$\mathcal{L}_{\textrm{total}} = w_1 \mathcal{L}_{\textrm{PDE}} + w_2 \mathcal{L}_{\textrm{PDE}}$$

see [PINN] for more details.


Two successive residual networks (`model_residual`, `model_residual_it`) learn to cancel the PDE violations of the latest solution (`model`, `corr`, `corr2`). Key ideas:

- **Jacobian-based residuals.** `data_out` converts primitive predictions into the reduced Jacobian blocks `M`, `AX`, and differential terms (`dP/dt`, `dP/dx`), enabling a linear form `M(dp/dt - dpdt_r) + AX(dp/dx - dpdx_r) + S p`.
- **Residual-guided sampling.** `build_residual_mixture_coords` biases collocation points toward regions with high Jacobian residual norms, mixing in uniform samples for coverage.
- **Iterative correction.** Residual networks subtract from the baseline to produce `corr(x)` and `corr2(x)`, mimicking deferred corrections while sharing samplers and conditioning data.
- **Muon optimizer.** Training leverages the custom `PINNMuonOptimizer` (from `mm.py`) for mixed second-order and Adam-like updates.

## Jacobian form of the RMHD system
- **Model:** For instance, we can take 1D RMHD with a background guide field (`B_x = 5`). Primitive variables are density, velocity, pressure, and transverse magnetic components.
- **Formulation:** Rather than conservative fluxes, the notebook works with the Jacobian matrices (`compute_M`, `compute_AX`) imported from `jacobians.py`. These encode the linearized PDE system used to define residuals.
- **Boundary/conditioning data:** Snapshot files in `data1d/` provide time slices at `t = 0.0, 0.036, 0.1` for supervised anchoring. Open boundary data at `t=0` supply additional constraints.

At each collocation point we evaluate the primitive state $\mathbf{p}(x,t)$, and the RMHD system is written in Jacobian form

$$
M(\mathbf{p}) \partial_t \mathbf{p} + A_x(\mathbf{p}) \partial_x \mathbf{p} + S(\mathbf{p}) \mathbf{p} = \mathbf{0},
$$

where $M$ is the time Jacobian, $A_x$ is the spatial Jacobian, and $S = \partial_t M + \partial_x A_x$ captures source-like terms arising from spatially varying operators. During training we compare against precomputed targets

$$
\mathcal{R}(\hat{\mathbf{p}}) = M_r \big(\partial_t \hat{\mathbf{p}} - \partial_t \mathbf{p}_r \big)
      + A_{X,r} \big(\partial_x \hat{\mathbf{p}} - \partial_x \mathbf{p}_r \big)
      + S_r\,\hat{\mathbf{p}},
$$

and minimize $\Vert \mathcal{R}(\hat{\mathbf{p}})\Vert_2^2$ so that the PINN adheres to the Jacobian PDE while matching conditioning data.

## Notebook Workflow
1. **Train baseline PINN (`model`).** Uses Muon optimizer plus domain/data/boundary losses.
2. **Build Jacobian operators and train `model_residual`.** Residual-guided sampling + stored $(M_r, A_{X,r}, S_r, \partial_t\mathbf{p}_r, \partial_x\mathbf{p}_r)$ define `lin_eq`.
3. **Clean up tensors, recompute operators with `corr(x)`, and train `model_residual_it` to obtain `corr2(x)`.

## Repository Structure
```
RMHD-local/
├── rmhdpinn.ipynb        # Main notebook described here
├── RMHDEquations2D.py    # Reference RMHD equations (unused but informative)
├── jacobians.py          # Computes M/AX Jacobians used in the notebook
├── mm.py                 # PINNMuonOptimizer and Muon helper utilities
├── gauss_newton*.py      # Alternative solvers/experiments
├── data1d/               # Snapshot data (npz files) for conditioning
└── README.md             # This document
```

## Prerequisites
- Python 3.10+ with `pip`
- PyTorch ≥ 2.0 (CPU, CUDA, or Apple MPS build)
- NumPy, Matplotlib, SciPy, tqdm, IPython, Muon optimizer dependency
- JupyterLab or VS Code notebooks
- RMHD snapshot files inside `data1d/`

Recommended setup (from the repo root):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy matplotlib scipy tqdm jupyter
pip install git+https://github.com/KellerJordan/Muon
```

## Running the Notebook
1. Launch Jupyter Lab or VS Code and open `rmhdpinn.ipynb`.
2. Execute the environment and data-loading cells (Sections “Importing data” and “RMHD residual helpers”).
3. Train the baseline PINN (`# Training Loop`). Track metrics and plots.
4. Run the residual sampler + Jacobian storage cells.
5. Train `model_residual`, inspect corrections, and run the cleanup cell.
6. Execute the iteration-2 Jacobian builder and train `model_residual_it`.
7. Use the plotting cells to compare baseline vs corrections, and sample new predictions via `corr` / `corr2`.

## Extending the Framework
- **More iterations:** Add additional correction stages by repeating the Jacobian-storage + residual-training pattern.
- **Higher dimensions:** Replace `jacobians.py` with a higher-dimensional RMHD Jacobian provider and adjust the sampler.
- **Hybrid losses:** Combine Jacobian residuals with conservative-form residuals for robustness.
- **Deployment:** Export trained models by scripting the inference calls (`model`, `corr`, `corr2`) and saving weights with `torch.save`.

Feel free to open issues or PRs if you adapt the notebook to new RMHD scenarios or improve the training strategy.
