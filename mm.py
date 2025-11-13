import torch.distributed as dist
import torch
import types
import muon as _muon  # import the module first

# single-device stub for muon’s internal 'dist'
if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
    _muon.dist = types.SimpleNamespace(
        get_world_size=lambda *a, **k: 1,
        get_rank=lambda *a, **k: 0,
        all_gather=lambda *a, **k: None,   # no-op
        is_initialized=lambda *a, **k: False,
    )

from muon import Muon 

class PINNMuonOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr_muon: float = 1e-3,
        weight_decay_muon: float = 0.0,
        momentum_muon: float = 0.95,
        lr_other: float = 1e-3,
        weight_decay_other: float = 0.0,
        betas_other: tuple[float, float] = (0.9, 0.95),
    ) -> None:
        self.model = model
        self.matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        self.other_params  = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

        self.muon_opt = Muon(
            self.matrix_params,
            lr=lr_muon,
            weight_decay=weight_decay_muon,
            momentum=momentum_muon,
        ) if self.matrix_params else None

        self.adam_opt = torch.optim.AdamW(
            self.other_params,
            lr=lr_other,
            betas=betas_other,
            weight_decay=weight_decay_other,
        ) if self.other_params else None

    def zero_grad(self) -> None:
        if self.muon_opt: self.muon_opt.zero_grad()
        if self.adam_opt: self.adam_opt.zero_grad()

    def step(self) -> None:
        if self.muon_opt: self.muon_opt.step()
        if self.adam_opt: self.adam_opt.step()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict() if self.muon_opt else {},
            "adam": self.adam_opt.state_dict() if self.adam_opt else {},
        }

    def load_state_dict(self, d: dict) -> None:
        if self.muon_opt and "muon" in d: self.muon_opt.load_state_dict(d["muon"])
        if self.adam_opt and "adam" in d: self.adam_opt.load_state_dict(d["adam"])