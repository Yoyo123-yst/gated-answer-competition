"""Budget-safe pixel update helpers."""

from __future__ import annotations

import torch


def project_linf(proposal: torch.Tensor, clean: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    bounded = torch.maximum(torch.minimum(proposal, clean + epsilon), clean - epsilon)
    return bounded.clamp(0.0, 1.0)
