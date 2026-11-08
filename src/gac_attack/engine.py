"""Model-independent one-step Gated Answer Competition update mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .objective import combine_gradients
from .update import project_linf


@dataclass(frozen=True)
class AttackConfig:
    epsilon: float = 8 / 255
    alpha: float = 1 / 255
    lambda_slot: float = 0.5


@dataclass(frozen=True)
class StepResult:
    next_pixels: torch.Tensor
    projection_count: int


def run_step(
    *,
    current: torch.Tensor,
    clean: torch.Tensor,
    native_gradient: torch.Tensor,
    slot_gradient: torch.Tensor,
    editable_mask: torch.Tensor,
    config: AttackConfig,
) -> StepResult:
    if not (current.shape == clean.shape == native_gradient.shape == slot_gradient.shape == editable_mask.shape):
        raise ValueError("current, clean, gradients, and editable_mask must have the same shape")
    combined = combine_gradients(native_gradient, slot_gradient, lambda_slot=config.lambda_slot)
    direction = torch.where(editable_mask, combined.sign(), torch.zeros_like(combined))
    proposal = current - config.alpha * direction
    next_pixels = project_linf(proposal, clean, epsilon=config.epsilon)
    projected = (proposal != next_pixels) & editable_mask
    return StepResult(next_pixels=next_pixels, projection_count=int(projected.sum().item()))
