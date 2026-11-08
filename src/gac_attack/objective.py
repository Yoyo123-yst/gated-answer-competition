"""Pure math for the Gated Answer Competition auxiliary objective."""

from __future__ import annotations

import torch


VALID_OBJECTIVE_MODES = frozenset({"gac_legacy", "canonical"})


def gac_gradient(
    positive_gradient: torch.Tensor,
    candidate_gradients: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    tau: float,
    mode: str = "gac_legacy",
) -> torch.Tensor:
    """Return gradient of sequence score against the current image.

    `gac_legacy` preserves the frozen experimental implementation, whose
    candidate term includes an extra `1 / tau`.  `canonical` is the gradient
    of S+ - tau * logsumexp(S- / tau).
    """
    if mode not in VALID_OBJECTIVE_MODES:
        raise ValueError(f"unknown objective mode: {mode}")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if candidate_gradients.shape[0] != candidate_scores.numel():
        raise ValueError("candidate gradient count must equal score count")
    weights = torch.softmax(candidate_scores / tau, dim=0)
    weighted_negative = torch.tensordot(weights, candidate_gradients, dims=([0], [0]))
    if mode == "gac_legacy":
        weighted_negative = weighted_negative / tau
    return positive_gradient - weighted_negative


def combine_gradients(native_gradient: torch.Tensor, slot_gradient: torch.Tensor, *, lambda_slot: float) -> torch.Tensor:
    if lambda_slot < 0:
        raise ValueError("lambda_slot must be non-negative")
    native_scale = native_gradient.abs().mean().clamp_min(torch.finfo(native_gradient.dtype).eps)
    slot_scale = slot_gradient.abs().mean().clamp_min(torch.finfo(slot_gradient.dtype).eps)
    return native_gradient / native_scale + lambda_slot * slot_gradient / slot_scale
