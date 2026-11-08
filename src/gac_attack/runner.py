"""Reusable GAC attack loop; model-specific FastV operations live in an adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .engine import AttackConfig, run_step


@dataclass(frozen=True)
class AttackObservation:
    raw_output: str
    tokens: tuple[int, ...]
    status: str
    keep_hash: str


@dataclass(frozen=True)
class SavedImage:
    """The exact saved-and-reloaded image used by the next model call."""

    pixels: torch.Tensor
    path: str
    sha256: str


@dataclass(frozen=True)
class ReplayResult:
    """Result from a new-process endpoint replay supplied by the adapter."""

    passed: bool
    raw_output: str
    tokens: tuple[int, ...]
    status: str
    png_sha256: str


@dataclass(frozen=True)
class TrajectoryStep:
    step: int
    pixels: torch.Tensor
    observation: AttackObservation
    projection_count: int
    saved: SavedImage
    replay: ReplayResult | None = None
    endpoint_accepted: bool = False


class GACAdapter(Protocol):
    """Bridge a specific FastV implementation to the public attack loop.

    `slot_gradient` must implement the fixed clean reference, clean keep set,
    and online candidate selection defined by the chosen release configuration.
    `save_reload` must write and reload the same PNG format used at endpoint
    evaluation. `fresh_replay` must create a new model process and return a
    result for that exact saved-image SHA256.
    """

    def observe(self, pixels: torch.Tensor) -> AttackObservation: ...
    def native_gradient(self, pixels: torch.Tensor) -> torch.Tensor: ...
    def slot_gradient(self, pixels: torch.Tensor) -> torch.Tensor: ...
    def save_reload(self, pixels: torch.Tensor, step: int) -> SavedImage: ...
    def fresh_replay(self, saved: SavedImage) -> ReplayResult: ...


class AttackRunner:
    def __init__(self, adapter: GACAdapter, config: AttackConfig):
        self.adapter = adapter
        self.config = config

    def run(self, clean: torch.Tensor, editable_mask: torch.Tensor, *, max_steps: int) -> list[TrajectoryStep]:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        clean_saved = self.adapter.save_reload(clean, 0)
        clean_reference = clean_saved.pixels
        current = clean_reference
        initial_observation = self.adapter.observe(current)
        trajectory = [
            TrajectoryStep(
                step=0,
                pixels=current,
                observation=initial_observation,
                projection_count=0,
                saved=clean_saved,
            )
        ]
        for step in range(1, max_steps + 1):
            native = self.adapter.native_gradient(current)
            slot = self.adapter.slot_gradient(current)
            if not bool(torch.isfinite(native).all() and torch.isfinite(slot).all()):
                raise RuntimeError("NONFINITE_GRADIENT")
            update = run_step(
                current=current,
                clean=clean_reference,
                native_gradient=native,
                slot_gradient=slot,
                editable_mask=editable_mask,
                config=self.config,
            )
            saved = self.adapter.save_reload(update.next_pixels, step)
            current = saved.pixels
            observation = self.adapter.observe(current)
            replay = self.adapter.fresh_replay(saved) if observation.status == "WRONG" else None
            endpoint_accepted = bool(
                replay is not None
                and replay.passed
                and replay.status == "WRONG"
                and replay.png_sha256 == saved.sha256
            )
            trajectory.append(
                TrajectoryStep(
                    step=step,
                    pixels=current,
                    observation=observation,
                    projection_count=update.projection_count,
                    saved=saved,
                    replay=replay,
                    endpoint_accepted=endpoint_accepted,
                )
            )
            if endpoint_accepted:
                break
        return trajectory
