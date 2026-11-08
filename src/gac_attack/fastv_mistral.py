"""FastV selection hooks for compatible LLaVA-Next/Mistral decoder stacks.

This module intentionally exposes selector primitives only.  Callers must
verify model-version parity before treating its keep set as an endpoint input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import torch


def visual_layout(input_ids: torch.Tensor, *, image_token_id: int) -> list[int]:
    positions = [index for index, token in enumerate(input_ids[0].tolist()) if int(token) == image_token_id]
    if not positions:
        raise RuntimeError("NO_VISUAL_PLACEHOLDER_TOKENS")
    return positions


def keep_hash(positions: list[int]) -> str:
    encoded = ",".join(str(position) for position in positions).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class FastVTrace:
    visual_positions: list[int]
    scores: torch.Tensor | None = None
    keep_history: list[list[int]] = field(default_factory=list)
    active_positions: list[int] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_fastv_mistral(
    model: Any,
    input_ids: torch.Tensor,
    *,
    score_layer: int = 2,
    prune_layer: int = 3,
    keep_ratio: float = 0.25,
    gradient_safe: bool = True,
) -> FastVTrace:
    """Install a single-prefill FastV selector for Mistral-style attention.

    The implementation requires Mistral's `position_embeddings` argument and
    should be guarded by caller-side natural-generation parity checks.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must lie in (0, 1]")
    try:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
    except ImportError as exc:  # pragma: no cover - only exercised without optional dependency
        raise RuntimeError("install transformers with the llava-next extra") from exc

    image_token_id = int(model.config.image_token_index)
    trace = FastVTrace(visual_positions=visual_layout(input_ids, image_token_id=image_token_id))
    layers = model.model.language_model.layers
    score_attention = layers[score_layer].self_attn
    pruning_block = layers[prune_layer]

    def score_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or hidden.shape[1] == 1:
            return
        embeddings = kwargs.get("position_embeddings")
        if embeddings is None:
            raise RuntimeError("MISTRAL_POSITION_EMBEDDINGS_MISSING")
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        query = module.q_proj(hidden).view(shape).transpose(1, 2)
        key = module.k_proj(hidden).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, embeddings[0], embeddings[1])
        key = repeat_kv(key, module.num_key_value_groups)
        logits = torch.matmul(query[:, :, -1:, :], key.transpose(-1, -2)) * module.scaling
        attention_mask = kwargs.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.dim() == 4:
            logits = logits + attention_mask[:, :, :, -key.shape[-2]:]
        trace.scores = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)

    def prune_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or hidden.shape[1] == 1 or trace.scores is None:
            return None
        visual = torch.tensor(trace.visual_positions, device=hidden.device, dtype=torch.long)
        scores = trace.scores.mean(dim=1)[0, -1].index_select(0, visual)
        keep_count = max(1, int(round(len(trace.visual_positions) * keep_ratio)))
        relative = torch.topk(scores, k=keep_count, sorted=True).indices
        kept = torch.sort(visual[relative]).values
        kept_list = [int(item) for item in kept.detach().cpu().tolist()]
        visual_set = set(trace.visual_positions)
        trace.active_positions = sorted([index for index in range(hidden.shape[1]) if index not in visual_set] + kept_list)
        trace.keep_history.append(kept_list)
        indices = torch.tensor(trace.active_positions, device=hidden.device, dtype=torch.long)
        pruned = hidden.index_select(1, indices).contiguous()
        position_ids = indices.unsqueeze(0)
        kwargs["position_ids"] = position_ids
        kwargs["cache_position"] = indices
        kwargs["position_embeddings"] = model.model.language_model.rotary_emb(pruned, position_ids)
        kwargs["attention_mask"] = None
        trace.scores = None
        return (pruned,), kwargs

    if gradient_safe:
        def refresh_rope(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            embeddings = kwargs.get("position_embeddings")
            if hidden is None or embeddings is None:
                return args, kwargs
            if trace.active_positions and hidden.shape[1] != embeddings[0].shape[-2]:
                positions = torch.tensor(trace.active_positions, device=hidden.device, dtype=torch.long).unsqueeze(0)
                kwargs["position_ids"] = positions
                kwargs["cache_position"] = positions[0]
                kwargs["position_embeddings"] = model.model.language_model.rotary_emb(hidden, positions)
            else:
                kwargs["position_embeddings"] = (embeddings[0].detach().clone(), embeddings[1].detach().clone())
            return args, kwargs

        trace.handles.extend(block.register_forward_pre_hook(refresh_rope, with_kwargs=True) for block in layers)
    trace.handles.append(score_attention.register_forward_hook(score_hook, with_kwargs=True))
    trace.handles.append(pruning_block.register_forward_pre_hook(prune_hook, with_kwargs=True))
    return trace
