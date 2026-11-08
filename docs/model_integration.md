# GAC LLaVA-Next / FastV adapter integration

The public package does not guess visual-token ownership or emulate processor geometry. An adapter must obtain these quantities from the official processor and the actual model topology.

## One-time clean reference

For a frozen clean contract, record official processor input and page SHA256 values, image-token span and auxiliary-slot positions, natural clean FastV keep set \(S_c\), and `C = S_c intersect auxiliary_slot` using pre-pruning original indices. Capture B3 clean references for every required prefix. These references remain fixed through the run.

## Per-step methods

`native_gradient(pixels)` uses current pixels, the natural selector, and the frozen answer-score loss. `slot_gradient(pixels)` must:

1. capture current B3 pre-pruning hidden state;
2. replace only `C` in the fixed-clean B3 state with current hidden state without detaching that path;
3. force complete `S_c` only inside this diagnostic branch;
4. score online answer-competition sequences from current pixels; and
5. return the pixel gradient of the selected sequence objective.

`save_reload(pixels, step)` must return the reloaded pixel tensor, output path,
and SHA256. `fresh_replay(saved)` must load that exact file in a new model
process. At endpoint evaluation remove forced keep sets and state patches;
evaluate natural FastV only.

## Assertions

- prompt, `input_ids`, image sizes, and visual span match the frozen contract;
- candidate positions are nonempty and use original pre-pruning indices;
- non-target hidden values, masks, positions, and caches remain unchanged;
- the hook executes only for the intended prefill event and is always removed;
- natural generated logits and keep set match a direct model call; and
- every next iteration reloads the saved image.
- clean-relative L-infinity projection uses the saved/reloaded step-0 tensor,
  not an earlier in-memory approximation.

With `use_cache=False`, later output tokens recompute their prefix. First-answer B3 injection is therefore an initial-decision intervention, not persistent activation replacement.
