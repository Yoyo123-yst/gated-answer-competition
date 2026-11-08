# Sequence-Slot FastV Attack

Clean research release of the **Phase72 Sequence-Slot** objective for visual-token-pruned LLaVA-Next/Mistral systems.

The method updates only a frozen auxiliary-image region. At each step it combines the ordinary natural-FastV answer-score gradient with a selection-conditioned sequence gradient computed through a fixed clean selection reference and an online candidate subset of visual states.

This repository contains code, schemas, tests, and integration interfaces. It deliberately does **not** contain model weights, MP-DocVQA documents, answer labels, clean contracts, attack PNGs, keep sets, or historical experiment outputs.

## What is frozen from Phase72

| setting | value |
| --- | --- |
| pixel budget | `epsilon = 8/255` |
| update size | `alpha = 1/255` |
| maximum updates | `13` |
| auxiliary weight | `lambda_slot = 0.50` |
| sequence temperature | `tau = 0.20` |
| FastV topology | layer-2 scoring, layer-3 pruning |
| editable region | frozen auxiliary input slot only |
| endpoint | saved/reloaded natural FastV `WRONG` + fresh-process replay |

`phase72_legacy` is the default objective mode. It preserves the historical candidate-gradient temperature factor as a **surrogate gradient**. `canonical` is an explicit separate research option and must not be called a Phase72 reproduction without new validation.

## Install

```bash
git clone https://github.com/Yoyo123-yst/sequence-slot-fastv-attack.git
cd sequence-slot-fastv-attack
python -m pip install -e '.[dev]'
pytest -q
```

For LLaVA-Next model integration, install the optional dependencies:

```bash
python -m pip install -e '.[llava-next]'
```

## Clean input contract

Every run starts from a clean-only contract. A public example is available at [`examples/contract.example.json`](examples/contract.example.json).

```bash
sequence-slot validate-contract --contract examples/contract.example.json --root examples
```

Replace the example page path and placeholder SHA256 first; the provided file is a schema template, not a runnable document. The command validates only declared page paths and SHA256 values. Do not add attack results, keep sets, first-error steps, or outcome labels to this file.

## Integration model

`AttackRunner` is model-agnostic. Supply a `SequenceSlotAdapter` that performs the official model/processor forward and implements four operations:

1. `observe(pixels)`: natural FastV decode, evaluator result, and keep hash;
2. `native_gradient(pixels)`: normal natural-FastV answer-score gradient;
3. `slot_gradient(pixels)`: online Sequence-Slot gradient;
4. `save_reload(pixels)`: PNG serialization and reload.

The runner owns the update rule, L-infinity projection, saved-image iteration, SHA256 records, and endpoint replay gate. The adapter owns official preprocessing, image token span, layer-2 selector, layer-3 pruning, B3 capture/injection, generation parity, actual PNG I/O, and fresh-process replay. See [`docs/model_integration.md`](docs/model_integration.md).

For a compatible Mistral decoder, `sequence_slot_attack.fastv_mistral` includes
the layer-2 scoring/layer-3 pruning hook and trace collection primitive. It is
not a replacement for a processor or a model-version parity gate. **This
release intentionally does not ship a concrete end-to-end LLaVA-Next adapter:**
official preprocessing and B3 injection must be bound to the exact model and
Transformers version in use. It is therefore a reusable method release, not a
claim that `pip install` alone reproduces historical Phase72 endpoints.

## Objective

For clean sequence score \(S^+\), candidate sequence scores \(S^-_i\), and temperature \(\tau\), the auxiliary objective is

\[
L_{seq}=S^+-\tau\log\sum_i\exp(S^-_i/\tau).
\]

In `canonical` mode, the candidate contribution is the derivative of this
formula. The default `phase72_legacy` mode instead uses the frozen historical
surrogate

\[
g_{slot}=g^+-\frac{1}{\tau}\sum_i\operatorname{softmax}(S^-/\tau)_i g_i^-.
\]

The update follows the historical score-descending convention:

\[
g=\frac{g_{native}}{\operatorname{mean}|g_{native}|}+\lambda_{slot}\frac{g_{slot}}{\operatorname{mean}|g_{slot}|},\qquad x_{t+1}=\Pi_{\lVert x-x_0\rVert_\infty\le\epsilon}(x_t-\alpha\operatorname{sign}(g)).
\]

## Endpoint hygiene

Internal forced-keep and activation-patching results are diagnostics, not attack success. A reportable endpoint requires a saved PNG and SHA256, reload through the declared image path, natural FastV evaluation as strict `WRONG`, and a matching fresh-process replay. The runner marks an endpoint accepted only when the adapter reports all of those conditions.

## Development

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall -q src
```

## License and attribution

Released under the MIT license. This code does not ship upstream LLaVA-Next, Transformers, FastV, or MP-DocVQA assets. Cite and follow licenses for all upstream models, datasets, and implementations.
