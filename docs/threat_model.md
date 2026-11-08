# Threat model and claim boundary

The attacker may change pixels only in a pre-declared auxiliary document-image region. The perturbation is bounded by L-infinity `8/255`; the historical release configuration uses at most 13 sign-gradient updates of `1/255`.

The defender evaluates a saved/reloaded attacked image with the natural FastV selector, unchanged prompt, processor, decoding parameters, and strict frozen evaluator. No forced keep set, activation patch, or teacher-forcing path is permitted during endpoint evaluation.

This package implements Gated Answer Competition, not a claim that every FastV model or document dataset is vulnerable. Any rate or generality claim requires a separately frozen cohort, per-sample contracts, saved-image endpoint records, and fresh-process replays.
