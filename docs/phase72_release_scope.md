# Phase72 Sequence-Slot release scope

## Goal

Provide a readable, configuration-driven release of the Phase72
Sequence-Slot objective for LLaVA-Next/Mistral FastV research.

## Included

- A frozen input-contract schema with SHA256 validation.
- The strict ANLS-style endpoint evaluator used by the project.
- Pure Sequence-Slot objective and gradient-combination utilities.
- A model-agnostic attack loop interface, Mistral FastV hook primitive, and a
  LLaVA-Next/FastV adapter boundary for downstream integration.
- CLI validation and example configuration files.

## Excluded

- Model weights, MP-DocVQA pages, answer strings, input contracts, PNGs,
  attack trajectories, keep sets, and all historical experiment results.
- Project-specific absolute paths and scripts that replay private artifacts.

## Compatibility promise

`phase72_legacy` is the default objective mode. It preserves the historical
surrogate-gradient aggregation convention, including the candidate-gradient
temperature factor. `canonical` is exposed separately for future validation
and is never presented as a reproduction of Phase72 results.

This release does not include a concrete end-to-end LLaVA-Next adapter because
official preprocessing and B3 injection are version-sensitive. It is a public
method implementation with a tested adapter contract, not a turnkey claim of
historical endpoint reproduction.

## Endpoint rule

An attack endpoint is accepted only after saved-PNG reload and an independent
fresh-process replay produce a natural FastV `WRONG` output under the frozen
evaluator.  Internal forced-keep or activation-patching results are diagnostic
only and are not attack endpoints.
