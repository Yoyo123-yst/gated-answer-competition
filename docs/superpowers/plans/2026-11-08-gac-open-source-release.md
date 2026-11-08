# GAC Open-Source Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release a clean, testable implementation of Gated Answer Competition without modifying or publishing the historical experiment workspace.

**Architecture:** The release isolates deterministic contracts, evaluation, objective mathematics, and pixel updates in a dependency-light package. A model adapter owns official LLaVA-Next/FastV forward details so historical artifacts and absolute paths never enter the repository.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

**Spec:** `docs/gac_release_scope.md`

## Global Constraints

- Work only in this standalone repository.
- Keep `gac_legacy` as the default objective mode.
- Exclude datasets, images, model weights, attack artifacts, absolute paths, and private contracts.
- Treat saved/reloaded natural FastV plus fresh replay as the only attack endpoint.

## Review Focus

- Changed clean page bytes must invalidate an input contract.
- Exact, ambiguous, wrong, and invalid evaluator statuses must remain separate.
- Legacy and canonical objective modes must differ only by their documented temperature factor.
- Pixel updates must honor both editable masks and clean-relative L-infinity bounds.
- CLI validation must run without loading a model.

### Task 1: Deterministic core

**Files:** `src/gac_attack/{contracts,evaluator,objective,update,engine}.py`, `tests/test_*.py`

- [x] Write failing tests for contracts, endpoint evaluation, objective modes, and projection.
- [x] Implement the dependency-light core.
- [x] Run the complete unit suite.

### Task 2: Public execution interface

**Files:** `src/gac_attack/{runner,cli}.py`, `tests/test_{runner,cli}.py`

- [x] Write failing tests using a deterministic toy adapter.
- [x] Implement runner and clean-contract validation CLI.
- [x] Run tests and CLI smoke check.

### Task 3: Release material

**Files:** `README.md`, `docs/`, packaging metadata, license, ignore rules.

- [x] Document assumptions, compatibility boundary, model integration, and endpoint gate.
- [x] Verify and audit tracked files.
- [x] Commit and publish.
