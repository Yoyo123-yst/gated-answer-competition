import torch

from sequence_slot_attack.objective import combine_gradients, sequence_slot_gradient
from sequence_slot_attack.update import project_linf


def test_legacy_and_canonical_modes_only_differ_in_temperature_factor():
    g_plus = torch.tensor([2.0, 4.0])
    candidate_grads = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    scores = torch.tensor([0.2, 0.4])

    legacy = sequence_slot_gradient(g_plus, candidate_grads, scores, tau=0.5, mode="phase72_legacy")
    canonical = sequence_slot_gradient(g_plus, candidate_grads, scores, tau=0.5, mode="canonical")

    weighted = torch.softmax(scores / 0.5, dim=0) @ candidate_grads
    assert torch.allclose(canonical, g_plus - weighted)
    assert torch.allclose(legacy, g_plus - weighted / 0.5)


def test_combination_normalizes_each_gradient_before_weighting():
    native = torch.tensor([2.0, -2.0])
    slot = torch.tensor([4.0, 4.0])
    combined = combine_gradients(native, slot, lambda_slot=0.5)
    assert torch.allclose(combined, torch.tensor([1.5, -0.5]))


def test_projection_respects_clean_relative_and_pixel_bounds():
    clean = torch.tensor([0.0, 0.5, 1.0])
    proposal = torch.tensor([-1.0, 0.9, 2.0])
    result = project_linf(proposal, clean, epsilon=0.25)
    assert torch.allclose(result, torch.tensor([0.0, 0.75, 1.0]))
