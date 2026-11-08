import torch

from sequence_slot_attack.engine import AttackConfig, run_step


def test_run_step_updates_only_editable_pixels_and_keeps_budget():
    clean = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    current = clean.clone()
    editable = torch.tensor([[[[True, False], [False, True]]]])
    config = AttackConfig(epsilon=0.1, alpha=0.05, lambda_slot=0.5)

    result = run_step(
        current=current,
        clean=clean,
        native_gradient=torch.ones_like(clean),
        slot_gradient=torch.ones_like(clean),
        editable_mask=editable,
        config=config,
    )

    assert torch.allclose(result.next_pixels, torch.tensor([[[[0.45, 0.5], [0.5, 0.45]]]]))
    assert result.projection_count == 0
