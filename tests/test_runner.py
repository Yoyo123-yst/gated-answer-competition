import torch

from sequence_slot_attack.engine import AttackConfig
from sequence_slot_attack.runner import AttackObservation, AttackRunner, ReplayResult, SavedImage


class ToyAdapter:
    def __init__(self):
        self.clean = torch.full((1, 1, 1, 2), 0.5)
        self.mask = torch.tensor([[[[True, False]]]])
        self.states = []

    def observe(self, pixels):
        self.states.append(pixels.detach().clone())
        return AttackObservation(raw_output="correct", tokens=(1, 2), status="CORRECT", keep_hash="toy")

    def native_gradient(self, pixels):
        return torch.ones_like(pixels)

    def slot_gradient(self, pixels):
        return torch.ones_like(pixels)

    def save_reload(self, pixels, step):
        quantized = (pixels.detach().clone() * 10).round() / 10
        return SavedImage(pixels=quantized, path=f"step_{step:03d}.png", sha256=f"hash-{step}")

    def fresh_replay(self, saved):
        return ReplayResult(passed=True, raw_output="wrong", tokens=(9,), status="WRONG", png_sha256=saved.sha256)


def test_runner_keeps_two_independent_saved_steps():
    adapter = ToyAdapter()
    runner = AttackRunner(adapter, AttackConfig(epsilon=0.1, alpha=0.05, lambda_slot=0.5))

    trajectory = runner.run(adapter.clean, adapter.mask, max_steps=2)

    assert [item.step for item in trajectory] == [0, 1, 2]
    assert torch.allclose(trajectory[-1].pixels, torch.tensor([[[[0.4, 0.5]]]]))
    assert all(item.observation.status == "CORRECT" for item in trajectory)


def test_runner_uses_reloaded_clean_as_its_linf_reference():
    adapter = ToyAdapter()
    runner = AttackRunner(adapter, AttackConfig(epsilon=0.1, alpha=0.05, lambda_slot=0.0))

    trajectory = runner.run(torch.tensor([[[[0.54, 0.5]]]]), adapter.mask, max_steps=1)

    # save_reload quantizes 0.54 to 0.5; the update must be budgeted from 0.5.
    assert torch.allclose(trajectory[-1].pixels, torch.tensor([[[[0.4, 0.5]]]]))


def test_runner_records_a_verified_replay_before_accepting_wrong_endpoint():
    class WrongAdapter(ToyAdapter):
        def observe(self, pixels):
            return AttackObservation(raw_output="wrong", tokens=(9,), status="WRONG", keep_hash="toy")

    adapter = WrongAdapter()
    trajectory = AttackRunner(adapter, AttackConfig()).run(adapter.clean, adapter.mask, max_steps=2)

    assert len(trajectory) == 2
    assert trajectory[-1].saved.path == "step_001.png"
    assert trajectory[-1].replay is not None
    assert trajectory[-1].replay.passed is True
