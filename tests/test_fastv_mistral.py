import pytest
import torch

from sequence_slot_attack.fastv_mistral import visual_layout


def test_visual_layout_returns_original_image_placeholder_positions():
    input_ids = torch.tensor([[1, 99, 99, 5, 99]])
    assert visual_layout(input_ids, image_token_id=99) == [1, 2, 4]


def test_visual_layout_rejects_a_prompt_without_image_placeholders():
    with pytest.raises(RuntimeError, match="NO_VISUAL"):
        visual_layout(torch.tensor([[1, 2]]), image_token_id=99)
