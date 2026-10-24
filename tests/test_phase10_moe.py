import numpy as np
import torch

from src.phase10_moe import _masked_weights


def test_masked_weights_are_normalized_and_unavailable_views_are_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    availability = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    weights = _masked_weights(logits, availability)
    assert np.allclose(weights.sum(dim=1).numpy(), [1.0, 1.0])
    assert weights[0, 1].item() == 0.0
    assert torch.isfinite(weights).all()


def test_all_missing_views_fail_explicitly():
    try:
        _masked_weights(
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
        )
    except RuntimeError:
        return
    raise AssertionError("all-missing modalities must raise RuntimeError")
