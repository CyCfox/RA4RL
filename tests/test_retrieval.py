import numpy as np
import pytest

from ra4rl.retrieval.teacher import mix, rrf


def test_rrf_rank_one_based_and_stable_ties():
    assert rrf(["b", "a"], ["a", "b"], limit=2) == ["a", "b"]
    assert rrf(["z", "x"], ["x", "b"], limit=3)[0] == "x"


def test_logits_softmax_mixture_and_invalid_teachers():
    distributions = np.eye(7)[:2]
    assert np.allclose(mix([1000, 1000], distributions), [.5, .5, 0, 0, 0, 0, 0])
    expected = 1 / (1 + np.exp(-2))
    assert mix([2, 0], distributions)[0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        mix([float("nan"), 0], distributions)
    with pytest.raises(ValueError):
        mix([0, 1], np.zeros((2, 7)))
