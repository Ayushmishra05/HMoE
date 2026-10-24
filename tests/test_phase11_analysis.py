from src.phase11_analysis import _wilson


def test_wilson_interval_contains_observed_rate():
    lower, upper = _wilson(10, 100)
    assert 0 <= lower < 0.1 < upper <= 1


def test_wilson_zero_successes_is_finite():
    lower, upper = _wilson(0, 100)
    assert lower >= 0
    assert upper > lower
