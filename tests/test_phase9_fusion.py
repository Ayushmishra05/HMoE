import numpy as np

from src.phase9_fusion import _prepare_view, _project_views


def test_missing_view_is_zero_with_explicit_availability_policy():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    available = np.array([True, False])
    transformed, metadata = _prepare_view(
        values, np.array([True, True]), available
    )
    assert np.isfinite(transformed).all()
    assert np.array_equal(transformed[1], np.zeros(2, dtype=np.float32))
    assert metadata["available_count"] == 1


def test_projection_is_fit_on_available_training_rows():
    views = {
        "behavioral": np.array(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32
        ),
        "text": np.array(
            [[1.0, 0.0], [0.0, 0.0], [2.0, 2.0]], dtype=np.float32
        ),
    }
    availability = {
        "behavioral": np.array([1.0, 1.0, 1.0]),
        "text": np.array([1.0, 1.0, 1.0]),
    }
    projected = _project_views(
        views,
        np.array([True, True, False]),
        availability,
    )
    assert projected["behavioral"].shape == (3, 32)
    assert projected["text"].shape == (3, 32)
    assert np.isfinite(projected["text"]).all()
