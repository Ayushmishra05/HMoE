import numpy as np

from src.phase12_fusion_question import _evaluate


def test_evaluate_reports_pairwise_stability_for_fixed_population():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(100, 4)).astype(np.float32)
    import polars as pl

    aligned = pl.DataFrame(
        {
            "customer_unique_id": [str(i) for i in range(100)],
            "split": ["train"] * 70 + ["test"] * 30,
        }
    )
    result = _evaluate(matrix, aligned)
    assert result["dimension"] == 4
    assert len(result["test_stability_pairwise_ari"]) == 10
    assert len(result["test_cluster_counts"]) == 5
