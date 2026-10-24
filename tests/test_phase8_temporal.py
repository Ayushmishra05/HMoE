from datetime import datetime

import numpy as np
import polars as pl

from src.phase8_temporal import SEQUENCE_FEATURES, build_sequences


def _events() -> pl.DataFrame:
    rows = []
    for index in range(4):
        rows.append(
            {
                "customer_unique_id": "c1",
                "order_id": f"o{index}",
                "event_timestamp": datetime(2018, 1, index + 1),
                **{feature: float(index + 1) for feature in SEQUENCE_FEATURES},
            }
        )
    rows.append(
        {
            "customer_unique_id": "c1",
            "order_id": "future",
            "event_timestamp": datetime(2018, 3, 1),
            **{feature: 99.0 for feature in SEQUENCE_FEATURES},
        }
    )
    return pl.DataFrame(rows)


def test_cutoff_ordering_and_truncation():
    cohort = pl.DataFrame({"customer_unique_id": ["c1", "c2"]})
    sequences, masks, stats = build_sequences(
        _events(), cohort, max_length=2
    )
    assert sequences.shape == (2, 2, len(SEQUENCE_FEATURES))
    assert masks[0].tolist() == [1.0, 1.0]
    assert sequences[0, :, 0].tolist() == [3.0, 4.0]
    assert masks[1].tolist() == [0.0, 0.0]
    assert stats["truncated_customers"] == 1
    assert stats["cutoff_violations"] == 0


def test_tie_breaking_is_deterministic_and_values_are_finite():
    events = _events().with_columns(
        pl.when(pl.col("order_id") == "o1")
        .then(datetime(2018, 1, 1))
        .otherwise(pl.col("event_timestamp"))
        .alias("event_timestamp")
    )
    cohort = pl.DataFrame({"customer_unique_id": ["c1"]})
    first, first_mask, _ = build_sequences(events, cohort)
    second, second_mask, _ = build_sequences(events, cohort)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert np.isfinite(first).all()
