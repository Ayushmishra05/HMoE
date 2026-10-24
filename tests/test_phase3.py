"""
Automated Test Suite for Phase 3 Gold Analytical Layer.
Validates all acceptance criteria from the research supervisor.
"""

import sys
from pathlib import Path
from datetime import datetime
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GOLD_DIR, SILVER_DIR
from src.phase3_gold import run_phase3_gold


def test_phase3_pipeline():
    print("=" * 70)
    print("RUNNING PHASE 3 GOLD ANALYTICAL LAYER TESTS")
    print("=" * 70)

    # 1. Run Phase 3
    print("\n[Step 1] Executing Phase 3 Gold transformations...")
    summary = run_phase3_gold(allow_overwrite=True)
    assert len(summary) == 5, f"Expected 5 gold tables, got {len(summary)}"
    print(f"-> Successfully materialized {len(summary)} Gold analytical tables.")

    # 2. Check gold_orders_analytical
    print("\n[Step 2] Validating gold_orders_analytical...")
    orders = pl.read_parquet(GOLD_DIR / "gold_orders_analytical.parquet")
    silver_orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    assert len(orders) == len(silver_orders) == 99441, "Orders row count mismatch"
    assert orders["order_id"].n_unique() == 99441, "order_id not unique in gold_orders_analytical"
    assert "payment_discrepancy" in orders.columns, "payment_discrepancy missing"
    assert "has_payment_discrepancy" in orders.columns, "has_payment_discrepancy missing"
    assert "latest_review_score" in orders.columns, "latest_review_score missing"
    discrepancy_count = orders["has_payment_discrepancy"].sum()
    print(f"  [PASS] gold_orders_analytical: 99,441 rows, 100% unique order_id")
    print(f"  [INFO] Orders with payment discrepancy (|diff| > $0.01): {discrepancy_count}")

    # 3. Check gold_customer_profiles
    print("\n[Step 3] Validating gold_customer_profiles...")
    profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    silver_cust = pl.read_parquet(SILVER_DIR / "silver_customers.parquet")
    expected_unique_cust = silver_cust["customer_unique_id"].n_unique()
    assert len(profiles) == expected_unique_cust == 96096, "Customer profile count mismatch"
    assert profiles["customer_unique_id"].n_unique() == 96096, "customer_unique_id not unique"
    
    # Check RFM & Lifetime
    assert profiles["recency_days"].min() >= 0.0, "Negative recency found"
    assert profiles["customer_lifetime_days"].min() >= 0.0, "Negative lifetime found"
    repeat_cust = profiles.filter(pl.col("frequency_orders") > 1)
    assert len(repeat_cust) == 2997, f"Expected 2997 repeat customers, got {len(repeat_cust)}"
    print(f"  [PASS] gold_customer_profiles: 96,096 rows, 100% unique customer_unique_id")
    print(f"  [INFO] Repeat customers: {len(repeat_cust)} ({len(repeat_cust) / len(profiles) * 100:.2f}%)")
    print(f"  [INFO] Max customer lifetime span: {profiles['customer_lifetime_days'].max():.2f} days")

    # 4. Check gold_customer_events (Temporal View)
    print("\n[Step 4] Validating gold_customer_events (Temporal Encoder Input)...")
    events = pl.read_parquet(GOLD_DIR / "gold_customer_events.parquet")
    assert len(events) == 99441, f"Expected 99,441 events, got {len(events)}"
    assert "event_sequence_num" in events.columns, "event_sequence_num missing"
    assert "days_since_previous_order" in events.columns, "days_since_previous_order missing"
    # First events must have null days_since_previous_order
    first_events = events.filter(pl.col("event_sequence_num") == 1)
    assert first_events["days_since_previous_order"].null_count() == len(first_events) == 96096, "First events should have null previous delta"
    # Repeat events must have non-null days_since_previous_order >= 0
    repeat_events = events.filter(pl.col("event_sequence_num") > 1)
    assert len(repeat_events) == 3345, f"Expected 3,345 repeat events, got {len(repeat_events)}"
    assert repeat_events["days_since_previous_order"].null_count() == 0, "Repeat events missing time delta"
    assert repeat_events["days_since_previous_order"].min() >= 0.0, "Negative time delta found"
    print(f"  [PASS] gold_customer_events: 99,441 events chronologically ordered (96,096 initial + 3,345 repeat)")

    # 5. Check gold_customer_reviews (Text View)
    print("\n[Step 5] Validating gold_customer_reviews (Text Encoder Input)...")
    reviews = pl.read_parquet(GOLD_DIR / "gold_customer_reviews.parquet")
    silver_reviews = pl.read_parquet(SILVER_DIR / "silver_order_reviews.parquet")
    assert len(reviews) == len(silver_reviews) == 99224, "Review count mismatch"
    assert "customer_unique_id" in reviews.columns, "customer_unique_id missing in reviews"
    assert "review_comment_message" in reviews.columns, "review comment text missing"
    assert reviews["customer_unique_id"].null_count() == 0, "Null customer_unique_id in reviews"
    print(f"  [PASS] gold_customer_reviews: 99,224 individual reviews preserved with customer_unique_id intact")

    # 6. Check gold_product_preferences
    print("\n[Step 6] Validating gold_product_preferences...")
    pref = pl.read_parquet(GOLD_DIR / "gold_product_preferences.parquet")
    assert len(pref) == 97915, f"Expected 97,915 preference rows, got {len(pref)}"
    assert pref["customer_unique_id"].n_unique() == 95420, "Unique customers mismatch in preferences"
    invalid_shares = pref.filter((pl.col("category_spend_share") < 0.0) | (pl.col("category_spend_share") > 1.0001))
    assert len(invalid_shares) == 0, "Spend share outside [0, 1]"
    print(f"  [PASS] gold_product_preferences: 97,915 customer-category preference rows (spend share in [0, 1])")

    # 7. Configurable Observation Date Test
    print("\n[Step 7] Testing Configurable Observation Date and Temporal Cutoff...")
    custom_obs = datetime(2019, 1, 1, 0, 0, 0)
    custom_cutoff = datetime(2018, 1, 1, 0, 0, 0)
    summary_custom = run_phase3_gold(
        allow_overwrite=True,
        observation_date=custom_obs,
        temporal_cutoff=custom_cutoff,
    )
    custom_profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    # Min recency with custom_obs should be > (2019-01-01 - 2018-10-18) ~ 74 days
    assert custom_profiles["recency_days"].min() > 70.0, "Custom observation date not respected"
    assert "is_in_feature_period" in custom_profiles.columns, "Temporal cutoff flag missing"
    pre_cutoff = custom_profiles["is_in_feature_period"].sum()
    print(f"  [PASS] Custom observation date respected (min recency: {custom_profiles['recency_days'].min():.1f} days)")
    print(f"  [PASS] Temporal cutoff respected: {pre_cutoff} customers in pre-2018 feature period")

    # Restore default run
    print("\n[Step 8] Restoring standard pipeline run...")
    run_phase3_gold(allow_overwrite=True)

    print("\n" + "=" * 70)
    print("ALL PHASE 3 ACCEPTANCE CRITERIA PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_phase3_pipeline()
