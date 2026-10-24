"""
Automated Test Suite for Phase 5 Temporal Evaluation Framework and Anti-Leakage Invariants.
Validates:
1. Strict observation cutoff enforcement (no future timestamps in historical profiles).
2. Anti-leakage: future transactions do NOT increment historical frequency/monetary.
3. Anti-leakage: in-flight deliveries (delivered >= cutoff) do NOT leak into historical delivery metrics.
4. Anti-leakage: reviews submitted >= cutoff do NOT leak into historical review metrics.
5. Downstream ground-truth evaluation targets integrity, bounds, and consistency.
6. Deterministic SHA-256 train/val/test splitting reproducibility.
7. Experiment manifest schema and completeness.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GOLD_DIR,
    SILVER_DIR,
    DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    DEFAULT_FUTURE_EVALUATION_END,
    DEFAULT_RANDOM_SEED,
)
from src.phase5_temporal import assign_deterministic_splits, run_phase5_temporal


def test_observation_cutoff_temporal_strictness():
    """Verify that historical profiles contain ZERO timestamps on or after the observation cutoff."""
    profiles = pl.read_parquet(GOLD_DIR / "temporal_customer_profiles_cutoff.parquet")
    cutoff = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF
    
    # 1. Total cohort size
    assert len(profiles) == 57510, f"Expected 57,510 cohort customers, got {len(profiles)}"
    assert profiles["customer_unique_id"].n_unique() == 57510, "Duplicate customer_unique_id in cohort"
    
    # 2. Strict cutoff on purchase timestamps
    assert profiles["last_purchase_date"].max() < cutoff, "Leakage: last_purchase_date >= observation_cutoff"
    assert profiles["first_purchase_date"].max() < cutoff, "Leakage: first_purchase_date >= observation_cutoff"
    
    # 3. Recency strictly relative to cutoff
    assert profiles["recency_days"].min() > 0.0, "Leakage: recency <= 0 days (purchase in future)"
    
    # 4. Inclusion/Exclusion validation against full dataset
    silver_orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    silver_cust = pl.read_parquet(SILVER_DIR / "silver_customers.parquet")
    order_cust = silver_orders.join(silver_cust, on="customer_id", how="inner")
    
    cust_first_purchases = order_cust.group_by("customer_unique_id").agg(
        pl.col("order_purchase_timestamp").min().alias("earliest_purchase")
    )
    excluded_custs = cust_first_purchases.filter(pl.col("earliest_purchase") >= pl.lit(cutoff))
    assert len(excluded_custs) == 38586, f"Expected 38,586 excluded future customers, got {len(excluded_custs)}"
    assert len(profiles) + len(excluded_custs) == 96096, "Cohort + Excluded must equal 96,096 total customers"
    
    # No excluded customer appears in profiles
    overlap = set(profiles["customer_unique_id"].to_list()).intersection(set(excluded_custs["customer_unique_id"].to_list()))
    assert len(overlap) == 0, f"Leakage: {len(overlap)} future customers included in historical cohort"
    print("  [PASS] test_observation_cutoff_temporal_strictness passed successfully.")


def test_anti_leakage_future_transactions():
    """Verify that future transactions do not leak into historical customer frequency or monetary metrics."""
    pit_profiles = pl.read_parquet(GOLD_DIR / "temporal_customer_profiles_cutoff.parquet")
    gold_profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    targets = pl.read_parquet(GOLD_DIR / "temporal_customer_targets.parquet")
    
    # Find customers who returned and placed orders in the future window
    retained_cust_ids = targets.filter(pl.col("future_is_retained") == 1)["customer_unique_id"].to_list()
    assert len(retained_cust_ids) == 684, f"Expected 684 retained customers, got {len(retained_cust_ids)}"
    
    # Sample a returning customer and verify that pit profile != full lifetime profile
    sample_id = retained_cust_ids[0]
    pit_row = pit_profiles.filter(pl.col("customer_unique_id") == sample_id).to_dicts()[0]
    full_row = gold_profiles.filter(pl.col("customer_unique_id") == sample_id).to_dicts()[0]
    target_row = targets.filter(pl.col("customer_unique_id") == sample_id).to_dicts()[0]
    
    # Invariant: Full frequency == PIT frequency + Future frequency
    assert full_row["frequency_orders"] >= pit_row["frequency_orders"] + target_row["future_orders_count"], (
        "Mathematical inconsistency in order frequency reconciliation"
    )
    assert pit_row["frequency_orders"] < full_row["frequency_orders"] or pit_row["monetary_value"] < full_row["monetary_value"], (
        "Leakage: PIT profile already has full lifetime transaction metrics"
    )
    print("  [PASS] test_anti_leakage_future_transactions passed successfully.")


def test_anti_leakage_in_flight_deliveries():
    """Verify that packages delivered on or after the cutoff date are treated as in-flight at cutoff."""
    pit_profiles = pl.read_parquet(GOLD_DIR / "temporal_customer_profiles_cutoff.parquet")
    silver_orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    silver_cust = pl.read_parquet(SILVER_DIR / "silver_customers.parquet")
    cutoff = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF
    
    # Identify customers whose ONLY pre-cutoff order was delivered AFTER cutoff
    order_cust = silver_orders.join(silver_cust, on="customer_id", how="inner")
    pre_orders = order_cust.filter(pl.col("order_purchase_timestamp") < pl.lit(cutoff))
    
    # Group by customer
    cust_orders = pre_orders.group_by("customer_unique_id").agg([
        pl.len().alias("n_orders"),
        pl.col("order_delivered_customer_date").min().alias("min_delivery_date"),
        pl.col("order_delivered_customer_date").max().alias("max_delivery_date"),
    ])
    
    # Customers with exactly 1 pre-cutoff order that arrived after cutoff
    in_flight_custs = cust_orders.filter(
        (pl.col("n_orders") == 1) & (pl.col("min_delivery_date") >= pl.lit(cutoff))
    )["customer_unique_id"].to_list()
    assert len(in_flight_custs) > 0, "No in-flight delivery customers found to test"
    
    # Verify in PIT profile: delivered_orders_count MUST be 0.0 and delivery_duration_days MUST be null!
    sample_in_flight = pit_profiles.filter(pl.col("customer_unique_id").is_in(in_flight_custs[:10]))
    assert (sample_in_flight["delivered_orders_count"] == 0.0).sum() == len(sample_in_flight), (
        "Leakage: in-flight delivery was counted as delivered at cutoff"
    )
    assert sample_in_flight["average_delivery_duration_days"].null_count() == len(sample_in_flight), (
        "Leakage: in-flight delivery duration leaked into historical profile"
    )
    print("  [PASS] test_anti_leakage_in_flight_deliveries passed successfully.")


def test_anti_leakage_future_reviews():
    """Verify that reviews submitted on or after cutoff do not leak into historical review metrics."""
    pit_profiles = pl.read_parquet(GOLD_DIR / "temporal_customer_profiles_cutoff.parquet")
    silver_orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    silver_cust = pl.read_parquet(SILVER_DIR / "silver_customers.parquet")
    silver_reviews = pl.read_parquet(SILVER_DIR / "silver_order_reviews.parquet")
    cutoff = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF
    
    # Identify customers whose only order was pre-cutoff, but review was created AFTER cutoff
    order_cust = silver_orders.join(silver_cust, on="customer_id", how="inner").filter(
        pl.col("order_purchase_timestamp") < pl.lit(cutoff)
    )
    cust_order_counts = order_cust.group_by("customer_unique_id").agg(pl.len().alias("n_orders"))
    single_pre_custs = cust_order_counts.filter(pl.col("n_orders") == 1)["customer_unique_id"].to_list()
    
    cust_revs = (
        silver_reviews
        .join(order_cust.filter(pl.col("customer_unique_id").is_in(single_pre_custs)), on="order_id", how="inner")
        .select(["customer_unique_id", "review_creation_date", "review_score"])
    )
    
    # Customers whose review was created after cutoff
    late_rev_custs = cust_revs.filter(pl.col("review_creation_date") >= pl.lit(cutoff))["customer_unique_id"].unique().to_list()
    assert len(late_rev_custs) > 0, "No late review customers found to test"
    
    # In PIT profile, these customers MUST have review_count == 0 and average_review_score is null!
    sample_late_rev = pit_profiles.filter(pl.col("customer_unique_id").is_in(late_rev_custs[:10]))
    assert (sample_late_rev["review_count"] == 0).sum() == len(sample_late_rev), (
        "Leakage: future review was counted in historical review_count"
    )
    assert sample_late_rev["average_review_score"].null_count() == len(sample_late_rev), (
        "Leakage: future review score leaked into historical average_review_score"
    )
    print("  [PASS] test_anti_leakage_future_reviews passed successfully.")


def test_future_evaluation_targets_integrity():
    """Verify integrity, types, bounds, and consistency of downstream evaluation targets."""
    targets = pl.read_parquet(GOLD_DIR / "temporal_customer_targets.parquet")
    
    # 1. Row count matches cohort
    assert len(targets) == 57510, f"Expected 57,510 targets, got {len(targets)}"
    assert targets["customer_unique_id"].n_unique() == 57510, "Duplicate customer in targets"
    
    # 2. Binary flags consistency
    assert (targets["future_is_retained"] + targets["future_is_churned"] == 1).sum() == len(targets), (
        "Inconsistency: future_is_retained and future_is_churned must be strictly complementary"
    )
    assert ((targets["future_orders_count"] > 0) == (targets["future_is_retained"] == 1)).sum() == len(targets), (
        "future_orders_count > 0 must exactly equal future_is_retained == 1"
    )
    
    # 3. Retention rate check (~1.19%)
    retained = targets["future_is_retained"].sum()
    assert retained == 684, f"Expected 684 retained customers, got {retained}"
    
    # 4. Value bounds
    assert targets["future_monetary_value"].min() >= 0.0, "Negative future monetary value found"
    assert targets["future_items_count"].min() >= 0, "Negative future items count found"
    
    # 5. Days to next order bounds (must be within future evaluation window: 0 to 184 days)
    retained_targets = targets.filter(pl.col("future_is_retained") == 1)
    assert retained_targets["future_days_to_next_order"].null_count() == 0, "Retained customer missing days to next order"
    assert retained_targets["future_days_to_next_order"].min() >= 0.0, "Negative days to next order"
    assert retained_targets["future_days_to_next_order"].max() <= 184.0, "Days to next order exceeds future window"
    
    churned_targets = targets.filter(pl.col("future_is_churned") == 1)
    assert churned_targets["future_days_to_next_order"].null_count() == len(churned_targets), (
        "Churned customer should have null days to next order"
    )
    print("  [PASS] test_future_evaluation_targets_integrity passed successfully.")


def test_deterministic_splits_reproducibility():
    """Verify that deterministic SHA-256 splits are reproducible and leak-free."""
    cohort = pl.read_parquet(GOLD_DIR / "temporal_customer_cohort.parquet")
    
    # 1. Total count
    assert len(cohort) == 57510, "Cohort length mismatch"
    assert "split" in cohort.columns, "split column missing in cohort dataset"
    
    # 2. Check split distribution
    train_n = (cohort["split"] == "train").sum()
    val_n = (cohort["split"] == "val").sum()
    test_n = (cohort["split"] == "test").sum()
    assert train_n + val_n + test_n == 57510, "Splits do not sum to total cohort"
    
    # Expected approximate 70/15/15: Train ~40k, Val ~8.6k, Test ~8.6k
    assert 0.68 <= train_n / 57510 <= 0.72, f"Train split outside 70% bounds: {train_n/57510:.3f}"
    assert 0.13 <= val_n / 57510 <= 0.17, f"Val split outside 15% bounds: {val_n/57510:.3f}"
    assert 0.13 <= test_n / 57510 <= 0.17, f"Test split outside 15% bounds: {test_n/57510:.3f}"
    
    # 3. Reproducibility test: re-run split with same seed
    splits_re = assign_deterministic_splits(cohort["customer_unique_id"], seed=DEFAULT_RANDOM_SEED)
    assert (cohort["split"] == splits_re["split"]).sum() == len(cohort), "Splits not 100% reproducible with same seed"
    
    # Different seed produces different assignments
    splits_diff = assign_deterministic_splits(cohort["customer_unique_id"], seed=999)
    diff_count = (cohort["split"] != splits_diff["split"]).sum()
    assert diff_count > 10000, "Different seed did not produce distinct split assignments"
    print("  [PASS] test_deterministic_splits_reproducibility passed successfully.")


def test_experiment_manifest_schema_and_integrity():
    """Verify that experiment_manifest.json exists and contains all required audit metadata."""
    manifest_path = GOLD_DIR / "experiment_manifest.json"
    assert manifest_path.exists(), "experiment_manifest.json does not exist"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    required_keys = [
        "manifest_version", "created_at", "dataset_version", "feature_set_version",
        "temporal_parameters", "reproducibility", "cohort_cardinalities",
        "split_cardinalities", "anti_leakage_guarantees"
    ]
    for key in required_keys:
        assert key in manifest, f"Key '{key}' missing from experiment_manifest.json"
        
    assert manifest["cohort_cardinalities"]["cohort_customers"] == 57510, "Cohort count mismatch in manifest"
    assert manifest["cohort_cardinalities"]["excluded_future_customers"] == 38586, "Excluded count mismatch in manifest"
    assert manifest["cohort_cardinalities"]["retained_customers_in_future"] == 684, "Retained count mismatch in manifest"
    assert manifest["reproducibility"]["random_seed"] == DEFAULT_RANDOM_SEED, "Seed mismatch in manifest"
    print("  [PASS] test_experiment_manifest_schema_and_integrity passed successfully.")


def run_all_phase5_tests():
    print("=" * 70)
    print("RUNNING PHASE 5 TEMPORAL EVALUATION & LEAKAGE TEST SUITE")
    print("=" * 70)
    test_observation_cutoff_temporal_strictness()
    test_anti_leakage_future_transactions()
    test_anti_leakage_in_flight_deliveries()
    test_anti_leakage_future_reviews()
    test_future_evaluation_targets_integrity()
    test_deterministic_splits_reproducibility()
    test_experiment_manifest_schema_and_integrity()
    print("=" * 70)
    print("ALL PHASE 5 TESTS PASSED SUCCESSFULLY (ZERO LEAKAGE DETECTED)!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_phase5_tests()
