"""
Automated Test Suite for Phase 4 Customer Profiling Audit Invariants.
Validates all empirical, mathematical, and architectural invariants
discovered during the Phase 4 customer profiling audit.
"""

import sys
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GOLD_DIR, SILVER_DIR


def test_review_submission_rate_invariants():
    """Verify review_submission_rate definitions, bounds, and multi-review reconciliation."""
    profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    silver_reviews = pl.read_parquet(SILVER_DIR / "silver_order_reviews.parquet")
    
    # 1. Null check: exactly 716 customers with 0 reviews
    null_rev_rate = profiles.filter(pl.col("review_submission_rate").is_null())
    assert len(null_rev_rate) == 716, f"Expected 716 null review_submission_rate, got {len(null_rev_rate)}"
    assert null_rev_rate["review_count"].max() == 0, "Non-zero review_count found with null review_submission_rate"
    
    # 2. Multi-review intensity check: values > 1.0 occur due to multi-review orders
    gt_1 = profiles.filter(pl.col("review_submission_rate") > 1.0)
    assert len(gt_1) == 447, f"Expected 447 customers with rate > 1.0, got {len(gt_1)}"
    assert gt_1["review_submission_rate"].max() <= 2.0001, "Unexpectedly high review_submission_rate (>2.0)"
    
    # 3. Orders with multiple reviews in Silver
    order_rev_counts = silver_reviews.group_by("order_id").agg(pl.len().alias("n_revs"))
    multi_rev_orders = order_rev_counts.filter(pl.col("n_revs") > 1)
    assert len(multi_rev_orders) == 547, f"Expected 547 orders with >1 review, got {len(multi_rev_orders)}"
    assert multi_rev_orders["n_revs"].sum() == 1098, f"Expected 1098 review records for multi-review orders"
    print("  [PASS] test_review_submission_rate_invariants passed successfully.")


def test_customer_lifetime_and_repeat_invariants():
    """Verify mathematical consistency between frequency, lifetime, and repeat buyer flags."""
    profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    
    # 1. Repeat customer count
    repeat_cust = profiles.filter(pl.col("frequency_orders") > 1)
    single_cust = profiles.filter(pl.col("frequency_orders") == 1)
    assert len(repeat_cust) == 2997, f"Expected 2997 repeat customers, got {len(repeat_cust)}"
    assert len(single_cust) == 93099, f"Expected 93099 single-order customers, got {len(single_cust)}"
    assert len(repeat_cust) + len(single_cust) == 96096, "Customer partition count mismatch"
    
    # 2. Single-order lifetime invariant: lifetime MUST be strictly 0.0
    assert (single_cust["customer_lifetime_days"] == 0.0).sum() == len(single_cust), (
        "Single order customer found with non-zero lifetime"
    )
    
    # 3. Repeat customer lifetime behavior
    # Exactly 257 repeat customers placed simultaneous orders (lifetime == 0.0)
    rep_zero_lt = repeat_cust.filter(pl.col("customer_lifetime_days") == 0.0)
    rep_pos_lt = repeat_cust.filter(pl.col("customer_lifetime_days") > 0.0)
    assert len(rep_zero_lt) == 257, f"Expected 257 simultaneous repeat customers, got {len(rep_zero_lt)}"
    assert len(rep_pos_lt) == 2740, f"Expected 2740 positive lifetime repeat customers, got {len(rep_pos_lt)}"
    
    # 4. Strict non-negativity
    assert profiles["customer_lifetime_days"].min() >= 0.0, "Negative customer lifetime found"
    print("  [PASS] test_customer_lifetime_and_repeat_invariants passed successfully.")


def test_customer_profile_denominators_and_zero_safety():
    """Verify that all feature denominators and ratios are mathematically safe and well-defined."""
    profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    
    # 1. Zero items protection: when items == 0, diversity ratios MUST be 0.0
    zero_items = profiles.filter(pl.col("total_items_purchased") == 0)
    assert len(zero_items) == 676, f"Expected 676 customers with 0 items, got {len(zero_items)}"
    assert (zero_items["category_diversity"] == 0.0).sum() == len(zero_items), "Non-zero category_diversity for 0 items"
    assert (zero_items["seller_diversity"] == 0.0).sum() == len(zero_items), "Non-zero seller_diversity for 0 items"
    assert (zero_items["product_diversity"] == 0.0).sum() == len(zero_items), "Non-zero product_diversity for 0 items"
    
    # 2. Zero delivered orders protection: late_delivery_ratio MUST be 0.0
    zero_delivered = profiles.filter(pl.col("delivered_orders_count") == 0.0)
    assert (zero_delivered["late_delivery_ratio"] == 0.0).sum() == len(zero_delivered), (
        "Non-zero late_delivery_ratio for 0 delivered orders"
    )
    
    # 3. Review score conditional nulls
    zero_reviews = profiles.filter(pl.col("review_count") == 0)
    assert len(zero_reviews) == 716, f"Expected 716 customers with 0 reviews, got {len(zero_reviews)}"
    assert zero_reviews["low_rating_ratio"].null_count() == 716, "low_rating_ratio should be null for 0 reviews"
    assert zero_reviews["average_review_score"].null_count() == 716, "average_review_score should be null for 0 reviews"
    assert zero_reviews["review_submission_rate"].null_count() == 716, "review_submission_rate should be null for 0 reviews"
    print("  [PASS] test_customer_profile_denominators_and_zero_safety passed successfully.")


def test_zero_monetary_population_reconciliation():
    """Verify root cause and financial reconciliation of the 0.70% zero-monetary population."""
    profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    orders = pl.read_parquet(GOLD_DIR / "gold_orders_analytical.parquet")
    silver_orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    silver_items = pl.read_parquet(SILVER_DIR / "silver_order_items.parquet")
    
    # 1. Exactly 676 customers with monetary_value == 0.0
    zero_mon = profiles.filter(pl.col("monetary_value") == 0.0)
    assert len(zero_mon) == 676, f"Expected 676 zero-monetary customers, got {len(zero_mon)}"
    
    # 2. Map to gold orders: all 685 orders from zero-monetary customers have has_items == False
    zero_cust_ids = zero_mon["customer_unique_id"].to_list()
    zero_orders = orders.filter(pl.col("customer_unique_id").is_in(zero_cust_ids))
    assert len(zero_orders) == 685, f"Expected 685 orders for zero monetary customers, got {len(zero_orders)}"
    assert (zero_orders["has_items"] == False).sum() == 685, "Zero monetary order unexpectedly has items"
    
    # 3. Total orders without items in dataset: exactly 775 orders
    no_item_orders = silver_orders.join(silver_items, on="order_id", how="anti")
    assert len(no_item_orders) == 775, f"Expected 775 silver orders without items, got {len(no_item_orders)}"
    
    # Unfulfilled status verification (canceled, unavailable, created, invoiced)
    unfulfilled_statuses = ["unavailable", "canceled", "created", "invoiced", "shipped"]
    assert zero_orders["order_status"].is_in(unfulfilled_statuses).sum() == len(zero_orders), (
        "Zero monetary order has unexpected fulfilled status"
    )
    print("  [PASS] test_zero_monetary_population_reconciliation passed successfully.")


def test_feature_redundancy_and_correlations():
    """Verify discovered collinearities and mechanical redundancies between feature pairs."""
    profiles = pl.read_parquet(GOLD_DIR / "gold_customer_profiles.parquet")
    
    # 1. Monetary value vs Monetary payment value (r > 0.98)
    r_mon_pay = profiles.select(pl.corr("monetary_value", "monetary_payment_value")).item()
    assert r_mon_pay > 0.98, f"Expected r > 0.98, got {r_mon_pay}"
    
    # 2. Monetary value vs Average order value (r > 0.98)
    r_mon_aov = profiles.select(pl.corr("monetary_value", "average_order_value")).item()
    assert r_mon_aov > 0.98, f"Expected r > 0.98, got {r_mon_aov}"
    
    # 3. Category diversity vs Seller diversity (r > 0.95)
    r_cat_sel = profiles.select(pl.corr("category_diversity", "seller_diversity")).item()
    assert r_cat_sel > 0.95, f"Expected r > 0.95, got {r_cat_sel}"
    
    # 4. Average review score vs Low rating ratio (r < -0.85)
    clean_rev = profiles.select(["average_review_score", "low_rating_ratio"]).drop_nulls()
    r_rev_low = clean_rev.select(pl.corr("average_review_score", "low_rating_ratio")).item()
    assert r_rev_low < -0.85, f"Expected r < -0.85, got {r_rev_low}"
    
    # 5. Average installments vs Max installments (r > 0.99)
    r_inst = profiles.select(pl.corr("average_installments", "max_installments")).item()
    assert r_inst > 0.99, f"Expected r > 0.99, got {r_inst}"
    print("  [PASS] test_feature_redundancy_and_correlations passed successfully.")


def run_all_audit_tests():
    print("=" * 70)
    print("RUNNING PHASE 4 AUDIT INVARIANT TESTS")
    print("=" * 70)
    test_review_submission_rate_invariants()
    test_customer_lifetime_and_repeat_invariants()
    test_customer_profile_denominators_and_zero_safety()
    test_zero_monetary_population_reconciliation()
    test_feature_redundancy_and_correlations()
    print("=" * 70)
    print("ALL PHASE 4 AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_audit_tests()
