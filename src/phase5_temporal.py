"""
Phase 5: Temporal Evaluation Framework, Point-in-Time Reconstruction,
and Leakage Prevention for Customer Segmentation.

Artifacts Produced:
1. temporal_customer_profiles_cutoff.parquet:
   Historical customer feature vectors strictly as of observation_cutoff.
   Zero future leakage (orders, delivery outcomes, or reviews occurring after cutoff are omitted).
2. temporal_customer_targets.parquet:
   Ground-truth downstream evaluation targets over the future evaluation window
   [observation_cutoff, future_evaluation_end), keyed by customer_unique_id.
3. temporal_customer_cohort.parquet:
   Unified modeling cohort dataset combining historical features, deterministic
   train/val/test split flags, and future utility targets.
4. experiment_manifest.json:
   Machine-readable audit manifest recording cutoff dates, random seeds,
   feature versions, dataset versions, cohort cardinalities, and split sizes.

Strict Invariants:
- Deterministic and 100% reproducible across platforms.
- Strict point-in-time feature reconstruction:
  * Orders purchased >= cutoff are excluded from profiles.
  * Deliveries occurring >= cutoff are treated as undelivered/in-flight as of cutoff.
  * Reviews submitted >= cutoff are excluded from historical review metrics.
- Reproducible, independent hash-based train/val/test assignment without data leakage.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SILVER_DIR,
    GOLD_DIR,
    DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE,
    DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    DEFAULT_FUTURE_EVALUATION_END,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
    FEATURE_SET_VERSION,
    DATASET_VERSION,
)


def filter_point_in_time_orders(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    products: pl.DataFrame,
    payments: pl.DataFrame,
    reviews: pl.DataFrame,
    observation_cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    payment_discrepancy_tolerance: float = DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE,
) -> pl.DataFrame:
    """
    Constructs point-in-time analytical orders table as of observation_cutoff.
    Strictly purges all future transactions, in-flight delivery outcomes, and future reviews.
    """
    # 1. Filter orders strictly to purchases prior to cutoff
    pit_orders = orders.filter(pl.col("order_purchase_timestamp") < pl.lit(observation_cutoff))

    # 2. Filter reviews: only reviews created strictly before cutoff are available historically
    pit_reviews = reviews.filter(pl.col("review_creation_date") < pl.lit(observation_cutoff))

    # 3. Item-level aggregations per order
    items_enriched = items.join(
        products.select(["product_id", "product_category_name_english"]),
        on="product_id",
        how="left",
    )
    item_agg = items_enriched.group_by("order_id").agg([
        pl.col("price").sum().alias("total_item_price"),
        pl.col("freight_value").sum().alias("total_freight_value"),
        pl.col("item_total_value").sum().alias("total_order_value"),
        pl.len().alias("item_count"),
        pl.col("product_id").n_unique().alias("unique_product_count"),
        pl.col("seller_id").n_unique().alias("unique_seller_count"),
        pl.col("product_category_name_english").n_unique().alias("unique_category_count"),
    ])

    # 4. Payment-level aggregations per order
    pay_agg = payments.group_by("order_id").agg([
        pl.col("payment_value").sum().alias("total_payment_value"),
        pl.col("payment_type").n_unique().alias("payment_method_count"),
        pl.col("payment_installments").max().alias("max_installments"),
    ])

    # Primary payment type per order
    primary_pay = (
        payments.group_by(["order_id", "payment_type"])
        .agg(pl.col("payment_value").sum().alias("type_val"))
        .sort(["order_id", "type_val"], descending=[False, True])
        .group_by("order_id")
        .first()
        .select([pl.col("order_id"), pl.col("payment_type").alias("primary_payment_type")])
    )

    # 5. Review-level aggregation strictly on pre-cutoff reviews
    rev_sorted = pit_reviews.sort(["review_creation_date", "review_answer_timestamp"], descending=[True, True])
    rev_agg = rev_sorted.group_by("order_id").agg([
        pl.len().alias("review_count"),
        pl.col("review_score").first().alias("latest_review_score"),
        pl.col("review_id").first().alias("latest_review_id"),
    ])

    # 6. Customer mapping
    cust_map = customers.select(["customer_id", "customer_unique_id"]).unique()

    # 7. Assemble point-in-time analytical orders
    gold_pit_orders = (
        pit_orders
        .join(cust_map, on="customer_id", how="left")
        .join(item_agg, on="order_id", how="left")
        .join(pay_agg, on="order_id", how="left")
        .join(primary_pay, on="order_id", how="left")
        .join(rev_agg, on="order_id", how="left")
        .with_columns([
            # Fill missing items/payments
            pl.col("total_item_price").fill_null(0.0),
            pl.col("total_freight_value").fill_null(0.0),
            pl.col("total_order_value").fill_null(0.0),
            pl.col("item_count").fill_null(0).cast(pl.Int32),
            pl.col("unique_product_count").fill_null(0).cast(pl.Int32),
            pl.col("unique_seller_count").fill_null(0).cast(pl.Int32),
            pl.col("unique_category_count").fill_null(0).cast(pl.Int32),

            pl.col("total_payment_value").fill_null(0.0),
            pl.col("payment_method_count").fill_null(0).cast(pl.Int32),
            pl.col("max_installments").fill_null(0).cast(pl.Int32),
            pl.col("primary_payment_type").fill_null("none"),

            pl.col("review_count").fill_null(0).cast(pl.Int32),
            (pl.col("total_payment_value") - pl.col("total_order_value")).alias("payment_discrepancy"),
        ])
        .with_columns([
            (pl.col("payment_discrepancy").abs() > payment_discrepancy_tolerance).alias("has_payment_discrepancy"),

            # -----------------------------------------------------------------
            # Anti-Leakage: In-Flight Delivery Correction as of Cutoff Date
            # If delivered_customer_date >= cutoff or is null, the package was
            # still in transit as of cutoff date. Delivery duration and delay
            # could NOT be known at cutoff!
            # -----------------------------------------------------------------
            pl.when(
                pl.col("order_delivered_customer_date").is_not_null()
                & (pl.col("order_delivered_customer_date") < pl.lit(observation_cutoff))
            )
            .then(pl.col("is_delivered"))
            .otherwise(False)
            .alias("is_delivered_at_cutoff"),

            pl.when(
                pl.col("order_delivered_customer_date").is_not_null()
                & (pl.col("order_delivered_customer_date") < pl.lit(observation_cutoff))
            )
            .then(pl.col("delivery_duration_days"))
            .otherwise(None)
            .alias("delivery_duration_days_at_cutoff"),

            pl.when(
                pl.col("order_delivered_customer_date").is_not_null()
                & (pl.col("order_delivered_customer_date") < pl.lit(observation_cutoff))
            )
            .then(pl.col("delivery_delay_days"))
            .otherwise(None)
            .alias("delivery_delay_days_at_cutoff"),

            pl.when(
                pl.col("order_delivered_customer_date").is_not_null()
                & (pl.col("order_delivered_customer_date") < pl.lit(observation_cutoff))
            )
            .then(pl.col("is_delayed"))
            .otherwise(False)
            .alias("is_delayed_at_cutoff"),
        ])
        .select([
            "order_id",
            "customer_id",
            "customer_unique_id",
            "order_status",
            "order_purchase_timestamp",
            "total_item_price",
            "total_freight_value",
            "total_order_value",
            "total_payment_value",
            "payment_discrepancy",
            "has_payment_discrepancy",
            "item_count",
            "unique_product_count",
            "unique_seller_count",
            "unique_category_count",
            "payment_method_count",
            "max_installments",
            "primary_payment_type",
            "review_count",
            "latest_review_score",
            "latest_review_id",
            pl.col("delivery_duration_days_at_cutoff").alias("delivery_duration_days"),
            pl.col("delivery_delay_days_at_cutoff").alias("delivery_delay_days"),
            pl.col("is_delivered_at_cutoff").alias("is_delivered"),
            pl.col("is_delayed_at_cutoff").alias("is_delayed"),
            "has_items",
            "has_payments",
        ])
    )

    return gold_pit_orders


def build_point_in_time_customer_profiles(
    pit_orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    products: pl.DataFrame,
    payments: pl.DataFrame,
    reviews: pl.DataFrame,
    observation_cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
) -> pl.DataFrame:
    """
    Constructs customer profiles strictly using orders and events known as of observation_cutoff.
    Customer inclusion rule: At least one purchase strictly before observation_cutoff.
    Recency and lifetime are measured relative to observation_cutoff.
    """
    # 1. Order-level aggregations per customer_unique_id
    order_agg = (
        pit_orders
        .group_by("customer_unique_id")
        .agg([
            pl.len().cast(pl.Int32).alias("frequency_orders"),
            pl.col("total_order_value").sum().alias("monetary_value"),
            pl.col("total_payment_value").sum().alias("monetary_payment_value"),
            pl.col("total_order_value").mean().alias("average_order_value"),
            pl.col("total_order_value").median().alias("median_order_value"),
            pl.col("total_order_value").max().alias("max_order_value"),
            pl.col("item_count").sum().cast(pl.Int32).alias("total_items_purchased"),
            pl.col("item_count").mean().alias("average_items_per_order"),

            pl.col("order_purchase_timestamp").min().alias("first_purchase_date"),
            pl.col("order_purchase_timestamp").max().alias("last_purchase_date"),

            pl.col("delivery_duration_days").mean().alias("average_delivery_duration_days"),
            pl.col("delivery_delay_days").mean().alias("average_delivery_delay_days"),
            pl.col("is_delayed").filter(pl.col("is_delivered")).sum().cast(pl.Float64).alias("late_orders_count"),
            pl.col("is_delivered").sum().cast(pl.Float64).alias("delivered_orders_count"),
            (pl.col("order_status") == "canceled").sum().cast(pl.Int32).alias("canceled_order_count"),
        ])
        .with_columns([
            # Recency strictly relative to observation cutoff
            ((pl.lit(observation_cutoff) - pl.col("last_purchase_date")).dt.total_seconds() / 86400.0).alias("recency_days"),

            # Customer lifetime strictly within historical observation period
            ((pl.col("last_purchase_date") - pl.col("first_purchase_date")).dt.total_seconds() / 86400.0).alias("customer_lifetime_days"),

            # Late delivery ratio
            pl.when(pl.col("delivered_orders_count") > 0)
            .then(pl.col("late_orders_count") / pl.col("delivered_orders_count"))
            .otherwise(0.0)
            .alias("late_delivery_ratio"),
        ])
    )

    # 2. Product and category diversity on pre-cutoff orders
    items_enriched = (
        items
        .join(pit_orders.select(["order_id", "customer_unique_id"]), on="order_id", how="inner")
        .join(products.select(["product_id", "product_category_name_english"]), on="product_id", how="inner")
    )

    prod_agg = (
        items_enriched
        .group_by("customer_unique_id")
        .agg([
            pl.col("product_id").n_unique().cast(pl.Int32).alias("unique_products"),
            pl.col("product_category_name_english").n_unique().cast(pl.Int32).alias("unique_categories"),
            pl.col("seller_id").n_unique().cast(pl.Int32).alias("unique_sellers"),
        ])
    )

    top_cat = (
        items_enriched
        .group_by(["customer_unique_id", "product_category_name_english"])
        .agg(pl.col("item_total_value").sum().alias("cat_spend"))
        .sort(["customer_unique_id", "cat_spend"], descending=[False, True])
        .group_by("customer_unique_id")
        .first()
        .select([
            pl.col("customer_unique_id"),
            pl.col("product_category_name_english").alias("top_product_category"),
        ])
    )

    # 3. Payment behavior on pre-cutoff orders
    payments_cust = payments.join(
        pit_orders.select(["order_id", "customer_unique_id"]),
        on="order_id",
        how="inner",
    )
    pay_cust_agg = (
        payments_cust
        .group_by("customer_unique_id")
        .agg([
            pl.col("payment_type").n_unique().cast(pl.Int32).alias("payment_method_count"),
            pl.col("payment_installments").mean().alias("average_installments"),
            pl.col("payment_installments").max().cast(pl.Int32).alias("max_installments"),
        ])
    )
    preferred_pay = (
        payments_cust
        .group_by(["customer_unique_id", "payment_type"])
        .agg(pl.col("payment_value").sum().alias("type_spend"))
        .sort(["customer_unique_id", "type_spend"], descending=[False, True])
        .group_by("customer_unique_id")
        .first()
        .select([
            pl.col("customer_unique_id"),
            pl.col("payment_type").alias("preferred_payment_type"),
        ])
    )

    # 4. Review behavior strictly on pre-cutoff reviews
    pit_reviews = reviews.filter(pl.col("review_creation_date") < pl.lit(observation_cutoff))
    reviews_cust = pit_reviews.join(
        pit_orders.select(["order_id", "customer_unique_id"]),
        on="order_id",
        how="inner",
    )
    rev_cust_agg = (
        reviews_cust
        .group_by("customer_unique_id")
        .agg([
            pl.len().cast(pl.Int32).alias("review_count"),
            pl.col("review_score").mean().alias("average_review_score"),
            (pl.col("review_score") <= 2).sum().cast(pl.Float64).alias("low_rating_count"),
        ])
        .with_columns([
            pl.when(pl.col("review_count") > 0)
            .then(pl.col("low_rating_count") / pl.col("review_count"))
            .otherwise(None)
            .alias("low_rating_ratio")
        ])
    )

    # 5. Geographic metadata
    cust_geo = (
        customers
        .join(pit_orders.select(["customer_id", "order_purchase_timestamp"]), on="customer_id", how="inner")
        .sort(["customer_unique_id", "order_purchase_timestamp"], descending=[False, True])
        .group_by("customer_unique_id")
        .first()
        .select([
            "customer_unique_id",
            "customer_city",
            "customer_state",
            "customer_zip_code_prefix",
        ])
    )

    # 6. Assemble profile dataframe
    profiles = (
        order_agg
        .join(prod_agg, on="customer_unique_id", how="left")
        .join(top_cat, on="customer_unique_id", how="left")
        .join(pay_cust_agg, on="customer_unique_id", how="left")
        .join(preferred_pay, on="customer_unique_id", how="left")
        .join(rev_cust_agg, on="customer_unique_id", how="left")
        .join(cust_geo, on="customer_unique_id", how="left")
        .with_columns([
            pl.col("unique_products").fill_null(0).cast(pl.Int32),
            pl.col("unique_categories").fill_null(0).cast(pl.Int32),
            pl.col("unique_sellers").fill_null(0).cast(pl.Int32),
            pl.col("top_product_category").fill_null("none"),

            pl.when(pl.col("total_items_purchased") > 0)
            .then(pl.col("unique_categories") / pl.col("total_items_purchased"))
            .otherwise(0.0)
            .alias("category_diversity"),

            pl.when(pl.col("total_items_purchased") > 0)
            .then(pl.col("unique_sellers") / pl.col("total_items_purchased"))
            .otherwise(0.0)
            .alias("seller_diversity"),

            pl.when(pl.col("total_items_purchased") > 0)
            .then(pl.col("unique_products") / pl.col("total_items_purchased"))
            .otherwise(0.0)
            .alias("product_diversity"),

            pl.col("payment_method_count").fill_null(0).cast(pl.Int32),
            pl.col("average_installments").fill_null(0.0),
            pl.col("max_installments").fill_null(0).cast(pl.Int32),
            pl.col("preferred_payment_type").fill_null("none"),
            (pl.col("payment_method_count") / pl.col("frequency_orders")).alias("payment_method_diversity"),

            pl.col("review_count").fill_null(0).cast(pl.Int32),
            (
                pl.when(pl.col("review_count") > 0)
                .then(pl.col("review_count") / pl.col("frequency_orders"))
                .otherwise(None)
            ).alias("review_submission_rate"),
        ])
    )

    return profiles


def build_future_evaluation_targets(
    cohort_customer_ids: pl.Series,
    silver_orders: pl.DataFrame,
    silver_customers: pl.DataFrame,
    silver_items: pl.DataFrame,
    silver_products: pl.DataFrame,
    silver_reviews: pl.DataFrame,
    observation_cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    evaluation_end: datetime = DEFAULT_FUTURE_EVALUATION_END,
) -> pl.DataFrame:
    """
    Constructs downstream ground-truth utility evaluation targets for the cohort
    over the future evaluation window [observation_cutoff, evaluation_end).

    Evaluated Targets:
    - future_orders_count (Int32): Total orders placed in the evaluation window.
    - future_monetary_value (Float64): Total merchandise + freight spend in the window.
    - future_items_count (Int32): Total physical items purchased in the window.
    - future_is_retained (Int32): Binary retention indicator (1 if future_orders > 0 else 0).
    - future_is_churned (Int32): Binary churn proxy (1 if future_orders == 0 else 0).
    - future_days_to_next_order (Float64): Days from cutoff to earliest future purchase.
    - future_top_category (String): Category representing customer's highest future spend.
    - future_average_review_score (Float64): Mean review rating given to future orders.
    """
    cohort_list = cohort_customer_ids.unique().to_list()
    cohort_df = pl.DataFrame({"customer_unique_id": cohort_list})

    # 1. Join orders with customer_unique_id
    order_cust = (
        silver_orders
        .select(["order_id", "customer_id", "order_purchase_timestamp", "order_status"])
        .join(silver_customers.select(["customer_id", "customer_unique_id"]), on="customer_id", how="left")
    )

    # 2. Filter strictly to future orders placed by the cohort
    future_orders = order_cust.filter(
        (pl.col("order_purchase_timestamp") >= pl.lit(observation_cutoff))
        & (pl.col("order_purchase_timestamp") < pl.lit(evaluation_end))
        & pl.col("customer_unique_id").is_in(cohort_list)
    )

    # 3. Item values in future window
    items_enriched = (
        silver_items
        .join(future_orders.select(["order_id", "customer_unique_id"]), on="order_id", how="inner")
        .join(silver_products.select(["product_id", "product_category_name_english"]), on="product_id", how="left")
    )
    future_item_agg = items_enriched.group_by("customer_unique_id").agg([
        pl.col("item_total_value").sum().alias("future_monetary_value"),
        pl.len().cast(pl.Int32).alias("future_items_count"),
    ])

    # Future top category
    future_top_cat = (
        items_enriched
        .group_by(["customer_unique_id", "product_category_name_english"])
        .agg(pl.col("item_total_value").sum().alias("cat_spend"))
        .sort(["customer_unique_id", "cat_spend"], descending=[False, True])
        .group_by("customer_unique_id")
        .first()
        .select([
            pl.col("customer_unique_id"),
            pl.col("product_category_name_english").alias("future_top_category"),
        ])
    )

    # 4. Future reviews
    future_revs = (
        silver_reviews
        .join(future_orders.select(["order_id", "customer_unique_id"]), on="order_id", how="inner")
        .group_by("customer_unique_id")
        .agg([
            pl.col("review_score").mean().alias("future_average_review_score")
        ])
    )

    # 5. Future order aggregations
    future_order_agg = (
        future_orders
        .group_by("customer_unique_id")
        .agg([
            pl.len().cast(pl.Int32).alias("future_orders_count"),
            pl.col("order_purchase_timestamp").min().alias("earliest_future_purchase"),
        ])
        .with_columns([
            ((pl.col("earliest_future_purchase") - pl.lit(observation_cutoff)).dt.total_seconds() / 86400.0)
            .alias("future_days_to_next_order")
        ])
    )

    # 6. Assemble complete cohort targets (including non-returning customers with 0s)
    targets = (
        cohort_df
        .join(future_order_agg, on="customer_unique_id", how="left")
        .join(future_item_agg, on="customer_unique_id", how="left")
        .join(future_top_cat, on="customer_unique_id", how="left")
        .join(future_revs, on="customer_unique_id", how="left")
        .with_columns([
            pl.col("future_orders_count").fill_null(0).cast(pl.Int32),
            pl.col("future_monetary_value").fill_null(0.0),
            pl.col("future_items_count").fill_null(0).cast(pl.Int32),
            pl.col("future_top_category").fill_null("none"),
            (pl.col("future_orders_count").fill_null(0) > 0).cast(pl.Int32).alias("future_is_retained"),
            (pl.col("future_orders_count").fill_null(0) == 0).cast(pl.Int32).alias("future_is_churned"),
        ])
        .select([
            "customer_unique_id",
            "future_orders_count",
            "future_monetary_value",
            "future_items_count",
            "future_is_retained",
            "future_is_churned",
            "future_days_to_next_order",
            "future_top_category",
            "future_average_review_score",
        ])
    )

    assert len(targets) == len(cohort_df), "Targets row count mismatch with cohort"
    return targets


def assign_deterministic_splits(
    customer_ids: pl.Series,
    seed: int = DEFAULT_RANDOM_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> pl.DataFrame:
    """
    Assigns each customer to train, val, or test using deterministic SHA-256 hashing.
    100% reproducible across platforms and independent of row ordering.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    splits = []
    hash_values = []
    train_threshold = train_ratio
    val_threshold = train_ratio + val_ratio

    for cust_id in customer_ids:
        # Cryptographic salt with seed
        hash_input = f"{cust_id}_{seed}".encode("utf-8")
        # Map 8 hex chars (32 bits) uniformly to [0, 1)
        hex_digest = hashlib.sha256(hash_input).hexdigest()[:8]
        norm_val = int(hex_digest, 16) / 0xFFFFFFFF
        hash_values.append(norm_val)

        if norm_val < train_threshold:
            splits.append("train")
        elif norm_val < val_threshold:
            splits.append("val")
        else:
            splits.append("test")

    return pl.DataFrame({
        "customer_unique_id": customer_ids,
        "split": splits,
    })


def generate_experiment_manifest(
    manifest_path: Path,
    observation_cutoff: datetime,
    evaluation_end: datetime,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    total_raw_customers: int,
    cohort_customers: int,
    excluded_future_customers: int,
    retained_cohort_customers: int,
    split_counts: Dict[str, int],
) -> Dict[str, Any]:
    """
    Generates and persists the machine-readable experiment manifest JSON.
    """
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "dataset_version": DATASET_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "temporal_parameters": {
            "observation_cutoff": observation_cutoff.isoformat(),
            "evaluation_end": evaluation_end.isoformat(),
            "evaluation_window_days": (evaluation_end - observation_cutoff).days,
            "inclusion_rule": "first_purchase_date < observation_cutoff",
            "exclusion_rule": "first_purchase_date >= observation_cutoff",
        },
        "reproducibility": {
            "random_seed": seed,
            "split_ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio,
            },
            "split_algorithm": "SHA-256 deterministic hash mapping",
        },
        "cohort_cardinalities": {
            "total_dataset_customers": total_raw_customers,
            "cohort_customers": cohort_customers,
            "excluded_future_customers": excluded_future_customers,
            "cohort_coverage_pct": round(cohort_customers / total_raw_customers * 100, 2),
            "retained_customers_in_future": retained_cohort_customers,
            "retention_rate_pct": round(retained_cohort_customers / cohort_customers * 100, 4),
        },
        "split_cardinalities": split_counts,
        "anti_leakage_guarantees": [
            "Orders purchased on or after observation_cutoff strictly excluded from customer profiles.",
            "Deliveries completed on or after observation_cutoff treated as in-flight / undelivered.",
            "Reviews created on or after observation_cutoff strictly purged from historical review metrics.",
            "Downstream evaluation targets isolated in dedicated target table and evaluated strictly in future window.",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fw:
        json.dump(manifest, fw, indent=2)

    return manifest


def run_phase5_temporal(
    observation_cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    evaluation_end: datetime = DEFAULT_FUTURE_EVALUATION_END,
    seed: int = DEFAULT_RANDOM_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Dict[str, Any]:
    """
    Executes Phase 5 pipeline:
    1. Loads Silver tables.
    2. Builds point-in-time orders and customer profiles as of observation_cutoff.
    3. Builds future ground-truth evaluation targets over [observation_cutoff, evaluation_end).
    4. Computes deterministic train/val/test splits.
    5. Assembles unified temporal cohort table.
    6. Persists artifacts and writes experiment_manifest.json.
    """
    print("=" * 70)
    print("RUNNING PHASE 5: TEMPORAL EVALUATION & LEAKAGE PREVENTION")
    print(f"Observation Cutoff : {observation_cutoff}")
    print(f"Evaluation Window  : {observation_cutoff} to {evaluation_end}")
    print("=" * 70)

    print("\n[Step 1] Loading Silver tables...")
    orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    customers = pl.read_parquet(SILVER_DIR / "silver_customers.parquet")
    items = pl.read_parquet(SILVER_DIR / "silver_order_items.parquet")
    products = pl.read_parquet(SILVER_DIR / "silver_products.parquet")
    payments = pl.read_parquet(SILVER_DIR / "silver_order_payments.parquet")
    reviews = pl.read_parquet(SILVER_DIR / "silver_order_reviews.parquet")

    total_unique_cust = customers["customer_unique_id"].n_unique()

    # 1. Point-in-time orders
    print("\n[Step 2] Building Point-in-Time analytical orders (Anti-Leakage Filter)...")
    pit_orders = filter_point_in_time_orders(
        orders=orders,
        customers=customers,
        items=items,
        products=products,
        payments=payments,
        reviews=reviews,
        observation_cutoff=observation_cutoff,
    )
    print(f"-> Historical Orders prior to cutoff: {len(pit_orders):,} (from {len(orders):,} total)")

    # 2. Point-in-time customer profiles
    print("\n[Step 3] Building Point-in-Time Customer Profiles...")
    profiles = build_point_in_time_customer_profiles(
        pit_orders=pit_orders,
        customers=customers,
        items=items,
        products=products,
        payments=payments,
        reviews=reviews,
        observation_cutoff=observation_cutoff,
    )
    cohort_size = len(profiles)
    excluded_count = total_unique_cust - cohort_size
    print(f"-> Included Cohort Customers: {cohort_size:,} ({cohort_size / total_unique_cust * 100:.2f}%)")
    print(f"-> Excluded Future Customers: {excluded_count:,} ({excluded_count / total_unique_cust * 100:.2f}%)")

    # 3. Ground-truth evaluation targets
    print("\n[Step 4] Computing Downstream Ground-Truth Evaluation Targets...")
    targets = build_future_evaluation_targets(
        cohort_customer_ids=profiles["customer_unique_id"],
        silver_orders=orders,
        silver_customers=customers,
        silver_items=items,
        silver_products=products,
        silver_reviews=reviews,
        observation_cutoff=observation_cutoff,
        evaluation_end=evaluation_end,
    )
    retained_count = targets["future_is_retained"].sum()
    print(f"-> Retained Cohort Customers in Future: {retained_count:,} ({retained_count / cohort_size * 100:.2f}%)")

    # 4. Deterministic splits
    print("\n[Step 5] Assigning Deterministic SHA-256 Train/Val/Test Splits...")
    splits = assign_deterministic_splits(
        customer_ids=profiles["customer_unique_id"],
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    split_counts = {
        "train": (splits["split"] == "train").sum(),
        "val": (splits["split"] == "val").sum(),
        "test": (splits["split"] == "test").sum(),
    }
    print(f"-> Splits: Train={split_counts['train']:,} ({split_counts['train']/cohort_size*100:.1f}%), "
          f"Val={split_counts['val']:,} ({split_counts['val']/cohort_size*100:.1f}%), "
          f"Test={split_counts['test']:,} ({split_counts['test']/cohort_size*100:.1f}%)")

    # 5. Assemble unified cohort dataset
    print("\n[Step 6] Assembling Unified Temporal Cohort Parquet...")
    cohort_dataset = (
        profiles
        .join(splits, on="customer_unique_id", how="inner")
        .join(targets, on="customer_unique_id", how="inner")
    )
    assert len(cohort_dataset) == cohort_size, "Cohort dataset cardinality mismatch"

    # 6. Save Parquet tables
    print("\n[Step 7] Writing Phase 5 Parquet tables and Experiment Manifest...")
    profiles_path = GOLD_DIR / "temporal_customer_profiles_cutoff.parquet"
    targets_path = GOLD_DIR / "temporal_customer_targets.parquet"
    cohort_path = GOLD_DIR / "temporal_customer_cohort.parquet"
    manifest_path = GOLD_DIR / "experiment_manifest.json"

    profiles.write_parquet(profiles_path)
    targets.write_parquet(targets_path)
    cohort_dataset.write_parquet(cohort_path)

    # 7. Write Manifest
    manifest = generate_experiment_manifest(
        manifest_path=manifest_path,
        observation_cutoff=observation_cutoff,
        evaluation_end=evaluation_end,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        total_raw_customers=total_unique_cust,
        cohort_customers=cohort_size,
        excluded_future_customers=excluded_count,
        retained_cohort_customers=retained_count,
        split_counts=split_counts,
    )

    print(f"  [SAVED] {profiles_path.name} ({len(profiles):,} rows)")
    print(f"  [SAVED] {targets_path.name} ({len(targets):,} rows)")
    print(f"  [SAVED] {cohort_path.name} ({len(cohort_dataset):,} rows)")
    print(f"  [SAVED] {manifest_path.name}")
    print("\nPhase 5 Temporal Pipeline execution complete.")

    return {
        "cohort_size": cohort_size,
        "excluded_count": excluded_count,
        "retained_count": retained_count,
        "split_counts": split_counts,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_phase5_temporal()
