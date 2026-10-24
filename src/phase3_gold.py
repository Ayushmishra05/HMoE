"""
Phase 3: Gold Analytical Layer for Hierarchical Mixture of Experts (HMoE) Customer Segmentation.

Artifacts Produced:
1. gold_orders_analytical: 1 row per order_id (99,441 rows).
2. gold_customer_profiles: 1 row per customer_unique_id (96,096 rows).
3. gold_customer_events: 1 row per customer-order event (99,441 rows).
4. gold_customer_reviews: 1 row per review (99,224 rows).
5. gold_product_preferences: 1 row per (customer_unique_id, product_category_name_english).

Strict Research Guarantees:
- Reproducible and deterministic.
- Zero blind joins; explicit cardinality preservation.
- Observation date, payment discrepancy tolerance, and temporal cutoffs are explicit parameters.
- Missing modalities are preserved and flagged explicitly.
- No ML scaling, normalization, clustering, or embeddings applied yet.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import polars as pl

from src.config import (
    SILVER_DIR,
    GOLD_DIR,
    DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE,
    DEFAULT_OBSERVATION_DATE,
    DEFAULT_TEMPORAL_CUTOFF,
)


def build_gold_orders_analytical(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    products: pl.DataFrame,
    payments: pl.DataFrame,
    reviews: pl.DataFrame,
    payment_discrepancy_tolerance: float = DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE,
) -> pl.DataFrame:
    """
    Constructs gold_orders_analytical at grain: one row per order_id.
    Reconciles item financial totals with payment totals, extracts item diversity,
    and extracts deterministic latest review score.
    """
    # 1. Join items with product category
    items_enriched = items.join(
        products.select(["product_id", "product_category_name_english"]),
        on="product_id",
        how="left"
    )

    # 2. Item-level aggregations per order
    item_agg = items_enriched.group_by("order_id").agg([
        pl.col("price").sum().alias("total_item_price"),
        pl.col("freight_value").sum().alias("total_freight_value"),
        pl.col("item_total_value").sum().alias("total_order_value"),
        pl.len().alias("item_count"),
        pl.col("product_id").n_unique().alias("unique_product_count"),
        pl.col("seller_id").n_unique().alias("unique_seller_count"),
        pl.col("product_category_name_english").n_unique().alias("unique_category_count"),
    ])

    # 3. Payment-level aggregations per order
    pay_agg = payments.group_by("order_id").agg([
        pl.col("payment_value").sum().alias("total_payment_value"),
        pl.col("payment_type").n_unique().alias("payment_method_count"),
        pl.col("payment_installments").max().alias("max_installments"),
    ])

    # Primary payment type per order (highest payment value for that order)
    primary_pay = (
        payments.group_by(["order_id", "payment_type"])
        .agg(pl.col("payment_value").sum().alias("type_val"))
        .sort(["order_id", "type_val"], descending=[False, True])
        .group_by("order_id")
        .first()
        .select([pl.col("order_id"), pl.col("payment_type").alias("primary_payment_type")])
    )

    # 4. Review-level aggregation (deterministic latest review)
    # Explicitly sorted by review_creation_date descending, then review_answer_timestamp descending
    rev_sorted = reviews.sort(["review_creation_date", "review_answer_timestamp"], descending=[True, True])
    rev_agg = rev_sorted.group_by("order_id").agg([
        pl.len().alias("review_count"),
        pl.col("review_score").first().alias("latest_review_score"),
        pl.col("review_id").first().alias("latest_review_id"),
    ])

    # 5. Customer mapping for customer_unique_id
    cust_map = customers.select(["customer_id", "customer_unique_id"]).unique()

    # 6. Assemble gold_orders_analytical
    gold_orders = (
        orders
        .join(cust_map, on="customer_id", how="left")
        .join(item_agg, on="order_id", how="left")
        .join(pay_agg, on="order_id", how="left")
        .join(primary_pay, on="order_id", how="left")
        .join(rev_agg, on="order_id", how="left")
        .with_columns([
            # Fill missing financial/item values for orders without items
            pl.col("total_item_price").fill_null(0.0),
            pl.col("total_freight_value").fill_null(0.0),
            pl.col("total_order_value").fill_null(0.0),
            pl.col("item_count").fill_null(0).cast(pl.Int32),
            pl.col("unique_product_count").fill_null(0).cast(pl.Int32),
            pl.col("unique_seller_count").fill_null(0).cast(pl.Int32),
            pl.col("unique_category_count").fill_null(0).cast(pl.Int32),

            # Fill missing payment values for orders without payments
            pl.col("total_payment_value").fill_null(0.0),
            pl.col("payment_method_count").fill_null(0).cast(pl.Int32),
            pl.col("max_installments").fill_null(0).cast(pl.Int32),
            pl.col("primary_payment_type").fill_null("none"),

            # Fill review counts
            pl.col("review_count").fill_null(0).cast(pl.Int32),
        ])
        .with_columns([
            # Payment discrepancy: total_payment_value - total_order_value
            (pl.col("total_payment_value") - pl.col("total_order_value")).alias("payment_discrepancy"),
        ])
        .with_columns([
            # Boolean discrepancy indicator using configurable tolerance
            (pl.col("payment_discrepancy").abs() > payment_discrepancy_tolerance).alias("has_payment_discrepancy"),
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
            "delivery_duration_days",
            "delivery_delay_days",
            "is_delivered",
            "is_delayed",
            "has_items",
            "has_payments",
        ])
    )

    # Invariant: 1 row per order_id
    assert len(gold_orders) == len(orders), f"Row count mismatch in gold_orders_analytical: {len(gold_orders)} vs {len(orders)}"
    assert gold_orders["order_id"].n_unique() == len(gold_orders), "Duplicate order_id found in gold_orders_analytical"

    return gold_orders


def build_gold_customer_events(gold_orders: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs gold_customer_events at grain: one row per customer-order event.
    Preserves chronological order sequences for the temporal encoder.
    """
    events = (
        gold_orders
        .sort(["customer_unique_id", "order_purchase_timestamp"])
        .with_columns([
            pl.col("order_purchase_timestamp")
            .cum_count()
            .over("customer_unique_id")
            .cast(pl.Int32)
            .alias("event_sequence_num"),

            (
                (
                    pl.col("order_purchase_timestamp")
                    - pl.col("order_purchase_timestamp").shift(1).over("customer_unique_id")
                ).dt.total_seconds() / 86400.0
            ).alias("days_since_previous_order"),
        ])
        .select([
            "customer_unique_id",
            "order_id",
            pl.col("order_purchase_timestamp").alias("event_timestamp"),
            "event_sequence_num",
            "days_since_previous_order",
            pl.col("total_order_value").alias("order_value"),
            pl.col("total_payment_value").alias("payment_value"),
            "item_count",
            pl.col("unique_category_count").alias("category_count"),
            pl.col("unique_seller_count").alias("seller_count"),
            "delivery_duration_days",
            "delivery_delay_days",
            pl.col("latest_review_score").alias("review_score"),
            "order_status",
        ])
    )

    assert len(events) == len(gold_orders), f"Event row count mismatch: {len(events)} vs {len(gold_orders)}"
    return events


def build_gold_customer_reviews(
    reviews: pl.DataFrame,
    orders: pl.DataFrame,
    customers: pl.DataFrame,
) -> pl.DataFrame:
    """
    Constructs gold_customer_reviews at grain: one row per review.
    Preserves individual reviews with customer_unique_id for text encoding.
    """
    order_cust = (
        orders.select(["order_id", "customer_id", "order_purchase_timestamp"])
        .join(customers.select(["customer_id", "customer_unique_id"]), on="customer_id", how="left")
    )

    gold_reviews = (
        reviews
        .join(order_cust, on="order_id", how="left")
        .with_columns([
            (
                pl.col("review_comment_title").is_not_null()
                | pl.col("review_comment_message").is_not_null()
            ).alias("has_comment_text"),
        ])
        .select([
            "customer_unique_id",
            "customer_id",
            "order_id",
            "review_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "has_comment_text",
            "review_creation_date",
            "review_answer_timestamp",
            "review_response_time_hours",
            "order_purchase_timestamp",
        ])
    )

    assert len(gold_reviews) == len(reviews), f"Review row count mismatch: {len(gold_reviews)} vs {len(reviews)}"
    return gold_reviews


def build_gold_product_preferences(
    items: pl.DataFrame,
    products: pl.DataFrame,
    gold_orders: pl.DataFrame,
) -> pl.DataFrame:
    """
    Constructs gold_product_preferences at grain: (customer_unique_id, product_category_name_english).
    Computes spend and item counts per category per customer, along with category spend shares.
    """
    items_enriched = (
        items
        .join(gold_orders.select(["order_id", "customer_unique_id"]), on="order_id", how="inner")
        .join(products.select(["product_id", "product_category_name_english"]), on="product_id", how="inner")
    )

    pref = (
        items_enriched
        .group_by(["customer_unique_id", "product_category_name_english"])
        .agg([
            pl.len().cast(pl.Int32).alias("category_item_count"),
            pl.col("item_total_value").sum().alias("category_spend"),
        ])
        .with_columns([
            (
                pl.col("category_spend")
                / pl.col("category_spend").sum().over("customer_unique_id")
            ).alias("category_spend_share")
        ])
        .sort(["customer_unique_id", "category_spend"], descending=[False, True])
    )

    return pref


def build_gold_customer_profiles(
    gold_orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    products: pl.DataFrame,
    payments: pl.DataFrame,
    reviews: pl.DataFrame,
    observation_date: Optional[datetime] = DEFAULT_OBSERVATION_DATE,
    temporal_cutoff: Optional[datetime] = DEFAULT_TEMPORAL_CUTOFF,
) -> pl.DataFrame:
    """
    Constructs gold_customer_profiles at grain: one row per customer_unique_id.
    Synthesizes RFM, Customer Lifetime, Purchase Behavior, Product Diversity,
    Delivery Quality, Payment Diversity, and Review Behavior.
    """
    # Determine observation date deterministically if not explicitly provided
    if observation_date is None:
        max_ts = gold_orders["order_purchase_timestamp"].max()
        # Explicit rule: observation date = max purchase timestamp + 1 day
        observation_date = max_ts + timedelta(days=1)

    # 1. Order-level aggregations per customer_unique_id
    order_agg = (
        gold_orders
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

            # Timestamps
            pl.col("order_purchase_timestamp").min().alias("first_purchase_date"),
            pl.col("order_purchase_timestamp").max().alias("last_purchase_date"),

            # Delivery aggregations
            pl.col("delivery_duration_days").mean().alias("average_delivery_duration_days"),
            pl.col("delivery_delay_days").mean().alias("average_delivery_delay_days"),
            pl.col("is_delayed").filter(pl.col("is_delivered")).sum().cast(pl.Float64).alias("late_orders_count"),
            pl.col("is_delivered").sum().cast(pl.Float64).alias("delivered_orders_count"),
            (pl.col("order_status") == "canceled").sum().cast(pl.Int32).alias("canceled_order_count"),
        ])
        .with_columns([
            # Recency in days relative to observation date
            ((pl.lit(observation_date) - pl.col("last_purchase_date")).dt.total_seconds() / 86400.0).alias("recency_days"),

            # Customer lifetime days (tenure span)
            ((pl.col("last_purchase_date") - pl.col("first_purchase_date")).dt.total_seconds() / 86400.0).alias("customer_lifetime_days"),

            # Late delivery ratio
            pl.when(pl.col("delivered_orders_count") > 0)
            .then(pl.col("late_orders_count") / pl.col("delivered_orders_count"))
            .otherwise(0.0)
            .alias("late_delivery_ratio"),
        ])
    )

    # 2. Product and category diversity per customer
    items_enriched = (
        items
        .join(gold_orders.select(["order_id", "customer_unique_id"]), on="order_id", how="inner")
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

    # Top product category by spend per customer
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

    # 3. Payment behavior per customer
    payments_cust = payments.join(
        gold_orders.select(["order_id", "customer_unique_id"]),
        on="order_id",
        how="inner"
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

    # Preferred payment method (highest total spend across customer lifetime)
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

    # 4. Review behavior per customer
    reviews_cust = reviews.join(
        gold_orders.select(["order_id", "customer_unique_id"]),
        on="order_id",
        how="inner"
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

    # 5. Customer geographic metadata (from most recent customer order)
    cust_geo = (
        customers
        .join(gold_orders.select(["customer_id", "order_purchase_timestamp"]), on="customer_id", how="inner")
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

    # 6. Assemble Gold Customer Profile
    profiles = (
        order_agg
        .join(prod_agg, on="customer_unique_id", how="left")
        .join(top_cat, on="customer_unique_id", how="left")
        .join(pay_cust_agg, on="customer_unique_id", how="left")
        .join(preferred_pay, on="customer_unique_id", how="left")
        .join(rev_cust_agg, on="customer_unique_id", how="left")
        .join(cust_geo, on="customer_unique_id", how="left")
        .with_columns([
            # Fill product metrics for customers whose orders had no items
            pl.col("unique_products").fill_null(0).cast(pl.Int32),
            pl.col("unique_categories").fill_null(0).cast(pl.Int32),
            pl.col("unique_sellers").fill_null(0).cast(pl.Int32),
            pl.col("top_product_category").fill_null("none"),

            # Behavioral Diversity metrics
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

            # Fill payment metrics
            pl.col("payment_method_count").fill_null(0).cast(pl.Int32),
            pl.col("average_installments").fill_null(0.0),
            pl.col("max_installments").fill_null(0).cast(pl.Int32),
            pl.col("preferred_payment_type").fill_null("none"),

            # Payment diversity (payment methods used relative to frequency)
            (pl.col("payment_method_count") / pl.col("frequency_orders")).alias("payment_method_diversity"),

            # Review metrics
            pl.col("review_count").fill_null(0).cast(pl.Int32),
            (pl.col("review_count") / pl.col("frequency_orders")).alias("review_submission_rate"),
        ])
    )

    # Optional temporal cutoff split indicator
    if temporal_cutoff is not None:
        profiles = profiles.with_columns([
            (pl.col("first_purchase_date") < pl.lit(temporal_cutoff)).alias("is_in_feature_period")
        ])

    assert len(profiles) == customers["customer_unique_id"].n_unique(), (
        f"Profile count mismatch: {len(profiles)} vs {customers['customer_unique_id'].n_unique()}"
    )
    assert profiles["customer_unique_id"].n_unique() == len(profiles), (
        "Duplicate customer_unique_id in gold_customer_profiles"
    )

    return profiles


def run_phase3_gold(
    allow_overwrite: bool = False,
    observation_date: Optional[datetime] = DEFAULT_OBSERVATION_DATE,
    temporal_cutoff: Optional[datetime] = DEFAULT_TEMPORAL_CUTOFF,
    payment_discrepancy_tolerance: float = DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE,
) -> Dict[str, Any]:
    """
    Executes Phase 3 and writes all 5 Gold Parquet tables with overwrite protection.
    Returns metrics and validation summary.
    """
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Silver Parquet tables...")
    orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    customers = pl.read_parquet(SILVER_DIR / "silver_customers.parquet")
    items = pl.read_parquet(SILVER_DIR / "silver_order_items.parquet")
    products = pl.read_parquet(SILVER_DIR / "silver_products.parquet")
    payments = pl.read_parquet(SILVER_DIR / "silver_order_payments.parquet")
    reviews = pl.read_parquet(SILVER_DIR / "silver_order_reviews.parquet")

    # 1. Build gold_orders_analytical
    print("Building gold_orders_analytical...")
    gold_orders = build_gold_orders_analytical(
        orders, customers, items, products, payments, reviews,
        payment_discrepancy_tolerance=payment_discrepancy_tolerance
    )

    # 2. Build gold_customer_events
    print("Building gold_customer_events...")
    gold_events = build_gold_customer_events(gold_orders)

    # 3. Build gold_customer_reviews
    print("Building gold_customer_reviews...")
    gold_reviews = build_gold_customer_reviews(reviews, orders, customers)

    # 4. Build gold_product_preferences
    print("Building gold_product_preferences...")
    gold_pref = build_gold_product_preferences(items, products, gold_orders)

    # 5. Build gold_customer_profiles
    print("Building gold_customer_profiles...")
    gold_profiles = build_gold_customer_profiles(
        gold_orders, customers, items, products, payments, reviews,
        observation_date=observation_date,
        temporal_cutoff=temporal_cutoff,
    )

    gold_tables = {
        "gold_orders_analytical": gold_orders,
        "gold_customer_profiles": gold_profiles,
        "gold_customer_events": gold_events,
        "gold_customer_reviews": gold_reviews,
        "gold_product_preferences": gold_pref,
    }

    # Write tables to Gold directory
    summary = {}
    for name, df in gold_tables.items():
        out_path = GOLD_DIR / f"{name}.parquet"
        if out_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Gold file {out_path} already exists. Set allow_overwrite=True to overwrite."
            )
        df.write_parquet(out_path, compression="zstd")

        # Roundtrip verification
        readback = pl.read_parquet(out_path)
        assert len(readback) == len(df), f"Parquet roundtrip failed for {name}"

        summary[name] = {
            "output_path": str(out_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns,
            "size_bytes": out_path.stat().st_size,
        }

    return summary


if __name__ == "__main__":
    results = run_phase3_gold(allow_overwrite=True)
    print(f"Phase 3 Complete. Generated {len(results)} Gold analytical tables.")
