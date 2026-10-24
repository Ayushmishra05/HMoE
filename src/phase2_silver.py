"""
Phase 2: Silver Layer - Entity Normalization, Spatial Deduplication, and Domain Cleaning.

Transformations:
- Orders: Lead times, delay flags, quality flags (100% rows preserved).
- Customers: Formatted city/state, repeat customer indicator (100% rows preserved).
- Sellers: Formatted city/state (100% rows preserved).
- Products: Translation resolution with fallback for missing categories, volume calculation (100% rows preserved).
- Order Items: Item total values, freight ratios (100% rows preserved).
- Order Payments: Payment value flags (100% rows preserved).
- Order Reviews: Response time in hours, text availability indicators, multi-review flags (100% rows preserved).
- Geolocation Centroids: Deterministic spatial aggregation of 1,000,163 raw points into 19,015 canonical zip-code centroids.
"""

from pathlib import Path
from typing import Dict, Any
import polars as pl

from src.config import (
    BRONZE_DIR,
    SILVER_DIR,
)

# Explicit category translations for the 2 categories missing in Olist's translation table
CATEGORY_TRANSLATION_FALLBACKS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "small_appliances_kitchen_and_food_preparers",
}


def build_silver_orders(orders_bronze: pl.DataFrame, items_bronze: pl.DataFrame, payments_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_orders with domain lead times and quality flags.
    Preserves all 99,441 orders without dropping any records.
    """
    item_orders = items_bronze.select("order_id").unique()
    payment_orders = payments_bronze.select("order_id").unique()

    df = (
        orders_bronze
        .with_columns([
            # 1. Lead times & Durations (in days / hours)
            ((pl.col("order_delivered_customer_date") - pl.col("order_purchase_timestamp")).dt.total_seconds() / 86400.0)
            .alias("delivery_duration_days"),

            ((pl.col("order_estimated_delivery_date") - pl.col("order_purchase_timestamp")).dt.total_seconds() / 86400.0)
            .alias("estimated_delivery_duration_days"),

            ((pl.col("order_approved_at") - pl.col("order_purchase_timestamp")).dt.total_seconds() / 3600.0)
            .alias("approval_delay_hours"),

            ((pl.col("order_delivered_customer_date") - pl.col("order_delivered_carrier_date")).dt.total_seconds() / 86400.0)
            .alias("carrier_duration_days"),

            ((pl.col("order_delivered_customer_date") - pl.col("order_estimated_delivery_date")).dt.total_seconds() / 86400.0)
            .alias("delivery_delay_days"),

            # 2. Quality and Domain Flags
            (pl.col("order_status") == "delivered").alias("is_delivered"),
            
            # Anomaly: delivered status but delivery date missing
            ((pl.col("order_status") == "delivered") & pl.col("order_delivered_customer_date").is_null())
            .alias("is_delivered_missing_date"),

            # Anomaly: carrier timestamp earlier than purchase timestamp (45 known Olist records)
            (pl.col("order_delivered_carrier_date") < pl.col("order_purchase_timestamp"))
            .fill_null(False)
            .alias("is_carrier_before_purchase"),
        ])
        .with_columns([
            # Late delivery indicator: positive delay days
            (pl.col("delivery_delay_days") > 0).fill_null(False).alias("is_delayed"),
        ])
        .join(
            item_orders.with_columns(pl.lit(True).alias("has_items")),
            on="order_id",
            how="left"
        )
        .with_columns(pl.col("has_items").fill_null(False))
        .join(
            payment_orders.with_columns(pl.lit(True).alias("has_payments")),
            on="order_id",
            how="left"
        )
        .with_columns(pl.col("has_payments").fill_null(False))
    )

    return df


def build_silver_customers(customers_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_customers with standardized text and repeat customer flag.
    Preserves all 99,441 records.
    """
    # Count customer occurrences by customer_unique_id
    repeat_counts = customers_bronze.group_by("customer_unique_id").len().rename({"len": "customer_order_count"})

    df = (
        customers_bronze
        .with_columns([
            pl.col("customer_city").str.to_lowercase().str.strip_chars().alias("customer_city"),
            pl.col("customer_state").str.to_uppercase().str.strip_chars().alias("customer_state"),
        ])
        .join(repeat_counts, on="customer_unique_id", how="left")
        .with_columns([
            (pl.col("customer_order_count") > 1).alias("is_repeat_customer")
        ])
    )
    return df


def build_silver_sellers(sellers_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_sellers with standardized text.
    Preserves all 3,095 records.
    """
    return sellers_bronze.with_columns([
        pl.col("seller_city").str.to_lowercase().str.strip_chars().alias("seller_city"),
        pl.col("seller_state").str.to_uppercase().str.strip_chars().alias("seller_state"),
    ])


def build_silver_products(products_bronze: pl.DataFrame, translation_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_products with English translation mapping, volume, and quality indicators.
    Preserves all 32,951 records.
    """
    # Create complete translation dictionary with fallbacks
    trans_map = dict(zip(
        translation_bronze["product_category_name"].to_list(),
        translation_bronze["product_category_name_english"].to_list()
    ))
    trans_map.update(CATEGORY_TRANSLATION_FALLBACKS)

    # Convert map to DataFrame for deterministic Polars join
    full_trans_df = pl.DataFrame({
        "product_category_name": list(trans_map.keys()),
        "product_category_name_english": list(trans_map.values()),
    })

    df = (
        products_bronze
        .join(full_trans_df, on="product_category_name", how="left")
        .with_columns([
            # Missing category indicator
            pl.col("product_category_name").is_null().alias("is_category_missing"),
            pl.col("product_category_name_english").fill_null("unclassified").alias("product_category_name_english"),

            # Derived Volume in cm^3
            (pl.col("product_length_cm") * pl.col("product_height_cm") * pl.col("product_width_cm"))
            .alias("product_volume_cm3"),

            # Missing dimensions indicator
            (
                pl.col("product_weight_g").is_null()
                | pl.col("product_length_cm").is_null()
                | pl.col("product_height_cm").is_null()
                | pl.col("product_width_cm").is_null()
            ).alias("has_missing_dimensions"),
        ])
    )
    return df


def build_silver_order_items(items_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_order_items with total value and freight proportion.
    Preserves all 112,650 records.
    """
    df = (
        items_bronze
        .with_columns([
            (pl.col("price") + pl.col("freight_value")).alias("item_total_value"),
            (pl.col("freight_value") / (pl.col("price") + pl.col("freight_value"))).alias("freight_ratio"),
        ])
    )
    return df


def build_silver_order_payments(payments_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_order_payments with zero-value flags.
    Preserves all 103,886 records.
    """
    df = (
        payments_bronze
        .with_columns([
            (pl.col("payment_value") <= 0.0).alias("is_zero_value_payment"),
            pl.col("payment_type").str.strip_chars().str.to_lowercase().alias("payment_type"),
        ])
    )
    return df


def build_silver_order_reviews(reviews_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs silver_order_reviews with response duration, text indicators, and multi-review flag.
    Preserves all 99,224 records.
    """
    # Count reviews per order
    rev_counts = reviews_bronze.group_by("order_id").len().rename({"len": "order_review_count"})

    df = (
        reviews_bronze
        .with_columns([
            pl.col("review_comment_title").is_not_null().alias("has_comment_title"),
            pl.col("review_comment_message").is_not_null().alias("has_comment_message"),
            (
                (pl.col("review_answer_timestamp") - pl.col("review_creation_date")).dt.total_seconds() / 3600.0
            ).alias("review_response_time_hours"),
        ])
        .join(rev_counts, on="order_id", how="left")
        .with_columns([
            (pl.col("order_review_count") > 1).alias("is_multi_review_order"),
        ])
    )
    return df


def build_silver_geolocation_centroids(geo_bronze: pl.DataFrame) -> pl.DataFrame:
    """
    Deterministically collapses 1,000,163 raw multi-point coordinates
    into 19,015 canonical zip-code centroids.
    Calculates median lat/lng, canonical city/state, and point count.
    """
    centroids = (
        geo_bronze
        .group_by("geolocation_zip_code_prefix")
        .agg([
            pl.col("geolocation_lat").median().alias("geolocation_lat_median"),
            pl.col("geolocation_lng").median().alias("geolocation_lng_median"),
            pl.col("geolocation_city").first().str.to_lowercase().str.strip_chars().alias("geolocation_city_canonical"),
            pl.col("geolocation_state").first().str.to_uppercase().str.strip_chars().alias("geolocation_state_canonical"),
            pl.len().alias("points_count"),
        ])
        .sort("geolocation_zip_code_prefix")
    )
    return centroids


def run_phase2_silver(allow_overwrite: bool = False) -> Dict[str, Any]:
    """
    Executes Phase 2 transformations and writes all Silver Parquet tables.
    Returns metrics and summary for all processed tables.
    """
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Bronze Tables
    print("Loading Bronze Parquet tables...")
    orders_brz = pl.read_parquet(BRONZE_DIR / "olist_orders_dataset.parquet")
    customers_brz = pl.read_parquet(BRONZE_DIR / "olist_customers_dataset.parquet")
    sellers_brz = pl.read_parquet(BRONZE_DIR / "olist_sellers_dataset.parquet")
    products_brz = pl.read_parquet(BRONZE_DIR / "olist_products_dataset.parquet")
    items_brz = pl.read_parquet(BRONZE_DIR / "olist_order_items_dataset.parquet")
    payments_brz = pl.read_parquet(BRONZE_DIR / "olist_order_payments_dataset.parquet")
    reviews_brz = pl.read_parquet(BRONZE_DIR / "olist_order_reviews_dataset.parquet")
    geo_brz = pl.read_parquet(BRONZE_DIR / "olist_geolocation_dataset.parquet")
    trans_brz = pl.read_parquet(BRONZE_DIR / "product_category_name_translation.parquet")

    # 2. Build Silver Tables
    print("Building Silver entities...")
    silver_tables = {
        "silver_orders": build_silver_orders(orders_brz, items_brz, payments_brz),
        "silver_customers": build_silver_customers(customers_brz),
        "silver_sellers": build_silver_sellers(sellers_brz),
        "silver_products": build_silver_products(products_brz, trans_brz),
        "silver_order_items": build_silver_order_items(items_brz),
        "silver_order_payments": build_silver_order_payments(payments_brz),
        "silver_order_reviews": build_silver_order_reviews(reviews_brz),
        "silver_geolocation_centroids": build_silver_geolocation_centroids(geo_brz),
    }

    # 3. Write Silver Parquet Tables with Overwrite Protection
    summary = {}
    for name, df in silver_tables.items():
        out_path = SILVER_DIR / f"{name}.parquet"
        if out_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Silver file {out_path} already exists. Set allow_overwrite=True to overwrite."
            )
        df.write_parquet(out_path, compression="zstd")

        # Verify roundtrip
        df_read = pl.read_parquet(out_path)
        assert len(df_read) == len(df), f"Roundtrip failed for {name}"

        summary[name] = {
            "output_path": str(out_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns,
            "size_bytes": out_path.stat().st_size,
        }

    return summary


if __name__ == "__main__":
    results = run_phase2_silver(allow_overwrite=True)
    print(f"Phase 2 Complete. Generated {len(results)} silver tables.")
