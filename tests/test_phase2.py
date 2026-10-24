"""
Automated Test Suite for Phase 2 Silver Entity Normalization and Cleaning.
"""

import sys
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BRONZE_DIR, SILVER_DIR
from src.phase2_silver import run_phase2_silver


def test_phase2_pipeline():
    print("=" * 70)
    print("RUNNING PHASE 2 SILVER TRANSFORMATIONS AND VALIDATION")
    print("=" * 70)

    # 1. Run Phase 2
    print("\n[Step 1] Executing Phase 2 Silver transformations...")
    summary = run_phase2_silver(allow_overwrite=True)
    assert len(summary) == 8, f"Expected 8 silver tables, got {len(summary)}"
    print(f"-> Successfully materialized {len(summary)} silver tables.")

    # 2. Check Row Preservation against Bronze
    print("\n[Step 2] Validating 100% Entity Row Count Preservation...")
    expected_rows = {
        "silver_orders": ("olist_orders_dataset.parquet", 99441),
        "silver_customers": ("olist_customers_dataset.parquet", 99441),
        "silver_sellers": ("olist_sellers_dataset.parquet", 3095),
        "silver_products": ("olist_products_dataset.parquet", 32951),
        "silver_order_items": ("olist_order_items_dataset.parquet", 112650),
        "silver_order_payments": ("olist_order_payments_dataset.parquet", 103886),
        "silver_order_reviews": ("olist_order_reviews_dataset.parquet", 99224),
    }

    for silver_name, (brz_name, expected_count) in expected_rows.items():
        df_silver = pl.read_parquet(SILVER_DIR / f"{silver_name}.parquet")
        df_bronze = pl.read_parquet(BRONZE_DIR / brz_name)
        assert len(df_silver) == len(df_bronze) == expected_count, (
            f"Row count mismatch for {silver_name}: silver={len(df_silver)}, bronze={len(df_bronze)}"
        )
        print(f"  [PASS] {silver_name:<30}: {len(df_silver):>7} rows (100% preserved)")

    # 3. Geolocation Centroid Check
    print("\n[Step 3] Validating Geolocation Centroids Aggregation...")
    geo_centroids = pl.read_parquet(SILVER_DIR / "silver_geolocation_centroids.parquet")
    raw_geo = pl.read_parquet(BRONZE_DIR / "olist_geolocation_dataset.parquet")
    expected_zips = raw_geo["geolocation_zip_code_prefix"].n_unique()
    assert len(geo_centroids) == expected_zips == 19015, (
        f"Centroid count mismatch: got {len(geo_centroids)}, expected {expected_zips}"
    )
    assert geo_centroids["points_count"].sum() == len(raw_geo) == 1000163, (
        "Sum of centroid points does not match raw points"
    )
    assert geo_centroids["geolocation_zip_code_prefix"].n_unique() == len(geo_centroids), (
        "Centroid zip codes not unique"
    )
    print(f"  [PASS] silver_geolocation_centroids : 19,015 unique centroids derived from 1,000,163 GPS points (100% accounted)")

    # 4. Primary Key Uniqueness
    print("\n[Step 4] Validating Silver Primary Keys...")
    pks = {
        "silver_orders": ["order_id"],
        "silver_customers": ["customer_id"],
        "silver_sellers": ["seller_id"],
        "silver_products": ["product_id"],
        "silver_order_items": ["order_id", "order_item_id"],
        "silver_order_payments": ["order_id", "payment_sequential"],
        "silver_geolocation_centroids": ["geolocation_zip_code_prefix"],
    }
    for table_name, pk_cols in pks.items():
        df = pl.read_parquet(SILVER_DIR / f"{table_name}.parquet")
        assert df.select(pk_cols).n_unique() == len(df), f"PK violation in {table_name}"
        print(f"  [PASS] {table_name:<30}: {len(df):>7} unique keys across {len(df):>7} rows")

    # 5. Product Translation Resolution
    print("\n[Step 5] Validating Product Category Translation Resolution...")
    prod = pl.read_parquet(SILVER_DIR / "silver_products.parquet")
    assert prod["product_category_name_english"].null_count() == 0, (
        "Found null values in product_category_name_english"
    )
    unclass_count = prod.filter(pl.col("product_category_name_english") == "unclassified").shape[0]
    assert unclass_count == 610, f"Expected 610 unclassified products, got {unclass_count}"
    print(f"  [PASS] silver_products translations   : 0 unmapped nulls, 32,341 mapped, 610 unclassified flagged")

    # 6. Lead Time & Delivery Duration Sanity
    print("\n[Step 6] Validating Order Delivery Lead Times...")
    orders = pl.read_parquet(SILVER_DIR / "silver_orders.parquet")
    delivered_orders = orders.filter(pl.col("is_delivered") & pl.col("order_delivered_customer_date").is_not_null())
    negative_delivery = delivered_orders.filter(pl.col("delivery_duration_days") < 0)
    assert len(negative_delivery) == 0, f"Found negative delivery durations: {len(negative_delivery)}"
    delayed_count = orders.filter(pl.col("is_delayed")).shape[0]
    print(f"  [PASS] silver_orders delivery lead times: 0 negative durations across {len(delivered_orders)} delivered orders")
    print(f"  [INFO] Orders with late delivery flag: {delayed_count} ({delayed_count / len(orders) * 100:.2f}%)")

    # 7. Order Items Financial Integrity
    print("\n[Step 7] Validating Order Item Financial Totals...")
    items = pl.read_parquet(SILVER_DIR / "silver_order_items.parquet")
    calc_diff = items.filter((pl.col("item_total_value") - (pl.col("price") + pl.col("freight_value"))).abs() > 1e-6)
    assert len(calc_diff) == 0, "Item total mismatch"
    invalid_ratios = items.filter((pl.col("freight_ratio") < 0.0) | (pl.col("freight_ratio") > 1.0))
    assert len(invalid_ratios) == 0, "Freight ratios outside [0, 1]"
    print(f"  [PASS] silver_order_items financials  : 100% math consistency (price + freight == total)")

    print("\n" + "=" * 70)
    print("ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_phase2_pipeline()
