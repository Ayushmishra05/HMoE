"""
Configuration and Schema Specifications for HMoE Data Engineering Pipeline.
"""

from pathlib import Path
from datetime import datetime
import polars as pl

# ---------------------------------------------------------
# Directory Hierarchy
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "olist_dataset"
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DOCS_DIR = BASE_DIR / "docs"

# ---------------------------------------------------------
# Research Parameters for Phase 3 (Gold Layer)
# ---------------------------------------------------------
# Payment discrepancy tolerance: abs(total_payment - total_order) > tolerance
DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE = 0.01

# Observation Date for RFM recency calculation
# By default, None means auto-set to (max_purchase_timestamp in dataset + 1 day)
DEFAULT_OBSERVATION_DATE = None  # Can be set to a specific datetime, e.g. datetime(2018, 10, 18, 18, 0, 0)

# Optional temporal train/validation cutoff for future research splitting
# If set, customers/events can be filtered or flagged relative to this date
DEFAULT_TEMPORAL_CUTOFF = None

# Category translation fallbacks for categories missing in Olist translation table
CATEGORY_TRANSLATION_FALLBACKS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "small_appliances_kitchen_and_food_preparers",
}

# ---------------------------------------------------------
# Research Parameters for Phase 5 (Temporal Evaluation & Leakage Prevention)
# ---------------------------------------------------------
# Observation cutoff date: information strictly before this timestamp is historical;
# information on or after this timestamp is future/evaluation.
DEFAULT_TEMPORAL_OBSERVATION_CUTOFF = datetime(2018, 3, 1, 0, 0, 0)

# Future evaluation window end date (~6 months / 184 days after cutoff)
DEFAULT_FUTURE_EVALUATION_END = datetime(2018, 9, 1, 0, 0, 0)

# Reproducibility seed for deterministic cohort splitting
DEFAULT_RANDOM_SEED = 42

# Train / Validation / Test split ratios for customer cohort evaluation
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

# Dataset and feature set semantic versions for experiment tracking
FEATURE_SET_VERSION = "v1.0-gold-audited"
DATASET_VERSION = "olist-brazilian-ecommerce-gold"

# ---------------------------------------------------------
# Table Schemas and Types for Bronze Ingestion
# ---------------------------------------------------------
DATETIME_COLUMNS = {
    "olist_orders_dataset": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "olist_order_items_dataset": [
        "shipping_limit_date",
    ],
    "olist_order_reviews_dataset": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}

EXPLICIT_DTYPES = {
    "olist_customers_dataset": {
        "customer_id": pl.String,
        "customer_unique_id": pl.String,
        "customer_zip_code_prefix": pl.Int32,
        "customer_city": pl.String,
        "customer_state": pl.String,
    },
    "olist_geolocation_dataset": {
        "geolocation_zip_code_prefix": pl.Int32,
        "geolocation_lat": pl.Float64,
        "geolocation_lng": pl.Float64,
        "geolocation_city": pl.String,
        "geolocation_state": pl.String,
    },
    "olist_order_items_dataset": {
        "order_id": pl.String,
        "order_item_id": pl.Int32,
        "product_id": pl.String,
        "seller_id": pl.String,
        "price": pl.Float64,
        "freight_value": pl.Float64,
    },
    "olist_order_payments_dataset": {
        "order_id": pl.String,
        "payment_sequential": pl.Int32,
        "payment_type": pl.String,
        "payment_installments": pl.Int32,
        "payment_value": pl.Float64,
    },
    "olist_order_reviews_dataset": {
        "review_id": pl.String,
        "order_id": pl.String,
        "review_score": pl.Int32,
        "review_comment_title": pl.String,
        "review_comment_message": pl.String,
    },
    "olist_orders_dataset": {
        "order_id": pl.String,
        "customer_id": pl.String,
        "order_status": pl.String,
    },
    "olist_products_dataset": {
        "product_id": pl.String,
        "product_category_name": pl.String,
        "product_name_lenght": pl.Int32,
        "product_description_lenght": pl.Int32,
        "product_photos_qty": pl.Int32,
        "product_weight_g": pl.Float64,
        "product_length_cm": pl.Float64,
        "product_height_cm": pl.Float64,
        "product_width_cm": pl.Float64,
    },
    "olist_sellers_dataset": {
        "seller_id": pl.String,
        "seller_zip_code_prefix": pl.Int32,
        "seller_city": pl.String,
        "seller_state": pl.String,
    },
    "product_category_name_translation": {
        "product_category_name": pl.String,
        "product_category_name_english": pl.String,
    },
}

PRIMARY_KEYS = {
    "olist_customers_dataset": ["customer_id"],
    "olist_orders_dataset": ["order_id"],
    "olist_sellers_dataset": ["seller_id"],
    "olist_products_dataset": ["product_id"],
    "olist_order_items_dataset": ["order_id", "order_item_id"],
    "olist_order_payments_dataset": ["order_id", "payment_sequential"],
    "product_category_name_translation": ["product_category_name"],
}

FOREIGN_KEYS = [
    ("olist_orders_dataset", "customer_id", "olist_customers_dataset", "customer_id"),
    ("olist_order_items_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_order_items_dataset", "product_id", "olist_products_dataset", "product_id"),
    ("olist_order_items_dataset", "seller_id", "olist_sellers_dataset", "seller_id"),
    ("olist_order_payments_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_order_reviews_dataset", "order_id", "olist_orders_dataset", "order_id"),
]
