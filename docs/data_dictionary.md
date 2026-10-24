# HMoE Research Project Data Dictionary

This document provides the definitive data dictionary and architectural specification for the **Hierarchical Mixture of Experts (HMoE)** Customer Segmentation research pipeline on the Brazilian E-Commerce public dataset (Olist).

---

## 1. Architectural Overview & Research Alignment

The data pipeline follows a strict four-layer architecture:

```
[Raw Olist CSVs] (Read-Only)
       │
       ▼
   [Bronze] ─── Strict Schema Typing, Datetime Parsing, Lossless Parquet
       │
       ▼
   [Silver] ─── Entity Normalization, Spatial Deduplication, Quality Flags
       │
       ▼
    [Gold]  ─── Analytical Grains & Multimodal Views
       │
       ├── Behavioral View ──► Behavioral Expert (Customer Profiles & Preferences)
       ├── Temporal View   ──► Temporal Expert   (Customer Order Event Sequences)
       └── Text View       ──► Textual Expert    (Customer Individual Reviews)
       │
       ▼
[Modeling Layer] ──► Imputation, Scaling, Embeddings, Gating & MoE
```

---

## 2. Research Configuration Parameters

All research assumptions are configurable parameters in [`HMoE/src/config.py`](file:///c:/Users/ASUS/OneDrive/Documents/autoML/HMoE/src/config.py):

| Parameter | Type | Default Value | Description |
| :--- | :---: | :---: | :--- |
| `DEFAULT_OBSERVATION_DATE` | `Optional[datetime]` | `None` (auto: max purchase date + 1 day) | Anchor date for calculating customer recency in RFM. Can be set to a fixed date or split point. |
| `DEFAULT_PAYMENT_DISCREPANCY_TOLERANCE` | `float` | `0.01` | Dollar threshold above which `total_payment_value` vs `total_order_value` difference is flagged as `has_payment_discrepancy`. |
| `DEFAULT_TEMPORAL_CUTOFF` | `Optional[datetime]` | `None` | Optional temporal split timestamp to separate feature observation window from future evaluation window. |
| `CATEGORY_TRANSLATION_FALLBACKS` | `dict` | Explicit map | Handles categories present in listings but omitted from translation reference (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`). |

---

## 3. Gold Table Specifications

### 3.1 `gold_orders_analytical`
- **Grain**: Exactly one row per `order_id` (99,441 rows).
- **Primary Key**: `order_id`
- **Foreign Keys**: `customer_id`, `customer_unique_id`

| Column | Type | Description / Formula |
| :--- | :--- | :--- |
| `order_id` | `String` | Unique order identifier. |
| `customer_id` | `String` | Order-specific customer transaction identifier. |
| `customer_unique_id` | `String` | Real-world customer unique identifier. |
| `order_status` | `String` | Order lifecycle status (`delivered`, `shipped`, `canceled`, etc.). |
| `order_purchase_timestamp` | `Datetime` | Timestamp when the order was placed. |
| `total_item_price` | `Float64` | Sum of item catalog prices ($\sum \text{price}$); $0.0$ if no items. |
| `total_freight_value` | `Float64` | Sum of item freight charges ($\sum \text{freight}$); $0.0$ if no items. |
| `total_order_value` | `Float64` | Merchandise + Shipping total: $\text{total\_item\_price} + \text{total\_freight\_value}$. |
| `total_payment_value` | `Float64` | Sum of all payment installments/methods recorded for the order. |
| `payment_discrepancy` | `Float64` | Financial difference: $\text{total\_payment\_value} - \text{total\_order\_value}$. |
| `has_payment_discrepancy` | `Boolean` | Flag: $|\text{payment\_discrepancy}| > \text{tolerance}$ ($0.01$). |
| `item_count` | `Int32` | Total number of physical line items in the order. |
| `unique_product_count` | `Int32` | Distinct `product_id`s in the order. |
| `unique_seller_count` | `Int32` | Distinct `seller_id`s in the order. |
| `unique_category_count` | `Int32` | Distinct product categories in the order. |
| `payment_method_count` | `Int32` | Distinct payment methods utilized (e.g. credit card + voucher). |
| `max_installments` | `Int32` | Maximum payment installments chosen for this order. |
| `primary_payment_type` | `String` | Payment type with the largest dollar contribution to the order. |
| `review_count` | `Int32` | Number of reviews associated with this order. |
| `latest_review_score` | `Int32` | Score ($1$ to $5$) of the most recent review by creation/answer timestamp. |
| `latest_review_id` | `String` | Review identifier of the latest review. |
| `delivery_duration_days` | `Float64` | $(\text{delivered\_customer\_date} - \text{purchase\_timestamp}) / 86400$. |
| `delivery_delay_days` | `Float64` | $(\text{delivered\_customer\_date} - \text{estimated\_delivery\_date}) / 86400$. |
| `is_delivered` | `Boolean` | True if `order_status == 'delivered'`. |
| `is_delayed` | `Boolean` | True if `delivery_delay_days > 0`. |
| `has_items` | `Boolean` | True if order has items in order items table (False for 775 orders). |
| `has_payments` | `Boolean` | True if order has payment records (False for 1 order). |

---

### 3.2 `gold_customer_profiles`
- **Grain**: Exactly one row per `customer_unique_id` (96,096 rows).
- **Primary Key**: `customer_unique_id`
- **Role**: Primary feature store for the **Behavioral Expert**.

#### Feature Group: RFM & Lifetime
- `recency_days` (`Float64`): $(\text{OBSERVATION\_DATE} - \max(\text{order\_purchase\_timestamp})) / 86400.0$.
  - *Denominator*: $86,400.0$ seconds/day.
  - *Bounds*: $[1.00, 773.84]$ days. Mean: $289.20$, Median: $269.91$.
- `frequency_orders` (`Int32`): Count of distinct orders placed across customer lifetime.
  - *Bounds*: $[1, 17]$. Exactly $93,099$ ($96.88\%$) customers have frequency $= 1$.
- `monetary_value` (`Float64`): Cumulative merchandise + freight spend: $\sum \text{total\_order\_value}$.
  - *Bounds*: $[0.00, 13,664.08]$ BRL. Exactly $676$ customers ($0.70\%$) have $0.00$ BRL spend due to unfulfilled orders without physical items.
- `monetary_payment_value` (`Float64`): Cumulative payment spend: $\sum \text{total\_payment\_value}$.
  - *Bounds*: $[0.00, 13,664.08]$ BRL. Redundant with `monetary_value` ($r = 0.9848$).
- `first_purchase_date` (`Datetime`): Earliest `order_purchase_timestamp`.
- `last_purchase_date` (`Datetime`): Latest `order_purchase_timestamp`.
- `customer_lifetime_days` (`Float64`): Tenure span: $(\text{last\_purchase\_date} - \text{first\_purchase\_date}) / 86400.0$.
  - *Denominator*: $86,400.0$ seconds/day.
  - *Mathematical Invariants*:
    - Strictly $= 0.0$ for all $93,099$ single-order customers.
    - Strictly $> 0.0$ for $2,740$ repeat customers.
    - Exactly $= 0.0$ for $257$ repeat customers who placed multiple orders at the exact same purchase timestamp (simultaneous checkout).
    - Strictly non-negative ($[0.00, 633.08]$ days).

#### Feature Group: Purchase & Order Value Behavior
- `average_order_value` (`Float64`): $\text{monetary\_value} / \text{frequency\_orders}$.
  - *Denominator*: `frequency_orders` ($\ge 1$). Never null or zero-division. Identical to `monetary_value` for $96.88\%$ of customers ($r = 0.9818$).
- `median_order_value` (`Float64`): Median order value across customer orders.
- `max_order_value` (`Float64`): Peak single order value across customer orders.
- `total_items_purchased` (`Int32`): Cumulative physical items purchased ($\sum \text{item\_count}$).
- `average_items_per_order` (`Float64`): $\text{total\_items\_purchased} / \text{frequency\_orders}$.
  - *Denominator*: `frequency_orders` ($\ge 1$).

#### Feature Group: Product & Diversity Behavior
- `unique_products` (`Int32`): Distinct `product_id`s purchased.
- `unique_categories` (`Int32`): Distinct product categories purchased.
- `unique_sellers` (`Int32`): Distinct sellers purchased from.
- `top_product_category` (`String`): Product category representing the customer's highest spend.
- `category_diversity` (`Float64`): $\text{unique\_categories} / \text{total\_items\_purchased}$ ($0.0$ if no items).
  - *Denominator*: `total_items_purchased`. Protected against zero division via conditional branch ($0.0$ for the 676 customers with 0 items).
- `seller_diversity` (`Float64`): $\text{unique\_sellers} / \text{total\_items\_purchased}$ ($0.0$ if no items).
  - *Denominator*: `total_items_purchased`. Collinear with `category_diversity` ($r = 0.9533$).
- `product_diversity` (`Float64`): $\text{unique\_products} / \text{total\_items\_purchased}$ ($0.0$ if no items).
  - *Denominator*: `total_items_purchased`.

#### Feature Group: Delivery Experience
- `average_delivery_duration_days` (`Float64`): Mean customer delivery turnaround.
  - *Null count*: $2,740$ customers ($2.85\%$) whose orders were never delivered (canceled, unavailable, or in-transit).
- `average_delivery_delay_days` (`Float64`): Mean delay against estimate. Negative means early delivery.
- `late_orders_count` (`Float64`): Count of delivered orders where delivery was delayed.
- `delivered_orders_count` (`Float64`): Count of successfully delivered orders.
- `late_delivery_ratio` (`Float64`): $\text{late\_orders\_count} / \text{delivered\_orders\_count}$ ($0.0$ if 0 delivered orders).
  - *Denominator*: `delivered_orders_count`. Protected against zero division ($0.0$ if no delivered orders).
- `canceled_order_count` (`Int32`): Number of canceled orders across lifetime.

#### Feature Group: Payment Behavior
- `preferred_payment_type` (`String`): Payment method with largest cumulative dollar spend.
- `average_installments` (`Float64`): Mean installment count across orders.
- `max_installments` (`Int32`): Highest single installment count chosen. Redundant with `average_installments` ($r = 0.9931$).
- `payment_method_count` (`Int32`): Distinct payment methods used across lifetime.
- `payment_method_diversity` (`Float64`): $\text{payment\_method\_count} / \text{frequency\_orders}$.
  - *Denominator*: `frequency_orders` ($\ge 1$).

#### Feature Group: Review Behavior
- `review_count` (`Int32`): Total review records submitted across lifetime.
- `average_review_score` (`Float64`): Mean review rating ($1.0$ to $5.0$).
  - *Null count*: $716$ customers ($0.75\%$) who submitted 0 reviews.
- `low_rating_count` (`Float64`): Count of reviews with score $\le 2$.
- `low_rating_ratio` (`Float64`): $\text{low\_rating\_count} / \text{review\_count}$ ($null$ if 0 reviews).
  - *Denominator*: `review_count`. Mirror-redundant with `average_review_score` ($r = -0.8839$).
- `review_submission_rate` (`Float64`): $\text{review\_count} / \text{frequency\_orders}$ ($null$ if 0 reviews).
  - *Denominator*: `frequency_orders` ($\ge 1$).
  - *Multiplicity Note*: This is a **review volume intensity rate**, not a bounded percentage. Values $> 1.0$ occur for **447 customers** (up to $2.0$) because $547$ orders in Olist received multiple review submissions.

#### Feature Group: Customer Metadata (Geographic Context)
- `customer_city` (`String`): Canonical city from the customer's most recent order.
- `customer_state` (`String`): 2-letter state code from most recent order.
- `customer_zip_code_prefix` (`Int32`): Zip code prefix from most recent order.

---

### 3.3 `gold_customer_events`
- **Grain**: Exactly one row per customer-order event (99,441 rows).
- **Primary Keys**: (`customer_unique_id`, `event_sequence_num`)
- **Foreign Keys**: `order_id`
- **Role**: Sequential input for the **Temporal Expert**.

| Column | Type | Description |
| :--- | :--- | :--- |
| `customer_unique_id` | `String` | Real-world customer unique identifier. |
| `order_id` | `String` | Order identifier for this event. |
| `event_timestamp` | `Datetime` | Purchase timestamp (`order_purchase_timestamp`). |
| `event_sequence_num` | `Int32` | Chronological event index ($1, 2, 3, \dots$) per customer. |
| `days_since_previous_order` | `Float64` | Time delta in days from the customer's previous order ($null$ for event 1). |
| `order_value` | `Float64` | Order merchandise + shipping value. |
| `payment_value` | `Float64` | Total payment value recorded for this event. |
| `item_count` | `Int32` | Number of items in this order. |
| `category_count` | `Int32` | Distinct categories in this order. |
| `seller_count` | `Int32` | Distinct sellers in this order. |
| `delivery_duration_days` | `Float64` | Actual delivery duration for this order. |
| `delivery_delay_days` | `Float64` | Delivery delay relative to estimate for this order. |
| `review_score` | `Int32` | Latest review score awarded for this order. |
| `order_status` | `String` | Final status of the order. |

---

### 3.4 `gold_customer_reviews`
- **Grain**: Exactly one row per customer review (99,224 rows).
- **Primary Key**: `review_id`
- **Foreign Keys**: `order_id`, `customer_id`, `customer_unique_id`
- **Role**: Granular textual input for the **Textual Expert** (un-concatenated, preserving review level fidelity).

| Column | Type | Description |
| :--- | :--- | :--- |
| `customer_unique_id` | `String` | Customer unique identifier. |
| `customer_id` | `String` | Transaction customer identifier. |
| `order_id` | `String` | Associated order identifier. |
| `review_id` | `String` | Unique review identifier. |
| `review_score` | `Int32` | Numerical rating ($1$ to $5$). |
| `review_comment_title` | `String` | Customer review headline ($null$ if omitted). |
| `review_comment_message` | `String` | Full text feedback message ($null$ if omitted). |
| `has_comment_text` | `Boolean` | True if either title or message was written. |
| `review_creation_date` | `Datetime` | Timestamp when survey was sent to customer. |
| `review_answer_timestamp` | `Datetime` | Timestamp when customer answered the review survey. |
| `review_response_time_hours` | `Float64` | Time taken by customer to reply in hours. |
| `order_purchase_timestamp` | `Datetime` | Timestamp of original purchase. |

---

### 3.5 `gold_product_preferences`
- **Grain**: Exactly one row per (`customer_unique_id`, `product_category_name_english`) (97,915 rows across 95,420 purchasing customers).
- **Primary Key**: (`customer_unique_id`, `product_category_name_english`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `customer_unique_id` | `String` | Customer unique identifier. |
| `product_category_name_english` | `String` | English product category name. |
| `category_item_count` | `Int32` | Total items purchased in this category by this customer. |
| `category_spend` | `Float64` | Total dollar spend in this category by this customer. |
| `category_spend_share` | `Float64` | Category spend proportion relative to customer's total spend ($\in [0.0, 1.0]$). |

---

## 4. Phase 4 Feature Classification & Dependency Map

### 4.1 Feature Classification Taxonomy

All 38 features in `gold_customer_profiles` are classified into the following 6 functional research groups:

| Category | Features | Modeling Role |
| :--- | :--- | :--- |
| **Core Behavioral** | `frequency_orders`, `monetary_value`, `total_items_purchased`, `unique_categories`, `top_product_category`, `unique_sellers`, `unique_products` | Primary tabular inputs representing foundational purchasing scale, basket diversity, and merchant variety. |
| **Temporal** | `recency_days`, `first_purchase_date`, `last_purchase_date`, `customer_lifetime_days` | Time-to-event, churn indicators, and customer lifecycle span. |
| **Logistics** | `average_delivery_duration_days`, `average_delivery_delay_days`, `late_orders_count`, `delivered_orders_count`, `late_delivery_ratio`, `canceled_order_count` | Fulfillment quality and operational friction experienced by the customer. |
| **Payment** | `preferred_payment_type`, `average_installments`, `payment_method_count` | Financial preference and credit financing habits. |
| **Review / Satisfaction** | `review_count`, `average_review_score`, `low_rating_count` | Customer sentiment, dissatisfaction signals, and platform engagement. |
| **Derived / Redundant** | `monetary_payment_value`, `average_order_value`, `median_order_value`, `max_order_value`, `average_items_per_order`, `category_diversity`, `seller_diversity`, `product_diversity`, `max_installments`, `payment_method_diversity`, `low_rating_ratio`, `review_submission_rate` | Mechanically derived ratios and collinear pairs ($|r| \ge 0.70$ or $|\rho| \ge 0.70$). Preserved in Gold, but flagged for pruning or specialized handling prior to modeling. |

*(Note: `customer_city`, `customer_state`, and `customer_zip_code_prefix` represent geographic metadata).*

---

### 4.2 Feature Dependency & Redundancy Map

```
[Raw Tables]
├─ silver_orders ──────────► order_id, order_status, order_purchase_timestamp, delivery dates
├─ silver_order_items ─────► price, freight_value, product_id, seller_id
├─ silver_order_payments ──► payment_value, payment_type, payment_installments
└─ silver_order_reviews ───► review_score, review_id, review dates

[Order Level: gold_orders_analytical]
├─ total_order_value = total_item_price + total_freight_value
├─ total_payment_value = sum(payment_value)
├─ payment_discrepancy = total_payment_value - total_order_value
└─ review_count, latest_review_score, delivery_duration_days, delivery_delay_days

[Customer Level: gold_customer_profiles]
├─ Core Spend:
│    ├─ monetary_value = sum(total_order_value)
│    ├─ [REDUNDANT] monetary_payment_value = sum(total_payment_value)  (r = 0.9848)
│    ├─ [REDUNDANT] average_order_value = monetary_value / frequency   (r = 0.9818, identical for 96.88% of users)
│    ├─ [REDUNDANT] median_order_value                                 (identical for 96.88% of users)
│    └─ [REDUNDANT] max_order_value                                    (identical for 96.88% of users)
├─ Basket Size:
│    ├─ total_items_purchased = sum(item_count)
│    └─ [REDUNDANT] average_items_per_order = items / frequency        (identical for 96.88% of users)
├─ Diversity Ratios:
│    ├─ unique_categories, unique_sellers, unique_products
│    ├─ [REDUNDANT] category_diversity = categories / items            (rho = -0.7935 vs total_items)
│    ├─ [REDUNDANT] seller_diversity = sellers / items                 (r = 0.9533 vs category_diversity)
│    └─ [REDUNDANT] product_diversity = products / items
├─ Installments:
│    ├─ average_installments
│    └─ [REDUNDANT] max_installments                                   (r = 0.9931 vs average_installments)
├─ Reviews:
│    ├─ average_review_score
│    ├─ [REDUNDANT] low_rating_ratio = low_ratings / reviews           (r = -0.8839 vs average_score)
│    └─ [REDUNDANT] review_submission_rate = reviews / frequency       (multiplicity intensity rate)
└─ Lifecycle:
     ├─ frequency_orders
     └─ customer_lifetime_days = (last - first) / 86400                (rho = 0.9549 vs frequency)
```

---

## 5. Anomaly and Domain Policy Log

1. **Orders Without Items (775 orders)**:
   - Statuses: `unavailable` (603), `canceled` (164), `created` (5), `invoiced` (2), `shipped` (1).
   - *Policy*: Preserved in all tables. Item metrics filled with $0$, category set to `"none"`. Not discarded to avoid survival bias.
   - *Customer Impact*: Causes **676 customers (0.70%)** to have lifetime `monetary_value == 0.0` because their orders were canceled/unavailable prior to item fulfillment.
2. **Orders Without Payments (1 order)**:
   - Order `bfbd0f9bdef84302105ad712db648a6c` (delivered with 0 payment records).
   - *Policy*: Preserved with `total_payment_value = 0.0`, `has_payment_discrepancy = True`.
3. **Payment Discrepancies (1,151 orders)**:
   - *Policy*: Preserved. Discrepancy metric and boolean audit flag computed using configurable tolerance ($0.01$).
4. **Multiple Reviews Per Order (547 orders)**:
   - *Policy*: Preserved in full within `gold_customer_reviews`. Deterministic latest review score chosen for `gold_orders_analytical` using descending `review_creation_date` followed by `review_answer_timestamp`.
   - *Customer Impact*: Causes `review_submission_rate > 1.0` for **447 customers**.
5. **Simultaneous Repeat Orders (257 customers)**:
   - 257 repeat customers placed multiple orders with identical purchase timestamps (`first_purchase_date == last_purchase_date`), resulting in `customer_lifetime_days == 0.0`.
6. **Missing Product Categories (610 products)**:
   - *Policy*: Explicitly translated to English as `"unclassified"`. Flagged via `is_category_missing`.

---

## 6. Phase 5 Temporal Evaluation Framework & Point-in-Time Specifications

Phase 5 introduces strict **temporal cohorting and anti-leakage evaluation tables** to enable valid downstream evaluation of unsupervised customer segments against future commercial utility.

### 6.1 Architectural Parameters
- **Observation Cutoff Date ($T_{\text{obs}}$)**: `2018-03-01 00:00:00` (configurable via `DEFAULT_TEMPORAL_OBSERVATION_CUTOFF`).
- **Future Evaluation Window End ($T_{\text{eval}}$)**: `2018-09-01 00:00:00` (configurable via `DEFAULT_FUTURE_EVALUATION_END`).
- **Evaluation Duration**: $184$ days (~6 months).
- **Cohort Inclusion Rule**: Customers who placed at least one purchase strictly before $T_{\text{obs}}$ ($\min(\text{purchase\_timestamp}) < T_{\text{obs}}$).
- **Cohort Exclusion Rule**: Customers whose very first purchase occurred $\ge T_{\text{obs}}$ ($38,586$ future customers excluded).
- **Random Seed**: `42` (`DEFAULT_RANDOM_SEED`).
- **Split Proportions**: `70% Train` ($40,208$ cust), `15% Val` ($8,681$ cust), `15% Test` ($8,621$ cust).

---

### 6.2 `temporal_customer_profiles_cutoff`
- **Grain**: Exactly one row per eligible cohort customer as of $T_{\text{obs}}$ ($57,510$ rows).
- **Primary Key**: `customer_unique_id`
- **Role**: Clean historical feature matrix for unsupervised segmentation and MoE gating.
- **Anti-Leakage Guarantees**:
  1. *Transactions*: Orders purchased $\ge T_{\text{obs}}$ strictly excluded.
  2. *Deliveries*: Deliveries occurring $\ge T_{\text{obs}}$ are treated as in-flight (`is_delivered = False`, `delivery_duration_days = null`).
  3. *Reviews*: Reviews submitted $\ge T_{\text{obs}}$ are purged (`review_count = 0`, `average_review_score = null`).
  4. *Recency*: Calculated strictly relative to $T_{\text{obs}}$: $(T_{\text{obs}} - \max(\text{order\_purchase\_timestamp})) / 86400.0$.

---

### 6.3 `temporal_customer_targets`
- **Grain**: Exactly one row per cohort customer ($57,510$ rows).
- **Primary Key**: `customer_unique_id`
- **Role**: Downstream ground-truth commercial utility targets measured strictly within $[T_{\text{obs}}, T_{\text{eval}})$.

| Target Column | Type | Formula / Description |
| :--- | :--- | :--- |
| `customer_unique_id` | `String` | Real-world customer unique identifier. |
| `future_orders_count` | `Int32` | Total orders placed in $[T_{\text{obs}}, T_{\text{eval}})$. ($0$ for non-returning customers). |
| `future_monetary_value` | `Float64` | Cumulative merchandise + freight spend in $[T_{\text{obs}}, T_{\text{eval}})$. |
| `future_items_count` | `Int32` | Total physical items purchased in the future window. |
| `future_is_retained` | `Int32` | Binary retention indicator: $1$ if `future_orders_count > 0` else $0$ ($684$ customers, $1.19\%$). |
| `future_is_churned` | `Int32` | Binary churn proxy: $1$ if `future_orders_count == 0` else $0$ ($56,826$ customers, $98.81\%$). |
| `future_days_to_next_order` | `Float64` | $(\min(\text{future\_purchase\_timestamp}) - T_{\text{obs}}) / 86400.0$. ($null$ if not retained). |
| `future_top_category` | `String` | Product category with highest spend in future window ($"none"$ if not retained). |
| `future_average_review_score` | `Float64` | Mean review rating awarded by customer to future orders ($null$ if no future reviews). |

---

### 6.4 `temporal_customer_cohort`
- **Grain**: Exactly one row per cohort customer ($57,510$ rows).
- **Primary Key**: `customer_unique_id`
- **Role**: Unified master evaluation dataset combining historical features, deterministic `split` tag (`train`, `val`, `test`), and all future evaluation targets.

---

### 6.5 `experiment_manifest.json`
- **Format**: Machine-readable JSON metadata stored at `data/gold/experiment_manifest.json`.
- **Contents**: Semantic versions (`dataset_version`, `feature_set_version`), timestamps, temporal cutoff parameters, cryptographic split algorithm, cohort cardinalities, split counts, and explicit anti-leakage audit statements.

---

## 7. Phase 6 Classical Baselines & Downstream Utility Specifications

Phase 6 implements the benchmark suite of classical and deep customer-segmentation models using strictly pre-cutoff information defined in Phase 5 ($T_{\text{obs}} = \text{2018-03-01 00:00:00}$).

### 7.1 Implemented Baseline Models
1. **Baseline 1: `KMeans_RFM`** ($K=5$): Classical RFM (`recency_days`, `frequency_orders`, `monetary_value`).
2. **Baseline 2: `KMeans_RFM_Behavioral`** ($K=4$): RFM + core purchase scale (`total_items_purchased`, `unique_categories`, `unique_sellers`, `average_installments`).
3. **Baseline 3: `KMeans_RFM_Selected_NonRedundant`** ($K=5$): RFM + non-redundant behavioral metrics (`category_diversity`, `average_delivery_duration_days`, `average_review_score`, `average_installments`, `total_items_purchased`).
4. **Baseline 4: `KMeans_RFM_All_Engineered`** ($K=4$): 16 pre-cutoff numerical and ratio features.
5. **Baseline 5: `Gaussian_Mixture_Model (GMM)`** ($K=7$): Probabilistic soft clustering with diagonal covariance.
6. **Baseline 6: `HDBSCAN`** ($K=6$ + noise): Density-based hierarchical clustering (`min_cluster_size=200`, `min_samples=20`).
7. **Baseline 7: `DEC (Deep Embedding Clustering)`** ($K=5$): PyTorch MLP Autoencoder pretraining + Student-$t$ distribution soft assignment refinement via KL divergence.

### 7.2 Preprocessing & Data Isolation Protocol
- **Pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Isolation Invariant**: Preprocessing pipelines are **fit strictly on the `train` split** ($40,208$ customers) and applied via `.transform()` onto `val` ($8,681$) and `test` ($8,621$) to prevent train-to-test information leakage.
- **Zero Gold Modification**: Gold datasets remain immutable; models and assignments are stored in `data/gold/baselines/`.

### 7.3 Phase 6 Artifacts
- **`phase6_baseline_test_assignments.parquet`**: Cluster IDs for all $8,621$ blind test customers across all 7 baselines.
- **`phase6_baselines_manifest.json`**: Original machine-readable manifest recording K-selection curves, internal metrics, multi-seed stability (ARI/NMI), downstream utility metrics (retention rate, ANOVA $F$, lift), and cluster profiles.
- **`data/gold/reports/phase6_corrected_report.json`** and **`phase6_corrected_report.md`**: Reconciled report using the exact blind-test retention baseline, per-cluster Wilson 95% intervals, Cohen's $h$ effect sizes, and an explicit HDBSCAN transductive/stability audit. The original manifest is preserved unchanged.
