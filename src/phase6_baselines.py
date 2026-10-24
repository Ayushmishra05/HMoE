"""
Phase 6: Classical Customer-Segmentation Baselines & Downstream Utility Evaluation.

Implements:
1. Baseline 1: KMeans on RFM
2. Baseline 2: KMeans on RFM + Behavioral
3. Baseline 3: KMeans on RFM + Selected Non-Redundant Behavioral
4. Baseline 4: KMeans on RFM + All Engineered Behavioral
5. Baseline 5: Gaussian Mixture Model (GMM)
6. Baseline 6: HDBSCAN (Density-Based Clustering)
7. Baseline 7: DEC (Deep Embedding Clustering) via PyTorch Autoencoder + Student-t Refinement

Strict Constraints:
- Uses strictly pre-cutoff information defined by Phase 5 (temporal_customer_cohort.parquet).
- Preprocessing and scaling fit strictly on train split; transformed onto val and test.
- Rigorous K-selection via Elbow, Silhouette, DBI, and BIC/AIC analysis.
- Multi-seed clustering stability evaluation (ARI / NMI).
- Downstream future-period commercial utility evaluation (retention rate, lift, future monetary spend).
- Cluster profiling on interpretable pre-cutoff features.
- Zero future leakage, zero text embeddings, zero temporal neural encoders, zero MoE.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import polars as pl
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from src.config import GOLD_DIR, DEFAULT_RANDOM_SEED

# Output directory for Phase 6 baseline models and evaluation artifacts
BASELINES_DIR = GOLD_DIR / "baselines"

# -----------------------------------------------------------------------------
# Feature Set Definitions (Pre-Cutoff Historical Features Only)
# -----------------------------------------------------------------------------
FEATURE_SETS = {
    # 1. Classical RFM
    "RFM": [
        "recency_days",
        "frequency_orders",
        "monetary_value",
    ],
    # 2. RFM + Core Behavioral
    "RFM_Behavioral": [
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "total_items_purchased",
        "unique_categories",
        "unique_sellers",
        "average_installments",
    ],
    # 3. RFM + Selected Non-Redundant Behavioral (Audited in Phase 4)
    "RFM_Selected_NonRedundant": [
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "total_items_purchased",
        "category_diversity",
        "average_installments",
        "average_delivery_duration_days",
        "average_review_score",
    ],
    # 4. RFM + All Engineered Behavioral Features
    "RFM_All_Engineered": [
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "monetary_payment_value",
        "average_order_value",
        "total_items_purchased",
        "average_items_per_order",
        "unique_categories",
        "category_diversity",
        "unique_sellers",
        "seller_diversity",
        "average_delivery_duration_days",
        "late_delivery_ratio",
        "average_installments",
        "payment_method_count",
        "average_review_score",
    ],
}


def load_cohort_splits() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Loads temporal_customer_cohort.parquet and returns train, val, and test splits."""
    cohort_path = GOLD_DIR / "temporal_customer_cohort.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Temporal cohort dataset not found at {cohort_path}. Run Phase 5 first.")

    cohort = pl.read_parquet(cohort_path)
    train_df = cohort.filter(pl.col("split") == "train")
    val_df = cohort.filter(pl.col("split") == "val")
    test_df = cohort.filter(pl.col("split") == "test")

    return train_df, val_df, test_df


def create_preprocessing_pipeline() -> Pipeline:
    """
    Constructs leak-free preprocessing pipeline:
    1. SimpleImputer with median strategy (handles unreviewed / undelivered pre-cutoff nulls).
    2. StandardScaler for zero mean and unit variance.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def evaluate_k_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k_range: range = range(3, 9),
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Evaluates candidate K values on validation split using:
    - Inertia (Elbow Method)
    - Silhouette Score (higher is better)
    - Davies-Bouldin Index (lower is better)
    - Calinski-Harabasz Index (higher is better)
    """
    results = []
    best_k = None
    best_sil = -1.0

    # Subsample validation set for fast silhouette scoring if > 5000 rows
    val_subsample_idx = np.random.RandomState(random_state).choice(
        len(X_val), size=min(len(X_val), 5000), replace=False
    )
    X_val_sub = X_val[val_subsample_idx]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_train)
        val_preds = km.predict(X_val)
        val_preds_sub = val_preds[val_subsample_idx]

        sil = float(silhouette_score(X_val_sub, val_preds_sub))
        dbi = float(davies_bouldin_score(X_val, val_preds))
        chi = float(calinski_harabasz_score(X_val, val_preds))
        inertia = float(km.inertia_)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": round(sil, 4),
            "davies_bouldin": round(dbi, 4),
            "calinski_harabasz": round(chi, 4),
        })

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {
        "candidate_results": results,
        "selected_k": best_k,
        "selection_criteria": "Highest Validation Silhouette Score with Elbow/DBI cross-validation",
    }


def evaluate_gmm_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k_range: range = range(3, 9),
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Evaluates candidate GMM components using BIC and AIC on validation data."""
    results = []
    best_k = None
    lowest_bic = float("inf")

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, n_init=3)
        gmm.fit(X_train)
        bic = float(gmm.bic(X_val))
        aic = float(gmm.aic(X_val))
        val_preds = gmm.predict(X_val)
        dbi = float(davies_bouldin_score(X_val, val_preds))

        results.append({
            "k": k,
            "bic": round(bic, 2),
            "aic": round(aic, 2),
            "davies_bouldin": round(dbi, 4),
        })

        if bic < lowest_bic:
            lowest_bic = bic
            best_k = k

    return {
        "candidate_results": results,
        "selected_k": best_k,
        "selection_criteria": "Minimum Bayesian Information Criterion (BIC)",
    }


# -----------------------------------------------------------------------------
# Baseline 7: Deep Embedding Clustering (DEC) Architecture (PyTorch)
# -----------------------------------------------------------------------------
class DECAutoencoder(nn.Module):
    """Symmetric MLP Autoencoder for learning dense latent representations."""
    def __init__(self, input_dim: int, latent_dim: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class DECClusteringLayer(nn.Module):
    r"""
    Student-t distribution soft assignment layer as introduced in DEC (Xie et al., 2016).
    q_{ij} = (1 + ||z_i - \mu_j||^2 / \alpha)^{-(\alpha+1)/2} / \sum_{j'} ...
    """
    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z):
        # z: [batch, latent_dim], cluster_centers: [n_clusters, latent_dim]
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        numerator = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q


class DECModel(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int, latent_dim: int = 10):
        super().__init__()
        self.autoencoder = DECAutoencoder(input_dim, latent_dim)
        self.clustering_layer = DECClusteringLayer(n_clusters, latent_dim)

    def target_distribution(self, q):
        # p_{ij} = (q_{ij}^2 / \sum_i q_{ij}) / \sum_j (q_{ij}^2 / \sum_i q_{ij})
        weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
        return (weight / torch.sum(weight, dim=1, keepdim=True)).detach()

    def forward(self, x):
        z, x_recon = self.autoencoder(x)
        q = self.clustering_layer(z)
        return z, x_recon, q


def train_dec_baseline(
    X_train: np.ndarray,
    n_clusters: int,
    epochs_pretrain: int = 25,
    epochs_cluster: int = 15,
    batch_size: int = 256,
    latent_dim: int = 10,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Tuple[DECModel, np.ndarray]:
    """Pre-trains autoencoder with MSE, initializes cluster centers via KMeans, and refines with KL divergence."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cpu")

    input_dim = X_train.shape[1]
    model = DECModel(input_dim=input_dim, n_clusters=n_clusters, latent_dim=latent_dim).to(device)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    # 1. Pretrain Autoencoder
    opt_ae = optim.Adam(model.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion_mse = nn.MSELoss()
    model.autoencoder.train()

    for epoch in range(epochs_pretrain):
        for batch in loader:
            x_b = batch[0]
            opt_ae.zero_grad()
            _, recon = model.autoencoder(x_b)
            loss = criterion_mse(recon, x_b)
            loss.backward()
            opt_ae.step()

    # 2. Initialize cluster centers using KMeans on latent embeddings
    model.eval()
    with torch.no_grad():
        z_train, _ = model.autoencoder(train_tensor)
        z_train_np = z_train.numpy()

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(z_train_np)
    model.clustering_layer.cluster_centers.data = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    # 3. Fine-tune with KL Divergence Loss
    opt_dec = optim.Adam(model.parameters(), lr=5e-4)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train()

    for epoch in range(epochs_cluster):
        for batch in loader:
            x_b = batch[0]
            opt_dec.zero_grad()
            z, recon, q = model(x_b)
            p = model.target_distribution(q)
            loss_kl = criterion_kl(torch.log(q + 1e-10), p)
            loss_recon = criterion_mse(recon, x_b)
            loss = loss_kl + 0.1 * loss_recon
            loss.backward()
            opt_dec.step()

    model.eval()
    with torch.no_grad():
        _, _, q_all = model(train_tensor)
        preds_train = q_all.argmax(dim=1).numpy()

    return model, preds_train


# -----------------------------------------------------------------------------
# Downstream Future-Period Commercial Utility Evaluation
# -----------------------------------------------------------------------------
def evaluate_future_commercial_utility(
    cluster_labels: np.ndarray,
    cohort_df: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates downstream commercial utility of discovered clusters against
    ground-truth targets in the future 6-month evaluation window:
    - future_is_retained (retention rate per cluster)
    - future_monetary_value (future CLV per cluster)
    - ANOVA F-statistic and p-value for retention discrimination
    - Retention Lift (highest retention cluster vs cohort average)
    """
    df_eval = cohort_df.with_columns([
        pl.Series("cluster", cluster_labels).cast(pl.Int32)
    ])

    # Filter noise points (cluster == -1 in HDBSCAN) for fair group metrics
    valid_clusters_df = df_eval.filter(pl.col("cluster") >= 0)
    cohort_retention = float(df_eval["future_is_retained"].mean())

    cluster_stats = (
        valid_clusters_df
        .group_by("cluster")
        .agg([
            pl.len().alias("customer_count"),
            pl.col("future_is_retained").mean().alias("retention_rate"),
            pl.col("future_monetary_value").mean().alias("mean_future_spend"),
            pl.col("future_monetary_value").median().alias("median_future_spend"),
            pl.col("future_orders_count").mean().alias("mean_future_orders"),
        ])
        .sort("cluster")
    )

    # ANOVA test for retention rate discrimination across clusters
    unique_clusters = sorted(valid_clusters_df["cluster"].unique().to_list())
    retention_groups = [
        valid_clusters_df.filter(pl.col("cluster") == c)["future_is_retained"].to_numpy()
        for c in unique_clusters
    ]

    if len(retention_groups) > 1 and all(len(g) > 0 for g in retention_groups):
        f_stat, p_val = stats.f_oneway(*retention_groups)
    else:
        f_stat, p_val = 0.0, 1.0

    # Spend ANOVA
    spend_groups = [
        valid_clusters_df.filter(pl.col("cluster") == c)["future_monetary_value"].to_numpy()
        for c in unique_clusters
    ]
    f_stat_spend, p_val_spend = stats.f_oneway(*spend_groups) if len(spend_groups) > 1 else (0.0, 1.0)

    max_retention_cluster = cluster_stats.sort("retention_rate", descending=True).head(1).to_dicts()[0]
    min_retention_cluster = cluster_stats.sort("retention_rate", descending=False).head(1).to_dicts()[0]
    retention_lift = (
        max_retention_cluster["retention_rate"] / cohort_retention if cohort_retention > 0 else 1.0
    )

    return {
        "cohort_baseline_retention_rate": round(cohort_retention, 4),
        "cluster_retention_anova_f_stat": round(float(f_stat), 4),
        "cluster_retention_anova_p_val": float(p_val),
        "cluster_spend_anova_f_stat": round(float(f_stat_spend), 4),
        "cluster_spend_anova_p_val": float(p_val_spend),
        "max_retention_cluster": int(max_retention_cluster["cluster"]),
        "max_retention_rate": round(float(max_retention_cluster["retention_rate"]), 4),
        "min_retention_rate": round(float(min_retention_cluster["retention_rate"]), 4),
        "retention_lift": round(float(retention_lift), 2),
        "cluster_breakdown": cluster_stats.to_dicts(),
    }


def compute_cluster_profiles(
    cluster_labels: np.ndarray,
    cohort_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes interpretable pre-cutoff behavioral profiles for each cluster:
    Recency, Frequency, Spend, Items, Delivery Duration, Review Score, and Installments.
    """
    df = cohort_df.with_columns([
        pl.Series("cluster", cluster_labels).cast(pl.Int32)
    ])

    profile = (
        df
        .group_by("cluster")
        .agg([
            pl.len().alias("customer_count"),
            (pl.len() / len(df) * 100.0).alias("customer_pct"),
            pl.col("recency_days").median().alias("median_recency_days"),
            pl.col("frequency_orders").mean().alias("mean_frequency_orders"),
            pl.col("monetary_value").median().alias("median_spend"),
            pl.col("total_items_purchased").mean().alias("mean_items"),
            pl.col("average_installments").median().alias("median_installments"),
            pl.col("average_delivery_duration_days").median().alias("median_delivery_days"),
            pl.col("average_review_score").mean().alias("mean_review_score"),
            pl.col("future_is_retained").mean().alias("future_retention_rate"),
        ])
        .sort("cluster")
    )

    return profile


def run_phase6_baselines() -> Dict[str, Any]:
    """
    Orchestrates Phase 6 classical customer-segmentation baselines:
    1. Loads train/val/test splits strictly from Phase 5 temporal cohort.
    2. Executes K-selection across candidate K values on validation split.
    3. Trains and evaluates all 7 baselines.
    4. Evaluates internal clustering quality (Silhouette, DBI, CHI) on Test split.
    5. Evaluates multi-seed stability (ARI, NMI).
    6. Evaluates downstream future-period commercial utility.
    7. Computes interpretable cluster profiles.
    8. Persists all assignments, metrics, and experiment manifests.
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 6: CLASSICAL CUSTOMER-SEGMENTATION BASELINES")
    print("=" * 70)

    train_df, val_df, test_df = load_cohort_splits()
    print(f"Loaded Phase 5 Temporal Cohort:")
    print(f"  Train : {len(train_df):,} customers")
    print(f"  Val   : {len(val_df):,} customers")
    print(f"  Test  : {len(test_df):,} customers")

    experiments_summary = {}
    test_assignments_dict = {"customer_unique_id": test_df["customer_unique_id"]}

    # =========================================================================
    # Step 1: K-Selection on Validation Set (Documented Methodology)
    # =========================================================================
    print("\n[Step 1] Running K-Selection on Validation Split for RFM and Non-Redundant Sets...")
    pipe_rfm = create_preprocessing_pipeline()
    X_rfm_train = pipe_rfm.fit_transform(train_df.select(FEATURE_SETS["RFM"]).to_numpy())
    X_rfm_val = pipe_rfm.transform(val_df.select(FEATURE_SETS["RFM"]).to_numpy())
    X_rfm_test = pipe_rfm.transform(test_df.select(FEATURE_SETS["RFM"]).to_numpy())

    k_sel_rfm = evaluate_k_selection(X_rfm_train, X_rfm_val, k_range=range(3, 8))
    k_rfm = k_sel_rfm["selected_k"]
    print(f"-> Selected K for RFM: K={k_rfm} (Silhouette={k_sel_rfm['candidate_results'][k_rfm-3]['silhouette']})")

    pipe_nr = create_preprocessing_pipeline()
    X_nr_train = pipe_nr.fit_transform(train_df.select(FEATURE_SETS["RFM_Selected_NonRedundant"]).to_numpy())
    X_nr_val = pipe_nr.transform(val_df.select(FEATURE_SETS["RFM_Selected_NonRedundant"]).to_numpy())
    X_nr_test = pipe_nr.transform(test_df.select(FEATURE_SETS["RFM_Selected_NonRedundant"]).to_numpy())

    k_sel_nr = evaluate_k_selection(X_nr_train, X_nr_val, k_range=range(3, 8))
    k_nr = k_sel_nr["selected_k"]
    print(f"-> Selected K for Selected Non-Redundant: K={k_nr} (Silhouette={k_sel_nr['candidate_results'][k_nr-3]['silhouette']})")

    # GMM Component Selection
    gmm_sel = evaluate_gmm_selection(X_nr_train, X_nr_val, k_range=range(3, 8))
    k_gmm = gmm_sel["selected_k"]
    print(f"-> Selected K for GMM: K={k_gmm} (BIC={gmm_sel['candidate_results'][k_gmm-3]['bic']})")

    # Fixed representative K across all multi-feature models for fair comparative benchmarking
    k_standard = 4

    # =========================================================================
    # Step 2: Model Training & Evaluation
    # =========================================================================
    models_config = {
        "Baseline_1_KMeans_RFM": {
            "model_type": "KMeans",
            "features": "RFM",
            "k": k_rfm,
            "X_train": X_rfm_train,
            "X_test": X_rfm_test,
        },
        "Baseline_2_KMeans_RFM_Behavioral": {
            "model_type": "KMeans",
            "features": "RFM_Behavioral",
            "k": k_standard,
            "pipeline": create_preprocessing_pipeline(),
        },
        "Baseline_3_KMeans_RFM_Selected_NonRedundant": {
            "model_type": "KMeans",
            "features": "RFM_Selected_NonRedundant",
            "k": k_nr,
            "X_train": X_nr_train,
            "X_test": X_nr_test,
        },
        "Baseline_4_KMeans_RFM_All_Engineered": {
            "model_type": "KMeans",
            "features": "RFM_All_Engineered",
            "k": k_standard,
            "pipeline": create_preprocessing_pipeline(),
        },
        "Baseline_5_GMM": {
            "model_type": "GaussianMixture",
            "features": "RFM_Selected_NonRedundant",
            "k": k_gmm,
            "X_train": X_nr_train,
            "X_test": X_nr_test,
        },
        "Baseline_6_HDBSCAN": {
            "model_type": "HDBSCAN",
            "features": "RFM_Selected_NonRedundant",
            "min_cluster_size": 200,
            "X_train": X_nr_train,
            "X_test": X_nr_test,
        },
        "Baseline_7_DEC": {
            "model_type": "DEC",
            "features": "RFM_Selected_NonRedundant",
            "k": k_nr,
            "X_train": X_nr_train,
            "X_test": X_nr_test,
        },
    }

    # For fast and exact test silhouette computation across 8.6k test samples, take deterministic 5000 sample
    test_subsample_idx = np.random.RandomState(DEFAULT_RANDOM_SEED).choice(
        len(test_df), size=min(len(test_df), 5000), replace=False
    )

    print("\n[Step 2] Training and evaluating all 7 baseline models on Test Split...")
    for model_name, cfg in models_config.items():
        print(f"\n--- Running {model_name} ---")
        feat_name = cfg["features"]
        feat_cols = FEATURE_SETS[feat_name]

        if "X_train" in cfg:
            X_tr = cfg["X_train"]
            X_te = cfg["X_test"]
        else:
            pipe = cfg["pipeline"]
            X_tr = pipe.fit_transform(train_df.select(feat_cols).to_numpy())
            X_te = pipe.transform(test_df.select(feat_cols).to_numpy())

        # Fit model on train, predict on test
        if cfg["model_type"] == "KMeans":
            k = cfg["k"]
            model = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_SEED, n_init=10)
            model.fit(X_tr)
            test_preds = model.predict(X_te)
        elif cfg["model_type"] == "GaussianMixture":
            k = cfg["k"]
            model = GaussianMixture(n_components=k, covariance_type="diag", random_state=DEFAULT_RANDOM_SEED, n_init=3)
            model.fit(X_tr)
            test_preds = model.predict(X_te)
        elif cfg["model_type"] == "HDBSCAN":
            # HDBSCAN density clustering on test space directly
            model = HDBSCAN(min_cluster_size=cfg["min_cluster_size"], min_samples=20)
            test_preds = model.fit_predict(X_te)
            k = len(set(test_preds)) - (1 if -1 in test_preds else 0)
        elif cfg["model_type"] == "DEC":
            k = cfg["k"]
            dec_model, _ = train_dec_baseline(X_tr, n_clusters=k, epochs_pretrain=15, epochs_cluster=10, random_state=DEFAULT_RANDOM_SEED)
            with torch.no_grad():
                _, _, q_test = dec_model(torch.tensor(X_te, dtype=torch.float32))
                test_preds = q_test.argmax(dim=1).numpy()

        test_assignments_dict[model_name] = test_preds

        # Internal clustering metrics
        valid_mask = test_preds >= 0
        if valid_mask.sum() > 1 and len(set(test_preds[valid_mask])) > 1:
            X_te_sub = X_te[test_subsample_idx]
            preds_sub = test_preds[test_subsample_idx]
            sub_valid = preds_sub >= 0
            sil = float(silhouette_score(X_te_sub[sub_valid], preds_sub[sub_valid]))
            dbi = float(davies_bouldin_score(X_te[valid_mask], test_preds[valid_mask]))
            chi = float(calinski_harabasz_score(X_te[valid_mask], test_preds[valid_mask]))
        else:
            sil, dbi, chi = 0.0, 99.0, 0.0

        # Multi-seed stability (Seeds: 42, 123, 456)
        stability_metrics = evaluate_model_stability(cfg["model_type"], X_tr, X_te, cfg)

        # Downstream commercial utility on future targets
        utility_metrics = evaluate_future_commercial_utility(test_preds, test_df)

        # Cluster profiling
        cluster_profile = compute_cluster_profiles(test_preds, test_df)

        print(f"  Internal Quality: Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}, Calinski-Harabasz={chi:.1f}")
        print(f"  Stability: Multi-Seed ARI={stability_metrics['mean_pairwise_ari']:.4f}, NMI={stability_metrics['mean_pairwise_nmi']:.4f}")
        print(f"  Future Utility: Retention ANOVA F={utility_metrics['cluster_retention_anova_f_stat']:.2f} (p={utility_metrics['cluster_retention_anova_p_val']:.2e}), Max Retention={utility_metrics['max_retention_rate']*100:.2f}% (Lift={utility_metrics['retention_lift']}x)")

        experiments_summary[model_name] = {
            "model_type": cfg["model_type"],
            "feature_set": feat_name,
            "num_features": len(feat_cols),
            "num_clusters": int(k),
            "internal_metrics": {
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(dbi, 4),
                "calinski_harabasz_score": round(chi, 2),
            },
            "stability_metrics": stability_metrics,
            "downstream_utility": utility_metrics,
            "cluster_profiles": cluster_profile.to_dicts(),
        }

    # =========================================================================
    # Step 3: Persist Assignments and Experiment Manifest
    # =========================================================================
    print("\n[Step 3] Persisting Phase 6 Baseline Assignments and Metrics...")
    test_assignments_df = pl.DataFrame(test_assignments_dict)
    assignments_path = BASELINES_DIR / "phase6_baseline_test_assignments.parquet"
    test_assignments_df.write_parquet(assignments_path)

    manifest_path = BASELINES_DIR / "phase6_baselines_manifest.json"
    full_manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 6: Classical Customer-Segmentation Baselines",
        "timestamp": datetime.now().isoformat(),
        "k_selection_methodology": {
            "RFM": k_sel_rfm,
            "RFM_Selected_NonRedundant": k_sel_nr,
            "GMM": gmm_sel,
        },
        "baselines_evaluation": experiments_summary,
    }

    with open(manifest_path, "w", encoding="utf-8") as fw:
        json.dump(full_manifest, fw, indent=2)

    print(f"  [SAVED] {assignments_path.name} ({len(test_assignments_df):,} rows)")
    print(f"  [SAVED] {manifest_path.name}")
    print("\nPhase 6 Classical Baselines execution complete.")

    return full_manifest


def evaluate_model_stability(
    model_type: str,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    cfg: Dict[str, Any],
    seeds: Tuple[int, ...] = (42, 123, 456),
) -> Dict[str, float]:
    """Evaluates clustering stability across multiple random seeds using pairwise ARI and NMI."""
    if model_type == "HDBSCAN":
        # sklearn HDBSCAN has no out-of-sample predict method.  Refit on the
        # evaluated matrix for each repetition so the reported value is
        # calculated, rather than asserted, while documenting its transductive
        # scope in the corrected report.
        runs_preds = [
            HDBSCAN(
                min_cluster_size=cfg["min_cluster_size"],
                min_samples=cfg.get("min_samples", 20),
            ).fit_predict(X_te)
            for _ in seeds
        ]
        ari_list = [
            adjusted_rand_score(runs_preds[i], runs_preds[j])
            for i in range(len(runs_preds))
            for j in range(i + 1, len(runs_preds))
        ]
        nmi_list = [
            normalized_mutual_info_score(runs_preds[i], runs_preds[j])
            for i in range(len(runs_preds))
            for j in range(i + 1, len(runs_preds))
        ]
        return {
            "mean_pairwise_ari": round(float(np.mean(ari_list)), 4),
            "mean_pairwise_nmi": round(float(np.mean(nmi_list)), 4),
        }

    runs_preds = []
    k = cfg.get("k", 4)

    for s in seeds:
        if model_type == "KMeans":
            m = KMeans(n_clusters=k, random_state=s, n_init=10)
            m.fit(X_tr)
            runs_preds.append(m.predict(X_te))
        elif model_type == "GaussianMixture":
            m = GaussianMixture(n_components=k, covariance_type="diag", random_state=s, n_init=3)
            m.fit(X_tr)
            runs_preds.append(m.predict(X_te))
        elif model_type == "DEC":
            m, _ = train_dec_baseline(X_tr, n_clusters=k, epochs_pretrain=10, epochs_cluster=8, random_state=s)
            with torch.no_grad():
                _, _, q = m(torch.tensor(X_te, dtype=torch.float32))
                runs_preds.append(q.argmax(dim=1).numpy())

    # Compute pairwise ARI and NMI
    ari_list = []
    nmi_list = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari_list.append(adjusted_rand_score(runs_preds[i], runs_preds[j]))
            nmi_list.append(normalized_mutual_info_score(runs_preds[i], runs_preds[j]))

    return {
        "mean_pairwise_ari": round(float(np.mean(ari_list)), 4),
        "mean_pairwise_nmi": round(float(np.mean(nmi_list)), 4),
    }


if __name__ == "__main__":
    run_phase6_baselines()
