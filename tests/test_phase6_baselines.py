"""
Automated Test Suite for Phase 6 Classical Customer-Segmentation Baselines.
Validates:
1. Preprocessing and scaling fit strictly on train split without leakage.
2. Cutoff compliance: feature matrices exclude all future evaluation targets.
3. Cluster assignment integrity: all 8,621 test customers have valid cluster assignments.
4. Metric bounds and validity (Silhouette in [-1, 1], DBI >= 0, CHI >= 0).
5. Downstream utility consistency (retention lift >= 1.0, valid ANOVA statistics).
6. Multi-seed stability metrics (ARI and NMI in [0, 1]).
7. Experiment manifest schema and completeness.
"""

import sys
from pathlib import Path
import json
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GOLD_DIR
from src.phase6_baselines import (
    FEATURE_SETS,
    load_cohort_splits,
    create_preprocessing_pipeline,
    BASELINES_DIR,
)


def test_preprocessing_and_train_isolation():
    """Verify that preprocessing is fit strictly on train split and transforms val/test safely."""
    train_df, val_df, test_df = load_cohort_splits()
    feat_cols = FEATURE_SETS["RFM_Selected_NonRedundant"]
    
    pipe = create_preprocessing_pipeline()
    
    # 1. Fit on train
    X_train_raw = train_df.select(feat_cols).to_numpy()
    X_train_scaled = pipe.fit_transform(X_train_raw)
    
    # Check that scaled train has approx 0 mean and unit variance
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0.0, atol=1e-2), "Train mean is not 0"
    assert np.allclose(np.std(X_train_scaled, axis=0), 1.0, atol=1e-2), "Train std is not 1"
    
    # 2. Transform test
    X_test_raw = test_df.select(feat_cols).to_numpy()
    X_test_scaled = pipe.transform(X_test_raw)
    
    # Check that test shapes match
    assert X_test_scaled.shape == (8621, len(feat_cols)), "Transformed test shape mismatch"
    assert not np.isnan(X_test_scaled).any(), "NaN values found in transformed test matrix"
    print("  [PASS] test_preprocessing_and_train_isolation passed successfully.")


def test_cutoff_compliance_and_feature_isolation():
    """Verify that feature matrices exclude all future evaluation targets."""
    future_target_cols = [
        "future_orders_count", "future_monetary_value", "future_items_count",
        "future_is_retained", "future_is_churned", "future_days_to_next_order",
        "future_top_category", "future_average_review_score"
    ]
    for feat_set_name, cols in FEATURE_SETS.items():
        overlap = set(cols).intersection(set(future_target_cols))
        assert len(overlap) == 0, f"Future target leakage detected in feature set '{feat_set_name}': {overlap}"
    print("  [PASS] test_cutoff_compliance_and_feature_isolation passed successfully.")


def test_cluster_assignments_integrity():
    """Verify that all 8,621 test customers have non-null cluster assignments across all 7 baselines."""
    assignments_path = BASELINES_DIR / "phase6_baseline_test_assignments.parquet"
    assert assignments_path.exists(), "phase6_baseline_test_assignments.parquet does not exist"
    
    assignments = pl.read_parquet(assignments_path)
    assert len(assignments) == 8621, f"Expected 8,621 test assignments, got {len(assignments)}"
    
    expected_models = [
        "Baseline_1_KMeans_RFM",
        "Baseline_2_KMeans_RFM_Behavioral",
        "Baseline_3_KMeans_RFM_Selected_NonRedundant",
        "Baseline_4_KMeans_RFM_All_Engineered",
        "Baseline_5_GMM",
        "Baseline_6_HDBSCAN",
        "Baseline_7_DEC",
    ]
    for model_name in expected_models:
        assert model_name in assignments.columns, f"Missing model assignments for {model_name}"
        col = assignments[model_name]
        assert col.null_count() == 0, f"Null cluster assignments found in {model_name}"
        unique_c = col.n_unique()
        assert unique_c >= 2, f"Model {model_name} generated fewer than 2 clusters ({unique_c})"
    print("  [PASS] test_cluster_assignments_integrity passed successfully.")


def test_metrics_bounds_and_validity():
    """Verify that internal clustering and downstream utility metrics satisfy mathematical bounds."""
    manifest_path = BASELINES_DIR / "phase6_baselines_manifest.json"
    assert manifest_path.exists(), "phase6_baselines_manifest.json does not exist"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    evaluations = manifest["baselines_evaluation"]
    for model_name, eval_data in evaluations.items():
        iq = eval_data["internal_metrics"]
        st = eval_data["stability_metrics"]
        du = eval_data["downstream_utility"]
        
        # Internal quality bounds
        assert -1.0 <= iq["silhouette_score"] <= 1.0, f"Silhouette out of bounds for {model_name}"
        assert iq["davies_bouldin_index"] >= 0.0, f"DBI negative for {model_name}"
        assert iq["calinski_harabasz_score"] >= 0.0, f"CHI negative for {model_name}"
        
        # Stability bounds
        assert -1.0 <= st["mean_pairwise_ari"] <= 1.0, f"ARI out of bounds for {model_name}"
        assert 0.0 <= st["mean_pairwise_nmi"] <= 1.0, f"NMI out of bounds for {model_name}"
        
        # Downstream utility bounds
        assert du["retention_lift"] >= 1.0, f"Retention lift < 1.0 for {model_name}"
        assert du["max_retention_rate"] >= du["min_retention_rate"], f"Max retention < min retention for {model_name}"
        assert du["cluster_retention_anova_f_stat"] >= 0.0, f"Negative ANOVA F-stat for {model_name}"
    print("  [PASS] test_metrics_bounds_and_validity passed successfully.")


def test_manifest_schema_and_profiles():
    """Verify that the manifest records K-selection, profiles, and configurations properly."""
    manifest_path = BASELINES_DIR / "phase6_baselines_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert "k_selection_methodology" in manifest, "k_selection_methodology missing"
    assert "baselines_evaluation" in manifest, "baselines_evaluation missing"
    
    k_sel = manifest["k_selection_methodology"]
    assert "RFM" in k_sel and "RFM_Selected_NonRedundant" in k_sel and "GMM" in k_sel, "Missing k-selection keys"
    
    # Check profiles structure for Baseline 3
    b3 = manifest["baselines_evaluation"]["Baseline_3_KMeans_RFM_Selected_NonRedundant"]
    assert len(b3["cluster_profiles"]) == 5, f"Expected 5 cluster profiles for B3, got {len(b3['cluster_profiles'])}"
    for p in b3["cluster_profiles"]:
        assert "customer_count" in p and "median_spend" in p and "future_retention_rate" in p
    print("  [PASS] test_manifest_schema_and_profiles passed successfully.")


def run_all_phase6_tests():
    print("=" * 70)
    print("RUNNING PHASE 6 CLASSICAL BASELINES TEST SUITE")
    print("=" * 70)
    test_preprocessing_and_train_isolation()
    test_cutoff_compliance_and_feature_isolation()
    test_cluster_assignments_integrity()
    test_metrics_bounds_and_validity()
    test_manifest_schema_and_profiles()
    print("=" * 70)
    print("ALL PHASE 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_phase6_tests()
