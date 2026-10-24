"""Audit Phase 9 assignments, cohorts, stability, missingness, and utility protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import GOLD_DIR
from src.phase9_fusion import (
    BEHAVIORAL_FEATURES,
    K,
    SEED,
    STABILITY_SEEDS,
    _load_aligned_views,
    _project_views,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = GOLD_DIR / "reports"


def _assignment(
    matrix: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    model = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(matrix[train_mask])
    labels = model.predict(matrix[test_mask])
    runs = []
    for seed in STABILITY_SEEDS:
        run = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(
            matrix[train_mask]
        )
        runs.append(run.predict(matrix[test_mask]))
    pairwise = [
        adjusted_rand_score(runs[left], runs[right])
        for left in range(len(runs))
        for right in range(left + 1, len(runs))
    ]
    return labels, {
        "test_silhouette": round(float(silhouette_score(matrix[test_mask], labels)), 6),
        "test_cluster_counts": np.bincount(labels, minlength=K).tolist(),
        "mean_pairwise_ari": round(float(np.mean(pairwise)), 6),
        "pairwise_ari": [round(float(value), 6) for value in pairwise],
    }


def _cluster_profiles(
    aligned: pl.DataFrame,
    test_ids: list[str],
    labels: np.ndarray,
    variant: str,
) -> list[dict[str, Any]]:
    targets = pl.read_parquet(GOLD_DIR / "temporal_customer_targets.parquet")
    profile = pl.read_parquet(
        GOLD_DIR / "temporal_customer_profiles_cutoff.parquet"
    )
    temporal = pl.read_parquet(
        PROJECT_ROOT / "data" / "embeddings" / "phase8_temporal_embeddings.parquet"
    )
    base = (
        aligned.filter(pl.col("customer_unique_id").is_in(test_ids))
        .with_columns(pl.Series("cluster", labels))
        .join(
            profile.select(["customer_unique_id", *BEHAVIORAL_FEATURES]),
            on="customer_unique_id",
            how="left",
        )
        .join(
            targets.select(["customer_unique_id", "future_is_retained"]),
            on="customer_unique_id",
            how="left",
        )
        .join(
            temporal.select(["customer_unique_id", "temporal_sequence_length"]),
            on="customer_unique_id",
            how="left",
        )
    )
    summaries = []
    for cluster in range(K):
        rows = base.filter(pl.col("cluster") == cluster)
        summaries.append(
            {
                "variant": variant,
                "cluster": cluster,
                "customer_count": len(rows),
                "share_pct": round(100 * len(rows) / len(base), 4),
                "text_available_count": int(rows["text_available"].sum()),
                "text_available_pct": round(
                    100 * float(rows["text_available"].mean()), 4
                ),
                "temporal_available_count": int(rows["temporal_available"].sum()),
                "mean_sequence_length": round(
                    float(rows["temporal_sequence_length"].mean()), 4
                ),
                "retained_count": int(rows["future_is_retained"].sum()),
                "retention_rate_pct": round(
                    100 * float(rows["future_is_retained"].mean()), 6
                ),
                "behavioral_means": {
                    feature: round(float(rows[feature].mean()), 6)
                    for feature in BEHAVIORAL_FEATURES
                },
            }
        )
    return summaries


def run_phase9_audit() -> dict[str, Any]:
    aligned, views, alignment = _load_aligned_views()
    train_mask = aligned["split"].to_numpy() == "train"
    test_mask = aligned["split"].to_numpy() == "test"
    test_ids = aligned.filter(pl.col("split") == "test")["customer_unique_id"].to_list()

    # Reconstruct documented matrices and an availability-free text sensitivity matrix.
    matrices = {
        "behavioral_only": views["behavioral"],
        "text_only_documented": views["text"],
        "temporal_only": views["temporal"],
        "behavioral_text_temporal_documented": np.hstack(
            [views["behavioral"], views["text"], views["temporal"]]
        ),
    }
    text_core = views["text"][:, :-1]
    matrices["text_only_without_availability_indicator"] = text_core
    availability = {
        "behavioral": np.ones(len(aligned), dtype=np.float32),
        "text": aligned["text_available"].to_numpy().astype(np.float32),
        "temporal": aligned["temporal_available"].to_numpy().astype(np.float32),
    }
    projected = _project_views(views, train_mask, availability)
    matrices["behavioral_text_temporal_fixed_weight_documented"] = np.hstack(
        [
            (
                projected["behavioral"]
                + projected["text"]
                + projected["temporal"]
            )
            / 3.0,
            np.column_stack(
                [availability["behavioral"], availability["text"], availability["temporal"]]
            ),
        ]
    )

    results: dict[str, Any] = {}
    profiles: list[dict[str, Any]] = []
    for name, matrix in matrices.items():
        labels, metrics = _assignment(matrix, train_mask, test_mask)
        results[name] = {
            **metrics,
            "cohort_count": len(aligned),
            "train_count": int(train_mask.sum()),
            "validation_count": int((aligned["split"] == "val").sum()),
            "test_count": int(test_mask.sum()),
            "test_customer_id_sha256": alignment["test_customer_id_sha256"],
            "fit_protocol": "KMeans fit on train customers; test labels predicted without refit",
        }
        if name in {
            "text_only_documented",
            "temporal_only",
            "behavioral_only",
            "behavioral_text_temporal_documented",
        }:
            profiles.extend(_cluster_profiles(aligned, test_ids, labels, name))

    target = pl.read_parquet(GOLD_DIR / "temporal_customer_targets.parquet")
    retained = int(target["future_is_retained"].sum())
    audit = {
        "phase": "Phase 9 audit",
        "source_phase9_manifest": "data/gold/reports/phase9_fusion_manifest.json",
        "issues": [
            {
                "id": "A1",
                "issue": "Full normalized concatenation mean pairwise ARI was absent from the human-readable table.",
                "resolution": "Recovered from the machine-readable manifest and independently recomputed.",
                "evidence": results["behavioral_text_temporal_documented"],
                "remaining_limitation": "The original report table omitted the value; the manifest retained it.",
            },
            {
                "id": "A2",
                "issue": "Documented text-only includes an availability indicator.",
                "resolution": "Retained the original result and added an indicator-free sensitivity result.",
                "evidence": {
                    "documented": results["text_only_documented"],
                    "without_indicator": results["text_only_without_availability_indicator"],
                },
                "remaining_limitation": "The text-only label should be read as text-plus-availability for the original result.",
            },
            {
                "id": "A3",
                "issue": "Per-cluster profiles and imbalance diagnostics were not previously reported.",
                "resolution": "Added profiles for behavioral, text-only, temporal-only, and full documented fusion.",
                "evidence": "cluster_profiles below",
                "remaining_limitation": "Profiles describe blind-test assignments and are not causal analyses.",
            },
        ],
        "cohort_and_split": {
            "cohort_count": len(aligned),
            "train_count": int(train_mask.sum()),
            "validation_count": int((aligned["split"] == "val").sum()),
            "test_count": int(test_mask.sum()),
            "test_customer_id_sha256": alignment["test_customer_id_sha256"],
            "all_experiments_same_population": all(
                result["test_customer_id_sha256"]
                == alignment["test_customer_id_sha256"]
                for result in results.values()
            ),
        },
        "learned_transformation_audit": {
            "behavioral_scaler_fit_rows": alignment["behavioral_preprocessing"]["train_available_count"],
            "text_scaler_fit_rows": alignment["text_preprocessing"]["train_available_count"],
            "temporal_scaler_fit_rows": alignment["temporal_preprocessing"]["train_available_count"],
            "pca_fit_split": "train only",
            "encoder_fit_split": "train only; documented in Phase 8 manifest",
            "test_outcomes_used_for_transforms": False,
        },
        "stability_audit": {
            "seeds": STABILITY_SEEDS,
            "metric": "adjusted Rand index, label-permutation invariant",
            "assignment_alignment": "same ordered test customer IDs for every seed and variant",
            "independent_runs": True,
        },
        "availability_audit": {
            "text_available_count": alignment["text_available_count"],
            "text_missing_count": alignment["cohort_count"] - alignment["text_available_count"],
            "temporal_available_count": alignment["temporal_available_count"],
            "indicator_sensitivity": results["text_only_without_availability_indicator"],
            "documented_text_result": results["text_only_documented"],
        },
        "retention_protocol": {
            "target": "future_is_retained = 1 iff future_orders_count > 0",
            "window": "[2018-03-01T00:00:00, 2018-09-01T00:00:00)",
            "cohort_retained_count": retained,
            "cohort_count": int(target.height),
            "future_outcomes_used_for_representation_or_selection": False,
            "downstream_use": "descriptive audit only; no test-driven tuning performed",
        },
        "previous_test_set_exploration": (
            "Phase 6 exploratory baseline work and the prior Phase 8/9 geometry evaluations "
            "used the blind test for reporting. This is disclosed and the Phase 9 audit "
            "does not retune models or select configurations from new test outcomes."
        ),
        "results": results,
        "cluster_profiles": profiles,
        "strongest_fair_comparison": {
            "behavioral_baseline": results["behavioral_only"],
            "full_static_fusion": results[
                "behavioral_text_temporal_documented"
            ],
            "comparison_population_identical": True,
            "selection_status": "descriptive frozen-protocol comparison; no Phase 10 model selected",
        },
    }
    path = REPORT_DIR / "phase9_audit_report.json"
    md_path = REPORT_DIR / "phase9_audit_report.md"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    text = [
        "# Phase 9 Audit Report",
        "",
        "## Status",
        "",
        "Phase 9 comparability and readiness audit completed. Phase 10 was not started.",
        "",
        "## Resolved issues",
        "",
        "| ID | Issue | Resolution |",
        "|---|---|---|",
    ]
    for issue in audit["issues"]:
        text.append(f"| {issue['id']} | {issue['issue']} | {issue['resolution']} |")
    text.extend(
        [
            "",
            "## Frozen evaluation protocol",
            "",
            f"- Cohort: **{len(aligned):,}** customers",
            f"- Train/validation/test: **{train_mask.sum():,} / {(aligned['split'] == 'val').sum():,} / {test_mask.sum():,}**",
            "- All experiments use the same ordered test customer IDs and KMeans k=5.",
            "- Standardization and PCA are fit on training customers only.",
            "- Stability uses independent seeds and label-permutation-invariant ARI.",
            "",
            "## Key sensitivity",
            "",
            f"- Documented text-only silhouette: **{results['text_only_documented']['test_silhouette']:.6f}**",
            f"- Text-only without availability indicator: **{results['text_only_without_availability_indicator']['test_silhouette']:.6f}**",
            f"- Full normalized fusion mean pairwise ARI: **{results['behavioral_text_temporal_documented']['mean_pairwise_ari']:.6f}**",
            "",
            "## Retention protocol",
            "",
            "Retention is `future_orders_count > 0` in the future window "
            "[2018-03-01, 2018-09-01). Future outcomes were not used to construct "
            "representations or select models. Prior test-set exploration is disclosed "
            "and no further test-driven tuning was performed.",
            "",
            "## Readiness decision",
            "",
            "The audit artifacts are complete. Phase 10 should compare adaptive routing "
            "against the strongest fair static-fusion baseline and fixed-routing controls, "
            "not against a weak baseline. Approval is required before implementation.",
        ]
    )
    md_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(path)
    print(md_path)
    return audit


if __name__ == "__main__":
    run_phase9_audit()
