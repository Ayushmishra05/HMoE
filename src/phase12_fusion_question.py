"""Phase 12: frozen temporal versus text-plus-temporal comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import GOLD_DIR
from src.phase9_fusion import (
    K,
    SEED,
    STABILITY_SEEDS,
    _load_aligned_views,
    _project_views,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = GOLD_DIR / "reports"


def _evaluate(matrix: np.ndarray, aligned: pl.DataFrame) -> dict[str, Any]:
    split = aligned["split"].to_numpy()
    train_mask = split == "train"
    test_mask = split == "test"
    model = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(
        matrix[train_mask]
    )
    labels = model.predict(matrix[test_mask])
    seeded = []
    for seed in STABILITY_SEEDS:
        seeded_model = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(
            matrix[train_mask]
        )
        seeded.append(seeded_model.predict(matrix[test_mask]))
    pairwise_ari = [
        adjusted_rand_score(seeded[left], seeded[right])
        for left in range(len(seeded))
        for right in range(left + 1, len(seeded))
    ]
    return {
        "dimension": int(matrix.shape[1]),
        "metric_family": "Euclidean distance used by silhouette, Davies-Bouldin, and CH",
        "test_silhouette": round(float(silhouette_score(matrix[test_mask], labels)), 6),
        "test_davies_bouldin": round(
            float(davies_bouldin_score(matrix[test_mask], labels)), 6
        ),
        "test_calinski_harabasz": round(
            float(calinski_harabasz_score(matrix[test_mask], labels)), 6
        ),
        "test_stability_mean_pairwise_ari": round(float(np.mean(pairwise_ari)), 6),
        "test_stability_pairwise_ari": [
            round(float(value), 6) for value in pairwise_ari
        ],
        "test_cluster_counts": np.bincount(labels, minlength=K).tolist(),
    }


def run_phase12_fusion_question() -> dict[str, Any]:
    aligned, views, alignment = _load_aligned_views()
    split = aligned["split"].to_numpy()
    train_mask = split == "train"
    availability = {
        "behavioral": np.ones(len(aligned), dtype=np.float32),
        "text": aligned["text_available"].to_numpy().astype(np.float32),
        "temporal": aligned["temporal_available"].to_numpy().astype(np.float32),
    }
    projected = _project_views(views, train_mask, availability)
    text_projected = projected["text"]
    temporal_projected = projected["temporal"]

    # The indicator-free sensitivity removes the explicit text availability
    # column and preserves missing text as a zero projected vector.
    text_without_indicator = views["text"][:, :-1]
    indicator_free_views = {
        "text": text_without_indicator,
        "temporal": views["temporal"],
    }
    indicator_free = _project_views(
        indicator_free_views,
        train_mask,
        {"text": availability["text"], "temporal": availability["temporal"]},
    )
    indicator_free["text"][availability["text"] == 0] = 0.0

    matrices = {
        "temporal_only": temporal_projected,
        "text_temporal_availability_aware": np.hstack(
            [
                (text_projected + temporal_projected) / 2,
                availability["text"][:, None],
                availability["temporal"][:, None],
            ]
        ),
        "text_temporal_indicator_free": (
            indicator_free["text"] + indicator_free["temporal"]
        )
        / 2,
    }
    results = {
        name: _evaluate(matrix, aligned) for name, matrix in matrices.items()
    }
    temporal_silhouette = results["temporal_only"]["test_silhouette"]
    for name, result in results.items():
        result["silhouette_delta_vs_temporal"] = round(
            result["test_silhouette"] - temporal_silhouette, 6
        )
        result["stability_delta_vs_temporal"] = round(
            result["test_stability_mean_pairwise_ari"]
            - results["temporal_only"]["test_stability_mean_pairwise_ari"],
            6,
        )

    manifest = {
        "phase": "Phase 12: multimodal fusion research question",
        "research_question": (
            "When does multimodal fusion help beyond a temporal-only representation?"
        ),
        "input_artifacts": [
            "data/gold/temporal_customer_cohort.parquet",
            "data/gold/temporal_customer_profiles_cutoff.parquet",
            "data/processed/customer_text_view.parquet",
            "data/embeddings/phase8_temporal_embeddings.parquet",
            "data/gold/reports/phase9_fusion_manifest.json",
        ],
        "protocol": {
            "cohort_count": len(aligned),
            "train_count": int(train_mask.sum()),
            "validation_count": int((split == "val").sum()),
            "test_count": int((split == "test").sum()),
            "test_customer_id_sha256": alignment["test_customer_id_sha256"],
            "k": K,
            "clustering": "KMeans(n_clusters=5, n_init=10), fit on train and predict on test",
            "stability": (
                "For each seed, refit KMeans on training customers, predict the same "
                "test customers, then compute pairwise label-permutation-invariant ARI."
            ),
            "distance_metric": "Euclidean for all internal clustering metrics",
            "test_outcomes_used_for_selection": False,
        },
        "availability": {
            "text_available_count": alignment["text_available_count"],
            "text_missing_count": len(aligned) - alignment["text_available_count"],
            "temporal_available_count": alignment["temporal_available_count"],
            "aware_protocol": (
                "Projected text and temporal views averaged with explicit text and "
                "temporal availability indicators."
            ),
            "indicator_free_protocol": (
                "Text availability indicator removed; unavailable text remains a zero "
                "projected vector after train-fitted PCA."
            ),
        },
        "results": results,
        "interpretation": {
            "text_adds_reproducible_value": (
                results["text_temporal_indicator_free"]["test_silhouette"]
                > temporal_silhouette
                and results["text_temporal_indicator_free"][
                    "test_stability_mean_pairwise_ari"
                ]
                >= results["temporal_only"]["test_stability_mean_pairwise_ari"]
            ),
            "retention_claim_status": (
                "Not assessed for model selection; behavioral-heavy retention "
                "finding remains exploratory until an independent holdout confirms it."
            ),
        },
        "results_are_descriptive": True,
        "reproduction_command": "python -m src.phase12_fusion_question",
    }
    manifest_path = REPORT_DIR / "phase12_fusion_question_manifest.json"
    report_path = REPORT_DIR / "phase12_fusion_question_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 12 Multimodal Fusion Research Question",
        "",
        "Research question: **When does multimodal fusion help beyond temporal-only representation?**",
        "",
        "All variants use the same 57,510-customer cohort, 8,621 test customers, "
        "train-only projection, KMeans k=5, and Euclidean internal metrics.",
        "",
        "| Variant | Dim. | Silhouette | Delta vs temporal | DBI | CH | Stability ARI | Delta ARI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['dimension']} | {result['test_silhouette']:.6f} | "
            f"{result['silhouette_delta_vs_temporal']:+.6f} | "
            f"{result['test_davies_bouldin']:.6f} | "
            f"{result['test_calinski_harabasz']:.3f} | "
            f"{result['test_stability_mean_pairwise_ari']:.6f} | "
            f"{result['stability_delta_vs_temporal']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol audit",
            "",
            "- Stability ARI compares independently refit KMeans test assignments; it is not a comparison to ground-truth labels.",
            "- Every internal metric uses Euclidean geometry on the same test population.",
            "- The availability-aware fusion includes explicit modality indicators.",
            "- The indicator-free sensitivity removes the text availability indicator and zeros unavailable projected text.",
            "- Future retention outcomes were not used for preprocessing, fitting, or selection.",
            "",
            "## Retention limitation",
            "",
            "The behavioral-heavy retention result is exploratory. Its binary retention "
            "finding has not been independently confirmed on a separate outcome holdout. "
            "No retention outcome was used in this comparison.",
            "",
            "## Decision",
            "",
            "Text adds reproducible value beyond temporal geometry only if the "
            "indicator-free comparison improves both silhouette and stability. This "
            "report records that decision rule before outcome inspection.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(manifest_path)
    print(report_path)
    return manifest


if __name__ == "__main__":
    run_phase12_fusion_question()
