"""Phase 9: static multimodal fusion baselines."""

from __future__ import annotations

import hashlib
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import GOLD_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = GOLD_DIR / "reports"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
BEHAVIORAL_FEATURES = [
    "recency_days",
    "frequency_orders",
    "monetary_value",
    "total_items_purchased",
    "unique_categories",
    "unique_sellers",
    "average_installments",
    "review_count",
]
K = 5
SEED = 42
STABILITY_SEEDS = [42, 43, 44, 45, 46]


def _row_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1e-12))


def _prepare_view(
    values: np.ndarray, train_mask: np.ndarray, available: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(values[train_mask & available])
    transformed = scaler.transform(values).astype(np.float32)
    transformed[~available] = 0.0
    transformed = _row_normalize(transformed)
    transformed[~available] = 0.0
    return transformed, {
        "scaler": "StandardScaler fit on available training customers only",
        "train_available_count": int((train_mask & available).sum()),
        "available_count": int(available.sum()),
        "dimension": int(transformed.shape[1]),
    }


def _load_aligned_views() -> tuple[pl.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    cohort = pl.read_parquet(GOLD_DIR / "temporal_customer_cohort.parquet")
    profile = pl.read_parquet(GOLD_DIR / "temporal_customer_profiles_cutoff.parquet")
    text = pl.read_parquet(PROJECT_ROOT / "data" / "processed" / "customer_text_view.parquet")
    temporal = pl.read_parquet(EMBEDDINGS_DIR / "phase8_temporal_embeddings.parquet")
    customer_ids = cohort["customer_unique_id"].to_list()
    cohort_ids = set(customer_ids)

    for name, frame in {"profile": profile, "text": text, "temporal": temporal}.items():
        ids = frame["customer_unique_id"].to_list()
        if len(ids) != len(set(ids)) or set(ids) != cohort_ids:
            raise ValueError(f"{name} view is not exactly aligned to the cohort")

    aligned = (
        cohort.select(["customer_unique_id", "split"])
        .join(profile.select(["customer_unique_id", *BEHAVIORAL_FEATURES]), on="customer_unique_id")
        .join(text.select(["customer_unique_id", "text_embedding", "text_available"]), on="customer_unique_id")
        .join(
            temporal.select(["customer_unique_id", "temporal_embedding", "temporal_available"]),
            on="customer_unique_id",
        )
    )
    if len(aligned) != len(cohort) or aligned["customer_unique_id"].n_unique() != len(cohort):
        raise ValueError("Aligned multimodal view is not one row per cohort customer")

    train_mask = aligned["split"].to_numpy() == "train"
    behavior = aligned.select(BEHAVIORAL_FEATURES).to_numpy().astype(np.float32)
    text_available = aligned["text_available"].to_numpy().astype(bool)
    temporal_available = aligned["temporal_available"].to_numpy().astype(bool)
    text_rows = aligned["text_embedding"].to_list()
    text_dimension = next(len(row) for row in text_rows if row is not None)
    text_values = np.asarray(
        [
            row if row is not None else [0.0] * text_dimension
            for row in text_rows
        ],
        dtype=np.float32,
    )
    temporal_values = np.asarray(
        aligned["temporal_embedding"].to_list(), dtype=np.float32
    )
    behavior_available = np.ones(len(aligned), dtype=bool)
    behavior_view, behavior_meta = _prepare_view(
        behavior, train_mask, behavior_available
    )
    text_view, text_meta = _prepare_view(text_values, train_mask, text_available)
    temporal_view, temporal_meta = _prepare_view(
        temporal_values, train_mask, temporal_available
    )
    views = {
        "behavioral": behavior_view,
        "text": np.hstack([text_view, text_available.astype(np.float32)[:, None]]),
        "temporal": temporal_view,
    }
    return aligned, views, {
        "cohort_count": len(aligned),
        "train_count": int(train_mask.sum()),
        "validation_count": int((aligned["split"] == "val").sum()),
        "test_count": int((aligned["split"] == "test").sum()),
        "text_available_count": int(text_available.sum()),
        "temporal_available_count": int(temporal_available.sum()),
        "behavioral_preprocessing": behavior_meta,
        "text_preprocessing": {**text_meta, "missing_policy": "zero transformed vector plus explicit availability mask"},
        "temporal_preprocessing": temporal_meta,
        "test_customer_id_sha256": hashlib.sha256(
            "\n".join(
                aligned.filter(pl.col("split") == "test")["customer_unique_id"].to_list()
            ).encode("utf-8")
        ).hexdigest(),
    }


def _evaluate_view(
    matrix: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> dict[str, Any]:
    model = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(matrix[train_mask])
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
        "k": K,
        "dimension": int(matrix.shape[1]),
        "test_silhouette": round(float(silhouette_score(matrix[test_mask], labels)), 6),
        "test_davies_bouldin": round(float(davies_bouldin_score(matrix[test_mask], labels)), 6),
        "test_calinski_harabasz": round(float(calinski_harabasz_score(matrix[test_mask], labels)), 6),
        "test_stability_mean_pairwise_ari": round(float(np.mean(pairwise_ari)), 6),
        "test_stability_pairwise_ari": [round(float(value), 6) for value in pairwise_ari],
        "test_cluster_counts": np.bincount(labels, minlength=K).tolist(),
    }


def _project_views(
    views: dict[str, np.ndarray],
    train_mask: np.ndarray,
    availability: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    projected: dict[str, np.ndarray] = {}
    for name, values in views.items():
        fit_values = values[train_mask & availability[name].astype(bool)]
        dimension = min(32, values.shape[1], len(fit_values))
        projector = PCA(n_components=dimension, random_state=SEED).fit(
            fit_values
        )
        transformed = projector.transform(values).astype(np.float32)
        if dimension < 32:
            transformed = np.pad(
                transformed, ((0, 0), (0, 32 - dimension)), mode="constant"
            )
        projected[name] = transformed
    return projected


def run_phase9_fusion() -> dict[str, Any]:
    """Build and evaluate the required static fusion matrix."""
    aligned, views, alignment = _load_aligned_views()
    train_mask = aligned["split"].to_numpy() == "train"
    test_mask = aligned["split"].to_numpy() == "test"
    combinations = {
        "behavioral_only": ["behavioral"],
        "text_only": ["text"],
        "temporal_only": ["temporal"],
        "behavioral_text": ["behavioral", "text"],
        "behavioral_temporal": ["behavioral", "temporal"],
        "text_temporal": ["text", "temporal"],
        "behavioral_text_temporal": ["behavioral", "text", "temporal"],
    }
    results = {}
    for name, components in combinations.items():
        matrix = np.hstack([views[component] for component in components])
        results[name] = {
            "components": components,
            "fusion": "normalized concatenation with equal fixed component weights",
            **_evaluate_view(matrix, train_mask, test_mask),
        }
    availability = {
        "behavioral": np.ones(len(aligned), dtype=np.float32),
        "text": aligned["text_available"].to_numpy().astype(np.float32),
        "temporal": aligned["temporal_available"].to_numpy().astype(np.float32),
    }
    projected = _project_views(views, train_mask, availability)
    fixed_weight_combinations = {
        "behavioral_text_fixed_weight": ["behavioral", "text"],
        "behavioral_temporal_fixed_weight": ["behavioral", "temporal"],
        "text_temporal_fixed_weight": ["text", "temporal"],
        "behavioral_text_temporal_fixed_weight": [
            "behavioral",
            "text",
            "temporal",
        ],
    }
    for name, components in fixed_weight_combinations.items():
        matrix = sum(projected[component] for component in components) / len(components)
        matrix = np.hstack(
            [matrix, np.column_stack([availability[component] for component in components])]
        )
        results[name] = {
            "components": components,
            "fusion": "train-fitted PCA to 32 dimensions, equal fixed weights, availability indicators",
            **_evaluate_view(matrix, train_mask, test_mask),
        }
    manifest = {
        "phase": "Phase 9: Static multimodal fusion",
        "input_artifacts": [
            "data/gold/temporal_customer_cohort.parquet",
            "data/gold/temporal_customer_profiles_cutoff.parquet",
            "data/processed/customer_text_view.parquet",
            "data/embeddings/phase8_temporal_embeddings.parquet",
        ],
        "output_artifacts": [
            "data/gold/reports/phase9_fusion_manifest.json",
            "data/gold/reports/phase9_fusion_report.md",
        ],
        "protocol": {
            "cohort_count": alignment["cohort_count"],
            "train_count": alignment["train_count"],
            "validation_count": alignment["validation_count"],
            "test_count": alignment["test_count"],
            "k": K,
            "clustering": "KMeans(n_clusters=5, n_init=10), fit on train and predict on test",
            "seeds": STABILITY_SEEDS,
            "test_customer_id_sha256": alignment["test_customer_id_sha256"],
            "test_outcomes_used_for_selection": False,
        },
        "alignment_audit": alignment,
        "fusion_policy": {
            "per_view_preprocessing": "train-only StandardScaler followed by per-customer L2 normalization",
            "fixed_weights": "equal weight per available view through normalized concatenation",
            "text_missingness": "zero transformed text vector plus explicit text_available scalar",
            "temporal_missingness": "explicit temporal_available flag; no missing temporal rows observed",
            "projected_fixed_weight": "each view projected to 32 train-fitted PCA dimensions, then averaged with equal fixed weights and availability indicators",
        },
        "results": results,
        "reproduction_command": "python -m src.phase9_fusion",
        "limitations": [
            "The static comparison evaluates geometry and stability; downstream retention utility remains a separate audited analysis.",
            "The text view has substantial missingness and uses an explicit availability feature rather than imputing observed text.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT_DIR / "phase9_fusion_manifest.json"
    report_path = REPORT_DIR / "phase9_fusion_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 9 Static Multimodal Fusion Report",
        "",
        f"All variants use the same **{alignment['cohort_count']:,}**-customer cohort, "
        f"including the same **{alignment['test_count']:,}** blind-test customers. "
        "Each view is standardized on training customers only, row-normalized, and clustered "
        "with KMeans at fixed k=5.",
        "",
        "| Variant | Components | Dim. | Silhouette | Davies-Bouldin | Calinski-Harabasz | Stability ARI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {', '.join(result['components'])} | {result['dimension']} | "
            f"{result['test_silhouette']:.6f} | {result['test_davies_bouldin']:.6f} | "
            f"{result['test_calinski_harabasz']:.3f} | "
            f"{result['test_stability_mean_pairwise_ari']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Text available customers: **{alignment['text_available_count']:,}**; "
            f"temporal available customers: **{alignment['temporal_available_count']:,}**.",
            "",
            "No future retention outcomes were used for preprocessing, fusion weighting, "
            "or clustering selection. Phase 10 has not started.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(manifest_path)
    print(report_path)
    return manifest


if __name__ == "__main__":
    run_phase9_fusion()
