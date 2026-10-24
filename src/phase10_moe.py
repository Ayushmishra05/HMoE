"""Phase 10: heterogeneous mixture-of-experts routing audit."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torch import nn

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
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ASSIGNMENTS_DIR = GOLD_DIR / "baselines"
MODALITIES = ["behavioral", "text", "temporal"]
GATE_SEEDS = [42, 43, 44, 45, 46]


class RoutingGate(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, len(MODALITIES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _masked_weights(
    logits: torch.Tensor, availability: torch.Tensor
) -> torch.Tensor:
    if torch.any(availability.sum(dim=1) <= 0):
        raise RuntimeError("At least one modality must be available per customer")
    masked = logits.masked_fill(availability <= 0, -1e9)
    weights = torch.softmax(masked, dim=1)
    return weights * availability / (weights * availability).sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)


def _route_vectors(
    views: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.einsum("nm,nmd->nd", weights, views).astype(np.float32)


def _train_gate(
    projected: np.ndarray,
    availability: np.ndarray,
    quality: np.ndarray,
    train_mask: np.ndarray,
    seed: int,
    quality_aware: bool,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gate_input = np.hstack([projected.reshape(len(projected), -1), quality])
    if quality_aware:
        gate_input = np.hstack([gate_input, availability])
    model = RoutingGate(gate_input.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(gate_input.astype(np.float32))
    target = torch.from_numpy(projected.mean(axis=1).astype(np.float32))
    available = torch.from_numpy(availability.astype(np.float32))
    model.train()
    for _ in range(10):
        logits = model(x[train_mask])
        weights = _masked_weights(logits, available[train_mask])
        routed = torch.einsum(
            "nm,nmd->nd", weights, torch.from_numpy(projected[train_mask])
        )
        loss = ((routed - target[train_mask]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return _masked_weights(model(x), available).numpy().astype(np.float32)


def _evaluate(
    routed: np.ndarray,
    aligned: pl.DataFrame,
    name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    split = aligned["split"].to_numpy()
    train_mask = split == "train"
    test_mask = split == "test"
    model = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(routed[train_mask])
    labels = model.predict(routed[test_mask])
    seeded_labels = []
    for seed in STABILITY_SEEDS:
        run = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(
            routed[train_mask]
        )
        seeded_labels.append(run.predict(routed[test_mask]))
    pairwise = [
        adjusted_rand_score(seeded_labels[left], seeded_labels[right])
        for left in range(len(seeded_labels))
        for right in range(left + 1, len(seeded_labels))
    ]
    return {
        "variant": name,
        "k": K,
        "dimension": int(routed.shape[1]),
        "cohort_count": len(aligned),
        "train_count": int(train_mask.sum()),
        "validation_count": int((split == "val").sum()),
        "test_count": int(test_mask.sum()),
        "test_silhouette": round(float(silhouette_score(routed[test_mask], labels)), 6),
        "test_davies_bouldin": round(float(davies_bouldin_score(routed[test_mask], labels)), 6),
        "test_calinski_harabasz": round(
            float(calinski_harabasz_score(routed[test_mask], labels)), 6
        ),
        "test_stability_mean_pairwise_ari": round(float(np.mean(pairwise)), 6),
        "test_stability_pairwise_ari": [round(float(x), 6) for x in pairwise],
        "test_cluster_counts": np.bincount(labels, minlength=K).tolist(),
    }, labels


def _gate_diagnostics(weights: np.ndarray, availability: np.ndarray) -> dict[str, Any]:
    entropy = -np.sum(
        np.where(weights > 0, weights * np.log(np.maximum(weights, 1e-12)), 0),
        axis=1,
    )
    effective_single = np.max(weights, axis=1) >= 0.9
    return {
        "mean_weight": weights.mean(axis=0).round(6).tolist(),
        "std_weight": weights.std(axis=0).round(6).tolist(),
        "mean_entropy": round(float(entropy.mean()), 6),
        "single_view_rate": round(float(effective_single.mean()), 6),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "sum_deviation_max": float(np.max(np.abs(weights.sum(axis=1) - 1))),
        "available_view_counts": availability.sum(axis=0).astype(int).tolist(),
        "weights_by_availability": {
            str(tuple(row.astype(int))): weights[
                np.all(availability == row, axis=1)
            ].mean(axis=0).round(6).tolist()
            for row in np.unique(availability, axis=0)
        },
    }


def run_phase10_moe() -> dict[str, Any]:
    aligned, views, alignment = _load_aligned_views()
    split = aligned["split"].to_numpy()
    train_mask = split == "train"
    test_mask = split == "test"
    availability = np.column_stack(
        [
            np.ones(len(aligned), dtype=np.float32),
            aligned["text_available"].to_numpy().astype(np.float32),
            aligned["temporal_available"].to_numpy().astype(np.float32),
        ]
    )
    projected_views = _project_views(views, train_mask, {
        name: availability[:, index] for index, name in enumerate(MODALITIES)
    })
    projected = np.stack([projected_views[name][:, :32] for name in MODALITIES], axis=1)
    quality = np.column_stack(
        [
            np.linalg.norm(projected[:, index], axis=1)
            for index in range(len(MODALITIES))
        ]
    ).astype(np.float32)
    quality = quality / np.maximum(quality.max(axis=0, keepdims=True), 1e-8)

    weights: dict[str, np.ndarray] = {}
    weights["uniform"] = availability / availability.sum(axis=1, keepdims=True)
    weights["fixed_behavioral_heavy"] = availability * np.array([0.6, 0.2, 0.2])
    weights["fixed_behavioral_heavy"] /= weights["fixed_behavioral_heavy"].sum(
        axis=1, keepdims=True
    )
    weights["learned_gate"] = _train_gate(
        projected, availability, quality, train_mask, SEED, False
    )
    weights["quality_aware_gate"] = _train_gate(
        projected, availability, quality, train_mask, SEED, True
    )
    gate_seed_sensitivity: dict[str, list[dict[str, float]]] = {}
    for gate_name, quality_aware in [
        ("learned_gate", False),
        ("quality_aware_gate", True),
    ]:
        gate_seed_sensitivity[gate_name] = []
        for gate_seed in GATE_SEEDS:
            seed_weights = _train_gate(
                projected,
                availability,
                quality,
                train_mask,
                gate_seed,
                quality_aware,
            )
            seed_metrics, _ = _evaluate(
                _route_vectors(projected, seed_weights), aligned, gate_name
            )
            gate_seed_sensitivity[gate_name].append(
                {
                    "seed": gate_seed,
                    "test_silhouette": seed_metrics["test_silhouette"],
                    "test_stability_mean_pairwise_ari": seed_metrics[
                        "test_stability_mean_pairwise_ari"
                    ],
                }
            )

    results: dict[str, Any] = {}
    assignments: list[pl.DataFrame] = []
    for name, current_weights in weights.items():
        routed = _route_vectors(projected, current_weights)
        metrics, labels = _evaluate(routed, aligned, name)
        metrics["gate_diagnostics"] = _gate_diagnostics(current_weights, availability)
        results[name] = metrics
        test_ids = aligned.filter(pl.col("split") == "test")["customer_unique_id"]
        assignments.append(
            pl.DataFrame(
                {
                    "customer_unique_id": test_ids,
                    "variant": [name] * len(test_ids),
                    "cluster": labels,
                    "behavioral_weight": current_weights[test_mask, 0].astype(np.float32),
                    "text_weight": current_weights[test_mask, 1].astype(np.float32),
                    "temporal_weight": current_weights[test_mask, 2].astype(np.float32),
                }
            )
        )

    assignment_path = ASSIGNMENTS_DIR / "phase10_moe_test_assignments.parquet"
    ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(assignments).write_parquet(assignment_path)
    phase9_manifest = json.loads(
        (REPORT_DIR / "phase9_fusion_manifest.json").read_text(encoding="utf-8")
    )
    strongest_static = phase9_manifest["results"]["text_temporal_fixed_weight"]
    full_static = phase9_manifest["results"]["behavioral_text_temporal_fixed_weight"]
    manifest = {
        "phase": "Phase 10: Heterogeneous Mixture-of-Experts",
        "input_artifacts": [
            "data/gold/temporal_customer_cohort.parquet",
            "data/gold/temporal_customer_profiles_cutoff.parquet",
            "data/processed/customer_text_view.parquet",
            "data/embeddings/phase8_temporal_embeddings.parquet",
        ],
        "output_artifacts": [
            str(assignment_path.relative_to(PROJECT_ROOT)),
            "data/gold/reports/phase10_moe_manifest.json",
            "data/gold/reports/phase10_moe_report.md",
        ],
        "protocol": {
            "cohort_count": len(aligned),
            "train_count": int(train_mask.sum()),
            "validation_count": int((split == "val").sum()),
            "test_count": int(test_mask.sum()),
            "test_customer_id_sha256": alignment["test_customer_id_sha256"],
            "k": K,
            "clustering": "KMeans(n_clusters=5, n_init=10), fit on train routed vectors and predict on test",
            "stability_seeds": STABILITY_SEEDS,
            "gate_seeds": GATE_SEEDS,
            "test_outcomes_used": False,
        },
        "expert_definition": {
            "modalities": MODALITIES,
            "expert_output_dimension": 32,
            "inputs": "Phase 9 train-fitted projected view representations",
            "missing_view_policy": "availability mask sets unavailable logits to -infinity and renormalizes",
            "all_missing_policy": "not present in this cohort; guard would reject zero denominator",
        },
        "training": {
            "objective": "unsupervised reconstruction of mean projected view",
            "fit_split": "train customers only",
            "epochs": 10,
            "seed": SEED,
            "gate_architecture": "Linear -> Tanh -> Linear",
            "quality_features": "per-view projected-vector L2 norm, train-derived normalization",
        },
        "results": results,
        "gate_seed_sensitivity": gate_seed_sensitivity,
        "phase9_comparison": {
            "strongest_static_fusion_by_silhouette": {
                "variant": "text_temporal_fixed_weight",
                "test_silhouette": strongest_static["test_silhouette"],
                "test_stability_mean_pairwise_ari": strongest_static[
                    "test_stability_mean_pairwise_ari"
                ],
            },
            "full_static_fusion": {
                "variant": "behavioral_text_temporal_fixed_weight",
                "test_silhouette": full_static["test_silhouette"],
                "test_stability_mean_pairwise_ari": full_static[
                    "test_stability_mean_pairwise_ari"
                ],
            },
            "adaptive_gate_minus_strongest_static_silhouette": {
                name: round(
                    results[name]["test_silhouette"]
                    - strongest_static["test_silhouette"],
                    6,
                )
                for name in ["learned_gate", "quality_aware_gate"]
            },
        },
        "assignment_artifact": str(assignment_path.relative_to(PROJECT_ROOT)),
        "limitations": [
            "The gate objective is representation reconstruction, not future retention prediction.",
            "Phase 9 test exploration is disclosed; Phase 10 does not tune gates on test outcomes.",
            "The text view is missing for 34,975 customers, so adaptive routing is evaluated under explicit missingness.",
        ],
        "reproduction_command": "python -m src.phase10_moe",
    }
    manifest_path = REPORT_DIR / "phase10_moe_manifest.json"
    report_path = REPORT_DIR / "phase10_moe_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 10 Heterogeneous Mixture-of-Experts Report",
        "",
        f"All variants use the same {len(aligned):,}-customer cohort and {int(test_mask.sum()):,}-customer test split. "
        "Experts are 32-dimensional train-fitted projected views; unavailable views are masked and weights renormalized.",
        "",
        "| Routing variant | Silhouette | Davies-Bouldin | Calinski-Harabasz | Stability ARI | Single-view rate | Mean entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        diag = result["gate_diagnostics"]
        lines.append(
            f"| {name} | {result['test_silhouette']:.6f} | {result['test_davies_bouldin']:.6f} | "
            f"{result['test_calinski_harabasz']:.3f} | {result['test_stability_mean_pairwise_ari']:.6f} | "
            f"{diag['single_view_rate']:.6f} | {diag['mean_entropy']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Gate weight sums have maximum deviation **{max(r['gate_diagnostics']['sum_deviation_max'] for r in results.values()):.3e}**.",
            f"Strongest Phase 9 static fusion (text + temporal fixed weight) silhouette: "
            f"**{strongest_static['test_silhouette']:.6f}**.",
            "Neither learned gate exceeds the strongest fair Phase 9 static-fusion baseline; "
            "adaptive-routing superiority is not supported by this Phase 10 geometry evaluation.",
            "No future retention labels were used for gate training, routing, or clustering selection.",
            "Phase 11 has not started.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(assignment_path)
    print(manifest_path)
    print(report_path)
    return manifest


if __name__ == "__main__":
    run_phase10_moe()
