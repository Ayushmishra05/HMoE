"""Phase 11: ablations, robustness, and evidence assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, fisher_exact

from src.config import GOLD_DIR
from src.phase9_fusion import (
    K,
    SEED,
    STABILITY_SEEDS,
    _load_aligned_views,
    _project_views,
)
from src.phase10_moe import (
    _evaluate,
    _route_vectors,
    _train_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = GOLD_DIR / "reports"
BASELINE_DIR = GOLD_DIR / "baselines"
MODALITIES = ["behavioral", "text", "temporal"]


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [round(float(centre - margin), 8), round(float(centre + margin), 8)]


def _retention_summary(
    assignments: np.ndarray, customer_ids: list[str], targets: pl.DataFrame
) -> list[dict[str, Any]]:
    target_map = dict(
        zip(
            targets["customer_unique_id"].to_list(),
            targets["future_is_retained"].to_list(),
        )
    )
    retained = np.array([target_map[customer_id] for customer_id in customer_ids])
    summaries = []
    for cluster in sorted(np.unique(assignments)):
        mask = assignments == cluster
        count = int(mask.sum())
        successes = int(retained[mask].sum())
        summaries.append(
            {
                "cluster": int(cluster),
                "customer_count": count,
                "retained_count": successes,
                "retention_rate_pct": round(100 * successes / count, 8),
                "wilson_95_ci_pct": [round(100 * x, 8) for x in _wilson(successes, count)],
            }
        )
    return summaries


def _retention_tests(
    assignments: np.ndarray, customer_ids: list[str], targets: pl.DataFrame
) -> dict[str, Any]:
    target_map = dict(
        zip(
            targets["customer_unique_id"].to_list(),
            targets["future_is_retained"].to_list(),
        )
    )
    retained = np.array([target_map[customer_id] for customer_id in customer_ids])
    clusters = sorted(np.unique(assignments))
    contingency = np.array(
        [
            [
                int(((assignments == cluster) & (retained == outcome)).sum())
                for outcome in [0, 1]
            ]
            for cluster in clusters
        ]
    )
    chi2, omnibus_p, _, _ = chi2_contingency(contingency, correction=False)
    pairs = []
    raw_p = []
    for left_index, left in enumerate(clusters):
        for right in clusters[left_index + 1 :]:
            table = np.array(
                [
                    [
                        int(((assignments == left) & (retained == 1)).sum()),
                        int(((assignments == left) & (retained == 0)).sum()),
                    ],
                    [
                        int(((assignments == right) & (retained == 1)).sum()),
                        int(((assignments == right) & (retained == 0)).sum()),
                    ],
                ]
            )
            _, p_value = fisher_exact(table, alternative="two-sided")
            pairs.append({"cluster_left": int(left), "cluster_right": int(right)})
            raw_p.append(p_value)
    ordered = sorted(enumerate(raw_p), key=lambda item: item[1])
    adjusted = [0.0] * len(raw_p)
    for rank, (index, p_value) in enumerate(ordered):
        adjusted[index] = min(1.0, (len(raw_p) - rank) * p_value)
    for rank in range(1, len(ordered)):
        previous_index = ordered[rank - 1][0]
        current_index = ordered[rank][0]
        adjusted[current_index] = max(adjusted[current_index], adjusted[previous_index])
    for pair, p_value, adjusted_p in zip(pairs, raw_p, adjusted):
        pair["p_value"] = round(float(p_value), 10)
        pair["holm_adjusted_p_value"] = round(float(adjusted_p), 10)
        pair["significant_at_0.05"] = bool(adjusted_p < 0.05)
    return {
        "pearson_chi_square": round(float(chi2), 8),
        "omnibus_p_value": round(float(omnibus_p), 10),
        "pairwise_fisher_holm": pairs,
    }


def _evaluate_matrix(
    matrix: np.ndarray, aligned: pl.DataFrame
) -> tuple[dict[str, Any], np.ndarray]:
    return _evaluate(matrix, aligned, "phase11")


def run_phase11_analysis() -> dict[str, Any]:
    aligned, views, alignment = _load_aligned_views()
    split = aligned["split"].to_numpy()
    train_mask = split == "train"
    test_mask = split == "test"
    test_ids = aligned.filter(pl.col("split") == "test")["customer_unique_id"].to_list()
    targets = pl.read_parquet(GOLD_DIR / "temporal_customer_targets.parquet")

    availability = {
        "behavioral": np.ones(len(aligned), dtype=np.float32),
        "text": aligned["text_available"].to_numpy().astype(np.float32),
        "temporal": aligned["temporal_available"].to_numpy().astype(np.float32),
    }
    projected_map = _project_views(views, train_mask, availability)
    projected = np.stack(
        [projected_map[name][:, :32] for name in MODALITIES], axis=1
    )
    projected_views = {
        name: projected[:, index] for index, name in enumerate(MODALITIES)
    }

    def fixed_fusion(names: list[str]) -> np.ndarray:
        matrix = sum(projected_views[name] for name in names) / len(names)
        indicators = np.column_stack([availability[name] for name in names])
        return np.hstack([matrix, indicators])

    def available_mean(names: list[str]) -> np.ndarray:
        weights = np.column_stack([availability[name] for name in names])
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        stacked = np.stack([projected_views[name] for name in names], axis=1)
        return np.einsum("nm,nmd->nd", weights, stacked)

    matrices = {
        "A1_behavioral_only": projected_views["behavioral"],
        "A2_text_only": projected_views["text"],
        "A3_temporal_only": projected_views["temporal"],
        "A4_text_temporal_fixed_weight": fixed_fusion(["text", "temporal"]),
        "A5_full_static_fixed_weight": fixed_fusion(MODALITIES),
    }
    uniform_weights = np.stack(list(availability.values()), axis=1)
    uniform_weights /= uniform_weights.sum(axis=1, keepdims=True)
    fixed_weights = uniform_weights * np.array([0.6, 0.2, 0.2])
    fixed_weights /= fixed_weights.sum(axis=1, keepdims=True)
    quality = np.linalg.norm(projected, axis=2)
    quality /= np.maximum(quality.max(axis=0, keepdims=True), 1e-8)
    learned_weights = _train_gate(
        projected, uniform_weights, quality, train_mask, SEED, False
    )
    quality_weights = _train_gate(
        projected, uniform_weights, quality, train_mask, SEED, True
    )
    matrices["A6_uniform_routing"] = _route_vectors(projected, uniform_weights)
    matrices["A6_fixed_behavioral_heavy"] = _route_vectors(projected, fixed_weights)
    matrices["A7_standard_learned_gate"] = _route_vectors(projected, learned_weights)
    matrices["A8_quality_aware_gate"] = _route_vectors(projected, quality_weights)
    for name in MODALITIES:
        retained = [modality for modality in MODALITIES if modality != name]
        matrices[f"A9_remove_{name}_from_uniform"] = available_mean(retained)

    results: dict[str, Any] = {}
    assignment_tables = []
    for name, matrix in matrices.items():
        metrics, labels = _evaluate_matrix(matrix, aligned)
        test_labels = labels
        retention = _retention_summary(test_labels, test_ids, targets)
        metrics["retention_summary"] = retention
        metrics["retention_tests"] = _retention_tests(
            test_labels, test_ids, targets
        )
        results[name] = metrics
        assignment_tables.append(
            pl.DataFrame(
                {
                    "customer_unique_id": test_ids,
                    "variant": [name] * len(test_ids),
                    "cluster": test_labels,
                }
            )
        )

    assignments_path = BASELINE_DIR / "phase11_ablation_test_assignments.parquet"
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(assignment_tables).write_parquet(assignments_path)
    retention_rate = float(targets.filter(pl.col("customer_unique_id").is_in(test_ids))[
        "future_is_retained"
    ].mean())
    manifest = {
        "phase": "Phase 11: Ablations, robustness, and evidence assessment",
        "input_artifacts": [
            "data/gold/temporal_customer_cohort.parquet",
            "data/gold/temporal_customer_targets.parquet",
            "data/gold/reports/phase9_fusion_manifest.json",
            "data/gold/reports/phase10_moe_manifest.json",
        ],
        "output_artifacts": [
            str(assignments_path.relative_to(PROJECT_ROOT)),
            "data/gold/reports/phase11_analysis_manifest.json",
            "data/gold/reports/phase11_analysis_report.md",
        ],
        "protocol": {
            "cohort_count": len(aligned),
            "train_count": int(train_mask.sum()),
            "validation_count": int((split == "val").sum()),
            "test_count": int(test_mask.sum()),
            "test_customer_id_sha256": alignment["test_customer_id_sha256"],
            "k": K,
            "clustering": "frozen KMeans(n_clusters=5, n_init=10), fit on train and predict on test",
            "stability_seeds": STABILITY_SEEDS,
            "test_driven_tuning": False,
        },
        "retention_protocol": {
            "definition": "future_is_retained == 1 iff future_orders_count > 0",
            "window": "[2018-03-01T00:00:00, 2018-09-01T00:00:00)",
            "test_retention_rate_pct": round(100 * retention_rate, 8),
            "inference": "Wilson 95% intervals, Pearson chi-square omnibus, Fisher exact pairwise tests with Holm correction",
            "outcomes_used_for_representation_or_selection": False,
        },
        "ablation_definitions": {
            "A1": "behavioral-only",
            "A2": "text-only projected view",
            "A3": "temporal-only projected view",
            "A4": "text + temporal equal fixed-weight fusion",
            "A5": "full three-view equal fixed-weight fusion",
            "A6": "uniform and behavioral-heavy fixed routing",
            "A7": "standard learned gate",
            "A8": "quality-aware learned gate",
            "A9": "remove-one-modality uniform routing controls",
        },
        "results": results,
        "robustness": {
            "missing_text_customers": alignment["cohort_count"]
            - alignment["text_available_count"],
            "missing_text_policy": "explicit availability masks; no rows dropped",
            "assignment_population_identical": True,
        },
        "prior_test_exploration": (
            "Phase 6 through Phase 10 used the blind test for exploratory reporting. "
            "Phase 11 applies the frozen protocol and does not retune or select configurations "
            "from new test outcomes."
        ),
        "interpretation": {
            "adaptive_routing_superiority_supported": False,
            "claim": "Do not claim H-MoE superiority without a positive, stable comparison against the strongest fair static baseline.",
        },
        "reproduction_command": "python -m src.phase11_analysis",
    }
    manifest_path = REPORT_DIR / "phase11_analysis_manifest.json"
    report_path = REPORT_DIR / "phase11_analysis_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 11 Ablation and Evidence Assessment Report",
        "",
        f"Frozen protocol: {len(aligned):,} customers; train/validation/test "
        f"{train_mask.sum():,}/{(split == 'val').sum():,}/{test_mask.sum():,}; k=5.",
        "",
        "| Ablation | Silhouette | DB | CH | Stability ARI | Retention omnibus p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['test_silhouette']:.6f} | "
            f"{result['test_davies_bouldin']:.6f} | {result['test_calinski_harabasz']:.3f} | "
            f"{result['test_stability_mean_pairwise_ari']:.6f} | "
            f"{result['retention_tests']['omnibus_p_value']:.10f} |"
        )
    lines.extend(["", "## Complete retention results", ""])
    for name, result in results.items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| Cluster | Customers | Retained | Rate (%) | Wilson 95% CI (%) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in result["retention_summary"]:
            ci = summary["wilson_95_ci_pct"]
            lines.append(
                f"| {summary['cluster']} | {summary['customer_count']} | "
                f"{summary['retained_count']} | {summary['retention_rate_pct']:.8f} | "
                f"[{ci[0]:.8f}, {ci[1]:.8f}] |"
            )
        lines.extend(
            [
                "",
                f"Pearson chi-square: "
                f"{result['retention_tests']['pearson_chi_square']:.8f}; "
                f"omnibus p: {result['retention_tests']['omnibus_p_value']:.10f}.",
                "",
                "Holm-adjusted Fisher pairwise comparisons:",
                "",
                "| Clusters | Raw p | Holm-adjusted p | Significant (0.05) |",
                "|---|---:|---:|:---:|",
            ]
        )
        for pair in result["retention_tests"]["pairwise_fisher_holm"]:
            lines.append(
                f"| {pair['cluster_left']} vs {pair['cluster_right']} | "
                f"{pair['p_value']:.10f} | {pair['holm_adjusted_p_value']:.10f} | "
                f"{'yes' if pair['significant_at_0.05'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Evidence decision",
            "",
            "Future retention labels were used only for downstream evaluation. "
            "No representation, gate, or clustering configuration was selected from new "
            "test outcomes in this phase. Adaptive-routing superiority is not assumed and "
            "is not supported unless it exceeds the strongest fair static baseline with "
            "stable, uncertainty-aware evidence.",
            "",
            f"Test retention baseline: **{100 * retention_rate:.8f}%**.",
            "Phase 11 is complete; no later phase was started.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(assignments_path)
    print(manifest_path)
    print(report_path)
    return manifest


if __name__ == "__main__":
    run_phase11_analysis()
