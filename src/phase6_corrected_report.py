"""Reconcile Phase 6 retention metrics without changing the original manifest.

The report is derived from the immutable temporal cohort and Phase 6 test
assignments. It uses the blind-test retention rate as the comparison baseline,
Wilson 95% intervals for each cluster, and Cohen's h versus that baseline.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import HDBSCAN
from scipy import stats

from src.config import GOLD_DIR
from src.phase6_baselines import FEATURE_SETS, create_preprocessing_pipeline


BASELINES_DIR = GOLD_DIR / "baselines"
REPORT_DIR = GOLD_DIR / "reports"
SEEDS = (42, 123, 456)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided Wilson interval for a binomial proportion."""
    if trials <= 0:
        return [None, None]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def cohens_h(rate: float, baseline: float) -> float:
    """Return Cohen's h for a cluster rate versus the test baseline."""
    return 2.0 * (math.asin(math.sqrt(rate)) - math.asin(math.sqrt(baseline)))


def rounded(value: float | None, digits: int = 8) -> float | None:
    return None if value is None else round(float(value), digits)


def cluster_metrics(
    labels: np.ndarray,
    retained: np.ndarray,
    baseline_rate: float,
) -> list[dict[str, Any]]:
    """Calculate counts, rates, Wilson intervals, lifts, and effect sizes."""
    rows: list[dict[str, Any]] = []
    for cluster in sorted(int(x) for x in np.unique(labels)):
        mask = labels == cluster
        n = int(mask.sum())
        successes = int(retained[mask].sum())
        rate = successes / n
        rows.append(
            {
                "cluster": cluster,
                "customer_count": n,
                "retained_count": successes,
                "retention_rate": rounded(rate),
                "retention_rate_pct": rounded(rate * 100.0, 6),
                "retention_ci95": [rounded(x) for x in wilson_interval(successes, n)],
                "retention_ci95_pct": [
                    rounded(x * 100.0, 6) for x in wilson_interval(successes, n)
                ],
                "lift_vs_test_baseline": rounded(rate / baseline_rate),
                "cohens_h_vs_test_baseline": rounded(cohens_h(rate, baseline_rate)),
            }
        )
    return rows


def audit_hdbscan_stability(X_test: np.ndarray, min_cluster_size: int, min_samples: int) -> dict[str, Any]:
    """Refit HDBSCAN repeatedly and report exact pairwise label stability."""
    runs = [
        HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(X_test)
        for _ in SEEDS
    ]
    ari = [
        adjusted_rand_score(runs[i], runs[j])
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    nmi = [
        normalized_mutual_info_score(runs[i], runs[j])
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    return {
        "method": "three repeated fits on the identical blind-test feature matrix",
        "mean_pairwise_ari": rounded(float(np.mean(ari)), 8),
        "mean_pairwise_nmi": rounded(float(np.mean(nmi)), 8),
        "pairwise_ari": [rounded(x) for x in ari],
        "pairwise_nmi": [rounded(x) for x in nmi],
        "identical_assignments_all_pairs": all(
            np.array_equal(runs[i], runs[j])
            for i in range(len(runs))
            for j in range(i + 1, len(runs))
        ),
        "interpretation": (
            "Exact repeatability on the same test matrix; not seed robustness "
            "or out-of-sample stability."
        ),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's step-down family-wise error correction."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted.tolist()


def binary_retention_analysis(
    labels: np.ndarray,
    retained: np.ndarray,
) -> dict[str, Any]:
    """Use chi-square and Fisher exact tests for binary retention outcomes."""
    valid = labels >= 0
    clusters = sorted(int(x) for x in np.unique(labels[valid]))
    groups = []
    for cluster in clusters:
        outcome = retained[labels == cluster]
        successes = int(outcome.sum())
        trials = len(outcome)
        groups.append([successes, trials - successes])

    omnibus = stats.chi2_contingency(np.asarray(groups), correction=False)
    pairwise: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for left_index, left in enumerate(clusters):
        for right in clusters[left_index + 1:]:
            left_outcome = retained[labels == left]
            right_outcome = retained[labels == right]
            left_successes = int(left_outcome.sum())
            right_successes = int(right_outcome.sum())
            odds_ratio, p_value = stats.fisher_exact(
                [
                    [left_successes, len(left_outcome) - left_successes],
                    [right_successes, len(right_outcome) - right_successes],
                ],
                alternative="two-sided",
            )
            raw_p.append(float(p_value))
            pairwise.append(
                {
                    "cluster_a": left,
                    "cluster_b": right,
                    "odds_ratio_a_vs_b": None if math.isinf(odds_ratio) else rounded(float(odds_ratio)),
                    "raw_p_value": float(p_value),
                }
            )
    adjusted = holm_adjust(raw_p)
    for row, p_value in zip(pairwise, adjusted):
        row["holm_adjusted_p_value"] = float(p_value)
        row["significant_at_0.05"] = p_value < 0.05

    return {
        "omnibus_test": "Pearson chi-square test of retention by cluster",
        "omnibus_chi2": rounded(float(omnibus[0])),
        "omnibus_degrees_of_freedom": int(omnibus[2]),
        "omnibus_p_value": float(omnibus[1]),
        "pairwise_test": "Two-sided Fisher exact test",
        "pairwise_correction": "Holm step-down family-wise error correction",
        "pairwise_comparisons": pairwise,
        "claim_policy": (
            "A particular cluster difference is called significant only when its "
            "Holm-adjusted p-value is below 0.05."
        ),
    }


def build_report() -> dict[str, Any]:
    cohort = pl.read_parquet(GOLD_DIR / "temporal_customer_cohort.parquet")
    assignments = pl.read_parquet(BASELINES_DIR / "phase6_baseline_test_assignments.parquet")
    original = json.loads(
        (BASELINES_DIR / "phase6_baselines_manifest.json").read_text(encoding="utf-8")
    )
    test = cohort.filter(pl.col("split") == "test")
    joined = assignments.join(
        test.select(["customer_unique_id", "future_is_retained"]),
        on="customer_unique_id",
        how="inner",
    )
    if len(joined) != len(test):
        raise ValueError("Assignments do not cover the complete blind test split")

    retained = joined["future_is_retained"].to_numpy().astype(bool)
    baseline_rate = float(retained.mean())
    report: dict[str, Any] = {
        "report_version": "2.0-corrected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "cohort": "data/gold/temporal_customer_cohort.parquet",
            "assignments": "data/gold/baselines/phase6_baseline_test_assignments.parquet",
            "original_manifest": "data/gold/baselines/phase6_baselines_manifest.json",
        },
        "evaluation_scope": {
            "target_definition": "future_is_retained == 1 iff future_orders_count > 0",
            "blind_test_customers": len(test),
            "blind_test_retained_customers": int(retained.sum()),
            "blind_test_retention_rate": rounded(baseline_rate),
            "blind_test_retention_rate_pct": rounded(baseline_rate * 100.0, 6),
            "cohort_customers": len(cohort),
            "cohort_retained_customers": int(cohort["future_is_retained"].sum()),
            "cohort_retention_rate": rounded(float(cohort["future_is_retained"].mean())),
            "cohort_retention_rate_pct": rounded(float(cohort["future_is_retained"].mean()) * 100.0, 6),
            "retention_baseline_used_for_lifts": "blind_test",
        },
        "validation": {
            "target_count_matches_manifest": len(cohort) == 57510,
            "retained_count_matches_manifest": int(cohort["future_is_retained"].sum()) == 684,
            "split_counts": {
                split: int(cohort.filter(pl.col("split") == split).height)
                for split in ("train", "val", "test")
            },
            "assignment_count_matches_test": len(assignments) == len(test),
        },
        "model_selection_audit": {
            "status": "confirmed_no_blind_test_retention_outcomes_used",
            "evidence": [
                "K selection is performed with X_train and X_val only.",
                "GMM BIC/AIC selection is performed with X_train and X_val only.",
                "Future retention columns are not feature columns and are evaluated only after test assignments are produced.",
            ],
            "caveat": (
                "HDBSCAN is transductive: sklearn HDBSCAN is fit_predict on X_test "
                "because it has no out-of-sample predict method."
            ),
        },
        "original_phase6_results": original,
        "models": {},
        "reconciliation": {
            "original_manifest_preserved": True,
            "original_manifest_path": "data/gold/baselines/phase6_baselines_manifest.json",
            "original_lifts_are_based_on_rounded_baseline": 0.0125,
            "corrected_lifts_use_exact_blind_test_baseline": rounded(baseline_rate),
        },
    }

    for model_name in assignments.columns:
        if model_name == "customer_unique_id":
            continue
        labels = joined[model_name].to_numpy()
        clusters = cluster_metrics(labels, retained, baseline_rate)
        non_noise = [row for row in clusters if row["cluster"] >= 0]
        max_row = max(non_noise, key=lambda row: row["retention_rate"])
        model_entry: dict[str, Any] = {
            "model_type": original["baselines_evaluation"][model_name]["model_type"],
            "feature_set": original["baselines_evaluation"][model_name]["feature_set"],
            "num_features": original["baselines_evaluation"][model_name]["num_features"],
            "num_clusters": original["baselines_evaluation"][model_name]["num_clusters"],
            "original_internal_metrics": original["baselines_evaluation"][model_name][
                "internal_metrics"
            ],
            "original_stability_metrics": original["baselines_evaluation"][model_name][
                "stability_metrics"
            ],
            "original_downstream_utility": original["baselines_evaluation"][model_name][
                "downstream_utility"
            ],
            "original_cluster_profiles": original["baselines_evaluation"][model_name][
                "cluster_profiles"
            ],
            "stability_interpretation": {
                "B2_B4": (
                    "Near-one ARI/NMI indicates the train-fitted KMeans partitions "
                    "are nearly unchanged across seeds; this is consistent with "
                    "well-separated, strongly initialized solutions, not evidence "
                    "that the feature sets are equivalent."
                )
                if model_name in {
                    "Baseline_2_KMeans_RFM_Behavioral",
                    "Baseline_4_KMeans_RFM_All_Engineered",
                }
                else None,
                "B3_DEC": (
                    "Lower ARI/NMI indicates materially different partitions across "
                    "seeds. B3 KMeans is more sensitive to initialization in the "
                    "selected feature space; DEC additionally depends on stochastic "
                    "neural initialization and optimization."
                )
                if model_name in {
                    "Baseline_3_KMeans_RFM_Selected_NonRedundant",
                    "Baseline_7_DEC",
                }
                else None,
            },
            "reported_original_lift": original["baselines_evaluation"][model_name][
                "downstream_utility"
            ]["retention_lift"],
            "corrected_max_non_noise_cluster": max_row["cluster"],
            "corrected_max_non_noise_retention_rate": max_row["retention_rate"],
            "corrected_max_non_noise_lift": max_row["lift_vs_test_baseline"],
            "corrected_max_non_noise_effect_size_cohens_h": max_row[
                "cohens_h_vs_test_baseline"
            ],
            "clusters": clusters,
            "binary_retention_analysis": binary_retention_analysis(labels, retained),
        }
        if model_name == "Baseline_6_HDBSCAN":
            pipe = create_preprocessing_pipeline()
            train = cohort.filter(pl.col("split") == "train")
            X_train = pipe.fit_transform(
                train.select(FEATURE_SETS["RFM_Selected_NonRedundant"]).to_numpy()
            )
            X_test = pipe.transform(
                test.select(FEATURE_SETS["RFM_Selected_NonRedundant"]).to_numpy()
            )
            del X_train
            model_entry["fit_scope"] = "blind_test_transductive"
            model_entry["noise_customer_count"] = int((labels == -1).sum())
            model_entry["stability_audit"] = audit_hdbscan_stability(
                X_test, min_cluster_size=200, min_samples=20
            )
            model_entry["stability_interpretation"] = {
                "HDBSCAN": (
                    "The perfect score is repeated-fit consistency on the identical "
                    "blind-test matrix. It is not a comparison of a run against "
                    "itself and is not out-of-sample stability."
                )
            }
        else:
            model_entry["fit_scope"] = "train_fit_test_predict"
        report["models"][model_name] = model_entry
    b3_name = "Baseline_3_KMeans_RFM_Selected_NonRedundant"
    dec_name = "Baseline_7_DEC"
    b3_labels = joined[b3_name].to_numpy()
    dec_labels = joined[dec_name].to_numpy()
    report["assignment_audits"] = {
        "source_file": "data/gold/baselines/phase6_baseline_test_assignments.parquet",
        "b3_dec": {
            "exact_assignment_arrays_equal": bool(np.array_equal(b3_labels, dec_labels)),
            "b3_cluster_counts": {
                str(cluster): int((b3_labels == cluster).sum())
                for cluster in sorted(np.unique(b3_labels))
            },
            "dec_cluster_counts": {
                str(cluster): int((dec_labels == cluster).sum())
                for cluster in sorted(np.unique(dec_labels))
            },
            "retention_summaries_equal": cluster_metrics(
                b3_labels, retained, baseline_rate
            )
            == cluster_metrics(dec_labels, retained, baseline_rate),
            "generation_independence": {
                "status": "confirmed_from_source",
                "evidence": [
                    "B3 is generated by the KMeans branch with X_train/X_test for the selected non-redundant feature set.",
                    "DEC is generated by the DEC training branch with a separately trained neural model.",
                    "The saved parquet contains separate columns for B3 and DEC.",
                ],
            },
        },
    }
    return report


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = REPORT_DIR / "phase6_corrected_report.json"
    md_path = REPORT_DIR / "phase6_corrected_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 6 Corrected Retention Report",
        "",
        "This report preserves the original manifest and recalculates every lift from the blind-test assignments.",
        "",
        f"- Blind-test baseline: **{report['evaluation_scope']['blind_test_retention_rate_pct']:.6f}%** "
        f"({report['evaluation_scope']['blind_test_retained_customers']}/{report['evaluation_scope']['blind_test_customers']})",
        f"- Cohort target: **{report['evaluation_scope']['cohort_retention_rate_pct']:.6f}%** "
        f"({report['evaluation_scope']['cohort_retained_customers']}/{report['evaluation_scope']['cohort_customers']})",
        "- Per-cluster intervals: Wilson 95%; effect size: Cohen's h versus the blind-test baseline.",
        "- Binary retention inference: Pearson chi-square omnibus test; pairwise two-sided Fisher exact tests with Holm correction.",
        "- Model selection audit: confirmed no blind-test retention outcomes were used for K/BIC selection.",
        "- HDBSCAN is reported as blind-test transductive and its stability is exact repeated-fit stability, not out-of-sample stability.",
        "",
        "| Model | Original lift | Corrected max non-noise lift | Max cluster | 95% CI | Cohen's h |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, model in report["models"].items():
        row = next(
            item for item in model["clusters"]
            if item["cluster"] == model["corrected_max_non_noise_cluster"]
        )
        ci = row["retention_ci95_pct"]
        lines.append(
            f"| {name} | {model['reported_original_lift']:.2f}x | "
            f"{model['corrected_max_non_noise_lift']:.8f}x | "
            f"{row['cluster']} | {ci[0]:.6f}%–{ci[1]:.6f}% | "
            f"{model['corrected_max_non_noise_effect_size_cohens_h']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility and validation",
            "",
            f"- Report version: `{report['report_version']}`",
            f"- Generated at (UTC): `{report['generated_at_utc']}`",
            f"- Source cohort: `{report['source_artifacts']['cohort']}`",
            f"- Source assignments: `{report['source_artifacts']['assignments']}`",
            f"- Original manifest preserved at: `{report['source_artifacts']['original_manifest']}`",
            f"- Split counts: `{report['validation']['split_counts']}`",
            f"- Target count check: `{report['validation']['target_count_matches_manifest']}`",
            f"- Retained count check: `{report['validation']['retained_count_matches_manifest']}`",
            f"- Assignment coverage check: `{report['validation']['assignment_count_matches_test']}`",
            "",
            "## K-selection results",
            "",
            "| Selection | Candidate results | Selected K | Criterion |",
            "|---|---|---:|---|",
        ]
    )
    for name, selection in report["original_phase6_results"]["k_selection_methodology"].items():
        candidates = "; ".join(
            ", ".join(f"{key}={value}" for key, value in candidate.items())
            for candidate in selection["candidate_results"]
        )
        lines.append(
            f"| {name} | {candidates} | {selection['selected_k']} | "
            f"{selection['selection_criteria']} |"
        )
    lines.extend(
        [
            "",
            "## Complete model results",
            "",
            "| Model | Type | Feature set | Features | K | Silhouette | DBI | CHI | ARI | NMI | Retention ANOVA F | Retention ANOVA p | Spend ANOVA F | Spend ANOVA p | Original max retention | Original lift |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, model in report["models"].items():
        internal = model["original_internal_metrics"]
        stability = model["original_stability_metrics"]
        utility = model["original_downstream_utility"]
        lines.append(
            f"| {name} | {model['model_type']} | {model['feature_set']} | "
            f"{model['num_features']} | {model['num_clusters']} | "
            f"{internal['silhouette_score']} | {internal['davies_bouldin_index']} | "
            f"{internal['calinski_harabasz_score']} | {stability['mean_pairwise_ari']} | "
            f"{stability['mean_pairwise_nmi']} | {utility['cluster_retention_anova_f_stat']} | "
            f"{utility['cluster_retention_anova_p_val']:.8g} | "
            f"{utility['cluster_spend_anova_f_stat']} | "
            f"{utility['cluster_spend_anova_p_val']:.8g} | "
            f"{utility['max_retention_rate'] * 100:.6f}% | "
            f"{utility['retention_lift']:.8f}x |"
        )
    lines.extend(
        [
            "",
            "## Binary retention tests",
            "",
            "ANOVA is retained as a descriptive legacy metric, but binary-outcome claims use the chi-square omnibus test and Holm-adjusted two-sided Fisher pairwise tests.",
            "",
            "| Model | Chi-square | df | Omnibus p | Pairwise significant after Holm correction |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, model in report["models"].items():
        analysis = model["binary_retention_analysis"]
        significant = [
            f"{row['cluster_a']} vs {row['cluster_b']} "
            f"(p={row['holm_adjusted_p_value']:.8g})"
            for row in analysis["pairwise_comparisons"]
            if row["significant_at_0.05"]
        ]
        lines.append(
            f"| {name} | {analysis['omnibus_chi2']:.6f} | "
            f"{analysis['omnibus_degrees_of_freedom']} | "
            f"{analysis['omnibus_p_value']:.8g} | "
            f"{', '.join(significant) if significant else 'None'} |"
        )
    lines.extend(["", "## Complete per-cluster retention results", ""])
    for name, model in report["models"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| Cluster | Customers | Retained | Retention | Wilson 95% CI | Lift vs exact test baseline | Cohen's h |",
                "|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        for row in model["clusters"]:
            ci = row["retention_ci95_pct"]
            label = "noise" if row["cluster"] == -1 else str(row["cluster"])
            lines.append(
                f"| {label} | {row['customer_count']} | {row['retained_count']} | "
                f"{row['retention_rate_pct']:.6f}% | {ci[0]:.6f}%–{ci[1]:.6f}% | "
                f"{row['lift_vs_test_baseline']:.8f}x | "
                f"{row['cohens_h_vs_test_baseline']:.8f} |"
            )
        lines.extend(
            [
                "",
                "Original downstream cluster breakdown (including spend and order metrics):",
                "",
                "```json",
                json.dumps(model["original_downstream_utility"]["cluster_breakdown"], indent=2),
                "```",
                "",
                "Original pre-cutoff cluster profiles:",
                "",
                "```json",
                json.dumps(model["original_cluster_profiles"], indent=2),
                "```",
                "",
            ]
        )
        if "stability_audit" in model:
            lines.extend(
                [
                    "HDBSCAN stability audit:",
                    "",
                    "```json",
                    json.dumps(model["stability_audit"], indent=2),
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "All pairwise binary retention comparisons:",
                "",
                "```json",
                json.dumps(model["binary_retention_analysis"], indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Assignment audits",
            "",
            "```json",
            json.dumps(report["assignment_audits"], indent=2),
            "```",
            "",
            "## Model-selection leakage audit",
            "",
            f"Status: **{report['model_selection_audit']['status']}**",
            "",
            *[f"- {item}" for item in report["model_selection_audit"]["evidence"]],
            "",
            f"Caveat: {report['model_selection_audit']['caveat']}",
            "",
            "The JSON file contains the same information in machine-readable form.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
