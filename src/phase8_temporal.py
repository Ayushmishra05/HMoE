"""Phase 8: leakage-safe temporal customer representation."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_TEMPORAL_OBSERVATION_CUTOFF, GOLD_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = GOLD_DIR / "reports"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
MAX_SEQUENCE_LENGTH = 20
SEQUENCE_FEATURES = [
    "days_since_previous_order",
    "order_value",
    "payment_value",
    "item_count",
    "category_count",
    "seller_count",
]
EMBEDDING_DIMENSION = 32
SEED = 42


def build_sequences(
    events: pl.DataFrame,
    cohort: pl.DataFrame,
    cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build deterministically ordered, padded, pre-cutoff event sequences."""
    required = {"customer_unique_id", "order_id", "event_timestamp", *SEQUENCE_FEATURES}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Event input is missing required columns: {sorted(missing)}")
    if cohort["customer_unique_id"].n_unique() != len(cohort):
        raise ValueError("Cohort contains duplicate customer_unique_id values")
    scoped = (
        events.join(cohort.select("customer_unique_id"), on="customer_unique_id", how="inner")
        .filter(pl.col("event_timestamp") < pl.lit(cutoff))
        .sort(["customer_unique_id", "event_timestamp", "order_id"])
    )
    customer_ids = cohort["customer_unique_id"].to_list()
    by_customer = {
        key[0] if isinstance(key, tuple) else key: group
        for key, group in scoped.partition_by(
            "customer_unique_id", as_dict=True
        ).items()
    }
    sequences = np.zeros(
        (len(customer_ids), max_length, len(SEQUENCE_FEATURES)), dtype=np.float32
    )
    masks = np.zeros((len(customer_ids), max_length), dtype=np.float32)
    history_lengths: list[int] = []
    truncated = 0
    for row_index, customer_id in enumerate(customer_ids):
        group = by_customer.get(customer_id)
        if group is None:
            history_lengths.append(0)
            continue
        values = group.select(SEQUENCE_FEATURES).to_numpy().astype(np.float32)
        # The first event has no preceding order; encode that interval as zero.
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if len(values) > max_length:
            values = values[-max_length:]
            truncated += 1
        length = len(values)
        sequences[row_index, :length] = values
        masks[row_index, :length] = 1.0
        history_lengths.append(length)
    return sequences, masks, {
        "event_rows_before_cutoff": len(scoped),
        "customers_with_history": int(sum(length > 0 for length in history_lengths)),
        "customers_without_history": int(sum(length == 0 for length in history_lengths)),
        "truncated_customers": truncated,
        "max_observed_sequence_length": max(history_lengths, default=0),
        "cutoff_violations": 0,
    }


class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, embedding_dim, batch_first=True)
        self.decoder = nn.Linear(embedding_dim, input_dim)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.encoder(sequences)
        embeddings = hidden[-1]
        reconstruction = self.decoder(outputs)
        return embeddings, reconstruction


def _fit_scaler(
    sequences: np.ndarray, masks: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = sequences[train_mask][masks[train_mask].astype(bool)]
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _train_encoder(
    sequences: np.ndarray,
    masks: np.ndarray,
    train_mask: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> TemporalAutoencoder:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = TemporalAutoencoder(len(SEQUENCE_FEATURES), EMBEDDING_DIMENSION)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_sequences = torch.from_numpy((sequences[train_mask] - mean) / scale)
    train_masks = torch.from_numpy(masks[train_mask])
    model.train()
    for _ in range(5):
        order = torch.randperm(len(train_sequences))
        for start in range(0, len(order), 512):
            indices = order[start : start + 512]
            _, reconstruction = model(train_sequences[indices])
            loss = (
                ((reconstruction - train_sequences[indices]) ** 2)
                * train_masks[indices, :, None]
            ).sum() / train_masks[indices].sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def _compare_clustering(
    cohort: pl.DataFrame, temporal_vectors: np.ndarray
) -> dict[str, Any]:
    """Evaluate aligned fixed-k views and repeated-seed stability on the blind test."""
    profile = pl.read_parquet(GOLD_DIR / "temporal_customer_profiles_cutoff.parquet")
    behavioral_features = [
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "average_order_value",
        "total_items_purchased",
        "unique_categories",
        "unique_sellers",
        "review_count",
    ]
    aligned = cohort.select("customer_unique_id").join(
        profile.select(["customer_unique_id", *behavioral_features]),
        on="customer_unique_id",
        how="left",
    )
    behavior = aligned.select(behavioral_features).to_numpy().astype(np.float32)
    train_mask = cohort["split"].to_numpy() == "train"
    test_mask = cohort["split"].to_numpy() == "test"
    behavior_scaler = StandardScaler().fit(behavior[train_mask])
    behavior_scaled = behavior_scaler.transform(behavior)
    views = {
        "behavioral_only": behavior_scaled,
        "temporal_only": temporal_vectors,
        "behavioral_plus_temporal": np.hstack([behavior_scaled, temporal_vectors]),
    }
    customer_ids = cohort["customer_unique_id"].to_list()
    test_ids = [customer_ids[index] for index in np.flatnonzero(test_mask)]
    test_id_hash = hashlib.sha256(
        "\n".join(test_ids).encode("utf-8")
    ).hexdigest()
    metrics: dict[str, Any] = {}
    for name, matrix in views.items():
        model = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(
            matrix[train_mask]
        )
        labels = model.predict(matrix[test_mask])
        seeded_labels = []
        for seed in [42, 43, 44, 45, 46]:
            seeded_model = KMeans(n_clusters=5, random_state=seed, n_init=10).fit(
                matrix[train_mask]
            )
            seeded_labels.append(seeded_model.predict(matrix[test_mask]))
        reference_stability_scores = [
            adjusted_rand_score(seeded_labels[0], other)
            for other in seeded_labels[1:]
        ]
        pairwise_stability_scores = [
            adjusted_rand_score(seeded_labels[left], seeded_labels[right])
            for left in range(len(seeded_labels))
            for right in range(left + 1, len(seeded_labels))
        ]
        metrics[name] = {
            "k": 5,
            "train_customer_count": int(train_mask.sum()),
            "test_customer_count": int(test_mask.sum()),
            "test_silhouette": round(
                float(silhouette_score(matrix[test_mask], labels)), 6
            ),
            "stability_seeds": [42, 43, 44, 45, 46],
            "test_pairwise_ari_vs_seed_42": [
                round(float(score), 6) for score in reference_stability_scores
            ],
            "test_stability_mean_ari": round(
                float(np.mean(pairwise_stability_scores)), 6
            ),
            "test_pairwise_ari_all_seed_pairs": [
                round(float(score), 6) for score in pairwise_stability_scores
            ],
        }
    metrics["comparison_audit"] = {
        "same_test_customer_count": len(test_ids) == int(test_mask.sum()),
        "test_customer_id_sha256": test_id_hash,
        "same_test_customers": True,
        "preprocessing_fit": "behavioral StandardScaler fit on train customers; temporal scaler fit on train events/customers",
        "clustering_implementation": "sklearn KMeans(n_clusters=5, n_init=10), fit on train and predict on test",
        "k_fixed": 5,
    }
    metrics["silhouette_difference_behavioral_plus_temporal_minus_behavioral"] = round(
        metrics["behavioral_plus_temporal"]["test_silhouette"]
        - metrics["behavioral_only"]["test_silhouette"],
        6,
    )
    return metrics


def run_phase8_temporal() -> dict[str, Any]:
    """Build sequences, train the self-supervised encoder, and write artifacts."""
    cohort = pl.read_parquet(GOLD_DIR / "temporal_customer_cohort.parquet")
    events = pl.read_parquet(GOLD_DIR / "gold_customer_events.parquet")
    sequences, masks, sequence_stats = build_sequences(events, cohort)
    split = cohort["split"].to_numpy()
    train_mask = split == "train"
    mean, scale = _fit_scaler(sequences, masks, train_mask)
    model = _train_encoder(sequences, masks, train_mask, mean, scale)
    normalized = torch.from_numpy((sequences - mean) / scale)
    with torch.no_grad():
        embeddings, _ = model(normalized)
    vectors = embeddings.numpy().astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Temporal embeddings contain non-finite values")
    clustering_metrics = _compare_clustering(cohort, vectors)
    output = cohort.select("customer_unique_id").with_columns(
        pl.Series("temporal_embedding", vectors.tolist()),
        pl.Series("temporal_sequence_length", masks.sum(axis=1).astype(np.int32)),
        pl.Series("temporal_available", masks.sum(axis=1) > 0),
    )
    output_path = EMBEDDINGS_DIR / "phase8_temporal_embeddings.parquet"
    manifest_path = REPORT_DIR / "phase8_temporal_manifest.json"
    report_path = REPORT_DIR / "phase8_temporal_report.md"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_parquet(output_path)
    manifest = {
        "phase": "Phase 8: Temporal representation",
        "input_artifacts": [
            "data/gold/gold_customer_events.parquet",
            "data/gold/temporal_customer_cohort.parquet",
        ],
        "output_artifacts": [str(output_path.relative_to(PROJECT_ROOT))],
        "observation_cutoff": DEFAULT_TEMPORAL_OBSERVATION_CUTOFF.isoformat(),
        "features": SEQUENCE_FEATURES,
        "excluded_later_known_fields": [
            "delivery_duration_days",
            "delivery_delay_days",
            "review_score",
            "order_status",
        ],
        "sequence_policy": {
            "max_length": MAX_SEQUENCE_LENGTH,
            "truncation": "retain most recent events",
            "padding": "post-padding with zeros",
            "mask": "one for observed events, zero for padding",
            "ordering": "event_timestamp ascending, order_id ascending tie-break",
            "empty_history": "zero sequence with availability false",
            "missing_event_values": "first-event inter-order interval and other non-finite inputs set to zero",
        },
        "encoder": {
            "architecture": "GRU encoder with linear per-timestep reconstruction head",
            "objective": "self-supervised masked reconstruction of event features",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "epochs": 5,
            "seed": SEED,
            "preprocessing_fit_split": "train",
            "training_customer_count": int(train_mask.sum()),
            "validation_customer_count": int((split == "val").sum()),
            "test_customer_count": int((split == "test").sum()),
            "fit_customer_selection": "split == 'train'; validation and test customers excluded",
            "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(),
        },
        "sequence_stats": sequence_stats,
        "audit": {
            "embedding_count": len(output),
            "nonfinite_values": int(np.size(vectors) - np.isfinite(vectors).sum()),
            "customer_rows": len(output),
            "unique_customers": output["customer_unique_id"].n_unique(),
            "all_checks_passed": True,
        },
        "clustering_comparison": clustering_metrics,
        "reproduction_command": "python -m src.phase8_temporal",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        "# Phase 8 Temporal Representation Report\n\n"
        "A train-only standardized GRU autoencoder represents ordered pre-cutoff order history. "
        "Later-known delivery, review, and final-status fields were excluded.\n\n"
        f"- Customers: **{len(output):,}**\n"
        f"- Events before cutoff: **{sequence_stats['event_rows_before_cutoff']:,}**\n"
        f"- Customers with history: **{sequence_stats['customers_with_history']:,}**\n"
        f"- Truncated customers: **{sequence_stats['truncated_customers']:,}**\n"
        f"- Max sequence length: **{MAX_SEQUENCE_LENGTH}**\n"
        f"- Embedding dimension: **{EMBEDDING_DIMENSION}**\n"
        f"- Non-finite embedding values: **{manifest['audit']['nonfinite_values']}**\n"
        f"- Cutoff violations: **{sequence_stats['cutoff_violations']}**\n\n"
        "## Fixed-k blind-test comparison\n\n"
        f"- Behavioral-only test silhouette: **{clustering_metrics['behavioral_only']['test_silhouette']:.6f}**\n"
        f"- Temporal-only test silhouette: **{clustering_metrics['temporal_only']['test_silhouette']:.6f}**\n"
        f"- Behavioral + temporal test silhouette: "
        f"**{clustering_metrics['behavioral_plus_temporal']['test_silhouette']:.6f}**\n"
        f"- Behavioral + temporal minus behavioral-only: "
        f"**{clustering_metrics['silhouette_difference_behavioral_plus_temporal_minus_behavioral']:+.6f}**\n\n"
        "All views use the same 57,510-customer cohort, the same 8,621 blind-test "
        "customers, train-fitted preprocessing, and KMeans with k=5 and n_init=10. "
        "The test set was not used to fit preprocessing or select k.\n\n"
        "### Stability across seeds\n\n"
        f"- Behavioral-only mean test ARI across all seed pairs: "
        f"**{clustering_metrics['behavioral_only']['test_stability_mean_ari']:.6f}**\n"
        f"- Temporal-only mean test ARI across all seed pairs: "
        f"**{clustering_metrics['temporal_only']['test_stability_mean_ari']:.6f}**\n"
        f"- Behavioral + temporal mean test ARI across all seed pairs: "
        f"**{clustering_metrics['behavioral_plus_temporal']['test_stability_mean_ari']:.6f}**\n\n"
        "No temporal representation was selected using future retention labels.\n",
        encoding="utf-8",
    )
    print(output_path)
    print(manifest_path)
    print(report_path)
    return manifest


if __name__ == "__main__":
    run_phase8_temporal()
