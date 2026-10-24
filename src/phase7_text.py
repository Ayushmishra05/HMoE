"""Phase 7: leakage-safe customer review-text representation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from src.config import GOLD_DIR, DEFAULT_TEMPORAL_OBSERVATION_CUTOFF


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
REPORT_DIR = GOLD_DIR / "reports"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_VERSION = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64


def normalize_review_text(title: str | None, message: str | None) -> str:
    """Normalize whitespace/case while preserving punctuation and negation."""
    parts = [value.strip() for value in (title, message) if value and value.strip()]
    return re.sub(r"\s+", " ", " ".join(parts)).lower()


def prepare_eligible_reviews(
    reviews: pl.DataFrame,
    cohort_ids: pl.Series,
    cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Filter, deduplicate, and normalize reviews strictly before the cutoff."""
    required = {
        "review_id",
        "customer_unique_id",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
    }
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"Review input is missing required columns: {sorted(missing)}")

    cohort = pl.DataFrame({"customer_unique_id": cohort_ids.unique()})
    scoped = (
        reviews
        .join(cohort, on="customer_unique_id", how="inner")
        .filter(pl.col("review_creation_date") < pl.lit(cutoff))
        .sort(["review_id", "review_creation_date"])
        .unique(subset=["review_id"], keep="first")
        .with_columns(
            pl.struct(["review_comment_title", "review_comment_message"])
            .map_elements(
                lambda row: normalize_review_text(
                    row["review_comment_title"], row["review_comment_message"]
                ),
                return_dtype=pl.String,
            )
            .alias("normalized_text")
        )
    )
    stats = {
        "eligible_reviews": len(scoped),
        "eligible_nonempty_reviews": int((scoped["normalized_text"].str.len_chars() > 0).sum()),
        "duplicate_review_ids_removed": len(
            reviews.join(cohort, on="customer_unique_id", how="inner")
            .filter(pl.col("review_creation_date") < pl.lit(cutoff))
        ) - len(scoped),
        "empty_text_reviews": int((scoped["normalized_text"].str.len_chars() == 0).sum()),
    }
    return scoped, stats


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_provenance(model: SentenceTransformer, model_name: str) -> dict[str, Any]:
    """Capture the immutable model revision and local Hugging Face cache location."""
    model_card = getattr(model, "model_card_data", None)
    revision = getattr(model_card, "base_model_revision", None)
    cache_root = os.environ.get("HF_HUB_CACHE")
    if not cache_root:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = str(HF_HUB_CACHE)
        except ImportError:
            cache_root = None
    snapshot_path = None
    if cache_root and revision:
        candidate = Path(cache_root) / f"models--{model_name.replace('/', '--')}" / "snapshots" / revision
        if candidate.exists():
            snapshot_path = str(candidate)
    return {
        "model_id": model_name,
        "revision": revision,
        "cache_root": cache_root,
        "snapshot_path": snapshot_path,
        "cache_present": snapshot_path is not None,
    }


def build_text_view(
    cohort: pl.DataFrame,
    reviews: pl.DataFrame,
    cutoff: datetime = DEFAULT_TEMPORAL_OBSERVATION_CUTOFF,
    model_name: str = MODEL_NAME,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Build one row per cohort customer and a review-level embedding cache."""
    if "customer_unique_id" not in cohort.columns:
        raise ValueError("Cohort must contain customer_unique_id")
    if cohort["customer_unique_id"].n_unique() != len(cohort):
        raise ValueError("Cohort contains duplicate customer_unique_id values")

    eligible, review_stats = prepare_eligible_reviews(
        reviews, cohort["customer_unique_id"], cutoff
    )
    nonempty = eligible.filter(pl.col("normalized_text").str.len_chars() > 0)
    model = SentenceTransformer(model_name)
    dimension = int(model.get_embedding_dimension())

    if len(nonempty):
        texts = nonempty["normalized_text"].to_list()
        vectors = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Text embeddings contain non-finite values")
        embedded = nonempty.with_columns(
            pl.Series("embedding", [row.tolist() for row in vectors]),
            pl.col("normalized_text").map_elements(_text_hash, return_dtype=pl.String).alias(
                "text_sha256"
            ),
        )
    else:
        embedded = nonempty.with_columns(
            pl.Series("embedding", [], dtype=pl.List(pl.Float32)),
            pl.Series("text_sha256", [], dtype=pl.String),
        )

    customer_text = (
        embedded
        .group_by("customer_unique_id")
        .agg(
            [
                pl.len().alias("text_review_count"),
                pl.col("review_creation_date").min().alias("text_first_review_date"),
                pl.col("review_creation_date").max().alias("text_last_review_date"),
                pl.col("normalized_text").str.join(" ").alias("normalized_text"),
            ]
        )
    )
    embedding_sums: dict[str, list[np.ndarray]] = {}
    for customer_id, embedding in embedded.select(
        ["customer_unique_id", "embedding"]
    ).iter_rows():
        embedding_sums.setdefault(customer_id, []).append(
            np.asarray(embedding, dtype=np.float32)
        )
    embedding_means = pl.DataFrame(
        {
            "customer_unique_id": list(embedding_sums),
            "text_embedding": [
                np.mean(vectors, axis=0).tolist()
                for vectors in embedding_sums.values()
            ],
        },
        schema={
            "customer_unique_id": pl.String,
            "text_embedding": pl.List(pl.Float32),
        },
    )
    customer_text = customer_text.join(
        embedding_means, on="customer_unique_id", how="inner"
    )
    text_view = (
        cohort.select("customer_unique_id")
        .join(customer_text, on="customer_unique_id", how="left")
        .with_columns(
            [
                pl.col("text_review_count").fill_null(0).cast(pl.Int32),
                pl.col("normalized_text").fill_null(""),
                pl.col("text_review_count").fill_null(0).gt(0).alias("text_available"),
            ]
        )
    )
    cache = embedded.select(
        [
            "review_id",
            "customer_unique_id",
            "review_creation_date",
            "normalized_text",
            "text_sha256",
            "embedding",
        ]
    )
    review_vectors = [
        np.asarray(row, dtype=np.float32) for row in cache["embedding"].to_list()
    ]
    customer_vectors = [
        np.asarray(row, dtype=np.float32)
        for row in text_view.filter(pl.col("text_available"))["text_embedding"].to_list()
    ]
    review_finite = bool(all(np.isfinite(row).all() for row in review_vectors))
    customer_finite = bool(all(np.isfinite(row).all() for row in customer_vectors))
    cutoff_violations = (
        int(
            cache.select(
                (pl.col("review_creation_date") >= pl.lit(cutoff)).sum()
            ).item()
        )
        if len(cache)
        else 0
    )
    aggregation_mismatches = 0
    expected_by_customer: dict[str, list[np.ndarray]] = {}
    for customer_id, vector in embedded.select(
        ["customer_unique_id", "embedding"]
    ).iter_rows():
        expected_by_customer.setdefault(customer_id, []).append(
            np.asarray(vector, dtype=np.float32)
        )
    actual_by_customer = {
        customer_id: np.asarray(vector, dtype=np.float32)
        for customer_id, vector in text_view.filter(pl.col("text_available")).select(
            ["customer_unique_id", "text_embedding"]
        ).iter_rows()
    }
    for customer_id, vectors in expected_by_customer.items():
        expected = np.mean(vectors, axis=0)
        actual = actual_by_customer.get(customer_id)
        if actual is None or not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
            aggregation_mismatches += 1
    if cutoff_violations or aggregation_mismatches or not review_finite or not customer_finite:
        raise ValueError(
            "Phase 7 embedding audit failed: "
            f"cutoff_violations={cutoff_violations}, "
            f"aggregation_mismatches={aggregation_mismatches}, "
            f"review_finite={review_finite}, customer_finite={customer_finite}"
        )
    metadata = {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "model_provenance": _model_provenance(model, model_name),
        "sentence_transformers_version": getattr(
            __import__("sentence_transformers"), "__version__", None
        ),
        "embedding_dimension": dimension,
        "pooling": "mean over eligible non-empty review embeddings per customer",
        "max_sequence_length": int(model.max_seq_length),
        "batch_size": EMBEDDING_BATCH_SIZE,
        "normalize_embeddings": False,
        "cutoff": cutoff.isoformat(),
        "review_stats": review_stats,
        "customer_count": len(text_view),
        "customers_with_text": int(text_view["text_available"].sum()),
        "customers_without_text": int((~text_view["text_available"]).sum()),
        "embedding_audit": {
            "review_embedding_count": len(review_vectors),
            "review_nonfinite_values": int(
                sum(np.size(row) - np.isfinite(row).sum() for row in review_vectors)
            ),
            "customer_embedding_count": len(customer_vectors),
            "customer_nonfinite_values": int(
                sum(np.size(row) - np.isfinite(row).sum() for row in customer_vectors)
            ),
            "review_cutoff_violations": cutoff_violations,
            "customer_rows": len(text_view),
            "customer_unique_ids": text_view["customer_unique_id"].n_unique(),
            "aggregation_customers_checked": len(expected_by_customer),
            "aggregation_mismatches": aggregation_mismatches,
            "all_checks_passed": True,
        },
    }
    return text_view, cache, metadata


def run_phase7_text() -> dict[str, Any]:
    """Generate Phase 7 customer text view, cache, manifest, and report."""
    cohort = pl.read_parquet(GOLD_DIR / "temporal_customer_cohort.parquet")
    reviews = pl.read_parquet(GOLD_DIR / "gold_customer_reviews.parquet")
    text_view, cache, metadata = build_text_view(cohort, reviews)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    view_path = PROCESSED_DIR / "customer_text_view.parquet"
    cache_path = EMBEDDINGS_DIR / "phase7_review_embeddings.parquet"
    manifest_path = REPORT_DIR / "phase7_text_manifest.json"
    report_path = REPORT_DIR / "phase7_text_report.md"
    text_view.write_parquet(view_path)
    cache.write_parquet(cache_path)
    manifest = {
        "phase": "Phase 7: Text representation",
        "input_artifacts": [
            "data/gold/temporal_customer_cohort.parquet",
            "data/gold/gold_customer_reviews.parquet",
        ],
        "output_artifacts": [str(view_path.relative_to(PROJECT_ROOT)), str(cache_path.relative_to(PROJECT_ROOT))],
        "observation_cutoff": DEFAULT_TEMPORAL_OBSERVATION_CUTOFF.isoformat(),
        "split_counts": cohort.group_by("split").len().sort("split").to_dicts(),
        "preprocessing_fit_scope": "frozen pretrained model; no fitting on customer outcomes",
        "reproduction_command": "python -m src.phase7_text",
        "metadata": metadata,
        "limitations": [
            "Review text is available only for customers with eligible non-empty pre-cutoff review text.",
            "Embeddings are descriptive and are not selected using future retention labels.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        "# Phase 7 Text Representation Report\n\n"
        "A frozen `all-MiniLM-L6-v2` sentence encoder was applied only to non-empty reviews "
        "created strictly before the Phase 5 observation cutoff. Customer embeddings are "
        "the mean of eligible review embeddings; missing text remains explicitly unavailable.\n\n"
        f"- Customers: **{len(text_view):,}**\n"
        f"- Customers with text: **{metadata['customers_with_text']:,}**\n"
        f"- Eligible reviews: **{metadata['review_stats']['eligible_reviews']:,}**\n"
        f"- Non-empty eligible reviews: **{metadata['review_stats']['eligible_nonempty_reviews']:,}**\n"
        f"- Duplicate review IDs removed: **{metadata['review_stats']['duplicate_review_ids_removed']:,}**\n"
        f"- Empty eligible reviews: **{metadata['review_stats']['empty_text_reviews']:,}**\n"
        f"- Embedding model/dimension: `{MODEL_NAME}` / `{metadata['embedding_dimension']}`\n"
        f"- Model revision: `{metadata['model_provenance']['revision']}`\n"
        f"- Embedding configuration: batch size `{metadata['batch_size']}`, "
        f"max sequence length `{metadata['max_sequence_length']}`, "
        f"mean pooling, normalized embeddings `{metadata['normalize_embeddings']}`\n"
        f"- Cutoff rule: `review_creation_date < {metadata['cutoff']}`\n\n"
        "## Embedding audit\n\n"
        f"- Review-level finite embeddings: **{metadata['embedding_audit']['review_embedding_count']:,}**; "
        f"non-finite values: **{metadata['embedding_audit']['review_nonfinite_values']}**\n"
        f"- Customer-level finite embeddings: **{metadata['embedding_audit']['customer_embedding_count']:,}**; "
        f"non-finite values: **{metadata['embedding_audit']['customer_nonfinite_values']}**\n"
        f"- Reviews at or after cutoff: **{metadata['embedding_audit']['review_cutoff_violations']}**\n"
        f"- Independent mean-pooling mismatches: **{metadata['embedding_audit']['aggregation_mismatches']}**\n"
        f"- Customer row uniqueness: **{metadata['embedding_audit']['customer_rows'] == metadata['embedding_audit']['customer_unique_ids']}**\n\n"
        "No Phase 8, 9, 10, or 11 work was started.\n",
        encoding="utf-8",
    )
    print(view_path)
    print(cache_path)
    print(manifest_path)
    print(report_path)
    return manifest


if __name__ == "__main__":
    run_phase7_text()
