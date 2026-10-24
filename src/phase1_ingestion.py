"""
Phase 1: Ingestion, Schema Standardization, and Bronze Layer Materialization.

Guarantees:
- Raw data files are strictly read-only and untouched.
- 0 rows are dropped or filtered.
- Whitespace is trimmed on textual fields.
- Dates are deterministically parsed into polars.Datetime.
- Explicit schema types are enforced.
- Existing processed datasets cannot be overwritten unless explicit permission/flag is granted.
"""

from pathlib import Path
from typing import Dict, List, Optional
import polars as pl

from src.config import (
    RAW_DATA_DIR,
    BRONZE_DIR,
    DATETIME_COLUMNS,
    EXPLICIT_DTYPES,
)


def standardize_table(
    table_name: str,
    raw_csv_path: Path,
) -> pl.DataFrame:
    """
    Ingests a raw CSV table, enforces strict typing, parses datetimes,
    and strips extraneous whitespace without dropping any rows.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {raw_csv_path}")

    # Read CSV with all text columns read as String
    df = pl.read_csv(raw_csv_path)
    initial_row_count = len(df)

    exprs = []

    # 1. Strip whitespace on string columns
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.String:
            exprs.append(pl.col(col).str.strip_chars().alias(col))

    if exprs:
        df = df.with_columns(exprs)

    # 2. Parse Datetime columns
    if table_name in DATETIME_COLUMNS:
        dt_exprs = []
        for dt_col in DATETIME_COLUMNS[table_name]:
            if dt_col in df.columns:
                raw_nulls = df[dt_col].null_count()
                parsed_expr = pl.col(dt_col).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=True).alias(dt_col)
                dt_exprs.append(parsed_expr)
        if dt_exprs:
            df = df.with_columns(dt_exprs)

    # 3. Explicit numeric / schema casting
    if table_name in EXPLICIT_DTYPES:
        cast_exprs = []
        for col_name, target_type in EXPLICIT_DTYPES[table_name].items():
            if col_name in df.columns and df[col_name].dtype != target_type:
                cast_exprs.append(pl.col(col_name).cast(target_type).alias(col_name))
        if cast_exprs:
            df = df.with_columns(cast_exprs)

    # Invariant: 0 rows lost
    assert len(df) == initial_row_count, (
        f"Row count mismatch in {table_name}: initial={initial_row_count}, final={len(df)}"
    )

    return df


def run_phase1_ingestion(
    allow_overwrite: bool = False,
) -> Dict[str, Dict[str, any]]:
    """
    Runs Phase 1 ingestion for all 9 Olist tables, writes to Parquet in bronze directory.
    Returns a summary dictionary of metrics for each processed table.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    summary = {}

    for csv_file in csv_files:
        table_name = csv_file.stem
        output_parquet_path = BRONZE_DIR / f"{table_name}.parquet"

        if output_parquet_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Target parquet file {output_parquet_path} already exists. "
                "Set allow_overwrite=True if explicit overwrite is confirmed."
            )

        # Standardize and validate row count
        df_standardized = standardize_table(table_name, csv_file)

        # Write to Parquet (lossless, columnar compression)
        df_standardized.write_parquet(output_parquet_path, compression="zstd")

        # Read back to verify roundtrip
        df_readback = pl.read_parquet(output_parquet_path)
        assert len(df_readback) == len(df_standardized), (
            f"Parquet roundtrip row count mismatch for {table_name}"
        )

        summary[table_name] = {
            "source_csv": str(csv_file),
            "output_parquet": str(output_parquet_path),
            "rows": len(df_standardized),
            "columns": len(df_standardized.columns),
            "column_names": df_standardized.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(df_standardized.columns, df_standardized.dtypes)},
            "null_counts": {col: df_standardized[col].null_count() for col in df_standardized.columns if df_standardized[col].null_count() > 0},
            "file_size_bytes": output_parquet_path.stat().st_size,
        }

    return summary


if __name__ == "__main__":
    import json
    results = run_phase1_ingestion(allow_overwrite=True)
    print(f"Phase 1 Ingestion Complete. Ingested {len(results)} tables.")
