"""
Automated Test Suite for Phase 1 Bronze Ingestion and Data Integrity.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_ingestion import run_phase1_ingestion
from src.validation import (
    validate_row_count_parity,
    validate_primary_keys,
    validate_referential_integrity,
    audit_null_values,
    audit_temporal_sanity,
)


def test_phase1_pipeline():
    print("=" * 70)
    print("RUNNING PHASE 1 INGESTION AND TESTS")
    print("=" * 70)

    # 1. Run ingestion
    print("\n[Step 1] Ingesting raw CSVs -> Bronze Parquet...")
    summary = run_phase1_ingestion(allow_overwrite=True)
    assert len(summary) == 9, f"Expected 9 ingested tables, got {len(summary)}"
    print(f"-> Successfully ingested {len(summary)} tables to Parquet.")

    # 2. Row count parity
    print("\n[Step 2] Validating 100% Row Count Parity...")
    parity = validate_row_count_parity()
    for table, res in parity.items():
        assert res["status"] == "PASSED", f"Parity failed for {table}: {res}"
        print(f"  [PASS] {table:<36}: {res['raw_rows']:>7} raw == {res['parquet_rows']:>7} parquet")

    # 3. Primary Key Uniqueness
    print("\n[Step 3] Validating Primary / Composite Key Uniqueness...")
    pk_results = validate_primary_keys()
    for table, res in pk_results.items():
        assert res["status"] == "PASSED", f"PK failed for {table}: {res}"
        print(f"  [PASS] {table:<36}: {res['unique_keys']:>7} unique keys across {res['total_rows']:>7} rows")

    # 4. Referential Integrity
    print("\n[Step 4] Validating Foreign Key Referential Integrity...")
    fk_results = validate_referential_integrity()
    for res in fk_results:
        assert res["status"] == "PASSED", f"FK constraint failed: {res}"
        print(f"  [PASS] {res['relationship']:<60}: {res['orphan_count']} orphan records")

    # 5. Null Value Audit
    print("\n[Step 5] Auditing Missing Values (Null Audit)...")
    null_audit = audit_null_values()
    for table, res in null_audit.items():
        null_cols = res["columns_with_nulls"]
        if not null_cols:
            print(f"  [INFO] {table:<36}: 0 null columns")
        else:
            cols_str = ", ".join([f"{c} ({v['null_count']} nulls, {v['null_percent']}%)" for c, v in null_cols.items()])
            print(f"  [INFO] {table:<36}: {cols_str}")

    # 6. Temporal Sanity
    print("\n[Step 6] Auditing Temporal Consistency...")
    temporal = audit_temporal_sanity()
    print(f"  Total orders checked: {temporal['total_orders']}")
    print(f"  Delivered orders checked: {temporal['delivered_orders_checked']}")
    print(f"  Deliveries before purchase timestamp: {temporal['delivery_before_purchase_count']}")
    print(f"  Estimated delivery before purchase: {temporal['estimated_before_purchase_count']}")
    print(f"  Delivered status with null delivery date: {temporal['delivered_status_with_null_date_count']}")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return {
        "summary": summary,
        "parity": parity,
        "pk": pk_results,
        "fk": fk_results,
        "null_audit": null_audit,
        "temporal": temporal,
    }


if __name__ == "__main__":
    test_phase1_pipeline()
