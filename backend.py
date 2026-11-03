"""Deterministic record verification for the store support assistant."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ORDER_ID_RE = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
FIXED_REFERENCE = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
ISSUES = {"DOUBLE_CHARGE", "PAYMENT_FAILURES", "REFUND_STATUS", "DELIVERY_STATUS"}

# Columns the verification logic relies on. Adjust if your CSV headers differ.
REQUIRED_COLUMNS = {
    "payments": {"order_id", "status", "amount_cents", "provider_ref", "timestamp"},
    "refunds": {"order_id", "status"},
    "deliveries": {"order_id", "status", "expected_by"},
}
# Used to pick the most recent record when an order has several refund/delivery rows.
UPDATE_COLUMNS = ("updated_at", "updated", "timestamp", "created_at")


@dataclass
class Verification:
    status: str
    issue: str | None
    order_id: str | None
    explanation: str
    evidence: list[dict]


def _parse_utc(value) -> datetime:
    """Parse an ISO timestamp. Naive timestamps are treated as UTC (not server-local time)."""
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_records(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    files = {"payments": "payments.csv", "refunds": "refunds.csv", "deliveries": "deliveries.csv"}
    str_cols = {"order_id": "string", "payment_id": "string", "refund_id": "string",
                "delivery_id": "string", "provider_ref": "string"}
    records = {}
    for key, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
        df = pd.read_csv(path, dtype=str_cols)
        df.columns = [c.strip() for c in df.columns]
        missing = REQUIRED_COLUMNS[key] - set(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing required column(s): {', '.join(sorted(missing))}")
        records[key] = df
    return records


def _row_dict(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _rows_for_order(df: pd.DataFrame, order_id: str) -> pd.DataFrame:
    return df[_norm(df["order_id"]) == order_id].copy()


def _latest_row(rows: pd.DataFrame) -> pd.Series:
    """Return the most recent row when an order has several records."""
    if len(rows) == 1:
        return rows.iloc[0]
    for col in UPDATE_COLUMNS:
        if col in rows.columns:
            try:
                stamps = [_parse_utc(v) for v in rows[col]]
            except (ValueError, TypeError):
                continue
            return rows.iloc[max(range(len(stamps)), key=stamps.__getitem__)]
    return rows.iloc[-1]


def _clean_ref(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def verify(issue: str, order_id: str, records: dict[str, pd.DataFrame]) -> Verification:
    issue = (issue or "").upper()
    order_id = (order_id or "").upper()
    if issue not in ISSUES or not re.fullmatch(r"ORD-\d{4}", order_id):
        return Verification("NEEDS_INFORMATION", issue, order_id, "The issue or order ID is not valid.", [])

    # ---------- Payments ----------
    if issue in {"DOUBLE_CHARGE", "PAYMENT_FAILURES"}:
        rows = _rows_for_order(records["payments"], order_id)
        if rows.empty:
            return Verification("NEEDS_INFORMATION", issue, order_id, "No payment records were found for this order.", [])

        if issue == "DOUBLE_CHARGE":
            success = rows[_norm(rows["status"]) == "SUCCESS"]
            # Compare every pair (input may be unsorted); same provider ref is NOT a distinct transaction.
            for i in range(len(success)):
                for j in range(i + 1, len(success)):
                    a, b = success.iloc[i], success.iloc[j]
                    ref_a, ref_b = _clean_ref(a["provider_ref"]), _clean_ref(b["provider_ref"])
                    try:
                        same_amount = int(a["amount_cents"]) == int(b["amount_cents"])
                    except (TypeError, ValueError):
                        continue
                    if ref_a and ref_b and ref_a != ref_b and same_amount:
                        return Verification(
                            "SIGNAL_FOUND", issue, order_id,
                            "Two successful payments have equal amounts and different provider references.",
                            [_row_dict(a), _row_dict(b)])
            return Verification("NOT_VERIFIED", issue, order_id,
                                "Payment records exist, but no qualifying duplicate-charge pair was found.", [])

        failed = rows[_norm(rows["status"]) == "FAILED"].copy()
        if len(failed) < 3:
            return Verification("NOT_VERIFIED", issue, order_id,
                                "Payment records exist, but fewer than three failed attempts were found.", [])
        try:
            failed["_dt"] = failed["timestamp"].map(_parse_utc)
        except (ValueError, TypeError):
            return Verification("ERROR", issue, order_id, "A payment record has an unreadable timestamp.", [])
        failed = failed.sort_values("_dt")
        # The tightest window of 3 failures is always among consecutive sorted rows.
        for i in range(len(failed) - 2):
            triple = failed.iloc[i:i + 3]
            if (triple.iloc[-1]["_dt"] - triple.iloc[0]["_dt"]).total_seconds() <= 600:
                evidence = [_row_dict(r.drop(labels=["_dt"])) for _, r in triple.iterrows()]
                return Verification("SIGNAL_FOUND", issue, order_id,
                                    "Three failed payment attempts occurred within ten minutes.", evidence)
        return Verification("NOT_VERIFIED", issue, order_id,
                            "Payment records exist, but no three failed attempts fall within ten minutes.", [])

    # ---------- Refunds ----------
    if issue == "REFUND_STATUS":
        rows = _rows_for_order(records["refunds"], order_id)
        if rows.empty:
            return Verification("NEEDS_INFORMATION", issue, order_id,
                                "No refund record was found for this order. This does not establish whether a refund was requested.", [])
        row = _latest_row(rows)
        note = f" (most recent of {len(rows)} refund records)" if len(rows) > 1 else ""
        return Verification("RECORD_FOUND", issue, order_id,
                            f"A refund record exists{note}; its status and update time are recorded below.",
                            [_row_dict(row)])

    # ---------- Deliveries ----------
    rows = _rows_for_order(records["deliveries"], order_id)
    if rows.empty:
        return Verification("NEEDS_INFORMATION", issue, order_id, "No delivery record was found for this order.", [])
    row = _latest_row(rows)
    status = str(row["status"]).strip().upper()
    evidence = [_row_dict(row)]
    if status == "DELIVERED":
        return Verification("RECORD_FOUND", issue, order_id,
                            "The delivery record is marked DELIVERED and includes its recorded delivery time.", evidence)
    if status == "IN_TRANSIT":
        try:
            expected = _parse_utc(row["expected_by"])
        except (ValueError, TypeError):
            return Verification("ERROR", issue, order_id, "The delivery record has an unreadable expected-by time.", evidence)
        if FIXED_REFERENCE > expected:
            return Verification("SIGNAL_FOUND", issue, order_id,
                                "The delivery is marked IN_TRANSIT and its expected deadline is before the fixed reference time (2026-11-02 12:00 UTC).",
                                evidence)
        return Verification("RECORD_FOUND", issue, order_id,
                            "The delivery is marked IN_TRANSIT, and it is not overdue at the fixed reference time (2026-11-02 12:00 UTC).",
                            evidence)
    return Verification("ERROR", issue, order_id, f"Unexpected delivery status in the records: {status}.", [])
