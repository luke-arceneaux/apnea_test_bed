"""Session metadata CSV: load, normalize, join to NightRef, and filter (dashboard parity)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.config import BUCKET_BY_SOURCE, DataSource
from backend.s3_client import TestbedS3Client
from backend.types import NightRef

METADATA_KEY = "metadata/session_metadata.csv"

# Canonical column names (matches data_pipeline/metadata_fields.py / dashboard CSV).
_ID_COLS = ("Subject_ID", "Session_ID", "Dataset")
_DATE_COL = "Date"
_START_TIME_COL = "Start Time"
_DURATION_COL = "Duration (min)"
_HYPOXIC_BURDEN_COL = "Hypoxic Burden (4%)"
_HYP_BUR_INDEX_COL = "HypBurIndex(4%)"
_SESSION_DT_COL = "SessionDateTime"

_CANONICAL_BY_LOWER = {
    "subject_id": "Subject_ID",
    "session_id": "Session_ID",
    "dataset": "Dataset",
    "date": "Date",
    "start time": "Start Time",
    "duration (min)": "Duration (min)",
    "odi": "ODI",
    "nst count": "NST Count",
    "desat count (3%)": "Desat Count (3%)",
    "desat count (4%)": "Desat Count (4%)",
    "desat count (sub 90%)": "Desat Count (sub 90%)",
    "hypoxic burden (4%)": "Hypoxic Burden (4%)",
    "min spo2": "Min SpO2",
    "t90_perc": "T90_perc",
    "t90_min": "T90_min",
    "supine_proportion": "supine_proportion",
    "prone_proportion": "prone_proportion",
    "left_proportion": "left_proportion",
    "right_proportion": "right_proportion",
    "upright_proportion": "upright_proportion",
    "non_supine_proportion": "non_supine_proportion",
    "device_id": "Device_ID",
    "chart_code": "Device_ID",
    "pdf report": "PDF Report",
    "interactive report": "Interactive Report",
    "spo2 snapshot": "SpO2 Snapshot",
    "version": "version",
}

# Reported sleep metrics from metadata (not recomputed in the testbed pipeline).
SESSION_REPORT_METRICS = (
    "Date",
    "Start Time",
    "Duration (min)",
    "ODI",
    "NST Count",
    "Desat Count (3%)",
    "Desat Count (4%)",
    "Desat Count (sub 90%)",
    "Hypoxic Burden (4%)",
    _HYP_BUR_INDEX_COL,
    "Min SpO2",
    "T90_perc",
    "T90_min",
    "supine_proportion",
    "prone_proportion",
    "left_proportion",
    "right_proportion",
    "upright_proportion",
    "non_supine_proportion",
    "Device_ID",
)

_DEFAULT_MIN_DURATION_MIN = 30.0
_DATE_CUTOFF = date(2026, 1, 1)

# Metadata fields shown in the catalog table (before analysis).
METADATA_TABLE_COLUMNS = (
    "Subject_ID",
    "Session_ID",
    "Date",
    "Start Time",
    "Duration (min)",
    "ODI",
    "NST Count",
    "Desat Count (3%)",
    "Desat Count (4%)",
    "Desat Count (sub 90%)",
    "HypBurIndex(4%)",
    "Min SpO2",
    "T90_perc",
    "T90_min",
    "supine_proportion",
)


def dataset_for_source(source: DataSource) -> str:
    """Metadata ``Dataset`` value for a testbed data source."""
    return source.value


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        _CANONICAL_BY_LOWER.get(str(c).strip().lower(), str(c).strip())
        for c in out.columns
    ]
    return out


def load_session_metadata(
    source: DataSource,
    s3: Optional[TestbedS3Client] = None,
    key: str = METADATA_KEY,
) -> pd.DataFrame:
    """Load and normalize session metadata CSV from the source bucket."""
    client = s3 or TestbedS3Client(BUCKET_BY_SOURCE[source])
    try:
        raw, _ = client.get_file_df(key)
    except Exception:
        return prepare_metadata_df(pd.DataFrame())
    return prepare_metadata_df(raw)


def prepare_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, parse dates/times, and add derived dashboard fields.
    """
    if df.empty:
        return df.copy()

    out = _canonicalize_columns(df)
    for col in _ID_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str)

    if _DATE_COL in out.columns:
        out[_DATE_COL] = pd.to_datetime(out[_DATE_COL], format="mixed").dt.date

    if _START_TIME_COL in out.columns:
        parsed_time = pd.to_datetime(out[_START_TIME_COL], format="mixed", errors="coerce")
        out[_START_TIME_COL] = parsed_time.dt.time

    if _DATE_COL in out.columns and _START_TIME_COL in out.columns:
        out[_SESSION_DT_COL] = pd.to_datetime(
            out[_DATE_COL].astype(str) + " " + out[_START_TIME_COL].astype(str),
            errors="coerce",
        )

    if _HYPOXIC_BURDEN_COL in out.columns and _DURATION_COL in out.columns:
        hours = out[_DURATION_COL].astype(float) / 60.0
        out[_HYP_BUR_INDEX_COL] = out[_HYPOXIC_BURDEN_COL].astype(float) / hours.replace(0, pd.NA)

    return out


def metadata_for_source(meta_df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
    """Rows for one data source's ``Dataset`` label."""
    if meta_df.empty or "Dataset" not in meta_df.columns:
        return meta_df.iloc[0:0].copy()
    want = dataset_for_source(source)
    return meta_df[meta_df["Dataset"].astype(str).str.lower() == want.lower()].copy()


def default_date_range(meta_df: pd.DataFrame) -> Tuple[date, date]:
    """
    Default session date filter bounds (dashboard ``filters.py`` logic).
    """
    if meta_df.empty or _DATE_COL not in meta_df.columns:
        today = date.today()
        return today, today

    min_date = meta_df[_DATE_COL].min()
    max_date = meta_df[_DATE_COL].max()
    if (meta_df[_DATE_COL] > _DATE_CUTOFF).any():
        default_min = _DATE_CUTOFF
    else:
        default_min = min_date
    return default_min, max_date


def filter_metadata_rows(
    meta_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    min_duration_min: float = _DEFAULT_MIN_DURATION_MIN,
) -> pd.DataFrame:
    """Apply dashboard date-range and minimum-duration filters."""
    if meta_df.empty:
        return meta_df.copy()

    out = meta_df.copy()
    if _DATE_COL in out.columns:
        out = out[(out[_DATE_COL] >= start_date) & (out[_DATE_COL] <= end_date)]
    if _DURATION_COL in out.columns:
        out = out[out[_DURATION_COL].astype(float) >= float(min_duration_min)]
    return out.reset_index(drop=True)


def _night_key(dataset: str, subject: str, session: str) -> Tuple[str, str, str]:
    return (str(dataset).lower(), str(subject), str(session))


def _ref_key(ref: NightRef) -> Tuple[str, str, str]:
    return _night_key(ref.dataset, ref.subject, ref.session)


@dataclass(frozen=True)
class NightRefWithMeta:
    """NightRef joined to one metadata row (strict catalog entries only)."""

    ref: NightRef
    metadata: Dict[str, Any]

    def report_metrics(self) -> Dict[str, Any]:
        """Subset of metadata fields for UI display (precomputed overnight stats)."""
        return {k: self.metadata[k] for k in SESSION_REPORT_METRICS if k in self.metadata}


def _row_to_metadata_dict(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (date, datetime)):
            out[k] = v.isoformat() if isinstance(v, datetime) else v.isoformat()
        else:
            out[k] = v
    return out


def join_refs_to_metadata(
    refs: List[NightRef],
    meta_df: pd.DataFrame,
    source: Optional[DataSource] = None,
) -> List[NightRefWithMeta]:
    """
    Inner join: only nights with a metadata row are returned.

    If ``source`` is set, ``meta_df`` is restricted to that dataset first.
    """
    scoped = metadata_for_source(meta_df, source) if source is not None else meta_df
    if scoped.empty:
        return []

    index: Dict[Tuple[str, str, str], pd.Series] = {}
    for _, row in scoped.iterrows():
        if "Dataset" not in row or "Subject_ID" not in row or "Session_ID" not in row:
            continue
        key = _night_key(row["Dataset"], row["Subject_ID"], row["Session_ID"])
        index[key] = row

    joined: List[NightRefWithMeta] = []
    for ref in refs:
        row = index.get(_ref_key(ref))
        if row is None:
            continue
        joined.append(NightRefWithMeta(ref=ref, metadata=_row_to_metadata_dict(row)))
    return joined


def list_nights_with_metadata(
    refs: List[NightRef],
    source: DataSource,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_duration_min: float = _DEFAULT_MIN_DURATION_MIN,
    meta_df: Optional[pd.DataFrame] = None,
    s3: Optional[TestbedS3Client] = None,
) -> List[NightRefWithMeta]:
    """
    Strict catalog: S3 nights that have metadata and pass date/duration filters.

    Nights without a metadata row are never included.
    """
    prepared = meta_df if meta_df is not None else load_session_metadata(source, s3=s3)
    scoped = metadata_for_source(prepared, source)
    if scoped.empty:
        return []

    if start_date is None or end_date is None:
        default_start, default_end = default_date_range(scoped)
        start_date = start_date if start_date is not None else default_start
        end_date = end_date if end_date is not None else default_end

    filtered = filter_metadata_rows(scoped, start_date, end_date, min_duration_min)
    return join_refs_to_metadata(refs, filtered, source=None)


def _format_display_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    return value


def catalog_to_display_dataframe(
    entries: List[NightRefWithMeta],
    *,
    columns: Tuple[str, ...] = METADATA_TABLE_COLUMNS,
) -> pd.DataFrame:
    """
    Build a dashboard-style metadata table for catalog nights (newest first).
    """
    if not entries:
        return pd.DataFrame(columns=list(columns))

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        row = {col: _format_display_value(entry.metadata.get(col)) for col in columns}
        rows.append(row)

    df = pd.DataFrame(rows)
    if _SESSION_DT_COL in entries[0].metadata:
        sort_vals = [_format_display_value(e.metadata.get(_SESSION_DT_COL)) for e in entries]
        df["_sort"] = pd.to_datetime(sort_vals, errors="coerce")
        df = df.sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort")

    numeric_cols = [
        c
        for c in columns
        if c
        not in ("Subject_ID", "Session_ID", "Date", "Start Time")
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present = [c for c in columns if c in df.columns]
    out = df[present].round(2)
    return out.reset_index(drop=True)
