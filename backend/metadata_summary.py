"""Summarize session metadata for analyzed nights (full, subset, or multi-night)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.session_metadata import METADATA_TABLE_COLUMNS, _HYP_BUR_INDEX_COL

# Numeric metadata metrics for summary (excludes identity / time-of-night fields).
SUMMARY_METRIC_COLUMNS: Tuple[str, ...] = tuple(
    c
    for c in METADATA_TABLE_COLUMNS
    if c not in ("Subject_ID", "Session_ID", "Date", "Start Time")
)

_POSITION_COLUMNS = (
    "supine_proportion",
)

_COUNT_COLUMNS = (
    "NST Count",
    "Desat Count (3%)",
    "Desat Count (4%)",
    "Desat Count (sub 90%)",
    "Hypoxic Burden (4%)",
    "T90_min",
)

_RATE_COLUMNS = ("ODI", _HYP_BUR_INDEX_COL)

_SPO2_TRACE_COLUMNS = ("Min SpO2", "T90_perc")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _subset_fraction(
    full_duration_min: float,
    subset: Optional[Tuple[float, float]],
    analyzed_seconds: float,
) -> Tuple[float, float]:
    """
    Return (analyzed_duration_min, fraction_of_full_night).

    ``analyzed_seconds`` comes from the conditioned trace (authoritative when subset).
    """
    if subset is not None:
        start_m, end_m = subset
        analyzed_min = max(0.0, float(end_m) - float(start_m))
    elif analyzed_seconds > 0:
        analyzed_min = analyzed_seconds / 60.0
    else:
        analyzed_min = full_duration_min

    if full_duration_min > 0:
        frac = min(1.0, analyzed_min / full_duration_min)
    else:
        frac = 1.0 if analyzed_min > 0 else 0.0
    return analyzed_min, frac


def _position_proportions(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "position" not in df.columns:
        return {}
    pos = df["position"].dropna().astype(str).str.upper()
    if pos.empty:
        return {}
    counts = pos.value_counts()
    total = float(counts.sum())
    mapping = {
        "SUPINE": "supine_proportion",
        "PRONE": "prone_proportion",
        "LEFT": "left_proportion",
        "RIGHT": "right_proportion",
        "UPRIGHT": "upright_proportion",
    }
    out: Dict[str, float] = {}
    for label, col in mapping.items():
        out[col] = float(counts.get(label, 0)) / total
    if "supine_proportion" in out:
        out["non_supine_proportion"] = 1.0 - out["supine_proportion"]
    return out


def per_night_summary_metrics(
    metadata: Dict[str, Any],
    *,
    subset: Optional[Tuple[float, float]] = None,
    analyzed_seconds: float = 0.0,
    spo2_metrics: Optional[Dict[str, float]] = None,
    conditioned_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Build numeric summary metrics for one night.

    Full-night runs use metadata values. Subsets scale count/burden fields by time
    fraction and prefer SpO2 stats from the analyzed trace. Position proportions
    are recomputed from the trace when a position column exists and a subset is used.
    """
    full_dur = _as_float(metadata.get("Duration (min)")) or 0.0
    analyzed_min, frac = _subset_fraction(full_dur, subset, analyzed_seconds)

    if subset is None and frac >= 0.999:
        out: Dict[str, float] = {}
        for col in SUMMARY_METRIC_COLUMNS:
            val = _as_float(metadata.get(col))
            if val is not None:
                out[col] = val
        return out

    out = {"Duration (min)": analyzed_min}

    for col in _COUNT_COLUMNS:
        base = _as_float(metadata.get(col))
        if base is not None:
            out[col] = base * frac

    trace = spo2_metrics or {}
    if trace.get("min_spo2") is not None:
        out["Min SpO2"] = float(trace["min_spo2"])
    else:
        base = _as_float(metadata.get("Min SpO2"))
        if base is not None:
            out["Min SpO2"] = base

    if trace.get("pct_below_90") is not None:
        out["T90_perc"] = float(trace["pct_below_90"])
    else:
        base = _as_float(metadata.get("T90_perc"))
        if base is not None:
            out["T90_perc"] = base

    nst = out.get("NST Count")
    if nst is not None and analyzed_min > 0:
        out["ODI"] = nst / (analyzed_min / 60.0)
    else:
        base = _as_float(metadata.get("ODI"))
        if base is not None:
            out["ODI"] = base

    burden = out.get("Hypoxic Burden (4%)")
    if burden is not None and analyzed_min > 0:
        out[_HYP_BUR_INDEX_COL] = burden / (analyzed_min / 60.0)
    else:
        base = _as_float(metadata.get(_HYP_BUR_INDEX_COL))
        if base is not None:
            out[_HYP_BUR_INDEX_COL] = base

    if subset is not None and conditioned_df is not None:
        pos_props = _position_proportions(conditioned_df)
        for col in _POSITION_COLUMNS:
            if col in pos_props:
                out[col] = pos_props[col]
            else:
                base = _as_float(metadata.get(col))
                if base is not None:
                    out[col] = base
    else:
        for col in METADATA_TABLE_COLUMNS:
            if col in _POSITION_COLUMNS or col.endswith("_proportion"):
                base = _as_float(metadata.get(col))
                if base is not None:
                    out[col] = base

    return out


def aggregate_night_summaries(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-night summary metrics across multiple nights."""
    if not rows:
        return {}
    if len(rows) == 1:
        return dict(rows[0])

    df = pd.DataFrame(rows)
    weights = df["Duration (min)"].fillna(0).astype(float)
    total_weight = float(weights.sum())
    out: Dict[str, float] = {}

    if "Duration (min)" in df.columns:
        out["Duration (min)"] = float(df["Duration (min)"].sum())

    if "Min SpO2" in df.columns:
        out["Min SpO2"] = float(df["Min SpO2"].min())

    for col in _COUNT_COLUMNS:
        if col in df.columns:
            out[col] = float(df[col].sum())

    weighted_cols = set(_RATE_COLUMNS + ("T90_perc",) + _POSITION_COLUMNS)
    for col in df.columns:
        if col in out or col == "Min SpO2" or col == "Duration (min)":
            continue
        if col in _COUNT_COLUMNS:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if col in weighted_cols or col.endswith("_proportion"):
            if total_weight > 0:
                out[col] = float((series * weights).sum() / total_weight)
            else:
                out[col] = float(series.mean())

    total_min = out.get("Duration (min)", 0.0)
    if total_min > 0:
        nst = out.get("NST Count")
        if nst is not None:
            out["ODI"] = nst / (total_min / 60.0)
        burden = out.get("Hypoxic Burden (4%)")
        if burden is not None:
            out[_HYP_BUR_INDEX_COL] = burden / (total_min / 60.0)

    return out


def summaries_for_results(
    per_night: List[Dict[str, Any]],
    metadata_by_night_id: Dict[str, Dict[str, Any]],
    subset: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build per-night and combined summary tables.

    Returns:
        per_night_df: one row per analyzed night
        combined_df: single-row aggregate (same columns)
        caption: human-readable description of adjustments
    """
    rows: List[Dict[str, float]] = []
    night_ids: List[str] = []

    for night in per_night:
        nid = night["night_id"]
        meta = metadata_by_night_id.get(nid)
        if not meta:
            continue
        metrics = per_night_summary_metrics(
            meta,
            subset=subset,
            analyzed_seconds=float(night.get("total_seconds", 0)),
            spo2_metrics=night.get("spo2_metrics"),
            conditioned_df=night.get("conditioned_df"),
        )
        rows.append(metrics)
        night_ids.append(nid)

    if not rows:
        empty = pd.DataFrame(columns=list(SUMMARY_METRIC_COLUMNS))
        return empty, empty, "No metadata available for analyzed nights."

    per_night_df = pd.DataFrame(rows)
    per_night_df.insert(0, "night_id", night_ids)
    ordered = ["night_id", *[c for c in SUMMARY_METRIC_COLUMNS if c in per_night_df.columns]]
    per_night_df = per_night_df[ordered].round(2)

    combined = aggregate_night_summaries(rows)
    combined_df = pd.DataFrame([combined]).round(2)
    combined_cols = [c for c in SUMMARY_METRIC_COLUMNS if c in combined_df.columns]
    combined_df = combined_df[combined_cols]

    if subset is not None:
        caption = (
            f"Metrics adjusted for analyzed window "
            f"({subset[0]:.0f}–{subset[1]:.0f} min): counts/burden scaled by time fraction; "
            f"SpO₂ and position (when available) from the analyzed trace."
        )
    elif len(rows) == 1:
        caption = "Overnight session metadata for the analyzed night."
    else:
        caption = (
            f"Aggregated across {len(rows)} nights "
            f"(sum: duration, counts, T90 min; min: Min SpO₂; "
            f"duration-weighted mean: rates, T90 %, position proportions)."
        )

    return per_night_df, combined_df, caption
