"""Time helpers — ``data_pipeline.preprocess.resample`` using numeric seconds only."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Zephyr companion CSVs (1 Hz integer ``time`` in seconds; drop clock columns).
SPO2_SIGNAL_COLS = ("time", "heartrate", "oxygen", "confidence", "status")
GRAVITY_SIGNAL_COLS = ("time", "x-axis", "y-axis", "z-axis", "position")

_GRID_STEP_SEC = 0.125  # 8 Hz
# 1 Hz CSV integers — forward-fill on upsample (not linear blend).
_DISCRETE_UPSAMPLE_COLS = frozenset({"status", "confidence"})


def _normalize_col_name(col: str) -> str:
    return col.strip().lower().replace(" ", "")


def print_dataframe_head(label: str, df: pd.DataFrame, n: int = 5) -> None:
    """Print columns, dtypes, and head rows for chat debugging."""
    if df is None or df.empty:
        print(f"[HEAD] {label}: (empty)")
        return
    print(f"[HEAD] {label}: rows={len(df)} cols={list(df.columns)}")
    print(f"[HEAD] {label} dtypes:\n{df.dtypes.to_string()}")
    if "time" in df.columns:
        t = df["time"]
        print(
            f"[HEAD] {label} time: dtype={t.dtype} "
            f"min={t.min()} max={t.max()} "
            f"sample={t.head(n).tolist()}"
        )
    print(f"[HEAD] {label} head:\n{df.head(n).to_string()}")


def prune_companion_columns(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Keep pipeline columns only; drop ``time(min)``, ``time(hr)``, ``action``, etc."""
    want = SPO2_SIGNAL_COLS if role == "spo2" else GRAVITY_SIGNAL_COLS
    lower_map = {_normalize_col_name(c): c for c in df.columns}
    keep = [lower_map[_normalize_col_name(c)] for c in want if _normalize_col_name(c) in lower_map]
    if not keep:
        print(f"[HEAD] prune/{role}: no pipeline columns in {list(df.columns)}")
        return df
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        print(f"[HEAD] prune/{role}: dropped {dropped}")
    out = df[keep].copy()
    print_dataframe_head(f"prune/{role}/kept", out)
    return out


def ensure_time_column(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """``time`` = numeric elapsed seconds (Zephyr/PSG CSVs use int or float seconds)."""
    if df.empty or "time" not in df.columns:
        return df
    out = df.copy()
    raw = out["time"]
    if label:
        print_dataframe_head(f"{label}/before-coerce", out)
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["time"])
    dropped = before - len(out)
    if dropped and label:
        print(f"[HEAD] {label}: dropped {dropped} rows with non-numeric time")
    return out


def align_time_for_merge(df: pd.DataFrame, col: str = "time", decimals: int = 3) -> pd.DataFrame:
    """``apnea_detection_auto``: ``df['time'].round(3)`` before outer merge."""
    out = ensure_time_column(df)
    if col in out.columns and not out.empty:
        out[col] = out[col].astype(float).round(decimals)
    return out


def _median_time_step(times: np.ndarray) -> float:
    if len(times) < 2:
        return 1.0
    d = np.diff(np.sort(times))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def _should_round_time_to_int_seconds(times: np.ndarray) -> bool:
    """1 Hz companions: round to int seconds. Already ~8 Hz: keep sub-second grid."""
    return _median_time_step(times) > 0.5


def _ffill_on_grid(t_src: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_src, grid, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def _categorical_ffill_on_grid(
    t_src: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Forward-fill string/categorical labels onto the 8 Hz grid (no numeric coercion)."""
    idx = np.searchsorted(t_src, grid, side="right") - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    return labels[idx]


def resample_to_8hz(
    data: pd.DataFrame,
    reindex: bool = True,
    label: str = "resample",
    role: Optional[str] = None,
) -> pd.DataFrame:
    """
    Upsample to 8 Hz on a uniform 0.125 s grid (numeric seconds — no timedelta index).

    Matches ``preprocess.resample`` output: ``time`` in float seconds, step 0.125.
    """
    if role:
        data = prune_companion_columns(data, role)
    if "time" not in data.columns:
        raise ValueError(f"{label}: missing required column 'time'")

    print_dataframe_head(f"{label}/input", data)
    data = ensure_time_column(data.copy())
    if data.empty:
        print(f"[HEAD] {label}: empty after numeric time coerce")
        return data

    data = data.sort_values("time")
    t_src = data["time"].to_numpy(dtype=np.float64)

    if reindex and _should_round_time_to_int_seconds(t_src):
        data["time"] = data["time"].round().astype(np.int64)
        t_src = data["time"].to_numpy(dtype=np.float64)
        print(f"[HEAD] {label}: rounded time to int seconds (1 Hz companion)")
    else:
        data["time"] = data["time"].astype(np.float64)
        t_src = data["time"].to_numpy(dtype=np.float64)
        print(f"[HEAD] {label}: keeping sub-second time (median step={_median_time_step(t_src):.4f}s)")

    data = data.drop_duplicates(subset=["time"])
    t_src = data["time"].to_numpy(dtype=np.float64)
    print_dataframe_head(f"{label}/pre-grid", data)

    t0, t1 = float(t_src[0]), float(t_src[-1])
    grid = np.arange(t0, t1 + _GRID_STEP_SEC * 0.5, _GRID_STEP_SEC)

    rows: dict[str, np.ndarray] = {"time": np.round(grid, 3)}
    position_col = "position" if "position" in data.columns else None

    for col in data.columns:
        if col == "time":
            continue
        series = data[col]
        if col == position_col:
            y = series.astype(object).to_numpy()
            rows[col] = _categorical_ffill_on_grid(t_src, y, grid)
        elif col in _DISCRETE_UPSAMPLE_COLS:
            y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            rows[col] = _ffill_on_grid(t_src, y, grid)
        else:
            y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            rows[col] = np.interp(grid, t_src, y)

    out = pd.DataFrame(rows)
    print_dataframe_head(f"{label}/output", out)
    return out
