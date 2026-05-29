"""Training / inference input feature registry and availability checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.config import DataSource

# Current production behavior: audio (NAF value) only.
DEFAULT_INPUT_FEATURES: Tuple[str, ...] = ("audio",)

# MVP optional channels (POC).
OPTIONAL_FEATURES: Tuple[str, ...] = ("heartrate", "position")

ALL_FEATURE_NAMES: Tuple[str, ...] = DEFAULT_INPUT_FEATURES + OPTIONAL_FEATURES

# PSG conditioning stubs — not real wearable signals.
_PSG_STUB_HEARTRATE = 70.0
_PSG_STUB_POSITION = "UPRIGHT"

_POSITION_CODES: Dict[str, float] = {
    "UPRIGHT": 0.0,
    "SUPINE": 1.0,
    "PRONE": 2.0,
    "LEFT": 3.0,
    "RIGHT": 4.0,
}


@dataclass(frozen=True)
class FeatureSpec:
    """Maps a logical feature name to conditioned DataFrame column(s)."""

    name: str
    columns: Tuple[str, ...]
    kind: str  # continuous | categorical


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "audio": FeatureSpec("audio", ("value",), "continuous"),
    "heartrate": FeatureSpec("heartrate", ("heartrate",), "continuous"),
    "position": FeatureSpec("position", ("position",), "categorical"),
}


def normalize_feature_list(features: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Validate and return a stable ordered feature tuple (audio first if present)."""
    if not features:
        return DEFAULT_INPUT_FEATURES
    names = [str(f).strip().lower() for f in features if str(f).strip()]
    if not names:
        return DEFAULT_INPUT_FEATURES
    unknown = [n for n in names if n not in FEATURE_SPECS]
    if unknown:
        raise ValueError(f"Unknown input feature(s): {unknown}. Valid: {list(ALL_FEATURE_NAMES)}")
    if "audio" not in names:
        raise ValueError("Input features must include 'audio'.")
    # Stable order: audio, then optional in registry order
    ordered: List[str] = []
    for key in ALL_FEATURE_NAMES:
        if key in names and key not in ordered:
            ordered.append(key)
    return tuple(ordered)


def feature_set_key(features: Sequence[str]) -> str:
    return "+".join(normalize_feature_list(features))


def features_require_retrain(features: Sequence[str]) -> bool:
    return normalize_feature_list(features) != DEFAULT_INPUT_FEATURES


def n_channels_for_features(features: Sequence[str]) -> int:
    """Number of model input channels for a feature set."""
    return sum(len(FEATURE_SPECS[n].columns) for n in normalize_feature_list(features))


def available_features(
    df: pd.DataFrame,
    source: Optional[DataSource] = None,
) -> Tuple[str, ...]:
    """Features with real (non-stub) data in the conditioned trace."""
    out: List[str] = []
    if "value" in df.columns and len(df) > 0:
        out.append("audio")
    if _heartrate_available(df, source):
        out.append("heartrate")
    if _position_available(df, source):
        out.append("position")
    return tuple(out)


def validate_requested_features(
    requested: Sequence[str],
    df: pd.DataFrame,
    source: Optional[DataSource] = None,
    *,
    night_id: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return normalized features or raise if any requested channel is unavailable."""
    normalized = normalize_feature_list(requested)
    avail = set(available_features(df, source))
    missing = [f for f in normalized if f not in avail]
    if missing:
        nid = f" for night '{night_id}'" if night_id else ""
        raise ValueError(
            f"Requested feature(s) not available{nid}: {missing}. "
            f"Available: {sorted(avail)}."
        )
    return normalized


def _heartrate_available(df: pd.DataFrame, source: Optional[DataSource]) -> bool:
    if "heartrate" not in df.columns or df.empty:
        return False
    series = pd.to_numeric(df["heartrate"], errors="coerce").dropna()
    if series.empty:
        return False
    if source in (DataSource.SCIDB, DataSource.MESA):
        return False
    if series.nunique() <= 1:
        return False
    if float(series.std()) < 1e-6:
        return False
    return True


def _position_available(df: pd.DataFrame, source: Optional[DataSource]) -> bool:
    if "position" not in df.columns or df.empty:
        return False
    if source in (DataSource.SCIDB, DataSource.MESA):
        return False
    pos = df["position"].dropna().astype(str).str.strip().str.upper()
    pos = pos[~pos.isin(("", "NAN", "NONE"))]
    if pos.empty:
        return False
    # Zephyr without gravity uses conditioning stub (constant UPRIGHT).
    if pos.nunique() == 1 and pos.iloc[0] == _PSG_STUB_POSITION:
        return False
    return True


def encode_position_value(raw: object) -> float:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float(_POSITION_CODES[_PSG_STUB_POSITION])
    key = str(raw).strip().upper()
    return float(_POSITION_CODES.get(key, _POSITION_CODES[_PSG_STUB_POSITION]))


def extract_channel_series(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Extract one channel as float64 aligned to df row order."""
    col = spec.columns[0]
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing for feature '{spec.name}'")
    if spec.kind == "categorical":
        return np.array([encode_position_value(v) for v in df[col].values], dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float64).values


def stack_feature_window(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    features: Sequence[str],
    *,
    length: int,
) -> np.ndarray:
    """
    Slice [start_idx:end_idx) for each feature channel and pad/trim to `length`.

    Returns shape (n_channels, length).
    """
    from backend.ml import prepare_sequence

    normalized = normalize_feature_list(features)
    channels: List[np.ndarray] = []
    for name in normalized:
        spec = FEATURE_SPECS[name]
        series = extract_channel_series(df, spec)
        segment = series[int(start_idx) : int(end_idx)]
        channels.append(prepare_sequence(segment, length=length))
    return np.stack(channels, axis=0)
