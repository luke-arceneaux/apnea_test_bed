"""Training window generation (collect_data logic) and train/val splits."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from backend.config import DataSource
from backend.features import DEFAULT_INPUT_FEATURES, normalize_feature_list, stack_feature_window
from backend.types import NightData, TrainingWindow, WindowGenerationParams

SplitStrategy = Literal["random", "by_night", "stratified_by_night"]  # stratified aliases to by_night


def params_for_source(source: DataSource) -> WindowGenerationParams:
    """Defaults mirroring apnea_detection_auto vs neurostim_auto collect_data."""
    if source in (DataSource.SCIDB, DataSource.MESA):
        return WindowGenerationParams(
            non_apnea_start_offset=15,
            non_apnea_end_offset=0,
            flatline_min_duration=13.0,
            skip_last_flatline=True,
        )
    return WindowGenerationParams(
        non_apnea_start_offset=16,
        non_apnea_end_offset=1,
        flatline_min_duration=15.0,
        skip_last_flatline=False,
    )


def flatline_events_from_times(flatline_times: Sequence[Tuple[float, float]]) -> pd.DataFrame:
    rows = [{"start": s, "duration": e - s} for s, e in flatline_times]
    if not rows:
        rows = [{"start": 0.0, "duration": 0.0}]
    return pd.DataFrame(rows)


def generate_training_windows(
    flatline_events: pd.DataFrame,
    audio_df: pd.DataFrame,
    total_sec: float,
    params: Optional[WindowGenerationParams] = None,
    source: Optional[DataSource] = None,
    input_features: Optional[Sequence[str]] = None,
) -> List[TrainingWindow]:
    """
    Replicate collect_data with configurable parameters.

    Labels: 0=non-apnea, 1=onset-apnea, 2=flatline.
    """
    if params is None:
        params = params_for_source(source) if source else WindowGenerationParams()
    features = normalize_feature_list(input_features)

    df = audio_df.copy()
    if "value" not in df.columns:
        raise ValueError("audio_df must contain a 'value' column")

    pytorch_data: List[TrainingWindow] = []
    n = flatline_events.shape[0]

    for i in range(n):
        row = flatline_events.iloc[i]
        duration = float(row["duration"])
        if duration < params.min_event_duration:
            continue

        # Class 1: onset-apnea
        try:
            apnea_start = float(row["start"]) - params.sec_before
            apnea_end = float(row["start"]) + params.sec_after
            apnea_seq = _slice_feature_window(
                df, apnea_start, apnea_end, features, params.window_samples
            )
            pytorch_data.append(TrainingWindow(1, apnea_start, apnea_seq))
        except (IndexError, ValueError) as exc:
            print(f"Error getting onset apnea sequence: {exc}")

        # Class 0: non-apnea
        try:
            if i + 1 >= n:
                reg_start = float(row["start"]) + duration
                reg_end = reg_start + params.flatline_window_sec
                if reg_end >= total_sec:
                    continue
            else:
                next_start = float(flatline_events.iloc[i + 1]["start"])
                reg_start = next_start - params.non_apnea_start_offset
                reg_end = next_start - params.non_apnea_end_offset
            reg_seq = _slice_feature_window(
                df, reg_start, reg_end, features, params.window_samples
            )
            pytorch_data.append(TrainingWindow(0, reg_start, reg_seq))
        except (IndexError, ValueError) as exc:
            print(f"Error getting non-apnea sequence: {exc}")

        # Class 2: flatline
        if params.skip_last_flatline and i + 1 >= n:
            continue
        if duration < params.flatline_min_duration:
            continue
        try:
            flat_start = float(row["start"])
            flat_end = flat_start + params.flatline_window_sec
            flat_seq = _slice_feature_window(
                df, flat_start, flat_end, features, params.window_samples
            )
            pytorch_data.append(TrainingWindow(2, flat_start, flat_seq))
        except (IndexError, ValueError) as exc:
            print(f"Error getting flatline sequence: {exc}")

    if params.balance_classes:
        pytorch_data = _balance_per_event(pytorch_data)

    return pytorch_data


def generate_windows_from_night(
    night: NightData,
    flatline_times: Sequence[Tuple[float, float]],
    params: Optional[WindowGenerationParams] = None,
    input_features: Optional[Sequence[str]] = None,
) -> List[TrainingWindow]:
    flatline_df = flatline_events_from_times(flatline_times)
    audio = night.audio if "value" in night.audio.columns else night.audio
    return generate_training_windows(
        flatline_df,
        audio,
        night.total_seconds,
        params=params,
        source=night.ref.source,
        input_features=input_features,
    )


def windows_to_arrays(
    windows: List[TrainingWindow],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, start_times, and stacked sequences."""
    labels = np.array([w.label for w in windows], dtype=np.int64)
    starts = np.array([w.start_time for w in windows], dtype=np.float64)
    seqs = np.stack([np.asarray(w.sequence, dtype=np.float64) for w in windows])
    return labels, starts, seqs


def split_windows(
    windows: List[TrainingWindow],
    strategy: SplitStrategy = "random",
    test_frac: float = 0.2,
    night_ids: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[List[TrainingWindow], List[TrainingWindow]]:
    """
    Split windows into train and validation sets.

    - random: shuffle all windows (current pipeline behavior)
    - by_night / stratified_by_night: require night_ids per window (same length);
      no night appears in both splits.
    """
    if not windows:
        return [], []

    rng = random.Random(seed)

    if strategy in ("random",) or night_ids is None:
        strategy = "random"
    elif strategy == "stratified_by_night":
        strategy = "by_night"

    if strategy == "random":
        shuffled = windows.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(test_frac * len(shuffled)))) if len(shuffled) > 1 else 0
        return shuffled[n_test:], shuffled[:n_test]

    # Group by night id
    by_night: Dict[str, List[TrainingWindow]] = {}
    for w, nid in zip(windows, night_ids):
        by_night.setdefault(nid, []).append(w)

    nights = list(by_night.keys())
    rng.shuffle(nights)
    n_test_nights = max(1, int(round(test_frac * len(nights))))
    test_nights = set(nights[:n_test_nights])

    train, val = [], []
    for nid, ws in by_night.items():
        if nid in test_nights:
            val.extend(ws)
        else:
            train.extend(ws)
    return train, val


def _slice_feature_window(
    df: pd.DataFrame,
    start: float,
    end: float,
    features: Sequence[str],
    window_samples: int,
) -> np.ndarray:
    start_idx = df.index[df["time"].astype(int) == int(start)].tolist()[0]
    end_idx = df.index[df["time"].astype(int) == int(end)].tolist()[0]
    return stack_feature_window(
        df,
        int(start_idx),
        int(end_idx),
        features,
        length=window_samples,
    )


def _balance_per_event(windows: List[TrainingWindow]) -> List[TrainingWindow]:
    """Keep up to one of each label per onset event group (approximate 1:1:1)."""
    # windows are emitted in order: 1, 0, 2 per event — return as-is (already balanced per event)
    return windows
