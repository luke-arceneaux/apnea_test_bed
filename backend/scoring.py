"""Variance-based flatline/onset extraction (Analytics.run_onset_extraction logic)."""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from backend.config import (
    DEFAULT_MIN_APNEA_SECONDS,
    DEFAULT_SEC_AFTER_ONSET,
    DEFAULT_SEC_BEFORE_ONSET,
    DEFAULT_VARIANCE_THRESHOLD,
)


def extract_flatline_times(
    df: pd.DataFrame,
    sample_rate: int = 8,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    min_apnea_seconds: int = DEFAULT_MIN_APNEA_SECONDS,
    seconds_before_apnea: int = DEFAULT_SEC_BEFORE_ONSET,
    seconds_after_apnea: int = DEFAULT_SEC_AFTER_ONSET,
    include_apnea: bool = True,
    max_duration_sec: int = 120,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Detect flatline segments from conditioned audio.

    Returns:
        flatline_times: list of (start_sec, end_sec)
        onset_times: list of (start_sec, end_sec) onset windows
    """
    work = df[["time", "value"]].copy()
    work["variance"] = (
        work["value"]
        .rolling(window=int(sample_rate * 4), min_periods=1)
        .apply(lambda x: np.var(x), raw=True)
    )
    work["binary_var"] = np.where(np.abs(work["variance"]) >= variance_threshold, 1, 0)
    bin_str = "".join(str(int(x)) for x in work["binary_var"].tolist())

    flatline_times: List[Tuple[float, float]] = []
    onset_times: List[Tuple[float, float]] = []
    min_apnea_length = int(sample_rate) * int(min_apnea_seconds)

    if not include_apnea:
        return flatline_times, onset_times

    pattern = (
        r"1{"
        + re.escape(f"{int(sample_rate)}")
        + "}0{"
        + re.escape(f"{min_apnea_length}")
        + r",}"
    )

    for match in re.finditer(pattern, bin_str):
        start_flatline_idx = match.start() + sample_rate
        end_flatline_idx = match.end()
        max_end_idx = start_flatline_idx + (max_duration_sec * sample_rate)
        if end_flatline_idx > max_end_idx:
            end_flatline_idx = max_end_idx

        start_onset = work.iloc[
            start_flatline_idx - (seconds_before_apnea * sample_rate)
        ]["time"]
        end_onset = work.iloc[
            start_flatline_idx + (seconds_after_apnea * sample_rate)
        ]["time"]
        onset_times.append((float(start_onset), float(end_onset)))

        start_flatline = work.iloc[start_flatline_idx]["time"]
        end_flatline = work.iloc[end_flatline_idx - 1]["time"]
        flatline_times.append((float(start_flatline), float(end_flatline)))

    return flatline_times, onset_times
