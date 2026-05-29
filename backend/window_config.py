"""Window width (seconds) → samples and training-label time spans."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

from backend.config import (
    DEFAULT_FLATLINE_WINDOW_SEC,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEC_AFTER_ONSET,
    DEFAULT_SEC_BEFORE_ONSET,
    DEFAULT_WINDOW_SAMPLES,
    DataSource,
)
from backend.training_windows import params_for_source
from backend.types import WindowGenerationParams


def sliding_window_width_sec(scoring: Dict[str, Any]) -> float:
    return float(scoring.get("sliding_window_width", DEFAULT_FLATLINE_WINDOW_SEC))


def window_samples_for_scoring(
    scoring: Dict[str, Any],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> int:
    """Tensor/window length in samples: width_sec × sample_rate."""
    return max(1, int(round(sliding_window_width_sec(scoring) * sample_rate)))


def scaled_onset_margins_sec(width_sec: int) -> Tuple[int, int]:
    """
    Onset label span scales with window width (reference 15 s: 10 s before, 5 s after).
    sec_before + sec_after == width_sec.
    """
    before = int(round(width_sec * DEFAULT_SEC_BEFORE_ONSET / DEFAULT_FLATLINE_WINDOW_SEC))
    before = max(0, min(before, width_sec))
    after = width_sec - before
    return before, after


def scoring_onset_margins(scoring: Dict[str, Any]) -> Tuple[int, int]:
    """Seconds before/after flatline start for onset extraction and class-1 windows."""
    width = int(round(sliding_window_width_sec(scoring)))
    return scaled_onset_margins_sec(width)


def scale_seconds_at_reference(
    reference_sec: float,
    width_sec: int,
    *,
    ref_width: float = DEFAULT_FLATLINE_WINDOW_SEC,
    min_value: float = 0.0,
) -> float:
    """Scale a duration defined at ``ref_width`` (default 15 s) to ``width_sec``."""
    if reference_sec <= 0:
        return reference_sec
    scaled = reference_sec * width_sec / ref_width
    if min_value > 0:
        scaled = max(min_value, scaled)
    return scaled


def scale_seconds_int(
    reference_sec: float,
    width_sec: int,
    *,
    ref_width: float = DEFAULT_FLATLINE_WINDOW_SEC,
    min_value: int = 0,
) -> int:
    """Integer seconds for index-based window placement (nearest second)."""
    if reference_sec <= 0:
        return 0
    return max(min_value, int(round(scale_seconds_at_reference(reference_sec, width_sec, ref_width=ref_width))))


def window_generation_params_from_scoring(
    scoring: Dict[str, Any],
    sample_rate: int,
    source: DataSource,
) -> WindowGenerationParams:
    """
    Training window geometry from scoring ``sliding_window_width``.

    All collect_data-style spans scale from their 15 s reference (``params_for_source``).
    """
    width = int(round(sliding_window_width_sec(scoring)))
    before, after = scoring_onset_margins(scoring)
    samples = window_samples_for_scoring(scoring, sample_rate)
    base = params_for_source(source)

    end_offset = scale_seconds_int(
        base.non_apnea_end_offset,
        width,
        min_value=1 if base.non_apnea_end_offset > 0 else 0,
    )
    return replace(
        base,
        sec_before=before,
        sec_after=after,
        flatline_window_sec=float(width),
        window_samples=samples,
        non_apnea_start_offset=scale_seconds_int(base.non_apnea_start_offset, width, min_value=1),
        non_apnea_end_offset=end_offset,
        flatline_min_duration=scale_seconds_at_reference(
            base.flatline_min_duration, width, min_value=1.0
        ),
        min_event_duration=scale_seconds_at_reference(base.min_event_duration, width, min_value=1.0),
    )


def assert_s3_ptl_window_compatible(window_samples: int) -> None:
    if window_samples != DEFAULT_WINDOW_SAMPLES:
        width_sec = window_samples / DEFAULT_SAMPLE_RATE
        raise ValueError(
            f"S3 .ptl engines expect {DEFAULT_WINDOW_SAMPLES} samples "
            f"({DEFAULT_FLATLINE_WINDOW_SEC} s @ {DEFAULT_SAMPLE_RATE} Hz). "
            f"Current window width is {width_sec:g} s ({window_samples} samples). "
            "Set window width to 15 s or use **Train new model**."
        )
