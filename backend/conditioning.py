"""
Signal conditioning mirrored from data_pipeline.preprocess.Preprocess.

Applies the same steps as apnea_detection_auto (Zephyr) and
apnea_detection_neurostim_auto (PSG) without queue workers or Analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.stats import zscore

from backend.config import DEFAULT_SAMPLE_RATE, DataSource
from backend.time_columns import (
    align_time_for_merge,
    ensure_time_column,
    print_dataframe_head,
    resample_to_8hz,
)
from backend.types import NightData, NightRef

# Audio conditioning steps (fixed pipeline order; multiselect enables a subset).
AUDIO_CONDITIONING_STEP_ORDER = ("savgol", "normalization", "standardization", "nonlinear")
DEFAULT_AUDIO_CONDITIONING_STEPS = ("savgol", "normalization", "standardization")


def audio_steps_from_config(cond: Dict[str, Any]) -> tuple[str, ...]:
    """Resolve enabled audio steps from config (new list or legacy filter flags)."""
    if raw := cond.get("audio_conditioning_steps"):
        selected = {str(s).lower() for s in raw}
        steps = tuple(s for s in AUDIO_CONDITIONING_STEP_ORDER if s in selected)
        return steps if steps else DEFAULT_AUDIO_CONDITIONING_STEPS

    steps: list[str] = []
    ft = str(cond.get("filter_type", "savitzky_golay")).lower()
    if ft in ("savitzky_golay", "savgol"):
        steps.append("savgol")
    norm = str(cond.get("normalization", "min_max")).lower()
    if norm and norm not in ("none", ""):
        steps.append("normalization")
    std = str(cond.get("standardization", "z_score")).lower()
    if std and std not in ("none", ""):
        steps.append("standardization")
    if cond.get("use_nonlinear_scaling"):
        steps.append("nonlinear")
    return tuple(steps) if steps else DEFAULT_AUDIO_CONDITIONING_STEPS


@dataclass
class ConditioningParams:
    """Configurable preprocessing (replaces DefaultConfig fields used by Preprocess)."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    target_sample_rate: int = DEFAULT_SAMPLE_RATE
    audio_steps: tuple[str, ...] = DEFAULT_AUDIO_CONDITIONING_STEPS
    downsample_step_size: int = 1
    force_sample_rate: Optional[int] = 8  # Zephyr pipeline uses 8 after load
    spo2_interp_window_sec: int = 30
    slope_threshold: float = 0.025
    scale_factor_high: float = 1.0
    scale_factor_low: float = 0.75
    apply_spo2_artifact_cleanup: bool = True
    zephyr_downsample_step: int = 1000
    is_csv_audio: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any], source: Optional[DataSource] = None) -> "ConditioningParams":
        cond = config.get("conditioning", config)
        # Pipeline grid is fixed at 8 Hz (see resample_to_8hz); ignore legacy downsample_rate in saved configs.
        params = cls(
            target_sample_rate=DEFAULT_SAMPLE_RATE,
            audio_steps=audio_steps_from_config(cond),
            spo2_interp_window_sec=int(cond.get("spo2_interpolation_window", 30)),
            slope_threshold=float(cond.get("slope_threshold", 0.025)),
            scale_factor_high=float(cond.get("scale_factor_high", 1.0)),
            scale_factor_low=float(cond.get("scale_factor_low", 0.75)),
        )
        if source == DataSource.ZEPHYR:
            params.downsample_step_size = params.zephyr_downsample_step if params.is_csv_audio else 1
            params.force_sample_rate = 8
            params.apply_spo2_artifact_cleanup = True
        elif source in (DataSource.SCIDB, DataSource.MESA):
            params.downsample_step_size = int(cond.get("downsample_step_size", 1))
            params.force_sample_rate = None
            params.apply_spo2_artifact_cleanup = False
        return params

    @classmethod
    def for_source(cls, source: DataSource, config: Optional[Dict[str, Any]] = None) -> "ConditioningParams":
        return cls.from_dict(config or {}, source=source)


@dataclass
class ConditionedNightData:
    """Merged, conditioned night ready for scoring / window extraction."""

    ref: NightRef
    df: pd.DataFrame
    sample_rate: int
    total_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Conditioner:
    """Standalone mirror of data_pipeline.preprocess.Preprocess."""

    def __init__(self, params: ConditioningParams):
        self.params = params
        self.sample_rate = params.sample_rate

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        return df.dropna().drop_duplicates()

    def get_sample_rate(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return self.params.target_sample_rate
        dt = df["time"].iloc[1] - df["time"].iloc[0]
        return max(1, int(round(1.0 / dt))) if dt > 0 else self.params.target_sample_rate

    def downsample(self, df: pd.DataFrame, step: int) -> pd.DataFrame:
        df = df.copy()
        self.sample_rate = self.get_sample_rate(df)
        df = df.iloc[::step, :].reset_index(drop=True)
        self.sample_rate = self.get_sample_rate(df)
        self.params.sample_rate = self.sample_rate
        return df

    def minmax_normalize(self, df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
        df = df.copy()
        minim, maxim = np.min(df[column]), np.max(df[column])
        denom = maxim - minim
        if denom == 0:
            df[column] = 0.0
        else:
            df[column] = (df[column] - minim) / denom
        return df

    def zscore_standardize(self, df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
        df = df.copy()
        df[column] = zscore(df[column].astype(float))
        return df

    def nonlinear_scaling(self, df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
        df = df.copy()
        sr = self.sample_rate
        df["slope"] = (
            df[column]
            .rolling(window=sr, min_periods=1)
            .apply(lambda x: (x[-1] - x[0]) / 2, raw=True)
        )
        df["scale_factor"] = np.where(
            np.abs(df["slope"]) > self.params.slope_threshold,
            self.params.scale_factor_high,
            self.params.scale_factor_low,
        )
        df[column] = df[column] * df["scale_factor"]
        return df.drop(columns=["slope", "scale_factor"])

    def apply_savgol(self, df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
        df = df.copy()
        if len(df) >= 11:
            df[column] = savgol_filter(df[column].values, window_length=11, polyorder=3)
        return df

    def butterworth_filter(
        self,
        data: pd.Series,
        cutoff: float = 1,
        order: int = 4,
        pass_type: str = "high",
    ) -> np.ndarray:
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=pass_type, analog=False)
        return filtfilt(b, a, data.astype(float).values)

    def interpolate_segments(self, df: pd.DataFrame, max_sec: int) -> pd.DataFrame:
        """Interpolate SpO2/HR artifact segments (Zephyr)."""
        df = df.copy()
        max_length = max_sec * self.sample_rate
        mask = df["artifact"].to_numpy()
        # Work in float64 — np.interp yields fractions; int64 columns reject those values.
        spo2_values = pd.to_numeric(df["oxygen"], errors="coerce").astype(np.float64).copy()
        hr_values = (
            pd.to_numeric(df["heartrate"], errors="coerce").astype(np.float64).copy()
            if "heartrate" in df.columns
            else None
        )

        i = 0
        while i < len(mask):
            if not mask[i]:
                i += 1
                continue
            start = i
            while i < len(mask) and mask[i]:
                i += 1
            left_idx, right_idx = start - 1, i
            length = i - start

            if hr_values is not None and 0 <= left_idx < len(hr_values) and right_idx < len(hr_values):
                interp_values = np.interp(
                    range(start, i),
                    [left_idx, right_idx],
                    [float(hr_values.iloc[left_idx]), float(hr_values.iloc[right_idx])],
                )
                hr_values.iloc[start:i] = interp_values

            if length <= max_length and 0 <= left_idx < len(spo2_values) and right_idx < len(spo2_values):
                interp_values = np.interp(
                    range(start, i),
                    [left_idx, right_idx],
                    [float(spo2_values.iloc[left_idx]), float(spo2_values.iloc[right_idx])],
                )
                spo2_values.iloc[start:i] = interp_values
                df.loc[df.index[start:i], "artifact"] = False

        df["oxygen"] = spo2_values
        if hr_values is not None:
            df["heartrate"] = hr_values
        return df

    def apply_audio_conditioning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply enabled steps in pipeline order on the value column."""
        enabled = set(self.params.audio_steps)
        if "savgol" in enabled:
            df = self.apply_savgol(df)
        if "normalization" in enabled:
            df = self.minmax_normalize(df)
        if "standardization" in enabled:
            df = self.zscore_standardize(df)
        if "nonlinear" in enabled:
            df = self.nonlinear_scaling(df)
        return df


def _empty_spo2_row() -> pd.DataFrame:
    return pd.DataFrame(
        [{"time": 0, "heartrate": 0, "oxygen": 0, "confidence": 0, "status": 0}]
    )


def _empty_position_row() -> pd.DataFrame:
    return pd.DataFrame(
        [{"time": 0, "x-axis": 0, "y-axis": 0, "z-axis": 0, "position": "UPRIGHT"}]
    )


def _mark_spo2_artifacts(spo2_df: pd.DataFrame, conditioner: Conditioner) -> pd.DataFrame:
    spo2_df = spo2_df.copy()
    if len(spo2_df) <= 15:
        spo2_df["artifact"] = True
        return spo2_df

    spo2_df["filtered_oxygen"] = conditioner.butterworth_filter(spo2_df["oxygen"])
    spo2_df["oxygen_sq"] = np.square(spo2_df["filtered_oxygen"])
    cond_raw = (spo2_df["oxygen"] > 100) | (spo2_df["oxygen"] < 50)
    cond_art = cond_raw | (spo2_df["oxygen_sq"] > 30)
    if "status" in spo2_df.columns and "confidence" in spo2_df.columns:
        cond_status = (spo2_df["status"] != 3) | (spo2_df["confidence"] < 90)
    else:
        cond_status = pd.Series(False, index=spo2_df.index)
    spo2_df["artifact"] = cond_art | cond_status
    return spo2_df


def _recalc_spo2_artifacts(spo2_df: pd.DataFrame, conditioner: Conditioner) -> pd.DataFrame:
    spo2_df = spo2_df.copy()
    if len(spo2_df) <= 15:
        return spo2_df
    spo2_df["filtered_oxygen"] = conditioner.butterworth_filter(spo2_df["oxygen"])
    spo2_df["oxygen_sq"] = np.square(spo2_df["filtered_oxygen"])
    cond_raw = (spo2_df["oxygen"] > 100) | (spo2_df["oxygen"] < 50)
    cond_art = cond_raw | (spo2_df["oxygen_sq"] > 30)
    if "status" in spo2_df.columns and "confidence" in spo2_df.columns:
        cond_status = (spo2_df["status"] != 3) | (spo2_df["confidence"] < 90)
    else:
        cond_status = pd.Series(False, index=spo2_df.index)
    spo2_df["artifact"] = cond_art | cond_status
    return spo2_df


def condition_night(
    night: NightData,
    params: Optional[ConditioningParams] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ConditionedNightData:
    """
    Run source-appropriate conditioning on loaded night data.

    Args:
        night: Output from load_night_data()
        params: Explicit conditioning params (overrides config)
        config: ParameterSet-style dict with 'conditioning' key
    """
    if params is None:
        params = ConditioningParams.for_source(night.ref.source, config)

    if night.ref.source == DataSource.ZEPHYR:
        return _condition_zephyr(night, params)
    return _condition_psg(night, params)


def _condition_zephyr(night: NightData, params: ConditioningParams) -> ConditionedNightData:
    conditioner = Conditioner(params)
    df = conditioner.clean_data(night.audio)

    if params.is_csv_audio and params.downsample_step_size > 1:
        df = conditioner.downsample(df, params.downsample_step_size)

    if params.force_sample_rate is not None:
        conditioner.sample_rate = params.force_sample_rate
        params.sample_rate = params.force_sample_rate
    else:
        conditioner.sample_rate = conditioner.get_sample_rate(df)
        params.sample_rate = conditioner.sample_rate

    df = ensure_time_column(df)
    df = df[~df["time"].duplicated(keep="last")]

    spo2_df = night.spo2.copy() if night.spo2 is not None and not night.spo2.empty else _empty_spo2_row()
    spo2_df.columns = spo2_df.columns.str.strip().str.lower()
    if "oxygen" in spo2_df.columns:
        spo2_df = spo2_df.dropna(subset=["oxygen"]).reset_index(drop=True)

    position_df = (
        night.position.copy()
        if night.position is not None and not night.position.empty
        else _empty_position_row()
    )
    position_df.columns = position_df.columns.str.strip().str.lower()
    spo2_df = ensure_time_column(spo2_df, label="condition/spo2-in")
    position_df = ensure_time_column(position_df, label="condition/gravity-in")
    print_dataframe_head("condition/spo2-in", spo2_df)
    print_dataframe_head("condition/gravity-in", position_df)

    if params.apply_spo2_artifact_cleanup:
        spo2_df = _mark_spo2_artifacts(spo2_df, conditioner)
        spo2_df = conditioner.interpolate_segments(spo2_df, params.spo2_interp_window_sec)

    spo2_df = resample_to_8hz(spo2_df, label="condition/spo2", role="spo2")
    position_df = resample_to_8hz(position_df, label="condition/gravity", role="gravity")

    if params.apply_spo2_artifact_cleanup:
        spo2_df = _recalc_spo2_artifacts(spo2_df, conditioner)

    oxy_start = float(spo2_df["time"].min()) if not spo2_df.empty else 0.0
    if position_df.empty or "position" not in position_df.columns:
        g_start = float(position_df["time"].min()) if not position_df.empty else oxy_start
    else:
        valid_pos = position_df.dropna(subset=["position"])
        g_start = float(valid_pos["time"].min()) if not valid_pos.empty else float(position_df["time"].min())
    start = max(oxy_start, g_start)

    df = df[df["time"] > start]
    spo2_df = spo2_df[spo2_df["time"] > start]
    position_df = position_df[position_df["time"] > start]

    df = align_time_for_merge(df)
    spo2_df = align_time_for_merge(spo2_df)
    position_df = align_time_for_merge(position_df)
    df = df.merge(spo2_df, how="outer", on="time")
    df = df.merge(position_df, how="outer", on="time")

    numeric_cols = ["oxygen", "confidence", "status", "value", "heartrate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    pos_cols = ["x-axis", "y-axis", "z-axis", "position"]
    for col in pos_cols:
        if col in df.columns:
            if col == "position":
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = df[col].ffill().fillna(0)

    df = conditioner.apply_audio_conditioning(df)
    df = df.reset_index(drop=True)

    total_sec = float(df["time"].iloc[-1]) if len(df) else 0.0
    return ConditionedNightData(
        ref=night.ref,
        df=df,
        sample_rate=conditioner.sample_rate,
        total_seconds=total_sec,
        metadata={"source": "zephyr", "conditioning": "full"},
    )


def _condition_psg(night: NightData, params: ConditioningParams) -> ConditionedNightData:
    conditioner = Conditioner(params)
    df = ensure_time_column(conditioner.clean_data(night.audio), label="condition/psg-in")
    print_dataframe_head("condition/psg-in", df)
    df = df[~df["time"].duplicated(keep="last")]

    if params.is_csv_audio and params.downsample_step_size > 1:
        df = conditioner.downsample(df, params.downsample_step_size)

    conditioner.sample_rate = conditioner.get_sample_rate(df)
    params.sample_rate = conditioner.sample_rate

    if conditioner.sample_rate != DEFAULT_SAMPLE_RATE:
        df = resample_to_8hz(df, label="condition/psg_audio")
        conditioner.sample_rate = DEFAULT_SAMPLE_RATE
        params.sample_rate = DEFAULT_SAMPLE_RATE

    spo2_df = night.spo2.copy() if night.spo2 is not None and not night.spo2.empty else _empty_spo2_row()
    spo2_df.columns = spo2_df.columns.str.strip().str.lower()
    if "oxygen" in spo2_df.columns:
        spo2_df = spo2_df.dropna(subset=["oxygen"]).reset_index(drop=True)
        spo2_df = ensure_time_column(spo2_df, label="condition/psg-spo2-in")
        if not spo2_df.empty:
            spo2_df = resample_to_8hz(spo2_df, label="condition/psg-spo2", role="spo2")
            spo2_df = align_time_for_merge(spo2_df)

    df = conditioner.apply_audio_conditioning(df)

    # PSG pipeline stubs missing wearable columns after conditioning
    df["artifact"] = 0
    df["status"] = 3
    df["confidence"] = 100
    df["heartrate"] = 70
    df["x-axis"] = 0
    df["y-axis"] = 0
    df["z-axis"] = 0
    df["position"] = "UPRIGHT"

    if not spo2_df.empty and "oxygen" in spo2_df.columns:
        df = align_time_for_merge(df)
        df = df.merge(spo2_df, how="outer", on="time")

    numeric_cols = ["oxygen", "confidence", "status", "value", "heartrate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in ["x-axis", "y-axis", "z-axis"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)
    if "position" in df.columns:
        df["position"] = df["position"].ffill().bfill()

    df = df.reset_index(drop=True)
    total_sec = float(df["time"].iloc[-1]) if len(df) else 0.0

    return ConditionedNightData(
        ref=night.ref,
        df=df,
        sample_rate=conditioner.sample_rate,
        total_seconds=total_sec,
        metadata={"source": night.ref.source.value, "conditioning": "psg"},
    )


def load_and_condition_night(
    night_ref: NightRef,
    config: Optional[Dict[str, Any]] = None,
    params: Optional[ConditioningParams] = None,
    **load_kwargs: Any,
) -> ConditionedNightData:
    """Convenience: load from S3 then condition."""
    from backend.data_loader import load_night_data

    night = load_night_data(night_ref, **load_kwargs)
    is_csv = night.ref.primary_key.lower().endswith(".csv")
    if params is None:
        params = ConditioningParams.for_source(night.ref.source, config)
    params.is_csv_audio = is_csv
    if night.ref.source == DataSource.ZEPHYR and is_csv:
        params.downsample_step_size = params.zephyr_downsample_step
    return condition_night(night, params=params)
