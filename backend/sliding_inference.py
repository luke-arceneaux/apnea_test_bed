"""Full-night model inference at live cadence (~1 Hz) and actuation extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.config import DEFAULT_WINDOW_SAMPLES
from backend.features import (
    DEFAULT_INPUT_FEATURES,
    extract_channel_series,
    normalize_feature_list,
    n_channels_for_features,
)
from backend.features import FEATURE_SPECS
from backend.live_actuation import (
    LiveActuationParams,
    actuations_to_detection_list,
    probs_to_actuations,
)


def build_sliding_windows(
    df: pd.DataFrame,
    window_len: int = DEFAULT_WINDOW_SAMPLES,
    step: int = 1,
    value_col: str = "value",
    time_col: str = "time",
    input_features: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding windows; `step` should be sample_rate * inference_interval_sec for live parity."""
    features = normalize_feature_list(input_features)
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}'")
    for name in features:
        col = FEATURE_SPECS[name].columns[0]
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' for feature '{name}'")

    times = df[time_col].astype(float).to_numpy()
    channel_rows = [extract_channel_series(df, FEATURE_SPECS[n]) for n in features]
    n_rows = len(times)
    if n_rows < window_len:
        return np.zeros((0, n_channels_for_features(features), window_len), dtype=np.float32), np.zeros(0, dtype=np.float64)

    if step < 1:
        step = 1

    starts = np.arange(0, n_rows - window_len + 1, step)
    if len(starts) == 0:
        n_ch = n_channels_for_features(features)
        return np.zeros((0, n_ch, window_len), dtype=np.float32), np.zeros(0, dtype=np.float64)

    n_ch = len(channel_rows)
    windows = np.zeros((len(starts), n_ch, window_len), dtype=np.float32)
    for i, s in enumerate(starts):
        for c, row in enumerate(channel_rows):
            windows[i, c, :] = row[s : s + window_len]

    window_times = times[starts]
    # Legacy single-channel: return (N, T) for JIT compatibility
    if n_ch == 1:
        return windows[:, 0, :], window_times
    return windows, window_times


def predict_sliding_probs(
    model: Any,
    df: pd.DataFrame,
    model_type: str,
    is_jit: bool,
    *,
    batch_size: int = 32,
    window_len: int = DEFAULT_WINDOW_SAMPLES,
    step: int = 8,
    device: str = "cpu",
    input_features: Optional[Sequence[str]] = None,
    n_channels: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model at live cadence; return (frame_times, probs) with shape (n, 3).

    Default step=8 at 8 Hz ≈ one inference per second (matches AudioProcessor slide).
    Output treated as probabilities (no softmax); same as Android Lite module.
    """
    from backend.ml import _forward, _jit_batch_tensor, is_tabular_model, predict_tabular_proba

    import torch

    features = normalize_feature_list(input_features)
    X, window_times = build_sliding_windows(
        df,
        window_len=window_len,
        step=step,
        input_features=features,
    )
    if len(X) == 0:
        return window_times, np.zeros((0, 3), dtype=np.float32)

    if is_tabular_model(model_type):
        all_probs: List[List[float]] = []
        for start in range(0, len(X), batch_size):
            chunk = np.ascontiguousarray(X[start : start + batch_size], dtype=np.float64)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            proba = predict_tabular_proba(model, chunk)
            all_probs.extend(proba.tolist())
        return window_times, np.asarray(all_probs, dtype=np.float32)

    run_device = "cpu" if is_jit else device
    sample_dtype = np.float32 if is_jit else np.float64
    all_probs = []

    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = np.ascontiguousarray(X[start : start + batch_size], dtype=sample_dtype)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            if is_jit:
                batch = _jit_batch_tensor(chunk, device=run_device, n_channels=n_channels)
            else:
                batch = torch.tensor(chunk, dtype=torch.float64, device=run_device)
            out = _forward(model, batch, model_type, is_jit=is_jit, n_channels=n_channels)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            all_probs.extend(out.detach().cpu().numpy().tolist())

    return window_times, np.asarray(all_probs, dtype=np.float32)


def run_live_model_actuations(
    df: pd.DataFrame,
    model: Any,
    model_config: Any,
    is_jit: bool,
    sample_rate: int,
    prediction_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Full-night inference at ~1 Hz → live post-processing → actuation list.
    """
    params = LiveActuationParams.from_prediction_config(prediction_config)
    step = max(1, int(round(params.inference_interval_sec * sample_rate)))

    infer_type = "cnn" if is_jit else model_config.model_type
    n_ch = getattr(model_config, "n_channels", 1)
    input_features = getattr(model_config, "input_features", DEFAULT_INPUT_FEATURES)
    window_len = int(getattr(model_config, "window_samples", DEFAULT_WINDOW_SAMPLES))
    frame_times, probs = predict_sliding_probs(
        model,
        df,
        infer_type,
        is_jit=is_jit,
        batch_size=model_config.batch_size,
        window_len=window_len,
        step=step,
        device=model_config.device,
        input_features=input_features,
        n_channels=n_ch,
    )

    session_start = float(df["time"].iloc[0]) if len(df) else 0.0
    actuations = probs_to_actuations(
        frame_times,
        probs,
        params,
        session_start_time=session_start,
    )

    positive_frames = sum(
        1
        for row in probs
        if row[1] >= params.activation_threshold or row[2] >= params.activation_threshold
    )

    meta = {
        "detection_source": "live_postprocessing",
        "n_inference_frames": int(len(probs)),
        "n_positive_frames": int(positive_frames),
        "n_actuations": len(actuations),
        "inference_step_samples": step,
        "window_samples": window_len,
        "inference_interval_sec": params.inference_interval_sec,
        "activation_threshold": params.activation_threshold,
        "activation_seconds": params.activation_seconds,
        "seconds_since_last_treatment": params.seconds_since_last_treatment,
        "input_features": list(input_features),
    }
    return actuations, meta


# Legacy helpers (scoring path + tests)
def scoring_detections_from_onsets(onset_times) -> List[Dict[str, Any]]:
    return [{"time": float(start), "source": "scoring"} for start, _ in onset_times]


def run_model_detections(
    df: pd.DataFrame,
    model: Any,
    model_config: Any,
    is_jit: bool,
    sample_rate: int,
    prediction_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Backward-compatible alias: returns detections derived from actuations."""
    actuations, meta = run_live_model_actuations(
        df, model, model_config, is_jit, sample_rate, prediction_config
    )
    return actuations_to_detection_list(actuations), meta
