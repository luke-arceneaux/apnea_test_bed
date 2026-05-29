"""
Live Zephyr post-inference actuation logic (mirrors DataManager / AudioProcessorCallbacks).

Reference: Android app — p_apnea OR p_flatline >= threshold, debounced frames,
cooldown strictly > seconds_since_last_treatment. Warmup (45 min) optional and off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class LiveActuationParams:
    """Parameters aligned with DataManager live haptic post-processing."""

    activation_threshold: float = 0.7
    activation_seconds: int = 3  # consecutive positive inference frames (~1 Hz)
    seconds_since_last_treatment: float = 5.0  # actuate only if elapsed > this value
    inference_interval_sec: float = 1.0
    haptic_duration_ms: int = 1000
    actuation_start_delay_min: float = 0.0  # 0 = disabled (test bed default)

    @classmethod
    def from_prediction_config(cls, config: Optional[Dict[str, Any]] = None) -> "LiveActuationParams":
        c = config or {}
        return cls(
            activation_threshold=float(c.get("activation_threshold", 0.7)),
            activation_seconds=int(c.get("activation_seconds", c.get("activation_time", 3))),
            seconds_since_last_treatment=float(
                c.get("seconds_since_last_treatment", c.get("refractory_time", 5))
            ),
            inference_interval_sec=float(c.get("inference_interval_sec", 1.0)),
            haptic_duration_ms=int(c.get("haptic_duration", 1000)),
            actuation_start_delay_min=float(c.get("actuation_start_delay_min", 0)),
        )


def is_positive_frame(p_apnea: float, p_flatline: float, threshold: float) -> bool:
    """Live app: (p_apnea >= threshold) OR (p_flatline >= threshold); p_non_apnea ignored."""
    return p_apnea >= threshold or p_flatline >= threshold


def probs_to_actuations(
    frame_times: np.ndarray,
    probs: np.ndarray,
    params: LiveActuationParams,
    *,
    session_start_time: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert per-frame model outputs to actuation events (haptic would fire).

    Args:
        frame_times: time at each inference tick (~1 Hz)
        probs: (n, 3) as [p_non_apnea, p_apnea, p_flatline]
        session_start_time: recording start (seconds) for optional warmup gate
    """
    if len(frame_times) == 0 or len(probs) == 0:
        return []

    warmup_sec = params.actuation_start_delay_min * 60.0
    actuations: List[Dict[str, Any]] = []
    apnea_count = 0
    last_actuation_time: Optional[float] = None

    for i in range(len(probs)):
        t = float(frame_times[i])
        if warmup_sec > 0 and t < session_start_time + warmup_sec:
            continue

        p0, p1, p2 = float(probs[i, 0]), float(probs[i, 1]), float(probs[i, 2])

        if is_positive_frame(p1, p2, params.activation_threshold):
            apnea_count += 1
            if apnea_count >= params.activation_seconds:
                # Mirror app: reset counters before cooldown check
                apnea_count = 0

                allow = True
                if last_actuation_time is not None:
                    elapsed = t - last_actuation_time
                    allow = elapsed > params.seconds_since_last_treatment

                if allow:
                    confidence = max(p1, p2)
                    actuations.append({
                        "time": t,
                        "confidence": confidence,
                        "p_apnea": p1,
                        "p_flatline": p2,
                        "p_non_apnea": p0,
                        "source": "live_postprocessing",
                        "haptic_duration_ms": params.haptic_duration_ms,
                    })
                    last_actuation_time = t
                    apnea_count = 0
        else:
            apnea_count = 0

    return actuations


def actuations_to_detection_list(actuations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map actuations to detection dicts for expert match and simulated efficacy."""
    return [
        {
            "time": float(a["time"]),
            "source": a.get("source", "live_postprocessing"),
            "confidence": a.get("confidence"),
            "haptic_duration_ms": a.get("haptic_duration_ms"),
            "p_apnea": a.get("p_apnea"),
            "p_flatline": a.get("p_flatline"),
        }
        for a in actuations
    ]
