"""Evaluation: validation metrics, expert comparison, simulated efficacy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.config import DEFAULT_EXPERT_MATCH_TOLERANCE_SEC
from backend.live_actuation import LiveActuationParams
from backend.metrics import classification_metrics


@dataclass
class StimulationParams:
    """
    Simulated haptic response on SpO2 given actuation (or onset) times.

    Built from the same prediction config as live post-processing so efficacy
    tracks threshold/debounce/cooldown (via which detections exist) and
    haptic duration / inference cadence (via recovery window shape).
    """

    activation_delay_sec: float = 0.0
    recovery_sec: float = 30.0
    spo2_recovery_fraction: float = 0.5
    min_drop_to_count_prevented: float = 3.0
    haptic_duration_sec: float = 1.0
    # Traceability / UI (mirror live post-processing knobs)
    activation_threshold: float = 0.7
    activation_seconds: int = 3
    inference_interval_sec: float = 1.0
    seconds_since_last_treatment: float = 5.0
    efficacy_trailing_observation_sec: float = 25.0
    scale_gain_by_confidence: bool = True
    postprocessing: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prediction_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        *,
        detection_at_actuation_time: bool = True,
    ) -> "StimulationParams":
        """
        Map prediction config to efficacy simulation.

        Args:
            detection_at_actuation_time: True when ``time`` is live actuation fire
                (debounce/cooldown already applied). False for scoring onsets.
        """
        c = config or {}
        live = LiveActuationParams.from_prediction_config(c)
        debounce_sec = live.activation_seconds * live.inference_interval_sec
        haptic_sec = live.haptic_duration_ms / 1000.0
        trailing = float(c.get("efficacy_trailing_observation_sec", 25.0))
        recovery_sec = debounce_sec + haptic_sec + trailing

        delay = 0.0
        if not detection_at_actuation_time:
            # Scoring detections: onset time → approximate time-to-actuation under same debounce
            delay = debounce_sec

        return cls(
            activation_delay_sec=delay,
            recovery_sec=recovery_sec,
            spo2_recovery_fraction=float(c.get("efficacy_spo2_recovery_fraction", 0.5)),
            min_drop_to_count_prevented=float(c.get("efficacy_min_drop_to_count_prevented", 3.0)),
            haptic_duration_sec=haptic_sec,
            activation_threshold=live.activation_threshold,
            activation_seconds=live.activation_seconds,
            inference_interval_sec=live.inference_interval_sec,
            seconds_since_last_treatment=live.seconds_since_last_treatment,
            efficacy_trailing_observation_sec=trailing,
            scale_gain_by_confidence=bool(c.get("efficacy_scale_gain_by_confidence", True)),
            postprocessing={
                "activation_threshold": live.activation_threshold,
                "activation_seconds": live.activation_seconds,
                "seconds_since_last_treatment": live.seconds_since_last_treatment,
                "inference_interval_sec": live.inference_interval_sec,
                "haptic_duration_ms": live.haptic_duration_ms,
                "actuation_start_delay_min": live.actuation_start_delay_min,
                "debounce_sec": debounce_sec,
                "recovery_sec": recovery_sec,
            },
        )


class Evaluator:
    """MVP evaluation methods from the implementation plan."""

    @staticmethod
    def eval_validation_split(
        predictions: Sequence[int],
        labels: Sequence[int],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Standard sklearn-style metrics on held-out windows."""
        return classification_metrics(predictions, labels, class_names=class_names)

    @staticmethod
    def eval_expert_annotations(
        detections: Sequence[Dict[str, float]],
        expert_events: Sequence[Dict[str, float]],
        tolerance_sec: float = DEFAULT_EXPERT_MATCH_TOLERANCE_SEC,
        detection_time_key: str = "time",
        expert_start_key: str = "start",
        expert_end_key: str = "end",
    ) -> Dict[str, Any]:
        """
        Match detections to expert events if detection falls within tolerance of event span.
        """
        matched = 0
        missed = len(expert_events)
        extra = 0
        matched_types: List[str] = []

        remaining = list(expert_events)
        for det in detections:
            t = float(det[detection_time_key])
            found = False
            for j, ev in enumerate(remaining):
                start = float(ev[expert_start_key]) - tolerance_sec
                end = float(ev.get(expert_end_key, ev[expert_start_key] + ev.get("duration", 0))) + tolerance_sec
                if start <= t <= end:
                    matched += 1
                    missed -= 1
                    matched_types.append(ev.get("apnea_type", "unknown"))
                    remaining.pop(j)
                    found = True
                    break
            if not found:
                extra += 1

        total_expert = len(expert_events)
        sensitivity = matched / total_expert if total_expert else 0.0
        ppv = matched / len(detections) if detections else 0.0

        return {
            "matched": matched,
            "missed": max(0, missed),
            "extra_positives": extra,
            "sensitivity": sensitivity,
            "precision": ppv,
            "total_expert_events": total_expert,
            "total_detections": len(detections),
            "matched_apnea_types": matched_types,
        }

    @staticmethod
    def _event_recovery_sec(det: Dict[str, Any], stim: StimulationParams) -> float:
        """Observation window after stim: debounce + haptic (+ optional per-event haptic)."""
        haptic_ms = det.get("haptic_duration_ms")
        if haptic_ms is not None:
            haptic_sec = float(haptic_ms) / 1000.0
        else:
            haptic_sec = stim.haptic_duration_sec
        debounce_sec = stim.activation_seconds * stim.inference_interval_sec
        return debounce_sec + haptic_sec + stim.efficacy_trailing_observation_sec

    @staticmethod
    def _event_improvement(
        det: Dict[str, Any],
        stim: StimulationParams,
        baseline: float,
        nadir: float,
    ) -> float:
        """Simulated SpO2 lift: fraction of (baseline − nadir) in the recovery window."""
        drop = baseline - nadir
        if drop <= 0:
            return 0.0
        fraction = stim.spo2_recovery_fraction
        if stim.scale_gain_by_confidence:
            conf = det.get("confidence")
            if conf is not None:
                fraction *= 0.5 + 0.5 * float(conf)
        return drop * fraction

    @staticmethod
    def eval_simulated_efficacy(
        detections: Sequence[Dict[str, float]],
        spo2_df: pd.DataFrame,
        stimulation: Optional[StimulationParams] = None,
        time_col: str = "time",
        spo2_col: str = "oxygen",
    ) -> Dict[str, Any]:
        """
        Estimate SpO2 benefit from haptic actuations using post-processing-aligned timing.

        Detection times should already reflect live debounce/cooldown when sourced from
        model actuations. Recovery window length scales with activation_seconds,
        inference_interval_sec, haptic_duration, and efficacy_trailing_observation_sec.
        """
        stim = stimulation or StimulationParams()
        if spo2_df.empty or not detections:
            return {
                "prevented_desaturations": 0,
                "mean_spo2_improvement": 0.0,
                "simulated_events": 0,
                "model": "instant_gain",
                "stimulation_params": stim.__dict__,
                "postprocessing": stim.postprocessing,
            }

        spo2 = spo2_df[[time_col, spo2_col]].dropna().copy()
        times = spo2[time_col].values.astype(float)
        values = spo2[spo2_col].values.astype(float)
        baseline = float(np.median(values))

        improvements = []
        prevented = 0

        for det in detections:
            t_det = float(det.get("time", det.get("start", 0))) + stim.activation_delay_sec
            recovery = Evaluator._event_recovery_sec(det, stim)
            window_end = t_det + recovery
            mask = (times >= t_det) & (times <= window_end)
            if not mask.any():
                continue
            segment = values[mask]
            nadir = float(segment.min())
            drop = baseline - nadir
            if drop >= stim.min_drop_to_count_prevented:
                prevented += 1
            improvements.append(Evaluator._event_improvement(det, stim, baseline, nadir))

        return {
            "prevented_desaturations": prevented,
            "mean_spo2_improvement": float(np.mean(improvements)) if improvements else 0.0,
            "simulated_events": len(detections),
            "model": "instant_gain",
            "stimulation_params": stim.__dict__,
            "postprocessing": stim.postprocessing,
            "mean_recovery_sec": float(
                np.mean([Evaluator._event_recovery_sec(d, stim) for d in detections])
            )
            if detections
            else 0.0,
        }
