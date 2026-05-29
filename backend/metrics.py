"""Metric helpers for audio, SpO2, and classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def classification_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    y_pred = np.asarray(predictions)
    y_true = np.asarray(labels)
    labels_sorted = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_sorted,
        target_names=class_names or [str(c) for c in labels_sorted],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_sorted).tolist(),
        "classification_report": report,
        "n_samples": int(len(y_true)),
    }


def calculate_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias used by package __init__."""
    return classification_metrics(predictions, labels, **kwargs)


def calculate_audio_metrics(df: pd.DataFrame, value_col: str = "value") -> Dict[str, float]:
    if df.empty or value_col not in df.columns:
        return {}
    v = df[value_col].astype(float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def calculate_spo2_metrics(df: pd.DataFrame, spo2_col: str = "oxygen") -> Dict[str, float]:
    if df.empty or spo2_col not in df.columns:
        return {}
    s = df[spo2_col].dropna().astype(float)
    return {
        "mean_spo2": float(s.mean()),
        "min_spo2": float(s.min()),
        "pct_below_90": float((s < 90).mean() * 100),
        "pct_below_85": float((s < 85).mean() * 100),
    }
