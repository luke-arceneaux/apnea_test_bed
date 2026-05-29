"""Timed checkpoints for analysis runs (console + optional UI callback)."""

from __future__ import annotations

import time
from typing import Callable, List, Optional


class RunStatus:
    def __init__(self, callback: Optional[Callable[[str], None]] = None) -> None:
        self._callback = callback
        self._run_start = time.perf_counter()
        self._step_start = self._run_start
        self.lines: List[str] = []

    def checkpoint(self, step: str, detail: str = "") -> None:
        now = time.perf_counter()
        step_sec = now - self._step_start
        total_sec = now - self._run_start
        label = step if not detail else f"{step} - {detail}"
        msg = f"[{total_sec:6.1f}s  +{step_sec:5.1f}s] {label}"
        print(msg, flush=True)
        self.lines.append(msg)
        if self._callback:
            self._callback(msg)
        self._step_start = now


def model_step_label(mode: str) -> str:
    return {
        "none": "Scoring",
        "train": "Training",
        "s3_ptl": "Model load & inference",
    }.get(mode, "Model")
