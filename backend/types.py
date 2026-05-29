"""Shared datatypes for loaded nights, windows, and evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import DEFAULT_WINDOW_SAMPLES, DataSource


@dataclass(frozen=True)
class NightRef:
    """Reference to one processable night in S3."""

    source: DataSource
    dataset: str
    signal: str
    subject: str
    session: str
    bucket: str
    primary_key: str

    @property
    def night_id(self) -> str:
        return f"{self.dataset}_{self.signal}_subj{self.subject}_sess{self.session}"

    @property
    def file_format_data(self) -> str:
        return self.night_id

    @property
    def expert_annotation_key(self) -> str:
        return f"expert_annotations/{self.dataset}/{self.dataset}_subj{self.subject}_sess{self.session}.csv"


@dataclass
class NightData:
    """Aligned night-level data returned by load_night_data."""

    ref: NightRef
    audio: pd.DataFrame
    spo2: pd.DataFrame
    position: Optional[pd.DataFrame] = None
    sample_rate: int = 8
    total_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingWindow:
    label: int  # 0=non-apnea, 1=onset-apnea, 2=flatline
    start_time: float
    sequence: Any  # numpy array in practice


@dataclass
class WindowGenerationParams:
    """Configurable collect_data-style parameters."""

    sec_before: int = 10
    sec_after: int = 5
    min_event_duration: float = 10.0
    flatline_min_duration: float = 15.0
    flatline_window_sec: float = 15.0
    window_samples: int = DEFAULT_WINDOW_SAMPLES
    # non-apnea gap before next event (source-specific defaults applied in generator)
    non_apnea_end_offset: int = 1
    non_apnea_start_offset: int = 16
    skip_last_flatline: bool = False
    balance_classes: bool = True
