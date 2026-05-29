"""Load and normalize PSG expert annotations."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from backend.config import DataSource
from backend.s3_client import TestbedS3Client
from backend.types import NightRef


EXPERT_COLUMNS = ("start", "duration", "apnea_type")


def expert_key_for_ref(ref: NightRef) -> str:
    return ref.expert_annotation_key


def load_expert_annotations(
    ref: NightRef,
    s3: Optional[TestbedS3Client] = None,
    apnea_types: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load expert annotations for a PSG night.

    Path: expert_annotations/{dataset}/{dataset}_subj{X}_sess{Y}.csv
    Returns None if missing or source is Zephyr without annotations.
    """
    if ref.source == DataSource.ZEPHYR:
        # Zephyr may have expert files under dataset name; still attempt load
        pass

    client = s3 or TestbedS3Client(ref.bucket)
    key = expert_key_for_ref(ref)
    print(f"[FILE READ] expert annotations: {key}")
    try:
        df, _ = client.get_file_df(key)
    except Exception as exc:
        print(f"[FILE READ] expert annotations missing or failed: {key} ({exc})")
        return None

    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in EXPERT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Expert annotation missing columns: {missing}")

    df = df.sort_values("start").reset_index(drop=True)
    if apnea_types:
        df = df[df["apnea_type"].isin(apnea_types)].reset_index(drop=True)
    return df


def expert_events_to_records(df: pd.DataFrame) -> List[dict]:
    """Convert annotation dataframe to list of event dicts for evaluation."""
    return [
        {
            "start": float(row["start"]),
            "duration": float(row["duration"]),
            "end": float(row["start"]) + float(row["duration"]),
            "apnea_type": row["apnea_type"],
        }
        for _, row in df.iterrows()
    ]
