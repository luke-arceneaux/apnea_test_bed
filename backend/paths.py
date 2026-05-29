"""Parse S3 keys and build path templates."""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from backend.config import (
    DataSource,
    PSG_DATASET_FOLDER,
    PSG_NAF_SIGNAL,
    PSG_SPO2_SIGNAL,
    ZEPHYR_GRAVITY_SIGNAL,
    ZEPHYR_NAF_SIGNAL,
    ZEPHYR_SPO2_SIGNAL,
)
from backend.types import NightRef

_FILE_PATTERN = re.compile(
    r"(?P<dataset>[^/_]+)_(?P<signal>[^/_]+)_subj(?P<subject>[^/_]+)_sess(?P<session>[^/.]+)"
)


def parse_night_key(key: str, source: DataSource) -> Optional[NightRef]:
    """
    Parse a raw-data S3 key into a NightRef.

    Examples:
      zephyr/stereo/zephyr_stereo_subj12_sess20240101120000.wav
      raw_data/scidb/cannula/scidb_cannula_subj1447_sess1.csv
      raw_data/mesa-commercial-use/Flow/mesa_Flow_subj1_sess2.csv
    """
    basename = os.path.splitext(os.path.basename(key))[0]
    match = _FILE_PATTERN.match(basename)
    if not match:
        return None
    groups = match.groupdict()
    from backend.config import BUCKET_BY_SOURCE

    return NightRef(
        source=source,
        dataset=groups["dataset"],
        signal=groups["signal"],
        subject=groups["subject"],
        session=groups["session"],
        bucket=BUCKET_BY_SOURCE[source],
        primary_key=key,
    )


def raw_data_key(
    source: DataSource,
    dataset: str,
    subject: str,
    session: str,
    signal: str,
    ext: str = "csv",
) -> str:
    """
    Build an S3 object key for raw night data.

    Layout:
      Zephyr: zephyr/{signal}/{dataset}_{signal}_subj{S}_sess{T}.{ext}
      PSG:    raw_data/{dataset-folder}/{signal}/{dataset}_{signal}_subj{S}_sess{T}.csv

    The ``signal`` segment is the folder name (stereo, oxygen, cannula, Flow, spo2, ...).
    """
    ext = ext.lstrip(".")
    basename = f"{dataset}_{signal}_subj{subject}_sess{session}.{ext}"
    if source == DataSource.ZEPHYR:
        return f"zephyr/{signal}/{basename}"
    folder = PSG_DATASET_FOLDER[source]
    return f"raw_data/{folder}/{signal}/{basename}"


def raw_data_key_from_ref(ref: NightRef, signal: str, ext: Optional[str] = None) -> str:
    """Build a companion or alternate-signal key for the same night as ``ref``."""
    if ext is None:
        _, primary_ext = os.path.splitext(ref.primary_key)
        psg_spo2_signals = tuple(s for s in PSG_SPO2_SIGNAL.values() if s)
        ext = "csv" if signal in (ZEPHYR_SPO2_SIGNAL, ZEPHYR_GRAVITY_SIGNAL, *psg_spo2_signals) else primary_ext.lstrip(".")
    return raw_data_key(ref.source, ref.dataset, ref.subject, ref.session, signal, ext)


def naf_signal_for_source(source: DataSource) -> str:
    """Primary NAF/audio signal folder for listing and loading."""
    if source == DataSource.ZEPHYR:
        return ZEPHYR_NAF_SIGNAL
    return PSG_NAF_SIGNAL[source]


def spo2_signal_for_source(source: DataSource) -> Optional[str]:
    """SpO2 folder signal name, or None when SpO2 is embedded in the NAF file (SCIDB)."""
    if source == DataSource.ZEPHYR:
        return ZEPHYR_SPO2_SIGNAL
    return PSG_SPO2_SIGNAL.get(source)


def gravity_signal_for_source(source: DataSource) -> Optional[str]:
    if source == DataSource.ZEPHYR:
        return ZEPHYR_GRAVITY_SIGNAL
    return None


def psg_raw_prefix(source: DataSource, signal: str) -> str:
    folder = PSG_DATASET_FOLDER[source]
    return f"raw_data/{folder}/{signal}/"


def zephyr_raw_prefix(signal: str) -> str:
    return f"zephyr/{signal}/"


def listable_signals(source: DataSource) -> Tuple[str, ...]:
    """NAF signals used when listing available nights (one entry point per night)."""
    return (naf_signal_for_source(source),)


def zephyr_companion_key(stereo_key: str, companion_signal: str) -> str:
    """
    Deprecated: use raw_data_key_from_ref. Kept for compatibility.

    Rebuilds companion paths using dataset/subject/session from the primary key
    (replaces folder + filename signal segment, not only the directory).
    """
    ref = parse_night_key(stereo_key, DataSource.ZEPHYR)
    if ref is None:
        # Fallback: pipeline-style string replace
        path = stereo_key.replace("temp/", "")
        path = path.replace("/stereo/", f"/{companion_signal}/")
        path = path.replace("stereo", companion_signal)
        return path.replace(".wav", ".csv").replace(".WAV", ".csv")
    return raw_data_key_from_ref(ref, companion_signal, ext="csv")
