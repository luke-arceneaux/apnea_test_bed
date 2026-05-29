"""Multi-source night listing and loading (Zephyr + PSG)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from backend.config import (
    BUCKET_BY_SOURCE,
    DataSource,
    DEFAULT_SAMPLE_RATE,
    PSG_SPO2_OXYGEN_COLUMN,
)
from backend.paths import (
    gravity_signal_for_source,
    listable_signals,
    naf_signal_for_source,
    parse_night_key,
    psg_raw_prefix,
    raw_data_key_from_ref,
    spo2_signal_for_source,
    zephyr_raw_prefix,
)
from backend.s3_client import TestbedS3Client
from backend.time_columns import (
    align_time_for_merge,
    ensure_time_column,
    print_dataframe_head,
    prune_companion_columns,
    resample_to_8hz,
)
from backend.temp_cleanup import remove_download_file
from backend.types import NightData, NightRef

# Optional: reuse pipeline preprocessing when available
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent / "data_pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))


def _get_s3(source: DataSource) -> TestbedS3Client:
    return TestbedS3Client(BUCKET_BY_SOURCE[source])


def list_available_nights(
    source: Union[DataSource, str],
    signal: Optional[str] = None,
    subject: Optional[str] = None,
    s3: Optional[TestbedS3Client] = None,
) -> List[NightRef]:
    """
    List nights available in S3 for a data source.

    Args:
        source: zephyr | scidb | mesa
        signal: optional filter (stereo, cannula, Flow, ...)
        subject: optional filter on subject id substring
    """
    src = DataSource(source) if isinstance(source, str) else source
    client = s3 or _get_s3(src)
    refs: List[NightRef] = []

    if src == DataSource.ZEPHYR:
        naf = signal or naf_signal_for_source(src)
        prefix = zephyr_raw_prefix(naf)
        keys = client.list_keys(prefix, suffix=".wav") + client.list_keys(prefix, suffix=".csv")
    else:
        naf = signal or naf_signal_for_source(src)
        keys = client.list_keys(psg_raw_prefix(src, naf), suffix=".csv")

    seen = set()
    for key in sorted(keys):
        ref = parse_night_key(key, src)
        if ref is None:
            continue
        if subject and subject not in ref.subject:
            continue
        if signal and ref.signal != signal:
            continue
        if ref.night_id in seen:
            continue
        seen.add(ref.night_id)
        refs.append(ref)

    return refs


def load_night_data(
    ref: NightRef,
    s3: Optional[TestbedS3Client] = None,
    temp_dir: str = "temp/",
    download_wav: bool = True,
) -> NightData:
    """
    Load one night: audio/NAF, SpO2, and optional position (Zephyr).

    Zephyr: zephyr/stereo + zephyr/oxygen + zephyr/gravity (signal in filename).
    SCIDB: raw_data/scidb/cannula — SpO2 column in same file as NAF.
    MESA: raw_data/mesa-commercial-use/Flow + separate spo2 CSV.
    """
    client = s3 or _get_s3(ref.source)
    print(f"[FILE READ] load_night_data: {ref.night_id} ({ref.source.value}) primary={ref.primary_key}")

    if ref.source == DataSource.ZEPHYR:
        return _load_zephyr_night(ref, client, temp_dir, download_wav)
    return _load_psg_night(ref, client)


def _load_zephyr_night(
    ref: NightRef,
    client: TestbedS3Client,
    temp_dir: str,
    download_wav: bool,
) -> NightData:
    key = ref.primary_key
    if key.endswith(".wav") and download_wav:
        local = client.download_file(key, os.path.join(temp_dir, key))
        print(f"[FILE READ] local WAV decode: {local}")
        try:
            from utils import wav2dfprocess_stereo  # type: ignore

            audio_df, _, _ = wav2dfprocess_stereo(local, downsample_step_size=1000)
        except ImportError:
            raise ImportError(
                "WAV loading requires data_pipeline.utils.wav2dfprocess_stereo on PYTHONPATH."
            )
        finally:
            remove_download_file(local, temp_root=temp_dir)
    else:
        audio_df, _ = client.get_file_df(key)

    audio_df = _normalize_audio_df(audio_df)
    print_dataframe_head("zephyr/audio", audio_df)

    spo2_sig = spo2_signal_for_source(ref.source)
    grav_sig = gravity_signal_for_source(ref.source)
    assert spo2_sig and grav_sig  # Zephyr only
    oxygen_key = raw_data_key_from_ref(ref, spo2_sig, ext="csv")
    gravity_key = raw_data_key_from_ref(ref, grav_sig, ext="csv")
    spo2_df = _load_optional_csv(client, oxygen_key)
    position_df = _load_optional_csv(client, gravity_key)

    # Match apnea_detection_auto: keep audio / SpO2 / gravity separate until conditioning.
    spo2_out = _extract_spo2_frame(spo2_df)
    position_out = position_df.copy() if position_df is not None and not position_df.empty else None

    sample_rate = _infer_sample_rate(audio_df)
    total_sec = float(audio_df["time"].iloc[-1]) if len(audio_df) else 0.0

    return NightData(
        ref=ref,
        audio=audio_df[["time", "value"]].copy(),
        spo2=spo2_out,
        position=position_out,
        sample_rate=sample_rate,
        total_seconds=total_sec,
        metadata={"raw_audio_rows": len(audio_df)},
    )


def _load_psg_night(ref: NightRef, client: TestbedS3Client) -> NightData:
    df, _ = client.get_file_df(ref.primary_key)
    df = _normalize_audio_df(df)
    print_dataframe_head(f"psg/audio/{ref.primary_key}", df)

    if "time" not in df.columns or "value" not in df.columns:
        raise ValueError(f"PSG file missing time/value columns: {ref.primary_key}")

    spo2_sig = spo2_signal_for_source(ref.source)
    if spo2_sig is None:
        # SCIDB: SpO2 is in the cannula CSV (``spo2`` -> ``oxygen``)
        spo2_df = _extract_spo2_frame(normalize_psg_spo2_to_oxygen(df, ref.source))
        metadata = {"columns": list(df.columns), "spo2_source": "embedded"}
    else:
        # MESA: NAF in Flow file; SpO2 in raw_data/.../spo2/
        spo2_key = raw_data_key_from_ref(ref, spo2_sig, ext="csv")
        spo2_raw = _load_optional_csv(client, spo2_key)
        spo2_df = (
            _extract_spo2_frame(normalize_psg_spo2_to_oxygen(spo2_raw, ref.source))
            if spo2_raw is not None
            else _empty_spo2_frame()
        )
        metadata = {"columns": list(df.columns), "spo2_key": spo2_key}

    sample_rate = _infer_sample_rate(df)
    total_sec = float(df["time"].iloc[-1]) if len(df) else 0.0

    return NightData(
        ref=ref,
        audio=df[["time", "value"]].copy(),
        spo2=spo2_df,
        position=None,
        sample_rate=sample_rate,
        total_seconds=total_sec,
        metadata=metadata,
    )


def _load_optional_csv(client: TestbedS3Client, key: str) -> Optional[pd.DataFrame]:
    print(f"[FILE READ] optional companion CSV: {key}")
    try:
        df, _ = client.get_file_df(key)
        df.columns = df.columns.str.strip().str.lower()
        print_dataframe_head(f"csv/raw/{key}", df)
        role = (
            "spo2"
            if "/oxygen/" in key or "/spo2/" in key
            else "gravity"
            if "/gravity/" in key
            else "companion"
        )
        psg_source = _psg_source_from_key(key)
        if psg_source is not None:
            df = normalize_psg_spo2_to_oxygen(df, psg_source)
        if role in ("spo2", "gravity"):
            df = prune_companion_columns(df, role)
        out = ensure_time_column(df, label=f"csv/{role}/{key}")
        print_dataframe_head(f"csv/ready/{key}", out)
        return out
    except Exception as exc:
        print(f"[FILE READ] optional companion CSV missing or failed: {key} ({exc})")
        return None


def _normalize_audio_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = ensure_time_column(df)
    return df.drop_duplicates(subset=["time"], keep="last")


def _psg_source_from_key(key: str) -> Optional[DataSource]:
    lower = key.lower()
    if "/scidb/" in lower:
        return DataSource.SCIDB
    if "/mesa-commercial-use/" in lower or "/mesa/" in lower:
        return DataSource.MESA
    return None


def normalize_psg_spo2_to_oxygen(df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
    """Rename PSG-specific SpO2 column to ``oxygen`` (SCIDB: spo2, MESA: value)."""
    raw_col = PSG_SPO2_OXYGEN_COLUMN.get(source)
    if raw_col is None:
        return df
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    if "oxygen" not in out.columns and raw_col in out.columns:
        out = out.rename(columns={raw_col: "oxygen"})
    return out


def _extract_spo2_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty or "oxygen" not in df.columns:
        return _empty_spo2_frame()
    cols = ["time", "oxygen"]
    for extra in ("heartrate", "confidence", "status"):
        if extra in df.columns:
            cols.append(extra)
    out = df[cols].dropna(subset=["oxygen"]).reset_index(drop=True)
    return out if not out.empty else _empty_spo2_frame()


def _empty_spo2_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "oxygen"])


def _merge_zephyr_frames(
    audio: pd.DataFrame,
    spo2: Optional[pd.DataFrame],
    position: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, int]:
    """Light merge aligned with pipeline (apnea_detection_auto: resample then outer merge on time)."""
    df = align_time_for_merge(audio, decimals=3)
    print_dataframe_head("merge/audio", df)

    if spo2 is not None and not spo2.empty and "oxygen" in spo2.columns:
        spo2 = spo2.dropna(subset=["oxygen"]).reset_index(drop=True)
        spo2 = ensure_time_column(spo2)
        if not spo2.empty:
            spo2 = resample_to_8hz(spo2, label="merge/spo2", role="spo2")
            spo2 = align_time_for_merge(spo2)
            print(f"[DEBUG time] merge/spo2: merging {len(spo2)} rows onto audio {len(df)} rows")
            df = df.merge(spo2, how="outer", on="time")

    if position is not None and not position.empty:
        position = ensure_time_column(position)
        if not position.empty:
            position = resample_to_8hz(position, label="merge/gravity", role="gravity")
            position = align_time_for_merge(position)
            print(f"[DEBUG time] merge/gravity: merging {len(position)} rows")
            df = df.merge(position, how="outer", on="time")

    sample_rate = _infer_sample_rate(df) if len(df) > 1 else DEFAULT_SAMPLE_RATE
    return df.sort_values("time").reset_index(drop=True), sample_rate


def _infer_sample_rate(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return DEFAULT_SAMPLE_RATE
    dt = df["time"].iloc[1] - df["time"].iloc[0]
    if dt <= 0:
        return DEFAULT_SAMPLE_RATE
    return max(1, int(round(1.0 / dt)))


def load_model(model_path: str, device: str = "cpu"):
    """
    Load a TorchScript .ptl from a local path (testing only).

    For S3 engines use backend.engines.load_ptl / load_ptl_by_key.
    """
    import torch

    path = Path(model_path)
    if path.suffix != ".ptl":
        raise ValueError("load_model expects a local .ptl path; use backend.engines for S3.")
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model, (1, 1, 120)
