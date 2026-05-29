"""S3 TorchScript engines (.ptl) — list, resolve, and load."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.config import BUCKET_BY_SOURCE, DataSource
from backend.s3_client import TestbedS3Client
from backend.temp_cleanup import ptl_cache_dir, remove_download_file
from backend.types import NightRef

_ENGINE_KEY_PATTERN = re.compile(
    r"engines/(?P<dataset>[^/]+)/(?P<signal>[^/]+)/(?P<basename>[^/]+)\.ptl$"
)


@dataclass(frozen=True)
class EngineRef:
    """Reference to one .ptl file in S3."""

    s3_key: str
    bucket: str
    dataset: str
    signal: str
    night_id: str  # basename without .ptl

    @property
    def label(self) -> str:
        return f"{self.night_id} ({self.signal})"


def engine_s3_key(ref: NightRef) -> str:
    """engines/{dataset}/{signal}/{dataset}_{signal}_subj{X}_sess{Y}.ptl"""
    return f"engines/{ref.dataset}/{ref.signal}/{ref.file_format_data}.ptl"


def list_s3_engines(
    source: DataSource,
    dataset: Optional[str] = None,
    signal: Optional[str] = None,
    s3: Optional[TestbedS3Client] = None,
) -> List[EngineRef]:
    """List .ptl engines in the bucket for a data source."""
    client = s3 or TestbedS3Client(BUCKET_BY_SOURCE[source])
    prefix = "engines/"
    if dataset and signal:
        prefix = f"engines/{dataset}/{signal}/"
    elif dataset:
        prefix = f"engines/{dataset}/"

    keys = client.list_keys(prefix, suffix=".ptl")
    refs: List[EngineRef] = []
    for key in sorted(keys):
        m = _ENGINE_KEY_PATTERN.match(key)
        if not m:
            continue
        g = m.groupdict()
        if dataset and g["dataset"] != dataset:
            continue
        if signal and g["signal"] != signal:
            continue
        refs.append(
            EngineRef(
                s3_key=key,
                bucket=client.bucket_name,
                dataset=g["dataset"],
                signal=g["signal"],
                night_id=g["basename"],
            )
        )
    return refs


def engine_for_night(
    ref: NightRef,
    s3: Optional[TestbedS3Client] = None,
) -> Optional[EngineRef]:
    """Return EngineRef if the matching .ptl exists for this night."""
    client = s3 or TestbedS3Client(ref.bucket)
    key = engine_s3_key(ref)
    try:
        client.client.head_object(Bucket=ref.bucket, Key=key)
    except Exception:
        return None
    return EngineRef(
        s3_key=key,
        bucket=ref.bucket,
        dataset=ref.dataset,
        signal=ref.signal,
        night_id=ref.night_id,
    )


def download_ptl(
    engine: EngineRef,
    s3: Optional[TestbedS3Client] = None,
    temp_dir: Optional[str] = None,
) -> Path:
    """Download .ptl to a temp file and return local path."""
    client = s3 or TestbedS3Client(engine.bucket)
    base = Path(temp_dir) if temp_dir else ptl_cache_dir()
    local = base / engine.s3_key.replace("/", "_")
    client.download_file(engine.s3_key, str(local), bucket_name=engine.bucket)
    return local


def load_ptl(
    engine: EngineRef,
    device: str = "cpu",
    s3: Optional[TestbedS3Client] = None,
    temp_dir: Optional[str] = None,
):
    """Load TorchScript mobile model from S3 (CPU / float32 inference)."""
    import torch

    # Mobile lite interpreters with XNNPACK expect CPU execution
    _ = device
    cache_root = Path(temp_dir) if temp_dir else ptl_cache_dir()
    local = download_ptl(engine, s3=s3, temp_dir=temp_dir)
    try:
        model = torch.jit.load(str(local), map_location="cpu")
        model.eval()
        return model, str(local)
    finally:
        remove_download_file(local, temp_root=cache_root)


def load_ptl_by_key(
    bucket: str,
    s3_key: str,
    device: str = "cpu",
    s3: Optional[TestbedS3Client] = None,
):
    """Load a .ptl when you already know bucket and key."""
    import torch

    _ = device
    client = s3 or TestbedS3Client(bucket)
    local = ptl_cache_dir() / s3_key.replace("/", "_")
    client.download_file(s3_key, str(local), bucket_name=bucket)
    try:
        model = torch.jit.load(str(local), map_location="cpu")
        model.eval()
        return model, str(local)
    finally:
        remove_download_file(local, temp_root=ptl_cache_dir())
