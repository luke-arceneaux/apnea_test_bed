"""Remove temporary S3 downloads (project temp/ and OS .ptl cache)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

DEFAULT_PROJECT_TEMP = Path("temp")
PTL_CACHE_DIRNAME = "testbed_engines"


def ptl_cache_dir(base: Optional[PathLike] = None) -> Path:
    root = Path(base) if base is not None else Path(tempfile.gettempdir())
    return root / PTL_CACHE_DIRNAME


def remove_download_file(path: PathLike, *, temp_root: Optional[PathLike] = None) -> bool:
    """
    Delete one downloaded file and prune empty parent directories up to temp_root.

    If temp_root is omitted, parents are pruned until removal fails (e.g. non-empty).
    """
    target = Path(path)
    if not target.exists():
        return False

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()

    root = Path(temp_root).resolve() if temp_root is not None else None
    parent = target.parent
    while parent != parent.parent:
        if root is not None:
            try:
                parent.resolve().relative_to(root)
            except ValueError:
                break
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def cleanup_project_temp(temp_dir: PathLike = DEFAULT_PROJECT_TEMP) -> bool:
    """Remove the entire project temp/ tree if it exists."""
    root = Path(temp_dir)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def cleanup_ptl_cache(base: Optional[PathLike] = None) -> bool:
    """Remove {system_temp}/testbed_engines/ and all cached .ptl downloads."""
    cache = ptl_cache_dir(base)
    if not cache.exists():
        return False
    shutil.rmtree(cache)
    return True


def cleanup_run_artifacts(
    *,
    temp_dir: PathLike = DEFAULT_PROJECT_TEMP,
    clear_project_temp: bool = True,
    clear_ptl_cache: bool = True,
) -> dict:
    """
    Clean temp artifacts after an analysis run.

    WAV files are normally removed immediately after decode; this clears any
    leftovers plus the OS-level .ptl cache.
    """
    return {
        "project_temp": cleanup_project_temp(temp_dir) if clear_project_temp else False,
        "ptl_cache": cleanup_ptl_cache() if clear_ptl_cache else False,
    }
