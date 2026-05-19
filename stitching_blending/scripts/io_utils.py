"""Image and table I/O utilities for the prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile


def read_image(path: str | Path) -> np.ndarray:
    """Read a TIFF or common image file without changing dtype."""
    image = tifffile.imread(Path(path))
    # BigStitcher example TIFFs store RGB/channel data as C,H,W. The rest of the
    # prototype expects H,W,C for multi-channel images, so normalize that common
    # microscopy layout at the I/O boundary.
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[1] > 16 and image.shape[2] > 16:
        image = np.moveaxis(image, 0, -1)
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image, preserving dtype for TIFF output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, image)


def write_preview_png(path: str | Path, image: np.ndarray) -> None:
    """Write a display-friendly 8-bit PNG preview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = image.astype(np.float32)
    finite = np.isfinite(preview)
    if not finite.any():
        preview = np.zeros_like(preview, dtype=np.uint8)
    else:
        lo = float(np.nanpercentile(preview[finite], 0.5))
        hi = float(np.nanpercentile(preview[finite], 99.5))
        if hi <= lo:
            hi = lo + 1.0
        preview = np.clip((preview - lo) / (hi - lo), 0.0, 1.0)
        preview = (preview * 255).astype(np.uint8)
    cv2.imwrite(str(path), preview)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV with parent-directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def image_dtype_range(dtype: np.dtype) -> tuple[float, float]:
    """Return the valid numeric range for an image dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        return 0.0, 1.0
    raise TypeError(f'Unsupported image dtype: {dtype}')


def cast_preserving_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Clip and cast a floating accumulator back to the requested dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        lo, hi = image_dtype_range(dtype)
        return np.clip(np.rint(image), lo, hi).astype(dtype)
    return image.astype(dtype, copy=False)
