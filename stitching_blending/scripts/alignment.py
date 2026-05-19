"""Pairwise alignment with phase correlation and NCC fallback."""

from __future__ import annotations

from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd

try:  # Support both package and direct script execution contexts.
    from .overlap import NeighborPair, extract_overlap_strips
except ImportError:  # pragma: no cover
    from overlap import NeighborPair, extract_overlap_strips


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image.mean(axis=2)
    result = image.astype(np.float32, copy=False)
    result = result - float(np.mean(result))
    std = float(np.std(result))
    if std > 1e-6:
        result = result / std
    return result


def phase_correlation_shift(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Estimate target shift relative to source with Fourier phase correlation."""
    source_f = _to_gray_float(source)
    target_f = _to_gray_float(target)
    window = cv2.createHanningWindow((source_f.shape[1], source_f.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(source_f, target_f, window)
    return float(dx), float(dy), float(response)


def ncc_shift(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_shift_px: int = 10,
) -> tuple[float, float, float]:
    """Estimate target shift relative to source by exhaustive normalized NCC.

    The search evaluates integer shifts in ``[-max_shift_px, max_shift_px]``.
    For each shift, the overlapping source/target crop is compared with
    ``TM_CCOEFF_NORMED``. This is slower than phase correlation, but robust for
    small controlled fallbacks and avoids adding another dependency.
    """
    source_f = _to_gray_float(source)
    target_f = _to_gray_float(target)
    best_score = -np.inf
    best_shift = (0, 0)
    h, w = source_f.shape

    for dy in range(-max_shift_px, max_shift_px + 1):
        for dx in range(-max_shift_px, max_shift_px + 1):
            src_y0 = max(0, dy)
            src_y1 = min(h, h + dy)
            tgt_y0 = max(0, -dy)
            tgt_y1 = min(h, h - dy)
            src_x0 = max(0, dx)
            src_x1 = min(w, w + dx)
            tgt_x0 = max(0, -dx)
            tgt_x1 = min(w, w - dx)
            if src_y1 - src_y0 < 4 or src_x1 - src_x0 < 4:
                continue
            a = source_f[src_y0:src_y1, src_x0:src_x1]
            b = target_f[tgt_y0:tgt_y1, tgt_x0:tgt_x1]
            score = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_score = score
                best_shift = (dx, dy)

    if not np.isfinite(best_score):
        return 0.0, 0.0, 0.0
    # The crop loop shifts target coordinates against source coordinates; invert
    # the winning offset so NCC reports the same target-relative shift convention
    # as cv2.phaseCorrelate.
    return float(-best_shift[0]), float(-best_shift[1]), float(best_score)


def estimate_shift(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_shift_px: int = 10,
    phase_confidence_threshold: float = 0.08,
    ncc_acceptance_threshold: float = 0.3,
) -> dict[str, float | str | bool]:
    """Estimate a pairwise shift and choose NCC when phase confidence is low."""
    dx, dy, confidence = phase_correlation_shift(source, target)
    method = 'phase_correlation'
    acceptance_threshold = phase_confidence_threshold
    if confidence < phase_confidence_threshold or abs(dx) > max_shift_px or abs(dy) > max_shift_px:
        dx, dy, confidence = ncc_shift(source, target, max_shift_px=max_shift_px)
        method = 'ncc'
        acceptance_threshold = ncc_acceptance_threshold

    accepted = bool(confidence >= acceptance_threshold and abs(dx) <= max_shift_px and abs(dy) <= max_shift_px)
    return {
        'dx_px': float(dx),
        'dy_px': float(dy),
        'confidence': float(confidence),
        'method': method,
        'accepted': accepted,
    }


def align_neighbor_pairs(
    tiles: dict[str, np.ndarray],
    pairs: list[NeighborPair],
    *,
    max_shift_px: int = 10,
    phase_confidence_threshold: float = 0.08,
) -> pd.DataFrame:
    """Estimate pairwise neighbor displacements for a tile graph."""
    rows: list[dict[str, object]] = []
    for pair in pairs:
        source = tiles[pair.source_tile_id]
        target = tiles[pair.target_tile_id]
        strips = extract_overlap_strips(source, target, pair)
        if strips is None:
            estimate = {'dx_px': 0.0, 'dy_px': 0.0, 'confidence': 0.0, 'method': 'none', 'accepted': False}
        else:
            estimate = estimate_shift(
                strips[0],
                strips[1],
                max_shift_px=max_shift_px,
                phase_confidence_threshold=phase_confidence_threshold,
            )

        rows.append(
            {
                **asdict(pair),
                'dx_px': float(pair.expected_dx_px + float(estimate['dx_px'])),
                'dy_px': float(pair.expected_dy_px + float(estimate['dy_px'])),
                'shift_x_px': float(estimate['dx_px']),
                'shift_y_px': float(estimate['dy_px']),
                'confidence': float(estimate['confidence']),
                'method': str(estimate['method']),
                'accepted': bool(estimate['accepted']),
            }
        )
    return pd.DataFrame.from_records(rows)
