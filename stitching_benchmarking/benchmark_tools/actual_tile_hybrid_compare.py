#!/usr/bin/env python3
"""Compare current LVP overlap stitching against a hybrid FFT fast path.

This script is intentionally self-contained under ``stitching_benchmarking``.
It reads local protocol output folders, runs:

1. current LVP overlap registration/blending, via ``stitch_registered_tiles``;
2. hybrid FFT registration, falling back per edge to current LVP local NCC when
   confidence checks fail;

and writes stitched TIFFs plus visual/report artifacts under
``stitching_benchmarking/actual_tile_runs`` by default.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pathlib
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

BENCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import modules.common_utils as common_utils  # noqa: E402
from modules.objectives_loader import ObjectiveLoader  # noqa: E402
from modules.protocol import Protocol  # noqa: E402
from modules.protocol_execution_record import ProtocolExecutionRecord  # noqa: E402
from modules.stitch_algorithms import estimate_overlap_offset, stitch_registered_tiles  # noqa: E402


DEFAULT_OUTPUT_ROOT = BENCH_ROOT / "actual_tile_runs"
TIFF_EXTS = {".tif", ".tiff", ".ome.tif", ".ome.tiff"}


@dataclass
class StitchRun:
    image: np.ndarray
    registered_tiles: list[dict[str, Any]]
    metadata: dict[str, Any]
    timings: dict[str, float]


def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(BENCH_ROOT))
    except ValueError:
        return str(path)


def _safe_name(text: object) -> str:
    allowed = []
    for char in str(text):
        allowed.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(allowed).strip("_") or "unnamed"


def read_image(path: pathlib.Path) -> np.ndarray:
    image = tf.imread(path)
    if image.ndim == 3 and image.shape[-1] > 3:
        return image[:, :, :3]
    return image


def write_tiff(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tf.imwrite(path, image, compression="lzw")


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)


# The FFT estimator, overlap-view slicer, tile blender and grid-key helpers below
# intentionally re-implement production's stitching primitives rather than importing
# them. Production fuses registration and blending into stitch_registered_tiles and
# returns a bare (dx, dy, score) from its phase estimator; this comparison tool needs
# the stages kept separate and each estimator to report WHY it accepted or rejected an
# offset (no_overlap / too_small / low_signal / correction_too_large) so strategies can
# be compared side by side. Consolidating onto the production API would erase those
# diagnostics -- keep these copies distinct.
def _overlap_views(
    left: np.ndarray,
    right: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = left.shape[:2]
    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(w, dx + w)
    y1 = min(h, dy + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return left[y0:y1, x0:x1], right[y0 - dy : y1 - dy, x0 - dx : x1 - dx]


def _robust_range(image: np.ndarray) -> float:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return 1.0
    lo, hi = np.percentile(values, [1, 99.5])
    return float(max(hi - lo, 1.0))


def _overlap_has_signal(ref_view: np.ndarray, mov_view: np.ndarray) -> bool:
    ref_range = _robust_range(ref_view)
    mov_range = _robust_range(mov_view)
    ref_std = float(np.std(ref_view))
    mov_std = float(np.std(mov_view))
    return ref_std >= max(1.0, ref_range * 0.01) and mov_std >= max(1.0, mov_range * 0.01)


def estimate_fft_offset(
    reference: np.ndarray,
    moving: np.ndarray,
    nominal_dx: int,
    nominal_dy: int,
    *,
    max_correction_px: int,
    min_overlap_px: int,
) -> tuple[int, int, float, bool, str]:
    ref_gray = _gray_float(reference)
    mov_gray = _gray_float(moving)
    views = _overlap_views(ref_gray, mov_gray, nominal_dx, nominal_dy)
    if views is None:
        return 0, 0, -1.0, False, "no_overlap"
    ref_view, mov_view = views
    if ref_view.shape[0] < min_overlap_px or ref_view.shape[1] < min_overlap_px:
        return 0, 0, -1.0, False, "overlap_too_small"
    if not _overlap_has_signal(ref_view, mov_view):
        return 0, 0, -1.0, False, "overlap_low_signal"

    win = cv2.createHanningWindow((ref_view.shape[1], ref_view.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_view.astype(np.float32), mov_view.astype(np.float32), win)
    corr_x = round(-shift[0])
    corr_y = round(-shift[1])
    if abs(corr_x) > max_correction_px or abs(corr_y) > max_correction_px:
        return corr_x, corr_y, float(response), False, "fft_correction_too_large"
    return corr_x, corr_y, float(response), True, ""


def _blend_registered_tiles(registered: list[dict[str, Any]]) -> np.ndarray:
    if not registered:
        raise ValueError("Need at least one tile to blend")
    sample = registered[0]["tile"]
    tile_h, tile_w = sample.shape[:2]
    if any(tile["tile"].ndim != sample.ndim for tile in registered):
        raise ValueError("Cannot stitch a mix of mono and color tiles in one group")
    min_x = min(int(tile["registered_x_px"]) for tile in registered)
    min_y = min(int(tile["registered_y_px"]) for tile in registered)
    max_x = max(int(tile["registered_x_px"]) + tile_w for tile in registered)
    max_y = max(int(tile["registered_y_px"]) + tile_h for tile in registered)
    if sample.ndim == 2:
        accumulator = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)
        weights = np.zeros(accumulator.shape, dtype=np.float32)
    else:
        accumulator = np.zeros((max_y - min_y, max_x - min_x, sample.shape[2]), dtype=np.float32)
        weights = np.zeros((max_y - min_y, max_x - min_x, 1), dtype=np.float32)

    for tile in registered:
        image = tile["tile"]
        x0 = int(tile["registered_x_px"]) - min_x
        y0 = int(tile["registered_y_px"]) - min_y
        y1 = y0 + image.shape[0]
        x1 = x0 + image.shape[1]
        accumulator[y0:y1, x0:x1] += image.astype(np.float32)
        weights[y0:y1, x0:x1] += 1.0
    np.divide(accumulator, weights, out=accumulator, where=weights > 0)
    if np.issubdtype(sample.dtype, np.integer):
        info = np.iinfo(sample.dtype)
        np.clip(accumulator, info.min, info.max, out=accumulator)
    return accumulator.astype(sample.dtype)


def _grid_keys(tiles: list[dict[str, Any]]) -> tuple[list[int], list[int], dict[tuple[int, int], int]]:
    x_values = sorted({int(tile["x_px"]) for tile in tiles})
    y_values = sorted({int(tile["y_px"]) for tile in tiles})
    by_position = {(int(tile["x_px"]), int(tile["y_px"])): idx for idx, tile in enumerate(tiles)}
    return x_values, y_values, by_position


def align_hybrid_fft_lvp(
    tiles: list[dict[str, Any]],
    *,
    fft_response_threshold: float,
    max_correction_px: int,
    min_overlap_px: int,
    consistency_px: int,
    force_fft: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not tiles:
        return [], {}

    x_values, y_values, by_position = _grid_keys(tiles)
    corrected = [dict(tile) for tile in tiles]
    offsets: dict[int, tuple[int, int]] = {}
    edge_corrections: list[tuple[int, int]] = []
    edge_records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    used: Counter[str] = Counter()

    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}
    anchor = by_position[(x_values[0], y_values[0])]
    offsets[anchor] = (0, 0)
    queue: deque[int] = deque([anchor])

    while queue:
        idx = queue.popleft()
        base_dx, base_dy = offsets[idx]
        x = int(tiles[idx]["x_px"])
        y = int(tiles[idx]["y_px"])
        xi = x_index[x]
        yi = y_index[y]
        neighbors = []
        if xi > 0:
            neighbors.append((x_values[xi - 1], y))
        if xi + 1 < len(x_values):
            neighbors.append((x_values[xi + 1], y))
        if yi > 0:
            neighbors.append((x, y_values[yi - 1]))
        if yi + 1 < len(y_values):
            neighbors.append((x, y_values[yi + 1]))

        for nx, ny in neighbors:
            nidx = by_position.get((nx, ny))
            if nidx is None or nidx in offsets:
                continue
            nominal_dx = nx - x
            nominal_dy = ny - y

            edge_t0 = time.perf_counter()
            fft_x, fft_y, response, fft_basic_ok, reason = estimate_fft_offset(
                tiles[idx]["tile"],
                tiles[nidx]["tile"],
                nominal_dx,
                nominal_dy,
                max_correction_px=max_correction_px,
                min_overlap_px=min_overlap_px,
            )
            fft_ms = (time.perf_counter() - edge_t0) * 1000.0

            force_usable_fft = force_fft and reason not in {
                "no_overlap",
                "overlap_too_small",
                "overlap_low_signal",
            }
            use_fft = (fft_basic_ok and response >= fft_response_threshold) or force_usable_fft
            if fft_basic_ok and response < fft_response_threshold:
                reason = "fft_response_low"

            if use_fft and edge_corrections and not force_fft:
                med_x = float(np.median([item[0] for item in edge_corrections]))
                med_y = float(np.median([item[1] for item in edge_corrections]))
                if abs(fft_x - med_x) > consistency_px or abs(fft_y - med_y) > consistency_px:
                    use_fft = False
                    reason = "fft_inconsistent_with_neighbors"

            if use_fft:
                corr_x, corr_y, score = fft_x, fft_y, response
                method = "fft"
                lvp_ms = 0.0
                if force_fft and reason:
                    reasons[f"forced_{reason}"] += 1
            else:
                lvp_t0 = time.perf_counter()
                corr_x, corr_y, score = estimate_overlap_offset(
                    reference=tiles[idx]["tile"],
                    moving=tiles[nidx]["tile"],
                    nominal_dx=nominal_dx,
                    nominal_dy=nominal_dy,
                    max_correction_px=min(max_correction_px, 12),
                    min_overlap_px=min_overlap_px,
                )
                lvp_ms = (time.perf_counter() - lvp_t0) * 1000.0
                method = "lvp_fallback"
                reasons[reason or "fft_rejected"] += 1

            used[method] += 1
            edge_corrections.append((int(corr_x), int(corr_y)))
            offsets[nidx] = (base_dx + int(corr_x), base_dy + int(corr_y))
            corrected[nidx]["registration_offset_x_px"] = offsets[nidx][0]
            corrected[nidx]["registration_offset_y_px"] = offsets[nidx][1]
            corrected[nidx]["registration_score"] = float(score)
            corrected[nidx]["registration_method"] = method
            queue.append(nidx)
            edge_records.append(
                {
                    "from_tile_index": tiles[idx].get("source_index", idx),
                    "to_tile_index": tiles[nidx].get("source_index", nidx),
                    "from_x_px": x,
                    "from_y_px": y,
                    "to_x_px": nx,
                    "to_y_px": ny,
                    "method": method,
                    "fft_response": response,
                    "fft_corr_x_px": fft_x,
                    "fft_corr_y_px": fft_y,
                    "used_corr_x_px": int(corr_x),
                    "used_corr_y_px": int(corr_y),
                    "fallback_reason": "" if use_fft else (reason or "fft_rejected"),
                    "fft_ms": fft_ms,
                    "lvp_fallback_ms": lvp_ms,
                }
            )

    for idx, tile in enumerate(corrected):
        corr_x, corr_y = offsets.get(idx, (0, 0))
        tile["registration_offset_x_px"] = corr_x
        tile["registration_offset_y_px"] = corr_y
        tile["registered_x_px"] = int(tile["x_px"]) + corr_x
        tile["registered_y_px"] = int(tile["y_px"]) + corr_y
        tile.setdefault("registration_method", "anchor" if idx == anchor else "unregistered")

    metadata = {
        "mode": "forced_fft" if force_fft else "hybrid_fft_lvp",
        "edge_count": len(edge_records),
        "fft_edges": int(used["fft"]),
        "lvp_fallback_edges": int(used["lvp_fallback"]),
        "fallback_reasons": dict(reasons),
        "edge_records": edge_records,
    }
    return corrected, metadata


def _load_protocol_steps(protocol_path: pathlib.Path) -> tuple[Protocol, pathlib.Path]:
    tiling_path = REPO_ROOT / "data" / "tiling.json"
    return Protocol.from_file(file_path=protocol_path, tiling_configs_file_loc=tiling_path), tiling_path


def _find_protocol_record(root: pathlib.Path) -> pathlib.Path | None:
    candidates = [root / ProtocolExecutionRecord.DEFAULT_FILENAME, root.parent / ProtocolExecutionRecord.DEFAULT_FILENAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = sorted(root.rglob(ProtocolExecutionRecord.DEFAULT_FILENAME))
    return found[0] if found else None


def _dataframe_from_protocol_record(input_path: pathlib.Path) -> tuple[pathlib.Path, pd.DataFrame] | None:
    record_path = _find_protocol_record(input_path)
    if record_path is None:
        return None
    root = record_path.parent
    record = ProtocolExecutionRecord.from_file(record_path)
    protocol_path = root / record.protocol_file_loc()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file from record not found: {protocol_path}")
    protocol, _ = _load_protocol_steps(protocol_path)
    rows = []
    for _row_idx, rec in record._records.iterrows():
        filename = pathlib.Path(str(rec["Filename"]))
        image_path = root / filename
        if not image_path.is_file():
            continue
        step = protocol.step(idx=int(rec["Step Index"]))
        rows.append(
            {
                "Filepath": filename,
                "Name": step.get("Name", ""),
                "Scan Count": int(rec.get("Scan Count", 0)),
                "Step Index": int(rec["Step Index"]),
                "X": float(step["X"]),
                "Y": float(step["Y"]),
                "Z": float(step.get("Z", 0)),
                "Z-Slice": int(step.get("Z-Slice", -1)),
                "Well": step.get("Well", ""),
                "Color": step.get("Color", ""),
                "Objective": step.get("Objective", ""),
                "Tile": step.get("Tile", ""),
                "Tile Group ID": step.get("Tile Group ID", ""),
                "Custom Step": bool(step.get("Custom Step", False)),
                "Timestamp": rec.get("Timestamp", ""),
            }
        )
    return root, pd.DataFrame(rows)


def _read_lvp_table(path: pathlib.Path) -> pd.DataFrame | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if path.suffix.lower() == ".tsv":
        for marker in ("Images", "Steps"):
            if marker in lines:
                idx = lines.index(marker)
                if idx + 1 < len(lines):
                    return pd.read_csv(path, sep="\t", skiprows=idx + 1)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _dataframe_from_direct_table(input_path: pathlib.Path) -> tuple[pathlib.Path, pd.DataFrame] | None:
    search_root = input_path if input_path.is_dir() else input_path.parent
    table_exts = {".tsv", ".csv"}
    tsvs = [input_path] if input_path.is_file() and input_path.suffix.lower() in table_exts else [
        *sorted(search_root.rglob("*.tsv")),
        *sorted(search_root.rglob("*.csv")),
    ]
    required = {"Filepath", "X", "Y", "Objective"}
    for tsv in tsvs:
        df = _read_lvp_table(tsv)
        if df is None or not required.issubset(df.columns):
            continue
        root = tsv.parent
        df = df.copy()
        df["Filepath"] = df["Filepath"].astype(str).map(pathlib.Path)
        df = df[(df["Filepath"].map(lambda p, root=root: (root / p).is_file()))]
        if len(df) > 1:
            return root, df
    return None


def load_input_dataframe(input_path: pathlib.Path) -> tuple[pathlib.Path, pd.DataFrame]:
    loaded = _dataframe_from_protocol_record(input_path)
    if loaded is None:
        loaded = _dataframe_from_direct_table(input_path)
    if loaded is None:
        raise FileNotFoundError(
            f"Could not load protocol image metadata from {input_path}. "
            "Expected protocol_record.tsv + protocol TSV, or a TSV with Filepath/X/Y/Objective columns."
        )
    root, df = loaded
    if df.empty:
        raise ValueError(f"No readable image rows found in {input_path}")
    return root, df.fillna("")


def group_dataframe(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    group_cols = [
        col
        for col in ["Scan Count", "Z-Slice", "Well", "Color", "Objective", "Tile Group ID", "Custom Step"]
        if col in df.columns
    ]
    if not group_cols:
        return [("all_tiles", df)]
    groups = []
    for key, group in df.groupby(group_cols, dropna=False):
        if len(group) < 2:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        group_id = "__".join(
            f"{_safe_name(col)}-{_safe_name(value)}"
            for col, value in zip(group_cols, key, strict=True)
        )
        groups.append((group_id, group.copy()))
    return groups


def resolve_pixel_size_um(df: pd.DataFrame, override: float | None) -> float:
    if override is not None and override > 0:
        return float(override)
    objective = str(df.iloc[0].get("Objective", "")).strip()
    if not objective:
        raise ValueError("Objective is blank; pass --pixel-size-um")
    loader = ObjectiveLoader(source_path=REPO_ROOT)
    info = loader.get_objective_info(objective_id=objective)
    return float(common_utils.get_pixel_size(focal_length=info["focal_length"], binning_size=1))


def build_tiles(root: pathlib.Path, df: pd.DataFrame, pixel_size_um: float) -> list[dict[str, Any]]:
    frame = df.copy()
    frame["X"] = frame["X"].astype(float)
    frame["Y"] = frame["Y"].astype(float)
    images = {row["Filepath"]: read_image(root / row["Filepath"]) for _, row in frame.iterrows()}
    sample = next(iter(images.values()))
    image_h, image_w = sample.shape[:2]
    x_max = frame["X"].max()
    y_min = frame["Y"].min()
    frame["x_pix"] = ((x_max - frame["X"]) * 1000 / pixel_size_um).round().astype(int)
    frame["y_pix"] = ((frame["Y"] - y_min) * 1000 / pixel_size_um).round().astype(int)
    tiles = []
    for source_index, (_, row) in enumerate(frame.iterrows()):
        tiles.append(
            {
                "tile": images[row["Filepath"]],
                "x_px": int(row["x_pix"]),
                "y_px": int(row["y_pix"]),
                "source_index": source_index,
                "filepath": str(row["Filepath"]),
                "width_px": image_w,
                "height_px": image_h,
            }
        )
    return tiles


def run_current_lvp(tiles: list[dict[str, Any]]) -> StitchRun:
    t0 = time.perf_counter()
    image, registered = stitch_registered_tiles(tiles, max_correction_px=12, min_overlap_px=16)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return StitchRun(
        image=image,
        registered_tiles=registered,
        metadata={"algorithm": "current_lvp_overlap"},
        timings={"total_ms": total_ms},
    )


def run_hybrid(tiles: list[dict[str, Any]], args: argparse.Namespace) -> StitchRun:
    t0 = time.perf_counter()
    reg_t0 = time.perf_counter()
    registered, metadata = align_hybrid_fft_lvp(
        tiles,
        fft_response_threshold=args.fft_response_threshold,
        max_correction_px=args.max_correction_px,
        min_overlap_px=args.min_overlap_px,
        consistency_px=args.consistency_px,
        force_fft=args.force_fft,
    )
    registration_ms = (time.perf_counter() - reg_t0) * 1000.0
    blend_t0 = time.perf_counter()
    image = _blend_registered_tiles(registered)
    blend_ms = (time.perf_counter() - blend_t0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    metadata["algorithm"] = "forced_fft" if args.force_fft else "hybrid_fft_lvp"
    return StitchRun(
        image=image,
        registered_tiles=registered,
        metadata=metadata,
        timings={"total_ms": total_ms, "registration_ms": registration_ms, "blend_ms": blend_ms},
    )


def _fit_to(image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    target_h, target_w = shape[:2]
    h, w = image.shape[:2]
    y0_src = max(0, (h - target_h) // 2)
    x0_src = max(0, (w - target_w) // 2)
    cropped = image[y0_src : y0_src + min(h, target_h), x0_src : x0_src + min(w, target_w)]
    out_shape = shape if image.ndim == len(shape) else (*shape[:2], image.shape[2]) if image.ndim == 3 else shape[:2]
    out = np.zeros(out_shape, dtype=image.dtype)
    y0_dst = max(0, (target_h - cropped.shape[0]) // 2)
    x0_dst = max(0, (target_w - cropped.shape[1]) // 2)
    out[y0_dst : y0_dst + cropped.shape[0], x0_dst : x0_dst + cropped.shape[1], ...] = cropped
    return out


def compare_without_ground_truth(current: np.ndarray, hybrid: np.ndarray) -> dict[str, float]:
    fitted = _fit_to(hybrid, current.shape)
    cur = _gray_float(current)
    hyb = _gray_float(fitted)
    diff = cur.astype(np.float64) - hyb.astype(np.float64)
    denom = float(np.sqrt(np.mean(cur.astype(np.float64) ** 2))) or 1.0
    return {
        "mae_vs_current": float(np.mean(np.abs(diff))),
        "rmse_vs_current": float(np.sqrt(np.mean(diff * diff))),
        "nrmse_vs_current": float(np.sqrt(np.mean(diff * diff)) / denom),
    }


def _normalize_preview(image: np.ndarray, max_side: int = 620) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        channels = [arr]
    else:
        channels = [arr[:, :, idx] for idx in range(min(arr.shape[2], 3))]
    previews = []
    for ch in channels:
        sample = ch[np.isfinite(ch)]
        lo, hi = np.percentile(sample, [0.5, 99.7]) if sample.size else (0, 1)
        if hi <= lo:
            hi = lo + 1.0
        previews.append((np.clip((ch - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))
    if len(previews) == 1:
        out = cv2.cvtColor(previews[0], cv2.COLOR_GRAY2RGB)
    else:
        while len(previews) < 3:
            previews.append(np.zeros_like(previews[0]))
        out = np.stack(previews[:3], axis=2)
    h, w = out.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1:
        out = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return out


def write_visual_comparison(
    *,
    current: StitchRun,
    hybrid: StitchRun,
    output_path: pathlib.Path,
    title: str,
    metrics: dict[str, float],
) -> None:
    def panel(image: np.ndarray, heading: str, lines: list[str]) -> np.ndarray:
        preview = _normalize_preview(image)
        title_h = 128
        canvas = np.full((preview.shape[0] + title_h, max(preview.shape[1], 620), 3), 255, dtype=np.uint8)
        canvas[title_h : title_h + preview.shape[0], : preview.shape[1]] = preview
        cv2.putText(canvas, heading, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 2, cv2.LINE_AA)
        y = 56
        for line in lines[:4]:
            cv2.putText(canvas, line[:88], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
        return canvas

    hybrid_meta = hybrid.metadata
    panels = [
        panel(
            current.image,
            "Current LVP overlap",
            [
                title,
                f"time {current.timings['total_ms']:.1f} ms",
                "method: current local NCC registration",
            ],
        ),
        panel(
            hybrid.image,
            "Forced FFT" if hybrid_meta.get("mode") == "forced_fft" else "Hybrid FFT -> LVP fallback",
            [
                f"time {hybrid.timings['total_ms']:.1f} ms",
                f"FFT edges {hybrid_meta.get('fft_edges', 0)} / {hybrid_meta.get('edge_count', 0)}",
                f"LVP fallback edges {hybrid_meta.get('lvp_fallback_edges', 0)}",
                f"NRMSE vs current {metrics['nrmse_vs_current']:.4g}",
            ],
        ),
    ]
    height = max(item.shape[0] for item in panels)
    width = max(item.shape[1] for item in panels)
    padded = []
    for item in panels:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[: item.shape[0], : item.shape[1]] = item
        padded.append(canvas)
    sheet = np.concatenate(padded, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def write_html_report(rows: list[dict[str, Any]], out_dir: pathlib.Path) -> None:
    table_rows = []
    for row in rows:
        visual = html.escape(row["visual_path"])
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['input_name'])}</td>"
            f"<td>{html.escape(row['group_id'])}</td>"
            f"<td>{row['tile_count']}</td>"
            f"<td>{row['current_total_ms']:.1f}</td>"
            f"<td>{row['hybrid_total_ms']:.1f}</td>"
            f"<td>{row['speedup_vs_current']:.3g}</td>"
            f"<td>{row['hybrid_fft_edges']}/{row['hybrid_edge_count']}</td>"
            f"<td>{row['hybrid_lvp_fallback_edges']}</td>"
            f"<td>{row['nrmse_vs_current']:.4g}</td>"
            f"<td><a href='{visual}'>comparison</a></td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Actual Tile Hybrid Stitch Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Actual Tile Hybrid Stitch Comparison</h1>
  <p>Error metrics are image difference vs current LVP because real tile folders do not include ground truth.</p>
  <table>
    <thead>
      <tr>
        <th>Input</th><th>Group</th><th>Tiles</th><th>Current ms</th><th>Hybrid ms</th>
        <th>Speedup</th><th>FFT edges</th><th>LVP fallback edges</th><th>NRMSE vs current</th><th>Visual</th>
      </tr>
    </thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "comparison_report.html").write_text(doc, encoding="utf-8")


def process_input(input_path: pathlib.Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    root, df = load_input_dataframe(input_path)
    groups = group_dataframe(df)
    if args.max_groups:
        groups = groups[: args.max_groups]
    rows = []
    input_name = _safe_name(input_path.name or input_path.parent.name)
    for group_id, group in groups:
        pixel_size_um = resolve_pixel_size_um(group, args.pixel_size_um)
        tiles = build_tiles(root, group, pixel_size_um)
        if len(tiles) < 2:
            continue
        out_dir = pathlib.Path(args.output_root) / input_name / group_id
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"stitch input={input_path} group={group_id} tiles={len(tiles)}", flush=True)

        current = run_current_lvp(tiles)
        hybrid = run_hybrid(tiles, args)
        write_tiff(out_dir / "current_lvp_overlap.tiff", current.image)
        write_tiff(out_dir / "hybrid_fft_lvp.tiff", hybrid.image)
        pd.DataFrame(hybrid.metadata.get("edge_records", [])).to_csv(out_dir / "hybrid_edge_decisions.csv", index=False)

        metrics = compare_without_ground_truth(current.image, hybrid.image)
        visual_path = out_dir / "comparison.png"
        write_visual_comparison(
            current=current,
            hybrid=hybrid,
            output_path=visual_path,
            title=f"{input_name} / {group_id}",
            metrics=metrics,
        )
        row = {
            "input_path": str(input_path),
            "input_name": input_name,
            "root_path": str(root),
            "group_id": group_id,
            "tile_count": len(tiles),
            "pixel_size_um": pixel_size_um,
            "current_total_ms": current.timings["total_ms"],
            "hybrid_total_ms": hybrid.timings["total_ms"],
            "speedup_vs_current": current.timings["total_ms"] / hybrid.timings["total_ms"] if hybrid.timings["total_ms"] else math.nan,
            "hybrid_edge_count": int(hybrid.metadata.get("edge_count", 0)),
            "hybrid_fft_edges": int(hybrid.metadata.get("fft_edges", 0)),
            "hybrid_lvp_fallback_edges": int(hybrid.metadata.get("lvp_fallback_edges", 0)),
            "hybrid_fallback_reasons": json.dumps(hybrid.metadata.get("fallback_reasons", {}), sort_keys=True),
            "current_output_path": _rel(out_dir / "current_lvp_overlap.tiff"),
            "hybrid_output_path": _rel(out_dir / "hybrid_fft_lvp.tiff"),
            "edge_decisions_path": _rel(out_dir / "hybrid_edge_decisions.csv"),
            "visual_path": _rel(visual_path),
            **metrics,
        }
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Local protocol output folders or TSV files")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pixel-size-um", type=float, default=None, help="Override stage um/pixel conversion")
    parser.add_argument("--max-groups", type=int, default=0, help="Limit groups per input for quick checks")
    parser.add_argument("--fft-response-threshold", type=float, default=0.25)
    parser.add_argument("--max-correction-px", type=int, default=24)
    parser.add_argument("--min-overlap-px", type=int, default=16)
    parser.add_argument("--consistency-px", type=int, default=10)
    parser.add_argument(
        "--force-fft",
        action="store_true",
        help=(
            "Use FFT results even when response, correction-size, or consistency checks fail. "
            "Only no-overlap/too-small/low-signal edges still fall back."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = pathlib.Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for raw in args.inputs:
        rows.extend(process_input(pathlib.Path(raw).resolve(), args))
    if not rows:
        raise SystemExit("No stitch groups were processed")
    df = pd.DataFrame(rows)
    csv_path = out_root / "comparison_summary.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_html_report(rows, out_root)
    print(f"wrote {_rel(csv_path)}", flush=True)
    print(f"wrote {_rel(out_root / 'comparison_report.html')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
