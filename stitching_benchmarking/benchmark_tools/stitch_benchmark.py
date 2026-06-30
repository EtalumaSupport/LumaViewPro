#!/usr/bin/env python3
"""Generate synthetic tile datasets and benchmark LVP stitching methods.

The command is intentionally self-contained under ``stitching_benchmarking``:
it reads the source atlas, writes generated tiles/results/reports next to it,
and imports the current LVP stitching core as the primary baseline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import pathlib
import queue
import sys
import threading
import time
import tracemalloc
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd
import psutil
import tifffile as tf
from skimage import metrics as sk_metrics

BENCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.stitch_algorithms import (  # noqa: E402
    estimate_overlap_offset,
    feature_stitch,
    stitch_registered_tiles,
)


SOURCE_ROOT = BENCH_ROOT / "LS800imageAtlas"
GENERATED_ROOT = BENCH_ROOT / "generated_tiles"
RESULTS_ROOT = BENCH_ROOT / "results"
REPORTS_ROOT = BENCH_ROOT / "reports"

OVERLAP = 0.10
GRID_SIZES = (3, 5, 7)
TIERS = ("clean", "realistic", "stress", "failure")
METHODS = (
    "lvp_current_overlap",
    "lvp_simple_grid",
    "fft_phase_correlation",
    "coarse_to_fine_ncc",
    "opencv_feature",
)
PIXEL_SIZE_UM = 1.0


@dataclass(frozen=True)
class SourceImage:
    path: pathlib.Path
    magnification: str
    channel: str

    @property
    def source_id(self) -> str:
        return f"{self.magnification}__{self.channel}"


@dataclass
class TimedResult:
    image: np.ndarray | None
    registered_tiles: list[dict[str, Any]]
    timings: dict[str, float]
    status: bool
    error: str
    metadata: dict[str, Any]


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(BENCH_ROOT))


def _safe_name(text: str) -> str:
    allowed = []
    for char in text:
        allowed.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(allowed).strip("_")


def _seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def discover_sources(source_root: pathlib.Path = SOURCE_ROOT) -> list[SourceImage]:
    sources: list[SourceImage] = []
    for path in sorted(source_root.rglob("*.tif*")):
        rel = path.relative_to(source_root)
        if len(rel.parts) == 3 and rel.parts[1] in {"Red", "Green", "Blue"}:
            sources.append(SourceImage(path=path, magnification=rel.parts[0], channel=rel.parts[1]))
        elif len(rel.parts) == 2 and path.name.startswith("composite_"):
            sources.append(SourceImage(path=path, magnification=rel.parts[0], channel="composite"))
    return sources


def read_image(path: pathlib.Path) -> np.ndarray:
    image = tf.imread(path)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape for {path}: {image.shape}")


def write_tiff(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tf.imwrite(path, image, compression="lzw")


def grid_positions(length: int, grid: int, overlap: float = OVERLAP) -> tuple[int, list[int]]:
    tile_len = int(round(length / (grid - (grid - 1) * overlap)))
    tile_len = max(32, min(tile_len, length))
    if grid == 1:
        return tile_len, [0]
    max_start = length - tile_len
    positions = [int(round(value)) for value in np.linspace(0, max_start, grid)]
    return tile_len, positions


def _normalize_preview(image: np.ndarray, max_side: int = 420) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        sample = arr[arr > 0]
        if sample.size == 0:
            sample = arr.reshape(-1)
        lo, hi = np.percentile(sample, [1, 99.5])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            out = np.zeros(arr.shape, dtype=np.uint8)
        else:
            out = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    else:
        channels = []
        for idx in range(arr.shape[2]):
            ch = arr[:, :, idx]
            sample = ch[ch > 0]
            if sample.size == 0:
                sample = ch.reshape(-1)
            lo, hi = np.percentile(sample, [1, 99.5])
            if hi <= lo:
                lo, hi = float(ch.min()), float(ch.max())
            if hi <= lo:
                channels.append(np.zeros(ch.shape, dtype=np.uint8))
            else:
                channels.append((np.clip((ch - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))
        out = np.stack(channels[:3], axis=2)
    h, w = out.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        out = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return out


def _apply_vignette(tile: np.ndarray, strength: float) -> np.ndarray:
    h, w = tile.shape[:2]
    y, x = np.ogrid[-1:1:complex(h), -1:1:complex(w)]
    radius = np.sqrt(x * x + y * y)
    mask = 1.0 - strength * np.clip(radius, 0, 1)
    if tile.ndim == 3:
        mask = mask[:, :, None]
    return tile.astype(np.float32) * mask


def _apply_perturbations(
    tile: np.ndarray,
    *,
    tier: str,
    rng: np.random.Generator,
    tile_index: int,
    tile_count: int,
) -> np.ndarray | None:
    arr = tile.astype(np.float32)
    dtype = tile.dtype
    max_value = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0

    if tier == "failure":
        if tile_index == tile_count // 2:
            return None
        if tile_index == tile_count // 2 - 1:
            arr[...] = 0
        elif tile_index == tile_count // 2 + 1:
            arr[...] = max_value
        else:
            arr *= rng.uniform(0.55, 1.35)
            arr += rng.uniform(-0.025, 0.025) * max_value
            arr = _apply_vignette(arr.astype(dtype, copy=False), 0.25).astype(np.float32)
    elif tier == "stress":
        arr *= rng.uniform(0.60, 1.45)
        arr += rng.uniform(-0.035, 0.035) * max_value
        arr = _apply_vignette(np.clip(arr, 0, max_value).astype(dtype), 0.22).astype(np.float32)
        if rng.random() < 0.45:
            arr = cv2.GaussianBlur(arr, (5, 5), rng.uniform(0.6, 1.2))
        noise_sigma = max_value * 0.012
        arr += rng.normal(0, noise_sigma, size=arr.shape)
    elif tier == "realistic":
        arr *= rng.uniform(0.88, 1.12)
        arr += rng.uniform(-0.01, 0.01) * max_value
        arr = _apply_vignette(np.clip(arr, 0, max_value).astype(dtype), 0.08).astype(np.float32)
        if rng.random() < 0.25:
            arr = cv2.GaussianBlur(arr, (3, 3), rng.uniform(0.25, 0.55))
        arr += rng.normal(0, max_value * 0.002, size=arr.shape)
    elif tier != "clean":
        raise ValueError(f"Unknown perturbation tier: {tier}")

    if np.issubdtype(dtype, np.integer):
        arr = np.clip(arr, 0, max_value)
    return arr.astype(dtype)


def _stage_jitter(tier: str, rng: np.random.Generator) -> tuple[int, int]:
    if tier == "clean":
        return 0, 0
    if tier == "realistic":
        return int(rng.integers(-5, 6)), int(rng.integers(-5, 6))
    if tier in {"stress", "failure"}:
        return int(rng.integers(-15, 16)), int(rng.integers(-15, 16))
    raise ValueError(f"Unknown perturbation tier: {tier}")


def make_ideal_ground_truth(
    source: np.ndarray,
    x_positions: list[int],
    y_positions: list[int],
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    if source.ndim == 2:
        acc = np.zeros(source.shape, dtype=np.float32)
        weights = np.zeros(source.shape, dtype=np.float32)
    else:
        acc = np.zeros(source.shape, dtype=np.float32)
        weights = np.zeros((*source.shape[:2], 1), dtype=np.float32)
    for y in y_positions:
        for x in x_positions:
            tile = source[y : y + tile_h, x : x + tile_w]
            acc[y : y + tile_h, x : x + tile_w] += tile.astype(np.float32)
            weights[y : y + tile_h, x : x + tile_w] += 1.0
    np.divide(acc, weights, out=acc, where=weights > 0)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        np.clip(acc, info.min, info.max, out=acc)
    return acc.astype(source.dtype)


def generate_dataset(
    source: SourceImage,
    *,
    grid: int,
    tier: str,
    output_root: pathlib.Path = GENERATED_ROOT,
    force: bool = False,
) -> pathlib.Path:
    dataset_id = _safe_name(f"{source.source_id}__grid{grid}x{grid}__overlap10__{tier}")
    dataset_dir = output_root / dataset_id
    config_path = dataset_dir / "dataset_config.json"
    if config_path.exists() and not force:
        return dataset_dir

    image = read_image(source.path)
    h, w = image.shape[:2]
    tile_w, x_positions = grid_positions(w, grid)
    tile_h, y_positions = grid_positions(h, grid)
    rng = np.random.default_rng(_seed(source.source_id, grid, tier))
    tiles_dir = dataset_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    gt = make_ideal_ground_truth(image, x_positions, y_positions, tile_w, tile_h)
    write_tiff(dataset_dir / "ground_truth.tiff", gt)

    rows: list[dict[str, Any]] = []
    tile_count = grid * grid
    tile_index = 0
    for row_idx, nominal_y in enumerate(y_positions):
        for col_idx, nominal_x in enumerate(x_positions):
            jitter_x, jitter_y = _stage_jitter(tier, rng)
            actual_x = int(np.clip(nominal_x + jitter_x, 0, w - tile_w))
            actual_y = int(np.clip(nominal_y + jitter_y, 0, h - tile_h))
            tile = image[actual_y : actual_y + tile_h, actual_x : actual_x + tile_w].copy()
            tile = _apply_perturbations(
                tile,
                tier=tier,
                rng=rng,
                tile_index=tile_index,
                tile_count=tile_count,
            )
            filename = f"tile_r{row_idx:02d}_c{col_idx:02d}.tiff"
            included = tile is not None
            if included:
                write_tiff(tiles_dir / filename, tile)
            rows.append(
                {
                    "tile_index": tile_index,
                    "row": row_idx,
                    "col": col_idx,
                    "filename": f"tiles/{filename}",
                    "included": included,
                    "nominal_x_px": nominal_x,
                    "nominal_y_px": nominal_y,
                    "actual_x_px": actual_x,
                    "actual_y_px": actual_y,
                    "true_offset_x_px": actual_x - nominal_x,
                    "true_offset_y_px": actual_y - nominal_y,
                    "width_px": tile_w,
                    "height_px": tile_h,
                }
            )
            tile_index += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(dataset_dir / "tiles_manifest.csv", index=False)

    coords = manifest[manifest["included"]].copy()
    max_nominal_x = float(coords["nominal_x_px"].max()) if not coords.empty else 0.0
    coords["Filepath"] = coords["filename"]
    coords["X"] = (max_nominal_x - coords["nominal_x_px"].astype(float)) * PIXEL_SIZE_UM / 1000.0
    coords["Y"] = coords["nominal_y_px"].astype(float) * PIXEL_SIZE_UM / 1000.0
    coords["Objective"] = source.magnification
    coords["Color"] = source.channel
    coords["Well"] = "Synthetic"
    coords["Tile Group ID"] = "0"
    coords[
        [
            "Filepath",
            "X",
            "Y",
            "Objective",
            "Color",
            "Well",
            "Tile Group ID",
            "tile_index",
            "row",
            "col",
        ]
    ].to_csv(dataset_dir / "fake_lvp_coords.csv", index=False)

    config = {
        "dataset_id": dataset_id,
        "source_path": str(source.path.relative_to(BENCH_ROOT)),
        "magnification": source.magnification,
        "channel": source.channel,
        "grid": grid,
        "overlap_fraction": OVERLAP,
        "tier": tier,
        "source_shape": list(image.shape),
        "tile_shape": [tile_h, tile_w] + ([] if image.ndim == 2 else [image.shape[2]]),
        "pixel_size_um": PIXEL_SIZE_UM,
        "generated_at_unix_s": time.time(),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dataset_dir


def generate_all(args: argparse.Namespace) -> list[pathlib.Path]:
    source_root = pathlib.Path(args.source_root).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    sources = discover_sources(source_root)
    if args.source_limit:
        sources = sources[: args.source_limit]
    generated: list[pathlib.Path] = []
    for source in sources:
        for grid in args.grids:
            for tier in args.tiers:
                dataset_dir = generate_dataset(
                    source,
                    grid=grid,
                    tier=tier,
                    output_root=output_root,
                    force=args.force,
                )
                generated.append(dataset_dir)
                print(f"generated {dataset_dir.relative_to(BENCH_ROOT)}", flush=True)
    return generated


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


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


def _ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-6:
        return -1.0
    return float(np.sum(a * b) / denom)


def estimate_phase_offset(
    reference: np.ndarray,
    moving: np.ndarray,
    nominal_dx: int,
    nominal_dy: int,
    *,
    max_correction_px: int = 24,
    min_overlap_px: int = 16,
) -> tuple[int, int, float]:
    views = _overlap_views(_gray_float(reference), _gray_float(moving), nominal_dx, nominal_dy)
    if views is None:
        return 0, 0, -1.0
    ref_view, mov_view = views
    if ref_view.shape[0] < min_overlap_px or ref_view.shape[1] < min_overlap_px:
        return 0, 0, -1.0
    win = cv2.createHanningWindow((ref_view.shape[1], ref_view.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_view.astype(np.float32), mov_view.astype(np.float32), win)
    corr_x = int(round(-shift[0]))
    corr_y = int(round(-shift[1]))
    corr_x = int(np.clip(corr_x, -max_correction_px, max_correction_px))
    corr_y = int(np.clip(corr_y, -max_correction_px, max_correction_px))
    return corr_x, corr_y, float(response)


def estimate_coarse_to_fine_ncc(
    reference: np.ndarray,
    moving: np.ndarray,
    nominal_dx: int,
    nominal_dy: int,
    *,
    max_correction_px: int = 24,
    min_overlap_px: int = 16,
) -> tuple[int, int, float]:
    ref_gray = _gray_float(reference)
    mov_gray = _gray_float(moving)
    scale = 0.25
    ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mov_small = cv2.resize(mov_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    coarse_best = (0, 0, -1.0)
    coarse_radius = max(1, int(math.ceil(max_correction_px * scale)))
    for cy in range(-coarse_radius, coarse_radius + 1):
        for cx in range(-coarse_radius, coarse_radius + 1):
            views = _overlap_views(
                ref_small,
                mov_small,
                int(round((nominal_dx * scale) + cx)),
                int(round((nominal_dy * scale) + cy)),
            )
            if views is None:
                continue
            ref_view, mov_view = views
            if ref_view.shape[0] < max(4, int(min_overlap_px * scale)) or ref_view.shape[1] < max(4, int(min_overlap_px * scale)):
                continue
            score = _ncc_score(ref_view, mov_view)
            if score > coarse_best[2]:
                coarse_best = (cx, cy, score)

    base_x = int(round(coarse_best[0] / scale))
    base_y = int(round(coarse_best[1] / scale))
    best = (base_x, base_y, -1.0)
    for corr_y in range(base_y - 3, base_y + 4):
        for corr_x in range(base_x - 3, base_x + 4):
            if abs(corr_x) > max_correction_px or abs(corr_y) > max_correction_px:
                continue
            views = _overlap_views(ref_gray, mov_gray, nominal_dx + corr_x, nominal_dy + corr_y)
            if views is None:
                continue
            ref_view, mov_view = views
            if ref_view.shape[0] < min_overlap_px or ref_view.shape[1] < min_overlap_px:
                continue
            score = _ncc_score(ref_view, mov_view)
            if score > best[2]:
                best = (corr_x, corr_y, score)
    return best


def align_with_estimator(
    tiles: list[dict[str, Any]],
    estimator: Callable[..., tuple[int, int, float]],
    *,
    max_correction_px: int = 24,
    min_overlap_px: int = 16,
) -> list[dict[str, Any]]:
    if not tiles:
        return []
    x_values = sorted({int(tile["x_px"]) for tile in tiles})
    y_values = sorted({int(tile["y_px"]) for tile in tiles})
    by_position = {(int(tile["x_px"]), int(tile["y_px"])): idx for idx, tile in enumerate(tiles)}
    x_index = {x: idx for idx, x in enumerate(x_values)}
    y_index = {y: idx for idx, y in enumerate(y_values)}
    corrected = [dict(tile) for tile in tiles]
    offsets: dict[int, tuple[int, int]] = {}
    edge_times: list[float] = []
    edge_scores: list[float] = []
    failed_edges = 0
    anchor = by_position[(x_values[0], y_values[0])]
    offsets[anchor] = (0, 0)
    pending: deque[int] = deque([anchor])
    while pending:
        idx = pending.popleft()
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
            t0 = time.perf_counter()
            corr_x, corr_y, score = estimator(
                tiles[idx]["tile"],
                tiles[nidx]["tile"],
                nx - x,
                ny - y,
                max_correction_px=max_correction_px,
                min_overlap_px=min_overlap_px,
            )
            edge_times.append((time.perf_counter() - t0) * 1000.0)
            edge_scores.append(float(score))
            if score < -0.5:
                failed_edges += 1
                corr_x, corr_y = 0, 0
            offsets[nidx] = (base_dx + int(corr_x), base_dy + int(corr_y))
            corrected[nidx]["registration_score"] = float(score)
            corrected[nidx]["registration_failed_edge"] = bool(score < -0.5)
            pending.append(nidx)

    for idx, tile in enumerate(corrected):
        corr_x, corr_y = offsets.get(idx, (0, 0))
        tile["registration_offset_x_px"] = corr_x
        tile["registration_offset_y_px"] = corr_y
        tile["registered_x_px"] = int(tile["x_px"]) + corr_x
        tile["registered_y_px"] = int(tile["y_px"]) + corr_y
    for tile in corrected:
        tile["registration_edge_time_ms_avg"] = float(np.mean(edge_times)) if edge_times else 0.0
        tile["registration_score_avg"] = float(np.mean(edge_scores)) if edge_scores else 0.0
        tile["registration_failed_edges"] = failed_edges
    return corrected


def blend_registered_tiles(registered: list[dict[str, Any]]) -> np.ndarray:
    if not registered:
        raise ValueError("Need at least one tile to blend")
    sample = registered[0]["tile"]
    tile_h, tile_w = sample.shape[:2]
    min_x = min(int(tile["registered_x_px"]) for tile in registered)
    min_y = min(int(tile["registered_y_px"]) for tile in registered)
    max_x = max(int(tile["registered_x_px"]) + tile_w for tile in registered)
    max_y = max(int(tile["registered_y_px"]) + tile_h for tile in registered)
    if sample.ndim == 2:
        acc = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)
        weights = np.zeros(acc.shape, dtype=np.float32)
    else:
        acc = np.zeros((max_y - min_y, max_x - min_x, sample.shape[2]), dtype=np.float32)
        weights = np.zeros((max_y - min_y, max_x - min_x, 1), dtype=np.float32)
    for tile in registered:
        image = tile["tile"]
        x0 = int(tile["registered_x_px"]) - min_x
        y0 = int(tile["registered_y_px"]) - min_y
        x1 = x0 + image.shape[1]
        y1 = y0 + image.shape[0]
        acc[y0:y1, x0:x1] += image.astype(np.float32)
        weights[y0:y1, x0:x1] += 1.0
    np.divide(acc, weights, out=acc, where=weights > 0)
    if np.issubdtype(sample.dtype, np.integer):
        info = np.iinfo(sample.dtype)
        np.clip(acc, info.min, info.max, out=acc)
    return acc.astype(sample.dtype)


def load_tiles(dataset_dir: pathlib.Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    manifest = pd.read_csv(dataset_dir / "tiles_manifest.csv")
    included = manifest[manifest["included"].astype(bool)].copy()
    tiles: list[dict[str, Any]] = []
    for _, row in included.iterrows():
        image = read_image(dataset_dir / str(row["filename"]))
        tiles.append(
            {
                "tile": image,
                "x_px": int(row["nominal_x_px"]),
                "y_px": int(row["nominal_y_px"]),
                "tile_index": int(row["tile_index"]),
                "true_x_px": int(row["actual_x_px"]),
                "true_y_px": int(row["actual_y_px"]),
            }
        )
    return tiles, manifest


def method_lvp_current_overlap(dataset_dir: pathlib.Path) -> TimedResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    tiles, _ = load_tiles(dataset_dir)
    timings["read_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    image, registered = stitch_registered_tiles(tiles, max_correction_px=12, min_overlap_px=16)
    timings["registration_blend_ms"] = (time.perf_counter() - t0) * 1000.0
    return TimedResult(image, registered, timings, True, "", {"native_method": "lvp_current_overlap"})


def method_lvp_simple_grid(dataset_dir: pathlib.Path) -> TimedResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    tiles, _ = load_tiles(dataset_dir)
    timings["read_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    registered = []
    for tile in tiles:
        item = dict(tile)
        item["registered_x_px"] = int(tile["x_px"])
        item["registered_y_px"] = int(tile["y_px"])
        item["registration_offset_x_px"] = 0
        item["registration_offset_y_px"] = 0
        item["registration_score"] = ""
        registered.append(item)
    image = blend_registered_tiles(registered)
    timings["registration_blend_ms"] = (time.perf_counter() - t0) * 1000.0
    return TimedResult(image, registered, timings, True, "", {"native_method": "lvp_simple_grid"})


def method_fft_phase_correlation(dataset_dir: pathlib.Path) -> TimedResult:
    return _method_custom_registration(dataset_dir, "fft_phase_correlation", estimate_phase_offset)


def method_coarse_to_fine_ncc(dataset_dir: pathlib.Path) -> TimedResult:
    return _method_custom_registration(dataset_dir, "coarse_to_fine_ncc", estimate_coarse_to_fine_ncc)


def _method_custom_registration(
    dataset_dir: pathlib.Path,
    method: str,
    estimator: Callable[..., tuple[int, int, float]],
) -> TimedResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    tiles, _ = load_tiles(dataset_dir)
    timings["read_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    registered = align_with_estimator(tiles, estimator, max_correction_px=24, min_overlap_px=16)
    timings["registration_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    image = blend_registered_tiles(registered)
    timings["blend_ms"] = (time.perf_counter() - t0) * 1000.0
    return TimedResult(image, registered, timings, True, "", {"native_method": method})


def method_opencv_feature(dataset_dir: pathlib.Path) -> TimedResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    tiles, _ = load_tiles(dataset_dir)
    feature_images = []
    feature_counts = []
    orb = cv2.ORB_create(nfeatures=750)
    for tile in tiles:
        image = tile["tile"]
        if image.dtype != np.uint8:
            arr = image.astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            image_u8 = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        else:
            image_u8 = image
        if image_u8.ndim == 2:
            bgr = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image_u8[:, :, :3], cv2.COLOR_RGB2BGR)
        feature_images.append(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        feature_counts.append(len(orb.detect(gray, None)))
    timings["read_ms"] = (time.perf_counter() - t0) * 1000.0
    if len(feature_images) < 2:
        return TimedResult(
            None,
            [],
            timings,
            False,
            "OpenCV feature stitch needs at least 2 tiles",
            {"native_method": "opencv_feature", "feature_keypoints_min": 0},
        )
    if not feature_counts or min(feature_counts) < 8:
        return TimedResult(
            None,
            [],
            timings,
            False,
            f"OpenCV feature stitch skipped: insufficient keypoints min={min(feature_counts) if feature_counts else 0}",
            {
                "native_method": "opencv_feature",
                "feature_keypoints_min": min(feature_counts) if feature_counts else 0,
                "feature_keypoints_mean": float(np.mean(feature_counts)) if feature_counts else 0.0,
            },
        )
    t0 = time.perf_counter()
    try:
        stitched_bgr = feature_stitch(feature_images, n_results=1)
    except cv2.error as exc:
        timings["feature_match_blend_ms"] = (time.perf_counter() - t0) * 1000.0
        return TimedResult(
            None,
            [],
            timings,
            False,
            f"OpenCV feature stitch failed: {exc}",
            {
                "native_method": "opencv_feature",
                "feature_keypoints_min": min(feature_counts),
                "feature_keypoints_mean": float(np.mean(feature_counts)),
            },
        )
    timings["feature_match_blend_ms"] = (time.perf_counter() - t0) * 1000.0
    if stitched_bgr is None:
        return TimedResult(
            None,
            [],
            timings,
            False,
            "OpenCV feature stitch returned no image",
            {
                "native_method": "opencv_feature",
                "feature_keypoints_min": min(feature_counts),
                "feature_keypoints_mean": float(np.mean(feature_counts)),
            },
        )
    image = cv2.cvtColor(stitched_bgr, cv2.COLOR_BGR2RGB)
    return TimedResult(
        image,
        [],
        timings,
        True,
        "",
        {
            "native_method": "opencv_feature",
            "feature_keypoints_min": min(feature_counts),
            "feature_keypoints_mean": float(np.mean(feature_counts)),
        },
    )


METHOD_FUNCS: dict[str, Callable[[pathlib.Path], TimedResult]] = {
    "lvp_current_overlap": method_lvp_current_overlap,
    "lvp_simple_grid": method_lvp_simple_grid,
    "fft_phase_correlation": method_fft_phase_correlation,
    "coarse_to_fine_ncc": method_coarse_to_fine_ncc,
    "opencv_feature": method_opencv_feature,
}


def _measure_method(func: Callable[[pathlib.Path], TimedResult], dataset_dir: pathlib.Path) -> tuple[TimedResult, dict[str, float]]:
    gc.collect()
    process = psutil.Process()
    samples: queue.Queue[float] = queue.Queue()
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                samples.put(process.memory_info().rss / (1024 * 1024))
            except Exception:
                pass
            time.sleep(0.005)

    tracemalloc.start()
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    try:
        result = func(dataset_dir)
    finally:
        total_ms = (time.perf_counter() - t0) * 1000.0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stop.set()
        thread.join(timeout=1.0)
    rss_values = list(samples.queue)
    memory = {
        "total_ms": total_ms,
        "peak_tracemalloc_mb": peak / (1024 * 1024),
        "final_tracemalloc_mb": current / (1024 * 1024),
        "peak_rss_mb": max(rss_values) if rss_values else process.memory_info().rss / (1024 * 1024),
    }
    return result, memory


def run_with_fallback(method: str, dataset_dir: pathlib.Path) -> tuple[TimedResult, dict[str, Any]]:
    attempts = [method]
    if method not in {"lvp_current_overlap", "lvp_simple_grid"}:
        attempts.append("lvp_current_overlap")
    if attempts[-1] != "lvp_simple_grid":
        attempts.append("lvp_simple_grid")

    first_error = ""
    fallback_from = ""
    for attempt in attempts:
        try:
            result, memory = _measure_method(METHOD_FUNCS[attempt], dataset_dir)
            if not result.status or result.image is None:
                raise ValueError(result.error or "method returned no image")
            used_fallback = attempt != method
            if used_fallback:
                fallback_from = fallback_from or method
                result.metadata["fallback_warning"] = f"FALLBACK: {fallback_from} -> {attempt}"
                result.metadata["fallback_reason"] = first_error
            result.timings.update(memory)
            result.metadata["requested_method"] = method
            result.metadata["actual_method"] = attempt
            result.metadata["used_fallback"] = used_fallback
            return result, {
                "used_fallback": used_fallback,
                "fallback_from": fallback_from if used_fallback else "",
                "fallback_to": attempt if used_fallback else "",
                "fallback_reason": first_error if used_fallback else "",
            }
        except Exception as exc:
            if not first_error:
                if isinstance(exc, ValueError):
                    first_error = str(exc)
                else:
                    first_error = f"{type(exc).__name__}: {exc}"
                fallback_from = attempt
            continue

    return (
        TimedResult(None, [], {"total_ms": 0.0}, False, first_error, {"requested_method": method}),
        {
            "used_fallback": False,
            "fallback_from": "",
            "fallback_to": "",
            "fallback_reason": first_error,
        },
    )


def _fit_to_gt(image: np.ndarray, gt_shape: tuple[int, ...]) -> np.ndarray:
    target_h, target_w = gt_shape[:2]
    out_shape = gt_shape if len(gt_shape) == image.ndim else gt_shape[:2]
    if len(out_shape) == 3 and image.ndim == 2:
        image = cv2.cvtColor(_normalize_preview(image, max_side=max(image.shape)), cv2.COLOR_RGB2GRAY)
    if len(out_shape) == 2 and image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]
    y0_src = max(0, (h - target_h) // 2)
    x0_src = max(0, (w - target_w) // 2)
    cropped = image[y0_src : y0_src + min(h, target_h), x0_src : x0_src + min(w, target_w)]
    out = np.zeros(out_shape, dtype=image.dtype)
    y0_dst = max(0, (target_h - cropped.shape[0]) // 2)
    x0_dst = max(0, (target_w - cropped.shape[1]) // 2)
    out[y0_dst : y0_dst + cropped.shape[0], x0_dst : x0_dst + cropped.shape[1], ...] = cropped
    return out


def _metric_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)


def compute_metrics(
    *,
    gt: np.ndarray,
    image: np.ndarray | None,
    registered_tiles: list[dict[str, Any]],
    manifest: pd.DataFrame,
) -> dict[str, float | str]:
    if image is None:
        return {
            "ssim": "",
            "psnr": "",
            "nrmse": "",
            "nmi": "",
            "seam_mae": "",
            "tile_error_mean_px": "",
            "tile_error_median_px": "",
            "tile_error_max_px": "",
            "registration_score_min": "",
            "registration_score_mean": "",
            "registration_failed_edges": "",
        }
    fitted = _fit_to_gt(image, gt.shape)
    gt_gray = _metric_gray(gt)
    im_gray = _metric_gray(fitted)
    data_range = float(np.iinfo(gt.dtype).max) if np.issubdtype(gt.dtype, np.integer) else float(gt_gray.max() - gt_gray.min() or 1.0)
    scores: dict[str, float | str] = {
        "ssim": float(sk_metrics.structural_similarity(gt_gray, im_gray, data_range=data_range)),
        "nrmse": float(sk_metrics.normalized_root_mse(gt_gray, im_gray)),
        "nmi": float(sk_metrics.normalized_mutual_information(gt_gray, im_gray)),
    }
    mse = float(np.mean((gt_gray.astype(np.float64) - im_gray.astype(np.float64)) ** 2))
    scores["psnr"] = float("inf") if mse == 0.0 else float(sk_metrics.peak_signal_noise_ratio(gt_gray, im_gray, data_range=data_range))

    bands = []
    included = manifest[manifest["included"].astype(bool)]
    for _, row in included.iterrows():
        x = int(row["nominal_x_px"])
        y = int(row["nominal_y_px"])
        w = int(row["width_px"])
        h = int(row["height_px"])
        overlap_w = max(1, int(round(w * OVERLAP)))
        overlap_h = max(1, int(round(h * OVERLAP)))
        if x > 0:
            bands.append(np.abs(gt_gray[y : y + h, x : x + overlap_w].astype(np.float32) - im_gray[y : y + h, x : x + overlap_w].astype(np.float32)))
        if y > 0:
            bands.append(np.abs(gt_gray[y : y + overlap_h, x : x + w].astype(np.float32) - im_gray[y : y + overlap_h, x : x + w].astype(np.float32)))
    scores["seam_mae"] = float(np.mean([band.mean() for band in bands])) if bands else ""

    true_by_idx = {
        int(row["tile_index"]): (float(row["actual_x_px"]), float(row["actual_y_px"]))
        for _, row in included.iterrows()
    }
    errors = []
    reg_scores = []
    failed_edges = []
    for tile in registered_tiles:
        idx = int(tile.get("tile_index", -1))
        if idx in true_by_idx and "registered_x_px" in tile and "registered_y_px" in tile:
            tx, ty = true_by_idx[idx]
            errors.append(math.hypot(float(tile["registered_x_px"]) - tx, float(tile["registered_y_px"]) - ty))
        value = tile.get("registration_score", "")
        if value != "":
            try:
                reg_scores.append(float(value))
            except Exception:
                pass
        if "registration_failed_edges" in tile:
            try:
                failed_edges.append(int(tile["registration_failed_edges"]))
            except Exception:
                pass
    scores["tile_error_mean_px"] = float(np.mean(errors)) if errors else ""
    scores["tile_error_median_px"] = float(np.median(errors)) if errors else ""
    scores["tile_error_max_px"] = float(np.max(errors)) if errors else ""
    scores["registration_score_min"] = float(np.min(reg_scores)) if reg_scores else ""
    scores["registration_score_mean"] = float(np.mean(reg_scores)) if reg_scores else ""
    scores["registration_failed_edges"] = int(max(failed_edges)) if failed_edges else ""
    return scores


def write_visual_grid(
    *,
    dataset_dir: pathlib.Path,
    result_items: list[dict[str, Any]],
    report_path: pathlib.Path,
) -> None:
    gt = read_image(dataset_dir / "ground_truth.tiff")
    panels = []
    title_h = 96

    def fmt(value: Any) -> str:
        try:
            if value == "":
                return ""
            return f"{float(value):.4g}"
        except Exception:
            return str(value)

    def panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
        preview = _normalize_preview(image)
        canvas = np.full((preview.shape[0] + title_h, max(preview.shape[1], 420), 3), 255, dtype=np.uint8)
        canvas[title_h : title_h + preview.shape[0], : preview.shape[1]] = preview
        y = 20
        for idx, line in enumerate(lines[:4]):
            color = (180, 0, 0) if "FALLBACK" in line else (0, 0, 0)
            cv2.putText(canvas, line[:78], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y += 22
        return canvas

    panels.append(panel(gt, ["GROUND TRUTH", dataset_dir.name]))
    for item in result_items:
        image = item.get("image")
        if image is None:
            image = np.zeros_like(gt)
        method = item["method"]
        warning = item.get("fallback_warning", "")
        lines = [
            method,
            warning if warning else f"native: {item.get('actual_method', method)}",
            f"time={fmt(item.get('total_ms', ''))}ms rss={fmt(item.get('peak_rss_mb', ''))}MB",
            f"ssim={fmt(item.get('ssim', ''))} nrmse={fmt(item.get('nrmse', ''))} tile={fmt(item.get('tile_error_mean_px', ''))}px",
        ]
        panels.append(panel(image, lines))

    width = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < width:
            q = np.full((p.shape[0], width, 3), 255, dtype=np.uint8)
            q[:, : p.shape[1]] = p
            p = q
        padded.append(p)
    grid = np.concatenate(padded, axis=0)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(report_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def benchmark_dataset(
    dataset_dir: pathlib.Path,
    *,
    results_root: pathlib.Path,
    reports_root: pathlib.Path,
    methods: list[str],
) -> list[dict[str, Any]]:
    config = json.loads((dataset_dir / "dataset_config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(dataset_dir / "tiles_manifest.csv")
    gt = read_image(dataset_dir / "ground_truth.tiff")
    dataset_results_dir = results_root / config["dataset_id"]
    dataset_results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    visuals = []
    for method in methods:
        print(f"benchmark {config['dataset_id']} method={method}", flush=True)
        result, fallback = run_with_fallback(method, dataset_dir)
        output_path = dataset_results_dir / f"{method}.tiff"
        write_ms = 0.0
        if result.image is not None:
            t0 = time.perf_counter()
            write_tiff(output_path, result.image)
            write_ms = (time.perf_counter() - t0) * 1000.0
        result.timings["write_ms"] = write_ms
        metrics = compute_metrics(gt=gt, image=result.image, registered_tiles=result.registered_tiles, manifest=manifest)
        row: dict[str, Any] = {
            **config,
            "dataset_dir": _rel(dataset_dir),
            "method": method,
            "actual_method": result.metadata.get("actual_method", ""),
            "status": result.status,
            "error": result.error,
            **fallback,
            **result.timings,
            **metrics,
            "output_path": _rel(output_path) if result.image is not None else "",
        }
        rows.append(row)
        visuals.append(
            {
                "method": method,
                "actual_method": result.metadata.get("actual_method", method),
                "fallback_warning": result.metadata.get("fallback_warning", ""),
                "image": result.image,
                **result.timings,
                **metrics,
            }
        )
        if fallback["used_fallback"]:
            print(f"WARNING {config['dataset_id']} {method}: FALLBACK {fallback['fallback_from']} -> {fallback['fallback_to']} ({fallback['fallback_reason']})", flush=True)

    visual_path = reports_root / "grids" / f"{config['dataset_id']}.png"
    write_visual_grid(dataset_dir=dataset_dir, result_items=visuals, report_path=visual_path)
    for row in rows:
        row["visual_grid_path"] = _rel(visual_path)
    return rows


def write_reports(rows: list[dict[str, Any]], reports_root: pathlib.Path) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        baseline = (
            df[df["method"] == "lvp_current_overlap"][
                ["dataset_id", "total_ms", "ssim", "nrmse", "tile_error_mean_px"]
            ]
            .rename(
                columns={
                    "total_ms": "baseline_lvp_total_ms",
                    "ssim": "baseline_lvp_ssim",
                    "nrmse": "baseline_lvp_nrmse",
                    "tile_error_mean_px": "baseline_lvp_tile_error_mean_px",
                }
            )
            .drop_duplicates("dataset_id")
        )
        df = df.merge(baseline, on="dataset_id", how="left")
        df["speedup_vs_lvp_current_overlap"] = df["baseline_lvp_total_ms"] / df["total_ms"]
        df["ssim_delta_vs_lvp_current_overlap"] = df["ssim"] - df["baseline_lvp_ssim"]
        df["nrmse_delta_vs_lvp_current_overlap"] = df["nrmse"] - df["baseline_lvp_nrmse"]
        df["tile_count"] = df["grid"].astype(float) * df["grid"].astype(float)
        df["total_ms_per_tile"] = df["total_ms"] / df["tile_count"]
        df["estimated_96well_10x10_total_ms"] = df["total_ms_per_tile"] * 96 * 100
        df["estimated_96well_10x10_total_min"] = df["estimated_96well_10x10_total_ms"] / 60000.0
    csv_path = reports_root / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)

    fallback_df = df[df["used_fallback"].astype(bool)] if "used_fallback" in df else pd.DataFrame()
    html_rows = []
    for _, row in df.sort_values(["dataset_id", "used_fallback", "method"], ascending=[True, False, True]).iterrows():
        warning = ""
        if bool(row.get("used_fallback", False)):
            warning = f"<strong style='color:#b00020'>FALLBACK: {html.escape(str(row.get('fallback_from', '')))} -> {html.escape(str(row.get('fallback_to', '')))}</strong>"
        visual = row.get("visual_grid_path", "")
        visual_link = f"<a href='../{html.escape(str(visual))}'>grid</a>" if visual else ""
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('dataset_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('method', '')))}</td>"
            f"<td>{html.escape(str(row.get('actual_method', '')))}</td>"
            f"<td>{warning}</td>"
            f"<td>{row.get('total_ms', '')}</td>"
            f"<td>{row.get('peak_rss_mb', '')}</td>"
            f"<td>{row.get('ssim', '')}</td>"
            f"<td>{row.get('nrmse', '')}</td>"
            f"<td>{row.get('tile_error_mean_px', '')}</td>"
            f"<td>{row.get('speedup_vs_lvp_current_overlap', '')}</td>"
            f"<td>{row.get('estimated_96well_10x10_total_min', '')}</td>"
            f"<td>{visual_link}</td>"
            "</tr>"
        )

    _write_optimization_direction(df, reports_root)

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stitching Benchmark Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>Stitching Benchmark Summary</h1>
  <p>Rows with fallbacks are explicitly marked and should not be treated as native method successes.</p>
  <p>Fallback rows: {len(fallback_df)} / {len(df)}</p>
  <table>
    <thead>
      <tr>
        <th>Dataset</th><th>Requested</th><th>Actual</th><th>Fallback</th>
        <th>Total ms</th><th>Peak RSS MB</th><th>SSIM</th><th>NRMSE</th>
        <th>Tile mean err px</th><th>Speedup vs LVP</th><th>Est 96-well 10x10 min</th><th>Visual</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (reports_root / "benchmark_summary.html").write_text(html_doc, encoding="utf-8")
    print(f"wrote {_rel(csv_path)}", flush=True)
    print(f"wrote {_rel(reports_root / 'benchmark_summary.html')}", flush=True)


def _write_optimization_direction(df: pd.DataFrame, reports_root: pathlib.Path) -> None:
    if df.empty:
        body = "# Stitching Optimization Direction\n\nNo benchmark rows were produced.\n"
    else:
        fallback_count = int(df["used_fallback"].astype(bool).sum()) if "used_fallback" in df else 0
        native = df[~df["used_fallback"].astype(bool)] if "used_fallback" in df else df
        method_summary = (
            native.groupby("method", dropna=False)
            .agg(
                runs=("method", "count"),
                median_total_ms=("total_ms", "median"),
                median_peak_rss_mb=("peak_rss_mb", "median"),
                median_ssim=("ssim", "median"),
                median_nrmse=("nrmse", "median"),
                median_speedup_vs_lvp=("speedup_vs_lvp_current_overlap", "median"),
                median_est_96well_10x10_min=("estimated_96well_10x10_total_min", "median"),
            )
            .reset_index()
            .sort_values("median_speedup_vs_lvp", ascending=False)
        )
        summary_md = _markdown_table(method_summary)
        body = f"""# Stitching Optimization Direction

Rows benchmarked: {len(df)}
Fallback rows: {fallback_count}

## Baseline Interpretation

`lvp_current_overlap` is the reference method. Experimental methods are useful
only if they are faster than this baseline while staying close on SSIM/NRMSE and
tile-position error, and while avoiding fallbacks.

## Native Method Summary

{summary_md}

## Recommended Direction

1. Use the current LVP overlap stitcher as the quality baseline.
2. Prefer FFT phase correlation if it consistently shows speedup with low tile
   error on `realistic` and `stress` tiers.
3. Prefer coarse-to-fine NCC if it preserves LVP-like registration quality but
   beats current local NCC on larger `5x5` and `7x7` grids.
4. Parallelize neighbor-pair registration only after the report confirms
   registration dominates total time.
5. Treat GPU acceleration as later-stage research for this MacBook; the first
   practical target is CPU/OpenCV/vectorized registration plus lower allocation
   pressure.

## Citations Used For Direction

- Fiji Grid/Collection Stitching: https://imagej.net/plugins/grid-collection-stitching
- BigStitcher: https://imagej.net/plugins/bigstitcher
- MIST: https://www.nature.com/articles/s41598-017-04567-y
- OpenCV phase correlation: https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html
- scikit-image metrics: https://scikit-image.org/docs/stable/api/skimage.metrics.html
"""
    path = reports_root / "optimization_direction.md"
    path.write_text(body, encoding="utf-8")
    print(f"wrote {_rel(path)}", flush=True)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No native method rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_benchmarks(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets_root = pathlib.Path(args.datasets_root).resolve()
    results_root = pathlib.Path(args.results_root).resolve()
    reports_root = pathlib.Path(args.reports_root).resolve()
    configs = sorted(datasets_root.glob("*/dataset_config.json"))
    if args.dataset:
        wanted = set(args.dataset)
        configs = [path for path in configs if path.parent.name in wanted]
    if args.limit:
        configs = configs[: args.limit]
    rows: list[dict[str, Any]] = []
    for config_path in configs:
        rows.extend(
            benchmark_dataset(
                config_path.parent,
                results_root=results_root,
                reports_root=reports_root,
                methods=args.methods,
            )
        )
    write_reports(rows, reports_root)
    return rows


def _add_common_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(GENERATED_ROOT))
    parser.add_argument("--grids", type=int, nargs="+", default=list(GRID_SIZES))
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    parser.add_argument("--source-limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic tile datasets")
    _add_common_generate_args(gen)

    run = sub.add_parser("run", help="Run benchmarks for generated datasets")
    run.add_argument("--datasets-root", default=str(GENERATED_ROOT))
    run.add_argument("--results-root", default=str(RESULTS_ROOT))
    run.add_argument("--reports-root", default=str(REPORTS_ROOT))
    run.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    run.add_argument("--dataset", nargs="+", default=[])
    run.add_argument("--limit", type=int, default=0)

    pipe = sub.add_parser("pipeline", help="Generate datasets and run benchmarks")
    _add_common_generate_args(pipe)
    pipe.add_argument("--results-root", default=str(RESULTS_ROOT))
    pipe.add_argument("--reports-root", default=str(REPORTS_ROOT))
    pipe.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    pipe.add_argument("--run-limit", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "generate":
        generate_all(args)
    elif args.cmd == "run":
        run_benchmarks(args)
    elif args.cmd == "pipeline":
        generated = generate_all(args)
        generated_names = [path.name for path in generated]
        if args.run_limit:
            generated_names = generated_names[: args.run_limit]
        run_args = argparse.Namespace(
            datasets_root=args.output_root,
            results_root=args.results_root,
            reports_root=args.reports_root,
            methods=args.methods,
            dataset=generated_names,
            limit=0,
        )
        run_benchmarks(run_args)
    else:
        parser.error(f"unknown command {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
