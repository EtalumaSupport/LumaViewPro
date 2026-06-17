#!/usr/bin/env python3
"""Run position-aware vs simple tiling stitch on a captured sample folder."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.stitcher import Stitcher


def _read_lvp_protocol(path: pathlib.Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        steps_idx = lines.index("Steps")
    except ValueError as exc:
        raise RuntimeError(f"{path} does not contain a Steps section") from exc

    header_idx = steps_idx + 1
    rows = [line.split("\t") for line in lines[header_idx:] if line.strip()]
    if not rows:
        raise RuntimeError(f"{path} has no step rows")

    header, data = rows[0], rows[1:]
    return pd.DataFrame(data, columns=header)


def _filename_for_step(sample_dir: pathlib.Path, name: str) -> str | None:
    matches = sorted(sample_dir.glob(f"{name}_*.tif*"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"multiple TIFFs found for protocol step {name!r}: {matches}")
    return matches[0].name


def _prep_stitch_df(sample_dir: pathlib.Path, protocol_df: pd.DataFrame) -> pd.DataFrame:
    df = protocol_df.copy()
    df = df[df["Acquire"].str.lower() == "image"].copy()
    df["Filepath"] = [_filename_for_step(sample_dir, name) for name in df["Name"]]
    df = df[df["Filepath"].notna()].copy()
    for col in ("X", "Y"):
        df[col] = df[col].astype(float)
    return df[["Filepath", "X", "Y", "Objective", "Color", "Well", "Tile Group ID"]]


def _normalize_preview(image: np.ndarray) -> np.ndarray:
    arr = image
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = arr.astype(np.float32)
    p_low, p_high = np.percentile(arr, [1, 99.5])
    if p_high <= p_low:
        p_low, p_high = float(arr.min()), float(arr.max())
    if p_high <= p_low:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
    return (arr * 255).astype(np.uint8)


def _thumbnail(image: np.ndarray, max_width: int = 1800) -> np.ndarray:
    preview = _normalize_preview(image)
    if preview.shape[1] <= max_width:
        return preview
    scale = max_width / preview.shape[1]
    size = (max_width, max(1, int(preview.shape[0] * scale)))
    return cv2.resize(preview, size, interpolation=cv2.INTER_AREA)


def _comparison_canvas(simple: np.ndarray, position: np.ndarray) -> np.ndarray:
    simple_thumb = _thumbnail(simple)
    position_thumb = _thumbnail(position)
    height = max(simple_thumb.shape[0], position_thumb.shape[0])

    def pad_to_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == height:
            return image
        padded = np.zeros((height, image.shape[1]), dtype=image.dtype)
        padded[: image.shape[0], :] = image
        return padded

    divider = np.full((height, 12), 255, dtype=np.uint8)
    return np.concatenate([pad_to_height(simple_thumb), divider, pad_to_height(position_thumb)], axis=1)


def _write_registration_csv(path: pathlib.Path, registered_tiles: list[dict]) -> None:
    fields = [
        "index",
        "x_px",
        "y_px",
        "registered_x_px",
        "registered_y_px",
        "registration_offset_x_px",
        "registration_offset_y_px",
        "registration_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for idx, tile in enumerate(registered_tiles):
            row = {field: tile.get(field, "") for field in fields}
            row["index"] = idx
            writer.writerow(row)


def run(sample_dir: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading protocol from: {sample_dir / 'unsaved_protocol.tsv'}", flush=True)
    protocol_df = _read_lvp_protocol(sample_dir / "unsaved_protocol.tsv")
    stitch_df = _prep_stitch_df(sample_dir, protocol_df)
    print(f"Prepared {len(stitch_df)} image rows", flush=True)

    stitcher = Stitcher(has_turret=False)
    summary_rows = []

    for (well, tile_group_id), group in stitch_df.groupby(["Well", "Tile Group ID"], sort=True):
        group = group.sort_values(["Y", "X"]).reset_index(drop=True)
        print(f"Stitching well {well}, group {tile_group_id}, {len(group)} tiles", flush=True)

        position_result = stitcher._position_stitcher(
            path=sample_dir,
            df=group[["Filepath", "X", "Y", "Objective", "Color"]],
        )
        simple_result = Stitcher._simple_position_stitcher(
            path=sample_dir,
            df=group[["Filepath", "X", "Y", "Color"]],
        )
        if not position_result["status"]:
            raise RuntimeError(f"position-aware stitch failed for {well}: {position_result['error']}")
        if not simple_result["status"]:
            raise RuntimeError(f"simple stitch failed for {well}: {simple_result['error']}")

        position_image = position_result["image"]
        simple_image = simple_result["image"]

        prefix = f"{well}_group{tile_group_id}"
        simple_path = output_dir / f"{prefix}_simple_grid_original_method.tiff"
        position_path = output_dir / f"{prefix}_position_aware_current_method.tiff"
        compare_path = output_dir / f"{prefix}_comparison_simple_left_current_right.png"
        reg_path = output_dir / f"{prefix}_position_registration_offsets.csv"

        tf.imwrite(simple_path, simple_image, compression="lzw")
        tf.imwrite(position_path, position_image, compression="lzw")
        cv2.imwrite(str(compare_path), _comparison_canvas(simple_image, position_image))
        _write_registration_csv(reg_path, position_result["metadata"]["registered_tiles"])

        summary_rows.append(
            {
                "well": well,
                "tile_group_id": tile_group_id,
                "tile_count": len(group),
                "simple_shape": "x".join(map(str, simple_image.shape)),
                "position_aware_shape": "x".join(map(str, position_image.shape)),
                "simple_output": simple_path.name,
                "position_aware_output": position_path.name,
                "comparison_preview": compare_path.name,
                "registration_offsets": reg_path.name,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-dir",
        type=pathlib.Path,
        default=pathlib.Path("/Users/mariamhusain/Downloads/20260611_074617"),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = sample_dir / "tiling_sample"
    output_dir = output_dir.expanduser().resolve()

    summary_path = run(sample_dir=sample_dir, output_dir=output_dir)
    print(f"Wrote stitching comparison outputs to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
