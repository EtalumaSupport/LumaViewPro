# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""GUI-free stitching orchestration helpers.

The production Stitcher post-processor owns grouping and record keeping; this
module owns per-group algorithm selection. The overlap-registration math stays
in modules.stitch_algorithms so the newest sparse-grid registration and
float32 average-blend behavior is shared by all callers.
"""

import logging
import pathlib
from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.stitch_algorithms import (
    crop_to_content,
    feature_stitch,
    stitch_registered_tiles,
)

logger = logging.getLogger('LVP.modules.stitching_core')


def _center_metadata(df: pd.DataFrame) -> dict:
    x_center = df['X'].unique().mean()
    y_center = df['Y'].unique().mean()
    return {
        'x': round(x_center, common_utils.max_decimal_precision(parameter='x')),
        'y': round(y_center, common_utils.max_decimal_precision(parameter='y')),
    }


def _read_tile(path: pathlib.Path, filename: str) -> np.ndarray:
    return tf.imread(str(path / filename))


def _write_output(
    *,
    path: pathlib.Path,
    output_file_loc: pathlib.Path | None,
    image: np.ndarray,
    first_tile_path: pathlib.Path,
    color: str,
    center: dict,
) -> np.ndarray | None:
    if output_file_loc is None:
        return image

    output_file_loc_abs = path / output_file_loc
    output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
    metadata = image_utils.build_postproc_output_metadata(
        input_path=first_tile_path,
        channel=color,
        plate_pos_mm_override=center,
    )
    image_utils.write_tiff(
        data=image,
        file_loc=output_file_loc_abs,
        metadata=metadata,
        ome=False,
        color=color,
    )
    return None


def _result(
    *,
    image: np.ndarray,
    path: pathlib.Path,
    output_file_loc: pathlib.Path | None,
    df: pd.DataFrame,
    metadata: dict,
) -> dict:
    first_row = df.iloc[0]
    color = str(first_row.get('Color', ''))
    center = metadata['center']
    return_image = _write_output(
        path=path,
        output_file_loc=output_file_loc,
        image=image,
        first_tile_path=path / first_row['Filepath'],
        color=color,
        center=center,
    )
    return {
        'status': True,
        'error': None,
        'image': return_image,
        'metadata': metadata,
    }


def _failure(algorithm: str, error: str, center: dict | None = None) -> dict:
    metadata = {'algorithm': algorithm}
    if center is not None:
        metadata['center'] = center
    return {
        'status': False,
        'error': error,
        'image': None,
        'metadata': metadata,
    }


def _failure_reason(failures: list[dict[str, str]]) -> str:
    return '; '.join(f"{item['algorithm']}: {item['error']}" for item in failures)


def _run_fallback_chain(
    chain: list[tuple[str, Callable[[], dict]]],
    context: dict | None = None,
) -> dict:
    failures: list[dict[str, str]] = []
    last_result: dict | None = None
    context = context or {}

    for algorithm, runner in chain:
        result = runner()
        last_result = result
        if bool(result.get('status')):
            metadata = result.setdefault('metadata', {})
            metadata.setdefault('algorithm', algorithm)
            if failures:
                metadata['fallback_from'] = failures[0]['algorithm']
                metadata['fallback_failures'] = failures
                metadata['fallback_reason'] = _failure_reason(failures)
                logger.warning(
                    '[Stitch] %s failed; using %s for well=%s color=%s tile_group=%s. Reason: %s',
                    failures[0]['algorithm'],
                    metadata.get('algorithm', algorithm),
                    context.get('well', ''),
                    context.get('color', ''),
                    context.get('tile_group_id', ''),
                    metadata['fallback_reason'],
                )
            return result

        failures.append(
            {
                'algorithm': algorithm,
                'error': result.get('error') or result.get('message') or 'no image returned',
            }
        )

    if last_result is None:
        return _failure('channel_aware_stitcher', 'no stitching algorithms configured')
    last_result.setdefault('metadata', {})['fallback_failures'] = failures
    last_result['metadata']['fallback_reason'] = _failure_reason(failures)
    return last_result


def _to_uint8_bgr_for_feature_stitch(image: np.ndarray) -> np.ndarray:
    """Convert microscope tiles to BGR uint8 for OpenCV's Stitcher."""
    if image.dtype == np.uint8:
        image_u8 = image
    else:
        image_f = image.astype(np.float32)
        finite = np.isfinite(image_f)
        lo = float(image_f[finite].min()) if finite.any() else 0.0
        hi = float(image_f[finite].max()) if finite.any() else 1.0
        if hi <= lo:
            hi = lo + 1.0
        image_u8 = (np.clip((image_f - lo) / (hi - lo), 0.0, 1.0) * 255).astype(
            np.uint8
        )

    if image_u8.ndim == 2:
        return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    if image_u8.ndim == 3 and image_u8.shape[2] == 3:
        return cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    if image_u8.ndim == 3 and image_u8.shape[2] == 4:
        return cv2.cvtColor(image_u8, cv2.COLOR_RGBA2BGR)
    raise ValueError(f'Unsupported image shape for feature stitch: {image.shape}')


def bf_feature_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Use OpenCV feature stitching for brightfield groups."""
    center = _center_metadata(df)
    try:
        feature_images = [
            _to_uint8_bgr_for_feature_stitch(_read_tile(path, row['Filepath']))
            for _, row in df.iterrows()
        ]
        stitched_img = feature_stitch(feature_images)
        if stitched_img is None:
            return _failure('bf_feature_stitcher', 'BF feature stitching failed', center)
        stitched_img = crop_to_content(stitched_img)
        stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        return _failure(
            'bf_feature_stitcher',
            f'BF feature stitching failed: {type(exc).__name__}: {exc}',
            center,
        )

    return _result(
        image=stitched_img,
        path=path,
        output_file_loc=output_file_loc,
        df=df,
        metadata={'center': center, 'algorithm': 'bf_feature_stitcher'},
    )


def overlap_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    pixel_size_um: float | None,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Use the newest stage-position + overlap-registration stitch math."""
    center = _center_metadata(df)
    if pixel_size_um is None or pixel_size_um <= 0:
        return _failure('overlap_stitcher', 'pixel_size_um must be greater than 0', center)

    try:
        frame = df.copy()
        frame['X'] = frame['X'].astype(float)
        frame['Y'] = frame['Y'].astype(float)

        images = {
            row['Filepath']: _read_tile(path, row['Filepath'])
            for _, row in frame.iterrows()
        }
        sample_row = frame.iloc[0]
        sample = images[sample_row['Filepath']]
        image_h = sample.shape[0]
        image_w = sample.shape[1]

        x_max = frame['X'].max()
        y_min = frame['Y'].min()
        frame['x_pix'] = ((x_max - frame['X']) * 1000 / pixel_size_um).round().astype(int)
        frame['y_pix'] = ((frame['Y'] - y_min) * 1000 / pixel_size_um).round().astype(int)

        if int(frame['x_pix'].max() + image_w) <= 0 or int(
            frame['y_pix'].max() + image_h
        ) <= 0:
            return _failure('overlap_stitcher', 'invalid stitched image dimensions', center)

        tiles = [
            {
                'tile': images[row['Filepath']],
                'x_px': int(row['x_pix']),
                'y_px': int(row['y_pix']),
            }
            for _, row in frame.iterrows()
        ]
        stitched_img, registered_tiles = stitch_registered_tiles(tiles)
    except Exception as exc:
        return _failure(
            'overlap_stitcher',
            f'overlap stitching failed: {type(exc).__name__}: {exc}',
            center,
        )

    return _result(
        image=stitched_img,
        path=path,
        output_file_loc=output_file_loc,
        df=df,
        metadata={
            'center': center,
            'algorithm': 'overlap_stitcher',
            'pixel_size_um': pixel_size_um,
            'registered_tiles': registered_tiles,
        },
    )


def stage_position_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    pixel_size_um: float | None,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Place tiles by recorded stage position without registration correction."""
    center = _center_metadata(df)
    if pixel_size_um is None or pixel_size_um <= 0:
        return _failure('stage_position_stitcher', 'pixel_size_um must be greater than 0', center)

    try:
        frame = df.copy()
        frame['X'] = frame['X'].astype(float)
        frame['Y'] = frame['Y'].astype(float)
        images = {
            row['Filepath']: _read_tile(path, row['Filepath'])
            for _, row in frame.iterrows()
        }
        sample = images[frame.iloc[0]['Filepath']]
        image_h = sample.shape[0]
        image_w = sample.shape[1]

        x_max = frame['X'].max()
        y_min = frame['Y'].min()
        frame['x_pix'] = ((x_max - frame['X']) * 1000 / pixel_size_um).round().astype(int)
        frame['y_pix'] = ((frame['Y'] - y_min) * 1000 / pixel_size_um).round().astype(int)

        min_x = int(frame['x_pix'].min())
        min_y = int(frame['y_pix'].min())
        max_x = int(frame['x_pix'].max() + image_w)
        max_y = int(frame['y_pix'].max() + image_h)
        if max_x <= min_x or max_y <= min_y:
            return _failure('stage_position_stitcher', 'invalid stitched image dimensions', center)

        if image_utils.is_color_image(sample):
            stitched_img = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=sample.dtype)
        else:
            stitched_img = np.zeros((max_y - min_y, max_x - min_x), dtype=sample.dtype)

        placements = []
        for _, row in frame.iterrows():
            image = images[row['Filepath']]
            x0 = int(row['x_pix']) - min_x
            y0 = int(row['y_pix']) - min_y
            x1 = x0 + image.shape[1]
            y1 = y0 + image.shape[0]
            if image.ndim == 3:
                stitched_img[y0:y1, x0:x1, :] = image
            else:
                stitched_img[y0:y1, x0:x1] = image
            placements.append(
                {
                    'filepath': row['Filepath'],
                    'canvas_x_px': x0,
                    'canvas_y_px': y0,
                    'width_px': image.shape[1],
                    'height_px': image.shape[0],
                }
            )
    except Exception as exc:
        return _failure(
            'stage_position_stitcher',
            f'stage-position stitching failed: {type(exc).__name__}: {exc}',
            center,
        )

    return _result(
        image=stitched_img,
        path=path,
        output_file_loc=output_file_loc,
        df=df,
        metadata={
            'center': center,
            'algorithm': 'stage_position_stitcher',
            'pixel_size_um': pixel_size_um,
            'placements': placements,
        },
    )


def simple_position_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Place tiles by X/Y rank, preserving the legacy no-overlap behavior."""
    try:
        frame = df.copy()
        num_x_tiles = frame['X'].nunique()
        num_y_tiles = frame['Y'].nunique()
        center = _center_metadata(frame)

        sample_row = frame.iloc[0]
        sample_filename = sample_row['Filepath']
        sample = _read_tile(path, sample_filename)
        source_image_h = sample.shape[0]
        source_image_w = sample.shape[1]

        frame = frame.sort_values(['X', 'Y'], ascending=False)
        frame['x_index'] = frame.groupby(by=['X']).ngroup()
        frame['y_index'] = frame.groupby(by=['Y']).ngroup()
        frame['x_pix_range'] = frame['x_index'] * source_image_w
        frame['y_pix_range'] = frame['y_index'] * source_image_h

        stitched_im_x = source_image_w * num_x_tiles
        stitched_im_y = source_image_h * num_y_tiles

        reverse_x = True
        reverse_y = False
        if reverse_x:
            frame['x_pix_range'] = stitched_im_x - frame['x_pix_range']
        if reverse_y:
            frame['y_pix_range'] = stitched_im_y - frame['y_pix_range']

        if image_utils.is_color_image(sample):
            stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype=sample.dtype)
        else:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x), dtype=sample.dtype)

        for _, row in frame.iterrows():
            image = _read_tile(path, row['Filepath'])
            im_x = image.shape[1]
            im_y = image.shape[0]
            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    if image.ndim == 3:
                        stitched_img[y_val - im_y : y_val, x_val - im_x : x_val, :] = image
                    else:
                        stitched_img[y_val - im_y : y_val, x_val - im_x : x_val] = image
                else:
                    if image.ndim == 3:
                        stitched_img[y_val - im_y : y_val, x_val : x_val + im_x, :] = image
                    else:
                        stitched_img[y_val - im_y : y_val, x_val : x_val + im_x] = image
            else:
                if reverse_x:
                    if image.ndim == 3:
                        stitched_img[y_val : y_val + im_y, x_val - im_x : x_val, :] = image
                    else:
                        stitched_img[y_val : y_val + im_y, x_val - im_x : x_val] = image
                else:
                    if image.ndim == 3:
                        stitched_img[y_val : y_val + im_y, x_val : x_val + im_x, :] = image
                    else:
                        stitched_img[y_val : y_val + im_y, x_val : x_val + im_x] = image
    except Exception as exc:
        return _failure(
            'simple_position_stitcher',
            f'simple-position stitching failed: {type(exc).__name__}: {exc}',
            None,
        )

    return _result(
        image=stitched_img,
        path=path,
        output_file_loc=output_file_loc,
        df=df,
        metadata={'center': center, 'algorithm': 'simple_position_stitcher'},
    )


def channel_aware_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    pixel_size_um: float | None = None,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Route channels through the preferred algorithm and fallbacks.

    BF: feature stitch -> overlap registration -> stage-position -> simple grid.
    Fluorescence/other: overlap registration -> stage-position -> simple grid.
    """
    color = str(df.iloc[0].get('Color', ''))
    shared = {
        'path': path,
        'df': df,
        'output_file_loc': output_file_loc,
    }
    chain: list[tuple[str, Callable[[], dict]]] = []
    if color == 'BF':
        chain.append(
            (
                'bf_feature_stitcher',
                lambda: bf_feature_stitcher(**shared),
            )
        )
    chain.extend(
        [
            (
                'overlap_stitcher',
                lambda: overlap_stitcher(**shared, pixel_size_um=pixel_size_um),
            ),
            (
                'stage_position_stitcher',
                lambda: stage_position_stitcher(**shared, pixel_size_um=pixel_size_um),
            ),
            (
                'simple_position_stitcher',
                lambda: simple_position_stitcher(**shared),
            ),
        ]
    )
    row0 = df.iloc[0]
    return _run_fallback_chain(
        chain,
        context={
            'well': row0.get('Well', ''),
            'color': row0.get('Color', ''),
            'tile_group_id': row0.get('Tile Group ID', ''),
        },
    )
