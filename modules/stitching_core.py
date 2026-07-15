# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""GUI-free stitching orchestration helpers.

The production Stitcher post-processor owns grouping and record keeping; this
module owns per-group algorithm selection. The overlap-registration math stays
in modules.stitch_algorithms so the newest sparse-grid registration and
float32 average-blend behavior is shared by all callers.
"""

import logging
import pathlib
import time
from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.stitch_algorithms import (
    crop_to_content,
    estimate_phase_offset,
    feature_stitch,
    stitch_registered_tiles,
)

logger = logging.getLogger('LVP.modules.stitching_core')


def _center_metadata(df: pd.DataFrame) -> dict:
    # Bounding-box midpoint, not the mean of unique positions: the stitched
    # image spans the full X/Y extent, so its center is (min + max) / 2.
    # Averaging unique positions drifts off-center on irregularly-spaced or
    # non-rectangular (sparse) grids, where the distinct coordinates are not
    # symmetric about the extent.
    x_center = (df['X'].min() + df['X'].max()) / 2
    y_center = (df['Y'].min() + df['Y'].max()) / 2
    return {
        'x': round(x_center, common_utils.max_decimal_precision(parameter='x')),
        'y': round(y_center, common_utils.max_decimal_precision(parameter='y')),
    }


def _read_tile_with_depth(path: pathlib.Path, filename: str) -> tuple[np.ndarray, int]:
    return image_utils.load_pixels(path / filename, collapse_legacy_false_color=False)


def _write_output(
    *,
    path: pathlib.Path,
    output_file_loc: pathlib.Path | None,
    image: np.ndarray,
    first_tile_path: pathlib.Path,
    color: str,
    center: dict,
    significant_bits: int,
    algorithm: str,
) -> np.ndarray | None:
    if output_file_loc is None:
        return image

    t0 = time.perf_counter()
    output_file_loc_abs = path / output_file_loc
    output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
    metadata = image_utils.build_postproc_output_metadata(
        input_path=first_tile_path,
        channel=color,
        significant_bits=significant_bits,
        plate_pos_mm_override=center,
        algorithm=algorithm,
    )
    image_utils.write_tiff(
        data=image,
        file_loc=output_file_loc_abs,
        metadata=metadata,
        ome=False,
        color=color,
        significant_bits=metadata['significant_bits'],
        save_encoding=image_utils.resolve_output_save_encoding(image),
    )
    logger.info(
        '[StitchPerf] write output %.1fms file=%s shape=%s dtype=%s',
        (time.perf_counter() - t0) * 1000.0,
        output_file_loc,
        getattr(image, 'shape', ''),
        getattr(image, 'dtype', ''),
    )
    return None


def _result(
    *,
    image: np.ndarray,
    path: pathlib.Path,
    output_file_loc: pathlib.Path | None,
    df: pd.DataFrame,
    metadata: dict,
    significant_bits: int,
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
        significant_bits=significant_bits,
        algorithm=metadata['algorithm'],
    )
    return {
        'status': True,
        'error': None,
        'image': return_image,
        'significant_bits': significant_bits,
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
    return '; '.join(f'{item["algorithm"]}: {item["error"]}' for item in failures)


def _run_fallback_chain(
    chain: list[tuple[str, Callable[[], dict]]],
    context: dict | None = None,
) -> dict:
    failures: list[dict[str, str]] = []
    last_result: dict | None = None
    context = context or {}

    for algorithm, runner in chain:
        t0 = time.perf_counter()
        result = runner()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        last_result = result
        logger.info(
            '[StitchPerf] algorithm %s finished in %.1fms status=%s well=%s color=%s tile_group=%s',
            algorithm,
            elapsed_ms,
            bool(result.get('status')),
            context.get('well', ''),
            context.get('color', ''),
            context.get('tile_group_id', ''),
        )
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


def _bgr_from_uint8(image_u8: np.ndarray) -> np.ndarray:
    """Present a uint8 tile as 3-channel BGR for OpenCV's Stitcher."""
    if image_u8.ndim == 2:
        return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    if image_u8.ndim == 3 and image_u8.shape[2] == 3:
        return cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    if image_u8.ndim == 3 and image_u8.shape[2] == 4:
        return cv2.cvtColor(image_u8, cv2.COLOR_RGBA2BGR)
    raise ValueError(f'Unsupported image shape for feature stitch: {image_u8.shape}')


def _feature_stitch_bgr_tiles(images: list[np.ndarray]) -> list[np.ndarray]:
    """Convert a tile GROUP to BGR uint8 for OpenCV's Stitcher, scaling every
    deep-input tile against one shared intensity range.

    Per-tile min/max normalization maps each tile's own extremes to 0..255, so
    a dim tile and a bright tile of the same specimen get different transfer
    functions and the shared seam shows a brightness step. Deep (12/16-bit)
    tiles are scaled against a group-wide lo/hi so intensities stay comparable
    across the montage; tiles already uint8 are display-ready and pass through
    unscaled.
    """
    deep = [image for image in images if image.dtype != np.uint8]
    if deep:
        finite_pixels = [np.asarray(image, dtype=np.float32) for image in deep]
        finite_pixels = [pixels[np.isfinite(pixels)].ravel() for pixels in finite_pixels]
        pooled = np.concatenate([pixels for pixels in finite_pixels if pixels.size])
        lo = float(pooled.min()) if pooled.size else 0.0
        hi = float(pooled.max()) if pooled.size else 1.0
        if hi <= lo:
            hi = lo + 1.0

    bgr_tiles = []
    for image in images:
        if image.dtype == np.uint8:
            image_u8 = image
        else:
            image_f = np.asarray(image, dtype=np.float32)
            image_u8 = (np.clip((image_f - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        bgr_tiles.append(_bgr_from_uint8(image_u8))
    return bgr_tiles


def bf_feature_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Use OpenCV feature stitching for brightfield groups."""
    center = _center_metadata(df)
    try:
        read_t0 = time.perf_counter()
        raw_images = [_read_tile_with_depth(path, row['Filepath'])[0] for _, row in df.iterrows()]
        feature_images = _feature_stitch_bgr_tiles(raw_images)
        logger.info(
            '[StitchPerf] bf_feature read+convert %.1fms tiles=%d',
            (time.perf_counter() - read_t0) * 1000.0,
            len(feature_images),
        )
        stitched_img = feature_stitch(feature_images)
        if stitched_img is None:
            return _failure('bf_feature_stitcher', 'BF feature stitching failed', center)
        crop_t0 = time.perf_counter()
        stitched_img = crop_to_content(stitched_img)
        stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
        logger.info(
            '[StitchPerf] bf_feature crop+convert %.1fms output_shape=%s dtype=%s',
            (time.perf_counter() - crop_t0) * 1000.0,
            stitched_img.shape,
            stitched_img.dtype,
        )
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
        # Couple the tag to the ACTUAL output: the OpenCV feature path emits
        # 8-bit BGR regardless of input depth. Tagging this uint8 montage with a
        # 16-bit input depth would render it ~256x dark on read-back.
        significant_bits=stitched_img.dtype.itemsize * 8,
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

        read_t0 = time.perf_counter()
        images = {}
        input_depths = []
        for _, row in frame.iterrows():
            image, significant_bits = _read_tile_with_depth(path, row['Filepath'])
            images[row['Filepath']] = image
            input_depths.append(significant_bits)
        tile_bytes = sum(int(image.nbytes) for image in images.values())
        logger.info(
            '[StitchPerf] overlap read %.1fms tiles=%d bytes=%d',
            (time.perf_counter() - read_t0) * 1000.0,
            len(images),
            tile_bytes,
        )
        sample_row = frame.iloc[0]
        sample = images[sample_row['Filepath']]
        image_h = sample.shape[0]
        image_w = sample.shape[1]

        x_max = frame['X'].max()
        y_min = frame['Y'].min()
        frame['x_pix'] = ((x_max - frame['X']) * 1000 / pixel_size_um).round().astype(int)
        frame['y_pix'] = ((frame['Y'] - y_min) * 1000 / pixel_size_um).round().astype(int)

        if int(frame['x_pix'].max() + image_w) <= 0 or int(frame['y_pix'].max() + image_h) <= 0:
            return _failure('overlap_stitcher', 'invalid stitched image dimensions', center)
        nominal_output_shape = (
            int(frame['y_pix'].max() + image_h),
            int(frame['x_pix'].max() + image_w),
        )

        tiles = [
            {
                'tile': images[row['Filepath']],
                'x_px': int(row['x_pix']),
                'y_px': int(row['y_pix']),
            }
            for _, row in frame.iterrows()
        ]
        stitch_t0 = time.perf_counter()
        stitched_img, registered_tiles = stitch_registered_tiles(
            tiles,
            output_shape=nominal_output_shape,
        )
        logger.info(
            '[StitchPerf] overlap register+blend %.1fms output_shape=%s dtype=%s',
            (time.perf_counter() - stitch_t0) * 1000.0,
            stitched_img.shape,
            stitched_img.dtype,
        )
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
        significant_bits=image_utils.resolve_output_depth(input_depths),
    )


def fft_phase_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    pixel_size_um: float | None,
    output_file_loc: pathlib.Path | None = None,
) -> dict:
    """Use stage-position placement plus FFT phase-correlation registration."""
    center = _center_metadata(df)
    if pixel_size_um is None or pixel_size_um <= 0:
        return _failure('fft_phase_stitcher', 'pixel_size_um must be greater than 0', center)

    try:
        frame = df.copy()
        frame['X'] = frame['X'].astype(float)
        frame['Y'] = frame['Y'].astype(float)

        read_t0 = time.perf_counter()
        images = {}
        input_depths = []
        for _, row in frame.iterrows():
            image, significant_bits = _read_tile_with_depth(path, row['Filepath'])
            images[row['Filepath']] = image
            input_depths.append(significant_bits)
        tile_bytes = sum(int(image.nbytes) for image in images.values())
        logger.info(
            '[StitchPerf] fft-phase read %.1fms tiles=%d bytes=%d',
            (time.perf_counter() - read_t0) * 1000.0,
            len(images),
            tile_bytes,
        )
        sample_row = frame.iloc[0]
        sample = images[sample_row['Filepath']]
        image_h = sample.shape[0]
        image_w = sample.shape[1]

        x_max = frame['X'].max()
        y_min = frame['Y'].min()
        frame['x_pix'] = ((x_max - frame['X']) * 1000 / pixel_size_um).round().astype(int)
        frame['y_pix'] = ((frame['Y'] - y_min) * 1000 / pixel_size_um).round().astype(int)

        if int(frame['x_pix'].max() + image_w) <= 0 or int(frame['y_pix'].max() + image_h) <= 0:
            return _failure('fft_phase_stitcher', 'invalid stitched image dimensions', center)
        nominal_output_shape = (
            int(frame['y_pix'].max() + image_h),
            int(frame['x_pix'].max() + image_w),
        )

        tiles = [
            {
                'tile': images[row['Filepath']],
                'x_px': int(row['x_pix']),
                'y_px': int(row['y_pix']),
            }
            for _, row in frame.iterrows()
        ]
        stitch_t0 = time.perf_counter()
        stitched_img, registered_tiles = stitch_registered_tiles(
            tiles,
            max_correction_px=24,
            min_overlap_px=16,
            output_shape=nominal_output_shape,
            estimator=estimate_phase_offset,
        )
        logger.info(
            '[StitchPerf] fft-phase register+blend %.1fms output_shape=%s dtype=%s',
            (time.perf_counter() - stitch_t0) * 1000.0,
            stitched_img.shape,
            stitched_img.dtype,
        )
    except Exception as exc:
        return _failure(
            'fft_phase_stitcher',
            f'FFT phase stitching failed: {type(exc).__name__}: {exc}',
            center,
        )

    return _result(
        image=stitched_img,
        path=path,
        output_file_loc=output_file_loc,
        df=df,
        metadata={
            'center': center,
            'algorithm': 'fft_phase_stitcher',
            'pixel_size_um': pixel_size_um,
            'registered_tiles': registered_tiles,
        },
        significant_bits=image_utils.resolve_output_depth(input_depths),
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
        read_t0 = time.perf_counter()
        images = {}
        input_depths = []
        for _, row in frame.iterrows():
            image, significant_bits = _read_tile_with_depth(path, row['Filepath'])
            images[row['Filepath']] = image
            input_depths.append(significant_bits)
        tile_bytes = sum(int(image.nbytes) for image in images.values())
        logger.info(
            '[StitchPerf] stage-position read %.1fms tiles=%d bytes=%d',
            (time.perf_counter() - read_t0) * 1000.0,
            len(images),
            tile_bytes,
        )
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

        place_t0 = time.perf_counter()
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
        logger.info(
            '[StitchPerf] stage-position place %.1fms output_shape=%s dtype=%s',
            (time.perf_counter() - place_t0) * 1000.0,
            stitched_img.shape,
            stitched_img.dtype,
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
        significant_bits=image_utils.resolve_output_depth(input_depths),
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
        sample, sample_depth = _read_tile_with_depth(path, sample_filename)
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

        place_t0 = time.perf_counter()
        tile_count = 0
        tile_bytes = 0
        input_depths = []
        for _, row in frame.iterrows():
            if row['Filepath'] == sample_filename:
                image, significant_bits = sample, sample_depth
            else:
                image, significant_bits = _read_tile_with_depth(path, row['Filepath'])
            input_depths.append(significant_bits)
            tile_count += 1
            tile_bytes += int(image.nbytes)
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
        logger.info(
            '[StitchPerf] simple-grid read+place %.1fms tiles=%d bytes=%d output_shape=%s dtype=%s',
            (time.perf_counter() - place_t0) * 1000.0,
            tile_count,
            tile_bytes,
            stitched_img.shape,
            stitched_img.dtype,
        )
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
        significant_bits=image_utils.resolve_output_depth(input_depths),
    )


def channel_aware_stitcher(
    path: pathlib.Path,
    df: pd.DataFrame,
    pixel_size_um: float | None = None,
    output_file_loc: pathlib.Path | None = None,
    stitching_mode: str = 'quality',
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
    mode = str(stitching_mode or 'quality')
    chain: list[tuple[str, Callable[[], dict]]] = []
    if mode == 'fast_preview':
        chain.extend(
            [
                (
                    'fft_phase_stitcher',
                    lambda: fft_phase_stitcher(**shared, pixel_size_um=pixel_size_um),
                ),
                (
                    'simple_position_stitcher',
                    lambda: simple_position_stitcher(**shared),
                ),
            ]
        )
    elif color == 'BF':
        chain.append(
            (
                'bf_feature_stitcher',
                lambda: bf_feature_stitcher(**shared),
            )
        )
    if mode != 'fast_preview':
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
