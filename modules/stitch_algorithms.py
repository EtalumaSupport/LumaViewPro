# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Advanced stitching algorithms for overlapping tile images.

Contains feature-based stitching (OpenCV SIFT/SURF keypoint matching),
color transfer for illumination normalization, and border cleanup.

Originally developed by Ori Amir and Kevin Peter Hickerson,
The Earthineering Company (2022). Cleaned up and integrated 2026-03.

The simple grid-placement stitcher (no overlap) lives in stitcher.py.
These algorithms handle the harder case: overlapping tiles with potential
lens distortion and illumination variation.
"""

from collections import deque
import time

import cv2
import numpy as np
from lvp_logger import logger


# ---------------------------------------------------------------------------
# Color transfer (Reinhard et al., 2001)
# ---------------------------------------------------------------------------


def _image_stats(image):
    """Compute mean and std for each channel of an L*a*b* image."""
    (l_chan, a, b) = cv2.split(image)
    return (l_chan.mean(), l_chan.std(), a.mean(), a.std(), b.mean(), b.std())


def color_transfer(source, target):
    """Transfer color distribution from source to target using L*a*b* stats.

    Based on "Color Transfer between Images" by Reinhard et al., 2001.
    Useful for normalizing illumination differences between tiles captured
    at different positions (LED illumination variation across field).

    Parameters
    ----------
    source : numpy.ndarray
        Reference image (BGR, uint8) whose color distribution to match.
    target : numpy.ndarray
        Image (BGR, uint8) to adjust.

    Returns
    -------
    numpy.ndarray
        Color-adjusted target image (BGR, uint8).
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = _image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = _image_stats(target)

    (l_chan, a, b) = cv2.split(target)
    l_chan -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    l_chan = (lStdTar / lStdSrc) * l_chan if lStdSrc > 0 else l_chan
    a = (aStdTar / aStdSrc) * a if aStdSrc > 0 else a
    b = (bStdTar / bStdSrc) * b if bStdSrc > 0 else b

    l_chan += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    l_chan = np.clip(l_chan, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l_chan, a, b])
    transfer = cv2.cvtColor(transfer.astype('uint8'), cv2.COLOR_LAB2BGR)
    return transfer


# ---------------------------------------------------------------------------
# Feature-based stitching (OpenCV Stitcher with SIFT/SURF)
# ---------------------------------------------------------------------------

MAX_TRIES = 20
N_RESULTS = 5


def feature_stitch(images, n_results=N_RESULTS):
    """Stitch overlapping images using OpenCV's feature-based stitcher.

    Uses SIFT/SURF keypoint detection to find matching features between
    overlapping tiles and computes homographies to align them. Handles
    lens distortion and slight position errors automatically.

    Runs the stitcher multiple times and picks the result with highest
    total luminance (best coverage / least black border).

    Parameters
    ----------
    images : list of numpy.ndarray
        List of BGR uint8 images to stitch. Must have overlapping regions.
    n_results : int, optional
        Number of successful stitch attempts to collect before picking
        the best one. Higher = better quality but slower. Default 5.

    Returns
    -------
    numpy.ndarray or None
        Stitched composite image (BGR, uint8), or None if stitching failed
        (insufficient keypoints or no overlap detected).
    """
    if not images or len(images) < 2:
        logger.warning('[Stitch] Need at least 2 images for feature stitching')
        return None

    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_SCANS)
    results = []
    total_attempts = 0
    t0 = time.perf_counter()

    for _ in range(n_results):
        tries = 0
        while tries < MAX_TRIES:
            tries += 1
            total_attempts += 1
            error, stitched_img = stitcher.stitch(images)
            if error == cv2.Stitcher_OK:
                results.append(stitched_img)
                break
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not results:
        logger.warning(
            '[Stitch] Feature stitching failed -- insufficient '
            'matching keypoints or no overlap detected'
        )
        logger.info(
            '[StitchPerf] feature_stitch %.1fms attempts=%d successes=0 tiles=%d',
            elapsed_ms,
            total_attempts,
            len(images),
        )
        return None

    # Pick the result with highest total luminance (best coverage)
    im_total_luminance = np.array([im.sum() for im in results])
    best = results[np.argmax(im_total_luminance)]
    logger.info(
        f'[Stitch] Feature stitch succeeded -- {len(results)}/{n_results} attempts produced results'
    )
    logger.info(
        '[StitchPerf] feature_stitch %.1fms attempts=%d successes=%d '
        'tiles=%d best_shape=%s dtype=%s',
        elapsed_ms,
        total_attempts,
        len(results),
        len(images),
        best.shape,
        best.dtype,
    )
    return best


# ---------------------------------------------------------------------------
# Post-processing: border cleanup
# ---------------------------------------------------------------------------


def _grab_contours(cnts):
    """Extract contours from cv2.findContours result (OpenCV 4.x returns 2-tuple)."""
    return cnts[0] if len(cnts) == 2 else cnts[1]


def crop_to_content(image):
    """Crop a stitched image to remove irregular black borders.

    Feature-based stitching produces non-rectangular output with black
    borders where the homography warped beyond the source images. This
    function finds the largest rectangular region containing only content
    (no black pixels) and crops to it.

    Parameters
    ----------
    image : numpy.ndarray
        Stitched image (BGR, uint8) with potential black borders.

    Returns
    -------
    numpy.ndarray
        Cropped image with black borders removed.
    """
    padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    contours = _grab_contours(
        cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    area = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype='uint8')
    x, y, w, h = cv2.boundingRect(area)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    min_rect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)
        sub = cv2.subtract(min_rect, thresh)

    contours = _grab_contours(
        cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    area = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(area)
    return padded[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# Position-aware stitching with overlap registration
# ---------------------------------------------------------------------------


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32)


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

    left_view = left[y0:y1, x0:x1]
    right_view = right[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return left_view, right_view


def _ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-6:
        return -1.0
    return float(np.sum(a * b) / denom)


def estimate_overlap_offset(
    reference: np.ndarray,
    moving: np.ndarray,
    nominal_dx: int,
    nominal_dy: int,
    max_correction_px: int = 12,
    min_overlap_px: int = 16,
) -> tuple[int, int, float]:
    """Estimate moving-tile correction from overlap content.

    Returns (correction_x, correction_y, score), where correction values are
    added to the moving tile's nominal position relative to the reference
    tile. The search is intentionally local: nominal stage coordinates already
    provide the coarse placement, and overlap registration only handles small
    acquisition/position errors.
    """
    ref_gray = _gray_float(reference)
    mov_gray = _gray_float(moving)

    best = (0, 0, -1.0)
    for corr_y in range(-max_correction_px, max_correction_px + 1):
        for corr_x in range(-max_correction_px, max_correction_px + 1):
            views = _overlap_views(
                ref_gray,
                mov_gray,
                dx=nominal_dx + corr_x,
                dy=nominal_dy + corr_y,
            )
            if views is None:
                continue
            ref_view, mov_view = views
            if ref_view.shape[0] < min_overlap_px or ref_view.shape[1] < min_overlap_px:
                continue
            score = _ncc_score(ref_view, mov_view)
            if score > best[2]:
                best = (corr_x, corr_y, score)

    return best


def estimate_phase_offset(
    reference: np.ndarray,
    moving: np.ndarray,
    nominal_dx: int,
    nominal_dy: int,
    max_correction_px: int = 12,
    min_overlap_px: int = 16,
) -> tuple[int, int, float]:
    """Bounded FFT phase-correlation correction for a known overlap edge.

    This is deliberately not OpenCV's unconstrained panorama stitcher.  The
    recorded stage position defines the only search region; a low-signal,
    no-overlap, or implausibly-large correction returns the nominal placement.
    """
    ref_gray = _gray_float(reference)
    mov_gray = _gray_float(moving)
    views = _overlap_views(ref_gray, mov_gray, nominal_dx, nominal_dy)
    if views is None:
        return 0, 0, -1.0
    ref_view, mov_view = views
    if ref_view.shape[0] < min_overlap_px or ref_view.shape[1] < min_overlap_px:
        return 0, 0, -1.0
    if float(ref_view.std()) <= 1e-6 or float(mov_view.std()) <= 1e-6:
        return 0, 0, -1.0
    window = cv2.createHanningWindow((ref_view.shape[1], ref_view.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        ref_view.astype(np.float32), mov_view.astype(np.float32), window
    )
    corr_x = int(round(-shift[0]))
    corr_y = int(round(-shift[1]))
    if abs(corr_x) > max_correction_px or abs(corr_y) > max_correction_px:
        return 0, 0, float(response)
    return corr_x, corr_y, float(response)


def _grid_keys(tiles: list[dict]) -> tuple[list[int], list[int], dict[tuple[int, int], int]]:
    x_values = sorted({int(tile['x_px']) for tile in tiles})
    y_values = sorted({int(tile['y_px']) for tile in tiles})
    by_position = {(int(tile['x_px']), int(tile['y_px'])): idx for idx, tile in enumerate(tiles)}
    return x_values, y_values, by_position


def align_tile_positions(
    tiles: list[dict],
    max_correction_px: int = 12,
    min_overlap_px: int = 16,
    estimator=estimate_overlap_offset,
) -> list[dict]:
    """Return tiles with overlap-registered x/y placement corrections.

    Propagates registration offsets from a top-left anchor across the tile
    lattice via a 4-neighbor (left/right/up/down) breadth-first flood, so
    every tile reachable from the anchor through present neighbors is
    registered -- not only those on a pure right/down path. Sparse or ragged
    groups (a partially off-stage region drops interior tiles but keeps the
    rest in one tile group) therefore register across the gap instead of
    stranding the tiles past a hole at zero offset. Tiles with no overlap
    path to the anchor (a disconnected component, or two tiles whose nominal
    positions round to the same lattice key) keep nominal placement and are
    logged.
    """
    if not tiles:
        return []

    x_values, y_values, by_position = _grid_keys(tiles)
    corrected = [dict(tile) for tile in tiles]
    offsets: dict[int, tuple[int, int]] = {}

    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}

    anchor = by_position[(x_values[0], y_values[0])]
    offsets[anchor] = (0, 0)

    # 4-neighbor BFS over present lattice positions. Each dequeued tile
    # registers any not-yet-placed grid neighbor that exists, then enqueues
    # it. Exploring all four directions lets the flood route around a hole
    # (reach a tile via up/left when the right/down path is blocked) --
    # estimate_overlap_offset / _overlap_views handle the negative nominal
    # displacement of left/up edges symmetrically.
    queue: deque[int] = deque([anchor])
    registration_times_ms = []
    registration_scores = []
    while queue:
        idx = queue.popleft()
        base_dx, base_dy = offsets[idx]
        x = int(tiles[idx]['x_px'])
        y = int(tiles[idx]['y_px'])
        xi = x_index[x]
        yi = y_index[y]

        neighbors = []
        if xi - 1 >= 0:
            neighbors.append((x_values[xi - 1], y))
        if xi + 1 < len(x_values):
            neighbors.append((x_values[xi + 1], y))
        if yi - 1 >= 0:
            neighbors.append((x, y_values[yi - 1]))
        if yi + 1 < len(y_values):
            neighbors.append((x, y_values[yi + 1]))

        for nx, ny in neighbors:
            nidx = by_position.get((nx, ny))
            if nidx is None or nidx in offsets:
                continue
            edge_t0 = time.perf_counter()
            corr_x, corr_y, score = estimator(
                reference=tiles[idx]['tile'],
                moving=tiles[nidx]['tile'],
                nominal_dx=nx - x,
                nominal_dy=ny - y,
                max_correction_px=max_correction_px,
                min_overlap_px=min_overlap_px,
            )
            edge_ms = (time.perf_counter() - edge_t0) * 1000.0
            registration_times_ms.append(edge_ms)
            registration_scores.append(score)
            if edge_ms >= 1000.0:
                logger.warning(
                    '[StitchPerf] slow registration edge %.1fms from=(%s,%s) '
                    'to=(%s,%s) score=%.4f max_correction_px=%d',
                    edge_ms,
                    x,
                    y,
                    nx,
                    ny,
                    score,
                    max_correction_px,
                )
            offsets[nidx] = (base_dx + corr_x, base_dy + corr_y)
            corrected[nidx]['registration_score'] = score
            queue.append(nidx)

    unregistered = [idx for idx in range(len(tiles)) if idx not in offsets]
    if unregistered:
        logger.warning(
            f'align_tile_positions: {len(unregistered)} of {len(tiles)} tiles '
            f'had no overlap path to the anchor (disconnected component or '
            f'colliding nominal positions); placed at nominal stage position '
            f'without registration correction'
        )

    for idx, tile in enumerate(corrected):
        corr_x, corr_y = offsets.get(idx, (0, 0))
        tile['registration_offset_x_px'] = corr_x
        tile['registration_offset_y_px'] = corr_y
        tile['registered_x_px'] = int(tile['x_px']) + corr_x
        tile['registered_y_px'] = int(tile['y_px']) + corr_y

    if registration_times_ms:
        logger.info(
            '[StitchPerf] align_tile_positions edges=%d total=%.1fms '
            'avg=%.1fms max=%.1fms score_min=%.4f score_avg=%.4f '
            'max_correction_px=%d',
            len(registration_times_ms),
            sum(registration_times_ms),
            sum(registration_times_ms) / len(registration_times_ms),
            max(registration_times_ms),
            min(registration_scores),
            sum(registration_scores) / len(registration_scores),
            max_correction_px,
        )

    return corrected


def stitch_registered_tiles(
    tiles: list[dict],
    max_correction_px: int = 12,
    min_overlap_px: int = 16,
    output_shape: tuple[int, int] | None = None,
    estimator=estimate_overlap_offset,
    blend_mode: str = 'average',
) -> tuple[np.ndarray, list[dict]]:
    """Register tiles, then compose them with an explicit pixel policy."""
    if not tiles:
        raise ValueError('Need at least one tile to stitch')

    register_t0 = time.perf_counter()
    registered = align_tile_positions(
        tiles=tiles,
        max_correction_px=max_correction_px,
        min_overlap_px=min_overlap_px,
        estimator=estimator,
    )
    register_ms = (time.perf_counter() - register_t0) * 1000.0

    sample = registered[0]['tile']
    tile_h, tile_w = sample.shape[:2]
    # All tiles in a stitch group must share channel layout; a mix of mono
    # (ndim 2) and color (ndim 3) tiles would broadcast-fail in the
    # accumulator blend below. Surface it as a clear error up front so the
    # caller falls back to the simple grid stitch instead of hitting a
    # cryptic broadcast ValueError mid-blend.
    if any(tile['tile'].ndim != sample.ndim for tile in registered):
        raise ValueError('Cannot stitch a mix of mono and color tiles in one group')
    if output_shape is None:
        min_x = min(int(tile['registered_x_px']) for tile in registered)
        min_y = min(int(tile['registered_y_px']) for tile in registered)
        max_x = max(int(tile['registered_x_px']) + tile_w for tile in registered)
        max_y = max(int(tile['registered_y_px']) + tile_h for tile in registered)
    else:
        min_x = 0
        min_y = 0
        max_y, max_x = output_shape

    if sample.ndim == 2:
        acc_shape = (max_y - min_y, max_x - min_x)
        weight_shape = acc_shape
    else:
        acc_shape = (max_y - min_y, max_x - min_x, sample.shape[2])
        weight_shape = (max_y - min_y, max_x - min_x, 1)

    if blend_mode == 'source_preserving':
        output = np.zeros(acc_shape, dtype=sample.dtype)
        covered = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
        for tile in registered:
            image = tile['tile']
            x0 = int(tile['registered_x_px']) - min_x
            y0 = int(tile['registered_y_px']) - min_y
            dst_x0, dst_y0 = max(0, x0), max(0, y0)
            dst_x1 = min(acc_shape[1], x0 + image.shape[1])
            dst_y1 = min(acc_shape[0], y0 + image.shape[0])
            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                continue
            src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
            src_x1 = src_x0 + (dst_x1 - dst_x0)
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            take = ~covered[dst_y0:dst_y1, dst_x0:dst_x1]
            destination = output[dst_y0:dst_y1, dst_x0:dst_x1]
            source = image[src_y0:src_y1, src_x0:src_x1]
            if sample.ndim == 2:
                destination[take] = source[take]
            else:
                destination[take, :] = source[take, :]
            covered[dst_y0:dst_y1, dst_x0:dst_x1][take] = True
        logger.info(
            '[StitchPerf] stitch_registered_tiles register=%.1fms source-preserving '
            'tiles=%d output_shape=%s output_dtype=%s',
            register_ms,
            len(tiles),
            output.shape,
            output.dtype,
        )
        return output, registered
    if blend_mode != 'average':
        raise ValueError(f'Unknown stitch blend mode: {blend_mode}')

    # float32 (not float64): each whole-mosaic canvas can be multiple GB, and
    # the blend is an integer-pixel average. Products/sums of uint8/uint16
    # pixels over the handful of tiles overlapping any pixel stay well inside
    # float32's exact-integer range (2**24), so the averaged result is
    # byte-identical to float64 at half the memory.
    alloc_t0 = time.perf_counter()
    accumulator = np.zeros(acc_shape, dtype=np.float32)
    weights = np.zeros(weight_shape, dtype=np.float32)
    alloc_ms = (time.perf_counter() - alloc_t0) * 1000.0
    logger.info(
        '[StitchPerf] blend allocation %.1fms acc_shape=%s weight_shape=%s '
        'working_bytes=%d sample_dtype=%s',
        alloc_ms,
        acc_shape,
        weight_shape,
        int(accumulator.nbytes + weights.nbytes),
        sample.dtype,
    )

    blend_t0 = time.perf_counter()
    for tile in registered:
        image = tile['tile']
        x0 = int(tile['registered_x_px']) - min_x
        y0 = int(tile['registered_y_px']) - min_y
        y1 = y0 + image.shape[0]
        x1 = x0 + image.shape[1]

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(acc_shape[1], x1)
        dst_y1 = min(acc_shape[0], y1)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue

        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        accumulator[dst_y0:dst_y1, dst_x0:dst_x1] += image[src_y0:src_y1, src_x0:src_x1].astype(
            np.float32
        )
        weights[dst_y0:dst_y1, dst_x0:dst_x1] += 1.0

    # Divide in place -- accumulator becomes the averaged mosaic, dropping a
    # third whole-mosaic canvas. where=weights>0 leaves never-covered pixels
    # at their initialized 0.
    np.divide(accumulator, weights, out=accumulator, where=weights > 0)

    if np.issubdtype(sample.dtype, np.integer):
        info = np.iinfo(sample.dtype)
        np.clip(accumulator, info.min, info.max, out=accumulator)

    output = accumulator.astype(sample.dtype)
    blend_ms = (time.perf_counter() - blend_t0) * 1000.0
    logger.info(
        '[StitchPerf] stitch_registered_tiles register=%.1fms alloc=%.1fms '
        'blend=%.1fms tiles=%d output_shape=%s output_dtype=%s',
        register_ms,
        alloc_ms,
        blend_ms,
        len(tiles),
        output.shape,
        output.dtype,
    )
    return output, registered
