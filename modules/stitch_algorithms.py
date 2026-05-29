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

import cv2
import numpy as np
from lvp_logger import logger


# ---------------------------------------------------------------------------
# Color transfer (Reinhard et al., 2001)
# ---------------------------------------------------------------------------


def _image_stats(image):
    """Compute mean and std for each channel of an L*a*b* image."""
    (l, a, b) = cv2.split(image)
    return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())


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

    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    l = (lStdTar / lStdSrc) * l if lStdSrc > 0 else l
    a = (aStdTar / aStdSrc) * a if aStdSrc > 0 else a
    b = (bStdTar / bStdSrc) * b if bStdSrc > 0 else b

    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
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

    for _ in range(n_results):
        tries = 0
        while tries < MAX_TRIES:
            tries += 1
            error, stitched_img = stitcher.stitch(images)
            if error == cv2.Stitcher_OK:
                results.append(stitched_img)
                break

    if not results:
        logger.warning(
            '[Stitch] Feature stitching failed -- insufficient '
            'matching keypoints or no overlap detected'
        )
        return None

    # Pick the result with highest total luminance (best coverage)
    im_total_luminance = np.array([im.sum() for im in results])
    best = results[np.argmax(im_total_luminance)]
    logger.info(
        f'[Stitch] Feature stitch succeeded -- {len(results)}/{n_results} attempts produced results'
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


def _grid_keys(tiles: list[dict]) -> tuple[list[int], list[int], dict[tuple[int, int], int]]:
    x_values = sorted({int(tile['x_px']) for tile in tiles})
    y_values = sorted({int(tile['y_px']) for tile in tiles})
    by_position = {
        (int(tile['x_px']), int(tile['y_px'])): idx
        for idx, tile in enumerate(tiles)
    }
    return x_values, y_values, by_position


def align_tile_positions(
    tiles: list[dict],
    max_correction_px: int = 12,
    min_overlap_px: int = 16,
) -> list[dict]:
    """Return tiles with overlap-registered x/y placement corrections."""
    if not tiles:
        return []

    x_values, y_values, by_position = _grid_keys(tiles)
    corrected = [dict(tile) for tile in tiles]
    offsets: dict[int, tuple[int, int]] = {}

    anchor = by_position[(x_values[0], y_values[0])]
    offsets[anchor] = (0, 0)

    changed = True
    while changed:
        changed = False
        for y in y_values:
            for x_idx, x in enumerate(x_values):
                idx = by_position.get((x, y))
                if idx is None or idx not in offsets:
                    continue
                base_dx, base_dy = offsets[idx]

                neighbors = []
                if x_idx + 1 < len(x_values):
                    neighbors.append((x_values[x_idx + 1], y))
                y_idx = y_values.index(y)
                if y_idx + 1 < len(y_values):
                    neighbors.append((x, y_values[y_idx + 1]))

                for nx, ny in neighbors:
                    nidx = by_position.get((nx, ny))
                    if nidx is None or nidx in offsets:
                        continue
                    corr_x, corr_y, score = estimate_overlap_offset(
                        reference=tiles[idx]['tile'],
                        moving=tiles[nidx]['tile'],
                        nominal_dx=nx - x,
                        nominal_dy=ny - y,
                        max_correction_px=max_correction_px,
                        min_overlap_px=min_overlap_px,
                    )
                    offsets[nidx] = (base_dx + corr_x, base_dy + corr_y)
                    corrected[nidx]['registration_score'] = score
                    changed = True

    for idx, tile in enumerate(corrected):
        corr_x, corr_y = offsets.get(idx, (0, 0))
        tile['registration_offset_x_px'] = corr_x
        tile['registration_offset_y_px'] = corr_y
        tile['registered_x_px'] = int(tile['x_px']) + corr_x
        tile['registered_y_px'] = int(tile['y_px']) + corr_y

    return corrected


def stitch_registered_tiles(
    tiles: list[dict],
    max_correction_px: int = 12,
    min_overlap_px: int = 16,
    output_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Register overlapping tiles, then average-blend them into one image."""
    if not tiles:
        raise ValueError('Need at least one tile to stitch')

    registered = align_tile_positions(
        tiles=tiles,
        max_correction_px=max_correction_px,
        min_overlap_px=min_overlap_px,
    )

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

    accumulator = np.zeros(acc_shape, dtype=np.float64)
    weights = np.zeros(weight_shape, dtype=np.float64)

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

        accumulator[dst_y0:dst_y1, dst_x0:dst_x1] += image[
            src_y0:src_y1, src_x0:src_x1
        ].astype(np.float64)
        weights[dst_y0:dst_y1, dst_x0:dst_x1] += 1.0

    output = np.zeros(acc_shape, dtype=np.float64)
    np.divide(accumulator, weights, out=output, where=weights > 0)

    if np.issubdtype(sample.dtype, np.integer):
        info = np.iinfo(sample.dtype)
        output = np.clip(output, info.min, info.max)

    return output.astype(sample.dtype), registered
