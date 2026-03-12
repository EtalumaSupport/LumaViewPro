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
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

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
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
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
        logger.warning("[Stitch] Need at least 2 images for feature stitching")
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
        logger.warning("[Stitch] Feature stitching failed — insufficient "
                       "matching keypoints or no overlap detected")
        return None

    # Pick the result with highest total luminance (best coverage)
    im_total_luminance = np.array([im.sum() for im in results])
    best = results[np.argmax(im_total_luminance)]
    logger.info(f"[Stitch] Feature stitch succeeded — {len(results)}/{n_results} "
                f"attempts produced results")
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

    contours = _grab_contours(cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    area = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(area)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    min_rect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)
        sub = cv2.subtract(min_rect, thresh)

    contours = _grab_contours(cv2.findContours(
        min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    area = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(area)
    return padded[y:y + h, x:x + w]
