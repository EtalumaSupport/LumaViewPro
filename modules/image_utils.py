# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import ast
import contextlib
import dataclasses
import datetime
import enum
import functools
import io
import json
import os
import pathlib
import re
import struct
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tifffile as tf

from modules.api_surface import api_fields
from modules.exceptions import FrameDepthError, InputFramesUnusableError
from modules.layer_record import GREY, Color, stored_key
import modules.common_utils as common_utils
import modules.image_mode as image_mode
import modules.objectives_loader as objectives_loader
import modules.image_utils as image_utils

from fractions import Fraction

from lvp_logger import logger, version

if TYPE_CHECKING:
    from modules.objectives_loader import Objective
    from modules.scope_capabilities import ScopeCapabilities


TIFF_SUFFIXES = frozenset({'.tif', '.tiff'})

# Every image load_pixels reads: a TIFF through tifffile, the rest through
# cv2.imread. A superset of TIFF_SUFFIXES, so the two cannot disagree about
# what a TIFF is.
IMAGE_SUFFIXES = TIFF_SUFFIXES | frozenset({'.png', '.jpg', '.jpeg', '.bmp'})

# The OME UnitsLength token for micrometres is the MICRO SIGN form; the
# ASCII 'um' shorthand is schema-invalid and a strict OME parser refuses
# the whole file (Bio-Formats' lenient parsing long hid this). Escaped so
# the source file stays ASCII. The ImageJ metadata 'unit' field is a
# different consumer's vocabulary and deliberately keeps 'um'.
OME_UNIT_MICROMETER = '\u00b5m'


def is_tiff(path: pathlib.Path | str) -> bool:
    """True when this path names a TIFF the project can read.

    The single answer to "is this a TIFF", shared by every caller that scans a
    folder, gates a post-processing op, or decides how to write a file back.
    Matching on the SUFFIX rather than a filename glob is what makes the
    multi-part `.ome.tiff` form safe: its suffix is plain `.tiff`, so one file
    yields one answer. Glob patterns cannot express that -- `*.tiff` and
    `*.ome.tiff` both match the same file, so a caller that unions them counts
    it twice.
    """
    return pathlib.Path(path).suffix.lower() in TIFF_SUFFIXES


def is_image(path: pathlib.Path | str) -> bool:
    """True when this path names an image the project can read (``IMAGE_SUFFIXES``).

    The single answer to "is this an image", for every folder scan and every
    file picker. Case-blind, as ``is_tiff`` is: a folder count that compared
    suffixes as written skipped ``IMG.TIF`` without saying so.
    """
    return pathlib.Path(path).suffix.lower() in IMAGE_SUFFIXES


def find_tiff_files(
    directory: pathlib.Path | str,
    recursive: bool = False,
) -> list[pathlib.Path]:
    """Return every TIFF under `directory`, each exactly once, sorted.

    De-duplication is structural rather than a cleanup pass: files are
    enumerated once and filtered by suffix, so there is no pattern-union step in
    which one file could appear twice. A duplicated tile is not cosmetic
    downstream -- two rows sharing a nominal stage position collide on the
    stitcher's lattice key, and the loser is placed without registration
    correction, which reads as a seam in the montage.
    """
    directory = pathlib.Path(directory)
    walk = directory.rglob('*') if recursive else directory.glob('*')
    return sorted(p for p in walk if p.is_file() and is_tiff(p))


def is_color_shape(shape) -> bool:
    """True when an array of this shape is a 3-channel color image (H, W, 3).

    The single channel-count rule, shared by callers that hold a full image and
    callers that hold only a shape tuple (e.g. a stitcher sizing its canvas from
    a header read without decoding pixels), so the two cannot disagree.
    """
    return len(shape) == 3 and shape[2] == 3


def is_color_image(image) -> bool:
    return is_color_shape(image.shape)


def fit_frame_to_shape(image: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray | None:
    """Pad or crop a frame to target_shape, preserving the overlapping pixels.

    Some camera modes (and the simulator) can report one configured frame
    size while delivering a nearby sensor-valid one. A spatial size
    difference is black-padded (when the frame is smaller) or cropped (when
    larger). Returns None -- the frame should be skipped -- when the frame is
    fundamentally incompatible: a different number of dimensions, or a
    different channel count for a color frame.
    """
    if len(image.shape) != len(target_shape):
        return None
    if image.ndim == 3 and image.shape[2] != target_shape[2]:
        return None

    fitted = np.zeros(target_shape, dtype=image.dtype)
    height = min(image.shape[0], target_shape[0])
    width = min(image.shape[1], target_shape[1])
    if image.ndim == 2:
        fitted[:height, :width] = image[:height, :width]
    else:
        fitted[:height, :width, :] = image[:height, :width, :]
    return fitted


@api_fields(
    'minimum', 'maximum', 'mean', 'standard_deviation', 'percentile_1', 'median', 'percentile_99'
)
@dataclasses.dataclass(frozen=True)
class ImageStatistics:
    """What an image's pixel values are, in its own counts.

    Over every sample of the array, each colour channel's included, at the
    depth it holds: a 12-bit frame's counts run to 4095, a sum's to what it
    reached. Saturation and darkness are the frame record's, measured
    against the frame's full scale by the capture.

    Attributes:
        minimum: The lowest value.
        maximum: The highest value.
        mean: The mean value.
        standard_deviation: The population standard deviation.
        percentile_1: The lowest value at or above 1% of the samples.
        median: The lowest value at or above half of the samples.
        percentile_99: The lowest value at or above 99% of the samples.
    """

    minimum: int
    maximum: int
    mean: float
    standard_deviation: float
    percentile_1: int
    median: int
    percentile_99: int


def image_statistics(image: np.ndarray) -> ImageStatistics:
    """The exact statistics of an integer image, from one histogram of its values.

    One count per value: about 4 ms for a 1900x1900 frame, where the same
    numbers through ``np.percentile`` take 32 ms, so every still can carry
    them.
    """
    counts = np.bincount(image.ravel())
    total = int(counts.sum())
    values = np.arange(counts.size, dtype=np.float64)
    mean = float(counts @ values) / total
    variance = float(counts @ (values - mean) ** 2) / total
    cumulative = np.cumsum(counts)
    present = np.flatnonzero(counts)

    def at_or_above(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction * total))

    return ImageStatistics(
        minimum=int(present[0]),
        maximum=int(present[-1]),
        mean=mean,
        standard_deviation=variance**0.5,
        percentile_1=at_or_above(0.01),
        median=at_or_above(0.5),
        percentile_99=at_or_above(0.99),
    )


def decimate_for_preview(
    image: np.ndarray, target_wh: tuple[int, int] | None, *, categorical: bool = False
) -> np.ndarray:
    """Host-side downscale of a live-preview frame to roughly the on-screen
    widget size, so the main-thread Kivy ``blit_buffer`` uploads fewer bytes.

    Fallback path only. The default preview path uploads the full-resolution
    frame and lets the GPU minify it to the widget; that is cheap (the upload is
    a small fraction of a core) and constant-cost at any zoom. This host resize
    exists for machines where the full-res upload is not smooth: enable it with
    the ``preview_host_downscale`` setting. It is kept because it is fast at any
    ratio, unlike ``INTER_AREA`` which collapses to a slow per-output-pixel
    resample at a fractional contain-fit ratio (the common case).

    Handles both the 2-D ``uint8`` luminance preview and the 3-D ``(H, W, 3)``
    RGB bullseye preview -- both blit paths share the identical
    downscale-to-widget-size need, so they share one helper (``cv2.resize``
    preserves the channel axis). Only the channel count and the resampling
    differ; the contain-fit math is the same.

    ``categorical`` picks the resampling to match the data's meaning:

    - ``False`` (default, continuous-tone): bilinear (``INTER_LINEAR``), a
      fixed 4-tap that stays fast at any downscale ratio.
    - ``True`` (false-color / label image, e.g. the bullseye contour map):
      nearest-neighbor. The bullseye LUT is a topographic map -- mostly black
      with thin pure-color iso-intensity bands -- so area-averaging would blend
      those one-pixel contour lines toward black and erase the focus aid.
      Nearest-neighbor keeps each sampled label's exact color (what the GPU's
      texture minification did before the host-side downscale existed).

    Live-view frame rate is bounded by the per-frame texture upload on the main
    (Kivy) thread, which serializes against the capture and convert threads on
    the GIL -- not by the camera or the packed-format converter. A large-sensor
    frame (e.g. 1900x1900, ~3.6 MB) blitted at full resolution every frame
    starves the main thread; a frame already near the displayed size does not.
    The displayed pixels are unchanged because the GPU was downscaling the
    oversized texture to the widget anyway.

    Preview only. Capture, save, autofocus, histogram, and raw frame-listener
    paths take the full-resolution image through other code and are untouched.

    Args:
        image: 2-D ``uint8`` grayscale or 3-D ``(H, W, C)`` ``uint8`` RGB
            preview frame (already downconverted).
        target_wh: ``(width, height)`` of the display widget in pixels, or None
            when the widget has not been laid out yet -- in which case the
            image is returned unchanged (full-resolution blit, the prior
            behavior), so a missing target never degrades correctness.

    Returns:
        The image unchanged when no downscale applies (target unknown, a frame
        that is neither 2-D luminance nor 3-D with 3/4 channels, or a frame
        already at/below the target); otherwise a copy scaled to the frame's
        on-screen size (channels preserved). The widget shows the frame
        contain-fit (aspect preserved, letterboxed), so the displayed size is the
        frame scaled by ``min(tw/w, th/h)``; downscaling to exactly that uploads
        no more pixels than the screen shows and stays as sharp as the widget can
        display.

    A single contain-fit ratio (not an integer decimation step) is used so the
    downscale engages for frames only slightly larger than the widget. An
    integer step rounds down to 1 -- i.e. no downscale -- for any frame under
    2x the widget on both axes, leaving e.g. a ~2100x2100 sensor frame blitted
    at full resolution on a normal display and capping live view at ~5-6 fps.
    """
    if target_wh is None or image is None:
        return image
    # Accept 2-D luminance, or 3-D with a real color channel count (3=RGB,
    # 4=RGBA). Reject anything else -- notably (H, W, 1), which cv2.resize would
    # squeeze to a 2-D result, dropping the channel axis the RGB blit assumes.
    is_luma = image.ndim == 2
    is_color = image.ndim == 3 and image.shape[2] in (3, 4)
    if not (is_luma or is_color):
        return image
    tw, th = target_wh
    if tw < 1 or th < 1:
        return image
    h, w = image.shape[:2]
    factor = min(tw / w, th / h)
    if factor >= 1.0:
        return image  # frame already <= widget on the limiting axis; never upscale
    interpolation = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    new_w = max(1, round(w * factor))
    new_h = max(1, round(h * factor))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def scaled_preview_target(base_wh: tuple[int, int] | None, scale: float) -> tuple[int, int] | None:
    """Scale a widget's pixel box up by a zoom (Scatter) factor to get the
    on-screen size of a digitally-zoomed live image, for ``decimate_for_preview``.

    Zoom-in magnifies each texel, so the displayed image needs more resolution,
    not less: at a 1:1 zoom the scale equals sensor/widget, so the target reaches
    sensor size and the decimation factor falls to 1 (full-resolution blit).
    Zoom-out is clamped to 1.0 -- keep widget-size detail rather than
    over-shrinking a view that is already small on screen.

    Returns None when ``base_wh`` is None (widget not laid out yet), so the
    caller leaves the frame at full resolution.

    Rounds to the nearest displayed pixel (not truncates): at 1:1 the scale is
    ``sensor/widget``, and ``widget * sensor/widget`` lands a sub-LSB below the
    sensor size for ~6% of widget/sensor pairs. Truncating would make the target
    one pixel short of the sensor, so the exact-ratio downscale would do a
    pointless 1-pixel shrink instead of leaving the 1:1 view at true full
    resolution. Rounding keeps the target on the sensor size, so the factor is
    exactly 1.0 and the frame is blitted untouched.
    """
    if base_wh is None:
        return None
    try:
        s = float(scale)
    except (TypeError, ValueError):
        s = 1.0
    if not s or s < 1.0:
        s = 1.0
    return (round(base_wh[0] * s), round(base_wh[1] * s))


def click_offset_to_um(
    norm_offset: float, norm_extent: float, frame_extent: float, pixel_size_um: float
) -> float:
    """Physical (micron) distance from the frame center for a click landing
    ``norm_offset`` widget-pixels from the displayed image's edge, where the
    displayed image spans ``norm_extent`` widget-pixels and the full-resolution
    sensor frame is ``frame_extent`` pixels. Used by click-to-center to drive
    the stage so the clicked feature moves to the optical axis.

    ``frame_extent`` MUST be the full-resolution sensor frame, never the
    preview-display texture: the live preview is downscaled to ~widget size
    before the blit, but ``pixel_size_um`` is the size of one SENSOR pixel, so
    feeding the downscaled texture size under-reports the distance by the
    downscale factor and the stage stops short of the clicked point. Returns
    0.0 when the displayed extent is degenerate (no division possible).
    """
    if norm_extent <= 0:
        return 0.0
    frame_pos = norm_offset * frame_extent / norm_extent
    return (frame_pos - frame_extent / 2.0) * pixel_size_um


# The caller resolves a colour (``layer_record.Color``); nothing here maps a
# channel to one. A writer keeps a ``GREY`` frame mono rather than storing
# three equal planes.


def mono_to_rgb_falsecolor(
    mono: np.ndarray, color: Color, output: np.ndarray | None = None
) -> np.ndarray:
    """Map a 2D mono array to a 3-channel RGB array in ``color``.

    The single Phase-1 boundary helper for mono -> RGB widening. Use at
    encode boundaries (live preview Kivy texture, MP4 / AVI encode) and
    legacy-display paths -- NOT in the save pipeline (mono-native save
    keeps 2D + layer metadata; widening to RGB at save bakes the false
    color into the file).

    Args:
        mono: 2D ndarray of uint8 / uint16 dtype.
        color: Each plane is ``mono`` times its weight, rounded; ``GREY``
            tiles the mono into all three planes.
        output: An ``(H, W, 3)`` array of ``mono``'s dtype to write into,
            reused across frames; another shape or dtype is not used.

    Returns:
        ``(H, W, 3)`` ndarray with the same dtype as ``mono``. Source
        array is not modified.

    Raises:
        ValueError: ``mono`` is not 2D.
    """
    if mono.ndim != 2:
        raise ValueError(f'mono_to_rgb_falsecolor expects 2D input, got shape {mono.shape}')

    shape = (mono.shape[0], mono.shape[1], 3)
    if output is not None and output.shape == shape and output.dtype == mono.dtype:
        rgb = output
    else:
        rgb = np.empty(shape, dtype=mono.dtype)
    for plane, weight in enumerate(color):
        # A whole weight is copied rather than multiplied: the same pixels,
        # without a float pass per plane on every live and video frame.
        if weight == 1:
            rgb[:, :, plane] = mono
        elif weight == 0:
            rgb[:, :, plane] = 0
        else:
            rgb[:, :, plane] = np.rint(mono * weight)
    return rgb


# Module-level once-per-process flag for legacy-collapse log noise control.
_legacy_collapse_warned: bool = False


def read_tiff_with_legacy_collapse(path: pathlib.Path) -> np.ndarray:
    """Read a TIFF, collapsing pre-1d 3-channel false-color-replica files
    to mono. Mono inputs and true color outputs pass through.

    The pre-1d save pipeline widened mono fluorescence to 3-channel RGB
    with one populated channel (Blue/Green/Red/Lumi) before write. Post-
    1d the save path is mono. This reader bridges the two on-disk
    formats so post-1d consumers (VideoBuilder, CompositeGeneration) see
    a uniform 2D mono shape regardless of file age.

    Detection rule: 3-channel input with exactly one channel non-zero
    AND the other two channels entirely zero is treated as legacy
    false-color-replica and collapsed to mono. True color outputs
    (composite RGB with multiple channels carrying signal) pass through
    unchanged.

    Args:
        path: TIFF file path.

    Returns:
        2D mono ndarray for mono and collapsed-legacy files; 3D RGB
        ndarray for real color images.
    """
    return _collapse_legacy_false_color(tf.imread(str(path)), path)


def _collapse_legacy_false_color(img: np.ndarray, path: pathlib.Path) -> np.ndarray:
    """Apply the legacy false-color collapse rule to an already-read array.

    Split out from read_tiff_with_legacy_collapse so load_pixels can decode the
    pixels from a single open TIFF handle and still reuse the exact collapse
    logic, instead of opening the file a second time. See
    read_tiff_with_legacy_collapse for the detection rule and rationale.
    """
    global _legacy_collapse_warned
    if img.ndim == 3 and img.shape[2] == 3:
        nonzero_channels = [i for i in range(3) if img[..., i].any()]
        # Considered tightening this to a known-LUT match or a metadata
        # marker so a genuine single-color composite is never collapsed;
        # rejected because every 3-channel input that reaches this reader is
        # a false-color replica (a legacy mono->RGB save, or a
        # false-color-mode save) -- real composites carry signal in
        # multiple channels -- and the one-shot log line flags any collapse.
        # Revisit if a true single-channel RGB output ever feeds VideoBuilder
        # or CompositeGeneration.
        if len(nonzero_channels) == 1:
            if not _legacy_collapse_warned:
                logger.info(f'Legacy false-color TIFF detected at {path}; loaded as mono')
                _legacy_collapse_warned = True
            return img[..., nonzero_channels[0]].copy()
    return img


# Private TIFF tag carrying the payload's significant-bit count on outputs that
# have no OME-XML to hold it (plain + ImageJ TIFFs). In the TIFF private-tag
# range (32768-65535); FIJI / ImageJ / Windows Preview ignore unknown tags.
_TIFF_TAG_SIGNIFICANT_BITS = 65123

# Private TIFF tag carrying the depth whose full scale is white in the stored
# pixels, when that lies below the payload depth: a sum of 12-bit frames is
# tagged 14 bits and shown against one frame's 4095. A tag rather than a
# structured-metadata key because the OME container keeps only schema
# attributes, and this has to survive every container.
_TIFF_TAG_WHITE_BITS = 65124

# The values the OME schema's Plane has attributes for, each with its unit.
# tifffile's OME writer keeps these and drops every other Plane key without a
# word, so write_tiff carries the rest in the image's MapAnnotation in LVP's
# namespace: OME's own key/value slot, which Bio-Formats shows.
_OME_PLANE_VALUES = ('DeltaT', 'ExposureTime', 'PositionX', 'PositionY', 'PositionZ')
_OME_PLANE_ATTRIBUTES = frozenset(
    _OME_PLANE_VALUES + tuple(f'{value}Unit' for value in _OME_PLANE_VALUES)
)
_LVP_OME_ANNOTATION_NAMESPACE = 'etaluma.com/lvp'

# tifffile's OME writer serializes only Image, Pixels, Channel, Plane and the
# MapAnnotation, and drops every other element without a word, so the
# Instrument is spliced into the description it wrote. Registered as the
# default namespace so the spliced description keeps the unprefixed tags
# tifffile wrote; nothing else in LVP serializes XML with ElementTree.
_OME_NAMESPACE = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
ET.register_namespace('', _OME_NAMESPACE)

# The TIFF 6.0 DateTime tag's one form. A record's ``datetime`` is a datetime
# value, formatted here by the writer and parsed back by the reader.
_TIFF_DATETIME = '%Y:%m:%d %H:%M:%S'


class FactKind(enum.Enum):
    """What a fact of a saved file's record describes, which decides whether a
    file derived from it states it."""

    # The one frame: its time, its id, how it was judged. A derived file was
    # not captured, so it states none of them.
    CAPTURE = 'capture'
    # A layer's settings and light: stated only by a derived file on the
    # channel of its inputs.
    CHANNEL = 'channel'
    # The scope, the objective, the plate and the place: stated by a derived
    # file as its inputs state them.
    SHARED = 'shared'
    # The file itself: set by whoever writes it, never read from an input.
    FILE = 'file'


@dataclasses.dataclass(frozen=True)
class FileFact:
    """One fact a saved file's record can state.

    ``plane`` is its key in the file's Plane block, None for a fact that
    lives elsewhere in the file; ``unit`` is written beside it as
    ``<plane>Unit``. ``dropped_by_a_sum``: a derivation that adds frames
    no longer states it.
    """

    kind: FactKind
    plane: str | None = None
    unit: str | None = None
    dropped_by_a_sum: bool = False


# Every fact a saved file's record can state, by its key in the record: the
# one list the writer, the reader and the derived builders read, so a fact
# added here is written, read back and carried, and a fact with no row is
# none of them. Plane facts in the order the writer writes them.
FILE_FACTS: dict[str, FileFact] = {
    'objective': FileFact(FactKind.SHARED, 'Objective'),
    # PositionX and PositionY, in mm: a pair, because half a plate coordinate
    # is not one.
    'plate_pos_mm': FileFact(FactKind.SHARED),
    'z_pos_um': FileFact(FactKind.SHARED, 'PositionZ', 'um'),
    'illumination_ma': FileFact(FactKind.CHANNEL, 'Illumination', 'mA'),
    # Every LED lit for the grab, by layer; Illumination is the named
    # channel's alone, and the one older files carry.
    'illumination_by_channel_ma': FileFact(FactKind.CHANNEL, 'IlluminationByChannel', 'mA'),
    # When those currents were read (frame_record.LightTiming).
    'illumination_timing': FileFact(FactKind.CHANNEL, 'IlluminationTiming'),
    'exposure_time_ms': FileFact(FactKind.CHANNEL, 'ExposureTime', 'ms'),
    'gain_db': FileFact(FactKind.CHANNEL, 'Gain', 'dB'),
    # The camera's own black level parameter, in the camera's own units: no
    # unit is written because none is common to every camera.
    'black_level': FileFact(FactKind.CHANNEL, 'BlackLevel'),
    # ExposureTime is per frame; a summed image integrates this many of them,
    # and without the count the file's exposure understates its integration.
    # A sum of such frames integrates more than its count says, so it states none.
    'frames_summed': FileFact(FactKind.CHANNEL, 'FramesSummed', dropped_by_a_sum=True),
    # The pixel size already folds the binning in, so without it a reader
    # cannot tell a binned frame from a finer objective.
    'binning_size': FileFact(FactKind.SHARED, 'Binning'),
    # The frame's time, and what that time is (frame_record.TimestampSource):
    # a time is written only with its source, which the producer states -- a
    # still and a recorded frame take their time from different clocks, and a
    # defaulted source would name the wrong one.
    'timestamp_iso': FileFact(FactKind.CAPTURE, 'Timestamp'),
    'timestamp_source': FileFact(FactKind.CAPTURE, 'TimestampSource'),
    # The camera-side ChunkTimestamp, and its tick frequency for converting
    # ticks to seconds (1 GHz on Basler USB3, GevTimestampTickFrequency on
    # GigE); the ChunkFrameID / Framecounter integer.
    'timestamp_camera_ticks': FileFact(FactKind.CAPTURE, 'TimestampCameraTicks'),
    'timestamp_camera_tick_hz': FileFact(FactKind.CAPTURE, 'TimestampCameraTickHz'),
    'frame_id': FileFact(FactKind.CAPTURE, 'FrameID'),
    # Each change the frame was judged against: whether it counts, how many
    # frames and milliseconds after the change it came, and who counted the
    # frames that met the change's count.
    'frame_validity': FileFact(FactKind.CAPTURE, 'FrameValidity'),
    # What the capture found the frame to be: the fraction of its pixels at
    # full scale, and whether it was kept dark under a lit LED.
    'saturated_fraction': FileFact(FactKind.CAPTURE, 'SaturatedFraction'),
    'dark': FileFact(FactKind.CAPTURE, 'Dark'),
    # Whether the camera held a layer's stored settings for the frame, and
    # which: a frame taken after a script's raw gain or exposure write says
    # it held none.
    'camera_held_a_channel': FileFact(FactKind.CAPTURE, 'CameraHeldAChannel'),
    'camera_held_channel': FileFact(FactKind.CAPTURE, 'CameraHeldChannel'),
    # Taken where a failed autofocus left the stage, and why it failed.
    'fallback_focus': FileFact(FactKind.CAPTURE, 'FallbackFocus'),
    # A recorded frame's number in its recording, and whether an axis was
    # moving or homing as it arrived: delivery follows the exposure, so this
    # is the state at delivery.
    'recording_frame': FileFact(FactKind.CAPTURE, 'RecordingFrame'),
    'stage_moving': FileFact(FactKind.CAPTURE, 'StageMoving'),
    'channel': FileFact(FactKind.FILE),
    'channel_display': FileFact(FactKind.CHANNEL),
    'excitation_nm': FileFact(FactKind.CHANNEL),
    'filterset': FileFact(FactKind.SHARED),
    'pixel_size_um': FileFact(FactKind.SHARED),
    'instrument': FileFact(FactKind.SHARED),
    # The plate, and the well on it when the frame was taken over one: a
    # well is a place on a plate, so a record cannot name one without it.
    'plate': FileFact(FactKind.SHARED),
    'datetime': FileFact(FactKind.FILE),
    'significant_bits': FileFact(FactKind.FILE),
    'algorithm': FileFact(FactKind.FILE),
}

_PLANE_FACTS = {key: fact for key, fact in FILE_FACTS.items() if fact.plane is not None}


def read_tiff_significant_bits(path: pathlib.Path) -> int:
    """Meaningful payload bits recorded in a TIFF.

    A reader needs this to scale a uint16 file to 8-bit correctly: a 12-bit
    payload stored right-aligned (0..4095) reads full-white only when scaled by
    4095, not by 65535. The depth is read from the OME-XML SignificantBits tag
    on OME files, and from a durable private tag on plain / ImageJ files (which
    have no OME-XML to carry it). Falls back to the container width
    (itemsize * 8) for files that carry neither -- including older files whose
    stored values were left-justified to fill the container, for which
    container-width scaling is the correct interpretation.
    """
    with _open_saved_tiff(path) as tif:
        return _significant_bits_from_open(tif)


def _unreadable(path: pathlib.Path, detail: str = '') -> InputFramesUnusableError:
    """The one refusal for a pixel file that is missing or will not decode, naming it."""
    return InputFramesUnusableError(
        f'Could not read image {path}{": " + detail if detail else ""}.', 'unreadable'
    )


@contextlib.contextmanager
def _open_saved_tiff(path: pathlib.Path):
    """Open a saved TIFF whose first page can be decoded, or raise InputFramesUnusableError.

    A file read while it is still being written, or a damaged one, fails inside
    tifffile with whatever its parser hit first: a short header raises
    struct.error, a header with no page yet IndexError, and a page whose header
    tifffile has reserved but not yet filled in has no pixel type at all. The
    readers promise ``InputFramesUnusableError('unreadable')`` for an
    undecodable file -- a ValueError, which is what their callers catch to
    skip it -- so every such state is raised as that, naming the file, with
    tifffile's own error chained.
    """
    try:
        tif = tf.TiffFile(str(path))
    except struct.error as ex:
        raise _unreadable(path) from ex
    except ValueError as ex:
        raise _unreadable(path, str(ex)) from ex
    with tif:
        try:
            page = tif.pages[0]
        except IndexError as ex:
            raise _unreadable(path, 'it has no page') from ex
        if page.dtype is None:
            raise _unreadable(path, 'its first page has no pixel type')
        yield tif


def _significant_bits_from_open(tif: 'tf.TiffFile') -> int:
    """Resolve significant bits from an already-open TiffFile handle.

    Split out from read_tiff_significant_bits so load_pixels can read the depth
    from the same handle it decodes the pixels with, instead of opening the
    file a second time just for the tag. See read_tiff_significant_bits for the
    OME -> private-tag -> container-width resolution order and rationale.
    """
    ome = tif.ome_metadata
    if ome:
        match = re.search(r'SignificantBits="(\d+)"', ome)
        if match:
            return int(match.group(1))
    tag = tif.pages[0].tags.get(_TIFF_TAG_SIGNIFICANT_BITS)
    if tag is not None and tag.value:
        return int(tag.value)
    return tif.pages[0].dtype.itemsize * 8


def _white_bits_from_open(tif: 'tf.TiffFile') -> int | None:
    """The white a file states for its stored pixels, or None when it states none.

    None is a file written before the white was recorded, or one whose values
    were remapped; either renders at its payload depth, as convert_to_8bit
    does by default.
    """
    tag = tif.pages[0].tags.get(_TIFF_TAG_WHITE_BITS)
    return int(tag.value) if tag is not None else None


def load_pixels(
    path: pathlib.Path,
    *,
    collapse_legacy_false_color: bool = True,
) -> tuple[np.ndarray, int, int | None]:
    """Load a saved frame's pixels together with their significant-bit depth and white.

    The one sanctioned read for saved pixel data: it returns the array AND the
    meaningful payload depth in a single call, so a caller cannot obtain the
    pixels without the depth needed to scale them. A uint16 frame stored
    right-aligned (0..4095 for a 12-bit sensor) reads ~16x dark if scaled as a
    full 16-bit value, so the depth is not optional context -- it is part of
    what the pixels mean. Loading the two apart is the gap that lets a consumer
    silently mis-scale every frame it touches.

    Depth comes from read_tiff_significant_bits (OME SignificantBits -> private
    tag -> container width), the single canonical resolver, so every encoding
    ever written reads correctly: OME right-aligned, plain / ImageJ private-tag,
    legacy left-justified (SignificantBits=16), and 8-bit. Non-TIFF files
    (PNG / JPEG) carry no depth tag, so their container width is the depth.

    The white is returned with them for the same reason: a sum is stored at the
    bits it can reach and shown against one frame's white, so a caller that
    renders it against its depth alone shows it about N times dimmer than it
    was captured.

    Args:
        path: Path to a saved pixel file (TIFF, PNG, or JPEG).
        collapse_legacy_false_color: When True (the mono-uniform consumers --
            folder walk, cell-count preview), a 3-channel TIFF with a single
            populated channel is collapsed to its mono plane so those consumers
            see one shape regardless of file age. Color-capable consumers
            (stitch / zproject / composite / stack) pass False to keep the raw
            channel layout their own color handling expects; the depth is
            returned either way.

    Returns:
        (image, significant_bits, white_bits). image is the stored array with
        values verbatim (right-aligned, dtype preserved); significant_bits is
        the payload depth to hand to convert_to_8bit and the display path;
        white_bits is the depth whose full scale is white, for
        convert_to_8bit's ``white_bits``, or None when the file states none
        and renders at its depth.

    Raises:
        InputFramesUnusableError: ``'unreadable'``, the path does not exist
            or the file cannot be decoded as an image (a ValueError).
    """
    image, sig, white, _ = _load_pixels_and_timestamp(
        path,
        collapse_legacy_false_color=collapse_legacy_false_color,
        read_timestamp=False,
    )
    return image, sig, white


def load_pixels_with_timestamp(
    path: pathlib.Path,
    *,
    collapse_legacy_false_color: bool = True,
) -> tuple[np.ndarray, int, int | None, 'datetime.datetime | None']:
    """Load a frame's pixels, significant-bit depth, white and capture time in one open.

    The video builder needs all three per frame: the pixels to encode, the depth
    to scale them to 8-bit, and the per-frame capture time for the optional
    overlay. Reading them through a single TiffFile handle keeps an
    overlay-enabled build to one open + IFD parse per frame instead of two (one
    for pixels + depth, a second just for the timestamp metadata). For non-TIFF
    inputs (PNG / JPEG), which carry no per-frame timestamp metadata, the
    timestamp is None.

    See load_pixels for the pixel + depth + white contract (the three are
    identical to what load_pixels returns) and read_file_record for the
    metadata shapes the timestamp is recovered from.

    Returns:
        (image, significant_bits, white_bits, timestamp). timestamp is None
        when the file carries no readable per-frame capture time.
    """
    return _load_pixels_and_timestamp(
        path,
        collapse_legacy_false_color=collapse_legacy_false_color,
        read_timestamp=True,
    )


def _load_pixels_and_timestamp(
    path: pathlib.Path,
    *,
    collapse_legacy_false_color: bool,
    read_timestamp: bool,
) -> tuple[np.ndarray, int, int | None, 'datetime.datetime | None']:
    """Shared loader behind load_pixels and load_pixels_with_timestamp.

    Keeps a single source of truth for the pixel + depth read so the two public
    entry points cannot drift. When read_timestamp is True the per-frame capture
    time is recovered from the same open TIFF handle, so no caller opens a file
    twice. A timestamp read that fails leaves the pixels intact: the frame still
    loads, the overlay just falls back to its caller-supplied time.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise _unreadable(path, 'no such file')

    if is_tiff(path):
        # One open serves every read: the pixel array, the significant-bit depth,
        # and (when asked) the per-frame timestamp come from the same TiffFile
        # handle. load_pixels runs per tile in stitch / zproject / composite, so a
        # second open just for the depth tag -- or the timestamp -- would double
        # the file opens and IFD parses of every post-processing or video run.
        timestamp = None
        with _open_saved_tiff(path) as tif:
            try:
                image = tif.asarray()
            except (RuntimeError, ValueError) as ex:
                # Pixel data cut short: the deflate codec raises its own
                # RuntimeError subclass, and tifffile a TiffFileError (a
                # ValueError) for an LZW strip that cannot be reshaped.
                raise _unreadable(path, str(ex)) from ex
            sig = _significant_bits_from_open(tif)
            white = _white_bits_from_open(tif)
            if read_timestamp:
                timestamp = frame_time(_record_from_open(tif))
        if collapse_legacy_false_color:
            image = _collapse_legacy_false_color(image, path)
        # Debug (not info): load_pixels runs per-tile in stitch/zproject, so this
        # is high-volume; it records how a saved file was interpreted on read-back
        # (depth + whether a false-color file collapsed to mono) when diagnosing.
        logger.debug(
            f'[ImageLoad tiff] {path.name} dtype={image.dtype} shape={image.shape} '
            f'color={is_color_image(image)} significant_bits={sig} white_bits={white} '
            f'collapse={collapse_legacy_false_color}'
        )
        return image, sig, white, timestamp

    # Non-TIFF (PNG / JPEG): no depth carrier, so the container width is the
    # depth, and no per-frame timestamp metadata to recover. cv2 returns color
    # files in BGR channel order; the depth-sensitive payloads are mono, where
    # channel order is moot.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise _unreadable(path)
    sig = image.dtype.itemsize * 8
    logger.debug(
        f'[ImageLoad cv2] {path.name} dtype={image.dtype} shape={image.shape} '
        f'color={is_color_image(image)} significant_bits={sig} '
        f'collapse={collapse_legacy_false_color}'
    )
    return image, sig, None, None


def read_image_geometry(path: pathlib.Path) -> tuple[tuple[int, ...], np.dtype]:
    """Read an image's (shape, dtype) without decoding its pixels when possible.

    A caller that only needs the spatial size and dtype to size an output canvas
    (the stitcher's per-region setup) should not pay a full pixel decode -- nor a
    second open for the depth tag it then discards. A TIFF's IFD header carries
    the page shape and dtype, so a single TiffFile open answers without reading
    the pixel data. Non-TIFF inputs (PNG / JPEG) have no separable header, so
    they fall back to a full load through load_pixels.

    The returned shape matches what load_pixels would return for the raw file
    (no legacy false-color collapse): mono is (H, W); a 3-channel frame is
    (H, W, 3).

    Args:
        path: Path to a saved pixel file (TIFF, PNG, or JPEG).

    Returns:
        (shape, dtype) of the stored array.

    Raises:
        InputFramesUnusableError: ``'unreadable'``, the path does not exist
            or the file cannot be decoded as an image (a ValueError).
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise _unreadable(path, 'no such file')
    if is_tiff(path):
        with _open_saved_tiff(path) as tif:
            page = tif.pages[0]
            return tuple(page.shape), page.dtype
    image, _, _ = load_pixels(path, collapse_legacy_false_color=False)
    return image.shape, image.dtype


def resolve_output_depth(input_depths) -> int:
    """Pick the significant-bit depth a derived output should carry.

    A projection / stitch / composite copies its inputs' pixel values
    verbatim, so the output's honest depth is the inputs' depth. The inputs
    of one derived output are the same channel captured at the same camera
    bit-depth, so the depths are uniform in normal operation. When they are
    not (genuinely mixed-source data), the deepest input is the safe label:
    a shallower payload sits right-aligned in the same container and stays
    correct under the deeper output's scaling, whereas labeling the output
    shallower than its deepest input would clip that input on display.

    Args:
        input_depths: The significant_bits values load_pixels returned for
            the inputs being combined.

    Returns:
        The depth to hand to the output write.

    Raises:
        ValueError: No input depths were supplied (no inputs were loaded).
    """
    depths = set(input_depths)
    if not depths:
        raise ValueError('resolve_output_depth needs at least one input depth')
    if len(depths) > 1:
        logger.warning(
            f'[ImageUtils] Combining inputs of mixed significant-bit depth '
            f'{sorted(depths)}; tagging the output as {max(depths)}-bit.'
        )
    return max(depths)


def resolve_output_white(inputs: Iterable[tuple[int, int | None]]) -> int | None:
    """The white a derived output that keeps its inputs' values should state.

    A stitch or a projection copies its inputs' pixel values, so it is shown
    against the white they were shown against; an input that states none is
    shown at its depth, so that depth is its white. Inputs whose whites differ
    give the output no single white to state, and it then renders at its depth.

    Args:
        inputs: (significant_bits, white_bits) for each input, as load_pixels
            returned them.

    Raises:
        ValueError: No inputs were supplied (no inputs were loaded).
    """
    whites = {depth if white is None else white for depth, white in inputs}
    if not whites:
        raise ValueError('resolve_output_white needs at least one input')
    return whites.pop() if len(whites) == 1 else None


def summed_significant_bits(frames_summed: int, frame_bits: int) -> int:
    """The depth a sum of frames is tagged with: the bits its largest value needs.

    A sum of N frames of b bits reaches N x (2^b - 1), so that is the range its
    file claims -- 10 bits for four 8-bit frames, 14 for four 12-bit ones --
    up to the 16-bit container the sum is stored in, where it saturates. A
    single frame is its own depth. Tagging a sum with the container width
    instead would scale it against a range it never reaches and render it
    near-black wherever the tag sets the scale.

    Args:
        frames_summed: How many frames the sum holds; 1 for a single frame.
        frame_bits: The depth of each summed frame, as delivered.

    Returns:
        The significant bits the summed frame carries.
    """
    if frames_summed < 1:
        raise ValueError(f'a frame holds at least one capture, not {frames_summed}')
    return min(16, (frames_summed * ((1 << frame_bits) - 1)).bit_length())


def summed_full_scale(frames_summed: int, frame_bits: int) -> int:
    """The largest value a sum of frames can hold: where it saturates.

    N frames at full scale, or the uint16 container's ceiling where that comes
    first. A sum's saturation is measured against this, never against its tag:
    the tag is a whole number of bits, so for N not a power of two a sum of
    blown frames sits well below the tag's maximum (three blown 8-bit frames
    are 765 of a 10-bit 1023) and would read as unsaturated.

    Args:
        frames_summed: How many frames the sum holds; 1 for a single frame.
        frame_bits: The depth of each summed frame, as delivered.

    Returns:
        The full-scale value of the summed frame.
    """
    if frames_summed < 1:
        raise ValueError(f'a frame holds at least one capture, not {frames_summed}')
    return min(frames_summed * ((1 << frame_bits) - 1), int(np.iinfo(np.uint16).max))


def _read_ome_input_metadata(ome_xml: str) -> dict | None:
    """Recover the record from an OME description LVP wrote.

    The record comes in two parts. The file's own facts: the image's LVP
    MapAnnotation, which carries every fact the write put in no schema
    attribute (a Plane fact under its Plane name, any other under its record
    key), Pixels PhysicalSizeX, and the Channel's Name and excitation when the
    file has one Channel. And the Plane facts every ``<Plane>`` agrees on
    (agreed_facts): a still's one Plane states its own, and a hyperstack
    whose planes stand at different Z states no Z, never its first plane's.
    Each is read through the one Plane reading the structured containers use,
    so an OME file states what an ImageJ file of the same frame does. A file
    written before the annotation carried a fact states none of it: a derived
    output states only what its input did. Returns None on a parse failure,
    an annotated value that is not JSON among them, or when there is no
    Pixels element to read at all. tifffile writes no Plane element when the
    file states none of its schema attributes (no position, no exposure, no
    timing), and the file's own facts are read all the same.
    """
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None

    def _local(tag: str) -> str:
        return tag.rsplit('}', 1)[-1]

    image = next((el for el in root.iter() if _local(el.tag) == 'Image'), None)
    pixels = (
        next((el for el in image if _local(el.tag) == 'Pixels'), None)
        if image is not None
        else None
    )
    if pixels is None:
        return None
    planes = [el for el in pixels if _local(el.tag) == 'Plane']
    channels = [el for el in pixels if _local(el.tag) == 'Channel']

    def _float(attrs: dict, key: str):
        raw = attrs.get(key)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    annotated: dict = {}
    referenced = {el.get('ID') for el in image if _local(el.tag) == 'AnnotationRef'}
    for annotation in root.iter():
        if (
            _local(annotation.tag) == 'MapAnnotation'
            and annotation.get('ID') in referenced
            and annotation.get('Namespace') == _LVP_OME_ANNOTATION_NAMESPACE
        ):
            try:
                annotated |= {
                    entry.get('K'): json.loads(entry.text or '')
                    for entry in annotation.iter()
                    if _local(entry.tag) == 'M'
                }
            except json.JSONDecodeError:
                return None

    # The scale is a hard key whose None the writer reads as no PhysicalSize.
    flat: dict = {'pixel_size_um': _stated_scale(_float(pixels.attrib, 'PhysicalSizeX'))}
    # A record key is lower-case and a Plane name capitalized, so the two
    # cannot collide; what is not a record key is the Plane's to read.
    flat |= _plane_facts({key: value for key, value in annotated.items() if key not in FILE_FACTS})
    flat |= {key: value for key, value in annotated.items() if key in FILE_FACTS}
    if len(channels) == 1 and channels[0].attrib.get('Name'):
        flat['channel'] = channels[0].attrib['Name']
        excitation = _float(channels[0].attrib, 'ExcitationWavelength')
        if excitation is not None:
            flat['excitation_nm'] = excitation
    if planes:
        flat |= agreed_facts(
            _plane_facts(
                {
                    key: value
                    for key in _OME_PLANE_VALUES
                    if (value := _float(plane.attrib, key)) is not None
                }
            )
            for plane in planes
        )
    return flat


def _plane_facts(plane: dict) -> dict:
    """The flat facts one Plane block states, whichever container held it.

    The one reading of a Plane, for the shaped and ImageJ descriptions and for
    the OME container's schema attributes with its LVP annotation. Every fact
    is present only when the Plane states it: the producer omits a value that
    was unknown at capture (a failed camera read; an LED that was off, as on
    every dark or luminescence frame; an axis that had lost its reference, or
    a scope with no X and Y), so such a frame still forwards everything it
    does state and no stand-in is invented on the way to a derived output.
    """
    facts: dict = {}
    if 'PositionX' in plane and 'PositionY' in plane:
        facts['plate_pos_mm'] = {'x': plane['PositionX'], 'y': plane['PositionY']}
    for key, fact in _PLANE_FACTS.items():
        # A file written while the objective was a hard key states "none" as {}.
        if fact.plane in plane and plane[fact.plane] != {}:
            facts[key] = plane[fact.plane]
    if 'objective' in facts:
        facts['objective'] = objectives_loader.under_field_names(facts['objective'])
    return facts


# A numpy scalar left in the metadata dict (e.g. a stage position that stayed
# np.float64) serializes into the ImageJ description as its repr -- np.float64(N)
# under numpy >= 2 -- which is a call expression, not a literal, so literal_eval
# rejects it. Unwrap np.<type>(...) back to the inner value before parsing so a
# tile whose positions were never cast to plain floats still round-trips. The
# inner group forbids parentheses, so a numpy scalar (which never nests) is
# unwrapped without disturbing any surrounding literal.
_IMAGEJ_NUMPY_SCALAR_RE = re.compile(r'np\.\w+\(([^()]*)\)')


def _imagej_to_structured(imagej_metadata: dict) -> dict:
    """Normalize an ImageJ-container metadata dict to the shaped-metadata shape.

    write_tiff routes a 16-bit mono non-OME frame through tifffile's ImageJ
    writer, which serializes the structured Plane/Channel/PhysicalSize block
    into an "ImageJ=..." ImageDescription with the keys lowercased and the
    nested dicts stringified. Reverse that here -- restore the capitalized keys
    the readers expect and literal_eval the stringified subtrees -- so an
    ImageJ tile feeds the exact same extraction as a shaped tile. A subtree that
    does not parse is left as its raw string; the caller's required-key checks
    then reject the tile rather than crashing.
    """

    def parsed(key):
        value = imagej_metadata.get(key)
        if isinstance(value, str):
            try:
                return ast.literal_eval(_IMAGEJ_NUMPY_SCALAR_RE.sub(r'\1', value))
            except (ValueError, SyntaxError):
                return value
        return value

    structured: dict = {}
    if 'physicalsizex' in imagej_metadata:
        structured['PhysicalSizeX'] = imagej_metadata['physicalsizex']
    for lower, canonical in (
        ('plane', 'Plane'),
        ('channel', 'Channel'),
        ('instrument', 'Instrument'),
        ('plate', 'Plate'),
        ('filterset', 'FilterSet'),
        ('algorithm', 'Algorithm'),
    ):
        if lower in imagej_metadata:
            structured[canonical] = parsed(lower)
    return structured


def _structured_metadata(tif: 'tf.TiffFile') -> dict | None:
    """Resolve one open TIFF's structured acquisition metadata, container-agnostic.

    write_tiff serializes the same structured block into either a tifffile
    "shaped" JSON ImageDescription (8-bit / color) or an ImageJ ImageDescription
    (16-bit mono); both deserialize to the same dict shape so readers never have
    to know which container a tile landed in. OME-XML is a distinct shape the
    callers handle separately. Returns None when neither structured container is
    present (a bare or OME-only tile).
    """
    shaped = tif.shaped_metadata
    if shaped:
        return shaped[0]
    imagej = tif.imagej_metadata
    if imagej:
        return _imagej_to_structured(imagej)
    return None


def read_file_record(path: pathlib.Path) -> dict | None:
    """A saved frame's record, read from its file whatever container holds it.

    The one reader of a saved file's record, for every consumer: a shaped
    (8-bit, colour) or ImageJ (16-bit mono) TIFF's structured block, an OME
    TIFF's schema attributes with its LVP MapAnnotation, and every recorded
    frame LVP has shipped (v2.3 to v4.0.0-beta30 wrote a flat dict), each
    read into the record's keys (``FILE_FACTS``), so a fact a file states is
    found whichever container it landed in. A PNG or JPEG carries no record.

    Returns:
        The record, as ``write_tiff``'s ``metadata`` names its facts, or
        None when the file opens and carries no LVP record (a bare
        ``tifffile.imwrite``, another producer's file, a PNG). A scale the
        file does not state reads back as ``pixel_size_um`` None.

    Raises:
        InputFramesUnusableError: ``'unreadable'``, the path does not exist
            or the file cannot be opened or decoded (a ValueError).
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise _unreadable(path, 'no such file')
    if not is_tiff(path):
        return None
    with _open_saved_tiff(path) as tif:
        return _record_from_open(tif)


def is_ome_tiff(path: pathlib.Path | str) -> bool:
    """True when this path names an OME-TIFF: ``.ome.tif`` or ``.ome.tiff``, case-blind.

    LVP names every file it writes by its format, an OME file ``.ome.tiff``
    and a plain one ``.tiff``, so the name answers which it is, as the
    suffix answers ``is_tiff``.
    """
    return [suffix.lower() for suffix in pathlib.Path(path).suffixes[-2:]] in (
        ['.ome', '.tif'],
        ['.ome', '.tiff'],
    )


def derived_file_suffix(output_format: image_mode.OutputFormat) -> str:
    """The suffix a derived file of *output_format* is named with: ``.ome.tiff`` or ``.tiff``."""
    return '.ome.tiff' if output_format == image_mode.OutputFormat.OME_TIFF else '.tiff'


def derived_file_format(paths: Iterable[pathlib.Path | str]) -> image_mode.OutputFormat:
    """The format a file derived from *paths* is written in: the one its inputs share.

    OME-TIFF only when every input is an OME-TIFF (``is_ome_tiff``);
    otherwise plain TIFF -- a group mixing the two, or a PNG or JPEG. A
    derived file states a fact only when every input states the same, and
    its container is one.
    """
    paths = list(paths)
    if paths and all(is_ome_tiff(path) for path in paths):
        return image_mode.OutputFormat.OME_TIFF
    return image_mode.OutputFormat.TIFF


def _record_from_open(tif: 'tf.TiffFile') -> dict | None:
    """The record one open TIFF states, or None when it states none.

    Split out from read_file_record so a reader that decodes the pixels reads
    the record from the same handle instead of opening the file twice.
    """
    structured = _structured_metadata(tif)
    if structured is not None and 'Plane' in structured:
        record = _structured_record(structured)
    elif structured is not None:
        record = _legacy_frame_record(structured)
    elif tif.ome_metadata:
        record = _read_ome_input_metadata(tif.ome_metadata)
    else:
        record = None
    tag = tif.pages[0].tags.get('DateTime')
    if record is not None and tag is not None:
        stated = _parsed_tiff_datetime(tag.value)
        if stated is not None:
            record['datetime'] = stated
    return record


def _stated_scale(stated) -> float | None:
    """A file's scale, in um/pixel, or None when it states none.

    The writer omits PhysicalSizeX when it measured no scale, so a file
    without one is a file saying "no scale", and None is how the record says
    the same: write_tiff takes it as no PhysicalSize and no absolute
    resolution unit.
    """
    if stated is None:
        return None
    stated = float(stated)
    # A non-positive scale is not a measurement; treat it as absent rather than
    # dividing by it downstream.
    return stated if stated > 0 else None


def _structured_record(structured: dict) -> dict | None:
    """The record a shaped or ImageJ structured block states.

    None when the block lacks the channel every LVP still names (another
    producer's file, or a non-acquisition frame type): the derived output then
    states nothing it read.
    """
    try:
        flat: dict = {
            'pixel_size_um': _stated_scale(structured.get('PhysicalSizeX')),
            'channel': structured['Channel']['Name'][0],
        }
    except (KeyError, IndexError, TypeError):
        return None
    flat |= _plane_facts(structured['Plane'])

    # Spectral identity, written by the same producer: per-channel fields
    # from the Channel block, the unit-level filterset from its own key.
    channel_block = structured['Channel']
    if channel_block.get('DisplayName'):
        flat['channel_display'] = channel_block['DisplayName'][0]
    if channel_block.get('ExcitationWavelength'):
        flat['excitation_nm'] = channel_block['ExcitationWavelength'][0]
    if 'FilterSet' in structured:
        flat['filterset'] = structured['FilterSet']
    if 'Algorithm' in structured:
        flat['algorithm'] = structured['Algorithm']

    # The Instrument block also renders the objective, so an Instrument
    # whose microscope and detector state nothing states no instrument.
    inst = structured.get('Instrument', {})
    microscope = inst.get('Microscope', {})
    instrument = stated_fields(
        manufacturer=microscope.get('Manufacturer'),
        model=microscope.get('Model'),
        serial_number=microscope.get('SerialNumber'),
        camera_model=inst.get('Detector', {}).get('Model'),
    )
    if instrument:
        flat['instrument'] = instrument

    if 'Plate' in structured:
        p = structured['Plate']
        flat['plate'] = stated_fields(
            name=p.get('Name'),
            rows=p.get('Rows'),
            columns=p.get('Columns'),
            standard=p.get('Standard'),
            well_label=p.get('WellLabel'),
        )

    return flat


# A recorded frame's flat dict before a recorded frame was written as a still
# is: v2.3 and v3.0.0 wrote ``timestamp`` (with microseconds) and
# ``frame_num``; v4.0.0-beta29 and beta30 added ``timestamp_iso``, the scale
# and the camera's ticks and frame id. Each key as the record names it.
_LEGACY_FRAME_KEYS = {
    'frame_num': 'recording_frame',
    'timestamp_iso': 'timestamp_iso',
    'timestamp_camera_ticks': 'timestamp_camera_ticks',
    'timestamp_camera_tick_hz': 'timestamp_camera_tick_hz',
    'frame_id': 'frame_id',
}
_LEGACY_FRAME_TIMESTAMP = '%Y:%m:%d %H:%M:%S.%f'


def _legacy_frame_record(flat: dict) -> dict | None:
    """The record a recorded frame written before v4.0.0-beta31 states, or None
    when the flat dict is no recorded frame's (a bare tifffile.imwrite's
    ``{'shape': ...}``)."""
    record = {
        key: flat[legacy]
        for legacy, key in _LEGACY_FRAME_KEYS.items()
        if flat.get(legacy) is not None
    }
    if 'timestamp_iso' not in record and flat.get('timestamp'):
        record['timestamp_iso'] = datetime.datetime.strptime(
            flat['timestamp'], _LEGACY_FRAME_TIMESTAMP
        ).isoformat(timespec='microseconds')
    if not record:
        return None
    record['pixel_size_um'] = _stated_scale(flat.get('pixel_size_um'))
    return record


def read_pixel_size_um(path: pathlib.Path) -> float | None:
    """A saved file's scale, in um/pixel, or None when it states none.

    Reads the value the writer stored rather than inverting the TIFF
    resolution tag: the stored number is what LVP measured, and the tag is a
    derived encoding of it for external readers.

    Raises:
        InputFramesUnusableError: ``'unreadable'``, as read_file_record.
    """
    record = read_file_record(path)
    return record.get('pixel_size_um') if record is not None else None


def frame_time(record: dict | None) -> datetime.datetime | None:
    """The time a record states its frame was taken, or None when it states none."""
    stated = record.get('timestamp_iso') if record is not None else None
    return datetime.datetime.fromisoformat(stated) if stated else None


def _parsed_tiff_datetime(stated: str) -> datetime.datetime | None:
    """The time a file's DateTime tag states, or None when it states none LVP can read.

    LVP writes the TIFF form; files a derived build wrote before the form was
    one carry the ISO form, read too. A tag in neither form, from another
    producer, states no time this reader can give.
    """
    try:
        return datetime.datetime.strptime(stated, _TIFF_DATETIME)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.datetime.fromisoformat(stated)
    except (TypeError, ValueError):
        return None


def read_tiff_depth_and_record(path: pathlib.Path) -> tuple[int, dict | None]:
    """Header-only read of a TIFF's payload depth and its record.

    One open serves both; the pixel data is never decoded, so a pre-scan over
    hundreds of recorded frames costs one IFD parse each. See
    read_tiff_significant_bits for the depth resolution order and
    read_file_record for the record, None when the file states none;
    frame_time gives the record's capture time.
    """
    with _open_saved_tiff(path) as tif:
        return _significant_bits_from_open(tif), _record_from_open(tif)


def agreed_facts(records: Iterable[dict | None]) -> dict:
    """The facts every record states with one value.

    What a file derived from several inputs can state of them: a fact one
    input states differently, or does not state, is no fact of the output (a
    stitch of tiles at 3 and 9 dB states no gain, a projection of slices at
    different Z states no Z). An input with no record states nothing, so
    nothing is agreed.

    Raises:
        ValueError: No records were given (no inputs were read).
    """
    records = list(records)
    if not records:
        raise ValueError('agreed_facts needs at least one input record')
    if any(record is None for record in records):
        return {}
    first, *rest = records
    return {
        key: value
        for key, value in first.items()
        if all(key in record and record[key] == value for record in rest)
    }


def build_postproc_output_metadata(
    input_records: Iterable[dict | None],
    channel: str,
    *,
    significant_bits: int,
    adds_frames: bool,
    plate_pos_mm_override: dict | None = None,
    algorithm: str | None = None,
) -> dict:
    """Build a write_tiff metadata dict for a post-processing output.

    Takes every input's record (read_file_record) and states what they agree
    on (agreed_facts), as each fact's kind (``FILE_FACTS``) lets a derived
    file: the shared facts (objective, scale, position, instrument, and the
    plate with its well), and a layer's settings only when the output is on
    its inputs' channel. A capture's own facts (its time, its id, how it was
    judged) describe the original capture, not the derived image, and a
    file's own facts are the output's writer's to set.

    The depth is supplied by the caller, not re-read here: the caller
    loads its input pixels through load_pixels, so it already holds the
    significant-bit depth that travels with those pixels. Passing it in
    keeps a single depth read per output instead of opening the input a
    second time, and keeps the output's depth coupled to the pixels that
    were actually projected / stitched / merged.

    ``datetime`` is set to the post-processing wall-clock time so derived
    outputs are distinguishable from the captures that fed them.

    Callers in stitcher pass ``plate_pos_mm_override`` with the stitched
    region's geometric center; zprojector leaves it None and states the
    slices' position where they agree. An input with no record (an external
    PNG, a bare TIFF) gives an output that states no position, acquisition
    setting or scale: unknown stays unknown, because a number in a TIFF tag
    is read downstream as a measurement.

    Args:
        input_records: Every input's record, as read_file_record read it.
        channel: Layer key for the derived output (Ex405 / Ex488 / Ex589
            / Lumi / BF / PC / DF), or 'Composite'.
        adds_frames: Whether each output pixel adds its inputs' (a Sum), so
            it integrates more frames than any input states.
        plate_pos_mm_override: Optional plate_pos_mm dict to override
            the value read from the input.
        algorithm: Optional producing-algorithm name. Stamped into the
            output TIFF so a consumer can tell a degraded edge-to-edge
            montage from a registered stitch without re-deriving it.

    Returns:
        Dict ready to pass as ``write_tiff``'s ``metadata`` parameter.
    """
    agreed = agreed_facts(input_records)
    metadata = _derived_facts(
        agreed,
        on_its_inputs_channel=agreed.get('channel') == channel,
        adds_frames=adds_frames,
    )
    # The writer's hard key, in its "none" form when the inputs agree on no scale.
    metadata['pixel_size_um'] = agreed.get('pixel_size_um')
    metadata['channel'] = channel
    metadata['datetime'] = datetime.datetime.now()

    if plate_pos_mm_override is not None:
        metadata['plate_pos_mm'] = plate_pos_mm_override

    # A derived output inherits its inputs' depth: a stitch or projection copies
    # the input pixels verbatim, so the output's true significant-bit depth is
    # the input's. The caller carries it from the load_pixels read of those
    # inputs, so the output is tagged honestly instead of defaulting to
    # container width and reading back dark.
    metadata['significant_bits'] = significant_bits

    if algorithm is not None:
        metadata['algorithm'] = algorithm

    return metadata


def _derived_facts(agreed: dict, *, on_its_inputs_channel: bool, adds_frames: bool) -> dict:
    """The facts a file derived from its inputs states of what they agree on.

    The one rule for every derived file, a post-processing output and a
    hyperstack alike: the shared facts always; a layer's settings only while
    the layer travels, since a derived output on another channel (a
    composite) must not pair another layer's exposure, light or display name
    with its own channel, and a sum integrates more frames than an input
    states; no capture's own fact, since the derived file was not captured;
    no file's own fact, which is the output's writer's to set.
    """

    def _stated_by_the_output(fact: FileFact) -> bool:
        if fact.kind is FactKind.CHANNEL:
            return on_its_inputs_channel and not (adds_frames and fact.dropped_by_a_sum)
        return fact.kind is FactKind.SHARED

    return {key: value for key, value in agreed.items() if _stated_by_the_output(FILE_FACTS[key])}


def require_uniform_geometry(
    labeled_arrays: Iterable[tuple[str, np.ndarray]], operation: str
) -> None:
    """Reject combine inputs that were not stitched to one shared geometry.

    Combining images across a tile-group -- compositing channels, projecting a
    Z-stack, assembling a hyperstack -- requires every input to share one (H, W)
    canvas. Producing that geometry is the stitcher's responsibility; when it is
    violated, the raw numpy failure is a cryptic broadcast / stack error that
    hides which images disagree. This surfaces the violation as a clear,
    actionable message naming each input and its shape.

    Args:
        labeled_arrays: (label, ndarray) pairs -- the images about to be combined.
        operation: Verb phrase for the message, e.g. 'composite this well'.

    Raises:
        InputFramesUnusableError: ``'geometry'``, the inputs do not all
            share one (H, W).
    """
    shapes = [(label, tuple(array.shape[:2])) for label, array in labeled_arrays]
    if len({hw for _, hw in shapes}) > 1:
        detail = ', '.join(f'{label} {h}x{w}' for label, (h, w) in shapes)
        raise InputFramesUnusableError(
            f'Cannot {operation}: inputs were not stitched to one geometry '
            f'({detail}). Every layer / slice of a tile-group must be stitched '
            f'to the same canvas before it can be combined.',
            'geometry',
        )


def stated_fields(**fields) -> dict:
    """The fields that hold a value: a fact the catalogue or the instrument
    did not give is absent, never a stand-in. ``''`` is a catalogue's empty
    entry and states nothing either; a stand-in written into a file reads
    downstream as the instrument's own report."""
    return {key: value for key, value in fields.items() if value is not None and value != ''}


def _stated_blocks(**blocks: dict) -> dict:
    """The sub-blocks that state anything: an empty one would claim a part
    the file says nothing about."""
    return {key: block for key, block in blocks.items() if block}


def _instrument_and_plate_blocks(record: dict) -> dict:
    """The ``Instrument`` and ``Plate`` blocks a record renders to.

    The one writer of the two, for a still, a derived file and a
    hyperstack. Each is a rendering for Bio-Formats and Fiji, drawn from the
    record's ``objective``, ``instrument`` and ``plate``, not a second
    store: the objective's home is ``Plane.Objective``, and in an OME file
    every other fact's is the LVP annotation. Its keys are the OME schema's
    attribute names, so the OME splice writes the Instrument as it stands.
    A block that states nothing is absent. Whether there is a plate is the
    record's to say, where it is made; a Plate is rendered exactly when the
    record holds one. The OME container takes the Instrument alone: a schema
    Plate in each file would import into OMERO as a plate per file. Not yet
    rendered: an OME LightSource (LED wavelength and power are not tracked
    per colour), the Detector's gain and zoom, an OME FilterSet (filter
    wheel and dichroic are not tracked).
    """
    objective = record.get('objective') or {}
    instrument = record.get('instrument') or {}
    plate = record.get('plate') or {}
    working_distance = stated_fields(
        WorkingDistance=objective.get('working_distance_mm'),
    )
    if working_distance:
        working_distance['WorkingDistanceUnit'] = 'mm'
    blocks = {
        'Instrument': _stated_blocks(
            Microscope=stated_fields(
                Manufacturer=instrument.get('manufacturer'),
                Model=instrument.get('model'),
                SerialNumber=instrument.get('serial_number'),
            ),
            Objective=stated_fields(
                NominalMagnification=objective.get('magnification'),
                LensNA=objective.get('numerical_aperture'),
            )
            | working_distance,
            Detector=stated_fields(Model=instrument.get('camera_model')),
        ),
        'Plate': stated_fields(
            Name=plate.get('name'),
            Rows=plate.get('rows'),
            Columns=plate.get('columns'),
            WellLabel=plate.get('well_label'),
            Standard=plate.get('standard'),
        ),
    }
    return _stated_blocks(**blocks)


def build_hyperstack_output_metadata(
    input_records: Iterable[dict | None],
    *,
    channel_names: list[str],
    channel_colors: list[Color],
    plane_positions: dict,
    significant_bits: int,
    pixel_size_um: float | None,
) -> dict:
    """Build a TZCYX hyperstack OME metadata dict for ``write_hyperstack_tiff``.

    States what its input frames agree on (agreed_facts) as any derived file
    does (``_derived_facts``): the scope, the objective, and the plate with
    its well always, a layer's settings only on a one-channel stack of that
    layer, no capture's own fact. Per-plane positions arrive as parallel
    lists (one entry per T*Z*C plane in scan order); they land in the
    OME-XML ``<Plane>`` elements via tifffile's per-axis list convention.
    Every stated fact no schema attribute holds rides in the LVP
    MapAnnotation, and the ``Instrument`` block is spliced into the
    description by ``write_hyperstack_tiff``, since tifffile drops it.

    Pixel size is supplied by the caller, which sources it from the input
    frames' own PhysicalSizeX. Passing None omits the PhysicalSize keys
    entirely, for inputs that carry no scale; the output then makes no
    measurement claim rather than an invented one.

    Args:
        input_records: Every input frame's record, as read_file_record
            read it.
        channel_names: One channel name per C-axis position.
        channel_colors: Each channel's colour, in ``channel_names``' order.
        plane_positions: Dict with PositionX / PositionY / PositionZ
            lists, one entry per T*Z*C plane in scan order, plus a
            required DeltaT entry: the per-plane seconds from the
            earliest plane when capture times were measured, else None
            (no timing claim is written). Caller is responsible for
            list-length consistency with the data array.
        significant_bits: 8 for uint8 captures, 16 for uint16.
        pixel_size_um: Hyperstack pixel size in microns: the scale its
            input frames agree on, or None.

    Returns:
        OME-shaped metadata dict; pass to ``write_hyperstack_tiff``.
    """
    inflat = agreed_facts(input_records)
    facts = _derived_facts(
        inflat,
        on_its_inputs_channel=channel_names == [inflat.get('channel')],
        adds_frames=False,
    )

    num_planes = len(plane_positions['PositionX'])

    # Per-channel OME Color hints. OME-XML uses a signed 32-bit RGBA
    # integer per channel (R << 24 | G << 16 | B << 8 | A,
    # two's-complement-folded into int32). Channel.Color is the
    # container's ONLY color carrier: tifffile drops metadata['LUTs']
    # under ome=True, and embedding ImageJ LUTs instead was considered
    # and rejected -- that container is mutually exclusive with OME and
    # cannot represent BigTIFF. FIJI applies Channel.Color only when
    # opened via the Bio-Formats Importer in Composite/Colorized mode;
    # plain File>Open renders ImageJ default LUTs. Grey is white (-1).
    ome_colors = [_ome_rgba(color) for color in channel_colors]

    # A position is written only when every plane has one, like the timing
    # below: tifffile writes a missing value as the literal PositionX="None",
    # which no OME reader accepts as a number. A recording whose position the
    # scope did not know, or a scope with no stage, states no position. X and
    # Y travel as a pair because half a plate coordinate is not one; Z is
    # independent.
    # A pandas column holding a missing value hands it back as NaN, not None.
    def _every_plane_has(values) -> bool:
        return all(v is not None and not np.isnan(v) for v in values)

    plane: dict = {}
    if _every_plane_has(plane_positions['PositionX']) and _every_plane_has(
        plane_positions['PositionY']
    ):
        plane['PositionX'] = plane_positions['PositionX']
        plane['PositionY'] = plane_positions['PositionY']
        plane['PositionXUnit'] = ['mm'] * num_planes
        plane['PositionYUnit'] = ['mm'] * num_planes
    if _every_plane_has(plane_positions['PositionZ']):
        plane['PositionZ'] = plane_positions['PositionZ']
        plane['PositionZUnit'] = [OME_UNIT_MICROMETER] * num_planes
    # Timing is written only when the caller measured it (per-plane
    # seconds from the earliest plane; DeltaT is a required key, None when
    # unmeasured). Like the pixel-size claim below, an absent DeltaT is
    # the honest signal that timing was not recorded; an invented list
    # would read downstream as a measurement.
    if plane_positions['DeltaT'] is not None:
        plane['DeltaT'] = plane_positions['DeltaT']
        plane['DeltaTUnit'] = ['s'] * num_planes

    metadata: dict = {
        'axes': 'TZCYX',
        'SignificantBits': significant_bits,
        'Pixels': {},
        'Channel': {'Name': channel_names, 'Color': ome_colors},
        'Plane': plane,
    }

    # A scale is written only when one is known. Emitting the keys with a null
    # value would still read downstream as a PhysicalSize claim, and a reader
    # cannot tell an invented scale from a measured one -- the absence is the
    # honest signal that this file cannot be measured.
    if pixel_size_um is not None:
        metadata['Pixels'].update(
            {
                'PhysicalSizeX': pixel_size_um,
                'PhysicalSizeXUnit': OME_UNIT_MICROMETER,
                'PhysicalSizeY': pixel_size_um,
                'PhysicalSizeYUnit': OME_UNIT_MICROMETER,
            }
        )

    # The hyperstack's exposure and excitation have no schema attribute here:
    # tifffile writes per-plane values only for the keys it is given lists of.
    metadata['MapAnnotation'] = _ome_annotation(
        facts, plane_rendered=frozenset(plane), record_rendered=_OME_HYPERSTACK_RENDERED
    )
    instrument = _instrument_and_plate_blocks(facts).get('Instrument')
    if instrument:
        metadata['Instrument'] = instrument
    return metadata


def imread_color(path, *, is_color_native: bool = False) -> 'np.ndarray':
    """Color-camera-aware image read. Phase 2 activation pending.

    Stub helper that exists so Phase 2 (color-native camera path) can
    flip the camera capability flag without touching processing modules.
    Today the mono pipeline reads through ``tifffile.imread`` directly;
    when a color-camera customer arrives, this wrapper routes to a
    Bayer-aware reader.

    Args:
        path: TIFF / PNG file path.
        is_color_native: Camera-capability flag from
            ``scope.capabilities.is_color_native``.

    Raises:
        NotImplementedError: Phase 2 activation pending.
    """
    if not is_color_native:
        # Mono pipeline -- callers should use tifffile.imread directly.
        # The stub stays explicit until 1d migrates the production callers.
        raise NotImplementedError(
            'imread_color is only meaningful with a color-native camera. '
            'Use tifffile.imread for the mono path.'
        )
    raise NotImplementedError('Phase 2 activation pending')


def imwrite_color(path, data, *, is_color_native: bool = False, color: str | None = None) -> None:
    """Color-camera-aware image write. Phase 2 activation pending.

    Stub helper for the color-native save path. Mono fluorescence saves
    go through ``write_tiff`` with layer metadata; color-camera frames
    will route through this wrapper to keep BGR / RGB / channel-order
    decisions in one place.

    Args:
        path: Output TIFF / PNG path.
        data: Image array.
        is_color_native: Camera-capability flag.
        color: Optional channel name for fluorescence false-color (unused
            on the color-native path; carried so Phase 1f can collapse
            mono + color-native call sites to one helper).

    Raises:
        NotImplementedError: Phase 2 activation pending.
    """
    if not is_color_native:
        raise NotImplementedError(
            'imwrite_color is only meaningful with a color-native camera. '
            'Use write_tiff for the mono fluorescence + layer-metadata path.'
        )
    raise NotImplementedError('Phase 2 activation pending')


def videowriter_color(*, is_color_native: bool = False, **kwargs):
    """Color-camera-aware VideoWriter constructor. Phase 2 activation pending.

    Stub factory that returns a VideoWriter configured for the camera's
    native shape. Mono path will continue through the existing
    ``modules.video_writer.VideoWriter`` class; color-native cameras
    will get a Bayer-decoded path that skips ``add_false_color``.

    Args:
        is_color_native: Camera-capability flag.
        **kwargs: Forward-compatible kwargs for the future implementation.

    Raises:
        NotImplementedError: Phase 2 activation pending.
    """
    if not is_color_native:
        raise NotImplementedError(
            'videowriter_color is only meaningful with a color-native camera. '
            'Use modules.video_writer.VideoWriter for the mono path.'
        )
    raise NotImplementedError('Phase 2 activation pending')


def add_false_color(
    array: np.ndarray, color: Color, output: np.ndarray | None = None
) -> np.ndarray:
    """Bake ``color`` into a mono frame at a save boundary; a colour frame,
    or a grey colour, is returned as it is.

    RGB ordering: index 0=Red, 1=Green, 2=Blue. Matches the canonical
    save-path convention shared with composite_builder. OpenCV consumers
    (cv2.VideoWriter) convert RGB->BGR at their own boundary.
    """
    if image_utils.is_color_image(array) or color == GREY:
        return array
    return mono_to_rgb_falsecolor(array, color, output=output)


def image_file_to_image(image_file):
    logger.info(f'[LVP image_utils  ] Loading: {image_file}')
    if not cv2.haveImageReader(image_file):
        logger.error('[LVP image_utils  ] - Image not supported by OpenCV')
        return

    num_images = cv2.imcount(image_file)
    logger.info(f'[LVP image_utils  ] - {num_images} images detected')

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error('[LVP image_utils  ] - Unable to load file')
        return

    return image


def get_used_color_planes(image) -> list:
    if not is_color_image(image=image):
        return []

    used_color_planes = []
    for color_plane_idx in range(image.shape[2]):
        image_view = image[:, :, color_plane_idx]
        if np.any(image_view):
            used_color_planes.append(color_plane_idx)

    return used_color_planes


def rgb_image_to_gray(image: np.ndarray, baked: bool = False) -> np.ndarray:
    """The mono an image holds: a mono image as it is; an image ``baked``
    in a channel's colour by its brightest plane; any other colour image by
    its one populated plane, else by luminance.

    A baked plane is the mono times its weight, and every stored colour has
    a full weight on its brightest plane (``layer_record.parse_display_color``),
    so that plane is the mono whatever the colour was: a magenta image
    reads back at full intensity, where luminance reads it about 58% dim,
    and a build needs no record of the colour a file was baked in.
    """

    def _is_grayscale(image):
        shape = image.shape
        return bool((len(shape) <= 2) or (shape[2] == 1))

    def _values_in_one_plane(image):
        used_color_planes = get_used_color_planes(image=image)

        return len(used_color_planes) <= 1

    if _is_grayscale(image=image):
        return image

    if baked or _values_in_one_plane(image=image):
        return np.amax(image, axis=2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preview_jpeg(
    image: np.ndarray,
    significant_bits: int,
    max_width: int | None,
    *,
    white_bits: int | None = None,
) -> bytes:
    """An image as it is shown, for display only: 8-bit grey JPEG, no wider than *max_width*.

    Half the image's width when *max_width* is None; never wider than the
    image. ``white_bits`` puts white at one frame's full scale for a sum,
    as ``convert_to_8bit`` does. A colour image is shown as the mean of its
    channels, which reads the same whichever order the channels came in.
    """
    if image.ndim == 3:
        image = image.mean(axis=2).astype(image.dtype)
    image = convert_to_8bit(image, significant_bits, white_bits=white_bits)
    height, width = image.shape
    target = max(1, width // 2) if max_width is None else max_width
    image = decimate_for_preview(image, (target, max(1, round(height * target / width))))
    return encode_image(image, fmt='jpeg')


def encode_image(image: np.ndarray, fmt: str = 'png', jpeg_quality: int = 80) -> bytes:
    """Encode a numpy image array to binary image data.

    Args:
        image: 2D (grayscale) or 3D (color) numpy array.
        fmt: Output format -- 'png', 'jpeg', or 'tiff'.
        jpeg_quality: JPEG quality (1-100), only used for JPEG format.

    Returns:
        bytes: Encoded image data.

    Raises:
        ValueError: If format is unsupported or encoding fails.
    """
    fmt = fmt.lower()
    if fmt in ('jpg', 'jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ext = '.jpg'
    elif fmt == 'png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # fast compression
        ext = '.png'
    elif fmt in ('tiff', 'tif'):
        params = []
        ext = '.tiff'
    else:
        raise ValueError(f'Unsupported image format: {fmt}')

    success, buf = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError(f'Failed to encode image as {fmt}')
    return buf.tobytes()


def encode_display_jpg(
    array: np.ndarray,
    color: Color,
    significant_bits: int,
    jpeg_quality: int = 90,
    *,
    white_bits: int | None = None,
) -> bytes:
    """Encode an image to JPEG bytes the way it appears on screen.

    JPEG is 8-bit and cannot carry the mono-pixels-plus-color-metadata
    form the TIFF path uses, so the channel's displayed color is baked
    into the pixels here: the source is reduced to 8 bits and the false
    color for the channel is rendered in. BF (and already-color sources)
    stay grayscale / color. This is the "save what I see" export path;
    TIFF / OME-TIFF remain the scientific, full-depth, metadata-bearing
    formats.

    Args:
        array: Source image (2D mono, 8/12/16-bit) for one channel.
        color: The colour the channel is shown in; ``GREY`` for grey.
        significant_bits: Payload depth of ``array`` so the 8-bit downconvert
            scales against the real range -- a summed frame is not indexed as
            12-bit (out of range) and a 10-bit frame is not crushed.
        jpeg_quality: JPEG quality, 1-100.
        white_bits: The depth whose full scale is white, below
            ``significant_bits`` for a sum: one frame's depth, so the JPG of a
            sum is as bright as it is on screen. None is ``significant_bits``.

    Returns:
        bytes: JPEG-encoded image.
    """
    img8 = convert_to_8bit(array, significant_bits, white_bits=white_bits)
    if img8.ndim == 3:
        # Already a display RGB image (e.g. a crosshairs / bullseye
        # overlay). These share the false-color RGB convention, so take
        # it as RGB rather than re-applying the layer color.
        rgb = img8
    else:
        # Bake the channel's displayed color at this encode boundary
        # (mono_to_rgb_falsecolor is the canonical mono -> RGB widening;
        # grey comes back grayscale RGB).
        rgb = mono_to_rgb_falsecolor(img8, color)
    # The RGB produced above is R=0,G=1,B=2; cv2 (encode_image) treats a
    # 3-channel array as BGR, so reverse the channel axis or the saved
    # JPG would have red and blue swapped.
    bgr = rgb[:, :, ::-1]
    return encode_image(bgr, fmt='jpeg', jpeg_quality=jpeg_quality)


def convert_12bit_to_8bit(image, out=None):
    """Downconvert a 12-bit-payload frame to 8-bit via the canonical converter."""
    return convert_to_8bit(image, 12, out=out)


@functools.cache
def _lut_to_8bit(significant_bits: int, white_bits: int) -> np.ndarray:
    """Build (once per depth pair) a payload-to-8-bit LUT sized to the value range.

    The table spans ``0 .. (1 << significant_bits) - 1`` so every legal payload
    value indexes in bounds, and ``(1 << white_bits) - 1`` maps to 255: full
    scale when the two are equal, one frame's white for a sum, whose values
    above it are white. Cached: the handful of depth pairs in use each build a
    single shared table.
    """
    max_value = (1 << significant_bits) - 1
    white_value = (1 << white_bits) - 1
    # Linear rescale (value / white * 255) then a truncating .astype(uint8),
    # chosen over the legacy >>8 (i.e. /256) used for 16-bit. Both truncate and
    # map full scale to 255; they differ by at most 1 LSB at 32640 of the 65536
    # 16-bit inputs because the divisor differs (65535 vs 65536), NOT because one
    # rounds. The rescale carries a systematic ~0.5-LSB low bias against the exact
    # real-valued map -- it is the deliberate choice, and the converter pin test
    # locks the <=1-LSB bound so a change is caught here. A white below full
    # scale puts the values above it past 255, and they render white.
    return np.clip(np.arange(max_value + 1, dtype=np.float64) / white_value * 255, 0, 255).astype(
        np.uint8
    )


def convert_to_8bit(
    image: np.ndarray,
    significant_bits: int,
    out: np.ndarray | None = None,
    *,
    white_bits: int | None = None,
) -> np.ndarray:
    """Downconvert a frame to 8-bit, scaling against its significant bits.

    ``significant_bits`` names the meaningful payload range -- 12 for a Mono12
    frame, 10 for four 8-bit frames summed -- so the LUT span follows the real
    depth: a value above it is a depth-contract violation and raises, and a
    10-bit full-white frame maps to 255 instead of being treated as 12-bit.
    ``white_bits`` puts white lower than full scale, which is how a sum is
    rendered: against one frame's white, brighter than one frame, white
    wherever it passes that (``convert_sum_to_8bit``). Already-8-bit frames
    pass through. ``out`` reuses a caller buffer to avoid a per-call
    allocation on the preview path.
    """
    if image.dtype == np.uint8:
        return image
    significant_bits = int(significant_bits)
    white_bits = significant_bits if white_bits is None else int(white_bits)
    if white_bits > significant_bits:
        raise ValueError(
            f'white at {white_bits} bits lies above the {significant_bits}-bit range it renders'
        )
    lut = _lut_to_8bit(significant_bits, white_bits)
    # The LUT has exactly one entry per in-range value, so a payload above the
    # declared depth indexes it out of range and raises. Re-raise that as a typed
    # FrameDepthError so a depth-contract violation is loud and named -- without a
    # per-frame full-frame image.max() scan on the live-preview path. The peak is
    # read only on the (rare) failure. If the downconvert is ever changed to a
    # scale-and-clip that would NOT raise on an out-of-range value, the depth-guard
    # tests go red and this enforcement must be restored eagerly.
    try:
        if out is not None and out.shape == image.shape and out.dtype == np.uint8:
            np.take(lut, image, out=out)
            return out
        return lut[image]
    except IndexError:
        raise FrameDepthError(int(image.max()), significant_bits) from None


def convert_sum_to_8bit(
    image: np.ndarray, frames_summed: int, frame_bits: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Render a captured frame to 8 bits the way it is shown: a sum brighter.

    Summing is for a brighter image, so a sum of N frames is rendered against
    one frame's white: N times brighter than one frame, and white wherever it
    passes one frame's white. Its full range stays in the frame (and in a
    full-depth file); only the rendering saturates. A single frame renders
    against its own depth, as ``convert_to_8bit`` does.

    Args:
        image: The captured frame, a sum or a single frame.
        frames_summed: How many frames it holds.
        frame_bits: The depth of each frame, as delivered.
        out: Optional caller buffer, as for ``convert_to_8bit``.
    """
    return convert_to_8bit(
        image,
        summed_significant_bits(frames_summed, frame_bits),
        out=out,
        white_bits=frame_bits,
    )


def convert_16bit_to_8bit(image):
    """Downconvert a 16-bit-container frame to 8-bit via the canonical converter."""
    return convert_to_8bit(image, 16)


def get_tiff_colormap(color: Color, dtype: np.dtype | type | str) -> np.ndarray:
    """Build a TIFF colormap array for PALETTE photometric (8-bit only).

    Returns a (3, 256) uint16 array suitable for tifffile's ``colormap``
    parameter. Only used for 8-bit false-color images -- Windows Preview
    supports PALETTE with uint8 but NOT with uint16.

    The TIFF ColorMap tag (320) is a uint16 field whose full scale is 65535,
    so each ramp entry is the palette index scaled by 257 (65535 // 255). A
    0-255 ramp stored in the tag renders at ~0.4% brightness in compliant
    readers (ImageJ's native reader shows black); only Bio-Formats' lenient
    rescaling of narrow colormaps masks it.
    """
    if dtype not in ('uint8', np.uint8):
        raise NotImplementedError(f'TIFF colormap only supported for uint8, got {dtype}')

    ramp = np.arange(0, 256, 1, dtype=np.float64) * 257
    return np.stack([np.rint(ramp * weight) for weight in color]).astype(np.uint16)


def _color_bytes(color: Color) -> tuple[int, int, int]:
    """The 0-255 red, green and blue of ``color``."""
    r, g, b = (round(255 * weight) for weight in color)
    return r, g, b


def _ome_rgba(color: Color) -> int:
    """The signed-32-bit OME Channel.Color value of ``color``.

    OME-XML encodes Channel.Color as ``(R << 24) | (G << 16) | (B << 8) | A``
    with the unsigned 32-bit result reinterpreted as Python signed int32.
    Consumed by Bio-Formats when the user opens the hyperstack with
    Color mode Composite/Colorized; the default File>Open ignores it.
    """
    r, g, b = _color_bytes(color)
    unsigned = (r << 24) | (g << 16) | (b << 8) | 0xFF
    return unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned


def get_imagej_lut(color: Color) -> np.ndarray:
    """Build an ImageJ-style LUT: (3, 256) uint8 array.

    Stored in ImageJ metadata (not TIFF tag 320), so Windows Preview sees
    plain MINISBLACK grayscale while ImageJ auto-applies the color LUT.
    Used for 16-bit images where PALETTE photometric breaks Windows Preview.
    """
    ramp = np.arange(256, dtype=np.float64)
    return np.stack([np.rint(ramp * weight) for weight in color]).astype(np.uint8)


def maybe_apply_false_color(
    data: np.ndarray,
    color: Color,
    use_false_color_16bit: bool,
    output_buf: np.ndarray | None = None,
) -> np.ndarray:
    """Widen single-channel 16-bit fluorescence to 3-channel RGB when the
    false-color setting is on; pass other inputs through unchanged.

    Scope is 16-bit ONLY. 8-bit fluorescence already saves as 2D mono with a
    PALETTE colormap, which renders in color in both Windows Preview and FIJI
    at 1x file size, so it needs no widening. 16-bit cannot use PALETTE
    (Windows Preview rejects uint16 palette), so default OFF it saves mono +
    an ImageJ LUT -- color in FIJI, grayscale in Windows Preview. Turning the
    setting ON bakes the layer false color into 3-channel RGB so 12/16-bit
    fluorescence also renders in color in Windows Preview / Explorer, at ~3x
    the file size.

    use_false_color_16bit is the resolved decision (the caller passed write_tiff
    a save_encoding; widening is save_encoding == 'rgb'). Already-RGB inputs,
    8-bit inputs and a grey colour always pass through.
    """
    if data.dtype == np.uint16 and use_false_color_16bit:
        return add_false_color(data, color, output=output_buf)
    return data


@contextlib.contextmanager
def _written_whole(file_loc: pathlib.Path | str) -> Iterator[pathlib.Path]:
    """Yield a name beside ``file_loc`` to write to, moved onto it once whole.

    A writer opened on the target itself leaves a truncated file under the
    real name when the write stops part-way (a full disk, a crash), and a
    later build reads it as an unreadable frame. A write that raises leaves
    nothing at the target and no temporary behind; a crash leaves only the
    temporary. A '.tmp' name is one no image search picks up.
    """
    file_loc = pathlib.Path(file_loc)
    tmp_loc = file_loc.with_name(file_loc.name + '.tmp')
    try:
        yield tmp_loc
        os.replace(tmp_loc, file_loc)
    finally:
        tmp_loc.unlink(missing_ok=True)


def write_hyperstack_tiff(
    planes: Iterator[np.ndarray],
    file_loc: pathlib.Path,
    shape: tuple,
    dtype: np.dtype,
    hyperstack_metadata: dict,
    hyperstack_options: dict,
    hyperstack_resolution: tuple | None = None,
) -> None:
    """Stream a 5D TZCYX hyperstack to disk with caller-prepared OME metadata.

    ``planes`` is an iterator of 2D YX planes in C-order of ``shape``
    (T-major, then Z, then C); ``shape`` and ``dtype`` describe the full
    stack, which tifffile needs up front to lay the file out. The stack
    never materializes in RAM -- the memory bound is one plane regardless
    of stack size, which is what lets a 600-frame video well build on any
    field machine.

    ``hyperstack_options`` is required and must carry ``photometric``:
    without it, tifffile's shape heuristics can read a <=4-wide trailing
    dimension of a declared-shape write as samples and reject the OME axes.

    Separate from write_tiff because this path shares none of the
    per-image logic: maybe_apply_false_color expects 2D mono input,
    generate_tiff_data builds per-image metadata, and the container here
    is OME-only -- the ImageJ container cannot represent BigTIFF and
    tifffile suppresses its imagej flag under ome=True anyway, so channel
    color travels solely as OME Channel.Color (FIJI applies it via the
    Bio-Formats Importer in Composite mode; plain File>Open shows default
    LUTs). The caller (stack_builder) supplies the full OME dict + write
    options + resolution and they pass through verbatim. Keeping these
    apart lets write_tiff demand significant_bits as a required argument,
    which this path carries inside its OME dict rather than as a scalar.

    tifffile's OME serializer drops the dict's ``Instrument``, so it is
    spliced into the description once the file is written and before it
    takes its name (``_splice_ome_instrument``).
    """
    total_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    use_bigtiff = total_bytes > 3.8 * 1024 * 1024 * 1024
    if 'photometric' not in hyperstack_options:
        raise ValueError(
            'write_hyperstack_tiff requires photometric in hyperstack_options; '
            'without it a <=4-wide trailing dimension can be misread as samples'
        )
    tiff_metadata = {
        key: value for key, value in hyperstack_metadata.items() if key != 'Instrument'
    }
    # The planes stream from the caller, which can refuse one mid-write.
    with _written_whole(file_loc) as tmp_loc:
        with tf.TiffWriter(str(tmp_loc), ome=True, bigtiff=use_bigtiff) as tif:
            tif.write(
                planes,
                shape=shape,
                dtype=np.dtype(dtype),
                resolution=hyperstack_resolution,
                metadata=tiff_metadata,
                software=f'LumaViewPro {version}',
                # The hyperstack's time is its build's.
                datetime=datetime.datetime.now().strftime(_TIFF_DATETIME),
                **hyperstack_options,
            )
        _splice_ome_instrument(tmp_loc, hyperstack_metadata.get('Instrument'))


def _msb_align_to_container(data: np.ndarray, significant_bits: int) -> tuple[np.ndarray, int]:
    """Left-justify a right-aligned payload to fill its container width.

    A right-aligned N-bit payload (0..2^N-1 in a wider container) reads dark in
    viewers that ignore the significant-bits tag. Shifting it up by
    (container_bits - N) makes it fill the container so those viewers render it
    bright; the shift is lossless (the top bits were zero) and the stored values
    then make no narrower-than-container significant-bits claim, so container-
    width read-back is the correct scale. Returns the shifted array and the new
    significant_bits; a payload already at container width is returned unchanged.
    """
    container_bits = data.itemsize * 8
    shift = container_bits - significant_bits
    if shift > 0:
        data = data << shift
        significant_bits = container_bits
    return data, significant_bits


def encoding_fills_container_dtype(save_encoding: str, dtype: np.dtype, is_color: bool) -> bool:
    """Whether this encoding + payload left-justifies to fill its container.

    The one gate write_tiff and depth-reporting callers share, so the rule
    lives in exactly one place: a container-filling encoding of a narrow uint16
    mono payload gets shifted up; a color frame or an already-8-bit payload does
    not.
    """
    return (
        image_mode.encoding_fills_container(save_encoding) and dtype == np.uint16 and not is_color
    )


def written_significant_bits(
    save_encoding: str, significant_bits: int, dtype: np.dtype, is_color: bool
) -> int:
    """The significant-bit depth a write actually stamps on the file.

    A container-filling encoding (scaled / false-color) left-justifies a narrow
    uint16 mono payload to fill its container, so the file is stamped at the
    container width, not the acquired depth -- a 12-bit capture saved scaled is
    a 16-bit file. Every other case keeps the acquired depth. Callers report the
    on-disk depth through this instead of the pre-encode value, which describes
    the sensor, not the file. Mirrors the shift write_tiff applies.
    """
    if encoding_fills_container_dtype(save_encoding, dtype, is_color):
        return dtype.itemsize * 8
    return significant_bits


def write_tiff(
    data: np.ndarray,
    file_loc: pathlib.Path,
    metadata: dict,
    ome: bool,
    color: Color,
    significant_bits: int,
    save_encoding: str,
    white_bits: int | None,
    extratags: list | None = None,
    false_color_buf: np.ndarray | None = None,
    rgb_buf: np.ndarray | None = None,
) -> None:
    # The white travels with the depth, for the same reason: a sum is tagged at
    # the bits it can reach and shown against one frame's white, and a file
    # that drops the white reopens about N times dimmer than it was taken.
    # Required, None for a file whose values were remapped (a composite, an
    # enhanced image) and so are shown at their depth.
    if white_bits is not None and white_bits > significant_bits:
        raise ValueError(
            f'white at {white_bits} bits lies above the {significant_bits}-bit payload it describes'
        )

    # Depth travels with the pixels. A uint16 frame stored right-aligned
    # (0..4095 for a 12-bit sensor) is bit-identical to a dark 16-bit image, so
    # a write that does not state its significant-bit depth cannot label the
    # file correctly -- it silently claims full container width and every
    # narrow payload reads back ~16x dark. significant_bits is a required
    # argument so a depth-less write cannot be expressed.
    metadata = {**metadata, 'significant_bits': significant_bits}

    if extratags is None:
        extratags = []

    # save_encoding is the single consolidated image_mode output and the only
    # thing that decides the on-disk shape: 'rgb' widens to false color,
    # 'msb_aligned' left-justifies a narrow payload, right_aligned/8bit store
    # as-is. RGB widening is derived from it alone, so the same fact is never
    # carried by a second out-of-band flag that could disagree with it.
    # Brighten BEFORE colorizing so the scaled mono mode and the false-color RGB
    # mode share one container-fill step and a false-color frame inherits it.
    # Colorizing a still-narrow (right-aligned) payload would bake dark color
    # that no plain viewer can show; filling the mono payload first makes the
    # false color render bright. right_aligned/8bit keep their narrow payload --
    # the depth tag, not a shift, carries their scale.
    if encoding_fills_container_dtype(save_encoding, data.dtype, is_color_image(data)):
        data, sig = _msb_align_to_container(data, metadata['significant_bits'])
        # The white moves with the pixels it describes.
        if white_bits is not None:
            white_bits += sig - metadata['significant_bits']
        metadata = {**metadata, 'significant_bits': sig}

    use_false_color_16bit = save_encoding == image_mode.SAVE_ENCODING_RGB

    data = maybe_apply_false_color(
        data=data,
        color=color,
        use_false_color_16bit=use_false_color_16bit,
        output_buf=false_color_buf,
    )

    kwargs = {}
    # Enable BigTIFF for datasets >3.8 GB to prevent silent corruption at 4 GB boundary
    data_size_bytes = data.nbytes
    use_bigtiff = data_size_bytes > 3.8 * 1024 * 1024 * 1024
    if ome:
        kwargs = {
            'ome': True,
            'bigtiff': use_bigtiff,
        }
    elif is_color_image(data):
        # For now, prevent 16-bit color images from being converted to ImageJ type
        # such as composite (or bullseye). Could allow this once proper support is added.
        pass
    elif data.dtype == np.uint16:
        kwargs['imagej'] = True

    def _validate_type() -> str:
        type_count = 0
        image_type = None

        if ome:
            type_count += 1
            image_type = 'ome'

        if kwargs.get('imagej', False):
            type_count += 1
            image_type = 'imagej'

        if type_count > 1:
            raise ValueError('Tiff must only be one type at most (OME or ImageJ)')

        return image_type

    image_type = _validate_type()

    support_data = generate_tiff_data(
        data=data, metadata=metadata, image_type=image_type, color=color, white_bits=white_bits
    )
    tiff_datetime = metadata['datetime'].strftime(_TIFF_DATETIME)

    with _written_whole(file_loc) as tmp_loc:
        with tf.TiffWriter(str(tmp_loc), **kwargs) as tif:
            if (image_type is None) and is_color_image(image=data):
                # Handles case where an actual color image is provided (such as the bullseye in engineering mode)
                tif.write(
                    data,
                    resolution=support_data['resolution'],
                    metadata=support_data['metadata'],
                    datetime=tiff_datetime,
                    software=f'LumaViewPro {version}',
                    extratags=support_data['extratags'],
                    **support_data['options'],
                )

            elif (image_type == 'ome') and is_color_image(image=data):
                tif.write(
                    data,
                    resolution=support_data['resolution'],
                    metadata=support_data['metadata'],
                    datetime=tiff_datetime,
                    software=f'LumaViewPro {version}',
                    extratags=support_data['extratags'],
                    **support_data['options'],
                )

            else:
                # 8-bit fluorescence: PALETTE photometric with colormap -- gives
                # false color in both Windows Preview and ImageJ.
                # 16-bit fluorescence: MINISBLACK photometric -- Windows Preview
                # compatible (shows grayscale). Color via ImageJ LUT metadata
                # (imagej type) or OME Channel metadata (ome type).
                # Grey: always MINISBLACK, no colormap needed.
                colormap_array = None
                if data.dtype == np.uint8 and color != GREY:
                    colormap_array = get_tiff_colormap(color=color, dtype=data.dtype)

                tif.write(
                    data,
                    resolution=support_data['resolution'],
                    metadata=support_data['metadata'],
                    datetime=tiff_datetime,
                    software=f'LumaViewPro {version}',
                    colormap=colormap_array,
                    extratags=support_data['extratags'],
                    **support_data['options'],
                )
        if image_type == 'ome':
            _splice_ome_instrument(tmp_loc, support_data['instrument'])


def generate_tiff_data(
    data: np.ndarray,
    metadata: dict,
    image_type: str,
    color: Color,
    *,
    white_bits: int | None,
) -> dict:

    axes = 'YX'

    if is_color_image(data):
        photometric = tf.PHOTOMETRIC.RGB
        axes = 'YXS'  # 3rd dimension is samples (RGB channels)
    elif color != GREY and data.dtype == np.uint8:
        # 8-bit: PALETTE with colormap -- works in Windows Preview and ImageJ.
        photometric = tf.PHOTOMETRIC.PALETTE
    else:
        # Grey, and 16-bit: MINISBLACK -- Windows Preview can't handle PALETTE
        # with uint16. A 16-bit colour is provided via ImageJ LUT metadata
        # (ImageJ type) or OME Channel metadata (OME type).
        photometric = tf.PHOTOMETRIC.MINISBLACK

    # Read once, above every arm: the scale claim is one rule, and the arm that
    # returned before this read is exactly the arm that emitted a wrong claim.
    # A hard key, not a .get default -- a metadata dict with no scale must fail
    # loudly here rather than silently resolve to a fabricated size.
    pixel_size_um = metadata['pixel_size_um']

    # The micron token is container-dependent: OME-XML (UTF-8) requires
    # the schema's micro-sign form, while tifffile encodes the ImageJ /
    # shaped descriptions as ASCII, where the micro sign crashes the
    # write -- and ImageJ's own convention is 'um'.
    micron = OME_UNIT_MICROMETER if image_type == 'ome' else 'um'

    plane = _plane_block(metadata, micron)
    # No source is assumed for a time: a defaulted one would name the wrong
    # clock for every producer that forgot to state its own.
    if 'timestamp_iso' in metadata and 'timestamp_source' not in metadata:
        raise KeyError('timestamp_source')

    # Spectral identity rides in the Channel block. ExcitationWavelength
    # is a schema OME Channel attribute, so the OME container serializes
    # it into real OME-XML; DisplayName is not schema, so the OME
    # container carries it in the LVP annotation.
    channel_block: dict = {'Name': [metadata['channel']]}
    if 'channel_display' in metadata and image_type != 'ome':
        channel_block['DisplayName'] = [metadata['channel_display']]
    if 'excitation_nm' in metadata:
        channel_block['ExcitationWavelength'] = [metadata['excitation_nm']]
        channel_block['ExcitationWavelengthUnit'] = ['nm']

    # Base metadata shared by all structured types
    tiff_metadata = {
        'axes': axes,
        'Channel': channel_block,
        'Plane': plane,
    }
    blocks = _instrument_and_plate_blocks(metadata)
    # The OME container keeps only the schema's attributes, so every other
    # fact of the record rides in the image's LVP MapAnnotation: without it a
    # summed frame states its per-frame exposure and no count, a dark or
    # unsettled frame reads as a good one, and the file loses its scope and
    # plate. Set in this arm only: tifffile would also serialize the key into
    # the ImageJ and shaped descriptions.
    if image_type == 'ome':
        # The schema's own statement of the payload depth, rendered from the
        # value private tag 65123 carries in every container.
        tiff_metadata['SignificantBits'] = metadata['significant_bits']
        tiff_metadata['Plane'] = {
            key: value for key, value in plane.items() if key in _OME_PLANE_ATTRIBUTES
        }
        tiff_metadata['MapAnnotation'] = _ome_annotation(
            metadata,
            plane_rendered=_OME_PLANE_ATTRIBUTES,
            record_rendered=_OME_STILL_RENDERED,
        )
    else:
        # The filterset is a unit-level fact, not per-channel; it rides as
        # its own structured key.
        if 'filterset' in metadata:
            tiff_metadata['FilterSet'] = metadata['filterset']
        # Producing algorithm, when the caller supplied one (post-processing
        # outputs), so a consumer can tell a degraded edge-to-edge montage
        # from a registered stitch off the file alone.
        if metadata.get('algorithm'):
            tiff_metadata['Algorithm'] = metadata['algorithm']
        tiff_metadata |= {key: blocks[key] for key in ('Instrument', 'Plate') if key in blocks}

    # A scale is written only when one is known. Emitting the keys with a null
    # value would still read downstream as a PhysicalSize claim, and a reader
    # cannot tell an invented scale from a measured one -- the absence is the
    # honest signal that this file cannot be measured. The unit comes from
    # the container-dependent micron token resolved with the Plane units
    # above (OME micro-sign form vs ASCII 'um' for the ImageJ container).
    if pixel_size_um is not None:
        tiff_metadata['PhysicalSizeX'] = pixel_size_um
        tiff_metadata['PhysicalSizeXUnit'] = micron
        tiff_metadata['PhysicalSizeY'] = pixel_size_um
        tiff_metadata['PhysicalSizeYUnit'] = micron

    # ImageJ adds unit, display range, channel modality and LUT
    if image_type == 'imagej':
        tiff_metadata['unit'] = 'um'
        # The display range Fiji opens the file at: the white LVP shows it
        # against. A file that states none is stretched by Fiji to its own
        # darkest and brightest pixel, which LVP never does, so a dim frame
        # opened there looked brighter than it was taken.
        tiff_metadata['min'] = 0
        tiff_metadata['max'] = (
            1 << (metadata['significant_bits'] if white_bits is None else white_bits)
        ) - 1
        # Modality states WHAT was imaged, so it is derived from the acquiring
        # channel -- the same value Channel.Name carries. Deriving it from the
        # render color instead made a false-color-off fluorescence frame record
        # itself as brightfield, contradicting its own Channel.Name in the same
        # file and tracking a viewer checkbox into permanent metadata. A
        # stored old name is its stock key's kind.
        identity_channel = metadata['channel']
        identity_key = stored_key(identity_channel)
        if identity_key in common_utils.get_transmitted_layers():
            modality = identity_channel
        elif identity_key in common_utils.get_image_layers():
            modality = 'MIF'
        elif identity_channel == 'Unknown':
            # A derived output whose source channel could not be determined
            # says so here too, rather than borrowing a modality it cannot
            # support. Only post-processing produces this; a capture always
            # knows the channel it was acquired on.
            modality = 'Unknown'
        else:
            raise ValueError(f'Unexpected channel ({identity_channel}) for tiff modality')
        tiff_metadata['Channel']['Modality'] = [modality]
        # Embed color LUT in ImageJ metadata (not TIFF tag 320).
        # Windows Preview ignores ImageJ metadata -> sees MINISBLACK -> works.
        # ImageJ reads its own metadata -> auto-applies color LUT -> shows color.
        # mode='color' tells FIJI to apply the LUT (otherwise defaults to grayscale).
        tiff_metadata['LUTs'] = [get_imagej_lut(color)]
        if color != GREY:
            tiff_metadata['mode'] = 'color'
        # imagej path mirrors the default/ome maxworkers=0 below for the
        # same Windows kernel-handle-leak reason; this path triggers on
        # 16-bit fluorescence + non-color + image_layer, which the
        # bench-witnessed 8-bit Bug E soak did not exercise. Adjacent
        # symmetry: same tifffile.write() ThreadPoolExecutor pattern,
        # same leak risk; deflate single-threaded cost is negligible.
        options = {
            'photometric': photometric,
            'compression': 'deflate',
            'maxworkers': 0,
        }
        # Resolution for ImageJ types is in pixels/pixel
        resolution = (
            resolution_for_pixel_size(pixel_size_um, per_centimeter=False)
            if pixel_size_um is not None
            else None
        )
    else:
        # ome and default use same options. maxworkers=0 disables tifffile's
        # per-write ThreadPoolExecutor; the executor's internal queue holds
        # a Windows kernel Event handle that intermittently outlives cleanup,
        # giving ~1 leaked handle per save (instrumentation confirmed via
        # lib/handle_trace.py over a 28-min bench run: mean +0.967/call).
        # LZW compression now runs single-threaded -- +~10ms per 5MP save,
        # negligible vs typical 1-2 saves/sec protocol cadence.
        options = {
            'photometric': photometric,
            'compression': 'lzw',
            # tifffile always emits an XResolution tag, defaulting to 1/1. Under
            # CENTIMETER that reads as one pixel per centimetre -- a concrete and
            # wildly wrong scale claim. NONE is the TIFF convention for "ratio
            # only, no absolute unit", which is what an unknown scale means.
            'resolutionunit': 'CENTIMETER' if pixel_size_um is not None else 'NONE',
            'maxworkers': 0,
        }
        resolution = resolution_for_pixel_size(pixel_size_um) if pixel_size_um is not None else None

    # Carry the payload depth in a durable private TIFF tag so plain / ImageJ
    # outputs (which have no OME-XML) recover it on read-back; OME files get it
    # too, harmlessly, and the reader prefers their OME-XML value. SHORT (type
    # 3), one value, written to the first page only.
    return {
        'metadata': tiff_metadata,
        'extratags': _depth_extratags(metadata['significant_bits'], white_bits),
        'options': options,
        'resolution': resolution,
        # Spliced into an OME file's description once it is written.
        'instrument': blocks.get('Instrument'),
    }


# The facts an OME still or a hyperstack states in a schema attribute or a
# TIFF tag, whatever their Plane: the annotation carries every other fact its
# record states. The scale is a hard key, None when the file states none.
_OME_STILL_RENDERED = frozenset(
    {'pixel_size_um', 'channel', 'excitation_nm', 'significant_bits', 'datetime'}
)
_OME_HYPERSTACK_RENDERED = frozenset({'pixel_size_um', 'channel', 'significant_bits', 'datetime'})

# The record keys the Plane block renders: its facts, and the position pair.
_IN_THE_PLANE_BLOCK = frozenset(_PLANE_FACTS) | {'plate_pos_mm'}


def _plane_block(record: dict, micron: str) -> dict:
    """The Plane block a record renders to, every fact under its Plane name.

    Every fact of the record that lives there (FILE_FACTS), each with its
    unit beside it. Every one is optional: a producer omits the key when the
    value was unknown (a failed camera read, an LED that was off or on an
    absent board, an axis that had lost its reference, a scope with no X and
    Y), because a stand-in written here would be read downstream as the
    instrument's own report. A unit standing alone would declare a
    measurement the file does not carry, so it is written only with its value.
    """
    plane: dict = {}
    if 'plate_pos_mm' in record:
        plane['PositionX'] = record['plate_pos_mm']['x']
        plane['PositionY'] = record['plate_pos_mm']['y']
        plane['PositionXUnit'] = 'mm'
        plane['PositionYUnit'] = 'mm'
    for key, fact in _PLANE_FACTS.items():
        if key in record:
            plane[fact.plane] = record[key]
            if fact.unit is not None:
                plane[f'{fact.plane}Unit'] = micron if fact.unit == 'um' else fact.unit
    return plane


def _ome_annotation(
    record: dict, *, plane_rendered: frozenset[str], record_rendered: frozenset[str]
) -> dict:
    """An OME file's LVP MapAnnotation: every fact of its record the write
    put in no schema attribute.

    The writer that renders a schema attribute names what it rendered, so a
    fact is written once and never dropped: a Plane fact under its Plane name
    (the name files have always carried it under), any other fact under its
    record key. Each value is JSON, so a count, a flag, the objective, the
    instrument and the frame-validity list read back as written.
    """
    entries = {
        key: value
        for key, value in _plane_block(record, OME_UNIT_MICROMETER).items()
        if key not in plane_rendered
    }
    entries |= {
        key: record[key]
        for key in FILE_FACTS
        if key in record and key not in _IN_THE_PLANE_BLOCK and key not in record_rendered
    }
    return {
        'Namespace': _LVP_OME_ANNOTATION_NAMESPACE,
        'Value': {
            key: json.dumps(value, ensure_ascii=False, default=_numpy_scalar_value)
            for key, value in entries.items()
        },
    }


def _ome_qualified(tag: str) -> str:
    return f'{{{_OME_NAMESPACE}}}{tag}'


def _splice_ome_instrument(path: pathlib.Path, instrument: dict | None) -> None:
    """Splice a written OME file's ``Instrument`` into its description.

    tifffile's OME writer drops the element, so it is inserted into the
    description tifffile wrote, which is written last in the file and
    extends in place: the Instrument before the first Image, as the schema
    orders them, and each Image's InstrumentRef, then its ObjectiveSettings
    when the Instrument holds an Objective, immediately before its Pixels. A
    reference to an absent part breaks the schema's keyref, so the
    Instrument is written only when one of its parts states something, and
    each reference only with what it names. ``instrument`` is the renderer's
    block (``_instrument_and_plate_blocks``), keyed by the schema's names.
    The description goes back as UTF-8 bytes: it holds the micro sign, and
    tiffcomment refuses a non-ASCII str.
    """
    if not instrument:
        return
    root = ET.fromstring(tf.tiffcomment(path))
    element = ET.Element(_ome_qualified('Instrument'), ID='Instrument:0')
    # The schema's order within an Instrument.
    for part, part_id in (
        ('Microscope', None),
        ('Detector', 'Detector:0:0'),
        ('Objective', 'Objective:0:0'),
    ):
        if part in instrument:
            attributes = {key: str(value) for key, value in instrument[part].items()}
            if part_id is not None:
                attributes = {'ID': part_id} | attributes
            ET.SubElement(element, _ome_qualified(part), attributes)
    images = root.findall(_ome_qualified('Image'))
    root.insert(list(root).index(images[0]), element)
    for image in images:
        at = list(image).index(image.find(_ome_qualified('Pixels')))
        references = [ET.Element(_ome_qualified('InstrumentRef'), ID='Instrument:0')]
        if 'Objective' in instrument:
            references.append(ET.Element(_ome_qualified('ObjectiveSettings'), ID='Objective:0:0'))
        for offset, reference in enumerate(references):
            image.insert(at + offset, reference)
    description = io.BytesIO()
    ET.ElementTree(root).write(description, encoding='utf-8', xml_declaration=True)
    tf.tiffcomment(path, comment=description.getvalue())


def _numpy_scalar_value(value):
    """A numpy scalar as the Python number it holds, for the JSON of a saved annotation.

    A producer's value can stay a numpy scalar (a stage position left
    np.float64), and the JSON encoder refuses np.int64 and np.bool_, which
    would fail a good save over the type of a number.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'a saved file cannot state a {type(value).__name__}')


def _depth_extratags(significant_bits: int, white_bits: int | None) -> list[tuple]:
    """The private tags stating a file's depth and, when it states one, its white.

    SHORT (type 3), one value each, on the first page only.
    """
    tags = [(_TIFF_TAG_SIGNIFICANT_BITS, 3, 1, int(significant_bits), True)]
    if white_bits is not None:
        tags.append((_TIFF_TAG_WHITE_BITS, 3, 1, int(white_bits), True))
    return tags


def ms_exposure_to_rational(ms_exposure):
    exposure_seconds = ms_exposure / 1000
    fraction = Fraction(exposure_seconds).limit_denominator(1_000_000)
    # Metadata uses rational number of seconds
    return fraction.numerator, fraction.denominator


def subject_dist_to_rational(distance):
    distance_meters = distance / 1_000_000  # Convert um to m
    fraction = Fraction(distance_meters).limit_denominator(1_000_000)
    return fraction.numerator, fraction.denominator


# A TIFF resolution tag is an unsigned RATIONAL (two uint32), but Bioformats
# reads the numerator back as a signed int32 -- a numerator above 2^31 surfaces
# as a negative PhysicalSize for a high-magnification (small) pixel size.
# Choosing the denominator here, instead of handing tifffile a float to
# rationalize, keeps the numerator within int32 by construction.
_TIFF_RATIONAL_NUMERATOR_MAX = 2**31 - 1


def _int32_safe_rational(value: float) -> tuple[int, int]:
    """Approximate a positive value as a (numerator, denominator) pair whose
    numerator stays within signed int32 range.
    """
    max_denominator = max(1, int(_TIFF_RATIONAL_NUMERATOR_MAX // value))
    fraction = Fraction(value).limit_denominator(min(1_000_000, max_denominator))
    return fraction.numerator, fraction.denominator


def resolution_for_pixel_size(pixel_size_um: float, *, per_centimeter: bool = True) -> tuple:
    """Build the (X, Y) TIFF resolution tag for a given pixel size.

    Each axis is returned as an int32-safe RATIONAL so a Bioformats reader
    cannot interpret a high-magnification pixel size as a negative PhysicalSize.

    Args:
        pixel_size_um: Image pixel size in microns (must be > 0).
        per_centimeter: True for the CENTIMETER resolutionunit (pixels per cm,
            1 cm = 1e4 um); False for the ImageJ pixels-per-pixel convention.

    Returns:
        ((x_num, x_den), (y_num, y_den)). X and Y match for square pixels but are
        returned per-axis to satisfy tifffile's resolution= contract.
    """
    pixels_per_unit = (1e4 if per_centimeter else 1.0) / pixel_size_um
    axis = _int32_safe_rational(pixels_per_unit)
    return (axis, axis)


_scale_bar_cache = {}


def _compute_scale_bar_overlay(
    height, width, dtype, is_color, objective, binning_size, color, significant_bits, pixel_size_um
):
    """Pre-render scale bar overlay and mask. Returns (overlay, mask, cache_key)."""
    # Scale bar should be 1/8 to 1/4 the image length
    min_px = int(width / 8)
    max_px = int(width / 4)
    mid_px = (min_px + max_px) // 2

    mid_um = mid_px * pixel_size_um
    min_um = min_px * pixel_size_um
    max_um = max_px * pixel_size_um

    good_numbers = np.array(
        [
            25,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            600,
            700,
            800,
            900,
            1000,
            1250,
            1500,
            1750,
            2000,
            2500,
            3000,
        ],
        dtype=float,
    )

    if min_um > good_numbers.max():
        while min_um > good_numbers.max():
            good_numbers *= 10
    elif max_um < good_numbers.min():
        while max_um < good_numbers.min():
            good_numbers = good_numbers / 10

    good_numbers_index = np.absolute(good_numbers - mid_um).argmin()
    scale_bar_length_um = good_numbers[good_numbers_index]
    scale_bar_length_pixels = int(scale_bar_length_um / pixel_size_um)

    scale_bar_thickness_pixels = min(3, int(height / 300))
    scale_bar_bottom_offset = int(height / 40)
    scale_bar_right_offset = int(width / 40)

    # Bright-background modes (BF, PC) need a black bar to stay visible.
    # Darkfield is transmitted-light but shows bright subjects on a dark
    # field, so it takes a white bar like the fluorescence channels.
    if color in common_utils.get_bright_background_layers():
        scale_bar_value = 0
    elif dtype == np.uint8:
        scale_bar_value = 255
    else:
        # White bar = the payload max for this frame's depth, so it downconverts
        # to full 8-bit white. A sum is tagged with the bits it can reach (four
        # 12-bit frames: 14 -> 16383); a single 12-bit frame is 4095. A fixed
        # 4095 would render a summed-frame bar gray in its full-depth file.
        scale_bar_value = (1 << significant_bits) - 1

    x_end = width - scale_bar_right_offset
    x_start = x_end - scale_bar_length_pixels
    y_start = scale_bar_bottom_offset
    y_end = y_start + scale_bar_thickness_pixels

    # Render the bar+text geometry with a NONZERO sentinel, not the bar's real
    # value. The mask below is built from nonzero canvas pixels; a black bar
    # (value 0) drawn onto a zeroed canvas would leave the mask empty, so
    # nothing would be written and the BF/PC bar would never appear. The real
    # value (scale_bar_value, 0 for black) is applied at the masked pixels in
    # add_scale_bar.
    sentinel = 255 if dtype == np.uint8 else (1 << significant_bits) - 1

    if is_color:
        canvas = np.zeros((height, width, 3), dtype=dtype)
        canvas[y_start : y_end + 1, x_start : x_end + 1, :] = sentinel
    else:
        canvas = np.zeros((height, width), dtype=dtype)
        canvas[y_start : y_end + 1, x_start : x_end + 1] = sentinel

    text_x_pos = x_start
    text_y_pos = y_end + 5
    font_scale = max(0.75, width / 2000)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    scale_bar_text = f'{scale_bar_length_um}um, {objective.magnification}x'

    while True:
        text_size, _ = cv2.getTextSize(
            text=scale_bar_text, fontFace=font_face, fontScale=font_scale, thickness=font_thickness
        )
        if text_size[0] < scale_bar_length_pixels:
            break
        font_scale *= 0.75

    cv2.putText(
        img=canvas,
        text=scale_bar_text,
        org=(text_x_pos, text_y_pos),
        fontFace=font_face,
        fontScale=font_scale,
        color=(sentinel, sentinel, sentinel),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=True,
    )

    # Mask marks the bar+text geometry (nonzero sentinel pixels), so the value
    # written in add_scale_bar can be anything -- including 0 for a black bar.
    if is_color:
        mask = np.any(canvas != 0, axis=2)
    else:
        mask = canvas != 0

    return canvas, mask, scale_bar_value


def add_scale_bar(
    image: np.ndarray,
    objective: 'Objective',
    binning_size: int,
    color: str | None = None,
    *,
    significant_bits: int,
    capabilities: 'ScopeCapabilities',
) -> np.ndarray:
    """``image`` with the scale bar drawn, on a copy; ``image`` is never written.

    The image is often a camera's delivered frame, which every consumer
    shares and the camera marks read-only. Returned unchanged when no bar
    can be drawn (an image too narrow, or no known pixel size).
    """
    global _scale_bar_cache

    height, width = image.shape[0], image.shape[1]

    MIN_IMAGE_WIDTH_PIXELS = 100
    if width < MIN_IMAGE_WIDTH_PIXELS:
        return image

    # A scale bar is a measurement claim; without a known pixel size there is
    # nothing to claim. Draw nothing rather than a bar of invented length.
    pixel_size_um = common_utils.get_pixel_size(
        focal_length=objective.focal_length_mm,
        binning_size=binning_size,
        capabilities=capabilities,
    )
    if pixel_size_um is None:
        return image

    dtype = image.dtype
    is_color = is_color_image(image=image)

    cache_key = (
        height,
        width,
        dtype,
        is_color,
        objective.focal_length_mm,
        objective.magnification,
        binning_size,
        color,
        # The white bar's value is the payload max for this depth, so two frames
        # of the same dtype but different significant bits (12-bit single vs
        # 16-bit summed) must not share a cached overlay.
        significant_bits,
    )

    if _scale_bar_cache.get('key') != cache_key:
        overlay, mask, value = _compute_scale_bar_overlay(
            height,
            width,
            dtype,
            is_color,
            objective,
            binning_size,
            color,
            significant_bits,
            pixel_size_um,
        )
        _scale_bar_cache = {'key': cache_key, 'overlay': overlay, 'mask': mask, 'value': value}

    cached = _scale_bar_cache
    mask = cached['mask']

    # Write the bar's pixel value at the masked geometry. Works for black
    # (value 0, BF/PC) and white (the payload max) alike -- the mask carries
    # the location, so the value can be 0 without erasing the bar.
    drawn = image.copy()
    drawn[mask] = cached['value']

    return drawn


def add_timestamp(image: np.ndarray, timestamp_str: str) -> np.ndarray:
    """Draw a timestamp on a copy of the image; ``image`` is never written.

    The image can be a camera's delivered frame, which every consumer
    shares and the camera marks read-only.

    Args:
        image: Input image array.
        timestamp_str: Text to draw.

    Returns:
        A C-contiguous copy of the image with the timestamp drawn.
    """
    height, width = image.shape[0], image.shape[1]

    dtype = image.dtype

    text_color_bg = (0, 0, 0)
    font_scale = max(0.75, width / 2000)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    text_size, _ = cv2.getTextSize(
        text=timestamp_str, fontFace=font_face, fontScale=font_scale, thickness=font_thickness
    )
    text_w, text_h = text_size

    bottom_offset = int(height / 40)
    left_offset = int(width / 40)

    top_offset = height - bottom_offset

    if dtype == np.uint8:
        text_intensity = 2**8 - 1
    else:  # 16-bit
        text_intensity = 2**16 - 1

    # A C-contiguous copy: np.flip() produces non-contiguous views that
    # OpenCV rejects with "Layout incompatible with cv::Mat".
    image = np.array(image, order='C', copy=True)
    cv2.rectangle(
        image,
        (left_offset, top_offset),
        (left_offset + text_w, top_offset + text_h),
        text_color_bg,
        -1,
    )

    cv2.putText(
        img=image,
        text=f'{timestamp_str}',
        org=(left_offset, int(top_offset + text_h + font_scale - 1)),
        fontFace=font_face,
        fontScale=font_scale,
        color=(text_intensity, text_intensity, text_intensity),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=False,
    )

    return image
