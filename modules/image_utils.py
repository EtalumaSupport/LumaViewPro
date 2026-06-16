# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import datetime
import enum
import json
import pathlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tifffile as tf

from modules.common_utils import ColorChannel
import modules.common_utils as common_utils
import modules.image_utils as image_utils

from fractions import Fraction

from lvp_logger import logger, version

# Pre-built lookup tables for bit-depth conversion (built once at import, ~4 KB each)
# Using the same float math as the original per-pixel conversion ensures identical results.
_LUT_12_TO_8 = np.clip(np.arange(4096, dtype=np.float32) / 4095 * 255, 0, 255).astype(np.uint8)

_LUT_16_TO_8 = (np.arange(65536, dtype=np.float64) / 256).astype(np.uint8)


def is_color_image(image) -> bool:
    return len(image.shape) == 3 and image.shape[2] == 3


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


def mono_to_rgb_falsecolor(mono: np.ndarray, layer: str) -> np.ndarray:
    """Map a 2D mono array to a 3-channel RGB array via the layer's false color.

    The single Phase-1 boundary helper for mono -> RGB widening. Use at
    encode boundaries (live preview Kivy texture, MP4 / AVI encode) and
    legacy-display paths -- NOT in the save pipeline (mono-native save
    keeps 2D + layer metadata; widening to RGB at save bakes the false
    color into the file).

    Args:
        mono: 2D ndarray of uint8 / uint16 dtype.
        layer: Fluorescence channel name. ``Red``, ``Green``, ``Blue``,
            ``Lumi`` place the mono signal in the matching RGB index.
            Transmitted layers (``BF``, ``PC``, ``DF``) tile the mono
            into all three channels (grayscale RGB). Unknown layers
            tile into all three channels as a safe fallback.

    Returns:
        ``(H, W, 3)`` ndarray with the same dtype as ``mono``. Source
        array is not modified.

    Raises:
        ValueError: ``mono`` is not 2D.
    """
    if mono.ndim != 2:
        raise ValueError(f'mono_to_rgb_falsecolor expects 2D input, got shape {mono.shape}')

    h, w = mono.shape
    rgb = np.zeros((h, w, 3), dtype=mono.dtype)

    if layer in ('Blue', 'Lumi'):
        rgb[:, :, 2] = mono
    elif layer == 'Green':
        rgb[:, :, 1] = mono
    elif layer == 'Red':
        rgb[:, :, 0] = mono
    else:
        rgb[:, :, 0] = mono
        rgb[:, :, 1] = mono
        rgb[:, :, 2] = mono
    return rgb


# Module-level once-per-process flag for legacy-collapse log noise control.
_legacy_collapse_warned: bool = False


# Private TIFF tag (range 32768-65535) carrying a JSON-encoded copy of
# the full hyperstack metadata dict, including the Instrument / Plate /
# Objective subtrees that tifffile's auto-OME-XML serializer silently
# drops. LVP-aware consumers recover the dropped fields by parsing the
# tag value as JSON via read_hyperstack_private_metadata; FIJI / ImageJ
# / generic OME readers ignore the private tag harmlessly. The OME-XML
# in ImageDescription (Image > Pixels > Channel + Plane) is unchanged
# by this sidecar -- both representations coexist on the same IFD.
LVP_HYPERSTACK_METADATA_TIFF_TAG: int = 51838


def _json_default_numpy(obj):
    """JSON-encode hook for numpy scalars / arrays that may sit in
    hyperstack metadata dicts (positions sourced from pandas DataFrame
    cells arrive as numpy.float64; channel-index lists as numpy arrays).
    """
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


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
    global _legacy_collapse_warned
    img = tf.imread(str(path))
    if img.ndim == 3 and img.shape[2] == 3:
        nonzero_channels = [i for i in range(3) if img[..., i].any()]
        # Considered tightening this to a known-LUT match or a metadata
        # marker so a genuine single-color composite is never collapsed;
        # rejected because every 3-channel input that reaches this reader is
        # a false-color replica (a legacy mono->RGB save, or a
        # false_color_16bit-on save) -- real composites carry signal in
        # multiple channels -- and the one-shot log line flags any collapse.
        # Revisit if a true single-channel RGB output ever feeds VideoBuilder
        # or CompositeGeneration.
        if len(nonzero_channels) == 1:
            if not _legacy_collapse_warned:
                logger.info(f'Legacy false-color TIFF detected at {path}; loaded as mono')
                _legacy_collapse_warned = True
            return img[..., nonzero_channels[0]].copy()
    return img


def _read_ome_input_metadata(ome_xml: str, datetime_value) -> dict | None:
    """Recover the flat metadata dict from a tifffile-auto-OME description.

    tifffile's auto-OME serializer preserves only a subset of the structured
    metadata into the ImageDescription XML: Plane PositionX/Y/Z + ExposureTime,
    Pixels PhysicalSizeX, and Channel Name. Gain/Illumination, Objective,
    Instrument, and Plate are dropped at write and cannot be recovered -- they
    take the same sentinel defaults build_postproc_output_metadata applies when
    no structured metadata is present. Returns None on a parse failure or a
    missing Plane position so the caller falls back to defaults.
    """
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None

    def _local(tag: str) -> str:
        return tag.rsplit('}', 1)[-1]

    pixels = next((el for el in root.iter() if _local(el.tag) == 'Pixels'), None)
    if pixels is None:
        return None
    plane = next((el for el in pixels if _local(el.tag) == 'Plane'), None)
    channel = next((el for el in pixels if _local(el.tag) == 'Channel'), None)
    if plane is None:
        return None

    def _float(attrs: dict, key: str):
        raw = attrs.get(key)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    pos_x = _float(plane.attrib, 'PositionX')
    pos_y = _float(plane.attrib, 'PositionY')
    if pos_x is None or pos_y is None:
        return None
    pos_z = _float(plane.attrib, 'PositionZ')
    exposure = _float(plane.attrib, 'ExposureTime')
    pixel_size = _float(pixels.attrib, 'PhysicalSizeX')

    flat: dict = {
        'plate_pos_mm': {'x': pos_x, 'y': pos_y},
        'z_pos_um': pos_z if pos_z is not None else 0.0,
        # Dropped by tifffile's auto-OME serializer; default to match the
        # no-structured-metadata path so the derived output is consistent.
        'objective': {},
        'exposure_time_ms': exposure if exposure is not None else 0.0,
        'gain_db': 0.0,
        'illumination_ma': 0.0,
        'pixel_size_um': pixel_size if pixel_size is not None else 1.0,
    }
    if channel is not None and channel.attrib.get('Name'):
        flat['channel'] = channel.attrib['Name']
    if datetime_value is not None:
        flat['datetime'] = datetime_value
    return flat


def read_postproc_input_metadata(path: pathlib.Path) -> dict | None:
    """Reverse the write_tiff metadata serialization for one input TIFF.

    Reads a TIFF written by ``write_tiff`` and reconstructs the flat
    metadata dict that the caller originally passed in. Used by
    post-processing modules (stitcher, zprojector) to propagate
    acquisition context from inputs to derived outputs.

    Returns None when the input has no recoverable structured metadata
    (bare ``tifffile.imwrite`` outputs that carry only ``{'shape': ...}``,
    or files written by a non-LumaViewPro pipeline). Returns None on any
    parse failure so callers can fall back to defaults without crashing.

    Args:
        path: TIFF file path.

    Returns:
        Flat dict matching ``write_tiff``'s ``metadata`` parameter shape,
        or None if the input carries no structured metadata.
    """
    try:
        with tf.TiffFile(str(path)) as tif:
            shaped = tif.shaped_metadata
            ome_xml = tif.ome_metadata
            datetime_tag = tif.pages[0].tags.get('DateTime')
            datetime_value = datetime_tag.value if datetime_tag else None
    except Exception:
        return None

    if not shaped:
        # OME-TIFF inputs carry no shaped_metadata; recover what tifffile's
        # auto-OME serializer preserved into the ImageDescription XML so an
        # OME-TIFF tile still forwards acquisition context to derived outputs.
        if ome_xml:
            return _read_ome_input_metadata(ome_xml, datetime_value)
        return None
    structured = shaped[0]
    if 'Plane' not in structured:
        # Bare tifffile.imwrite (only carries 'shape') or other non-LVP
        # producer; no acquisition context to forward.
        return None

    plane = structured['Plane']
    try:
        flat: dict = {
            'plate_pos_mm': {
                'x': plane['PositionX'],
                'y': plane['PositionY'],
            },
            'z_pos_um': plane['PositionZ'],
            'objective': plane.get('Objective', {}),
            'exposure_time_ms': plane['ExposureTime'],
            'gain_db': plane['Gain'],
            'illumination_ma': plane['Illumination'],
            'pixel_size_um': structured['PhysicalSizeX'],
            'channel': structured['Channel']['Name'][0],
        }
    except (KeyError, IndexError, TypeError):
        # Structured TIFF present but missing required acquisition keys
        # (older LVP file, third-party producer, or a non-acquisition
        # frame type); fall back to defaults rather than crashing the
        # post-processing job, per this function's documented contract.
        return None
    if datetime_value is not None:
        flat['datetime'] = datetime_value

    # Per-frame markers travel with the original capture, not with
    # derived outputs. Read them so the helper is reusable; the builder
    # for derived outputs strips them before passing to write_tiff.
    if 'Timestamp' in plane:
        flat['timestamp_iso'] = plane['Timestamp']
    if 'TimestampCameraTicks' in plane:
        flat['timestamp_camera_ticks'] = plane['TimestampCameraTicks']
    if 'TimestampCameraTickHz' in plane:
        flat['timestamp_camera_tick_hz'] = plane['TimestampCameraTickHz']
    if 'FrameID' in plane:
        flat['frame_id'] = plane['FrameID']

    if 'Instrument' in structured:
        inst = structured['Instrument']
        microscope = inst.get('Microscope', {})
        detector = inst.get('Detector', {})
        flat['instrument'] = {
            'manufacturer': microscope.get('Manufacturer'),
            'model': microscope.get('Model'),
            'serial_number': microscope.get('SerialNumber'),
            'firmware_version': microscope.get('FirmwareVersion'),
            'camera_model': detector.get('Model'),
        }

    if 'Plate' in structured:
        p = structured['Plate']
        flat['plate'] = {
            'name': p.get('Name'),
            'rows': p.get('Rows'),
            'columns': p.get('Columns'),
        }
        if 'Standard' in p:
            flat['plate']['standard'] = p['Standard']
        if 'WellLabel' in p:
            flat['well_label'] = p['WellLabel']

    return flat


def read_frame_timestamp(path: pathlib.Path) -> datetime.datetime | None:
    """Recover the per-frame capture timestamp from one recorded TIFF.

    Used by the video builder to draw a per-frame timestamp overlay at
    build time instead of burning it into the pixels at capture time, so
    the Create-Video timestamp toggle controls whether the timestamp
    appears. Tolerates the differing metadata shapes the capture paths
    write:

    - manual "Frames" recordings carry a flat ``timestamp_iso`` key;
    - protocol video-step frames carry a flat ``timestamp`` (and
      ``datetime``) key in ``%Y:%m:%d %H:%M:%S.%f`` form;
    - protocol scan images carry the timestamp under ``Plane.Timestamp``
      (ISO) in the structured metadata.

    Args:
        path: TIFF file path.

    Returns:
        The frame's capture time, or None when no readable timestamp is
        present (caller falls back to its own source or skips the overlay).
    """
    try:
        with tf.TiffFile(str(path)) as tif:
            shaped = tif.shaped_metadata
    except Exception:
        return None

    if not shaped:
        return None
    structured = shaped[0]

    # Structured scan images nest the timestamp under Plane; frame
    # recordings keep a flat dict. Check both, ISO first then the
    # colon-separated capture format.
    candidates = []
    plane = structured.get('Plane') if isinstance(structured, dict) else None
    if isinstance(plane, dict) and plane.get('Timestamp'):
        candidates.append(('iso', plane['Timestamp']))
    if structured.get('timestamp_iso'):
        candidates.append(('iso', structured['timestamp_iso']))
    if structured.get('timestamp'):
        candidates.append(('%Y:%m:%d %H:%M:%S.%f', structured['timestamp']))
    if structured.get('datetime'):
        candidates.append(('%Y:%m:%d %H:%M:%S', structured['datetime']))

    for fmt, value in candidates:
        try:
            if fmt == 'iso':
                return datetime.datetime.fromisoformat(value)
            return datetime.datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue

    return None


def build_postproc_output_metadata(
    input_path: pathlib.Path,
    channel: str,
    *,
    plate_pos_mm_override: dict | None = None,
    z_pos_um_override: float | None = None,
) -> dict:
    """Build a write_tiff metadata dict for a post-processing output.

    Reads acquisition context from one representative input TIFF
    (objective, exposure, gain, illumination, pixel size, instrument,
    plate, well label) and forwards it into the derived output. Per-frame
    timestamps and frame IDs are stripped because they describe the
    original capture, not the derived image.

    ``datetime`` is set to the post-processing wall-clock time so derived
    outputs are distinguishable from the captures that fed them.

    Callers in stitcher pass ``plate_pos_mm_override`` with the stitched
    region's geometric center; zprojector leaves both overrides None and
    inherits the input slice's position. Falls back to sentinel defaults
    when the input has no structured metadata (test fixtures, external
    files).

    Args:
        input_path: First-input TIFF; metadata is read from here.
        channel: Layer color for the derived output (Blue / Green / Red
            / Lumi / BF / PC / DF).
        plate_pos_mm_override: Optional plate_pos_mm dict to override
            the value read from the input.
        z_pos_um_override: Optional z_pos_um to override the value read
            from the input.

    Returns:
        Dict ready to pass as ``write_tiff``'s ``metadata`` parameter.
    """
    metadata = read_postproc_input_metadata(input_path)
    if metadata is None:
        metadata = {
            'plate_pos_mm': {'x': 0.0, 'y': 0.0},
            'z_pos_um': 0.0,
            'objective': {},
            'exposure_time_ms': 0.0,
            'gain_db': 0.0,
            'illumination_ma': 0.0,
            'pixel_size_um': 1.0,
        }
    else:
        for per_capture_field in (
            'timestamp_iso',
            'timestamp_camera_ticks',
            'timestamp_camera_tick_hz',
            'frame_id',
        ):
            metadata.pop(per_capture_field, None)

    metadata['channel'] = channel
    metadata['datetime'] = datetime.datetime.now().isoformat(timespec='seconds')

    if plate_pos_mm_override is not None:
        metadata['plate_pos_mm'] = plate_pos_mm_override
    if z_pos_um_override is not None:
        metadata['z_pos_um'] = z_pos_um_override

    return metadata


def build_composite_output_metadata(reference_input_path: pathlib.Path) -> dict:
    """Build a write_tiff metadata dict for a composite output.

    Composite outputs merge multiple input channels with different
    per-channel exposure / gain / illumination, so those fields zero
    out -- they describe the source captures, not the merged image.
    Shared acquisition context (objective, position, pixel size,
    instrument, plate, well_label) propagates from the reference input;
    composite input channels share these at the same site.

    ``channel`` is set to ``'Composite'`` to distinguish the derived
    output from its per-channel sources. ``datetime`` is wall-clock
    post-processing time.

    Args:
        reference_input_path: Any composite-input TIFF; shared metadata
            is read from here. Callers pass the first available channel
            (red -> green -> blue -> transmitted order).

    Returns:
        Dict ready to pass as write_tiff's ``metadata`` parameter.
    """
    metadata = build_postproc_output_metadata(
        input_path=reference_input_path,
        channel='Composite',
    )
    metadata['exposure_time_ms'] = 0.0
    metadata['gain_db'] = 0.0
    metadata['illumination_ma'] = 0.0
    return metadata


def build_hyperstack_output_metadata(
    reference_input_path: pathlib.Path,
    *,
    channel_names: list[str],
    plane_positions: dict,
    significant_bits: int,
    pixel_size_um: float,
) -> dict:
    """Build a TZCYX hyperstack OME metadata dict for ``tf.imwrite``.

    Reads acquisition context from one input-frame TIFF and reshapes
    it into the OME hyperstack schema that tifffile consumes when
    ``ome=True`` + ``axes='TZCYX'``. Per-plane positions arrive as
    parallel lists (one entry per T*Z*C plane in scan order); they land
    in the OME-XML ``<Plane>`` elements via tifffile's per-axis list
    convention.

    Pixel size is supplied separately rather than read from the input
    because hyperstacks may use a different binning configuration than
    the source captures; the caller passes the derived pixel_size_um.

    Tifffile-OME-XML constraint: the underlying tifffile serializer
    writes only Image > Pixels > Channel + Plane to OME-XML. Instrument,
    Objective, and Plate keys in the metadata dict are silently dropped
    by tifffile -- they remain in the returned dict for future-proofing
    (if tifffile gains broader OME schema coverage), but consumers
    cannot read those fields from current hyperstack outputs. Closing
    the Instrument/Plate provenance gap requires either hand-rolled
    OME-XML or a private-tag JSON sidecar; both deferred as separate
    work.

    Args:
        reference_input_path: One input frame TIFF; acquisition context
            is read from here. All hyperstack input frames share these
            shared fields (same site, same objective, same scope).
        channel_names: One channel name per C-axis position.
        plane_positions: Dict with PositionX / PositionY / PositionZ
            lists, one entry per T*Z*C plane in scan order. Caller is
            responsible for list-length consistency with the data
            array.
        significant_bits: 8 for uint8 captures, 16 for uint16.
        pixel_size_um: Hyperstack pixel size in microns.

    Returns:
        OME-shaped metadata dict; pass to ``tf.imwrite(..., metadata=)``.
    """
    inflat = read_postproc_input_metadata(reference_input_path) or {}

    num_planes = len(plane_positions['PositionX'])

    # Per-channel OME Color hints so FIJI's Bioformats reader auto-
    # opens the hyperstack in Composite view with the right color per
    # channel. OME-XML uses a signed 32-bit RGBA integer per channel
    # (R << 24 | G << 16 | B << 8 | A, two's-complement-folded into
    # int32). Tifffile drops metadata['LUTs'] when ome=True is set on
    # the writer (the OME-XML is the canonical color carrier in that
    # mode); Channel.Color reaches the same FIJI auto-rendering path
    # via Bioformats. Channels not mapped by LvpColormap fall back to
    # white (-1) which FIJI renders as plain grayscale.
    channel_colors: list[int] = []
    for name in channel_names:
        try:
            colormap_type = color_channel_to_colormap_type(color_channel=name)
        except Exception:
            colormap_type = LvpColormap.GRAY
        channel_colors.append(_lvp_colormap_to_ome_rgba(colormap_type))

    metadata: dict = {
        'axes': 'TZCYX',
        'SignificantBits': significant_bits,
        'Pixels': {
            'PhysicalSizeX': pixel_size_um,
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeY': pixel_size_um,
            'PhysicalSizeYUnit': 'um',
        },
        'Channel': {'Name': channel_names, 'Color': channel_colors},
        'Plane': {
            'PositionX': plane_positions['PositionX'],
            'PositionY': plane_positions['PositionY'],
            'PositionZ': plane_positions['PositionZ'],
            'PositionXUnit': ['mm'] * num_planes,
            'PositionYUnit': ['mm'] * num_planes,
            'PositionZUnit': ['um'] * num_planes,
        },
    }

    objective_dict = inflat.get('objective') or {}
    instrument = inflat.get('instrument') or {}
    plate = inflat.get('plate') or {}
    if instrument:
        metadata['Instrument'] = {
            'Microscope': {
                'Manufacturer': instrument.get('manufacturer') or 'Etaluma',
                'Model': instrument.get('model') or '',
                'SerialNumber': instrument.get('serial_number') or '',
                'FirmwareVersion': instrument.get('firmware_version') or '',
            },
            'Objective': {
                'Model': objective_dict.get('model') or '',
                'Manufacturer': objective_dict.get('manufacturer') or '',
                'Magnification': objective_dict.get('magnification'),
                'LensNA': objective_dict.get('aperture'),
                'WorkingDistance': objective_dict.get('working_distance'),
                'Immersion': objective_dict.get('immersion') or 'Air',
            },
            'Detector': {
                'Model': instrument.get('camera_model') or '',
                'Type': 'CMOS',
            },
        }
    if plate.get('rows') and plate.get('columns'):
        metadata['Plate'] = {
            'Name': plate.get('name') or '',
            'Rows': plate.get('rows'),
            'Columns': plate.get('columns'),
            'WellLabel': inflat.get('well_label', ''),
        }
        if plate.get('standard'):
            metadata['Plate']['Standard'] = plate['standard']

    return metadata


def read_hyperstack_private_metadata(path: pathlib.Path) -> dict | None:
    """Read the LVP private-tag JSON metadata from a hyperstack TIFF.

    Hyperstack writes carry the full metadata dict (including the
    Instrument / Plate / Objective subtrees that tifffile's auto-OME
    serializer drops) in a private TIFF tag alongside the standard
    OME-XML. This reader returns the parsed JSON dict so LVP-aware
    consumers can recover the dropped fields without parsing OME-XML.

    Args:
        path: Hyperstack TIFF file path.

    Returns:
        Parsed metadata dict matching what was passed to write_tiff's
        ``hyperstack_metadata`` parameter at write time. Returns None
        if the file has no private tag (third-party hyperstack TIFFs,
        or LVP files written before this tag was introduced), if the
        tag exists but is not valid JSON, or if the file cannot be
        opened.
    """
    try:
        with tf.TiffFile(str(path)) as tif:
            page0 = tif.pages[0]
            tag = page0.tags.get(LVP_HYPERSTACK_METADATA_TIFF_TAG)
            if tag is None:
                return None
            value = tag.value
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            return json.loads(value)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


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


def add_false_color(array, color, output=None):
    src_dtype = array.dtype
    if (not image_utils.is_color_image(array)) and (
        color in (*common_utils.get_fluorescence_layers(), *common_utils.get_luminescence_layers())
    ):
        if (
            output is not None
            and output.shape == (array.shape[0], array.shape[1], 3)
            and output.dtype == src_dtype
        ):
            img = output
            img[:] = 0
        else:
            img = np.zeros((array.shape[0], array.shape[1], 3), dtype=src_dtype)
        # RGB ordering: index 0=Red, 1=Green, 2=Blue. Matches the canonical
        # save-path convention shared with composite_builder. OpenCV consumers
        # (cv2.VideoWriter) convert RGB->BGR at their own boundary.
        if color in ('Blue', 'Lumi'):
            img[:, :, 2] = array
        elif color == 'Green':
            img[:, :, 1] = array
        elif color == 'Red':
            img[:, :, 0] = array

        # For HSL colorspace
        # elif color == 'Lumi':
        #     img[:,:,0] = 215 / 2 # Hue (OpenCV uses range of 0-180, so divide by 2)
        #     img[:,:,1] = array # Luminance
        #     img[:,:,2] = 255 # Saturation
    else:
        img = array

    # For HSL colorspace
    # if color == 'Lumi':
    #     img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)

    return img


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


def rgb_image_to_gray(image):

    def _is_grayscale(image):
        shape = image.shape
        return bool((len(shape) <= 2) or (shape[2] == 1))

    def _values_in_one_plane(image):
        used_color_planes = get_used_color_planes(image=image)

        return len(used_color_planes) <= 1

    if _is_grayscale(image=image):
        return image

    if _values_in_one_plane(image=image):
        return np.amax(image, axis=2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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


def encode_display_jpg(array, color, jpeg_quality: int = 90) -> bytes:
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
        color: Channel color label (BF, Blue, Green, Red, Lumi, ...).
        jpeg_quality: JPEG quality, 1-100.

    Returns:
        bytes: JPEG-encoded image.
    """
    img8 = convert_12bit_to_8bit(array)
    if img8.ndim == 3:
        # Already a display RGB image (e.g. a crosshairs / bullseye
        # overlay). These share the false-color RGB convention, so take
        # it as RGB rather than re-applying the layer color.
        rgb = img8
    else:
        # Bake the channel's displayed color at this encode boundary
        # (mono_to_rgb_falsecolor is the canonical mono -> RGB widening;
        # BF / transmitted layers come back grayscale RGB).
        rgb = mono_to_rgb_falsecolor(img8, color)
    # The RGB produced above is R=0,G=1,B=2; cv2 (encode_image) treats a
    # 3-channel array as BGR, so reverse the channel axis or the saved
    # JPG would have red and blue swapped.
    bgr = rgb[:, :, ::-1]
    return encode_image(bgr, fmt='jpeg', jpeg_quality=jpeg_quality)


def convert_12bit_to_8bit(image, out=None):
    if image.dtype == 'uint8':
        return image

    # Mirror PIW-5's convert_12bit_to_16bit(out=) pattern. The
    # caller-supplied out buffer eliminates the per-call fresh allocation
    # for the LUT indexing result -- saves ~120 MB/s allocator churn on
    # the 30fps Pylon 12-bit preview path. Mismatched shape/dtype falls
    # back to fresh allocation rather than failing.
    if out is not None and out.shape == image.shape and out.dtype == np.uint8:
        np.take(_LUT_12_TO_8, image, out=out)
        return out
    return _LUT_12_TO_8[image]


def convert_12bit_to_16bit(image, out=None):
    if image.dtype == 'uint8':
        return image

    # PIW-5: caller-supplied out buffer eliminates the per-save image.copy() (~24 MB).
    # Mismatched shape/dtype falls back to fresh allocation rather than failing.
    if out is not None and out.shape == image.shape and out.dtype == image.dtype:
        np.copyto(out, image)
        new_image = out
    else:
        new_image = image.copy()
    new_image *= 16
    return new_image


def convert_16bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image

    return _LUT_16_TO_8[image]


@enum.unique
class LvpColormap(enum.Enum):
    GRAY = 'gray'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


def color_channel_to_colormap_type(color_channel: str | ColorChannel) -> LvpColormap:
    if isinstance(color_channel, str):
        color_channel = ColorChannel[color_channel]

    lut = {
        ColorChannel.Lumi: LvpColormap.BLUE,
        ColorChannel.Blue: LvpColormap.BLUE,
        ColorChannel.Green: LvpColormap.GREEN,
        ColorChannel.Red: LvpColormap.RED,
        ColorChannel.BF: LvpColormap.GRAY,
        ColorChannel.PC: LvpColormap.GRAY,
        ColorChannel.DF: LvpColormap.GRAY,
    }

    return lut[color_channel]


def get_tiff_colormap(colormap: LvpColormap, dtype):
    """Build a TIFF colormap array for PALETTE photometric (8-bit only).

    Returns a (3, 256) array suitable for tifffile's ``colormap`` parameter.
    Only used for 8-bit false-color images -- Windows Preview supports PALETTE
    with uint8 but NOT with uint16.
    """
    if dtype not in ('uint8', np.uint8):
        raise NotImplementedError(f'TIFF colormap only supported for uint8, got {dtype}')

    max_value = np.iinfo(np.uint8).max + 1  # 256

    if colormap == LvpColormap.GRAY:
        return np.tile(np.arange(0, max_value, 1, dtype=np.uint8), (3, 1))

    cmap_array = np.zeros((3, 256), dtype=np.uint8)
    if colormap == LvpColormap.RED:
        cmap_array[0] = np.arange(0, max_value, 1, dtype=np.uint8)
    elif colormap == LvpColormap.GREEN:
        cmap_array[1] = np.arange(0, max_value, 1, dtype=np.uint8)
    elif colormap == LvpColormap.BLUE:
        cmap_array[2] = np.arange(0, max_value, 1, dtype=np.uint8)
    else:
        raise NotImplementedError(f'Unsupported colormap: {colormap}')
    return cmap_array


def _lvp_colormap_to_ome_rgba(colormap: 'LvpColormap') -> int:
    """Map an LVP colormap to the signed-32-bit OME Channel.Color value.

    OME-XML encodes Channel.Color as ``(R << 24) | (G << 16) | (B << 8) | A``
    with the unsigned 32-bit result reinterpreted as Python signed int32.
    Used by FIJI's Bioformats reader to assign a per-channel LUT when
    opening the hyperstack -- without it, FIJI defaults to grayscale.
    """
    color_map = {
        LvpColormap.RED: (255, 0, 0),
        LvpColormap.GREEN: (0, 255, 0),
        LvpColormap.BLUE: (0, 0, 255),
        LvpColormap.GRAY: (255, 255, 255),
    }
    r, g, b = color_map.get(colormap, (255, 255, 255))
    unsigned = (r << 24) | (g << 16) | (b << 8) | 0xFF
    return unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned


def get_imagej_lut(colormap: LvpColormap):
    """Build an ImageJ-style LUT: (3, 256) uint8 array.

    Stored in ImageJ metadata (not TIFF tag 320), so Windows Preview sees
    plain MINISBLACK grayscale while ImageJ auto-applies the color LUT.
    Used for 16-bit images where PALETTE photometric breaks Windows Preview.
    """
    ramp = np.arange(256, dtype=np.uint8)
    zeros = np.zeros(256, dtype=np.uint8)
    if colormap == LvpColormap.GRAY:
        return np.stack([ramp, ramp, ramp])
    elif colormap == LvpColormap.RED:
        return np.stack([ramp, zeros, zeros])
    elif colormap == LvpColormap.GREEN:
        return np.stack([zeros, ramp, zeros])
    elif colormap == LvpColormap.BLUE:
        return np.stack([zeros, zeros, ramp])
    else:
        raise NotImplementedError(f'Unsupported colormap: {colormap}')


def maybe_apply_false_color(
    data: np.ndarray,
    color: str,
    use_false_color_16bit: bool | None = None,
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

    Already-RGB inputs, 8-bit inputs, transmitted layers (BF/PC/DF), and
    unknown layer names always pass through. Callers may pass the resolved
    bool to skip the per-save settings_lock acquire; ``None`` triggers a
    one-shot read so derived-output paths (stitch / zproject) that call
    write_tiff without the flag still honor the user setting.
    """
    if not (
        data.dtype == np.uint16
        and not is_color_image(data)
        and color in common_utils.get_image_layers()
    ):
        return data
    try:
        if use_false_color_16bit is None:
            # Function-local import breaks the image_utils <-> app_context
            # cycle; app_context imports image_utils at module load.
            from modules import app_context as _app_ctx

            with _app_ctx.ctx.settings_lock:
                use_false_color_16bit = _app_ctx.ctx.settings.get('false_color_16bit', False)
        if use_false_color_16bit:
            return add_false_color(data, color, output=output_buf)
    except Exception:
        logger.exception(
            '[image_utils] maybe_apply_false_color: false-color application '
            'failed for color=%s; returning input',
            color,
        )
    return data


def write_tiff(
    data,
    file_loc: pathlib.Path,
    metadata: dict,
    ome: bool,
    color: str,
    video_frame: bool = False,
    extratags: list | None = None,
    use_false_color_16bit: bool | None = None,
    false_color_buf: np.ndarray | None = None,
    rgb_buf: np.ndarray | None = None,
    hyperstack_metadata: dict | None = None,
    hyperstack_options: dict | None = None,
    hyperstack_resolution: tuple | None = None,
):
    if hyperstack_metadata is not None:
        # Hyperstack TZCYX path: 5D data with caller-prepared OME
        # metadata (from build_hyperstack_output_metadata) and explicit
        # per-tiff options. Bypasses the per-image branches below --
        # maybe_apply_false_color expects 2D mono input, generate_tiff_data
        # generates per-image metadata, and _validate_type rejects the
        # ome=True + imagej=True combo that hyperstack readers (FIJI,
        # ImageJ) consume together. The caller (stack_builder) supplies
        # the full OME dict + write options + resolution; this branch
        # passes them through verbatim.
        #
        # JSON sidecar: tifffile's auto-OME serializer silently drops
        # Instrument / Plate / Objective from the metadata dict. The
        # full dict gets serialized into a private TIFF tag so
        # LVP-aware consumers can recover those fields; FIJI / ImageJ
        # ignore the unknown tag.
        use_bigtiff = data.nbytes > 3.8 * 1024 * 1024 * 1024
        write_options = hyperstack_options or {}
        # Strip rendering-hint keys from the JSON sidecar copy. LUTs +
        # Channel.Color are encoded into the file's TIFF / OME-XML
        # sections directly; the sidecar is for LVP-aware consumers
        # recovering the dropped-by-tifffile OME subtrees (Instrument /
        # Plate / Objective), not for re-deriving the file's render
        # hints. Bloats the sidecar by ~3 KB per channel of LUT data
        # for zero downstream value if left in.
        sidecar_metadata = {k: v for k, v in hyperstack_metadata.items() if k != 'LUTs'}
        sidecar_json = json.dumps(sidecar_metadata, default=_json_default_numpy)
        sidecar_extratag = (
            LVP_HYPERSTACK_METADATA_TIFF_TAG,
            's',
            0,
            sidecar_json,
            True,
        )
        caller_extratags = list(write_options.pop('extratags', []) or [])
        caller_extratags.append(sidecar_extratag)
        with tf.TiffWriter(
            str(file_loc),
            ome=True,
            imagej=True,
            bigtiff=use_bigtiff,
        ) as tif:
            tif.write(
                data,
                resolution=hyperstack_resolution,
                metadata=hyperstack_metadata,
                software=f'LumaViewPro {version}',
                extratags=caller_extratags,
                **write_options,
            )
        return

    if extratags is None:
        extratags = []

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
    elif not video_frame:
        if is_color_image(data):
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

        if video_frame:
            type_count += 1
            image_type = 'video_frame'

        if type_count > 1:
            raise ValueError('Tiff must only be one type at most (OME, ImageJ, or Video Frame)')

        return image_type

    image_type = _validate_type()

    support_data = generate_tiff_data(
        data=data, metadata=metadata, image_type=image_type, color=color
    )

    with tf.TiffWriter(str(file_loc), **kwargs) as tif:
        if image_type == 'video_frame':
            tif.write(
                data,
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f'LumaViewPro {version}',
                **support_data['options'],
            )

        elif (image_type is None) and is_color_image(image=data):
            # Handles case where an actual color image is provided (such as the bullseye in engineering mode)
            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f'LumaViewPro {version}',
                **support_data['options'],
            )

        elif (image_type == 'ome') and is_color_image(image=data):
            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f'LumaViewPro {version}',
                **support_data['options'],
            )

        else:
            # 8-bit fluorescence: PALETTE photometric with colormap -- gives
            # false color in both Windows Preview and ImageJ.
            # 16-bit fluorescence: MINISBLACK photometric -- Windows Preview
            # compatible (shows grayscale). Color via ImageJ LUT metadata
            # (imagej type) or OME Channel metadata (ome type).
            # BF/PC/DF: always MINISBLACK, no colormap needed.
            colormap_array = None
            if data.dtype == np.uint8 and color in common_utils.get_image_layers():
                colormap_type = color_channel_to_colormap_type(color_channel=color)
                if colormap_type != LvpColormap.GRAY:
                    colormap_array = get_tiff_colormap(colormap=colormap_type, dtype=data.dtype)

            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f'LumaViewPro {version}',
                colormap=colormap_array,
                extratags=support_data['extratags'],
                **support_data['options'],
            )


def generate_tiff_data(
    data,
    metadata: dict,
    image_type: str,
    color: str,
):

    axes = 'YX'

    modality = ''
    if is_color_image(data):
        photometric = tf.PHOTOMETRIC.RGB
        modality = 'RGB'
        axes = 'YXS'  # 3rd dimension is samples (RGB channels)
    elif color in common_utils.get_transmitted_layers():
        photometric = tf.PHOTOMETRIC.MINISBLACK
        modality = color
    elif color in common_utils.get_image_layers():
        # 8-bit: PALETTE with colormap -- works in Windows Preview and ImageJ.
        # 16-bit: MINISBLACK -- Windows Preview can't handle PALETTE with uint16.
        #         Color is provided via ImageJ LUT metadata (ImageJ type) or
        #         OME Channel metadata (OME type).
        if data.dtype == np.uint8:
            photometric = tf.PHOTOMETRIC.PALETTE
        else:
            photometric = tf.PHOTOMETRIC.MINISBLACK
        modality = 'MIF'
    else:
        raise ValueError(f'Unexpected color value ({color}) for tiff data generation')

    # Video frames pass through metadata as-is with no structured fields
    if image_type == 'video_frame':
        # maxworkers=0 mirrors the imagej + default/ome paths below for
        # the same Windows kernel-handle-leak reason -- tifffile's per-
        # write ThreadPoolExecutor holds an Event handle that outlives
        # cleanup. No production workflow saturates this path today;
        # added for adjacent-symmetry with the other two save paths.
        options = {
            'photometric': photometric,
            'compression': 'lzw',
            'resolutionunit': 'CENTIMETER',
            'maxworkers': 0,
        }
        if data.dtype == np.uint8:
            options['tile'] = (128, 128)
        return {
            'metadata': metadata,
            'extratags': [],
            'options': options,
        }

    # Shared plane metadata for all structured image types
    plane = {
        'PositionX': metadata['plate_pos_mm']['x'],
        'PositionY': metadata['plate_pos_mm']['y'],
        'PositionZ': metadata['z_pos_um'],
        'PositionXUnit': 'mm',
        'PositionYUnit': 'mm',
        'PositionZUnit': 'um',
        'Objective': metadata['objective'],
        'ExposureTime': metadata['exposure_time_ms'],
        'ExposureTimeUnit': 'ms',
        'Gain': metadata['gain_db'],
        'GainUnit': 'dB',
        'Illumination': metadata['illumination_ma'],
        'IlluminationUnit': 'mA',
    }

    # Per-frame timestamps. Each is optional -- callers that don't capture
    # them (older static metadata builders, Stage 2-pending paths) simply
    # omit the keys and the corresponding TIFF fields don't appear.
    # timestamp_iso is host wall-clock at metadata-build time;
    # timestamp_camera_ticks is the camera-side ChunkTimestamp value;
    # timestamp_camera_tick_hz is the camera tick frequency for converting
    # ticks to seconds (1 GHz on Basler USB3, GevTimestampTickFrequency on
    # GigE). frame_id is the ChunkFrameID/Framecounter integer.
    if 'timestamp_iso' in metadata:
        plane['Timestamp'] = metadata['timestamp_iso']
        plane['TimestampSource'] = 'host_wallclock'
    if 'timestamp_camera_ticks' in metadata:
        plane['TimestampCameraTicks'] = metadata['timestamp_camera_ticks']
    if 'timestamp_camera_tick_hz' in metadata:
        plane['TimestampCameraTickHz'] = metadata['timestamp_camera_tick_hz']
    if 'frame_id' in metadata:
        plane['FrameID'] = metadata['frame_id']

    # Base metadata shared by all structured types
    tiff_metadata = {
        'axes': axes,
        'SignificantBits': data.itemsize * 8,
        'PhysicalSizeX': metadata['pixel_size_um'],
        'PhysicalSizeXUnit': 'um',
        'PhysicalSizeY': metadata['pixel_size_um'],
        'PhysicalSizeYUnit': 'um',
        'Channel': {'Name': [metadata['channel']]},
        'Plane': plane,
    }

    # OME Plate + Instrument blocks (#491). The tifffile dict API
    # serializes well-known OME keys into OME-XML where possible; less-
    # standard keys ride along in the structured metadata and survive a
    # round-trip through tifffile.TiffFile(...).imagej_metadata /
    # .ome_metadata so consumers can extract them. The Objective sub-
    # block reuses the existing per-image objective dict that already
    # backs the Plane.Objective field. Not yet shipped: OME LightSource
    # (LED wavelength + power -- not tracked per-color), OME Detector
    # gain/zoom (partial -- only model today), OME FilterSet (filter
    # wheel + dichroic info -- not tracked).
    objective_dict = metadata.get('objective') or {}
    instrument = metadata.get('instrument') or {}
    plate = metadata.get('plate') or {}
    if instrument:
        tiff_metadata['Instrument'] = {
            'Microscope': {
                'Manufacturer': instrument.get('manufacturer') or 'Etaluma',
                'Model': instrument.get('model') or '',
                'SerialNumber': instrument.get('serial_number') or '',
                'FirmwareVersion': instrument.get('firmware_version') or '',
            },
            'Objective': {
                'Model': objective_dict.get('model') or '',
                'Manufacturer': objective_dict.get('manufacturer') or '',
                'Magnification': objective_dict.get('magnification'),
                'LensNA': objective_dict.get('aperture'),
                'WorkingDistance': objective_dict.get('working_distance'),
                'Immersion': objective_dict.get('immersion') or 'Air',
            },
            'Detector': {
                'Model': instrument.get('camera_model') or '',
                'Type': 'CMOS',
            },
        }
    if plate.get('rows') and plate.get('columns'):
        tiff_metadata['Plate'] = {
            'Name': plate.get('name') or '',
            'Rows': plate.get('rows'),
            'Columns': plate.get('columns'),
            'WellLabel': metadata.get('well_label', ''),
        }
        if plate.get('standard'):
            tiff_metadata['Plate']['Standard'] = plate['standard']

    # ImageJ adds unit, channel modality, LUT, and document block
    if image_type == 'imagej':
        tiff_metadata['unit'] = 'um'
        tiff_metadata['Channel']['Modality'] = [modality]
        tiff_metadata['Document'] = {
            'Manufacturer': metadata.get('camera_make', ''),
            'Device': metadata.get('microscope', ''),
            'Model': metadata.get('microscope_model', '') or '',
            'SerialNumber': instrument.get('serial_number') or '',
            'FirmwareVersion': instrument.get('firmware_version') or '',
            'CameraModel': instrument.get('camera_model') or '',
            'PlateName': plate.get('name') or '',
            'PlateRows': plate.get('rows') or '',
            'PlateColumns': plate.get('columns') or '',
            'WellLabel': metadata.get('well_label', ''),
            'WellSite': metadata.get('well_site', ''),
        }
        # Embed color LUT in ImageJ metadata (not TIFF tag 320).
        # Windows Preview ignores ImageJ metadata -> sees MINISBLACK -> works.
        # ImageJ reads its own metadata -> auto-applies color LUT -> shows color.
        # mode='color' tells FIJI to apply the LUT (otherwise defaults to grayscale).
        colormap_type = color_channel_to_colormap_type(color_channel=color)
        lut = get_imagej_lut(colormap_type)
        tiff_metadata['LUTs'] = [lut]
        if colormap_type != LvpColormap.GRAY:
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
        resolution = (1.0 / metadata['pixel_size_um'], 1.0 / metadata['pixel_size_um'])
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
            'resolutionunit': 'CENTIMETER',
            'maxworkers': 0,
        }
        resolution = (1e4 / metadata['pixel_size_um'], 1e4 / metadata['pixel_size_um'])

    # Tile setting: 8-bit images use tiles for ImageJ colormap compatibility
    if data.dtype == np.uint8:
        options['tile'] = (128, 128)

    return {
        'metadata': tiff_metadata,
        'extratags': [],
        'options': options,
        'resolution': resolution,
    }


def ms_exposure_to_rational(ms_exposure):
    exposure_seconds = ms_exposure / 1000
    fraction = Fraction(exposure_seconds).limit_denominator(1_000_000)
    # Metadata uses rational number of seconds
    return fraction.numerator, fraction.denominator


def subject_dist_to_rational(distance):
    distance_meters = distance / 1_000_000  # Convert um to m
    fraction = Fraction(distance_meters).limit_denominator(1_000_000)
    return fraction.numerator, fraction.denominator


_scale_bar_cache = {}


def _compute_scale_bar_overlay(height, width, dtype, is_color, objective, binning_size, color):
    """Pre-render scale bar overlay and mask. Returns (overlay, mask, cache_key)."""
    pixel_size_um = common_utils.get_pixel_size(
        focal_length=objective['focal_length'], binning_size=binning_size
    )

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
        scale_bar_value = 4095

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
    sentinel = 255 if dtype == np.uint8 else 4095

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

    scale_bar_text = f'{scale_bar_length_um}um, {objective["magnification"]}x'

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
    image,
    objective: dict,
    binning_size: int,
    color: str | None = None,
):
    global _scale_bar_cache

    height, width = image.shape[0], image.shape[1]

    MIN_IMAGE_WIDTH_PIXELS = 100
    if width < MIN_IMAGE_WIDTH_PIXELS:
        return image

    dtype = image.dtype
    is_color = is_color_image(image=image)

    cache_key = (
        height,
        width,
        dtype,
        is_color,
        objective['focal_length'],
        objective['magnification'],
        binning_size,
        color,
    )

    if _scale_bar_cache.get('key') != cache_key:
        overlay, mask, value = _compute_scale_bar_overlay(
            height, width, dtype, is_color, objective, binning_size, color
        )
        _scale_bar_cache = {'key': cache_key, 'overlay': overlay, 'mask': mask, 'value': value}

    cached = _scale_bar_cache
    mask = cached['mask']

    # Write the bar's pixel value at the masked geometry. Works for black
    # (value 0, BF/PC) and white (255 / 4095) alike -- the mask carries the
    # location, so the value can be 0 without erasing the bar.
    image[mask] = cached['value']

    return image


def add_timestamp(image, timestamp_str: str, in_place: bool = True):
    """Draw a timestamp on the image.

    Args:
        image: Input image array (modified in place by default).
        timestamp_str: Text to draw.
        in_place: If True, modify image directly (no copy). If False,
                  work on a copy and return it. Default True to avoid
                  allocating a full-frame copy.

    Returns:
        The image with timestamp drawn.
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

    if not in_place:
        image = image.copy()
    # Ensure array is C-contiguous -- np.flip() produces non-contiguous
    # views that OpenCV rejects with "Layout incompatible with cv::Mat"
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
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
