#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Image-save free functions extracted from `Lumascope`.

These were instance methods on `Lumascope` whose work is file-I/O,
path generation, and metadata-construction -- transformation steps
that take a `scope` (or its discrete settings) and produce a saved
image. They don't belong on the API root, which is the
hardware-composition root. The Lumascope wrappers have been retired;
callers import directly from this module:

    from modules.image_save import save_image
    save_image(scope, array=img, save_folder='./out', ...)

The 5 `*_static` duplicates that previously lived on Lumascope were
retired as dead code -- they had zero external callers; the static
chain existed only because it existed.
"""

from __future__ import annotations

import datetime
import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from lib.handle_trace import tick as _h_tick
from lvp_logger import logger, version
import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.image_mode as image_mode
import modules.image_utils as image_utils
from modules.exceptions import CaptureError, ConfigError
from modules.notification_center import notifications

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


_NUM_SEQ_DIGITS = 6


def write_video_frame(
    frame: np.ndarray,
    file_loc: pathlib.Path,
    metadata: dict,
    channel: str,
    false_color_on: bool,
    save_encoding: str,
    capture_depth: int,
) -> None:
    """Save one captured video frame to TIFF through the single canonical path.

    Shared by the manual-record and protocol-video Frames branches so both
    honor the image mode identically:

      - a false-color-off layer (and any transmitted BF/PC/DF) saves mono, even
        under the RGB encoding -- the mode never colorizes a colorless choice;
      - a 12-bit frame carries significant_bits so msb_aligned left-justifies
        and right_aligned marks the file at its true depth;
      - 8-bit fluorescence false color is baked to RGB here, because the
        video_frame TIFF write emits no palette colormap (unlike the still
        8-bit path).

    Args:
        frame: One captured frame -- mono uint8/uint16, or already-colored RGB.
        file_loc: Output .tiff path.
        metadata: Per-frame metadata dict (must include 'datetime').
        channel: Acquiring channel ('Red'/'Green'/'Blue'/'Lumi'/'BF'/...).
        false_color_on: Whether the layer's false-color toggle was on.
        save_encoding: Resolved image-mode save encoding ('8bit'/'right_aligned'/
            'msb_aligned'/'rgb').
        capture_depth: Acquired bit depth (8 or 12); stamps significant_bits for
            uint16 frames.

    Raises:
        CaptureError: if save_encoding is not a recognized image-mode encoding
            (a typo would otherwise fall through to a plain mono write).
    """
    if save_encoding not in image_mode.VALID_SAVE_ENCODINGS:
        raise CaptureError(
            f'unknown save_encoding {save_encoding!r}; a video frame cannot be saved '
            'with an unrecognized image-mode encoding'
        )
    # Rendering value, permanently: it collapses to 'BF' whenever false color
    # is off, so it stops naming the channel that was imaged. Nothing derived
    # from false_color_on may reach a field describing WHAT the frame is -- a
    # display setting recorded as identity is a claim about the specimen that
    # the file cannot walk back. Video frames carry no channel field today, so
    # nothing structural stops it here; the names are what keep them apart.
    render_color = channel if false_color_on else 'BF'
    # State the payload depth so the file is labeled honestly: an 8-bit frame
    # is 8-bit; a uint16 frame carries its acquired depth (12 for Mono12) so
    # msb_aligned can left-justify and right_aligned marks the true depth; a
    # summed / full-container 16-bit frame is 16.
    if frame.dtype == np.uint8:
        significant_bits = 8
    elif capture_depth and capture_depth < 16:
        significant_bits = capture_depth
    else:
        significant_bits = 16
    if (
        frame.dtype == np.uint8
        and false_color_on
        and not image_utils.is_color_image(frame)
        and render_color in common_utils.get_image_layers()
    ):
        frame = image_utils.add_false_color(array=frame, color=render_color)
    image_utils.write_tiff(
        data=frame,
        file_loc=file_loc,
        metadata=metadata,
        ome=False,
        color=render_color,
        video_frame=True,
        significant_bits=significant_bits,
        save_encoding=save_encoding,
    )


def get_next_save_path(scope: Lumascope, path) -> str:
    """Get the next save path given an existing save path.

    Increments the trailing numeric ID component on the filename and
    returns the new path string.

    Args:
        scope: Kept for signature uniformity with the other image_save
            free functions; not read by this function. Phase 8+ may
            decompose to discrete args.
        path: Path of the format
            ``./{save_folder}/{well_label}_{color}_{file_id}.tiff``.

    Returns:
        str: Next save path with ``file_id`` incremented.
    """
    # Handle both .tiff and .ome.tiff by detecting multiple extensions
    # if present -- pathlib doesn't handle multi-extension stems
    # natively.
    path2 = pathlib.Path(path)
    extension = ''.join(path2.suffixes)
    stem = path2.name[: len(path2.name) - len(extension)]
    seq_separator_idx = stem.rfind('_')
    stem_base = stem[:seq_separator_idx]
    seq_num_str = stem[seq_separator_idx + 1 :]
    seq_num = int(seq_num_str)

    next_seq_num = seq_num + 1
    next_seq_num_str = f'{next_seq_num:0>{_NUM_SEQ_DIGITS}}'

    new_path = path2.parent / f'{stem_base}_{next_seq_num_str}{extension}'
    return str(new_path)


def generate_image_save_path(
    scope: Lumascope,
    save_folder,
    file_root,
    append,
    tail_id_mode,
    output_format,
) -> pathlib.Path:
    """Generate a unique save path for an image given the naming inputs.

    Resolves collisions per ``tail_id_mode`` ("increment" auto-numbers
    until free, "if_collision" only adds a suffix on actual collision,
    ``None`` returns the bare path).

    Args:
        scope: Read for ``motion._last_turret_position`` when
            engineering mode is active. The engineering-mode flag itself
            lives on the app context, not on scope.
        save_folder: Directory to save into (str or Path).
        file_root: Filename prefix.
        append: String appended to filename (e.g. color label).
        tail_id_mode: One of ``"increment"``, ``"if_collision"``, or
            ``None``.
        output_format: ``"TIFF"`` or ``"OME-TIFF"``.

    Returns:
        pathlib.Path: Full save path with appropriate extension and
            disambiguation suffix.

    Raises:
        ConfigError: If ``tail_id_mode`` is not implemented.
    """
    if isinstance(save_folder, str):
        save_folder = pathlib.Path(save_folder)

    if file_root is None:
        file_root = ''

    # Append turret position in engineering mode. ctx may be unset in
    # bare-fixture tests; getattr fallback keeps the default branch.
    engineering_mode = getattr(_app_ctx.ctx, 'engineering_mode', False)
    if engineering_mode and scope.motion._last_turret_position is not None:
        append = f'{append}_T{scope.motion._last_turret_position}'

    if output_format == 'OME-TIFF':
        file_extension = '.ome.tiff'
    elif output_format == 'JPG':
        file_extension = '.jpg'
    else:
        file_extension = '.tiff'

    if tail_id_mode == 'increment':
        initial_id = '_000001'
        filename = f'{file_root}{append}{initial_id}{file_extension}'
        path = save_folder / filename

        # Obtain next save path if current path already exists
        while os.path.exists(path):
            path = get_next_save_path(scope, path)

    elif tail_id_mode == 'if_collision':
        # Write-time defense for duplicate step Names (#636). Use the
        # plain filename when no file exists; only add a numeric
        # suffix on actual collision. Keeps happy-path filenames
        # unchanged for well-formed protocols.
        base_path = save_folder / f'{file_root}{append}{file_extension}'
        if not os.path.exists(base_path):
            path = base_path
        else:
            n = 1
            while True:
                path = save_folder / f'{file_root}{append}_{n:06d}{file_extension}'
                if not os.path.exists(path):
                    break
                n += 1
            logger.warning(
                f'Protocol filename collision: {base_path.name} already '
                f'exists; saving as {path.name} instead. This usually means '
                f'the output folder already holds files from a previous run '
                f'-- capture into a fresh folder, or rename the protocol '
                f'steps so each produces a unique filename.'
            )

    elif tail_id_mode is None:
        filename = f'{file_root}{append}{file_extension}'
        path = save_folder / filename

    else:
        raise ConfigError(f'tail_id_mode: {tail_id_mode} not implemented')

    return path


def generate_image_metadata(scope: Lumascope, channel, x, y, z) -> dict:
    """Build TIFF metadata dict for the current capture settings and position.

    Args:
        scope: Read for objective / labware / stage-offset state,
            coordinate transformer, current camera + LED settings,
            and pending camera chunk metadata.
        channel (str): Channel the frame was acquired on (e.g. "Blue", "BF").
        x (float): Stage X position in um (or None).
        y (float): Stage Y position in um (or None).
        z (float): Stage Z position in um (or None).

    Returns:
        dict: Metadata including channel, positions, exposure, gain, pixel size.

    Raises:
        ConfigError: If objective, labware, or stage offset are not set.
        ValueError: If channel is not a known layer or 'Composite'.
    """
    # This is the last point that can tell a real channel from a placeholder,
    # an index, or a rendering value: past here it is written verbatim and read
    # back forever as a claim about what was imaged. The caller set stops being
    # enumerable the moment this ships, so the vocabulary is checked, not assumed.
    if channel not in common_utils.get_layers() and channel != 'Composite':
        raise ValueError(
            f'unknown channel {channel!r} for image metadata; expected one of '
            f"{common_utils.get_layers()} or 'Composite'"
        )

    objective = scope.runtime_state.get_current_objective()
    if objective is None:
        raise ConfigError('[SCOPE API ] Objective not set')

    if 'focal_length' not in objective:
        raise ConfigError('[SCOPE API ] Objective focal length not provided')

    labware = scope.runtime_state.get_labware()
    if labware is None:
        raise ConfigError('[SCOPE API ] Labware not set')

    if scope.runtime_state.get_stage_offset() is None:
        raise ConfigError('[SCOPE API ] Stage offset not set')

    if x is None:
        x = 0
    if y is None:
        y = 0
    if z is None:
        z = 0

    px, py = scope.runtime_state.stage_to_plate(sx=x, sy=y)
    well_label = scope.runtime_state.get_well_label()

    px = round(px, common_utils.max_decimal_precision('x'))
    py = round(py, common_utils.max_decimal_precision('y'))
    z = round(z, common_utils.max_decimal_precision('z'))

    pixel_size_um = common_utils.get_pixel_size(
        focal_length=objective['focal_length'],
        binning_size=scope.imaging._binning_size,
    )
    # A scope with no known pixel size writes no scale rather than an invented
    # one -- the writer omits PhysicalSizeX and the resolution tag when this is
    # None. A wrong scale is measured off the file forever and cannot be told
    # from a real one.
    if pixel_size_um is not None:
        pixel_size_um = round(
            pixel_size_um,
            common_utils.max_decimal_precision('pixel_size'),
        )

    now_host = datetime.datetime.now()
    microscope_model = scope.diagnostics.get_microscope_model()

    # Instrument + Plate metadata for OME-XML compatibility (#491).
    # Sourced from diagnostics + runtime_state; failures are non-fatal
    # since the per-image save must not block on diagnostics flakes.
    try:
        motor_info = scope.diagnostics.get_motor_info()
    except Exception:
        motor_info = {'serial_number': None, 'firmware_version': None}
    try:
        camera_info = scope.diagnostics.get_camera_info()
    except Exception:
        camera_info = {'model': None}
    plate_config = getattr(labware, 'config', None) or {}

    # Per-frame camera chunk metadata, captured at grab-time for THIS frame
    # (Pylon ace 2 / dart M / dart R carry ExposureTime + Gain + FrameID +
    # Timestamp every frame). These are the same chunk values frame_validity
    # checks the camera settled to, so they are the authoritative, race-free
    # source for the frame's gain/exposure metadata. The live
    # get_exposure_ms / get_gain_db calls are the fallback for cameras /
    # frames without chunk data (IDS stores frames without chunks; also
    # simulator and legacy). Both sources report what the hardware is
    # ACTUALLY set to, never the requested value -- so even if a settings
    # write silently failed, the recorded metadata stays truthful.
    try:
        handler = getattr(scope._camera_driver, 'cam_image_handler', None)
        chunks = handler.get_last_chunks() if handler is not None else None
    except Exception:
        chunks = None
    chunks = chunks or {}

    _chunk_exp_us = chunks.get('ExposureTime')
    _chunk_gain_db = chunks.get('Gain')
    # The live-confirmed surface, not get_gain_db()/get_exposure_ms(): the
    # value getters answer last-known-good on a failed read, which is
    # right for control flow but would record a gain/exposure this frame
    # was not captured at. get_live_camera_settings omits a field whose
    # read did not just succeed, so unknown stays unknown here.
    if _chunk_exp_us is None or _chunk_gain_db is None:
        _live_settings = scope.imaging.get_live_camera_settings()
    else:
        _live_settings = {}
    exposure_ms_value = (
        _chunk_exp_us / 1000.0 if _chunk_exp_us is not None else _live_settings.get('exposure_ms')
    )
    gain_db_value = _chunk_gain_db if _chunk_gain_db is not None else _live_settings.get('gain_db')

    # A non-physical gain / exposure (negative failed-read sentinel, or the
    # zero exposure an inactive camera reports) is not a real setting.
    # Unknown stays unknown: omit the key from the saved metadata rather
    # than record a value the hardware never had -- a -1.0 or 0.0 in
    # OME/TIFF metadata reads downstream as a real acquisition setting.
    _frame_settings = {}
    if common_utils.is_valid_exposure_ms(exposure_ms_value):
        _frame_settings['exposure_time_ms'] = round(
            exposure_ms_value, common_utils.max_decimal_precision('exposure')
        )
    else:
        logger.warning(
            'Exposure time for this frame is unknown (no chunk data and the '
            'live camera read failed or the camera is inactive); omitting '
            'exposure_time_ms from saved metadata'
        )
    if common_utils.is_valid_gain_db(gain_db_value):
        _frame_settings['gain_db'] = round(
            gain_db_value, common_utils.max_decimal_precision('gain')
        )
    else:
        logger.warning(
            'Gain for this frame is unknown (no chunk data and the live '
            'camera read failed); omitting gain_db from saved metadata'
        )

    # Spectral identity from the resolved layer record. Written whatever
    # rung identity resolved from, and a null value is ABSENT: broadband
    # layers have no excitation, 'Composite' has no record, an unresolved
    # identity has no filterset -- a null or stand-in written here would
    # read downstream as a measured property of the capture. The board
    # address (led_channel) deliberately stays out of metadata: it changes
    # with a rewire, board swap, or motorconfig regeneration, so recording
    # it would version the files to the wiring.
    identity = scope.layer_identity
    record = identity.find(channel)

    metadata = {
        'camera_make': 'Etaluma',
        'microscope': microscope_model,
        'microscope_model': microscope_model,
        'software': f'LumaViewPro {version}',
        'channel': channel,
        **({'channel_display': record.display_name} if record is not None else {}),
        **(
            {'excitation_nm': record.excitation_nm}
            if record is not None and record.excitation_nm is not None
            else {}
        ),
        **({'filterset': identity.filterset} if identity.filterset else {}),
        'datetime': now_host.strftime('%Y:%m:%d %H:%M:%S'),
        'sub_sec_time': f'{now_host.microsecond // 1000:03d}',
        'objective': objective,
        'focal_length': objective['focal_length'],
        'plate_pos_mm': {'x': px, 'y': py},
        'x_pos': px,
        'y_pos': py,
        'z_pos_um': z,
        **_frame_settings,
        # An LED that is off, never set, or on an absent board has no
        # drive current -- a normal state for dark and luminescence
        # captures, so the key is simply absent (no warning, unlike the
        # exposure/gain omissions above, which indicate a failed read).
        **(
            {'illumination_ma': round(_ma, common_utils.max_decimal_precision('illumination'))}
            if (_ma := scope.illumination.get_led_ma(channel=channel)) is not None
            else {}
        ),
        'binning_size': scope.imaging._binning_size,
        'pixel_size_um': pixel_size_um,
        # Zero-well labware (Blank) has no well: omit the key rather than
        # stamp an empty or fabricated label, mirroring the no-scale idiom
        # above -- a fake well is measured off the file forever.
        **({'well_label': well_label} if well_label else {}),
        'timestamp_iso': now_host.isoformat(timespec='microseconds'),
        'instrument': {
            'manufacturer': 'Etaluma',
            'model': microscope_model,
            'serial_number': motor_info.get('serial_number'),
            'firmware_version': motor_info.get('firmware_version'),
            'camera_model': camera_info.get('model'),
        },
        'plate': {
            'name': microscope_model or 'Plate',
            'rows': plate_config.get('rows'),
            'columns': plate_config.get('columns'),
            'standard': plate_config.get('standard'),
        },
    }

    # Camera-side timestamp + frame-id provenance from the same grab-time
    # chunk read above (Pylon ace 2 / dart M / dart R carry ChunkTimestamp;
    # IDS has ExposureTime/Gain but no ChunkTimestamp yet -- Stage 2 work).
    ts_ticks = chunks.get('Timestamp')
    if ts_ticks is not None:
        metadata['timestamp_camera_ticks'] = int(ts_ticks)
    tick_hz = getattr(scope._camera_driver, 'timestamp_tick_frequency_hz', None)
    if tick_hz is not None:
        metadata['timestamp_camera_tick_hz'] = int(tick_hz)
    frame_id = chunks.get('FrameID')
    if frame_id is not None:
        metadata['frame_id'] = int(frame_id)

    return metadata


def _apply_save_orientation(array):
    """Flip the camera array to the canonical save orientation.

    Single source of the save-time orientation convention, applied
    identically to every output format (TIFF, OME-TIFF, JPG) so they
    can never diverge. The camera delivers rows top-to-bottom; the
    saved image -- and the stitching / tiling pipeline that reads it
    back -- expects the vertically-flipped view.
    """
    return np.flip(array, 0)


def prepare_image_for_saving(
    scope: Lumascope,
    array: np.ndarray,
    save_folder: str,
    file_root: str,
    append: str,
    tail_id_mode: str,
    output_format: str,
    x,
    y,
    z,
    *,
    channel: str,
    significant_bits: int,
) -> dict:
    """Prepare an image array and metadata for saving to disk.

    Flips the image vertically, records the payload bit depth, and generates
    the save path and metadata. Pixel values are stored raw (right-aligned) --
    a 12-bit frame is saved as 0..4095, not left-justified to 0..65520 -- with
    the true depth carried in the SignificantBits tag instead.

    Args:
        scope: Passed to generate_image_metadata + generate_image_save_path.
        array: Raw image array from drivers.
        save_folder: Directory to save into.
        file_root: Filename prefix.
        append: String appended to filename (e.g. channel label).
        tail_id_mode: "increment" for auto-numbered files, or None.
        output_format: "TIFF" or "OME-TIFF".
        x: Stage X position in um.
        y: Stage Y position in um.
        z: Stage Z position in um.
        channel: Channel the frame was acquired on. Required and keyword-only:
            it is the sole durable carrier of channel identity, so a save that
            never states its channel must not be constructible. It is
            independent of how the frame is displayed.
        significant_bits: Payload depth ``array`` was captured at, recorded in
            the SignificantBits tag. Required: the caller passes the depth it
            captured the frame at (8 for a uint8 frame, the native depth for a
            single wider frame, 16 for a summed 16-bit container) -- a save
            cannot re-derive it from the camera's live state, which may already
            describe a newer format.

    Returns:
        dict: Contains 'image' (ndarray) and 'metadata' (dict with 'file_loc').
    """
    metadata = generate_image_metadata(scope, channel=channel, x=x, y=y, z=z)

    metadata['significant_bits'] = significant_bits

    array = _apply_save_orientation(array)

    path = generate_image_save_path(
        scope,
        save_folder=save_folder,
        file_root=file_root,
        append=append,
        tail_id_mode=tail_id_mode,
        output_format=output_format,
    )

    metadata['file_loc'] = path

    return {
        'image': array,
        'metadata': metadata,
    }


def save_image(
    scope: Lumascope,
    array,
    save_folder='./capture',
    file_root='img_',
    append='ms',
    tail_id_mode='increment',
    *,
    channel: str,
    false_color_on: bool,
    save_encoding: str,
    output_format: str = 'TIFF',
    x=None,
    y=None,
    z=None,
    false_color_buf: np.ndarray | None = None,
    rgb_buf: np.ndarray | None = None,
    jpeg_quality: int = 90,
    significant_bits: int,
) -> str:
    """Save an image array to a TIFF file with metadata.

    Args:
        scope: Passed to prepare_image_for_saving for path / metadata.
        array: Image array to save.
        save_folder: Directory to save into.
        file_root: Filename prefix.
        append: String appended to filename.
        tail_id_mode: "increment" for auto-numbered files, or None.
        channel: Channel the frame was acquired on -- what the image IS.
            Required and keyword-only; recorded as the file's identity and
            never altered by a display setting.
        false_color_on: Whether the channel's false-color toggle was on --
            how the image is DISPLAYED. Required and keyword-only; drives the
            colormap and the JPG bake, and reaches no identity field.
        output_format: "TIFF", "OME-TIFF", or "JPG".
        x: Stage X position in um.
        y: Stage Y position in um.
        z: Stage Z position in um.
        save_encoding: The derived on-disk encoding from the image_mode
            config layer (rgb / msb_aligned / right_aligned / 8bit). Required
            and keyword-only: it is the single value that drives the save
            shape, so no call site can omit the image mode and silently store
            a scaled payload right-aligned (dark).
        false_color_buf: Preallocated false-color buffer.
        rgb_buf: Preallocated RGB buffer.
        jpeg_quality: JPEG quality 1-100, used only when output_format
            is "JPG".

    Returns:
        str: Path to the saved file.

    Raises:
        CaptureError: If ``array`` is None (camera silent-stuck or
            grab-timeout). Surfaces to IOTask popup with a user-friendly
            message instead of a raw AttributeError downstream.
    """
    # Camera silent-stuck or grab-timeout produces None; raise typed
    # exception so the IOTask popup carries a user-friendly message
    # instead of a raw AttributeError. The deeper recovery work
    # (camera reset / USB reset on persistent stuck) lives elsewhere.
    if array is None:
        raise CaptureError(
            'Camera did not return an image. The capture was skipped; '
            'the protocol will retry on the next step.'
        )

    # The one place the two facts meet, and they meet on the rendering side
    # only. A channel with false color off is drawn like brightfield; it is
    # still that channel, and the metadata below says so.
    render_color = channel if false_color_on else 'BF'

    if output_format == 'JPG':
        # JPG is a convenience / sharing export: no metadata is embedded
        # and the pixels come from the raw array (to match the live
        # preview), so skip prepare_image_for_saving. Its
        # generate_image_metadata step requires objective / labware to be
        # configured, which an ad-hoc JPG snapshot taken before a protocol
        # is set up has no reason to need. Only the save path is required.
        image = None
        metadata = None
        file_loc = generate_image_save_path(
            scope,
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            tail_id_mode=tail_id_mode,
            output_format=output_format,
        )
    else:
        image_data = prepare_image_for_saving(
            scope,
            array=array,
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            tail_id_mode=tail_id_mode,
            output_format=output_format,
            channel=channel,
            x=x,
            y=y,
            z=z,
            significant_bits=significant_bits,
        )
        image = image_data['image']
        metadata = image_data['metadata']
        file_loc = metadata['file_loc']

    ome = output_format == 'OME-TIFF'

    try:
        if output_format == 'JPG':
            # Convenience / sharing export: bake the displayed channel
            # color into 8-bit pixels and write a JPEG. Shares the
            # orientation convention with the TIFF path via
            # _apply_save_orientation (the only step both formats have in
            # common); without it the JPG saved upside-down relative to the
            # TIFFs and the tiling pipeline that reads them. Bit depth,
            # color baking, and metadata are format-specific: JPG is an
            # 8-bit rendered display image, TIFF / OME-TIFF carry the
            # 16-bit data + metadata.
            jpg_bytes = image_utils.encode_display_jpg(
                _apply_save_orientation(array),
                render_color,
                significant_bits=significant_bits,
                jpeg_quality=jpeg_quality,
            )
            pathlib.Path(file_loc).write_bytes(jpg_bytes)
        else:
            image_utils.write_tiff(
                data=image,
                file_loc=file_loc,
                metadata=metadata,
                ome=ome,
                color=render_color,
                significant_bits=metadata['significant_bits'],
                save_encoding=save_encoding,
                false_color_buf=false_color_buf,
                rgb_buf=rgb_buf,
            )

        logger.debug(f'[SCOPE API ] Saving Image to {file_loc}')
    except Exception:
        logger.exception('[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist?')
        notifications.error(
            'FileIO',
            'Image Save Failed',
            f'Failed to save image to {file_loc}. Check disk space and permissions.',
        )
        raise

    # Handle-leak tracking; zero overhead when disabled. Enable via the
    # profiling.handle_trace_enabled setting.
    _h_tick('save_image')

    return file_loc


def save_live_image(
    scope: Lumascope,
    save_folder: str | pathlib.Path = './capture',
    file_root: str = 'img_',
    append: str = 'ms',
    tail_id_mode: str | None = 'increment',
    force_to_8bit: bool = True,
    output_format: str = 'TIFF',
    earliest_image_ts: datetime.datetime | None = None,
    timeout_s: float = 5.0,
    all_ones_check: bool = False,
    sum_count: int = 1,
    sum_delay_s: float = 0,
    sum_iteration_callback: Callable[..., None] | None = None,
    turn_off_all_leds_after: bool = False,
    use_executor: bool = False,
    jpeg_quality: int = 90,
    *,
    channel: str,
    false_color_on: bool,
    save_encoding: str,
) -> str | None:
    """Grab the current live image from the camera and save to a TIFF file.

    Combines capture_and_wait() and save_image() in one call. Optionally
    turns off all LEDs after capture.

    Args:
        scope: Source of imaging.capture_and_wait + illumination.leds_off.
        save_folder: Directory to save into.
        file_root: Filename prefix.
        append: String appended to filename.
        tail_id_mode: "increment" for auto-numbered files, or None.
        force_to_8bit: Convert 12-bit images to 8-bit.
        output_format: "TIFF" or "OME-TIFF".
        earliest_image_ts: Reject frames before this timestamp.
        timeout_s: Max seconds to wait for a valid frame.
        all_ones_check: Reject saturated frames.
        sum_count: Number of frames to sum.
        sum_delay_s: Delay between summed frames.
        sum_iteration_callback: Called after each summed frame.
        turn_off_all_leds_after: Turn off all LEDs after capture.
        use_executor: Reserved for future use.
        jpeg_quality: JPEG quality 1-100, used only when output_format
            is "JPG".
        channel: Channel the frame was acquired on, forwarded to save_image as
            the file's identity. Required and keyword-only.
        false_color_on: Whether the channel's false-color toggle was on,
            forwarded as the rendering choice. Required and keyword-only.
        save_encoding: The derived on-disk encoding from the image_mode
            config layer; required and keyword-only, forwarded to save_image
            so the live-capture path cannot drop the image mode.
    Returns:
        str | None: Path to saved file, or None on failure.
    """
    try:
        array = scope.imaging._capture_and_wait_impl(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            timeout_s=timeout_s,
            all_ones_check=all_ones_check,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
        )
    finally:
        # The off must hold even when the capture raises -- a caller that
        # asked for it is relying on this call to end illumination.
        if turn_off_all_leds_after:
            scope.illumination._leds_off_impl()

    if array is None:
        return None

    # Depth resolved here, right after the capture that produced the frame
    # (uint8 -> 8, summed -> 16, else the per-frame delivery stamp), and
    # handed down with it -- the shared capture-time depth rule.
    significant_bits = scope.imaging.capture_frame_depth(array, sum_count)

    path = save_image(
        scope,
        array,
        save_folder=save_folder,
        file_root=file_root,
        append=append,
        tail_id_mode=tail_id_mode,
        channel=channel,
        false_color_on=false_color_on,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        significant_bits=significant_bits,
        save_encoding=save_encoding,
    )

    # Record what the manual capture actually wrote, so a saved-file bundle is
    # self-describing. Report the sensor's acquired depth AND the depth stamped
    # on the file separately: a scaled encoding left-justifies a 12-bit capture
    # to fill the 16-bit container, so the file is 16-bit while the sensor gave
    # 12. Reporting only the acquired depth read as if the file were mis-tagged.
    saved_significant_bits = image_utils.written_significant_bits(
        save_encoding, significant_bits, array.dtype, image_utils.is_color_image(array)
    )
    logger.info(
        f'[ImageSave] manual capture encoding={save_encoding} '
        f'capture_bits={significant_bits} saved_significant_bits={saved_significant_bits} '
        f'dtype={array.dtype} shape={array.shape} -> {pathlib.Path(path).name}'
    )
    return path
