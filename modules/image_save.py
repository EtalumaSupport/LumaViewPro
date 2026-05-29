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
from typing import TYPE_CHECKING

import numpy as np

from lib.handle_trace import tick as _h_tick
from lvp_logger import logger, version
import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.exceptions import CaptureError, ConfigError
from modules.notification_center import notifications

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


_NUM_SEQ_DIGITS = 6


def get_next_save_path(scope: 'Lumascope', path) -> str:
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
    scope: 'Lumascope',
    save_folder,
    file_root,
    append,
    tail_id_mode,
    output_format,
) -> 'pathlib.Path':
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
                f'your protocol has multiple steps that produce the same '
                f'filename (same Name + Well + Tile + Z-Slice across '
                f'different Tile Group IDs). Consider including the Tile '
                f'Group ID in the step Name field to avoid the rename suffix.'
            )

    elif tail_id_mode is None:
        filename = f'{file_root}{append}{file_extension}'
        path = save_folder / filename

    else:
        raise ConfigError(f'tail_id_mode: {tail_id_mode} not implemented')

    return path


def generate_image_metadata(scope: 'Lumascope', color, x, y, z) -> dict:
    """Build TIFF metadata dict for the current capture settings and position.

    Args:
        scope: Read for objective / labware / stage-offset state,
            coordinate transformer, current camera + LED settings,
            and pending camera chunk metadata.
        color (str): Channel color name (e.g. "Blue", "BF").
        x (float): Stage X position in um (or None).
        y (float): Stage Y position in um (or None).
        z (float): Stage Z position in um (or None).

    Returns:
        dict: Metadata including channel, positions, exposure, gain, pixel size.

    Raises:
        ConfigError: If objective, labware, or stage offset are not set.
    """
    if scope.runtime_state._objective is None:
        raise ConfigError('[SCOPE API ] Objective not set')

    if 'focal_length' not in scope.runtime_state._objective:
        raise ConfigError('[SCOPE API ] Objective focal length not provided')

    if scope.runtime_state._labware is None:
        raise ConfigError('[SCOPE API ] Labware not set')

    if scope.runtime_state._stage_offset is None:
        raise ConfigError('[SCOPE API ] Stage offset not set')

    if x is None:
        x = 0
    if y is None:
        y = 0
    if z is None:
        z = 0

    px, py = scope.runtime_state._coordinate_transformer.stage_to_plate(
        labware=scope.runtime_state._labware,
        stage_offset=scope.runtime_state._stage_offset,
        sx=x,
        sy=y,
    )
    well_label = scope.runtime_state.get_well_label()

    px = round(px, common_utils.max_decimal_precision('x'))
    py = round(py, common_utils.max_decimal_precision('y'))
    z = round(z, common_utils.max_decimal_precision('z'))

    pixel_size_um = round(
        common_utils.get_pixel_size(
            focal_length=scope.runtime_state._objective['focal_length'],
            binning_size=scope.imaging._binning_size,
        ),
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
    plate_config = getattr(scope.runtime_state._labware, 'config', None) or {}

    metadata = {
        'camera_make': 'Etaluma',
        'microscope': microscope_model,
        'microscope_model': microscope_model,
        'software': f'LumaViewPro {version}',
        'channel': color,
        'datetime': now_host.strftime('%Y:%m:%d %H:%M:%S'),
        'sub_sec_time': f'{now_host.microsecond // 1000:03d}',
        'objective': scope.runtime_state._objective,
        'focal_length': scope.runtime_state._objective['focal_length'],
        'plate_pos_mm': {'x': px, 'y': py},
        'x_pos': px,
        'y_pos': py,
        'z_pos_um': z,
        'exposure_time_ms': round(
            scope.imaging.get_exposure_time(), common_utils.max_decimal_precision('exposure')
        ),
        'gain_db': round(scope.imaging.get_gain(), common_utils.max_decimal_precision('gain')),
        'illumination_ma': (
            round(_ma, common_utils.max_decimal_precision('illumination'))
            if (_ma := scope.illumination.get_led_ma(color=color)) is not None
            else 0
        ),
        'binning_size': scope.imaging._binning_size,
        'pixel_size_um': pixel_size_um,
        'well_label': well_label,
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

    # Camera-side timestamp + frame-id provenance, when the camera
    # supports chunk data (Pylon ace 2 / dart M / dart R always; IDS
    # has ExposureTime/Gain but no ChunkTimestamp yet -- Stage 2 work).
    # Read the most recent chunks; they're captured at-grab-time and
    # are the right values for the most recent frame on this thread.
    try:
        handler = getattr(scope._camera_driver, 'cam_image_handler', None)
        chunks = handler.get_last_chunks() if handler is not None else None
    except Exception:
        chunks = None
    if chunks is not None:
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


def prepare_image_for_saving(
    scope: 'Lumascope',
    array: np.ndarray,
    save_folder: str,
    file_root: str,
    append: str,
    color: str,
    tail_id_mode: str,
    output_format: str,
    true_color: str,
    x,
    y,
    z,
    out_12to16: np.ndarray | None = None,
) -> dict:
    """Prepare an image array and metadata for saving to disk.

    Flips the image vertically, converts bit depth if needed, generates
    the save path and metadata.

    Args:
        scope: Passed to generate_image_metadata + generate_image_save_path.
        array: Raw image array from drivers.
        save_folder: Directory to save into.
        file_root: Filename prefix.
        append: String appended to filename (e.g. color label).
        color: Color label for the filename.
        tail_id_mode: "increment" for auto-numbered files, or None.
        output_format: "TIFF" or "OME-TIFF".
        true_color: Actual channel color for metadata.
        x: Stage X position in um.
        y: Stage Y position in um.
        z: Stage Z position in um.
        out_12to16: Optional preallocated buffer for 12-to-16-bit
            conversion (avoids per-frame allocation in the hot path).

    Returns:
        dict: Contains 'image' (ndarray) and 'metadata' (dict with 'file_loc').
    """
    metadata = generate_image_metadata(scope, color=true_color, x=x, y=y, z=z)

    if array.dtype == np.uint16:
        array = image_utils.convert_12bit_to_16bit(array, out=out_12to16)

    array = np.flip(array, 0)

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
    scope: 'Lumascope',
    array,
    save_folder='./capture',
    file_root='img_',
    append='ms',
    color='BF',
    tail_id_mode='increment',
    output_format: str = 'TIFF',
    true_color: str = 'BF',
    x=None,
    y=None,
    z=None,
    use_false_color_16bit: bool | None = None,
    out_12to16: np.ndarray | None = None,
    false_color_buf: np.ndarray | None = None,
    rgb_buf: np.ndarray | None = None,
    jpeg_quality: int = 90,
) -> str:
    """Save an image array to a TIFF file with metadata.

    Args:
        scope: Passed to prepare_image_for_saving for path / metadata.
        array: Image array to save.
        save_folder: Directory to save into.
        file_root: Filename prefix.
        append: String appended to filename.
        color: Color label for the filename.
        tail_id_mode: "increment" for auto-numbered files, or None.
        output_format: "TIFF", "OME-TIFF", or "JPG".
        true_color: Actual channel color for metadata.
        x: Stage X position in um.
        y: Stage Y position in um.
        z: Stage Z position in um.
        use_false_color_16bit: Pre-resolved bool from sequenced_capture_runner;
            None falls back to image_utils.write_tiff's settings-lock read
            path (preserves behavior for ad-hoc callers).
        out_12to16: Preallocated 12-to-16-bit conversion buffer.
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
            color=color,
            tail_id_mode=tail_id_mode,
            output_format=output_format,
            true_color=true_color,
            x=x,
            y=y,
            z=z,
            out_12to16=out_12to16,
        )
        image = image_data['image']
        metadata = image_data['metadata']
        file_loc = metadata['file_loc']

    ome = output_format == 'OME-TIFF'

    try:
        if output_format == 'JPG':
            # Convenience / sharing export: bake the displayed channel
            # color into 8-bit pixels and write a JPEG. Encode from the
            # raw camera array (not the TIFF-prepared image) so the JPG
            # matches the live preview. No metadata is embedded; TIFF /
            # OME-TIFF remain the metadata-bearing scientific formats.
            jpg_bytes = image_utils.encode_display_jpg(
                array, color, jpeg_quality=jpeg_quality
            )
            pathlib.Path(file_loc).write_bytes(jpg_bytes)
        else:
            image_utils.write_tiff(
                data=image,
                file_loc=file_loc,
                metadata=metadata,
                ome=ome,
                color=color,
                use_false_color_16bit=use_false_color_16bit,
                false_color_buf=false_color_buf,
                rgb_buf=rgb_buf,
            )

        logger.info(f'[SCOPE API ] Saving Image to {file_loc}')
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
    scope: 'Lumascope',
    save_folder='./capture',
    file_root='img_',
    append='ms',
    color='BF',
    tail_id_mode='increment',
    force_to_8bit: bool = True,
    output_format: str = 'TIFF',
    true_color: str = 'BF',
    earliest_image_ts: datetime.datetime | None = None,
    timeout_s: float = 5.0,
    all_ones_check: bool = False,
    sum_count: int = 1,
    sum_delay_s: float = 0,
    sum_iteration_callback=None,
    turn_off_all_leds_after: bool = False,
    use_executor: bool = False,
    jpeg_quality: int = 90,
) -> str | None:
    """Grab the current live image from the camera and save to a TIFF file.

    Combines capture_and_wait() and save_image() in one call. Optionally
    turns off all LEDs after capture.

    Args:
        scope: Source of imaging.capture_and_wait + illumination.leds_off.
        save_folder: Directory to save into.
        file_root: Filename prefix.
        append: String appended to filename.
        color: Color label for the filename.
        tail_id_mode: "increment" for auto-numbered files, or None.
        force_to_8bit: Convert 12-bit images to 8-bit.
        output_format: "TIFF" or "OME-TIFF".
        true_color: Actual channel color for metadata.
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

    Returns:
        str | None: Path to saved file, or None on failure.
    """
    array = scope.imaging.capture_and_wait(
        force_to_8bit=force_to_8bit,
        earliest_image_ts=earliest_image_ts,
        timeout_s=timeout_s,
        all_ones_check=all_ones_check,
        sum_count=sum_count,
        sum_delay_s=sum_delay_s,
        sum_iteration_callback=sum_iteration_callback,
    )

    if turn_off_all_leds_after:
        scope.illumination.leds_off()

    if array is False:
        return None

    return save_image(
        scope,
        array,
        save_folder,
        file_root,
        append,
        color,
        tail_id_mode,
        output_format=output_format,
        true_color=true_color,
        jpeg_quality=jpeg_quality,
    )
