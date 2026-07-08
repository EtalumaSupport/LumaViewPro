# Copyright Etaluma, Inc.
"""Finalize a finished manual recording into TIFF frames or an MP4.

Extracted from ``MainDisplay.recording_complete`` so the write/finalize
logic can run (and be tested) off the Kivy widget. It owns no GUI state:
the live ``scope`` and an optional progress callback are passed in, and
failures surface through the notification center the same way the protocol
video path does. The caller runs this on the file_io_executor.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

import modules.image_save as image_save
from modules.notification_center import notifications
from modules.recording_manifest import build_session_manifest
from modules.stack_builder import StackBuilder
from modules.video_writer import VideoWriter, fps_from_frames

logger = logging.getLogger('LVP.modules.manual_video_finalize')


def finalize_manual_video(
    *,
    captured_frames: int,
    timestamps: list,
    chunks_per_frame: list,
    tick_freq_hz: float | None,
    video_frames: Any,
    video_duration: float,
    video_save_folder: Path | None,
    start_time_str: str,
    video_as_frames: bool,
    memmap_path: str | None,
    video_false_color: str | None,
    ui_snapshot: dict,
    scope: Any = None,
    progress_cb: Callable[[int], None] | None = None,
) -> str | None:
    """Write a finished manual recording to disk and return the memmap path.

    Args:
        captured_frames: Number of frames actually captured.
        timestamps: Per-frame capture timestamps (datetime).
        chunks_per_frame: Per-frame camera chunk metadata (Timestamp/FrameID).
        tick_freq_hz: Camera timestamp tick frequency, if known.
        video_frames: The frame source (a memmap or array) to read slots from.
        video_duration: Recording wall-clock duration in seconds.
        video_save_folder: Destination folder for TIFFs / MP4.
        start_time_str: Recording start stamp used in the MP4 filename.
        video_as_frames: True saves a TIFF-per-frame folder; False writes one MP4.
        memmap_path: Scratch memmap path, returned so the caller can release it.
        video_false_color: Layer false-color name, or None for grayscale.
        ui_snapshot: Main-thread snapshot of layer/capture/objective config.
        scope: Live Lumascope (for turret/position/camera-model metadata).
        progress_cb: Optional callback invoked with the 1-based frames-written
            count for UI progress; None disables progress reporting.

    Returns:
        The memmap_path passed in, so the caller's cleanup can untrack it.
    """

    def _report_progress(value: int) -> None:
        if progress_cb is not None:
            progress_cb(value)

    try:
        # Defensive check
        if video_frames is None:
            logger.error('Manual-Video] recording_complete called with no video frames')
            return memmap_path

        # Prevent division by zero
        if video_duration <= 0:
            video_duration = 0.1
            logger.warning('Manual-Video] Video duration was 0, using 0.1s')

        if captured_frames == 0:
            logger.error('Manual-Video] No frames captured, aborting video write')
            return memmap_path

        calculated_fps = fps_from_frames(captured_frames, video_duration)

        logger.info(
            f'Manual-Video] Images present in video array: {len(video_frames) > 0 if video_frames is not None else 0}'
        )
        logger.info(f'Manual-Video] Captured Frames: {captured_frames}')
        logger.info(f'Manual-Video] Video FPS: {calculated_fps}')
        logger.info('Manual-Video] Writing video...')

        # A finished recording is finalized entirely from its main-thread
        # config snapshot. The snapshot stores None for any field whose widget
        # read raised (e.g. active_layer_config raises when no layer accordion
        # is open); consuming a None field used to discard the recording behind
        # a single log line. Validate the fields THIS output path needs, once,
        # up front, so a newly added snapshot field cannot reintroduce that
        # silent discard by forgetting its own guard downstream.
        required_fields = ['active_layer_config']
        if video_as_frames:
            required_fields.append('image_capture_config')
            # OME-TIFF Hyperstack additionally reads the objective snapshot;
            # that need is only knowable once image_capture_config is present.
            capture_config = ui_snapshot.get('image_capture_config')
            if (
                capture_config is not None
                and capture_config.output_format_sequenced == 'OME-TIFF Hyperstack'
            ):
                required_fields.append('objective_info')
        missing = [field for field in required_fields if ui_snapshot.get(field) is None]
        if missing:
            logger.error(
                f'Manual-Video] Incomplete capture snapshot (missing {missing}); '
                'cannot finalize recording'
            )
            notifications.error(
                'Recording',
                'Recording Not Saved',
                'The recording could not be saved because the capture settings '
                'could not be read. Make sure an imaging layer is selected, then '
                'record again.',
            )
            return memmap_path

        color, _active_layer_config = ui_snapshot['active_layer_config']

        include_hyperstack_generation = False

        # Frames the chosen writer accepted but could not save. Both manual
        # write paths swallow a per-frame failure and continue (a short video
        # still beats none), then report the total once at the end -- the same
        # contract the protocol video path honors.
        dropped_frames = 0

        if video_as_frames:
            image_capture_config = ui_snapshot['image_capture_config']

            if image_capture_config.output_format_sequenced == 'OME-TIFF Hyperstack':
                include_hyperstack_generation = True
                _, objective = ui_snapshot['objective_info']
                stack_builder = StackBuilder(
                    has_turret=scope.motion.has_turret(),
                )
                frame_metadata = []

            save_folder = video_save_folder

            if not save_folder.exists():
                save_folder.mkdir(exist_ok=True, parents=True)

            for frame_num in range(captured_frames):
                image = video_frames[frame_num]
                ts = (
                    timestamps[frame_num]
                    if frame_num < len(timestamps)
                    else datetime.datetime.now()
                )
                # Filename includes per-frame timestamp so the folder is
                # browsable without a viewer that reads TIFF metadata.
                # Colon-free ISO variant for Windows path-safety; millisecond
                # precision. The timestamp is not drawn into the pixels here:
                # it travels in the frame metadata, and Create Video draws it
                # at build time only when the timestamp overlay is enabled.
                ts_filename = ts.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
                frame_name = f'ManualVideo_Frame_{frame_num:04}_{ts_filename}'

                output_file_loc = save_folder / f'{frame_name}.tiff'

                # Per-frame timestamp metadata. The 'datetime' / 'timestamp' /
                # 'frame_num' keys are kept for backward compatibility with
                # downstream readers that look for them. The
                # 'timestamp_iso' / 'timestamp_camera_ticks' /
                # 'timestamp_camera_tick_hz' / 'frame_id' keys mirror the
                # structured Plane fields used elsewhere; the video_frame
                # TIFF path serializes them into the description tag.
                metadata = {
                    'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
                    'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
                    'timestamp_iso': ts.isoformat(timespec='microseconds'),
                    'frame_num': frame_num,
                }
                chunks = chunks_per_frame[frame_num] if frame_num < len(chunks_per_frame) else None
                if chunks is not None:
                    ts_ticks = chunks.get('Timestamp')
                    if ts_ticks is not None:
                        metadata['timestamp_camera_ticks'] = int(ts_ticks)
                    if tick_freq_hz is not None:
                        metadata['timestamp_camera_tick_hz'] = int(tick_freq_hz)
                    frame_id = chunks.get('FrameID')
                    if frame_id is not None:
                        metadata['frame_id'] = int(frame_id)

                if include_hyperstack_generation:
                    current_position = scope.motion.get_current_position()
                    frame_metadata.append(
                        {
                            'Filepath': output_file_loc.name,
                            'Scan Count': frame_num,
                            'Color': color,
                            'Z-Slice': 0,
                            'X': current_position['X'],
                            'Y': current_position['Y'],
                            'Z': current_position['Z'],
                        }
                    )

                try:
                    image_save.write_video_frame(
                        frame=image,
                        file_loc=output_file_loc,
                        metadata=metadata,
                        layer_color=color,
                        false_color_on=video_false_color is not None,
                        save_encoding=image_capture_config.save_encoding,
                        capture_depth=image_capture_config.capture_depth,
                    )
                except Exception as e:
                    logger.exception(f'Manual-Video] Failed to write frame {frame_num}: {e}')
                    dropped_frames += 1

                # Update progress on main thread
                _report_progress(frame_num + 1)

            logger.info('Manual-Video] Video frames written to disk.')

            # Write session_manifest.json next to the TIFFs: a single summary
            # file per recording with provenance, rate stats, and per-frame
            # index. Failure to write does not abort the recording cleanup --
            # the TIFFs are the primary deliverable.
            try:
                from lvp_logger import version as _lvp_version

                camera_model = None
                camera_serial = None
                try:
                    if scope is not None and scope._camera_driver is not None:
                        camera_model = getattr(scope._camera_driver, 'model_name', None)
                        camera_serial = getattr(scope._camera_driver, '_device_serial', None)
                except Exception:
                    pass
                manifest = build_session_manifest(
                    timestamps=timestamps,
                    chunks_per_frame=chunks_per_frame,
                    tick_freq_hz=tick_freq_hz,
                    captured_frames=captured_frames,
                    video_duration=video_duration,
                    camera_model=camera_model,
                    camera_serial=camera_serial,
                    lvp_version=_lvp_version,
                    channel_color=color,
                )
                manifest_path = save_folder / 'session_manifest.json'
                with open(manifest_path, 'w') as fh:
                    json.dump(manifest, fh, indent=2, default=str)
                logger.info(f'Manual-Video] Session manifest written to {manifest_path}')
            except Exception as e:
                logger.warning(f'Manual-Video] Failed to write session_manifest.json: {e}')

            if include_hyperstack_generation:
                logger.info('Manual-Video] Creating hyperstack...')

                _, objective = ui_snapshot['objective_info']
                frame_metadata_df = pd.DataFrame(frame_metadata)
                stack_builder.create_single_recording_stack(
                    df=frame_metadata_df,
                    path=save_folder,
                    output_file_loc=save_folder / 'ManualVideo_Frame_HyperStack.ome.tiff',
                    focal_length=objective['focal_length'],
                    binning_size=ui_snapshot['binning'],
                )

                logger.info(
                    f'Manual-Video] Hyperstack created at {save_folder / "ManualVideo_Frame_HyperStack.ome.tiff"}'
                )

        else:
            if not video_save_folder.exists():
                video_save_folder.mkdir(exist_ok=True, parents=True)

            output_file_loc = video_save_folder / f'Video_{start_time_str}.mp4'

            video_writer = VideoWriter(
                output_path=output_file_loc,
                fps=calculated_fps,
                include_timestamp_overlay=True,
                # The memmap is mono; colorize false-color layers at the
                # encoder. None (transmitted / false-color off) gray-encodes.
                color=video_false_color,
            )

            for frame_num in range(captured_frames):
                try:
                    ts = (
                        timestamps[frame_num]
                        if frame_num < len(timestamps)
                        else datetime.datetime.now()
                    )
                    video_writer.add_frame(image=video_frames[frame_num], timestamp=ts)
                except Exception:
                    logger.exception('Manual-Video] FAILED TO WRITE FRAME')
                    dropped_frames += 1

                # Update progress on main thread
                _report_progress(frame_num + 1)

            video_writer.close()
            # Encode failures inside add_frame that did not raise are counted
            # by the writer; fold them in so the total reflects every loss.
            dropped_frames += video_writer.dropped_frames
            # The writer is the authority on where the file landed (a
            # collision suffix or the cv2 .avi fallback may have moved it
            # from the requested path).
            output_file_loc = video_writer.output_path
            logger.info(f'Manual-Video] Video written to {output_file_loc}')

        if dropped_frames > 0:
            notifications.warning(
                'Recording',
                'Video Frames Dropped',
                f'{dropped_frames} of {captured_frames} frame(s) could not be '
                'written, so the saved video is shorter than the recording. '
                'Check the log for the cause.',
            )

        logger.info('Manual-Video] Video writing finished.')

    except Exception:
        # Any failure to finalize must reach the user, not vanish behind a log
        # line. The snapshot precondition above catches the known-None fields;
        # this catch-all makes the silent discard impossible by construction for
        # everything else (a malformed config key, a zero-fps encoder init) --
        # the user is told the recording was lost instead of it disappearing.
        logger.exception('Manual-Video] Finalize failed; recording not saved')
        notifications.error(
            'Recording',
            'Recording Not Saved',
            'The recording could not be saved due to an unexpected error. '
            'Check the log for details.',
        )
        raise
    finally:
        # Cleanup memmap - must explicitly close the underlying mmap object
        # This MUST run even if we return early (e.g., no frames captured)
        if video_frames is not None:
            try:
                # Explicitly close the memory-mapped file
                # Note: No need to flush() before close - close() handles any pending writes
                if hasattr(video_frames, '_mmap') and video_frames._mmap is not None:
                    video_frames._mmap.close()
                del video_frames  # Delete the reference
            except Exception as e:
                logger.warning(f'[LVP Main  ] Error closing memmap: {e}')

        # NOTE: We intentionally do NOT delete the memmap file here because:
        # 1. Windows file deletion can block for several seconds even after closing
        # 2. This causes "Not Responding" freezes in the application
        # 3. The file will be automatically reused on the next recording (see record_init)
        # 4. Reusing the file is actually faster than creating a new one
        logger.info('[LVP Main  ] Memmap file closed and ready for reuse')

    # Return memmap_path so cleanup callback knows which path to remove from tracking
    return memmap_path
