# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Protocol video capture: frame capture loop and video writing (MP4 and
TIFF-frame paths)."""

import ctypes
import datetime
import pathlib
import queue
import sys
import time

import numpy as np

from lvp_logger import logger
import modules.image_save as image_save
from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.video_writer import VideoWriter, fps_from_frames


class VideoCaptureResult:
    """Result of a video capture session."""

    def __init__(
        self,
        captured_frames,
        calculated_fps,
        video_images,
        duration_sec,
        dropped_frames=0,
    ):
        self.captured_frames = captured_frames
        self.calculated_fps = calculated_fps
        self.video_images = video_images
        self.duration_sec = duration_sec
        # Frames the capture loop could not queue (consumer fell behind).
        # Surfaced at write time so a short recording is not silent.
        self.dropped_frames = dropped_frames


class VideoCaptureSession:
    """Manages a single video recording within a protocol step.

    Usage:
        session = VideoCaptureSession(scope, step, autogain_settings,
                                      is_protocol_running_fn, callbacks,
                                      leds_off_fn)
        result = session.capture()
        # result.video_images is a queue of (image, timestamp) tuples
        # Pass to write_video() on the file IO executor
    """

    def __init__(
        self,
        scope,
        step,
        autogain_settings,
        is_protocol_running_fn,
        callbacks,
        leds_off_fn,
    ):
        self._scope = scope
        self._step = step
        self._autogain_settings = autogain_settings
        self._is_protocol_running = is_protocol_running_fn
        self._callbacks = callbacks
        self._leds_off = leds_off_fn

    def capture(self) -> VideoCaptureResult | None:
        """Run the video capture loop. Blocking.

        Returns VideoCaptureResult, or None if cancelled/no frames captured.
        """
        step = self._step

        # Drain stale frames before video capture starts. get_image with
        # force_new_capture already counts the grabbed frame; a second count
        # here drained one frame early and admitted a not-yet-valid first
        # video frame.
        while self._scope.imaging.frames_until_valid() > 0:
            self._scope.imaging.get_image(force_new_capture=True)
        # Additional settle for auto-gain first frame
        time.sleep(max(step['Exposure'] / 1000, 0.05))

        # Disable autogain and then reenable it only for the first frame
        if step['Auto_Gain']:
            self._scope.imaging.set_auto_gain(state=False, settings=self._autogain_settings)
            self._scope.imaging.auto_gain_once(
                state=True,
                target_brightness=self._autogain_settings['target_brightness'],
                min_gain_db=self._autogain_settings['min_gain_db'],
                max_gain_db=self._autogain_settings['max_gain_db'],
                ae_max_exposure_ms=self._autogain_settings.get('max_exposure_ms'),
            )

        duration_sec = step['Video Config']['duration']

        # Clamp the FPS to be no faster than the exposure rate
        exposure = step['Exposure']
        if exposure <= 0:
            exposure = 10  # fallback to 10ms if exposure is missing/zero
            logger.warning(
                f'[PROTOCOL-VIDEO] Exposure is {step["Exposure"]}, defaulting to {exposure}ms'
            )
        exposure_freq = 1.0 / (exposure / 1000)
        fps = min(exposure_freq, 40)

        start_ts = time.time()
        stop_ts = start_ts + duration_sec
        captured_frames = 0
        dropped_frames = 0
        seconds_per_frame = 1.0 / fps
        video_images = queue.Queue(maxsize=500)

        if 'set_recording_title' in self._callbacks:
            _schedule_ui(
                lambda dt: self._callbacks['set_recording_title'](
                    elapsed_sec=0, total_sec=duration_sec
                ),
                0,
            )

        logger.info('[PROTOCOL-VIDEO] Capturing video...')

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception as e:
                logger.debug(f'[PROTOCOL-VIDEO] timeBeginPeriod failed: {e}')

        while time.time() < stop_ts:
            curr_time = time.time()
            elapsed = curr_time - start_ts
            if 'set_recording_title' in self._callbacks:
                _schedule_ui(
                    lambda dt, e=elapsed: self._callbacks['set_recording_title'](
                        elapsed_sec=e, total_sec=duration_sec
                    ),
                    0,
                )

            if not self._is_protocol_running():
                self._leds_off()
                # Drain queued frames to free memory on cancel
                _drain_queue(video_images)
                logger.info(f'[PROTOCOL-VIDEO] Cancelled - drained {captured_frames} queued frames')
                if 'reset_title' in self._callbacks:
                    _schedule_ui(lambda dt: self._callbacks['reset_title'](), 0)
                return None

            # Currently only support 8-bit images for video
            image = self._scope.imaging.get_image(force_to_8bit=True)

            if not isinstance(image, np.ndarray):
                logger.warning(
                    '[PROTOCOL-VIDEO] get_image() returned non-array '
                    f'({type(image).__name__}) - camera may have disconnected. '
                    'Ending video capture.'
                )
                break

            if isinstance(image, np.ndarray):
                # The frame enters the queue MONO; false color is applied at the
                # save edges (the MP4 VideoWriter color= and the per-frame TIFF
                # write), never on this capture thread.
                image = np.flip(image, 0)

                try:
                    video_images.put_nowait((image, datetime.datetime.now()))
                    captured_frames += 1
                except queue.Full:
                    # Do NOT `continue` here -- that bypasses the per-frame
                    # sleep below and turns the capture loop into a hot-spin
                    # against the writer thread when the queue is full. Drop
                    # the frame, count it, fall through to the normal
                    # frame-pacing sleep so the consumer has time to drain.
                    dropped_frames += 1
                    logger.warning(
                        f'[PROTOCOL-VIDEO] Frame queue full '
                        f'({video_images.maxsize}), dropping frame'
                    )

            # Slightly shorter sleep to compensate for processing overhead
            time.sleep(seconds_per_frame * 0.9)

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception as e:
                logger.debug(f'[PROTOCOL-VIDEO] timeEndPeriod failed: {e}')

        if captured_frames == 0:
            logger.warning(
                '[PROTOCOL] Zero frames captured during video recording - skipping write'
            )
            return None

        calculated_fps = fps_from_frames(captured_frames, duration_sec)

        logger.info(f'[PROTOCOL-VIDEO] Images present in video array: {not video_images.empty()}')
        logger.info(f'[PROTOCOL-VIDEO] Captured Frames: {captured_frames}')
        if dropped_frames:
            logger.warning(f'[PROTOCOL-VIDEO] Dropped Frames (queue full): {dropped_frames}')
        logger.info(f'[PROTOCOL-VIDEO] Video FPS: {calculated_fps}')

        return VideoCaptureResult(
            captured_frames=captured_frames,
            calculated_fps=calculated_fps,
            video_images=video_images,
            duration_sec=duration_sec,
            dropped_frames=dropped_frames,
        )


def write_video(
    result: VideoCaptureResult,
    save_folder: pathlib.Path,
    name: str,
    video_as_frames: bool,
    step: dict,
    callbacks: dict,
    save_encoding: str,
    capture_depth: int,
    timestamp_overlay: bool,
):
    """Write captured video frames to disk.

    Called on the file IO executor thread.

    Args:
        result: VideoCaptureResult from capture()
        save_folder: Directory to write to
        name: Base filename
        video_as_frames: True for TIFF frames, False for MP4
        step: Protocol step dict (for color info)
        callbacks: Dict with optional 'set_writing_title' and 'reset_title'
        timestamp_overlay: Burn each frame's capture timestamp into the
            encoded video. The user's choice, snapshotted at run start;
            required so no caller can silently decide it.

    Returns:
        pathlib.Path or None: Path to the output file/folder
    """
    video_images = result.video_images
    captured_frames = result.captured_frames
    # Frames lost during the write/encode stage, on top of any the capture
    # loop already dropped (result.dropped_frames).
    lost_in_write = 0

    if 'set_writing_title' in callbacks:
        _schedule_ui(lambda dt: callbacks['set_writing_title'](progress=0), 0)

    logger.info('[PROTOCOL-VIDEO] Writing video...')

    if video_as_frames:
        frame_folder = save_folder / f'{name}'
        if not frame_folder.exists():
            frame_folder.mkdir(exist_ok=True, parents=True)

        frame_num = 0
        while not video_images.empty():
            progress = frame_num / max(1, captured_frames) * 100
            if 'set_writing_title' in callbacks:
                _schedule_ui(lambda dt, p=progress: callbacks['set_writing_title'](progress=p), 0)

            image_pair = video_images.get_nowait()
            frame_num += 1
            image = image_pair[0]
            ts = image_pair[1]
            del image_pair
            video_images.task_done()

            frame_name = f'{name}_Frame_{frame_num:04}'
            output_file_loc = frame_folder / f'{frame_name}.tiff'

            # The timestamp is not drawn into the pixels here: it travels in the
            # frame metadata below, and Create Video draws it at build time only
            # when the timestamp overlay is enabled.
            metadata = {
                'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
                'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
                'frame_num': frame_num,
            }

            try:
                image_save.write_video_frame(
                    frame=image,
                    file_loc=output_file_loc,
                    metadata=metadata,
                    layer_color=step['Color'],
                    false_color_on=step['False_Color'],
                    save_encoding=save_encoding,
                    capture_depth=capture_depth,
                )
            except Exception as e:
                logger.error(f'[PROTOCOL-VIDEO] Failed to write frame {frame_num}: {e}')
                lost_in_write += 1

            del image

        _drain_queue(video_images)
        capture_result = frame_folder

    else:
        output_file_loc = save_folder / f'{name}.mp4'
        video_writer = VideoWriter(
            output_path=output_file_loc,
            fps=result.calculated_fps,
            include_timestamp_overlay=timestamp_overlay,
            # The queued frames are mono; colorize at the encoder. The layer
            # color applies when False_Color is on; 'BF' gray-encodes otherwise
            # -- the same gate the per-frame TIFF write uses above.
            color=step['Color'] if step['False_Color'] else 'BF',
        )
        try:
            frame_num = 0
            while not video_images.empty():
                progress = frame_num / max(1, captured_frames) * 100
                if 'set_writing_title' in callbacks:
                    _schedule_ui(
                        lambda dt, p=progress: callbacks['set_writing_title'](progress=p), 0
                    )

                try:
                    image_pair = video_images.get_nowait()
                except queue.Empty:
                    # Normal producer/consumer race against the empty() guard,
                    # not a failure -- stop draining.
                    break
                try:
                    video_writer.add_frame(image=image_pair[0], timestamp=image_pair[1])
                    frame_num += 1
                except Exception as e:
                    logger.error(f'[PROTOCOL-VIDEO] Failed to encode frame {frame_num}: {e}')
                    lost_in_write += 1
                finally:
                    del image_pair
                    video_images.task_done()
        finally:
            video_writer.close()
            # Encode failures inside add_frame are counted by the writer.
            lost_in_write += video_writer.dropped_frames
            # The writer is the authority on where the file landed (a
            # collision suffix may have moved it from the requested path);
            # record its path, not the request.
            output_file_loc = video_writer.output_path
            del video_writer

        _drain_queue(video_images)
        capture_result = output_file_loc

    if 'reset_title' in callbacks:
        _schedule_ui(lambda dt: callbacks['reset_title'](), 0)

    total_dropped = result.dropped_frames + lost_in_write
    if total_dropped > 0:
        # Log-only, never a modal: write_video runs only on the protocol path, and
        # an unattended protocol must not pop a dialog for a non-fatal dropped-frame
        # count (the video is shorter than the recording but still valid). Only a
        # fatal, run-aborting error pops during a protocol.
        logger.warning(
            f'[PROTOCOL-VIDEO] {total_dropped} frame(s) dropped from "{name}" -- '
            'the video is shorter than the recording.'
        )

    logger.info('[PROTOCOL-VIDEO] Video writing finished.')

    # Verify the file actually exists and has content
    if capture_result is not None and isinstance(capture_result, pathlib.Path):
        if capture_result.exists() and capture_result.stat().st_size > 0:
            logger.info(
                f'[PROTOCOL-VIDEO] Video saved at {capture_result} ({capture_result.stat().st_size} bytes)'
            )
        else:
            logger.error(
                f'[PROTOCOL-VIDEO] Video file MISSING or EMPTY at {capture_result}. '
                f'The codec may not be available on this system. '
                f'Exists={capture_result.exists()}, '
                f'Size={capture_result.stat().st_size if capture_result.exists() else "N/A"}'
            )
    else:
        logger.info(f'[PROTOCOL-VIDEO] Video saved at {capture_result}')

    return capture_result


def _drain_queue(q):
    """Drain any remaining items from a queue."""
    try:
        while not q.empty():
            q.get_nowait()
            q.task_done()
    except Exception as e:
        logger.debug(f'[PROTOCOL-VIDEO] Queue drain interrupted: {e}')
