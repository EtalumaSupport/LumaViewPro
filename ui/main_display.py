# Copyright Etaluma, Inc.
"""
MainDisplay -- primary application display (recording, camera, fit/zoom)
extracted from lumaviewpro.py.
"""

import datetime
import json
import logging
import math
import pathlib
import threading
import time

import numpy as np
import pandas as pd

from kivy.clock import Clock

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.config_helpers import get_manual_video_max_duration
import modules.image_utils as image_utils
from modules.recording_manifest import build_session_manifest
from modules.sequential_io_executor import IOTask
from modules.stack_builder import StackBuilder
from modules.video_writer import VideoWriter
from ui.ui_helpers import set_last_save_folder
from ui.composite_capture import CompositeCapture

logger = logging.getLogger('LVP.ui.main_display')


class MainDisplay(CompositeCapture):  # i.e. global lumaview
    def __init__(self, camera_type='ids', simulate=False, **kwargs):
        import modules.lumascope_api as lumascope_api

        super().__init__(**kwargs)
        self.scope = lumascope_api.Lumascope(camera_type=camera_type, simulate=simulate)
        # LVP-A-2: camera_temps_event moved to Lumascope.start_camera_temp_logging.
        self.recording = threading.Event()
        self.recording.clear()
        self.video_writing = threading.Event()  # Track if video is being written
        self.video_writing.clear()
        self._record_shape_warning_emitted = False
        self.recording_check = None
        self.recording_complete_event = None
        self.recording_title_update = None
        # Per-frame camera-callback driven recording (replaces the Kivy
        # Clock save timer). Slot index reserved by callback under
        # _record_lock; the IOTask on camera_executor writes the slot.
        self._record_lock = threading.Lock()
        self._save_interval_s = 0.0
        self._next_save_slot_ts = 0.0
        self._reserved_frames = 0
        self._max_frames = 0
        self.writing_progress_update = None
        self.video_writing_progress = 0
        self.video_writing_total_frames = 0
        # Reused scratch buffers for the record-path depth conversion and
        # false-color widening in record_helper. Sized lazily on the first
        # frame of a record and freed at finalize. Reuse is safe: record_helper
        # runs on the single-threaded camera_executor and copies its result
        # into the memmap slot before the next call can overwrite the scratch.
        self._record_convert_buf = None
        self._record_color_buf = None
        self._pause_led_snapshot = None  # save/restore via API

    def cam_toggle(self):
        try:
            logger.info('[LVP Main  ] MainDisplay.cam_toggle()')

            scope_display = self.ids['viewer_id'].ids['scope_display_id']
            if not self.scope.imaging.camera_active:
                gui_logger.button('CAM_TOGGLE', 'no-op (camera inactive)')
                return

            gui_logger.toggle('CAM_PLAY', not scope_display.play)
            if scope_display.play:
                scope_display.play = False
                # Stage B1: pause() instead of stop()+start() so the
                # display thread stays alive across pause-resume; no
                # Thread spawn/join overhead; generation does NOT bump
                # so the texture stays on the last rendered frame.
                scope_display.pause()
                if self.scope.led_connected:
                    self._pause_led_snapshot = self.scope.illumination.save_led_state(
                        'camera_pause'
                    )
                    self.scope.illumination.leds_off_async()
                    # LED observer handles UI button sync
            else:
                if self._pause_led_snapshot:
                    self.scope.illumination.restore_led_state(self._pause_led_snapshot)
                    self._pause_led_snapshot = None
                    # LED observer handles UI button sync

                scope_display.play = True
                scope_display.resume()
        except Exception as e:
            logger.error(f'[UI] cam_toggle failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def record_button(self):
        gui_logger.button('RECORD')
        from ui.notification_popup import show_notification_popup

        if self.recording.is_set():
            return

        # Check if video is currently being written
        if self.video_writing.is_set():
            # The video write shares the file worker with long jobs
            # (composite generation, z-projection): the finalize can sit
            # queued behind one for minutes while this flag stays set.
            # Name the real situation -- "wait for the video" alone reads
            # as a hang when a composite run is what is actually ahead.
            _ctx = _app_ctx.ctx
            _ahead = 0
            _busy_with = None
            try:
                _ahead = _ctx.file_io_executor.queue.qsize()
                _running = _ctx.file_io_executor.running_task
                if _running is not None:
                    _busy_with = getattr(_running.action, '__name__', str(_running.action))
            except Exception as e:
                logger.debug(f'[LVP Main  ] file-lane status read failed: {e}')
            logger.warning(
                f'[LVP Main  ] Cannot start recording - previous video write '
                f'still pending on the file worker (ahead={_ahead}, '
                f'busy_with={_busy_with})'
            )
            if _busy_with or _ahead:
                _msg = (
                    'The previous video is waiting for other file work '
                    '(such as composite generation) to finish first.'
                )
            else:
                _msg = 'The previous video is still being written.'
            Clock.schedule_once(
                lambda dt: show_notification_popup(
                    title='Video Being Written',
                    message=f'{_msg} It will complete automatically; '
                    'please try recording again afterward.',
                ),
                0,
            )
            # Reset button state
            try:
                self.ids['record_btn'].state = 'normal'
            except Exception as e:
                logger.warning(f'[LVP Main  ] Failed to reset record button state: {e}')
            return

        # H-3 fix: snapshot widget values on main thread before submitting
        # to camera executor, since .ids access is not thread-safe.
        ctx = _app_ctx.ctx
        # Gate on camera-connected before queueing record_init. Without
        # this, clicking record on a disconnected camera queues a task
        # that fails inside record_init with no clear user feedback.
        if not getattr(ctx.scope, 'camera_connected', True):
            from modules.notification_center import notifications

            notifications.warning(
                'Camera',
                'Camera not connected',
                'Cannot start recording -- camera is not connected. '
                'Check USB and reconnect, then try again.',
            )
            try:
                self.ids['record_btn'].state = 'normal'
            except Exception as e:
                logger.warning(f'[LVP Main  ] Failed to reset record button state: {e}')
            return
        false_color = None
        for layer in common_utils.get_layers():
            layer_accordion_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            if not layer_accordion_obj.collapse:
                if layer_obj.ids['false_color'].active:
                    false_color = layer
                break

        _app_ctx.ctx.camera_executor.put(
            IOTask(
                self.record_init,
                kwargs={'false_color': false_color},
            )
        )

    def open_save_folder_button(self):
        gui_logger.button('OPEN_SAVE_FOLDER')
        from ui.post_processing import open_last_save_folder

        open_last_save_folder()

    def record_init(self, false_color=None):
        from ui.notification_popup import show_notification_popup

        logger.info('[LVP Main  ] MainDisplay.record()')

        ctx = _app_ctx.ctx
        settings = ctx.settings

        # Guard against race condition: if another record_init() already started, abort
        if self.recording.is_set():
            logger.warning(
                '[LVP Main  ] Recording already in progress, ignoring duplicate record_init()'
            )
            return

        if not self.scope.imaging.camera_active:
            return

        # Atomically claim the recording operation
        self.recording.set()
        self._record_shape_warning_emitted = False

        self.video_as_frames = settings['video_as_frames']

        # false_color was snapshotted on main thread by record_button()
        color = false_color

        self.video_false_color = color

        manual_video = settings.get('manual_video', {})
        max_fps = manual_video.get('max_fps', 0)
        max_duration = get_manual_video_max_duration(settings)
        # max_fps == 0 means uncapped (camera free-run rate). The
        # spinner ships at 0; non-zero is the explicit user opt-in
        # that gates pre-flight + camera-rate-toggle below.
        self._user_requested_fps_limit = max_fps > 0

        frame_size = self.scope.imaging.camera_frame_size
        exposure = self.scope.imaging.camera_exposure_ms
        exposure_freq = 1.0 / (exposure / 1000)
        # Pre-flight: warn if the user requested an FPS limit that
        # exposure can't hit. Accept the achievable rate either way.
        if self._user_requested_fps_limit and max_fps > exposure_freq:
            try:
                from modules.notification_center import notifications

                notifications.warning(
                    'Recording',
                    'FPS budget exceeded',
                    f'Requested {max_fps:.1f} FPS at {exposure:.0f} ms exposure '
                    f"exceeds the camera's max {exposure_freq:.1f} FPS for that "
                    f'exposure. Recording will run at {exposure_freq:.1f} FPS '
                    f'instead. Reduce exposure to hit the requested rate.',
                )
            except Exception as e:
                logger.warning(f'[LVP Main  ] Could not notify FPS budget: {e}')
        if self._user_requested_fps_limit:
            video_fps = min(exposure_freq, max_fps)
        else:
            video_fps = exposure_freq

        max_frames = math.ceil(video_fps * max_duration)

        start_time = datetime.datetime.now()
        self.start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

        if self.video_as_frames:
            save_folder = (
                pathlib.Path(settings['live_folder']) / 'Manual' / f'Video_{self.start_time_str}'
            )
        else:
            save_folder = pathlib.Path(settings['live_folder']) / 'Manual'

        self.video_save_folder = save_folder
        # Set last save folder immediately so "Open Last Save Folder"
        # works even if the async cleanup callback hasn't fired yet (fixes #603)
        set_last_save_folder(save_folder)

        self.start_ts = time.time()
        self.stop_ts = self.start_ts + max_duration

        self.memmap_location = pathlib.Path(settings['live_folder']) / 'recording_temp.dat'

        if not settings['use_full_pixel_depth'] or not settings['video_as_frames']:
            dtype = 'uint8'
        else:
            dtype = 'uint16'

        # Calculate expected file size and shape
        if (color is None) or (dtype == 'uint16'):
            required_shape = (max_frames, frame_size['height'], frame_size['width'])
        else:
            required_shape = (max_frames, frame_size['height'], frame_size['width'], 3)

        bytes_per_element = 1 if dtype == 'uint8' else 2
        expected_size = int(np.prod(required_shape, dtype=np.int64)) * bytes_per_element

        # Issue #633 Stage 2C: pre-flight disk space. The memmap creation
        # below would error eventually, but at that point the user has
        # already clicked Record and waited; surface earlier with an
        # actionable message. 256 MB safety margin keeps the OS from
        # running out of breathing room mid-record.
        _DISK_SAFETY_MB = 256
        try:
            expected_mb = expected_size / (1024 * 1024)
            required_mb = expected_mb + _DISK_SAFETY_MB
            ok, free_mb = common_utils.check_disk_space_ok(self.memmap_location.parent, required_mb)
            if not ok:
                from modules.notification_center import notifications

                notifications.error(
                    'Recording',
                    'Insufficient disk space',
                    f'Recording would need {expected_mb / 1024:.1f} GB but only '
                    f'{free_mb / 1024:.1f} GB free. Free up space or reduce '
                    f'FPS / duration / pixel depth.',
                )
                self.recording.clear()
                return
        except Exception as e:
            # Non-fatal: a disk_usage probe failure is a worse error than
            # the eventual memmap-create failure. Log and proceed; the
            # OSError catch below is the structural backstop.
            logger.warning(f'[LVP Main  ] Disk-space pre-flight failed: {e}')

        # Check if we can reuse existing file (fast path - no truncation needed)
        reuse_existing = False
        if self.memmap_location.exists():
            try:
                actual_size = self.memmap_location.stat().st_size
                if actual_size == expected_size:
                    logger.info('[LVP Main  ] Reusing existing memmap file (same size)')
                    reuse_existing = True
                else:
                    logger.info(
                        f'[LVP Main  ] Memmap size changed ({actual_size} -> {expected_size}), recreating'
                    )
                    # Try to delete old file, but don't block if it fails
                    try:
                        self.memmap_location.unlink()
                    except (OSError, PermissionError) as e:
                        logger.warning(
                            f'[LVP Main  ] Could not remove old memmap: {e}, will overwrite'
                        )
            except Exception as e:
                logger.warning(f'[LVP Main  ] Could not check memmap file: {e}')

        # Create or reuse memmap
        try:
            # Use mode="r+" to reuse existing file without truncation (fast)
            # Use mode="w+" only when creating new file or size changed (requires truncation)
            memmap_mode = 'r+' if reuse_existing else 'w+'

            if (color is None) or (dtype == 'uint16'):
                self.current_video_frames = np.memmap(
                    str(self.memmap_location),
                    dtype=dtype,
                    mode=memmap_mode,
                    shape=(max_frames, frame_size['height'], frame_size['width']),
                )
            else:
                self.current_video_frames = np.memmap(
                    str(self.memmap_location),
                    dtype=dtype,
                    mode=memmap_mode,
                    shape=(max_frames, frame_size['height'], frame_size['width'], 3),
                )
        except OSError as e:
            logger.error(f'[LVP Main  ] Failed to create memmap file: {e}')
            logger.error(f'[LVP Main  ] If this persists, manually delete: {self.memmap_location}')
            Clock.schedule_once(
                lambda dt: show_notification_popup(
                    title='Recording Failed',
                    message=f'Could not create recording file. The file may be locked from a previous crash.\n\nTry manually deleting:\n{self.memmap_location.name}',
                ),
                0,
            )
            return

        self.current_captured_frames = 0
        # Pre-sized: callback reserves a slot index, IOTask writes at that
        # index. Concurrent callbacks-then-IOTasks can't race on append() and
        # produce out-of-order timestamps lists; slot-indexed writes match
        # slot-indexed reads in recording_complete.
        self.timestamps = [None] * max_frames
        self.chunks_per_frame = [None] * max_frames
        # Snapshot the camera-side timestamp tick frequency once. Used at
        # finalize time to convert ChunkTimestamp ticks to seconds; None
        # if the camera doesn't expose a Timestamp chunk.
        self.timestamp_tick_freq_hz = (
            getattr(self.scope._camera_driver, 'timestamp_tick_frequency_hz', None)
            if self.scope._camera_driver
            else None
        )

        logger.info('Manual-Video] Capturing video...')

        # Considered camera-side AcquisitionFrameRate cap; rejected because
        # the cap controls AVERAGE rate while jittering individual frames
        # widely around the target. Cadence is enforced host-side via the
        # callback slot scheduler below. The _fps_limit_was_enabled flag
        # stays as defense against future re-introduction.
        self._fps_limit_was_enabled = False

        # Considered host-side Kivy Clock timer; rejected because Clock's
        # ~16-30 ms Windows display-frame floor plus main-thread GIL
        # contention produces ~30% cadence stdev. Camera SDK ticks arrive
        # on the ingest thread at sub-ms accuracy. check_recording_state
        # stays on Clock for button-state and time-stop detection
        # (main-thread Kivy widget reads).
        self._save_interval_s = 1.0 / video_fps
        self._next_save_slot_ts = self.start_ts + self._save_interval_s
        self._reserved_frames = 0
        self._max_frames = max_frames
        capture_interval = 1.0 / video_fps
        self.recording_title_update = Clock.schedule_interval(self.update_recording_title, 0.1)
        self.recording_check = Clock.schedule_interval(self.check_recording_state, capture_interval)
        self.scope.imaging.add_frame_listener(self._on_camera_frame, name='manual_recording')

    def _on_camera_frame(self, image, frame_ts, chunks):
        """Camera-SDK-thread callback: reserve next save slot and enqueue write.

        Runs on the camera ingest thread (Pylon ``PylonImageGrab`` / IDS
        grab loop / SimCameraPump). Fast decision only -- the actual
        memmap write happens on ``camera_executor`` via ``record_helper``.
        """
        if not self.recording.is_set():
            return
        now = time.time()
        with self._record_lock:
            if not self.recording.is_set():
                return
            if self._reserved_frames >= self._max_frames:
                return
            if now < self._next_save_slot_ts:
                return
            # Backpressure: if the single CAMERA_WORKER is behind on memmap
            # writes, drop this frame BEFORE reserving a slot -- reserving then
            # dropping would leave an unwritten (black) slot. Bounds the
            # in-flight image backlog that was the manual-record RAM balloon.
            if not _app_ctx.ctx.camera_executor.admit_live_frame():
                return
            slot_index = self._reserved_frames
            self._reserved_frames += 1
            self._next_save_slot_ts += self._save_interval_s
        _app_ctx.ctx.camera_executor.put(
            IOTask(
                self.record_helper,
                kwargs={
                    'slot_index': slot_index,
                    'image': image,
                    'frame_ts': frame_ts,
                    'chunks': chunks,
                },
                droppable_live=True,
            )
        )

    def check_recording_state(self, dt=None):
        # Time-stop or capacity-stop: max_frames reserved means the
        # memmap is full; no more callbacks will reserve a slot.
        if time.time() >= self.stop_ts or self._reserved_frames >= self._max_frames:
            self._stop_recording_clocks()
            self.video_duration = time.time() - self.start_ts
            self.recording_complete_event = Clock.schedule_once(self._enqueue_recording_complete, 0)
            # Flip the record_btn back to 'normal' so the UI shows "ready"; the
            # next block (button-released stop) must NOT fall through and
            # double-schedule the complete event -- return here.
            self.ids['record_btn'].state = 'normal'
            return

        # Button not clicked yet, keep recording
        if self.ids['record_btn'].state == 'down':
            return

        # Button clicked, stop recording
        self._stop_recording_clocks()
        self.video_duration = time.time() - self.start_ts
        self.recording_complete_event = Clock.schedule_once(self._enqueue_recording_complete, 0)

    def _stop_recording_clocks(self):
        """Unschedule recording clocks and unregister the camera callback.

        Called from both ``check_recording_state`` stop branches. Order
        matters: unregister the callback BEFORE the finalize IOTask
        runs so no new ``record_helper`` task can be enqueued after
        ``_finalize_recording_state`` has snapshotted state. The
        camera_executor is FIFO, so any record_helper tasks already
        queued ahead of the finalize task complete first.
        """
        try:
            self.scope.imaging.remove_frame_listener(self._on_camera_frame)
        except Exception as e:
            logger.warning(f'[LVP Main  ] remove_frame_listener failed: {e}')
        if self.recording_check is not None:
            Clock.unschedule(self.recording_check)
            self.recording_check = None
        if self.recording_title_update is not None:
            Clock.unschedule(self.recording_title_update)
            self.recording_title_update = None

    def update_recording_title(self, dt=None):
        """Update window title-bar event suffix with recording elapsed time."""
        if self.recording.is_set():
            from ui.ui_helpers import set_title_event_text

            elapsed = time.time() - self.start_ts
            set_title_event_text(f'Recording Manual Video: {elapsed:.1f}s')

    def update_writing_progress(self, dt=None):
        """Update window title-bar event suffix with video writing progress."""
        if self.video_writing_total_frames > 0:
            from ui.ui_helpers import set_title_event_text

            progress_pct = (self.video_writing_progress / self.video_writing_total_frames) * 100
            set_title_event_text(f'Writing Manual Video: {progress_pct:.0f}%')

    def _enqueue_recording_complete(self, dt=None):
        """Enqueue recording finalization task on camera executor.

        This runs on the main thread (via Clock.schedule_once), so we snapshot
        all UI-dependent values here before handing off to background threads.
        """
        from modules.config_ui_getters import (
            get_active_layer_config,
            get_image_capture_config_from_ui,
            get_current_objective_info,
            get_binning_from_ui,
        )

        # H-4 fix: snapshot widget values on main thread
        ui_snapshot = {}
        try:
            ui_snapshot['active_layer_config'] = get_active_layer_config()
        except Exception as e:
            logger.warning(f'[LVP Main  ] Could not snapshot active_layer_config: {e}')
            ui_snapshot['active_layer_config'] = None
        try:
            ui_snapshot['image_capture_config'] = get_image_capture_config_from_ui()
        except Exception as e:
            logger.warning(f'[LVP Main  ] Could not snapshot image_capture_config: {e}')
            ui_snapshot['image_capture_config'] = None
        try:
            ui_snapshot['objective_info'] = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[LVP Main  ] Could not snapshot objective_info: {e}')
            ui_snapshot['objective_info'] = None
        try:
            ui_snapshot['binning'] = get_binning_from_ui()
        except Exception as e:
            logger.warning(f'[LVP Main  ] Could not snapshot binning: {e}')
            ui_snapshot['binning'] = 1

        _app_ctx.ctx.camera_executor.put(
            IOTask(
                self._finalize_recording_state,
                kwargs={'ui_snapshot': ui_snapshot},
            )
        )

    def _finalize_recording_state(self, dt=None, ui_snapshot=None):
        """Run on camera executor: Capture final state quickly and hand off to file writer."""
        memmap_path = None
        # Issue #633 Stage 2C: restore camera free-run before anything else
        # so live preview unsticks immediately after recording stops.
        # _fps_limit_was_enabled is set in record_init only when we
        # actually toggled the camera; never call disable when we didn't
        # enable, to avoid touching a knob the user may have set elsewhere.
        if getattr(self, '_fps_limit_was_enabled', False):
            try:
                self.scope.imaging.set_max_acquisition_frame_rate(False, 0.0)
                self._fps_limit_was_enabled = False
                logger.info('Manual-Video] Camera FPS limit disabled (free-run restored)')
            except Exception as e:
                logger.warning(
                    f'[LVP Main  ] Could not disable FPS limit: {e}; '
                    f'live preview may stay at recording rate until next config'
                )
        try:
            logger.info('Manual-Video] Finalizing recording state...')

            # Capture state (atomic with respect to camera thread, as we are ON camera thread)
            captured_frames = (
                self.current_captured_frames if hasattr(self, 'current_captured_frames') else 0
            )
            timestamps = self.timestamps[:] if hasattr(self, 'timestamps') else []
            chunks_per_frame = self.chunks_per_frame[:] if hasattr(self, 'chunks_per_frame') else []
            tick_freq_hz = (
                self.timestamp_tick_freq_hz if hasattr(self, 'timestamp_tick_freq_hz') else None
            )
            video_frames = (
                self.current_video_frames if hasattr(self, 'current_video_frames') else None
            )
            video_duration = self.video_duration if hasattr(self, 'video_duration') else 0
            video_save_folder = (
                self.video_save_folder if hasattr(self, 'video_save_folder') else None
            )
            start_time_str = self.start_time_str if hasattr(self, 'start_time_str') else ''
            video_as_frames = self.video_as_frames if hasattr(self, 'video_as_frames') else False
            video_false_color = (
                self.video_false_color if hasattr(self, 'video_false_color') else None
            )
            memmap_path = self.memmap_location if hasattr(self, 'memmap_location') else None

            # Release memmap reference from MainDisplay so file_io_executor has exclusive ownership
            self.current_video_frames = None
            # Drop the per-record conversion scratch buffers; a record is a
            # bounded event and these are multi-MB each.
            self._record_convert_buf = None
            self._record_color_buf = None

            # Clear recording event immediately - camera is now free
            if not self.recording.is_set():
                logger.warning('Manual-Video] Recording already cleared in finalize')
            else:
                self.recording.clear()

            # Set video writing event to block new recordings
            self.video_writing.set()

            # Initialize progress tracking on main thread
            total = max(1, captured_frames)
            Clock.schedule_once(lambda dt: setattr(self, 'video_writing_progress', 0), 0)
            Clock.schedule_once(
                lambda dt, t=total: setattr(self, 'video_writing_total_frames', t), 0
            )

            # Schedule progress updates
            self.writing_progress_update = Clock.schedule_interval(
                self.update_writing_progress, 0.1
            )

            # Prepare kwargs for file IO
            kwargs = {
                'captured_frames': captured_frames,
                'timestamps': timestamps,
                'chunks_per_frame': chunks_per_frame,
                'tick_freq_hz': tick_freq_hz,
                'video_frames': video_frames,
                'video_duration': video_duration,
                'video_save_folder': video_save_folder,
                'start_time_str': start_time_str,
                'video_as_frames': video_as_frames,
                'memmap_path': memmap_path,
                'video_false_color': video_false_color,
                'ui_snapshot': ui_snapshot or {},
            }

            # Hand off to file IO executor (doesn't block camera)
            _app_ctx.ctx.file_io_executor.put(
                IOTask(
                    self.recording_complete,
                    kwargs=kwargs,
                    callback=self._recording_cleanup_callback,
                    pass_result=True,
                )
            )

        except Exception as e:
            logger.exception(f'Manual-Video] Error in finalize_recording: {e}')
            # Ensure cleanup happens even if error
            Clock.schedule_once(lambda dt: self._recording_cleanup_gui(memmap_path=memmap_path), 0)

    def _recording_cleanup_callback(self, dt=None, result=None, exception=None):
        """Callback after file writing completes - run cleanup on GUI thread."""
        memmap_path = result
        Clock.schedule_once(lambda dt: self._recording_cleanup_gui(memmap_path=memmap_path), 0)

    def recording_complete(self, **kwargs):
        """Run on file_io_executor: Do heavy file writing without blocking camera."""
        # Retrieve captured state passed from _finalize_recording_state
        captured_frames = kwargs.get('captured_frames', 0)
        timestamps = kwargs.get('timestamps', [])
        chunks_per_frame = kwargs.get('chunks_per_frame', [])
        tick_freq_hz = kwargs.get('tick_freq_hz')
        video_frames = kwargs.get('video_frames')
        video_duration = kwargs.get('video_duration', 0)
        video_save_folder = kwargs.get('video_save_folder')
        start_time_str = kwargs.get('start_time_str', '')
        video_as_frames = kwargs.get('video_as_frames', False)
        memmap_path = kwargs.get('memmap_path')

        # H-4 fix: use UI values snapshotted on main thread by _enqueue_recording_complete()
        ui_snapshot = kwargs.get('ui_snapshot', {})

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

            calculated_fps = captured_frames // video_duration

            logger.info(
                f'Manual-Video] Images present in video array: {len(video_frames) > 0 if video_frames is not None else 0}'
            )
            logger.info(f'Manual-Video] Captured Frames: {captured_frames}')
            logger.info(f'Manual-Video] Video FPS: {calculated_fps}')
            logger.info('Manual-Video] Writing video...')

            if ui_snapshot.get('active_layer_config') is None:
                # The main-thread snapshot failed (raises when no layer
                # accordion is open). Without the layer's false-color
                # config the frames cannot be finalized; tell the user
                # instead of dying on an opaque unpack TypeError that
                # silently discarded the finished recording.
                logger.error(
                    'Manual-Video] No active layer config snapshot; cannot finalize recording'
                )
                from modules.notification_center import notifications

                notifications.error(
                    'Recording',
                    'Recording Not Saved',
                    'The recording could not be saved because no imaging '
                    'layer was selected. Open a layer tab and record again.',
                )
                return memmap_path

            color, _active_layer_config = ui_snapshot['active_layer_config']

            include_hyperstack_generation = False

            if video_as_frames:
                image_capture_config = ui_snapshot['image_capture_config']

                if image_capture_config['output_format']['sequenced'] == 'OME-TIFF Hyperstack':
                    include_hyperstack_generation = True
                    _, objective = ui_snapshot['objective_info']
                    stack_builder = StackBuilder(
                        has_turret=_app_ctx.ctx.scope.motion.has_turret(),
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

                    # Issue #633 Stage 2A: per-frame timestamp metadata. Existing
                    # 'datetime' / 'timestamp' / 'frame_num' keys preserved for
                    # backward compatibility with downstream readers that look
                    # for them. New 'timestamp_iso' / 'timestamp_camera_ticks' /
                    # 'timestamp_camera_tick_hz' / 'frame_id' keys mirror the
                    # structured Plane fields used elsewhere; the video_frame
                    # TIFF path serializes them into the description tag.
                    metadata = {
                        'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
                        'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
                        'timestamp_iso': ts.isoformat(timespec='microseconds'),
                        'frame_num': frame_num,
                    }
                    chunks = (
                        chunks_per_frame[frame_num] if frame_num < len(chunks_per_frame) else None
                    )
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
                        current_position = _app_ctx.ctx.scope.motion.get_current_position()
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
                        image_utils.write_tiff(
                            data=image,
                            metadata=metadata,
                            file_loc=output_file_loc,
                            video_frame=True,
                            ome=False,
                            color=color,
                        )
                    except Exception as e:
                        logger.exception(f'Protocol-Video] Failed to write frame {frame_num}: {e}')

                    # Update progress on main thread
                    progress = frame_num + 1
                    Clock.schedule_once(
                        lambda dt, p=progress: setattr(self, 'video_writing_progress', p), 0
                    )

                logger.info('Manual-Video] Video frames written to disk.')

                # Issue #633 Stage 2B: write session_manifest.json next to
                # the TIFFs. Single summary file per recording with provenance,
                # rate stats, and per-frame index. Failure to write does not
                # abort the recording cleanup -- the TIFFs are the primary
                # deliverable.
                try:
                    from lvp_logger import version as _lvp_version

                    camera_model = None
                    camera_serial = None
                    try:
                        scope = _app_ctx.ctx.scope
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

                    # Update progress on main thread
                    progress = frame_num + 1
                    Clock.schedule_once(
                        lambda dt, p=progress: setattr(self, 'video_writing_progress', p), 0
                    )

                video_writer.close()
                logger.info(f'Manual-Video] Mp4 written to {output_file_loc}')

            logger.info('Manual-Video] Video writing finished.')

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

    def _recording_cleanup_gui(self, memmap_path=None):
        """Final cleanup on GUI thread after video writing completes."""
        try:
            # Unschedule progress updates
            if hasattr(self, 'writing_progress_update') and self.writing_progress_update:
                Clock.unschedule(self.writing_progress_update)

            # Unschedule recording complete event if it exists
            if hasattr(self, 'recording_complete_event') and self.recording_complete_event:
                Clock.unschedule(self.recording_complete_event)

            # Set last save folder
            if hasattr(self, 'video_save_folder'):
                set_last_save_folder(self.video_save_folder)

            # Clear the title-bar event suffix; status bar will show FPS only.
            from ui.ui_helpers import set_title_event_text

            set_title_event_text(None)

            # Delete the memmap scratch file from the user's capture folder.
            # record_init kept it around for size-match reuse on the next run,
            # but that's only useful when the next recording matches geometry;
            # the cost is a multi-GB litter file in the user's live folder
            # after every record. record_init recreates as needed.
            if memmap_path is not None:
                try:
                    pathlib.Path(memmap_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(
                        f'Manual-Video] Could not remove memmap scratch file {memmap_path}: {e}'
                    )

            logger.info('Manual-Video] Recording cleanup complete')
        except Exception as e:
            logger.exception(f'Manual-Video] Error during GUI cleanup: {e}')
        finally:
            # Recording can only restart once this flag clears; clear on
            # every exit so a failure earlier in cleanup cannot leave the
            # Record button permanently dead.
            self.video_writing.clear()

    def record_helper(self, slot_index, image, frame_ts, chunks, dt=None):
        """Write one reserved frame slot on the camera_executor.

        Called via IOTask from ``_on_camera_frame``. ``slot_index`` was
        reserved under ``_record_lock`` in the callback, so two
        concurrent ``record_helper`` tasks always have distinct slots
        and slot-indexed writes never collide.
        """
        # Defensive: finalize may have nulled the memmap if a stray
        # task slipped past the FIFO discipline; the increment-and-write
        # is then a no-op rather than a NoneType crash.
        if self.current_video_frames is None:
            return
        if not isinstance(image, np.ndarray):
            return
        if slot_index >= self.current_video_frames.shape[0]:
            return

        settings = _app_ctx.ctx.settings
        force_to_8bit = not settings['use_full_pixel_depth'] or not settings['video_as_frames']

        if force_to_8bit and image.dtype != np.uint8:
            if self._record_convert_buf is None or self._record_convert_buf.shape != image.shape:
                self._record_convert_buf = np.empty(image.shape, dtype=np.uint8)
            image = image_utils.convert_12bit_to_8bit(image, out=self._record_convert_buf)
        elif image.dtype == np.uint16:
            if (
                self._record_convert_buf is None
                or self._record_convert_buf.shape != image.shape
                or self._record_convert_buf.dtype != np.uint16
            ):
                self._record_convert_buf = np.empty(image.shape, dtype=np.uint16)
            image = image_utils.convert_12bit_to_16bit(image, out=self._record_convert_buf)

        # Note: Currently, if image is 12/16-bit, then we ignore false coloring for video captures.
        if (image.dtype != np.uint16) and (self.video_false_color is not None):
            color_shape = (image.shape[0], image.shape[1], 3)
            if (
                self._record_color_buf is None
                or self._record_color_buf.shape != color_shape
                or self._record_color_buf.dtype != image.dtype
            ):
                self._record_color_buf = np.empty(color_shape, dtype=image.dtype)
            image = image_utils.add_false_color(
                array=image, color=self.video_false_color, output=self._record_color_buf
            )

        image = np.flip(image, 0)

        target_shape = self.current_video_frames.shape[1:]
        if image.shape != target_shape:
            image = self._fit_recording_frame_to_buffer(image, target_shape)
            if image is None:
                return

        self.current_video_frames[slot_index] = image
        self.timestamps[slot_index] = frame_ts if frame_ts is not None else datetime.datetime.now()
        self.chunks_per_frame[slot_index] = chunks
        self.current_captured_frames = max(self.current_captured_frames, slot_index + 1)

    def _fit_recording_frame_to_buffer(
        self,
        image: np.ndarray,
        target_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Pad/crop a frame whose live size differs from the recording buffer.

        Manual recording should not crash the camera worker when a delivered
        frame size differs from the pre-allocated buffer; black-pad/crop the
        spatial overhang (via image_utils.fit_frame_to_shape) and warn once.
        Fundamentally incompatible frames (different ndim / channel count) are
        skipped.
        """
        fitted = image_utils.fit_frame_to_shape(image, target_shape)
        if fitted is None:
            logger.warning(
                f'[Manual-Video] Skipping frame with incompatible shape '
                f'{image.shape}; expected {target_shape}'
            )
            return None

        if not self._record_shape_warning_emitted:
            logger.warning(
                f'[Manual-Video] Recording frame shape {image.shape} does not '
                f'match buffer {target_shape}; padding/cropping frame'
            )
            self._record_shape_warning_emitted = True
        return fitted

    def fit_image(self):
        gui_logger.button('FIT_IMAGE')
        logger.info('[LVP Main  ] MainDisplay.fit_image()')
        if not self.scope.imaging.camera_active:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0, 0)

    def one2one_image(self):
        try:
            gui_logger.button('ONE_TO_ONE_IMAGE')
            logger.info('[LVP Main  ] MainDisplay.one2one_image()')
            if not self.scope.imaging.camera_active:
                return
            scope = _app_ctx.ctx.scope
            w = self.width
            h = self.height
            scale_hor = float(scope.imaging.get_width()) / float(w)
            scale_ver = float(scope.imaging.get_height()) / float(h)
            scale = max(scale_hor, scale_ver)
            self.ids['viewer_id'].scale = scale
            self.ids['viewer_id'].pos = (int((w - scale * w) / 2), int((h - scale * h) / 2))
        except Exception as e:
            logger.error(f'[UI] one2one_image failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))
