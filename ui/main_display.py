# Copyright Etaluma, Inc.
"""
MainDisplay — primary application display (recording, camera, fit/zoom)
extracted from lumaviewpro.py.
"""

import datetime
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
import modules.image_utils as image_utils
import modules.scope_commands as scope_commands
from modules.sequential_io_executor import IOTask
from modules.ui_helpers import set_last_save_folder
from ui.composite_capture import CompositeCapture

logger = logging.getLogger('LVP.ui.main_display')


class MainDisplay(CompositeCapture): # i.e. global lumaview

    def __init__(self, camera_type='ids', simulate=False, **kwargs):
        import modules.lumascope_api as lumascope_api
        super(MainDisplay,self).__init__(**kwargs)
        self.scope = lumascope_api.Lumascope(camera_type=camera_type, simulate=simulate)
        self.camera_temps_event = None
        self.recording = threading.Event()
        self.recording.clear()
        self.video_writing = threading.Event()  # Track if video is being written
        self.video_writing.clear()
        self.recording_check = None
        self.recording_event = None
        self.recording_complete_event = None
        self.recording_title_update = None
        self._last_recorded_frame_ts = None  # For duplicate frame detection during recording
        self.writing_progress_update = None
        self.video_writing_progress = 0
        self.video_writing_total_frames = 0
        self.led_on_before_pause = False

    def log_camera_temps(self):
        if self.scope.camera_is_connected():
            temps = self.scope.get_camera_temps()
            for source, temp in temps.items():
                logger.info(f'[CAM Class ] Camera {source} Temperature : {temp:.2f} °C')
        else:
            if self.camera_temps_event is not None:
                Clock.unschedule(self.camera_temps_event)

    def cam_toggle(self):
        logger.info('[LVP Main  ] MainDisplay.cam_toggle()')

        ctx = _app_ctx.ctx
        settings = ctx.settings
        io_executor = ctx.io_executor

        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if not self.scope.camera_active:
            return

        if scope_display.play:
            scope_display.play = False
            scope_display.stop()
            if self.scope.led_connected:
                self.led_on_before_pause = self.scope.get_led_state(color=common_utils.get_opened_layer(ctx.image_settings))['enabled']
                scope_commands.leds_off(self.scope, io_executor)
                layer_obj = ctx.image_settings.layer_lookup(layer=common_utils.get_opened_layer(ctx.image_settings))
                layer_obj.update_led_toggle_ui()
        else:
            if self.led_on_before_pause:
                opened_layer = common_utils.get_opened_layer(ctx.image_settings)
                io_executor.put(IOTask(
                    action=self.scope.led_on,
                    kwargs={'channel': self.scope.color2ch(opened_layer), 'mA': settings[opened_layer]['ill']}
                ))
                layer_obj = ctx.image_settings.layer_lookup(layer=opened_layer)
                layer_obj.update_led_toggle_ui()

            scope_display.play = True
            scope_display.start()

    def record_button(self):
        from ui.notification_popup import show_notification_popup

        if self.recording.is_set():
            return

        # Check if video is currently being written
        if self.video_writing.is_set():
            logger.warning('[LVP Main  ] Cannot start recording - video is being written')
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Video Being Written",
                message="Please wait for the current video to finish writing before starting a new recording."
            ), 0)
            # Reset button state
            try:
                self.ids['record_btn'].state = 'normal'
            except Exception as e:
                logger.warning(f'[LVP Main  ] Failed to reset record button state: {e}')
            return

        # H-3 fix: snapshot widget values on main thread before submitting
        # to camera executor, since .ids access is not thread-safe.
        ctx = _app_ctx.ctx
        false_color = None
        for layer in common_utils.get_layers():
            layer_accordion_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            if not layer_accordion_obj.collapse:
                if layer_obj.ids['false_color'].active:
                    false_color = layer
                break

        _app_ctx.ctx.camera_executor.put(IOTask(
            self.record_init,
            kwargs={'false_color': false_color},
        ))

    def open_save_folder_button(self):
        from ui.post_processing import open_last_save_folder
        open_last_save_folder()

    def record_init(self, false_color=None):
        from ui.notification_popup import show_notification_popup

        logger.info('[LVP Main  ] MainDisplay.record()')

        ctx = _app_ctx.ctx
        settings = ctx.settings

        # Guard against race condition: if another record_init() already started, abort
        if self.recording.is_set():
            logger.warning('[LVP Main  ] Recording already in progress, ignoring duplicate record_init()')
            return

        if not self.scope.camera_active:
            return

        # Atomically claim the recording operation
        self.recording.set()

        self.video_as_frames = settings['video_as_frames']

        # false_color was snapshotted on main thread by record_button()
        color = false_color

        self.video_false_color = color

        if "manual_video" in settings:
            max_fps = settings["manual_video"]["max_fps"]
            max_duration = settings["manual_video"]["max_duration"]
        else:
            max_fps = 40
            max_duration = 30

        # Record at camera rate — duplicate frame detection in record_helper()
        # prevents storing the same frame twice. max_fps only used for
        # final video playback FPS calculation, not capture throttling.
        frame_size = self.scope.camera_frame_size
        exposure = self.scope.camera_exposure_ms
        exposure_freq = 1.0 / (exposure / 1000)
        video_fps = min(exposure_freq, max_fps)

        max_frames = math.ceil(video_fps * max_duration)

        start_time = datetime.datetime.now()
        self.start_time_str = start_time.strftime("%Y-%m-%d_%H.%M.%S")

        if self.video_as_frames:
            save_folder = pathlib.Path(settings['live_folder']) / "Manual" / f"Video_{self.start_time_str}"
        else:
            save_folder = pathlib.Path(settings['live_folder']) / "Manual"

        self.video_save_folder = save_folder

        self.start_ts = time.time()
        self.stop_ts = self.start_ts + max_duration
        seconds_per_frame = 1.0 / video_fps

        self.memmap_location = pathlib.Path(settings['live_folder']) / "recording_temp.dat"

        if not settings['use_full_pixel_depth'] or not settings['video_as_frames']:
            dtype = 'uint8'
        else:
            dtype = 'uint16'

        # Calculate expected file size and shape
        if (color is None) or (dtype == 'uint16'):
            required_shape = (max_frames, frame_size["height"], frame_size["width"])
        else:
            required_shape = (max_frames, frame_size["height"], frame_size["width"], 3)

        bytes_per_element = 1 if dtype == 'uint8' else 2
        expected_size = int(np.prod(required_shape, dtype=np.int64)) * bytes_per_element

        # Check if we can reuse existing file (fast path - no truncation needed)
        reuse_existing = False
        if self.memmap_location.exists():
            try:
                actual_size = self.memmap_location.stat().st_size
                if actual_size == expected_size:
                    logger.info('[LVP Main  ] Reusing existing memmap file (same size)')
                    reuse_existing = True
                else:
                    logger.info(f'[LVP Main  ] Memmap size changed ({actual_size} -> {expected_size}), recreating')
                    # Try to delete old file, but don't block if it fails
                    try:
                        self.memmap_location.unlink()
                    except (OSError, PermissionError) as e:
                        logger.warning(f'[LVP Main  ] Could not remove old memmap: {e}, will overwrite')
            except Exception as e:
                logger.warning(f'[LVP Main  ] Could not check memmap file: {e}')

        # Create or reuse memmap
        try:
            # Use mode="r+" to reuse existing file without truncation (fast)
            # Use mode="w+" only when creating new file or size changed (requires truncation)
            memmap_mode = "r+" if reuse_existing else "w+"

            if (color is None) or (dtype == 'uint16'):
                self.current_video_frames = np.memmap(str(self.memmap_location), dtype=dtype, mode=memmap_mode, shape=(max_frames, frame_size["height"], frame_size["width"]))
            else:
                self.current_video_frames = np.memmap(str(self.memmap_location), dtype=dtype, mode=memmap_mode, shape=(max_frames, frame_size["height"], frame_size["width"], 3))
        except (OSError, IOError) as e:
            logger.error(f'[LVP Main  ] Failed to create memmap file: {e}')
            logger.error(f'[LVP Main  ] If this persists, manually delete: {self.memmap_location}')
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Recording Failed",
                message=f"Could not create recording file. The file may be locked from a previous crash.\n\nTry manually deleting:\n{self.memmap_location.name}"
            ), 0)
            raise

        self.current_captured_frames = 0
        self.timestamps = []
        self._last_recorded_frame_ts = None  # Reset duplicate detection

        logger.info(f"Manual-Video] Capturing video...")

        # Schedule recording at camera exposure rate (not capped to max_fps).
        # Duplicate frame detection in record_helper() naturally handles the case
        # where the timer fires faster than the camera delivers new frames.
        capture_interval = 1.0 / exposure_freq
        # Schedule title updates to show recording progress
        self.recording_title_update = Clock.schedule_interval(self.update_recording_title, 0.1)  # Update every 100ms
        self.recording_check = Clock.schedule_interval(self.check_recording_state, capture_interval)
        self.recording_event = Clock.schedule_interval(self._enqueue_recording_frame, capture_interval)

    def _enqueue_recording_frame(self, dt=None):
        """Enqueue a recording frame task without creating closure."""
        _app_ctx.ctx.camera_executor.put(IOTask(self.record_helper))

    def check_recording_state(self, dt=None):
        # Over the max duration, stop video
        if time.time() >= self.stop_ts:
            Clock.unschedule(self.recording_check)
            Clock.unschedule(self.recording_event)
            if hasattr(self, 'recording_title_update') and self.recording_title_update:
                Clock.unschedule(self.recording_title_update)
            self.video_duration = time.time() - self.start_ts
            self.recording_complete_event = Clock.schedule_once(self._enqueue_recording_complete, 0)
            self.ids['record_btn'].state = 'normal'

        # Button not clicked yet, keep recording
        if self.ids['record_btn'].state == 'down':
            return

        # Button clicked, stop recording
        Clock.unschedule(self.recording_check)
        Clock.unschedule(self.recording_event)
        if hasattr(self, 'recording_title_update') and self.recording_title_update:
            Clock.unschedule(self.recording_title_update)
        self.video_duration = time.time() - self.start_ts
        self.recording_complete_event = Clock.schedule_once(self._enqueue_recording_complete, 0)

    def update_recording_title(self, dt=None):
        """Update window title with recording elapsed time."""
        if self.recording.is_set():
            elapsed = time.time() - self.start_ts
            from kivy.core.window import Window
            Window.set_title(f"Lumaview Pro {_app_ctx.ctx.version}   |   Recording Manual Video: {elapsed:.1f}s")

    def update_writing_progress(self, dt=None):
        """Update window title with video writing progress percentage."""
        if self.video_writing_total_frames > 0:
            progress_pct = (self.video_writing_progress / self.video_writing_total_frames) * 100
            from kivy.core.window import Window
            Window.set_title(f"Lumaview Pro {_app_ctx.ctx.version}   |   Writing Manual Video: {progress_pct:.0f}%")

    def _enqueue_recording_complete(self, dt=None):
        """Enqueue recording finalization task on camera executor.

        This runs on the main thread (via Clock.schedule_once), so we snapshot
        all UI-dependent values here before handing off to background threads.
        """
        from modules.config_ui_getters import (
            get_active_layer_config, get_image_capture_config_from_ui,
            get_current_objective_info, get_binning_from_ui,
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

        _app_ctx.ctx.camera_executor.put(IOTask(
            self._finalize_recording_state,
            kwargs={'ui_snapshot': ui_snapshot},
        ))

    def _finalize_recording_state(self, dt=None, ui_snapshot=None):
        """Run on camera executor: Capture final state quickly and hand off to file writer."""
        memmap_path = None
        try:
            logger.info("Manual-Video] Finalizing recording state...")

            # Capture state (atomic with respect to camera thread, as we are ON camera thread)
            captured_frames = self.current_captured_frames if hasattr(self, 'current_captured_frames') else 0
            timestamps = self.timestamps[:] if hasattr(self, 'timestamps') else []
            video_frames = self.current_video_frames if hasattr(self, 'current_video_frames') else None
            video_duration = self.video_duration if hasattr(self, 'video_duration') else 0
            video_save_folder = self.video_save_folder if hasattr(self, 'video_save_folder') else None
            start_time_str = self.start_time_str if hasattr(self, 'start_time_str') else ""
            video_as_frames = self.video_as_frames if hasattr(self, 'video_as_frames') else False
            video_false_color = self.video_false_color if hasattr(self, 'video_false_color') else None
            memmap_path = self.memmap_location if hasattr(self, 'memmap_location') else None

            # Release memmap reference from MainDisplay so file_io_executor has exclusive ownership
            self.current_video_frames = None

            # Clear recording event immediately - camera is now free
            if not self.recording.is_set():
                logger.warning("Manual-Video] Recording already cleared in finalize")
            else:
                self.recording.clear()

            # Set video writing event to block new recordings
            self.video_writing.set()

            # Initialize progress tracking on main thread
            total = max(1, captured_frames)
            Clock.schedule_once(lambda dt: setattr(self, 'video_writing_progress', 0), 0)
            Clock.schedule_once(lambda dt, t=total: setattr(self, 'video_writing_total_frames', t), 0)

            # Schedule progress updates
            self.writing_progress_update = Clock.schedule_interval(self.update_writing_progress, 0.1)

            # Prepare kwargs for file IO
            kwargs = {
                'captured_frames': captured_frames,
                'timestamps': timestamps,
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
            _app_ctx.ctx.file_io_executor.put(IOTask(
                self.recording_complete,
                kwargs=kwargs,
                callback=self._recording_cleanup_callback,
                pass_result=True
            ))

        except Exception as e:
            logger.exception(f"Manual-Video] Error in finalize_recording: {e}")
            # Ensure cleanup happens even if error
            Clock.schedule_once(lambda dt: self._recording_cleanup_gui(memmap_path=memmap_path), 0)

    def _recording_cleanup_callback(self, dt=None, result=None, exception=None):
        """Callback after file writing completes - run cleanup on GUI thread."""
        memmap_path = result
        Clock.schedule_once(lambda dt: self._recording_cleanup_gui(memmap_path=memmap_path), 0)

    def recording_complete(self, **kwargs):
        """Run on file_io_executor: Do heavy file writing without blocking camera."""
        from modules.stack_builder import StackBuilder
        from modules.video_writer import VideoWriter

        # Retrieve captured state passed from _finalize_recording_state
        captured_frames = kwargs.get('captured_frames', 0)
        timestamps = kwargs.get('timestamps', [])
        video_frames = kwargs.get('video_frames', None)
        video_duration = kwargs.get('video_duration', 0)
        video_save_folder = kwargs.get('video_save_folder', None)
        start_time_str = kwargs.get('start_time_str', "")
        video_as_frames = kwargs.get('video_as_frames', False)
        memmap_path = kwargs.get('memmap_path', None)
        video_false_color = kwargs.get('video_false_color', None)

        # H-4 fix: use UI values snapshotted on main thread by _enqueue_recording_complete()
        ui_snapshot = kwargs.get('ui_snapshot', {})

        try:
            # Defensive check
            if video_frames is None:
                logger.error("Manual-Video] recording_complete called with no video frames")
                return memmap_path

            # Prevent division by zero
            if video_duration <= 0:
                video_duration = 0.1
                logger.warning("Manual-Video] Video duration was 0, using 0.1s")

            if captured_frames == 0:
                logger.error("Manual-Video] No frames captured, aborting video write")
                return memmap_path

            calculated_fps = captured_frames // video_duration

            logger.info(f"Manual-Video] Images present in video array: {len(video_frames) > 0 if video_frames is not None else 0}")
            logger.info(f"Manual-Video] Captured Frames: {captured_frames}")
            logger.info(f"Manual-Video] Video FPS: {calculated_fps}")
            logger.info("Manual-Video] Writing video...")

            color, active_layer_config = ui_snapshot['active_layer_config']

            include_hyperstack_generation = False

            if video_as_frames:

                image_capture_config = ui_snapshot['image_capture_config']

                if image_capture_config['output_format']['sequenced'] == 'ImageJ Hyperstack':
                    include_hyperstack_generation = True
                    _, objective = ui_snapshot['objective_info']
                    stack_builder = StackBuilder(
                        has_turret=_app_ctx.ctx.scope.has_turret(),
                    )
                    frame_metadata = []

                save_folder = video_save_folder

                if not save_folder.exists():
                    save_folder.mkdir(exist_ok=True, parents=True)

                for frame_num in range(captured_frames):

                    image = video_frames[frame_num]
                    ts = timestamps[frame_num] if frame_num < len(timestamps) else datetime.datetime.now()
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    image = image_utils.add_timestamp(image=image, timestamp_str=ts_str)

                    frame_name = f"ManualVideo_Frame_{frame_num:04}"

                    output_file_loc = save_folder / f"{frame_name}.tiff"

                    metadata = {
                                "datetime": ts.strftime("%Y:%m:%d %H:%M:%S"),
                                "timestamp": ts.strftime("%Y:%m:%d %H:%M:%S.%f"),
                                "frame_num": frame_num
                            }

                    if include_hyperstack_generation:
                        current_position = _app_ctx.ctx.scope.get_current_position()
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
                            color=color
                        )
                    except Exception as e:
                        logger.exception(f"Protocol-Video] Failed to write frame {frame_num}: {e}")

                    # Update progress on main thread
                    progress = frame_num + 1
                    Clock.schedule_once(lambda dt, p=progress: setattr(self, 'video_writing_progress', p), 0)

                logger.info("Manual-Video] Video frames written to disk.")


                if include_hyperstack_generation:
                    logger.info("Manual-Video] Creating hyperstack...")

                    _, objective = ui_snapshot['objective_info']
                    frame_metadata_df = pd.DataFrame(frame_metadata)
                    stack_builder.create_single_recording_stack(
                        df=frame_metadata_df,
                        path=save_folder,
                        output_file_loc=save_folder / f"ManualVideo_Frame_HyperStack.ome.tiff",
                        focal_length=objective['focal_length'],
                        binning_size=ui_snapshot['binning'],
                    )

                    logger.info(f"Manual-Video] Hyperstack created at {save_folder / f'ManualVideo_Frame_HyperStack.ome.tiff'}")

            else:
                if not video_save_folder.exists():
                    video_save_folder.mkdir(exist_ok=True, parents=True)

                output_file_loc = video_save_folder / f"Video_{start_time_str}.mp4v"

                video_writer = VideoWriter(
                    output_file_loc=output_file_loc,
                    fps=calculated_fps,
                    include_timestamp_overlay=True
                )

                for frame_num in range(captured_frames):
                    try:
                        ts = timestamps[frame_num] if frame_num < len(timestamps) else datetime.datetime.now()
                        video_writer.add_frame(image=video_frames[frame_num], timestamp=ts)
                    except Exception:
                        logger.exception("Manual-Video] FAILED TO WRITE FRAME")

                    # Update progress on main thread
                    progress = frame_num + 1
                    Clock.schedule_once(lambda dt, p=progress: setattr(self, 'video_writing_progress', p), 0)

                video_writer.finish()
                logger.info(f"Manual-Video] Mp4 written to {output_file_loc}")

            logger.info("Manual-Video] Video writing finished.")

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

            # Clear video writing state - new recordings can now start
            self.video_writing.clear()

            # Reset window title
            from kivy.core.window import Window
            Window.set_title(f"Lumaview Pro {_app_ctx.ctx.version}")

            logger.info("Manual-Video] Recording cleanup complete")
        except Exception as e:
            logger.exception(f"Manual-Video] Error during GUI cleanup: {e}")

    def record_helper(self, dt=None):
        settings = _app_ctx.ctx.settings

        if not settings['use_full_pixel_depth'] or not settings['video_as_frames']:
            force_to_8bit = True
        else:
            force_to_8bit = False

        # Use get_image_from_buffer() instead of get_image() — avoids the extra
        # get_array() copy (~8MB at 4K) and the retry sleep loop. Returns the
        # latest frame reference directly via grab_latest().
        result = self.scope.get_image_from_buffer(force_to_8bit=force_to_8bit)
        if result is None or result[0] is False:
            return
        image, frame_ts = result

        # Skip duplicate frames — if camera hasn't delivered a new frame since
        # last recording, don't waste a memmap slot on identical data.
        if frame_ts is not None and frame_ts == self._last_recorded_frame_ts:
            return
        self._last_recorded_frame_ts = frame_ts

        if isinstance(image, np.ndarray):

            if image.dtype == np.uint16:
                image = image_utils.convert_12bit_to_16bit(image)

            # Note: Currently, if image is 12/16-bit, then we ignore false coloring for video captures.
            if (image.dtype != np.uint16) and (self.video_false_color is not None):
                image = image_utils.add_false_color(array=image, color=self.video_false_color)

            image = np.flip(image, 0)

            self.current_video_frames[self.current_captured_frames] = image
            self.timestamps.append(datetime.datetime.now())

            self.current_captured_frames += 1


    def fit_image(self):
        logger.info('[LVP Main  ] MainDisplay.fit_image()')
        if not self.scope.camera_active:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        logger.info('[LVP Main  ] MainDisplay.one2one_image()')
        if not self.scope.camera_active:
            return
        scope = _app_ctx.ctx.scope
        w = self.width
        h = self.height
        scale_hor = float(scope.get_width()) / float(w)
        scale_ver = float(scope.get_height()) / float(h)
        scale = max(scale_hor, scale_ver)
        self.ids['viewer_id'].scale = scale
        self.ids['viewer_id'].pos = (int((w-scale*w)/2),int((h-scale*h)/2))
