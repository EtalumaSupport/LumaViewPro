# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Video capture and stimulation — extracted from sequenced_capture_executor.py.

Self-contained video recording: frame capture loop, stimulation threads,
and video writing (MP4 and TIFF-frame paths).
"""

import ctypes
import datetime
import pathlib
import queue
import sys
import threading
import time

import numpy as np

from lvp_logger import logger
import modules.image_utils as image_utils
from modules.video_writer import VideoWriter


class VideoCaptureResult:
    """Result of a video capture session."""

    def __init__(self, captured_frames, calculated_fps, video_images,
                 duration_sec):
        self.captured_frames = captured_frames
        self.calculated_fps = calculated_fps
        self.video_images = video_images
        self.duration_sec = duration_sec


class VideoCaptureSession:
    """Manages a single video recording within a protocol step.

    Usage:
        session = VideoCaptureSession(scope, step, autogain_settings,
                                      is_protocol_running_fn, callbacks)
        result = session.capture()
        # result.video_images is a queue of (image, timestamp) tuples
        # Pass to write_video() on the file IO executor
    """

    def __init__(self, scope, step, autogain_settings,
                 is_protocol_running_fn, callbacks, leds_off_fn):
        self._scope = scope
        self._step = step
        self._autogain_settings = autogain_settings
        self._is_protocol_running = is_protocol_running_fn
        self._callbacks = callbacks
        self._leds_off = leds_off_fn

        self._stim_start_event = threading.Event()
        self._stim_stop_event = threading.Event()

    def capture(self) -> VideoCaptureResult | None:
        """Run the video capture loop. Blocking.

        Returns VideoCaptureResult, or None if cancelled/no frames captured.
        """
        step = self._step

        # Drain stale frames before video capture starts
        while self._scope.frame_validity.frames_until_valid() > 0:
            self._scope.get_image(force_new_capture=True)
            self._scope.frame_validity.count_frame()
        # Additional settle for auto-gain first frame
        time.sleep(max(step['Exposure'] / 1000, 0.05))

        # Disable autogain and then reenable it only for the first frame
        if step["Auto_Gain"]:
            self._scope.set_auto_gain(state=False,
                                      settings=self._autogain_settings)
            self._scope.auto_gain_once(
                state=True,
                target_brightness=self._autogain_settings['target_brightness'],
                min_gain=self._autogain_settings['min_gain'],
                max_gain=self._autogain_settings['max_gain'],
            )

        duration_sec = step['Video Config']['duration']

        # Clamp the FPS to be no faster than the exposure rate
        exposure = step['Exposure']
        exposure_freq = 1.0 / (exposure / 1000)
        fps = min(exposure_freq, 40)

        start_ts = time.time()
        stop_ts = start_ts + duration_sec
        captured_frames = 0
        seconds_per_frame = 1.0 / fps
        video_images = queue.Queue(maxsize=500)

        use_color = step['Color'] if step['False_Color'] else 'BF'

        # Start stimulation threads
        stim_threads = []
        for color in step['Stim_Config']:
            stim_config = step['Stim_Config'][color]
            if stim_config['enabled']:
                controller = StimulationController(
                    self._scope, color, stim_config)
                t = threading.Thread(
                    target=controller.run,
                    args=(self._stim_start_event, self._stim_stop_event))
                stim_threads.append(t)
                t.start()

        if "set_recording_title" in self._callbacks:
            from kivy.clock import Clock
            Clock.schedule_once(
                lambda dt: self._callbacks['set_recording_title'](progress=0),
                0)

        logger.info("Protocol-Video] Capturing video...")

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass

        self._stim_stop_event.clear()
        self._stim_start_event.set()

        while time.time() < stop_ts:
            curr_time = time.time()
            progress = (curr_time - start_ts) / duration_sec * 100
            if "set_recording_title" in self._callbacks:
                from kivy.clock import Clock
                Clock.schedule_once(
                    lambda dt, p=progress:
                        self._callbacks['set_recording_title'](progress=p),
                    0)

            if not self._is_protocol_running():
                self._leds_off()
                if "reset_title" in self._callbacks:
                    from kivy.clock import Clock
                    Clock.schedule_once(
                        lambda dt: self._callbacks['reset_title'](), 0)
                return None

            # Currently only support 8-bit images for video
            image = self._scope.get_image(force_to_8bit=True)

            if isinstance(image, np.ndarray):
                # Should never be used since forcing images to 8-bit
                if image.dtype == np.uint16:
                    image = image_utils.convert_12bit_to_16bit(image)

                if (image.dtype != np.uint16) and (step['False_Color']):
                    image = image_utils.add_false_color(
                        array=image, color=use_color)

                image = np.flip(image, 0)

                try:
                    video_images.put_nowait((image, datetime.datetime.now()))
                except queue.Full:
                    logger.warning(
                        f"[Protocol-Video] Frame queue full "
                        f"({video_images.maxsize}), dropping frame")
                    continue

                captured_frames += 1

            # Slightly shorter sleep to compensate for processing overhead
            time.sleep(seconds_per_frame * 0.9)

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass

        self._stim_stop_event.set()
        self._stim_start_event.clear()

        for t in stim_threads:
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning(
                    "[PROTOCOL] Stim thread did not exit within 5s timeout")

        if captured_frames == 0:
            logger.warning(
                "[PROTOCOL] Zero frames captured during video recording "
                "— skipping write")
            return None

        calculated_fps = max(1, int(captured_frames / duration_sec))

        logger.info(f"Protocol-Video] Images present in video array: "
                    f"{not video_images.empty()}")
        logger.info(f"Protocol-Video] Captured Frames: {captured_frames}")
        logger.info(f"Protocol-Video] Video FPS: {calculated_fps}")

        return VideoCaptureResult(
            captured_frames=captured_frames,
            calculated_fps=calculated_fps,
            video_images=video_images,
            duration_sec=duration_sec,
        )


def write_video(result: VideoCaptureResult, save_folder: pathlib.Path,
                name: str, video_as_frames: bool, step: dict,
                callbacks: dict):
    """Write captured video frames to disk.

    Called on the file IO executor thread.

    Args:
        result: VideoCaptureResult from capture()
        save_folder: Directory to write to
        name: Base filename
        video_as_frames: True for TIFF frames, False for MP4
        step: Protocol step dict (for color info)
        callbacks: Dict with optional 'set_writing_title' and 'reset_title'

    Returns:
        pathlib.Path or None: Path to the output file/folder
    """
    try:
        from kivy.clock import Clock
    except ImportError:
        class Clock:
            @staticmethod
            def schedule_once(func, timeout):
                pass

    video_images = result.video_images
    captured_frames = result.captured_frames

    if "set_writing_title" in callbacks:
        Clock.schedule_once(
            lambda dt: callbacks['set_writing_title'](progress=0), 0)

    logger.info("Protocol-Video] Writing video...")

    try:
        if video_as_frames:
            frame_folder = save_folder / f"{name}"
            if not frame_folder.exists():
                frame_folder.mkdir(exist_ok=True, parents=True)

            frame_num = 0
            while not video_images.empty():
                progress = frame_num / max(1, captured_frames) * 100
                if "set_writing_title" in callbacks:
                    Clock.schedule_once(
                        lambda dt, p=progress:
                            callbacks['set_writing_title'](progress=p),
                        0)

                image_pair = video_images.get_nowait()
                frame_num += 1
                image = image_pair[0]
                ts = image_pair[1]
                del image_pair

                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                image_w_timestamp = image_utils.add_timestamp(
                    image=image, timestamp_str=ts_str)
                del image
                video_images.task_done()

                frame_name = f"{name}_Frame_{frame_num:04}"
                output_file_loc = frame_folder / f"{frame_name}.tiff"

                metadata = {
                    "datetime": ts.strftime("%Y:%m:%d %H:%M:%S"),
                    "timestamp": ts.strftime("%Y:%m:%d %H:%M:%S.%f"),
                    "frame_num": frame_num,
                }

                try:
                    image_utils.write_tiff(
                        data=image_w_timestamp,
                        metadata=metadata,
                        file_loc=output_file_loc,
                        video_frame=True,
                        ome=False,
                        color=step['Color'],
                    )
                except Exception as e:
                    logger.error(
                        f"Protocol-Video] Failed to write frame "
                        f"{frame_num}: {e}")

            _drain_queue(video_images)
            capture_result = frame_folder

        else:
            output_file_loc = save_folder / f"{name}.mp4v"
            video_writer = VideoWriter(
                output_file_loc=output_file_loc,
                fps=result.calculated_fps,
                include_timestamp_overlay=True,
            )
            try:
                frame_num = 0
                while not video_images.empty():
                    progress = frame_num / max(1, captured_frames) * 100
                    if "set_writing_title" in callbacks:
                        Clock.schedule_once(
                            lambda dt, p=progress:
                                callbacks['set_writing_title'](progress=p),
                            0)

                    try:
                        image_pair = video_images.get_nowait()
                        video_writer.add_frame(
                            image=image_pair[0], timestamp=image_pair[1])
                        del image_pair
                        video_images.task_done()
                        frame_num += 1
                    except Exception as e:
                        logger.error(
                            f"Protocol-Video] FAILED TO WRITE FRAME: {e}")
            finally:
                video_writer.finish()
                del video_writer

            _drain_queue(video_images)
            capture_result = output_file_loc

    finally:
        pass  # Caller manages video_write_finished event

    if "reset_title" in callbacks:
        Clock.schedule_once(lambda dt: callbacks['reset_title'](), 0)

    logger.info("Protocol-Video] Video writing finished.")
    logger.info(f"Protocol-Video] Video saved at {capture_result}")

    return capture_result


def _drain_queue(q):
    """Drain any remaining items from a queue."""
    try:
        while not q.empty():
            q.get_nowait()
            q.task_done()
    except Exception:
        pass


class StimulationController:
    """Precise LED pulse train for optogenetic stimulation.

    Extracted from SequencedCaptureExecutor._stimulate().
    """

    def __init__(self, scope, color, stim_config):
        self._scope = scope
        self._color = color
        self._stim_config = stim_config

    def run(self, start_event: threading.Event,
            stop_event: threading.Event):
        """Thread target. Runs pulse train until stop_event or
        pulse_count reached."""
        stim_config = self._stim_config
        if not stim_config['enabled']:
            return

        color = self._color
        logger.info(f"[STIMULATOR] Stimulating {color} with {stim_config}")

        illumination = stim_config['illumination']
        frequency = stim_config['frequency']
        pulse_width = stim_config['pulse_width']
        pulse_count = stim_config['pulse_count']

        # Optional: reduce Windows timer quantum to 1 ms during stimulation
        time_period_set = False
        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                time_period_set = True
            except Exception:
                time_period_set = False

        try:
            period_s = 1.0 / float(frequency)
            pulse_s = float(pulse_width) / 1000.0
            ch = self._scope.color2ch(color=color)

            # Use fast path LED toggles if available via API
            def led_on_fast():
                if hasattr(self._scope, 'led_on_fast'):
                    self._scope.led_on_fast(channel=ch, mA=illumination)
                else:
                    self._scope.led_on(channel=ch, mA=illumination)

            def led_off_fast():
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=ch)
                else:
                    self._scope.led_off(channel=ch)

            start_epoch = time.perf_counter()

            start_event.wait()
            logger.info(
                f"[STIMULATOR] stim_start_event set for {color}")

            end_reason = "pulse_count_reached"
            pulses_executed = 0

            for i in range(pulse_count):
                if stop_event.is_set():
                    logger.info(
                        f"[STIMULATOR] {color} stop event set, "
                        f"ending stimulation.")
                    end_reason = "stop_event_set"
                    break

                # Target times for this pulse
                on_time = start_epoch + i * period_s
                off_time = on_time + pulse_s
                next_period_time = start_epoch + (i + 1) * period_s

                # Sleep until on_time (coarse) then spin
                while True:
                    now = time.perf_counter()
                    remaining = on_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        time.sleep(0.0001)

                led_on_fast()
                pulses_executed += 1

                # Sleep/spin until off_time
                while True:
                    now = time.perf_counter()
                    remaining = off_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        time.sleep(0.0001)

                led_off_fast()

                # Maintain period; wait until next_period_time
                while True:
                    now = time.perf_counter()
                    remaining = next_period_time - now
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    else:
                        time.sleep(0.0001)

        finally:
            if sys.platform.startswith('win') and time_period_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass

            logger.info(
                f"[STIMULATOR] {color} stimulation ended after executing "
                f"{pulses_executed} pulses.")
            logger.info(
                f"[STIMULATOR] {color} Ended due to {end_reason}")
            # Ensure LED off at the end
            try:
                ch = self._scope.color2ch(color=color)
                if hasattr(self._scope, 'led_off_fast'):
                    self._scope.led_off_fast(channel=ch)
                else:
                    self._scope.led_off(channel=ch)
            except Exception:
                pass
