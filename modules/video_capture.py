# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Video capture and stimulation - extracted from sequenced_capture_executor.py.

Self-contained video recording: frame capture loop, stimulation threads,
and video writing (MP4 and TIFF-frame paths).
"""

import ctypes
import datetime
import math
import pathlib
import queue
import statistics
import sys
import threading
import time
from typing import NamedTuple

import numpy as np

from lvp_logger import logger
import modules.image_utils as image_utils
from modules.kivy_utils import schedule_ui as _schedule_ui
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
                 is_protocol_running_fn, callbacks, leds_off_fn, *,
                 stim_profiling: bool = False,
                 run_dir: pathlib.Path | None = None):
        self._scope = scope
        self._step = step
        self._autogain_settings = autogain_settings
        self._is_protocol_running = is_protocol_running_fn
        self._callbacks = callbacks
        self._leds_off = leds_off_fn
        self._stim_profiling = stim_profiling
        self._run_dir = run_dir

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
        if exposure <= 0:
            exposure = 10  # fallback to 10ms if exposure is missing/zero
            logger.warning(
                f"[PROTOCOL-VIDEO] Exposure is {step['Exposure']}, "
                f"defaulting to {exposure}ms"
            )
        exposure_freq = 1.0 / (exposure / 1000)
        fps = min(exposure_freq, 40)

        start_ts = time.time()
        stop_ts = start_ts + duration_sec
        captured_frames = 0
        seconds_per_frame = 1.0 / fps
        video_images = queue.Queue(maxsize=500)

        use_color = step['Color'] if step['False_Color'] else 'BF'

        # Start one stimulation scheduler thread for all enabled channels.
        stim_thread = None
        enabled_stim_configs = {
            color: stim_config
            for color, stim_config in step['Stim_Config'].items()
            if stim_config['enabled']
        }
        if enabled_stim_configs:
            scheduler = StimulationController(
                self._scope,
                enabled_stim_configs,
                profiling_enabled=self._stim_profiling,
                run_dir=self._run_dir,
            )
            stim_thread = threading.Thread(
                target=scheduler.run,
                name="stim-scheduler",
                args=(self._stim_start_event, self._stim_stop_event),
            )
            stim_thread.start()

        if "set_recording_title" in self._callbacks:
            _schedule_ui(
                lambda dt: self._callbacks['set_recording_title'](progress=0),
                0)

        logger.info("[PROTOCOL-VIDEO] Capturing video...")

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception as e:
                logger.debug(f"[PROTOCOL-VIDEO] timeBeginPeriod failed: {e}")

        self._stim_stop_event.clear()
        self._stim_start_event.set()

        while time.time() < stop_ts:
            curr_time = time.time()
            progress = (curr_time - start_ts) / duration_sec * 100
            if "set_recording_title" in self._callbacks:
                _schedule_ui(
                    lambda dt, p=progress:
                        self._callbacks['set_recording_title'](progress=p),
                    0)

            if not self._is_protocol_running():
                # Stop stim thread BEFORE turning off LEDs to prevent
                # stim pulses from re-enabling LEDs after leds_off()
                self._stim_stop_event.set()
                if stim_thread is not None:
                    stim_thread.join(timeout=2.0)
                self._leds_off()
                # Drain queued frames to free memory on cancel
                _drain_queue(video_images)
                logger.info(
                    f"[PROTOCOL-VIDEO] Cancelled - drained {captured_frames} queued frames"
                )
                if "reset_title" in self._callbacks:
                        _schedule_ui(
                        lambda dt: self._callbacks['reset_title'](), 0)
                return None

            # Currently only support 8-bit images for video
            image = self._scope.get_image(force_to_8bit=True)

            if not isinstance(image, np.ndarray):
                logger.warning(
                    "[PROTOCOL-VIDEO] get_image() returned non-array "
                    f"({type(image).__name__}) - camera may have disconnected. "
                    "Ending video capture.")
                break

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
                        f"[PROTOCOL-VIDEO] Frame queue full "
                        f"({video_images.maxsize}), dropping frame")
                    continue

                captured_frames += 1

            # Slightly shorter sleep to compensate for processing overhead
            time.sleep(seconds_per_frame * 0.9)

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception as e:
                logger.debug(f"[PROTOCOL-VIDEO] timeEndPeriod failed: {e}")

        self._stim_stop_event.set()
        self._stim_start_event.clear()

        if stim_thread is not None:
            stim_thread.join(timeout=5.0)
            if stim_thread.is_alive():
                logger.warning(
                    "[STIMULATOR] Scheduler thread did not exit within 5s timeout")

        if captured_frames == 0:
            logger.warning(
                "[PROTOCOL] Zero frames captured during video recording "
                "- skipping write")
            return None

        calculated_fps = max(1, int(captured_frames / duration_sec))

        logger.info(f"[PROTOCOL-VIDEO] Images present in video array: "
                    f"{not video_images.empty()}")
        logger.info(f"[PROTOCOL-VIDEO] Captured Frames: {captured_frames}")
        logger.info(f"[PROTOCOL-VIDEO] Video FPS: {calculated_fps}")

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
    video_images = result.video_images
    captured_frames = result.captured_frames

    if "set_writing_title" in callbacks:
        _schedule_ui(
            lambda dt: callbacks['set_writing_title'](progress=0), 0)

    logger.info("[PROTOCOL-VIDEO] Writing video...")

    try:
        if video_as_frames:
            frame_folder = save_folder / f"{name}"
            if not frame_folder.exists():
                frame_folder.mkdir(exist_ok=True, parents=True)

            frame_num = 0
            while not video_images.empty():
                progress = frame_num / max(1, captured_frames) * 100
                if "set_writing_title" in callbacks:
                    _schedule_ui(
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
                        f"[PROTOCOL-VIDEO] Failed to write frame "
                        f"{frame_num}: {e}")

            _drain_queue(video_images)
            capture_result = frame_folder

        else:
            output_file_loc = save_folder / f"{name}.mp4"
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
                        _schedule_ui(
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
                            f"[PROTOCOL-VIDEO] FAILED TO WRITE FRAME: {e}")
            finally:
                video_writer.finish()
                del video_writer

            _drain_queue(video_images)
            capture_result = output_file_loc

    finally:
        pass  # Caller manages video_write_finished event

    if "reset_title" in callbacks:
        _schedule_ui(lambda dt: callbacks['reset_title'](), 0)

    logger.info("[PROTOCOL-VIDEO] Video writing finished.")

    # Verify the file actually exists and has content
    if capture_result is not None and isinstance(capture_result, pathlib.Path):
        if capture_result.exists() and capture_result.stat().st_size > 0:
            logger.info(f"[PROTOCOL-VIDEO] Video saved at {capture_result} ({capture_result.stat().st_size} bytes)")
        else:
            logger.error(f"[PROTOCOL-VIDEO] Video file MISSING or EMPTY at {capture_result}. "
                         f"The codec may not be available on this system. "
                         f"Exists={capture_result.exists()}, "
                         f"Size={capture_result.stat().st_size if capture_result.exists() else 'N/A'}")
    else:
        logger.info(f"[PROTOCOL-VIDEO] Video saved at {capture_result}")

    return capture_result


def _drain_queue(q):
    """Drain any remaining items from a queue."""
    try:
        while not q.empty():
            q.get_nowait()
            q.task_done()
    except Exception as e:
        logger.debug(f"[PROTOCOL-VIDEO] Queue drain interrupted: {e}")


class StimEdge(NamedTuple):
    target_offset_s: float
    action: str
    channel: int
    mA: float | None
    color: str


class StimulationController:
    """Single-thread stim scheduler for all enabled optogenetic channels."""

    _MAX_STIM_CURRENT_MA = 1000
    _SORT_EPSILON_S = 1e-6

    def __init__(self, scope, stim_configs, *,
                 profiling_enabled: bool = False,
                 run_dir: pathlib.Path | None = None):
        self._scope = scope
        self._stim_configs = stim_configs
        self._profiling_enabled = profiling_enabled and run_dir is not None
        self._run_dir = run_dir
        self._active_channels: list[tuple[str, int]] = []
        self._edges = self._build_edge_schedule()

    def _build_edge_schedule(self) -> list[StimEdge]:
        edges = []
        active_channels = {}

        for color, stim_config in self._stim_configs.items():
            if not stim_config.get('enabled'):
                continue

            illumination = stim_config.get('illumination')
            frequency = stim_config.get('frequency')
            pulse_width = stim_config.get('pulse_width')
            pulse_count = stim_config.get('pulse_count')

            if not isinstance(frequency, (int, float)) or frequency <= 0:
                logger.error(f"[STIMULATOR] {color}: invalid frequency {frequency} Hz - must be > 0. Skipping stimulation.")
                continue
            if not isinstance(pulse_width, (int, float)) or pulse_width <= 0:
                logger.error(f"[STIMULATOR] {color}: invalid pulse_width {pulse_width} ms - must be > 0. Skipping stimulation.")
                continue
            if not isinstance(pulse_count, int) or pulse_count <= 0:
                logger.error(f"[STIMULATOR] {color}: invalid pulse_count {pulse_count} - must be > 0. Skipping stimulation.")
                continue
            if not isinstance(illumination, (int, float)) or illumination <= 0:
                logger.error(f"[STIMULATOR] {color}: invalid illumination {illumination} mA - must be > 0. Skipping stimulation.")
                continue

            if illumination > self._MAX_STIM_CURRENT_MA:
                logger.warning(f"[STIMULATOR] {color}: illumination {illumination}mA exceeds max {self._MAX_STIM_CURRENT_MA}mA. Clamping.")
                illumination = self._MAX_STIM_CURRENT_MA

            period_s = 1.0 / float(frequency)
            pulse_s = float(pulse_width) / 1000.0
            if pulse_s >= period_s:
                logger.warning(f"[STIMULATOR] {color}: pulse_width ({pulse_width}ms) >= period ({period_s*1000:.1f}ms). Clamping pulse to 90% of period.")
                pulse_s = period_s * 0.9

            channel = self._scope.color2ch(color=color)
            active_channels[color] = channel

            for i in range(pulse_count):
                on_time = i * period_s
                off_time = on_time + pulse_s
                edges.append(StimEdge(
                    target_offset_s=on_time,
                    action="on",
                    channel=channel,
                    mA=illumination,
                    color=color,
                ))
                edges.append(StimEdge(
                    target_offset_s=off_time,
                    action="off",
                    channel=channel,
                    mA=None,
                    color=color,
                ))

        self._active_channels = sorted(
            active_channels.items(),
            key=lambda item: item[1],
        )
        edges.sort(key=lambda edge: (
            round(edge.target_offset_s / self._SORT_EPSILON_S),
            0 if edge.action == "off" else 1,
            edge.channel,
        ))
        return edges

    def _wait_until(self, target_time: float, stop_event: threading.Event) -> bool:
        while True:
            if stop_event.is_set():
                return False

            now = time.perf_counter()
            remaining = target_time - now
            if remaining <= 0:
                return True
            if remaining > 0.003:
                time.sleep(remaining - 0.002)
            else:
                time.sleep(0.0001)

    def _dispatch_edge(self, edge: StimEdge) -> float:
        """Dispatch a single stim edge. Returns perf_counter timestamp after the call."""
        if edge.action == "on":
            if hasattr(self._scope, 'led_on_fast'):
                self._scope.led_on_fast(channel=edge.channel, mA=edge.mA)
            else:
                self._scope.led_on(channel=edge.channel, mA=edge.mA)
        else:
            if hasattr(self._scope, 'led_off_fast'):
                self._scope.led_off_fast(channel=edge.channel)
            else:
                self._scope.led_off(channel=edge.channel)
        return time.perf_counter()

    def run(self, start_event: threading.Event,
            stop_event: threading.Event):
        """Thread target. Runs a merged pulse-edge schedule for all channels."""
        if not self._edges:
            return

        enabled_colors = [color for color, _ in self._active_channels]
        logger.info(f"[STIMULATOR] Starting merged scheduler for {enabled_colors}")

        time_period_set = False
        end_reason = "schedule_complete"
        executed_edges = 0
        lateness_ms = []

        # Per-edge profiling data: {color: [list of timing dicts]}
        profiling = self._profiling_enabled
        if profiling:
            # Track per-color: on_cmd durations, off_cmd durations, actual on-times
            profile_on_cmd: dict[str, list[dict]] = {c: [] for c, _ in self._active_channels}
            profile_off_cmd: dict[str, list[dict]] = {c: [] for c, _ in self._active_channels}
            profile_actual_on: dict[str, list[dict]] = {c: [] for c, _ in self._active_channels}
            # Track the last on-edge end time per channel for actual on-time calc
            last_on_end: dict[str, float] = {}
            pulses_executed: dict[str, int] = {c: 0 for c, _ in self._active_channels}

        if sys.platform.startswith('win'):
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                time_period_set = True
            except Exception:
                time_period_set = False

        try:
            while not start_event.wait(timeout=0.05):
                if stop_event.is_set():
                    end_reason = "stop_event_set_before_start"
                    return

            if stop_event.is_set():
                end_reason = "stop_event_set_before_start"
                return

            start_epoch = time.perf_counter()

            for edge in self._edges:
                if stop_event.is_set():
                    end_reason = "stop_event_set"
                    break

                if not self._wait_until(start_epoch + edge.target_offset_s, stop_event):
                    end_reason = "stop_event_set"
                    break

                if stop_event.is_set():
                    end_reason = "stop_event_set"
                    break

                t_before = time.perf_counter()
                dispatch_lateness = max(
                    0.0,
                    (t_before - (start_epoch + edge.target_offset_s)) * 1000.0,
                )
                lateness_ms.append(dispatch_lateness)

                try:
                    t_after = self._dispatch_edge(edge)
                except Exception as ex:
                    end_reason = "dispatch_error"
                    logger.error(f"[STIMULATOR] {edge.color}: {edge.action} edge failed: {ex}")
                    break

                if profiling:
                    cmd_duration_ms = (t_after - t_before) * 1000.0
                    timing = {
                        'offset_ms': (t_before - start_epoch) * 1000.0,
                        'duration_ms': cmd_duration_ms,
                    }
                    if edge.action == "on":
                        profile_on_cmd[edge.color].append(timing)
                        last_on_end[edge.color] = t_after
                    else:
                        profile_off_cmd[edge.color].append(timing)
                        on_end = last_on_end.pop(edge.color, None)
                        if on_end is not None:
                            actual_on_ms = (t_before - on_end) * 1000.0
                            profile_actual_on[edge.color].append({
                                'offset_ms': timing['offset_ms'],
                                'actual_on_ms': actual_on_ms,
                            })
                            pulses_executed[edge.color] = pulses_executed.get(edge.color, 0) + 1

                executed_edges += 1
        finally:
            if sys.platform.startswith('win') and time_period_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass

            for color, channel in self._active_channels:
                try:
                    if hasattr(self._scope, 'led_off_fast'):
                        self._scope.led_off_fast(channel=channel)
                    else:
                        self._scope.led_off(channel=channel)
                except Exception as ex:
                    logger.error(f"[STIMULATOR] {color}: failed to turn off LED in cleanup: {ex}")

            logger.info(
                f"[STIMULATOR] Merged scheduler ended after executing {executed_edges} edges. "
                f"Reason: {end_reason}"
            )
            if lateness_ms:
                logger.info(
                    f"[STIMULATOR] Timing lateness mean={sum(lateness_ms)/len(lateness_ms):.3f}ms "
                    f"max={max(lateness_ms):.3f}ms"
                )

            if profiling:
                self._save_profiling_data(
                    profile_on_cmd, profile_off_cmd, profile_actual_on,
                    pulses_executed, end_reason,
                )

    # ---- Profiling output ----

    @staticmethod
    def _timing_stats(values: list[float]) -> dict:
        """Compute summary statistics for a list of timing values (ms)."""
        if not values:
            return {}
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'p95': sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values),
            'p99': sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values),
        }

    @staticmethod
    def _write_timing_stats(f, label: str, values: list[float]):
        """Write a timing stats block to a file handle."""
        if not values:
            f.write(f"  {label}: no data\n")
            return
        stats = StimulationController._timing_stats(values)
        f.write(f"  {label}:\n")
        f.write(f"    count:  {stats['count']}\n")
        f.write(f"    mean:   {stats['mean']:.4f} ms\n")
        f.write(f"    std:    {stats['std']:.4f} ms\n")
        f.write(f"    min:    {stats['min']:.4f} ms\n")
        f.write(f"    max:    {stats['max']:.4f} ms\n")
        f.write(f"    p95:    {stats['p95']:.4f} ms\n")
        f.write(f"    p99:    {stats['p99']:.4f} ms\n")

    @staticmethod
    def _write_outlier_details(f, values: list[float], label: str,
                               expected_ms: float | None = None):
        """Write 3-sigma outlier analysis to a file handle."""
        if len(values) < 2:
            return
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        threshold = mean + 3 * std
        outliers = [(i, v) for i, v in enumerate(values) if v > threshold]
        if outliers:
            f.write(f"  {label} 3-sigma outliers (>{threshold:.4f} ms):\n")
            for idx, val in outliers:
                f.write(f"    pulse {idx}: {val:.4f} ms\n")
        if expected_ms is not None and expected_ms > 0:
            deviations = [(i, v) for i, v in enumerate(values)
                          if abs(v - expected_ms) > 3.0]
            if deviations:
                f.write(f"  {label} >3ms deviation from expected {expected_ms:.1f} ms:\n")
                for idx, val in deviations:
                    f.write(f"    pulse {idx}: {val:.4f} ms (delta={val - expected_ms:+.4f})\n")

    def _save_profiling_data(self, profile_on_cmd, profile_off_cmd,
                             profile_actual_on, pulses_executed, end_reason):
        """Save per-color profiling files to run_dir/stimulation_profile/."""
        try:
            profile_dir = self._run_dir / "stimulation_profile"
            profile_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            for color, _ in self._active_channels:
                stim_config = self._stim_configs.get(color, {})
                expected_pulse_ms = stim_config.get('pulse_width', 0)
                frequency = stim_config.get('frequency', 0)

                on_durations = [t['duration_ms'] for t in profile_on_cmd.get(color, [])]
                off_durations = [t['duration_ms'] for t in profile_off_cmd.get(color, [])]
                actual_on_times = [t['actual_on_ms'] for t in profile_actual_on.get(color, [])]

                filepath = profile_dir / f"stimulation_profile_{color}_{timestamp}.txt"
                with open(filepath, 'w') as f:
                    f.write(f"Stimulation Profile: {color}\n")
                    f.write(f"{'=' * 50}\n")
                    f.write(f"Frequency:       {frequency} Hz\n")
                    f.write(f"Pulse Width:     {expected_pulse_ms} ms\n")
                    f.write(f"Illumination:    {stim_config.get('illumination', '?')} mA\n")
                    f.write(f"Pulses executed: {pulses_executed.get(color, 0)}\n")
                    f.write(f"End reason:      {end_reason}\n")
                    f.write(f"\n--- Statistics ---\n")
                    self._write_timing_stats(f, "LED ON command time", on_durations)
                    self._write_timing_stats(f, "LED OFF command time", off_durations)
                    self._write_timing_stats(f, "Actual LED on-time", actual_on_times)

                    f.write(f"\n--- Outlier Analysis ---\n")
                    self._write_outlier_details(f, actual_on_times, "Actual on-time",
                                                expected_ms=expected_pulse_ms)
                    self._write_outlier_details(f, on_durations, "ON command")
                    self._write_outlier_details(f, off_durations, "OFF command")

                    f.write(f"\n--- Per-Pulse Event Log ---\n")
                    f.write(f"{'Pulse':>6} {'ON cmd (ms)':>12} {'OFF cmd (ms)':>13} {'Actual ON (ms)':>15}\n")
                    n_pulses = max(len(on_durations), len(off_durations), len(actual_on_times))
                    for i in range(n_pulses):
                        on_d = f"{on_durations[i]:.4f}" if i < len(on_durations) else "—"
                        off_d = f"{off_durations[i]:.4f}" if i < len(off_durations) else "—"
                        act = f"{actual_on_times[i]:.4f}" if i < len(actual_on_times) else "—"
                        f.write(f"{i:>6} {on_d:>12} {off_d:>13} {act:>15}\n")

                logger.info(f"[STIMULATOR] Profiling data saved to {filepath}")

        except Exception as ex:
            logger.error(f"[STIMULATOR] Failed to save profiling data: {ex}")
            # Dump summary to log as fallback
            for color, _ in self._active_channels:
                actual_on_times = [t['actual_on_ms'] for t in profile_actual_on.get(color, [])]
                if actual_on_times:
                    stats = self._timing_stats(actual_on_times)
                    logger.info(
                        f"[STIMULATOR] {color} actual on-time: "
                        f"mean={stats['mean']:.3f}ms std={stats['std']:.3f}ms "
                        f"min={stats['min']:.3f}ms max={stats['max']:.3f}ms"
                    )
