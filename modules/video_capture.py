# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Video capture and stimulation - extracted from sequenced_capture_runner.py.

Self-contained video recording: frame capture loop, stimulation threads,
and video writing (MP4 and TIFF-frame paths).
"""

import ctypes
import datetime
import pathlib
import queue
import statistics
import sys
import threading
import time
from typing import NamedTuple

import numpy as np

from lvp_logger import logger
import modules.image_save as image_save
from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.video_writer import VideoWriter, fps_from_frames


# One shared vocabulary of stim-schedule end_reason values, referenced by BOTH
# the producer (StimulationController.run) and the consumer classifier below so
# the two cannot drift. A reason is "clean" only when the schedule finished as
# intended: it ran every edge to completion, or it was deliberately stopped (the
# video is shorter than the stim schedule, or the run was cancelled). Every other
# value -- including the incomplete-by-default sentinel a never-finished run
# leaves behind -- means the sample received only a fraction of its configured
# pulses, so the recording earns a stim-status sidecar. The clean state is made
# unrepresentable unless earned: the default is INCOMPLETE, and a clean reason is
# assigned only at the point the schedule genuinely reaches that state.

# Earned-clean reasons.
STIM_END_SCHEDULE_COMPLETE = 'schedule_complete'  # dispatched every edge, no break
STIM_END_STOP_EVENT_SET = 'stop_event_set'  # stopped on purpose (short video / cancel)
STIM_END_STOP_BEFORE_START = 'stop_event_set_before_start'  # stopped before the first edge

# Not-clean reasons -- the sample was under-dosed.
STIM_END_INCOMPLETE = 'incomplete'  # default sentinel: run() never published a real reason
STIM_END_EMPTY_SCHEDULE = 'empty_schedule'  # enabled but zero edges built (misconfigured step)
STIM_END_DISPATCH_ERROR = 'dispatch_error'  # an LED edge raised mid-schedule
STIM_END_JOIN_TIMEOUT = 'join_timeout'  # scheduler thread still alive past its join
STIM_END_CAPTURE_FAULT = 'capture_fault'  # capture loop ended on a camera fault, not a normal end

_CLEAN_STIM_END_REASONS = frozenset(
    {
        STIM_END_SCHEDULE_COMPLETE,
        STIM_END_STOP_EVENT_SET,
        STIM_END_STOP_BEFORE_START,
    }
)


def _write_stim_status_sidecar(save_folder, name, stim_end_reason):
    """Drop a stim-status sidecar next to a recording when the schedule did not
    end cleanly.

    A stim schedule that ended on a fault (not clean completion nor an
    intentional stop) means the sample received only a fraction of its pulses,
    so the incomplete stimulation is recorded on disk rather than the run looking
    like a normal stim recording. This is the one place that classifies a reason
    and writes the file, so the zero-frame path and the normal write path cannot
    disagree about what counts as clean.

    No popup: an unattended protocol does not interrupt for a non-fatal issue.
    """
    if stim_end_reason is None or stim_end_reason in _CLEAN_STIM_END_REASONS:
        return
    if save_folder is None or name is None:
        logger.warning(
            f'[PROTOCOL-VIDEO] Stimulation ended early (end_reason={stim_end_reason}) '
            'but no save location is known, so no stim-status sidecar was written.'
        )
        return
    status_path = save_folder / f'{name}_stim_status.txt'
    try:
        status_path.write_text(f'Stimulation did not complete: end_reason={stim_end_reason}\n')
    except OSError as ex:
        logger.error(f'[PROTOCOL-VIDEO] Could not write stim status sidecar: {ex}')
    logger.warning(
        f'[PROTOCOL-VIDEO] Stimulation schedule for "{name}" ended early '
        f'(end_reason={stim_end_reason}); recording saved with a stim-status sidecar.'
    )


class VideoCaptureResult:
    """Result of a video capture session."""

    def __init__(
        self,
        captured_frames,
        calculated_fps,
        video_images,
        duration_sec,
        dropped_frames=0,
        stim_end_reason=None,
    ):
        self.captured_frames = captured_frames
        self.calculated_fps = calculated_fps
        self.video_images = video_images
        self.duration_sec = duration_sec
        # Frames the capture loop could not queue (consumer fell behind).
        # Surfaced at write time so a short recording is not silent.
        self.dropped_frames = dropped_frames
        # How the stimulation schedule ended (None if no stim ran). A non-clean
        # reason is recorded next to the video so an incomplete stim run is not
        # saved as a normal one.
        self.stim_end_reason = stim_end_reason


class VideoCaptureSession:
    """Manages a single video recording within a protocol step.

    Usage:
        session = VideoCaptureSession(scope, step, autogain_settings,
                                      is_protocol_running_fn, callbacks)
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
        *,
        stim_profiling: bool = False,
        run_dir: pathlib.Path | None = None,
        save_folder: pathlib.Path | None = None,
        name: str | None = None,
    ):
        self._scope = scope
        self._step = step
        self._autogain_settings = autogain_settings
        self._is_protocol_running = is_protocol_running_fn
        self._callbacks = callbacks
        self._leds_off = leds_off_fn
        self._stim_profiling = stim_profiling
        self._run_dir = run_dir
        # Save location for the zero-frame stim-status sidecar: when no frames are
        # captured there is no write_video call, but an incomplete stim still
        # dosed the sample wrong and must be recorded on disk.
        self._save_folder = save_folder
        self._name = name

        self._stim_start_event = threading.Event()
        self._stim_stop_event = threading.Event()
        # Distinguishes a camera-fault stop from a normal/intentional stop. Both
        # set _stim_stop_event, but only a fault means the schedule was truncated
        # against the sample's intent, so the scheduler reads this to classify.
        self._stim_fault_event = threading.Event()

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

        # Start one stimulation scheduler thread for all enabled channels.
        stim_thread = None
        scheduler = None
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
                name='stim-scheduler',
                args=(self._stim_start_event, self._stim_stop_event, self._stim_fault_event),
                daemon=True,
            )
            stim_thread.start()

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

        self._stim_stop_event.clear()
        self._stim_start_event.set()

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
                # Stop stim thread BEFORE turning off LEDs to prevent
                # stim pulses from re-enabling LEDs after leds_off()
                self._stim_stop_event.set()
                if stim_thread is not None:
                    stim_thread.join(timeout=2.0)
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
                # Mark the upcoming stim stop as a fault, not a normal end, so a
                # schedule truncated by this disconnect classifies incomplete
                # instead of looking like an intentional short recording. Set
                # before _stim_stop_event below so the scheduler sees the fault
                # the moment it observes the stop.
                self._stim_fault_event.set()
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

        self._stim_stop_event.set()
        self._stim_start_event.clear()

        if stim_thread is not None:
            stim_thread.join(timeout=5.0)

        # Snapshot the schedule's end_reason into a local that owns the
        # classification for THIS recording. A thread that exited published its
        # real reason in run()'s finally during the join above, so the snapshot
        # reads it. A wedged thread (join returned while still alive) is forced to a
        # join-timeout marker here. The snapshot is taken once and never re-read
        # from the cross-thread field afterward: if a wedged thread later unwedges
        # and its finally publishes a clean reason, it cannot reach back and stamp
        # this recording clean, so an under-dosed sample always reads incomplete.
        stim_end_reason = scheduler._end_reason if scheduler is not None else None
        if stim_thread is not None and stim_thread.is_alive():
            logger.warning('[STIMULATOR] Scheduler thread did not exit within 5s timeout')
            stim_end_reason = STIM_END_JOIN_TIMEOUT

        if captured_frames == 0:
            logger.warning(
                '[PROTOCOL] Zero frames captured during video recording - skipping write'
            )
            # No frames means no write_video call downstream, but an incomplete
            # stim still dosed the sample wrong. Record it here so a frame-less
            # recording is not silently treated as a clean stim run.
            if scheduler is not None:
                _write_stim_status_sidecar(self._save_folder, self._name, stim_end_reason)
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
            stim_end_reason=stim_end_reason,
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
            include_timestamp_overlay=True,
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

    # Flag an under-dosed sample when the stim schedule did not end cleanly. The
    # classification and the file write live in one shared helper so this path and
    # the zero-frame path in capture() cannot disagree about what counts as clean.
    _write_stim_status_sidecar(save_folder, name, result.stim_end_reason)

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

    def __init__(
        self,
        scope,
        stim_configs,
        *,
        profiling_enabled: bool = False,
        run_dir: pathlib.Path | None = None,
    ):
        self._scope = scope
        # Stim pulses fire while a protocol (or autofocus) owns the LED
        # lease. Capture that owner once -- not per pulse, which would add
        # lock latency to the timing-critical edge loop -- so each pulse is
        # attributed to the run and not refused as an out-of-turn write.
        # None when no run owns the LEDs (standalone video capture).
        self._lease_owner = scope.illumination.led_lease_owner
        self._stim_configs = stim_configs
        self._profiling_enabled = profiling_enabled and run_dir is not None
        self._run_dir = run_dir
        self._active_channels: list[tuple[str, int]] = []
        self._edges = self._build_edge_schedule()
        # How the stimulation actually ended, read by the recording to decide
        # whether to flag an under-dosed sample. Defaults to an INCOMPLETE
        # sentinel so a run that never reaches its finally (an early return, a
        # wedged thread) is never mistaken for clean; run() overwrites this with
        # a clean reason only once the schedule genuinely earns it.
        self._end_reason = STIM_END_INCOMPLETE

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
                logger.error(
                    f'[STIMULATOR] {color}: invalid frequency {frequency} Hz - must be > 0. Skipping stimulation.'
                )
                continue
            if not isinstance(pulse_width, (int, float)) or pulse_width <= 0:
                logger.error(
                    f'[STIMULATOR] {color}: invalid pulse_width {pulse_width} ms - must be > 0. Skipping stimulation.'
                )
                continue
            if not isinstance(pulse_count, int) or pulse_count <= 0:
                logger.error(
                    f'[STIMULATOR] {color}: invalid pulse_count {pulse_count} - must be > 0. Skipping stimulation.'
                )
                continue
            if not isinstance(illumination, (int, float)) or illumination <= 0:
                logger.error(
                    f'[STIMULATOR] {color}: invalid illumination {illumination} mA - must be > 0. Skipping stimulation.'
                )
                continue

            if illumination > self._MAX_STIM_CURRENT_MA:
                logger.warning(
                    f'[STIMULATOR] {color}: illumination {illumination}mA exceeds max {self._MAX_STIM_CURRENT_MA}mA. Clamping.'
                )
                illumination = self._MAX_STIM_CURRENT_MA

            period_s = 1.0 / float(frequency)
            pulse_s = float(pulse_width) / 1000.0
            if pulse_s >= period_s:
                logger.warning(
                    f'[STIMULATOR] {color}: pulse_width ({pulse_width}ms) >= period ({period_s * 1000:.1f}ms). Clamping pulse to 90% of period.'
                )
                pulse_s = period_s * 0.9

            channel = self._scope.illumination.color2ch(color=color)
            active_channels[color] = channel

            for i in range(pulse_count):
                on_time = i * period_s
                off_time = on_time + pulse_s
                edges.append(
                    StimEdge(
                        target_offset_s=on_time,
                        action='on',
                        channel=channel,
                        mA=illumination,
                        color=color,
                    )
                )
                edges.append(
                    StimEdge(
                        target_offset_s=off_time,
                        action='off',
                        channel=channel,
                        mA=None,
                        color=color,
                    )
                )

        self._active_channels = sorted(
            active_channels.items(),
            key=lambda item: item[1],
        )
        edges.sort(
            key=lambda edge: (
                round(edge.target_offset_s / self._SORT_EPSILON_S),
                0 if edge.action == 'off' else 1,
                edge.channel,
            )
        )
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
            # else: busy-wait the last <3ms. A time.sleep(100us) here yields
            # the GIL and the OS scheduler can take 100us-20+ms to resume us,
            # lengthening the next pulse by whatever it waits. Matters for
            # OFF edges that are 10 ms after their ON: measured on 2026-04-19
            # that a yielding spin produced pulse-width stddev 5.9 ms and
            # worst-case 26.3 ms on a 10 ms target. Busy-waiting matches OG's
            # approach and brings pulse-width stddev back to ~1 ms.

    def _dispatch_edge(self, edge: StimEdge) -> float:
        """Dispatch a single stim edge. Returns perf_counter timestamp after the call."""
        if edge.action == 'on':
            if hasattr(self._scope, 'led_on_fast'):
                self._scope.illumination.led_on_fast(channel=edge.channel, mA=edge.mA)
            else:
                self._scope.illumination.led_on(
                    channel=edge.channel, mA=edge.mA, _lease_owner=self._lease_owner
                )
        else:
            if hasattr(self._scope, 'led_off_fast'):
                self._scope.illumination.led_off_fast(channel=edge.channel)
            else:
                self._scope.illumination.led_off(
                    channel=edge.channel, _lease_owner=self._lease_owner
                )
        return time.perf_counter()

    @staticmethod
    def _stop_reason(fault_event: threading.Event | None) -> str:
        """Classify a stop: a camera fault truncated the schedule (incomplete) vs
        a normal/intentional stop (clean)."""
        if fault_event is not None and fault_event.is_set():
            return STIM_END_CAPTURE_FAULT
        return STIM_END_STOP_EVENT_SET

    def run(
        self,
        start_event: threading.Event,
        stop_event: threading.Event,
        fault_event: threading.Event | None = None,
    ):
        """Thread target. Runs a merged pulse-edge schedule for all channels.

        fault_event distinguishes a camera-fault stop from a normal/intentional
        one: both trip stop_event, but a fault means the schedule was truncated
        against the sample's intent, so it classifies incomplete.
        """
        if not self._edges:
            # An enabled stim that built zero edges is a misconfigured step (e.g.
            # pulse_count or frequency out of range): it delivers no pulses, so it
            # returns here BEFORE the try/finally and must not read clean.
            self._end_reason = STIM_END_EMPTY_SCHEDULE
            return

        enabled_colors = [color for color, _ in self._active_channels]
        logger.info(f'[STIMULATOR] Starting merged scheduler for {enabled_colors}')

        time_period_set = False
        # Start incomplete; only the for-else (every edge dispatched) earns clean.
        end_reason = STIM_END_INCOMPLETE
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
                    end_reason = STIM_END_STOP_BEFORE_START
                    return

            if stop_event.is_set():
                end_reason = STIM_END_STOP_BEFORE_START
                return

            start_epoch = time.perf_counter()

            for edge in self._edges:
                if stop_event.is_set():
                    end_reason = self._stop_reason(fault_event)
                    break

                if not self._wait_until(start_epoch + edge.target_offset_s, stop_event):
                    end_reason = self._stop_reason(fault_event)
                    break

                if stop_event.is_set():
                    end_reason = self._stop_reason(fault_event)
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
                    end_reason = STIM_END_DISPATCH_ERROR
                    logger.error(f'[STIMULATOR] {edge.color}: {edge.action} edge failed: {ex}')
                    break

                if profiling:
                    cmd_duration_ms = (t_after - t_before) * 1000.0
                    timing = {
                        'offset_ms': (t_before - start_epoch) * 1000.0,
                        'duration_ms': cmd_duration_ms,
                    }
                    if edge.action == 'on':
                        profile_on_cmd[edge.color].append(timing)
                        last_on_end[edge.color] = t_after
                    else:
                        profile_off_cmd[edge.color].append(timing)
                        on_end = last_on_end.pop(edge.color, None)
                        if on_end is not None:
                            actual_on_ms = (t_before - on_end) * 1000.0
                            profile_actual_on[edge.color].append(
                                {
                                    'offset_ms': timing['offset_ms'],
                                    'actual_on_ms': actual_on_ms,
                                }
                            )
                            pulses_executed[edge.color] = pulses_executed.get(edge.color, 0) + 1

                executed_edges += 1
            else:
                # for-else: reached only when no break fired, i.e. every edge was
                # dispatched. This is the single place the schedule earns a clean
                # reason -- the sample received its full configured dose.
                end_reason = STIM_END_SCHEDULE_COMPLETE
        finally:
            # Publish the final reason before cleanup so the recording can stamp it.
            self._end_reason = end_reason
            if sys.platform.startswith('win') and time_period_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass

            for color, channel in self._active_channels:
                try:
                    if hasattr(self._scope, 'led_off_fast'):
                        self._scope.illumination.led_off_fast(channel=channel)
                    else:
                        self._scope.illumination.led_off(
                            channel=channel, _lease_owner=self._lease_owner
                        )
                except Exception as ex:
                    logger.error(f'[STIMULATOR] {color}: failed to turn off LED in cleanup: {ex}')

            logger.info(
                f'[STIMULATOR] Merged scheduler ended after executing {executed_edges} edges. '
                f'Reason: {end_reason}'
            )
            if lateness_ms:
                logger.info(
                    f'[STIMULATOR] Timing lateness mean={sum(lateness_ms) / len(lateness_ms):.3f}ms '
                    f'max={max(lateness_ms):.3f}ms'
                )

            if profiling:
                self._save_profiling_data(
                    profile_on_cmd,
                    profile_off_cmd,
                    profile_actual_on,
                    pulses_executed,
                    end_reason,
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
            f.write(f'  {label}: no data\n')
            return
        stats = StimulationController._timing_stats(values)
        f.write(f'  {label}:\n')
        f.write(f'    count:  {stats["count"]}\n')
        f.write(f'    mean:   {stats["mean"]:.4f} ms\n')
        f.write(f'    std:    {stats["std"]:.4f} ms\n')
        f.write(f'    min:    {stats["min"]:.4f} ms\n')
        f.write(f'    max:    {stats["max"]:.4f} ms\n')
        f.write(f'    p95:    {stats["p95"]:.4f} ms\n')
        f.write(f'    p99:    {stats["p99"]:.4f} ms\n')

    @staticmethod
    def _write_outlier_details(
        f, values: list[float], label: str, expected_ms: float | None = None
    ):
        """Write 3-sigma outlier analysis to a file handle."""
        if len(values) < 2:
            return
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        threshold = mean + 3 * std
        outliers = [(i, v) for i, v in enumerate(values) if v > threshold]
        if outliers:
            f.write(f'  {label} 3-sigma outliers (>{threshold:.4f} ms):\n')
            for idx, val in outliers:
                f.write(f'    pulse {idx}: {val:.4f} ms\n')
        if expected_ms is not None and expected_ms > 0:
            deviations = [(i, v) for i, v in enumerate(values) if abs(v - expected_ms) > 3.0]
            if deviations:
                f.write(f'  {label} >3ms deviation from expected {expected_ms:.1f} ms:\n')
                for idx, val in deviations:
                    f.write(f'    pulse {idx}: {val:.4f} ms (delta={val - expected_ms:+.4f})\n')

    def _save_profiling_data(
        self, profile_on_cmd, profile_off_cmd, profile_actual_on, pulses_executed, end_reason
    ):
        """Save per-color profiling files to run_dir/stimulation_profile/."""
        try:
            profile_dir = self._run_dir / 'stimulation_profile'
            profile_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            for color, _ in self._active_channels:
                stim_config = self._stim_configs.get(color, {})
                expected_pulse_ms = stim_config.get('pulse_width', 0)
                frequency = stim_config.get('frequency', 0)

                on_durations = [t['duration_ms'] for t in profile_on_cmd.get(color, [])]
                off_durations = [t['duration_ms'] for t in profile_off_cmd.get(color, [])]
                actual_on_times = [t['actual_on_ms'] for t in profile_actual_on.get(color, [])]

                filepath = profile_dir / f'stimulation_profile_{color}_{timestamp}.txt'
                with open(filepath, 'w') as f:
                    f.write(f'Stimulation Profile: {color}\n')
                    f.write(f'{"=" * 50}\n')
                    f.write(f'Frequency:       {frequency} Hz\n')
                    f.write(f'Pulse Width:     {expected_pulse_ms} ms\n')
                    f.write(f'Illumination:    {stim_config.get("illumination", "?")} mA\n')
                    f.write(f'Pulses executed: {pulses_executed.get(color, 0)}\n')
                    f.write(f'End reason:      {end_reason}\n')
                    f.write('\n--- Statistics ---\n')
                    self._write_timing_stats(f, 'LED ON command time', on_durations)
                    self._write_timing_stats(f, 'LED OFF command time', off_durations)
                    self._write_timing_stats(f, 'Actual LED on-time', actual_on_times)

                    f.write('\n--- Outlier Analysis ---\n')
                    self._write_outlier_details(
                        f, actual_on_times, 'Actual on-time', expected_ms=expected_pulse_ms
                    )
                    self._write_outlier_details(f, on_durations, 'ON command')
                    self._write_outlier_details(f, off_durations, 'OFF command')

                    f.write('\n--- Per-Pulse Event Log ---\n')
                    f.write(
                        f'{"Pulse":>6} {"ON cmd (ms)":>12} {"OFF cmd (ms)":>13} {"Actual ON (ms)":>15}\n'
                    )
                    n_pulses = max(len(on_durations), len(off_durations), len(actual_on_times))
                    for i in range(n_pulses):
                        on_d = f'{on_durations[i]:.4f}' if i < len(on_durations) else '--'
                        off_d = f'{off_durations[i]:.4f}' if i < len(off_durations) else '--'
                        act = f'{actual_on_times[i]:.4f}' if i < len(actual_on_times) else '--'
                        f.write(f'{i:>6} {on_d:>12} {off_d:>13} {act:>15}\n')

                logger.info(f'[STIMULATOR] Profiling data saved to {filepath}')

        except Exception as ex:
            logger.error(f'[STIMULATOR] Failed to save profiling data: {ex}')
            # Dump summary to log as fallback
            for color, _ in self._active_channels:
                actual_on_times = [t['actual_on_ms'] for t in profile_actual_on.get(color, [])]
                if actual_on_times:
                    stats = self._timing_stats(actual_on_times)
                    logger.info(
                        f'[STIMULATOR] {color} actual on-time: '
                        f'mean={stats["mean"]:.3f}ms std={stats["std"]:.3f}ms '
                        f'min={stats["min"]:.3f}ms max={stats["max"]:.3f}ms'
                    )
