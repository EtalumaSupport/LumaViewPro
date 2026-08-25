# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Protocol video step recording: one step drives the engine.

The protocol-side twin of the manual controller
(``modules/manual_recording.py``): builds the immutable
``RecordingConfig`` from run-scoped snapshots, registers the camera
frame listener, drives ``VideoRecordingEngine`` for one video step on
the protocol thread, and finishes off that thread (MP4 close,
execution-record row, drop report). The step's capture cadence is the
engine's slot selection -- there is no polled frame sampler and no
pacing sleep; Stop closes selection within one wait tick.

Claim topology: the protocol run already holds the session's
exclusive-activity claim, which fences recordings and runs against each
other at the Session tier. A per-step recording nests INSIDE that
running claim, so the engine receives a nested claim handle that always
grants and releases as a no-op -- the run's own claim is the fence.
Steps execute sequentially on the protocol thread; the only overlap is
a previous step's drain tail against the next step, which shares no
state with it.

Notification policy: this controller posts through the central
``NotificationCenter`` unconditionally; the center's protocol mute owns
non-fatal suppression during a run, and fatal events (run aborts,
writer-lane death via the engine) pass through it.
"""

import datetime
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lvp_logger import logger, version as lvp_version
from modules.common_utils import (
    DISK_FLOOR_CHECK_INTERVAL_S,
    MIN_PER_WRITE_DISK_MB,
    MIN_REQUIRED_DISK_MB,
    check_disk_space_ok,
    estimate_step_write_mb,
)
import modules.image_save as image_save
import modules.image_utils as image_utils
from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.notification_center import notifications
from modules.recording_frames import (
    CameraTickRebaser,
    orient_and_fit,
    protocol_frame_filename_template,
    resolve_recording_pixel_size,
    tiff_frame_metadata,
)
from modules.recording_manifest import gather_host_provenance
from modules.video_cadence import (
    StallWatch,
    effective_recording_fps,
    prologue_stall_threshold_s,
    stall_threshold_s,
)
from modules.video_recording import (
    END_REASON_START_FAILED,
    RecordingConfig,
    VideoRecordingEngine,
)
from modules.video_writer import VideoWriter

# Outcomes of run_blocking(), consumed by ProtocolImageWriter's video leg.
COMPLETED = 'completed'  # recording ended normally (full or honest short delivery)
CANCELLED = 'cancelled'  # run stop/abort ended the step early; kept frames are final
NO_FRAMES = 'no_frames'  # the camera delivered nothing -- a capture failure (strike)
CAMERA_LOST = 'camera_lost'  # camera went inactive mid-step; kept frames are final (strike)
ABORTED = 'aborted'  # the controller aborted the run (disk); no strike, no row here

# The wait loop's tick: how quickly Stop/abort is honored during a video
# step, and the ceiling on how stale the recording title can be beyond
# its 1 s update throttle.
_WAIT_TICK_S = 0.1
_TITLE_UPDATE_INTERVAL_S = 1.0


class _NestedClaim:
    """Always-granting claim for a recording nested inside a protocol run.

    The run's own session claim (held for the whole run) is the real
    exclusivity fence; the engine's claim acquisition inside the run
    must not contend with it. Release is a no-op for the same reason:
    the run releases its claim at run end, never per step.
    """

    def try_claim(self, owner: str) -> bool:
        return True

    def release(self, owner: str) -> None:
        return None


class ProtocolVideoStep:
    """One protocol video step: snapshot, record via the engine, finish.

    Args:
        scope: The Lumascope instance (frame listener, camera identity,
            imaging prologue).
        step: The protocol step (reads Video Config, Color, False_Color,
            Auto_Gain, Exposure).
        save_folder: The run's save folder for this step.
        name: The step's artifact base name (caller tokens preserved).
        video_as_frames: Run-level flag -- TIFF frames vs MP4.
        capture_config: The run's frozen ImageCaptureConfig (encoding,
            depth).
        timestamp_overlay: Run-scoped snapshot of the overlay choice.
        global_max_fps: Run-scoped snapshot of the global "Video max
            FPS" cap (0 = uncapped); D15 -- never read live mid-run.
        autogain_settings: The runner's autogain settings dict.
        callbacks: The run callbacks dict (set_recording_title,
            set_writing_title, reset_title used here); dispatched via
            the UI scheduler.
        aborted_event: The run's abort event; checked every wait tick.
        is_run_in_progress: Callable; False ends the step early.
        abort_run_fatal: PIW's fatal-abort funnel, for disk faults.
        abort_run_on_writer_death: Arms the run abort after the engine
            has already surfaced writer-lane death at critical severity
            (no second popup).
        record_step_row: Records the finished step's execution-record
            row: ``record_step_row(capture_result_file_name, frame_count,
            duration_sec, timestamp)``.
        record_dropped_capture: Records a no-artifact row:
            ``record_dropped_capture(reason, capture_time)``.
        clock: Injectable time source (seconds); tests drive it.
    """

    def __init__(
        self,
        *,
        scope: Any,
        step: Any,
        save_folder: Path,
        name: str,
        video_as_frames: bool,
        capture_config: Any,
        timestamp_overlay: bool,
        global_max_fps: float,
        autogain_settings: dict,
        callbacks: dict,
        aborted_event: threading.Event,
        is_run_in_progress: Callable[[], bool],
        abort_run_fatal: Callable[[str, str, str], None],
        abort_run_on_writer_death: Callable[[], None],
        record_step_row: Callable[..., None],
        record_dropped_capture: Callable[..., None],
        clock: Callable[[], float] = time.time,
    ):
        self._scope = scope
        self._step = step
        self._save_folder = Path(save_folder)
        self._name = name
        self._video_as_frames = video_as_frames
        self._capture_config = capture_config
        self._timestamp_overlay = timestamp_overlay
        self._global_max_fps = global_max_fps
        self._autogain_settings = autogain_settings
        self._callbacks = callbacks
        self._aborted = aborted_event
        self._is_run_in_progress = is_run_in_progress
        self._abort_run_fatal = abort_run_fatal
        self._abort_run_on_writer_death = abort_run_on_writer_death
        self._record_step_row = record_step_row
        self._record_dropped_capture = record_dropped_capture
        self._clock = clock

        self._engine: VideoRecordingEngine | None = None
        self._writer: VideoWriter | None = None
        self._rebaser: CameraTickRebaser | None = None
        self._tick_freq_hz: float | None = None
        self._output_dir: Path | None = None
        self._frames_seen = 0
        self._last_disk_check_ts = 0.0
        self._start_dt: datetime.datetime | None = None
        self._finish_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Drain-state surface (the runner's end-of-run wait and the app-close
    # gate read these)
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """True until the drain and the post-drain finish complete."""
        thread = self._finish_thread
        engine = self._engine
        return (engine is not None and (engine.is_recording or engine.is_draining)) or (
            thread is not None and thread.is_alive()
        )

    @property
    def pending_writes(self) -> int:
        """Frames enqueued but not yet on disk."""
        return self._engine.pending_writes if self._engine is not None else 0

    def wait_until_finished(self, timeout: float | None = None) -> bool:
        """Block until the drain and finish complete; True when finished."""
        thread = self._finish_thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def discard_pending(self) -> None:
        """Drop the unwritten backlog loudly (the app-close discard path)."""
        if self._engine is not None:
            self._engine.discard_pending()

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def run_blocking(self) -> str:
        """Record this video step; blocks the protocol thread until
        selection closes (the drain finishes on its own).

        Returns one of COMPLETED / CANCELLED / NO_FRAMES / CAMERA_LOST /
        ABORTED.
        """
        step = self._step
        video_config = step['Video Config']
        duration_s = float(video_config['duration'])
        fps = effective_recording_fps(float(video_config['fps']), self._global_max_fps)

        # Per-step disk pre-flight at the effective rate: the whole-run
        # estimate ran at scan start, but a long recording must re-check
        # against what is free NOW, before frames start landing.
        required_mb = max(
            MIN_PER_WRITE_DISK_MB,
            estimate_step_write_mb(
                step,
                video_as_frames=self._video_as_frames,
                global_max_fps=self._global_max_fps,
            ),
        )
        try:
            ok, free_mb = check_disk_space_ok(self._save_folder, required_mb)
        except Exception as e:
            logger.warning(f'[PROTOCOL-VIDEO] Pre-flight disk probe failed (proceeding): {e}')
            ok, free_mb = True, 0.0
        if not ok:
            self._abort_run_fatal(
                'FileIO',
                'Disk Space Critical',
                f'Only {free_mb:.0f} MB free -- the video step needs ~{required_mb:.0f} MB. '
                'Aborting protocol to prevent data loss.',
            )
            return ABORTED

        prologue_outcome = self._prologue(step)
        if prologue_outcome is not None:
            return prologue_outcome

        scope = self._scope
        identity = scope.imaging.camera_identity
        frame_size = scope.imaging.frame_size_cached
        self._tick_freq_hz = identity['timestamp_tick_frequency_hz']
        # One scale snapshot per step, alongside the other start-of-recording
        # camera facts: the objective cannot change while a step records.
        self._pixel_size_um = resolve_recording_pixel_size(scope)
        self._rebaser = CameraTickRebaser(self._tick_freq_hz, self._clock)
        self._start_dt = datetime.datetime.now()

        if self._video_as_frames:
            self._output_dir = self._save_folder / self._name
        else:
            self._output_dir = self._save_folder
        self._output_dir.mkdir(exist_ok=True, parents=True)

        false_color_on = bool(step['False_Color'])
        config = RecordingConfig(
            fps=fps,
            duration_s=duration_s,
            width=frame_size['width'],
            height=frame_size['height'],
            bit_depth=self._capture_config.capture_depth,
            output_dir=self._output_dir,
            filename_template=protocol_frame_filename_template(self._name),
            timestamp_overlay=self._timestamp_overlay,
            manifest_extra={
                'step_name': self._name,
                'channel_color': step['Color'] if false_color_on else None,
                'camera': {
                    'model': identity['model'],
                    'serial': identity['serial'],
                    'timestamp_tick_hz': self._tick_freq_hz,
                },
                'provenance': {
                    'host': gather_host_provenance(),
                    'software': {'lvp_version': lvp_version},
                },
            },
            # MP4 artifacts share the step's flat folder across steps and
            # scans, so the manifest is named per recording; the frames
            # leg owns a per-recording folder and keeps the default name.
            manifest_filename=(
                'recording_manifest.json'
                if self._video_as_frames
                else f'{self._name}_manifest.json'
            ),
        )

        if not self._video_as_frames:
            self._writer = VideoWriter(
                output_path=self._output_dir / f'{self._name}.mp4',
                fps=fps,
                width=frame_size['width'],
                height=frame_size['height'],
                # None (true grayscale) when false color is off -- the
                # same contract the manual leg encodes with; the old
                # protocol leg's 'BF' gray-colormap encode diverged from
                # it for identical pixels.
                color=step['Color'] if false_color_on else None,
                include_timestamp_overlay=self._timestamp_overlay,
                vfr=True,
            )

        engine = VideoRecordingEngine(
            write_frame=self._write_frame,
            claim=_NestedClaim(),
            clock=self._clock,
            notify=notifications,
        )
        engine.start(config)
        try:
            self._engine = engine
            scope.imaging.add_frame_listener(
                self._on_camera_frame, name=f'protocol_video:{self._name}'
            )
        except BaseException:
            # The nested claim leaks nothing to the session, but a step left
            # recording never satisfies the runner's end-of-run wait, so the
            # whole run hangs on a step that never began.
            self._unwind_failed_start(engine)
            raise
        logger.info(
            f'[PROTOCOL-VIDEO] Recording started: {fps:.2f} fps, {duration_s:.0f} s, '
            f'{"frames" if self._video_as_frames else "mp4"} -> {self._output_dir}'
        )

        outcome, end_reason = self._wait_for_recording(
            engine,
            duration_s,
            stall_threshold_s(fps, float(step['Exposure']) / 1000.0),
        )

        try:
            scope.imaging.remove_frame_listener(self._on_camera_frame)
        except Exception as e:
            logger.warning(f'[PROTOCOL-VIDEO] remove_frame_listener failed: {e}')
        engine.stop(end_reason)

        if self._frames_seen == 0:
            # The camera delivered nothing for the whole step -- a capture
            # failure, not a short recording. The (empty) drain finishes
            # inline; the caller owns the failure row and the strike.
            engine.wait_for_drain(timeout=5.0)
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception as e:
                    logger.warning(f'[PROTOCOL-VIDEO] Writer close after empty step: {e}')
            self._reset_title()
            # A user Stop or a run abort that arrived before any frame is
            # not a capture failure: the early exit keeps its meaning, or
            # a zero-frame Stop would land a bogus strike toward the
            # 3-strike run abort. A camera lost before delivering anything
            # IS "delivered nothing" -- it keeps the failure row + strike.
            if outcome in (CANCELLED, ABORTED):
                return outcome
            return NO_FRAMES

        self._finish_thread = threading.Thread(
            target=self._finish_after_drain,
            name='ProtocolVideoFinish',
            daemon=True,
        )
        self._finish_thread.start()
        return outcome

    # ------------------------------------------------------------------
    # Prologue (validity pre-roll + autogain first frame), stop-checkable
    # ------------------------------------------------------------------

    def _prologue(self, step) -> str | None:
        """Drain stale frames, settle, arm the first-frame autogain.

        Returns None when the recording may start, or the outcome to
        return when it may not: CANCELLED on a run stop (checked first,
        so Stop always wins), NO_FRAMES when the feed is dead. Dead
        means zero frame ARRIVALS for the stall threshold -- arrivals,
        deliberately not validity progress: a stuck settle check pins
        ``frames_until_valid`` while frames keep arriving, and blaming
        the camera for a motion fault would send the run's abort message
        pointing at the wrong hardware.
        """
        scope = self._scope
        exposure_s = float(step['Exposure']) / 1000.0
        watch = StallWatch(prologue_stall_threshold_s(exposure_s))
        arrivals = 0
        while scope.imaging.frames_until_valid() > 0:
            if self._stop_requested():
                return CANCELLED
            if watch.stalled(arrivals, self._clock()):
                logger.error(
                    '[PROTOCOL-VIDEO] No camera frames arrived during the '
                    'pre-recording drain (stall threshold '
                    f'{prologue_stall_threshold_s(exposure_s):.0f} s); '
                    'treating the feed as dead'
                )
                return NO_FRAMES
            if scope.imaging.get_image(force_new_capture=True) is not None:
                arrivals += 1
        time.sleep(max(step['Exposure'] / 1000, 0.05))

        if step['Auto_Gain']:
            # Run-internal camera writes bind the impls: the camera lane
            # is disabled for the whole run, so the public dispatchers
            # would refuse their own run's work.
            scope.imaging._set_auto_gain_impl(state=False, settings=self._autogain_settings)
            scope.imaging._auto_gain_once_impl(
                state=True,
                target_brightness=self._autogain_settings['target_brightness'],
                min_gain_db=self._autogain_settings['min_gain_db'],
                max_gain_db=self._autogain_settings['max_gain_db'],
                ae_max_exposure_ms=self._autogain_settings.get('max_exposure_ms'),
            )
        return CANCELLED if self._stop_requested() else None

    def _stop_requested(self) -> bool:
        return self._aborted.is_set() or not self._is_run_in_progress()

    # ------------------------------------------------------------------
    # The wait loop (protocol thread)
    # ------------------------------------------------------------------

    def _wait_for_recording(
        self, engine: VideoRecordingEngine, duration_s: float, stall_threshold: float
    ) -> tuple[str, str]:
        """Poll until the recording ends; say how and why.

        Returns ``(outcome, end_reason)`` as one value so the manifest's
        reason can never drift from the outcome that produced it. Check
        order is the precedence: Stop always wins, the disconnect latch
        is the fast camera-death path, and the stall watch catches the
        feed that dies without an event -- delivery just stops while
        ``active_cached`` stays True.
        """
        start_ts = self._clock()
        last_title_ts = 0.0
        outcome, end_reason = COMPLETED, 'duration_elapsed'
        watch = StallWatch(stall_threshold)
        while engine.is_recording:
            now = self._clock()
            elapsed = now - start_ts
            if self._stop_requested():
                outcome, end_reason = CANCELLED, 'run_stop'
                break
            if not self._scope.imaging.active_cached:
                logger.warning(
                    '[PROTOCOL-VIDEO] Camera went inactive mid-step; ending the recording'
                )
                outcome, end_reason = CAMERA_LOST, 'camera_disconnected'
                break
            if watch.stalled(self._frames_seen, now):
                logger.warning(
                    '[PROTOCOL-VIDEO] Camera feed stalled: no frames for '
                    f'{stall_threshold:.0f} s with the camera still active; '
                    'ending the recording'
                )
                outcome, end_reason = CAMERA_LOST, 'camera_stalled'
                break
            if elapsed >= duration_s:
                # The frame budget is the primary boundary; this wall cap
                # ends a step whose camera delivers below the configured
                # rate (the budget would otherwise never fill). Kept
                # frames are an honest short delivery in the manifest.
                break
            if now - last_title_ts >= _TITLE_UPDATE_INTERVAL_S:
                last_title_ts = now
                self._set_title('set_recording_title', elapsed_sec=elapsed, total_sec=duration_s)
            time.sleep(_WAIT_TICK_S)
        return outcome, end_reason

    # ------------------------------------------------------------------
    # Camera-thread ingest
    # ------------------------------------------------------------------

    def _on_camera_frame(self, image, timestamp, chunks) -> None:
        """SDK-thread listener: rebase the timestamp, offer to the engine."""
        engine = self._engine
        if engine is None or not engine.is_recording:
            return
        self._frames_seen += 1
        engine.ingest_frame(image, self._rebaser.frame_time_s(timestamp, chunks), chunks)

    # ------------------------------------------------------------------
    # Writer-lane edge
    # ------------------------------------------------------------------

    def _write_frame(self, image, timestamp_s, frame_number, config, chunks) -> Path:
        """Write one kept frame as its final artifact (runs on the lane)."""
        self._check_disk_floor(config)

        image = orient_and_fit(image, config.width, config.height)

        step = self._step
        if self._video_as_frames:
            if config.bit_depth == 8 and image.dtype != np.uint8:
                image = image_utils.convert_to_8bit(image, config.bit_depth)
            metadata, _ts_filename = tiff_frame_metadata(
                timestamp_s, frame_number, chunks, self._tick_freq_hz, self._pixel_size_um
            )
            file_loc = config.output_dir / config.filename_template.format(n=frame_number)
            image_save.write_video_frame(
                frame=image,
                file_loc=file_loc,
                metadata=metadata,
                channel=step['Color'],
                false_color_on=bool(step['False_Color']),
                save_encoding=self._capture_config.save_encoding,
                capture_depth=self._capture_config.capture_depth,
            )
            return file_loc

        writer = self._writer
        significant_bits = self._capture_config.capture_depth if image.dtype != np.uint8 else None
        writer.add_frame(image=image, timestamp=timestamp_s, significant_bits=significant_bits)
        return writer.output_path

    def _check_disk_floor(self, config) -> None:
        """Rolling floor probe on the write lane; a breach aborts the RUN.

        The protocol's disk contract is run-fatal (matching the per-write
        guard the stills lane keeps): a disk that hits the floor mid-step
        aborts the whole run loudly rather than letting later steps fail
        one by one. Frames already on disk stay.
        """
        now = self._clock()
        if now - self._last_disk_check_ts < DISK_FLOOR_CHECK_INTERVAL_S:
            return
        self._last_disk_check_ts = now
        try:
            ok, free_mb = check_disk_space_ok(config.output_dir, MIN_REQUIRED_DISK_MB)
        except Exception as e:
            logger.warning(f'[PROTOCOL-VIDEO] Disk-floor probe failed: {e}')
            return
        if not ok and self._engine is not None and self._engine.is_recording:
            logger.error(
                f'[PROTOCOL-VIDEO] Free disk fell to {free_mb:.0f} MB (floor '
                f'{MIN_REQUIRED_DISK_MB} MB); aborting the run'
            )
            self._engine.stop('disk_floor')
            self._abort_run_fatal(
                'FileIO',
                'Disk Space Critical',
                f'Free disk fell to {free_mb:.0f} MB during a video step. '
                'Aborting protocol to prevent data loss.',
            )

    # ------------------------------------------------------------------
    # Post-drain finish (per-step thread; the protocol thread moves on)
    # ------------------------------------------------------------------

    def _unwind_failed_start(self, engine: VideoRecordingEngine) -> None:
        """Undo a step start that raised after the engine committed.

        Order matters: stop delivering frames, then end the recording so
        the writer lane drains, and only then dispose the artifact --
        disposing first would leave the lane writing into a closed encoder.
        """
        try:
            self._scope.imaging.remove_frame_listener(self._on_camera_frame)
        except Exception as e:
            logger.debug(f'[PROTOCOL-VIDEO] listener removal during start unwind: {e}')

        try:
            engine.stop(END_REASON_START_FAILED)
        except Exception:
            logger.exception('[PROTOCOL-VIDEO] engine did not end cleanly during start unwind')

        if self._writer is not None:
            try:
                self._writer.close()
                if self._writer.frame_count == 0:
                    self._writer.output_path.unlink(missing_ok=True)
            except Exception:
                logger.exception('[PROTOCOL-VIDEO] writer disposal failed during start unwind')
            self._writer = None

        self._engine = None

    def _finish_after_drain(self) -> None:
        """Wait out the drain, close artifacts, record the row, report."""
        engine = self._engine
        total = max(1, engine.frames_selected)
        while not engine.wait_for_drain(timeout=1.0):
            if 'set_writing_title' in self._callbacks:
                done = total - engine.pending_writes
                self._set_title('set_writing_title', progress=done / total * 100)

        result = None
        writer_dropped = 0
        try:
            # Close the encoder before reading measured truth: an unclosed
            # container is a corrupt file on disk, while the result is only
            # a report -- and result() raises when the engine never finished
            # finalizing, which is exactly when the close matters most.
            if self._writer is not None:
                self._writer.close()
                # The MP4 writer swallows per-frame encode errors into its
                # own counter; fold them into the user-facing total. The
                # manifest carries the engine-counted failures.
                writer_dropped = self._writer.dropped_frames
            result = engine.result()
        except Exception:
            logger.exception('[PROTOCOL-VIDEO] Post-drain finish failed')
            notifications.error(
                'Protocol',
                'Video Finalize Failed',
                'A video step finished but its output could not be fully '
                'assembled. Frames already written are on disk; check the log.',
            )
        finally:
            self._reset_title()
            if result is None:
                # The finish failed before any measured truth existed. The
                # step still owes the run a row: a video step that vanishes
                # from the execution record reads as one that never ran.
                self._record_dropped_capture(
                    reason='video_finalize_failed', capture_time=self._start_dt
                )
            else:
                dropped = result.write_failures + writer_dropped
                if result.aborted:
                    # The engine already surfaced writer-lane death at
                    # critical severity; arm the run abort without a second
                    # popup and leave an honest no-artifact row.
                    self._abort_run_on_writer_death()
                    self._record_dropped_capture(
                        reason='video_write_failed', capture_time=self._start_dt
                    )
                elif result.frames_written == 0:
                    # The row's reason is the engine's recorded end reason --
                    # a camera death and a user stop must not share a label.
                    self._record_dropped_capture(
                        reason=f'video_{result.end_reason}', capture_time=self._start_dt
                    )
                    # No file exists to name: the mp4 muxer writes nothing at
                    # all for an empty stream, so the configured output path
                    # was never created. Announcing it would contradict the
                    # dropped-capture row recorded immediately above.
                    logger.info(
                        '[PROTOCOL-VIDEO] No video written: the step captured 0 '
                        'frames, so no file was produced'
                    )
                else:
                    if self._writer is not None:
                        artifact_name = self._writer.output_path.name
                        logger.info(f'[PROTOCOL-VIDEO] Video written to {self._writer.output_path}')
                    else:
                        artifact_name = self._output_dir.name
                    self._record_step_row(
                        capture_result_file_name=artifact_name,
                        frame_count=result.frames_written,
                        duration_sec=result.measured_duration_s,
                        timestamp=self._start_dt,
                    )
                if dropped > 0 and not result.aborted:
                    # The center's protocol mute suppresses this popup during
                    # an unattended run; the manifest and end-of-run report
                    # carry the counts either way.
                    notifications.warning(
                        'Protocol',
                        'Video Frames Dropped',
                        f'{dropped} of {result.frames_selected} frame(s) in a video step '
                        'could not be written, so that video is shorter than its '
                        'recording. Check the log for the cause.',
                    )
                logger.info(
                    f'[PROTOCOL-VIDEO] Finished: {result.frames_written} written, '
                    f'{result.write_failures} failed, measured '
                    f'{result.measured_fps:.2f} fps over {result.measured_duration_s:.2f} s'
                )

    # ------------------------------------------------------------------
    # UI titles (dispatched to the UI scheduler; callbacks may be absent)
    # ------------------------------------------------------------------

    def _set_title(self, key: str, **kwargs) -> None:
        cb = self._callbacks.get(key)
        if cb is not None:
            _schedule_ui(lambda dt, cb=cb, kw=dict(kwargs): cb(**kw), 0)

    def _reset_title(self) -> None:
        cb = self._callbacks.get('reset_title')
        if cb is not None:
            _schedule_ui(lambda dt, cb=cb: cb(), 0)
