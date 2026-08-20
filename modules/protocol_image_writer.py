# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Image/video capture and file-write orchestration for protocol execution.

Runs on the protocol-executor thread (_capture) and file-IO thread
(_write_capture).  Extracted from ``sequenced_capture_runner.py``
during the protocol-decomposition refactor.
"""

from __future__ import annotations

import datetime
import functools
import pathlib
import logging
import threading
import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from lvp_logger import protocol_logger as logger

import modules.common_utils as common_utils
from modules.image_save import save_image
from modules.protocol import Protocol
import modules.protocol_recording as protocol_recording
from modules.protocol_recording import ProtocolVideoStep
from modules.sequential_io_executor import IOTask, PROTOCOL_QUEUE_WEDGED

from lib import profile_trace

if TYPE_CHECKING:
    from modules.image_mode import ImageCaptureConfig
    from modules.lumascope_api import Lumascope
    from modules.protocol_callbacks import ProtocolCallbacks
    from modules.protocol_execution_record import ProtocolExecutionRecord
    from modules.sequential_io_executor import SequentialIOExecutor


# Full-write-queue wait with zero tasks retired before the writer is declared
# wedged and the run aborts loudly. 12x the per-frame critical write budget
# (capture_save_disk_ms, PERFORMANCE_BUDGETS.md) -- unambiguously stuck, yet
# short enough that an attended user gets a named error instead of a
# frozen-looking run. A running task that declared a longer
# slow_task_threshold_sec raises the bar for itself. Bench-tunable.
WRITE_STALL_FATAL_S = 30.0


class CapturedFrame(NamedTuple):
    """A captured frame coupled with the payload depth it was captured at.

    The save runs asynchronously on the file-IO thread; a bare array would
    force the writer to re-derive depth at save time, when the camera may
    be at a different pixel format or unreadable. Coupling the depth to
    the frame at capture makes handing over a frame without its depth
    unrepresentable.
    """

    image: np.ndarray
    significant_bits: int


class ProtocolImageWriter:
    """Handles image/video capture and file writing during protocol runs.

    Created by SequencedCaptureRunner at the start of each run with
    the references it needs.  All state is borrowed from the executor --
    this class owns no mutable state of its own.
    """

    LOGGER_NAME = 'SequencedCaptureRunner'

    def __init__(
        self,
        *,
        scope: Lumascope,
        callbacks: ProtocolCallbacks,
        aborted: threading.Event,
        file_io_executor: SequentialIOExecutor,
        abort_fn,  # callable -- bound to protocol_thread.abort
        # THIS run's fatal-abort flag, allocated fresh per run by the runner.
        # Per-run, not runner-lifetime: queued write tasks keep draining after
        # the run ends and can fire a fatal abort (disk-full) from the OLD
        # run's writer after the NEXT run has started -- a shared flag set
        # then would fatal-brand and force-darken the successor run; a
        # per-run object lets the late set land on a dead flag.
        fatal_abort_event: threading.Event,
        execution_record: ProtocolExecutionRecord,
        # Functions borrowed from the parent executor
        leds_off_fn,
        is_run_in_progress_fn,
        # The run's one immutable capture/save intent. Required so a run
        # cannot be built without its image mode; holding the whole frozen
        # config (rather than a loose save_encoding) leaves no second
        # channel for the capture depth and the save encoding to diverge.
        image_capture_config: ImageCaptureConfig,
        # Whether video steps burn the capture timestamp into each encoded
        # frame. The user's choice, snapshotted at run start like the rest
        # of the run config; required so no writer can silently decide it.
        timestamp_overlay: bool,
        # The run's snapshot of the global "Video max FPS" cap (0 =
        # uncapped). Required so video-step sizing and the recording rate
        # can never read live settings mid-run.
        video_max_fps: float,
    ):
        self._scope = scope
        self._callbacks = callbacks
        self._aborted = aborted
        self._file_io_executor = file_io_executor
        self._abort_fn = abort_fn
        self._fatal_abort_event = fatal_abort_event
        self._execution_record = execution_record
        self._leds_off = leds_off_fn
        self._is_run_in_progress = is_run_in_progress_fn
        self._config = image_capture_config
        self._timestamp_overlay = timestamp_overlay
        self._video_max_fps = video_max_fps
        self._video_steps: list[ProtocolVideoStep] = []
        self._consecutive_capture_failures = 0
        self._MAX_CONSECUTIVE_CAPTURE_FAILURES = 3

    def _abort_run_fatal(self, domain: str, title: str, message: str) -> None:
        """The one fatal-abort path: every run-killing fault routes here.

        Ordering is load-bearing:
        1. abort -- a non-blocking, idempotent Event.set that immediately
           closes the protocol thread's aborted-gates against lighting the
           next step; it involves no I/O, so nothing may precede it.
        2. fatal flag -- read by cleanup (terminal-dark assertion) and by
           the step-boundary gate; set before the LEDs go dark so a step
           racing this call cannot observe dark-but-not-fatal.
        3. force_off -- darkens the sample NOW, on this thread, via the
           direct driver path (no executor hop), because the fault that
           brought us here may be wedging the teardown that normally turns
           the LEDs off; a live sample must not stay illuminated while a
           dead disk times out. Worst case ~5 s behind an in-flight
           confirmed LED write on the driver lock.
        4. the fatal popup -- last, after the hardware is safe.
        Safe to re-enter: every step is idempotent, so a second fault
        surfacing while this runs (e.g. the failure-record write itself
        wedging) changes nothing.
        """
        self._abort_fn()
        self._fatal_abort_event.set()
        self._scope.illumination.force_off()
        from modules.notification_center import notifications

        notifications.critical(domain, title, message)

    def _abort_run_on_writer_death(self) -> None:
        """Arm the run abort after the engine surfaced writer-lane death.

        The engine's critical notification already reached the user
        through the protocol mute; this is _abort_run_fatal's ordering
        minus a second popup: abort, fatal flag, force-dark.
        """
        self._abort_fn()
        self._fatal_abort_event.set()
        self._scope.illumination.force_off()

    @property
    def video_busy(self) -> bool:
        """True while any video step's drain or post-drain finish runs."""
        return any(step.is_busy for step in self._video_steps)

    @property
    def video_pending_writes(self) -> int:
        """Frames across all video steps enqueued but not yet on disk."""
        return sum(step.pending_writes for step in self._video_steps)

    def discard_video_pending(self) -> None:
        """Drop every video step's unwritten backlog loudly (app-close
        discard); frames already on disk stay."""
        for step in self._video_steps:
            step.discard_pending()

    def wait_for_video_drains(self, timeout_s: float = 600.0) -> bool:
        """Block until every video step's drain and finish complete.

        Called by the runner before the execution record reconciles, so a
        drain tail's row cannot land after the record closes. True when
        everything finished inside the timeout.
        """
        deadline = time.monotonic() + timeout_s
        finished = True
        for step in self._video_steps:
            remaining = deadline - time.monotonic()
            if not step.wait_until_finished(max(0.0, remaining)):
                logger.error(
                    '[Protocol-Writer] A video step was still draining after '
                    f'{timeout_s:.0f} s; its execution-record row may be missing'
                )
                finished = False
        return finished

    def _record_video_step_row(
        self,
        *,
        step,
        step_index,
        scan_count,
        name,
        capture_result_file_name,
        frame_count,
        duration_sec,
        timestamp,
    ) -> None:
        """Record one finished video step: the attempt and its row, paired.

        The stills lane pairs note_capture_attempt with add_step inside
        write_capture; the video lane pairs them here so the end-of-run
        reconcile stays balanced.
        """
        if self._execution_record is None:
            return
        self._execution_record.note_capture_attempt()
        try:
            self._execution_record.add_step(
                capture_result_file_name=capture_result_file_name,
                step_name=name if name else 'unknown',
                step_index=step_index,
                scan_count=scan_count,
                timestamp=timestamp,
                frame_count=frame_count,
                duration_sec=duration_sec,
            )
        except Exception as ex:
            logger.error(f'[Protocol-Writer] Failed to record video step row: {ex}')

    def _record_dropped_capture(
        self,
        *,
        step,
        step_index,
        scan_count,
        capture_time,
        name,
        reason,
    ):
        """Log a capture that produced no file in the execution record.

        Called when a step ends without a write: a cancelled / zero-frame
        video, a failed save, or a wedged file writer. Without this the
        missing capture would be silently absent from the run record.
        """
        if self._execution_record is None:
            return
        try:
            self._execution_record.add_step(
                capture_result_file_name=reason,
                step_name=name if name else 'unknown',
                step_index=step_index,
                scan_count=scan_count,
                timestamp=capture_time,
                frame_count=0,
                duration_sec=0.0,
            )
        except Exception as ex:
            logger.error(f'[Protocol-Writer] Failed to record dropped capture: {ex}')

    def _note_capture_failure(
        self,
        *,
        step,
        curr_step,
        scan_count,
        name,
        enable_image_saving,
        separate_folder_per_channel,
        cause: str,
    ) -> None:
        """One home for a step-level capture failure: the strike counter,
        the 3-strike fatal abort, the capture_failed row, and LED cleanup.

        Shared by the stills leg (no frame drained) and the video leg (no
        frames delivered for the whole step) so the two capture paths
        cannot drift on failure accounting. A camera-lost video step with
        kept frames takes _note_capture_strike instead: its finish thread
        owns the step's row (measured, truncated truth), so the
        capture_failed row here would contradict it.
        """
        self._note_capture_strike(
            step=step, curr_step=curr_step, scan_count=scan_count, cause=cause
        )
        # Still record the step with "capture_failed" so the
        # record isn't silently missing this step.
        self._submit_write(
            kwargs={
                'enable_image_saving': enable_image_saving,
                'separate_folder_per_channel': separate_folder_per_channel,
            },
            step=step,
            step_index=curr_step,
            scan_count=scan_count,
            capture_time=datetime.datetime.now(),
            name=name,
        )
        self._leds_off()

    def _note_capture_strike(self, *, step, curr_step, scan_count, cause: str) -> None:
        """The strike counter + the 3-strike fatal abort, row-free."""
        self._consecutive_capture_failures += 1
        logger.error(
            f'[PROTOCOL] Capture failed for step {curr_step} ({step.get("Name", "?")}), '
            f'scan {scan_count} -- {cause} (failure '
            f'{self._consecutive_capture_failures}/{self._MAX_CONSECUTIVE_CAPTURE_FAILURES})'
        )
        aborting = self._consecutive_capture_failures >= self._MAX_CONSECUTIVE_CAPTURE_FAILURES
        # The fatal funnel runs BEFORE the cleanup side effects
        # (recording the failed step, leds_off): abort + dark +
        # popup must not wait on a record write that can block
        # against a failing disk. In the queue-full case this
        # means the capture_failed row below is cancelled where
        # it previously waited for a slot -- accepted.
        if aborting:
            step_color = step.get('Color', '')
            # led_connected term: color2ch also returns None
            # when no LED board is present at all -- a
            # board-less run's failures are not a missing
            # channel and must keep the camera wording.
            undrivable = (
                step_color in common_utils.get_layers_with_led()
                and self._scope.led_connected
                and self._scope.illumination.color2ch(step_color) is None
            )
            if undrivable:
                # The failures were guaranteed by the scope's
                # channel set, not by the camera -- blaming the
                # camera here misnames the cause the user can
                # actually act on.
                self._abort_run_fatal(
                    'Protocol',
                    'Channel not available',
                    f"This microscope has no '{step_color}' LED "
                    f'channel, so its steps cannot capture here. '
                    f'The protocol was stopped after '
                    f'{self._consecutive_capture_failures} failed '
                    f"captures. Remove the '{step_color}' steps to "
                    'run this protocol on this microscope.',
                )
            else:
                self._abort_run_fatal(
                    'Protocol',
                    'Camera Failure',
                    f'Camera failed {self._consecutive_capture_failures} consecutive captures. Aborting protocol.',
                )

    def _submit_write(
        self,
        *,
        kwargs: dict,
        step,
        step_index,
        scan_count,
        capture_time,
        name,
        slow_task_threshold_sec: float | None = None,
    ) -> bool:
        """One owner for enqueueing a write_capture task onto the bounded
        file queue.

        Blocks (back-pressure) instead of dropping when the queue is full:
        the run paces to disk drain, so a grabbed frame is never silently
        lost. Abort stays responsive via the writer's aborted event, polled
        between slot attempts.

        The step-identity kwargs (step, indices, timestamp, name) are
        normalized onto every write task so the execution-record row and any
        stall report can always name the step and its file -- some legs
        historically omitted them and their failures logged as 'unknown'.

        Returns True when the task was handed to the executor (or the
        executor declined it because no protocol is in session -- the
        run-teardown race the non-blocking path also tolerated). False only
        when the run is over: the wait was cancelled by an abort, or the
        writer was declared wedged -- in which case this method has already
        fired the fatal user notification, recorded the lost capture, and
        aborted the run.
        """
        kwargs.setdefault('step', step)
        kwargs.setdefault('step_index', step_index)
        kwargs.setdefault('scan_count', scan_count)
        kwargs.setdefault('capture_time', capture_time)
        kwargs.setdefault('name', name)
        result = self._file_io_executor.protocol_put_wait(
            IOTask(
                action=self.write_capture,
                kwargs=kwargs,
                silent_on_failure=True,
                slow_task_threshold_sec=slow_task_threshold_sec,
            ),
            should_abort=self._aborted.is_set,
            stall_timeout_s=WRITE_STALL_FATAL_S,
        )
        if result is PROTOCOL_QUEUE_WEDGED:
            stuck = self._file_io_executor.describe_running_task()
            self._abort_run_fatal(
                'Protocol',
                'File Writer Stalled',
                f'Saving stopped making progress ({stuck}), so the protocol '
                f'was stopped to avoid losing more captures. Check that the '
                f'save drive is connected and responsive, then run the '
                f'protocol again. A partial file from the stuck write may '
                f'remain on disk and stay locked until the writer releases '
                f'it.',
            )
            # The record shares the dead save target; latch it so this row
            # attempt (and any later one) is a loud no-op instead of a
            # synchronous write blocking THIS thread against the dead disk
            # until the OS gives up -- which is what used to delay the abort
            # (and the LED-off behind it) by the whole OS timeout. The
            # writer_stalled row is lost; its only trace is this run's
            # cleanup error log, accepted.
            if self._execution_record is not None:
                self._execution_record.mark_target_unresponsive()
            self._record_dropped_capture(
                step=step,
                step_index=step_index,
                scan_count=scan_count,
                capture_time=capture_time,
                name=name,
                reason='writer_stalled',
            )
            return False
        # None with the abort flag set is a cancelled wait; a bare None is
        # the executor declining outside a session (tolerated, as before).
        return not (result is None and self._aborted.is_set())

    def _capture_evidence(self, image, significant_bits: int) -> str:
        """One-line provenance for a captured frame: brightness statistics
        plus the chunk-verified exposure / gain and capture-hold timing.

        Saved-frame defects (a frame exposed under the previous channel's
        settings saturates or mis-exposes) previously left no log trace at
        all; this line makes every protocol capture auditable from a
        support bundle. Brightness is computed on a strided sample so the
        cost stays negligible at full frame rate. ``significant_bits`` is
        the frame's true bit depth, required because the container dtype
        can be wider than the data (12-bit frames ride in uint16); a
        container-derived full scale reads a saturated frame as sat=0%.
        """
        try:
            parts = []
            if image is not None and getattr(image, 'size', 0) > 0:
                sample = image[::8, ::8]
                full_scale = (1 << significant_bits) - 1
                sat_fraction = float(np.count_nonzero(sample >= 0.99 * full_scale)) / sample.size
                parts.append(f'mean={float(sample.mean()):.1f}')
                parts.append(f'sat={sat_fraction * 100.0:.1f}%')
            info = self._scope.imaging.last_capture_info or {}
            exp_us = info.get('chunk_exposure_us')
            gain_db = info.get('chunk_gain_db')
            parts.append(f'exp_ms={exp_us / 1000.0:.2f}' if exp_us is not None else 'exp_ms=na')
            parts.append(f'gain_db={gain_db:.2f}' if gain_db is not None else 'gain_db=na')
            if info.get('hold_ms') is not None:
                parts.append(f'hold_ms={info["hold_ms"]:.0f}')
            if info.get('drained') is not None:
                parts.append(f'drained={info["drained"]}')
            return ' '.join(parts)
        except Exception as ex:
            # Evidence is best-effort; never let it break the capture path.
            logger.debug(f'[Protocol-Writer] capture evidence unavailable: {ex}')
            return ''

    def capture(
        self,
        save_folder,
        step,
        output_format: str,
        protocol,
        *,
        scan_count=None,
        sum_count: int = 1,
        enable_image_saving: bool = True,
        autogain_settings: dict | None = None,
        video_as_frames: bool = False,
        separate_folder_per_channel: bool = False,
        curr_step: int = 0,
    ) -> bool:
        """Orchestrate image/video acquisition for a single protocol step.

        Runs on the protocol-executor thread. Capture depth and save
        encoding come from the writer's held run config (self._config) --
        the one carrier for the run's capture/save intent -- so the values
        the camera captures with and the values the file-IO thread saves
        with cannot diverge for one run.

        Returns:
            True if the capture completed normally and left the step channel
            lit, so the caller should drive the step-boundary LED decision
            through the authority. False if the capture returned early
            (aborted, failed, cancelled, or a fatal fault -- stalled writer,
            3-strike camera, disk floor -- which aborts the run); the caller
            must not apply a boundary hold in that case. Failure paths turn
            the LED off themselves; fatal faults route through
            _abort_run_fatal, which force-darkens immediately, and cleanup
            re-asserts dark as the run's terminal LED state.
        """
        if self._aborted.is_set():
            return False
        if not self._is_run_in_progress():
            return False

        # N5 (STALL-1 H5 disambiguator): proto-state trace.
        # See docs/STALL1_INSTRUMENTATION_EXPERIMENT.md (Firmware repo) sec.4 N5.
        # Wraps capture body in try/finally -- single row per capture invocation
        # captures duration + outcome + step identity, regardless of return path.
        # Disambiguates "real stall" vs "between-step pause" in the timeline.
        _trace_enabled = profile_trace.ENABLE_PROFILE_TRACE
        _proto_t0 = time.perf_counter() if _trace_enabled else None
        _proto_outcome = 'unknown'
        # step is dict-like (supports .get) but not always a dict subclass --
        # smoke 1 showed isinstance(step, dict) returned False even though
        # step.get('Name', '?') works fine (the existing CAPTURE DIAG line
        # at protocol_image_writer.py:114 uses the same pattern). Drop the
        # isinstance gate; rely on try/except.
        try:
            _proto_step_name = step.get('Name', '?')
            _proto_color = step.get('Color', '?')
        except Exception:
            _proto_step_name = '?'
            _proto_color = '?'
        if _trace_enabled:
            logger.info(
                f'[PROTO STATE] capture_start step={_proto_step_name} '
                f'color={_proto_color} curr_step={curr_step} '
                f'scan_count={scan_count}'
            )

        try:
            is_video = step['Acquire'] == 'video'

            # #610 diagnostic: trace camera settings decision at each capture.
            # camera_gain/camera_exp are read LIVE on purpose (the point is the
            # ACTUAL camera state vs the step's intent), so the reads are gated
            # on debug being enabled -- otherwise two SDK reads run every step
            # even though the line is dropped in normal operation.
            if logger.isEnabledFor(logging.DEBUG):
                _ag = step['Auto_Gain']
                _curr_gain = self._scope.imaging.get_gain()
                _curr_exp = self._scope.imaging.get_exposure_ms()
                logger.debug(
                    f'[CAPTURE DIAG] step={step.get("Name", "?")} color={step["Color"]} '
                    f'Auto_Gain={_ag!r} (type={type(_ag).__name__}) '
                    f'step_gain={step["Gain"]} step_exp={step["Exposure"]} '
                    f'camera_gain={_curr_gain} camera_exp={_curr_exp}'
                )

            if not step['Auto_Gain']:
                logger.debug(
                    f'[CAPTURE DIAG] Applying step camera settings: gain={step["Gain"]}, exp={step["Exposure"]}'
                )
                # STALL-1 fix: removed the `with self._scope.imaging.update_camera_config():`
                # wrapper that was here. update_camera_config() does StopGrabbing +
                # StartGrabbing, which Pylon SDK only requires for buffer-geometry
                # changes (Width/Height/PixelFormat/Binning/Offset) -- NOT for Gain
                # or ExposureTime, which are live-updateable. The wrapper was
                # paying a full grab-loop teardown+rebuild per protocol step,
                # producing the observed ~11s per-step duration during 12-bit
                # protocol runs (camera delivering ~1 fps instead of ~50 fps,
                # despite LVP-side processing being fast).
                #
                # AF code already sets gain/exposure without a wrapper
                # (modules/autofocus_runner.py:200,202). This change brings
                # protocol behavior in line with AF.
                #
                # If a "Node is locked while streaming" GenICam exception fires
                # here, that means Gain or ExposureTime is locked on the current
                # SDK/firmware combo -- revert this change and add a
                # `requires_buffer_realloc=True` audit. Per Basler convention
                # both should be live-changeable.
                # The non-dispatching bodies: this runs on the protocol
                # thread while the run has the camera executor disabled, so
                # the public dispatchers would refuse every per-step write.
                self._scope.imaging._set_gain_impl(step['Gain'])
                self._scope.imaging._set_exposure_ms_impl(step['Exposure'])
            else:
                # Auto_Gain step: scan_iterate already lit the LED and armed AG
                # against the lit scene; the apply is skipped here to avoid
                # restarting AG mid-grab. The capture_and_wait drain below waits
                # the auto_gain settle frames before grabbing.
                logger.debug(
                    f'[CAPTURE DIAG] Auto_Gain step: armed in scan_iterate '
                    f'with target gain={step["Gain"]}dB exp={step["Exposure"]}ms; '
                    f'settle drained in capture_and_wait'
                )

            # Objective short name for filename
            objective_short_name = None
            if self._scope.capabilities.has_turret:
                obj_info = self._scope.runtime_state.get_objective_info(
                    objective_id=step['Objective']
                )
                if obj_info is not None:
                    objective_short_name = obj_info.get('short_name')
                else:
                    logger.warning(
                        f'[PROTOCOL] Turret available but no objective info for ID '
                        f"'{step['Objective']}' -- using None for filename"
                    )

            # Build base name from protocol's custom root + step name
            try:
                capture_root = protocol.capture_root()
            except Exception:
                capture_root = ''

            # In engineering mode, include turret position in filename.
            # engineering_mode lives on the app context (ctx); fall back to
            # False when ctx is unset (bare-fixture test paths).
            import modules.app_context as _app_ctx_im

            turret_pos = None
            engineering_mode = getattr(_app_ctx_im.ctx, 'engineering_mode', False)
            if engineering_mode and self._scope.capabilities.has_turret:
                try:
                    turret_pos = int(self._scope.motion.get_current_position('T'))
                except Exception as e:
                    logger.debug(
                        '[%s] get_current_position(T) failed; turret '
                        'position omitted from filename: %s: %s',
                        self.LOGGER_NAME,
                        type(e).__name__,
                        e,
                    )

            # The objective is stamped onto the saved filename here (the one
            # writer), separate from the step's identity Name. capture_root is
            # a path prefix kept out of the name seed, so a root that happens
            # to contain a token cannot perturb the derived name.
            step_name = common_utils.build_step_name(
                common_utils.step_components(
                    step,
                    scan_count=scan_count,
                    objective=objective_short_name,
                    turret_position=turret_pos,
                    post=(common_utils.POST_TOKEN_VIDEO,) if is_video else (),
                )
            )
            if capture_root not in (None, ''):
                name = f'{capture_root}_{step_name}'
            else:
                name = step_name
            # Ensure the filename base has no invalid path characters
            try:
                name = Protocol.sanitize_step_name(input=name)
            except Exception:
                logger.exception(
                    '[%s] sanitize_step_name failed for name=%r; using '
                    'unsanitized name, file save may fail if name contains '
                    'invalid path characters',
                    self.LOGGER_NAME,
                    name,
                )

            # The step's channel is already lit and confirmed on by the runner's
            # STEP_LIGHT illuminate before this leaf is called; the leaf is a
            # pure grab+save and drives no LED on the success path (its failure /
            # video / cancel offs remain as error cleanup below).
            sum_iteration_callback = None
            use_color = step['Color'] if step['False_Color'] else 'BF'

            capture_depth = self._config.capture_depth

            if enable_image_saving:
                if is_video:
                    recorder = ProtocolVideoStep(
                        scope=self._scope,
                        step=step,
                        save_folder=pathlib.Path(save_folder),
                        name=name,
                        video_as_frames=video_as_frames,
                        capture_config=self._config,
                        timestamp_overlay=self._timestamp_overlay,
                        global_max_fps=self._video_max_fps,
                        autogain_settings=autogain_settings,
                        callbacks=self._callbacks.to_dict(),
                        aborted_event=self._aborted,
                        is_run_in_progress=self._is_run_in_progress,
                        abort_run_fatal=self._abort_run_fatal,
                        abort_run_on_writer_death=self._abort_run_on_writer_death,
                        record_step_row=functools.partial(
                            self._record_video_step_row,
                            step=step,
                            step_index=curr_step,
                            scan_count=scan_count,
                            name=name,
                        ),
                        record_dropped_capture=functools.partial(
                            self._record_dropped_capture,
                            step=step,
                            step_index=curr_step,
                            scan_count=scan_count,
                            name=name,
                        ),
                    )
                    self._video_steps.append(recorder)
                    outcome = recorder.run_blocking()
                    self._leds_off()

                    if outcome == protocol_recording.NO_FRAMES:
                        self._note_capture_failure(
                            step=step,
                            curr_step=curr_step,
                            scan_count=scan_count,
                            name=name,
                            enable_image_saving=enable_image_saving,
                            separate_folder_per_channel=separate_folder_per_channel,
                            cause='the video step received no camera frames',
                        )
                        _proto_outcome = 'video_no_frames'
                        return False
                    if outcome == protocol_recording.ABORTED:
                        _proto_outcome = 'video_disk_abort'
                        return False
                    if outcome == protocol_recording.CAMERA_LOST:
                        # Kept frames drain and the finish thread records
                        # the step's measured (truncated) row; the strike
                        # is the failure accounting. This branch exists to
                        # skip the counter reset below -- a camera loss
                        # must accumulate toward the 3-strike abort.
                        self._note_capture_strike(
                            step=step,
                            curr_step=curr_step,
                            scan_count=scan_count,
                            cause='camera went inactive mid-video-step; kept frames are on disk',
                        )
                        _proto_outcome = 'video_camera_lost'
                        return False

                    self._consecutive_capture_failures = 0
                    # The drain and the execution-record row finish on the
                    # step's own thread; the run moves on. Video always
                    # extinguishes -- leds_off called above.
                    _proto_outcome = f'video_{outcome}'
                    return False

                else:
                    # Frame validity drains stale frames, then grabs a valid one.
                    # dark_floor_check: a step that drives its LED must never
                    # save a black frame (stale pre-LED integration, or an
                    # external consumer starving the feed); a step with
                    # illumination 0, or a colour the scope drives no LED
                    # for (luminescence), is dark by design.
                    captured_image = self._scope.imaging._capture_and_wait_impl(
                        force_to_8bit=capture_depth == 8,
                        all_ones_check=True,
                        dark_floor_check=step['Illumination'] > 0
                        and step['Color'] in common_utils.get_layers_with_led(),
                        timeout_s=1.0,
                        sum_count=sum_count,
                        sum_delay_s=step['Exposure'] / 1000,
                        sum_iteration_callback=sum_iteration_callback,
                    )

                    if captured_image is None:
                        self._note_capture_failure(
                            step=step,
                            curr_step=curr_step,
                            scan_count=scan_count,
                            name=name,
                            enable_image_saving=enable_image_saving,
                            separate_folder_per_channel=separate_folder_per_channel,
                            cause='camera inactive or frame drain failed',
                        )
                        _proto_outcome = 'capture_failed'
                        return False

                    self._consecutive_capture_failures = 0  # Reset on success

                    # Depth travels with the frame so the evidence line's
                    # saturation threshold, the hold-display downconvert, AND
                    # the eventual file save all scale against the real range
                    # (summed -> 16-bit). Resolved here at capture time -- the
                    # async save must not re-derive it later, when the camera
                    # may be at a different format or unreadable.
                    frame_significant_bits = self._scope.imaging.capture_frame_depth(
                        captured_image, sum_count
                    )
                    logger.info(
                        f'Protocol Image Captured: {name} '
                        f'{self._capture_evidence(captured_image, frame_significant_bits)}'
                    )

                    # Hold the captured image on screen for at least 500 ms so
                    # the user can see the saved frame before the live preview
                    # overwrites it. NOT a delay -- the next protocol save bumps
                    # the hold deadline forward, so display tracks the
                    # most-recent saved frame in real time. Best-effort; missing
                    # scope_display (early init / standalone tools) is fine.
                    try:
                        import modules.app_context as _app_ctx

                        ctx = _app_ctx.ctx
                        if ctx is not None and getattr(ctx, 'scope_display', None) is not None:
                            ctx.scope_display.hold_protocol_saved_image(
                                captured_image, frame_significant_bits
                            )
                    except Exception as _e:
                        logger.debug(f'[PROTOCOL] hold_protocol_saved_image failed: {_e}')

                    _success_capture_time = datetime.datetime.now()
                    if not self._submit_write(
                        kwargs={
                            'save_folder': save_folder,
                            'use_color': use_color,
                            'output_format': output_format,
                            'captured_image': CapturedFrame(
                                image=captured_image,
                                significant_bits=frame_significant_bits,
                            ),
                            'enable_image_saving': enable_image_saving,
                            'separate_folder_per_channel': separate_folder_per_channel,
                        },
                        step=step,
                        step_index=curr_step,
                        scan_count=scan_count,
                        capture_time=_success_capture_time,
                        name=name,
                    ):
                        _proto_outcome = 'write_aborted'
                        return False
                    _proto_outcome = 'success'

            else:
                _not_saving_capture_time = datetime.datetime.now()
                if not self._submit_write(
                    kwargs={
                        'enable_image_saving': enable_image_saving,
                        'separate_folder_per_channel': separate_folder_per_channel,
                    },
                    step=step,
                    step_index=curr_step,
                    scan_count=scan_count,
                    capture_time=_not_saving_capture_time,
                    name=name,
                ):
                    _proto_outcome = 'not_saving_write_aborted'
                    return False
                _proto_outcome = 'not_saving'

            # Completed normally with the step channel still lit. The
            # step-boundary LED decision -- hold within a z-stack or across a
            # same-color move, or go dark -- belongs to the LED authority and
            # is driven by the caller, so the leaf no longer turns the LED off
            # here. The failure paths above keep their own offs: those are
            # error cleanup, not the boundary decision.
            return True
        except Exception:
            _proto_outcome = 'exception'
            raise
        finally:
            if _trace_enabled and _proto_t0 is not None:
                _proto_dt_ms = (time.perf_counter() - _proto_t0) * 1000.0
                logger.info(
                    f'[PROTO STATE] capture_end step={_proto_step_name} '
                    f'outcome={_proto_outcome} dt_ms={_proto_dt_ms:.1f}'
                )
                profile_trace.trace(
                    'proto_state_trace.csv',
                    'ts_ms,duration_ms,step_name,color,curr_step,scan_count,outcome',
                    [
                        int(time.time() * 1000),
                        f'{_proto_dt_ms:.3f}',
                        _proto_step_name,
                        _proto_color,
                        curr_step,
                        scan_count,
                        _proto_outcome,
                    ],
                    recording_id=profile_trace.NO_RECORDING,
                )

    def write_capture(
        self,
        save_folder=None,
        use_color=None,
        name=None,
        output_format=None,
        *,
        step=None,
        captured_image=None,
        step_index=None,
        scan_count=None,
        capture_time=None,
        enable_image_saving=True,
        separate_folder_per_channel=False,
    ):
        """Write a captured still image to disk and record it in the run log.

        Runs on the file-IO thread. Encoding, depth, and JPEG quality come
        from the writer's held run config -- the one carrier for the run's
        capture/save intent. Video steps never ride this lane: their frames
        write continuously on the recording engine's own writer thread.

        Args:
            captured_image: ``CapturedFrame`` (frame + the payload depth it
                was captured at), or None to record a capture_failed row.
                The depth travels with the frame because this save is
                asynchronous -- deriving depth here would read the camera's
                state at save time, when the format may have changed or the
                camera may be unreadable.
        """
        # Count the attempt up front so end-of-run reconciliation can detect a
        # capture that returns without leaving a row in the execution record.
        if self._execution_record is not None:
            self._execution_record.note_capture_attempt()

        # Check disk space before writing -- long protocols can fill disk.
        # Require headroom for THIS step's predicted write, never below the
        # per-write minimum.
        if save_folder is not None:
            try:
                required_mb = max(
                    common_utils.MIN_PER_WRITE_DISK_MB,
                    common_utils.estimate_step_write_mb(step, global_max_fps=self._video_max_fps),
                )
                ok, free_mb = common_utils.check_disk_space_ok(save_folder, required_mb)
                if not ok:
                    # Runs on the file-IO thread: the funnel's abort-first
                    # ordering matters here -- the protocol thread may be
                    # mid-capture, and abort must close its step-lighting
                    # gates before force_off darkens the sample.
                    self._abort_run_fatal(
                        'FileIO',
                        'Disk Space Critical',
                        f'Only {free_mb:.0f} MB free. Aborting protocol to prevent data loss.',
                    )
                    return
            except Exception as e:
                logger.warning(f'[Protocol-Writer] Disk space check failed (proceeding): {e}')

        if enable_image_saving:
            if captured_image is None:
                logger.warning(
                    f'[PROTOCOL] _write_capture: captured_image is None for step {step_index} ({step.get("Name", "?") if step is not None else "?"}), scan {scan_count}, recording as capture_failed'
                )
                if self._execution_record is not None:
                    self._execution_record.add_step(
                        capture_result_file_name='capture_failed',
                        step_name=name if name else 'unknown',
                        step_index=step_index,
                        scan_count=scan_count,
                        timestamp=capture_time,
                        frame_count=0,
                        duration_sec=0.0,
                    )
                return

            # The frame arrives coupled with the payload depth it was
            # captured at (uint8 -> 8, summed -> 16, else the per-frame
            # delivery stamp) -- recorded at capture time on the executor
            # thread, because by the time this save runs the camera may
            # be at a different format or unreadable.
            # A raise from save_image must not leave the record without
            # a row for this step.
            try:
                capture_result = save_image(
                    self._scope,
                    array=captured_image.image,
                    save_folder=save_folder,
                    file_root=None,
                    append=name,
                    color=use_color,
                    # Defense-in-depth against duplicate step Names that
                    # slip past load-time validation (#636). Plain
                    # filename when no file exists; numeric suffix only
                    # on actual collision.
                    tail_id_mode='if_collision',
                    output_format=output_format,
                    jpeg_quality=self._config.jpg_quality,
                    true_color=step['Color'],
                    x=step['X'],
                    y=step['Y'],
                    z=step['Z'],
                    save_encoding=self._config.save_encoding,
                    significant_bits=captured_image.significant_bits,
                )
            except Exception:
                self._record_dropped_capture(
                    step=step,
                    step_index=step_index,
                    scan_count=scan_count,
                    capture_time=capture_time,
                    name=name,
                    reason='save_failed',
                )
                raise

            if capture_result is None:
                capture_result_filepath_name = 'unsaved'
            elif isinstance(capture_result, dict):
                capture_result_filepath_name = capture_result['metadata']['file_loc']
            elif separate_folder_per_channel:
                capture_result_filepath_name = pathlib.Path(step['Color']) / capture_result.name
            else:
                capture_result_filepath_name = capture_result.name

        else:
            capture_result_filepath_name = 'unsaved'

        if self._execution_record is not None:
            try:
                self._execution_record.add_step(
                    capture_result_file_name=capture_result_filepath_name,
                    step_name=name,
                    step_index=step_index,
                    scan_count=scan_count,
                    timestamp=capture_time,
                    # This lane writes stills only: a video step's row lands
                    # from its own finish thread (_record_video_step_row)
                    # with the MEASURED frame count and duration.
                    frame_count=1,
                    duration_sec=0.0,
                )
                logger.debug('[Protocol-Writer] Added step to protocol execution record')
            except Exception as ex:
                logger.error(
                    f'[Protocol-Writer] Failed to add step to protocol execution record: {ex}'
                )
