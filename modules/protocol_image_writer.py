# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Image/video capture and file-write orchestration for protocol execution.

Runs on the protocol-executor thread (_capture) and file-IO thread
(_write_capture).  Extracted from ``sequenced_capture_runner.py``
during the protocol-decomposition refactor.
"""

from __future__ import annotations

import datetime
import pathlib
import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from lvp_logger import protocol_logger as logger

import modules.common_utils as common_utils
from modules.image_save import save_image
from modules.protocol import Protocol
from modules.video_capture import VideoCaptureSession, write_video
from modules.sequential_io_executor import IOTask, PROTOCOL_QUEUE_FULL

try:
    from modules import profile_trace
except ImportError:
    profile_trace = None

if TYPE_CHECKING:
    from modules.lumascope_api import Lumascope
    from modules.protocol_callbacks import ProtocolCallbacks
    from modules.protocol_execution_record import ProtocolExecutionRecord
    from modules.sequential_io_executor import SequentialIOExecutor


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
        execution_record: ProtocolExecutionRecord,
        # Functions borrowed from the parent executor
        leds_off_fn,
        is_run_in_progress_fn,
        stim_profiling: bool = False,
        run_dir: pathlib.Path | None = None,
        # Resolved once at run start; per-save writes reuse it instead of
        # re-reading settings under lock. Required so a run cannot be built
        # without its image mode and silently default to 8-bit (saving a
        # 12-bit-scaled run right-aligned / dark).
        save_encoding: str,
    ):
        self._scope = scope
        self._callbacks = callbacks
        self._aborted = aborted
        self._file_io_executor = file_io_executor
        self._abort_fn = abort_fn
        self._execution_record = execution_record
        self._leds_off = leds_off_fn
        self._is_run_in_progress = is_run_in_progress_fn
        self._stim_profiling = stim_profiling
        self._run_dir = run_dir
        self._save_encoding = save_encoding
        self._consecutive_capture_failures = 0
        self._MAX_CONSECUTIVE_CAPTURE_FAILURES = 3

    def _record_dropped_capture(
        self,
        *,
        step,
        step_index,
        scan_count,
        capture_time,
        name,
        reason='capture_failed_queue_full',
    ):
        """F-2: log a dropped protocol capture in the execution record.

        Called when ``file_io_executor.protocol_put`` returns
        ``PROTOCOL_QUEUE_FULL`` because the bounded file-IO queue
        rejected the write. Without this the dropped capture would be
        silently absent from the run record.
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

    def _capture_evidence(self, image) -> str:
        """One-line provenance for a captured frame: brightness statistics
        plus the chunk-verified exposure / gain and capture-hold timing.

        Saved-frame defects (a frame exposed under the previous channel's
        settings saturates or mis-exposes) previously left no log trace at
        all; this line makes every protocol capture auditable from a
        support bundle. Brightness is computed on a strided sample so the
        cost stays negligible at full frame rate.
        """
        try:
            parts = []
            if image is not None and getattr(image, 'size', 0) > 0:
                sample = image[::8, ::8]
                max_value = np.iinfo(image.dtype).max
                sat_fraction = float(np.count_nonzero(sample >= 0.99 * max_value)) / sample.size
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
        image_capture_config: dict | None = None,
        autogain_settings: dict | None = None,
        video_as_frames: bool = False,
        separate_folder_per_channel: bool = False,
        curr_step: int = 0,
    ) -> bool:
        """Orchestrate image/video acquisition for a single protocol step.

        Runs on the protocol-executor thread.

        Returns:
            True if the capture completed normally and left the step channel
            lit, so the caller should drive the step-boundary LED decision
            through the authority. False if the capture returned early
            (aborted, failed, cancelled, or a dropped write); the caller must
            not apply a boundary hold in that case -- the failure paths have
            already turned the LED off, and a dropped write leaves the frame's
            LED for the next step's illuminate to resolve, as before.
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
        _trace_enabled = profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE
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
                _curr_exp = self._scope.imaging.get_exposure_time()
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
                self._scope.imaging.set_gain(step['Gain'])
                self._scope.imaging.set_exposure_time(step['Exposure'])
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
            if self._scope.motion.has_turret():
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
            if engineering_mode and self._scope.motion.has_turret():
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
                    post=('video',) if is_video else (),
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

            # capture_depth and save_encoding are a coupled pair from the
            # image_mode config; read required (not .get-with-default) so a
            # config that somehow lost one fails loudly here instead of saving a
            # 12-bit-scaled frame right-aligned (dark). Read before the
            # save/not-save split so every write_capture dispatch -- including
            # the not-saving record row -- carries the run's depth in scope.
            capture_depth = image_capture_config['capture_depth']

            if enable_image_saving:
                jpeg_quality = image_capture_config.get('jpg_quality', 90)

                if is_video:
                    session = VideoCaptureSession(
                        scope=self._scope,
                        step=step,
                        autogain_settings=autogain_settings,
                        is_protocol_running_fn=self._is_run_in_progress,
                        callbacks=self._callbacks.to_dict(),
                        leds_off_fn=self._leds_off,
                        stim_profiling=self._stim_profiling,
                        run_dir=self._run_dir,
                    )
                    video_result = session.capture()

                    if video_result is None:
                        # Cancelled or zero frames -- no file to write, but
                        # still leave a row so the run record isn't silently
                        # missing the step (image captures record one too).
                        self._leds_off()
                        self._record_dropped_capture(
                            step=step,
                            step_index=curr_step,
                            scan_count=scan_count,
                            capture_time=datetime.datetime.now(),
                            name=name,
                            reason='video_cancelled',
                        )
                        _proto_outcome = 'video_cancelled'
                        return False

                    self._leds_off()

                    _capture_time = datetime.datetime.now()
                    _put_result = self._file_io_executor.protocol_put(
                        IOTask(
                            action=self.write_capture,
                            kwargs={
                                'is_video': is_video,
                                'video_as_frames': video_as_frames,
                                'video_result': video_result,
                                'save_folder': save_folder,
                                'use_color': use_color,
                                'name': name,
                                'output_format': output_format,
                                'save_encoding': image_capture_config['save_encoding'],
                                'capture_depth': capture_depth,
                                'step': step,
                                'captured_image': None,
                                'step_index': curr_step,
                                'scan_count': scan_count,
                                'capture_time': _capture_time,
                                'enable_image_saving': enable_image_saving,
                                'separate_folder_per_channel': separate_folder_per_channel,
                            },
                            silent_on_failure=True,
                        )
                    )
                    if _put_result is PROTOCOL_QUEUE_FULL:
                        self._record_dropped_capture(
                            step=step,
                            step_index=curr_step,
                            scan_count=scan_count,
                            capture_time=_capture_time,
                            name=name,
                        )
                        _proto_outcome = 'video_dropped_queue_full'
                        return False
                    _proto_outcome = 'video_success'
                    return False  # Video always extinguishes; leds_off called above

                else:
                    # Frame validity drains stale frames, then grabs a valid one
                    captured_image = self._scope.imaging.capture_and_wait(
                        force_to_8bit=capture_depth == 8,
                        all_ones_check=True,
                        timeout_s=1.0,
                        sum_count=sum_count,
                        sum_delay_s=step['Exposure'] / 1000,
                        sum_iteration_callback=sum_iteration_callback,
                    )

                    if captured_image is None:
                        self._consecutive_capture_failures += 1
                        logger.error(
                            f'[PROTOCOL] Capture failed for step {curr_step} ({step.get("Name", "?")}), scan {scan_count} -- camera inactive or frame drain failed (failure {self._consecutive_capture_failures}/{self._MAX_CONSECUTIVE_CAPTURE_FAILURES})'
                        )
                        # Still record the step with "capture_failed" so the record isn't silently missing.
                        # If the file-IO queue is also full, fall back to recording directly (synchronously)
                        # so the failure isn't doubly hidden.
                        _failed_capture_time = datetime.datetime.now()
                        _put_result = self._file_io_executor.protocol_put(
                            IOTask(
                                action=self.write_capture,
                                kwargs={
                                    'step': step,
                                    'step_index': curr_step,
                                    'scan_count': scan_count,
                                    'capture_time': _failed_capture_time,
                                    'enable_image_saving': enable_image_saving,
                                    'separate_folder_per_channel': separate_folder_per_channel,
                                    'save_encoding': image_capture_config['save_encoding'],
                                    'capture_depth': capture_depth,
                                },
                                silent_on_failure=True,
                            )
                        )
                        if _put_result is PROTOCOL_QUEUE_FULL:
                            self._record_dropped_capture(
                                step=step,
                                step_index=curr_step,
                                scan_count=scan_count,
                                capture_time=_failed_capture_time,
                                name=name,
                            )
                        self._leds_off()
                        if (
                            self._consecutive_capture_failures
                            >= self._MAX_CONSECUTIVE_CAPTURE_FAILURES
                        ):
                            from modules.notification_center import notifications

                            notifications.critical(
                                'Protocol',
                                'Camera Failure',
                                f'Camera failed {self._consecutive_capture_failures} consecutive captures. Aborting protocol.',
                            )
                            self._abort_fn()
                        _proto_outcome = 'capture_failed'
                        return False

                    self._consecutive_capture_failures = 0  # Reset on success
                    logger.info(
                        f'Protocol Image Captured: {name} {self._capture_evidence(captured_image)}'
                    )

                    # Hold the captured image on screen for at least 500 ms so
                    # the user can see the saved frame before the live preview
                    # overwrites it. NOT a delay -- the next protocol save bumps
                    # the hold deadline forward, so display tracks the
                    # most-recent saved frame in real time. Best-effort; missing
                    # scope_display (early init / standalone tools) is fine.
                    # Depth travels with the frame so the hold-display downconvert
                    # scales against the real range (summed -> 16-bit).
                    if captured_image.dtype == np.uint8:
                        hold_significant_bits = 8
                    elif sum_count > 1:
                        hold_significant_bits = 16
                    else:
                        hold_significant_bits = self._scope.imaging.significant_bits
                    try:
                        import modules.app_context as _app_ctx

                        ctx = _app_ctx.ctx
                        if ctx is not None and getattr(ctx, 'scope_display', None) is not None:
                            ctx.scope_display.hold_protocol_saved_image(
                                captured_image, hold_significant_bits
                            )
                    except Exception as _e:
                        logger.debug(f'[PROTOCOL] hold_protocol_saved_image failed: {_e}')

                    _success_capture_time = datetime.datetime.now()
                    _put_result = self._file_io_executor.protocol_put(
                        IOTask(
                            action=self.write_capture,
                            kwargs={
                                'save_folder': save_folder,
                                'use_color': use_color,
                                'name': name,
                                'output_format': output_format,
                                'jpeg_quality': jpeg_quality,
                                'step': step,
                                'captured_image': captured_image,
                                'step_index': curr_step,
                                'scan_count': scan_count,
                                'capture_time': _success_capture_time,
                                'enable_image_saving': enable_image_saving,
                                'separate_folder_per_channel': separate_folder_per_channel,
                                'save_encoding': image_capture_config['save_encoding'],
                                'capture_depth': capture_depth,
                            },
                            silent_on_failure=True,
                        )
                    )
                    if _put_result is PROTOCOL_QUEUE_FULL:
                        self._record_dropped_capture(
                            step=step,
                            step_index=curr_step,
                            scan_count=scan_count,
                            capture_time=_success_capture_time,
                            name=name,
                        )
                        _proto_outcome = 'dropped_queue_full'
                        return False
                    _proto_outcome = 'success'

            else:
                _not_saving_capture_time = datetime.datetime.now()
                _put_result = self._file_io_executor.protocol_put(
                    IOTask(
                        action=self.write_capture,
                        kwargs={
                            'step': step,
                            'enable_image_saving': enable_image_saving,
                            'separate_folder_per_channel': separate_folder_per_channel,
                            'save_encoding': image_capture_config['save_encoding'],
                            'capture_depth': capture_depth,
                        },
                        silent_on_failure=True,
                    )
                )
                if _put_result is PROTOCOL_QUEUE_FULL:
                    self._record_dropped_capture(
                        step=step,
                        step_index=curr_step,
                        scan_count=scan_count,
                        capture_time=_not_saving_capture_time,
                        name=name,
                    )
                    _proto_outcome = 'not_saving_dropped_queue_full'
                else:
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
                )

    def write_capture(
        self,
        is_video=False,
        video_as_frames=False,
        video_result=None,
        save_folder=None,
        use_color=None,
        name=None,
        output_format=None,
        jpeg_quality=90,
        # save_encoding / capture_depth feed only the video leg (write_video);
        # the image leg saves via self._save_encoding. Required (no silent
        # 8-bit fallback) and keyword-only so every caller states them.
        *,
        save_encoding,
        capture_depth,
        step=None,
        captured_image=None,
        step_index=None,
        scan_count=None,
        capture_time=None,
        enable_image_saving=True,
        separate_folder_per_channel=False,
    ):
        """Write captured image/video to disk and record in execution log.

        Runs on the file-IO thread.
        """
        # Count the attempt up front so end-of-run reconciliation can detect a
        # capture that returns without leaving a row in the execution record.
        if self._execution_record is not None:
            self._execution_record.note_capture_attempt()

        captured_frames = 0
        duration_sec = 0.0

        # M8: Check disk space before writing -- long protocols can fill disk.
        if save_folder is not None:
            try:
                ok, free_mb = common_utils.check_disk_space_ok(save_folder, 500)
                if not ok:
                    from modules.notification_center import notifications

                    notifications.critical(
                        'FileIO',
                        'Disk Space Critical',
                        f'Only {free_mb:.0f} MB free. Aborting protocol to prevent data loss.',
                    )
                    self._abort_fn()
                    return
            except Exception:
                pass  # If we can't check, proceed anyway

        if enable_image_saving:
            if is_video:
                # A write failure must still leave a row in the execution
                # record -- the record is what post-processing and run
                # accounting key off. The queue-full and capture-failed
                # legs already record their failures; image-on-disk
                # missing AND row missing was the last silent-gap leg.
                try:
                    capture_result = write_video(
                        result=video_result,
                        save_folder=save_folder,
                        name=name,
                        video_as_frames=video_as_frames,
                        step=step,
                        callbacks=self._callbacks.to_dict(),
                        save_encoding=save_encoding,
                        capture_depth=capture_depth,
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

                captured_frames = video_result.captured_frames
                duration_sec = video_result.duration_sec

            else:
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

                # A summed 2D uint16 frame fills the 16-bit container, so its
                # stored depth is 16; a single uint16 frame keeps the camera's
                # native depth. uint8 frames never sum to a wider container.
                is_uint16_2d = (
                    hasattr(captured_image, 'dtype')
                    and captured_image.dtype == np.uint16
                    and getattr(captured_image, 'ndim', 0) == 2
                )
                # A summed full-depth frame lives in a 16-bit container; declare
                # that depth so SignificantBits matches the stored values. The
                # step's Sum column carries the count on this save thread.
                # Single uint16 frames fall through to the camera-native default.
                # step may be a pandas Series; test `is None` rather than
                # truthiness (bool() on a Series raises).
                step_sum = step.get('Sum', 1) if step is not None else 1
                summed_significant_bits = 16 if (is_uint16_2d and step_sum > 1) else None
                # Same failure-row contract as the video leg above: a
                # raise from save_image must not leave the record without
                # a row for this step.
                try:
                    capture_result = save_image(
                        self._scope,
                        array=captured_image,
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
                        jpeg_quality=jpeg_quality,
                        true_color=step['Color'],
                        x=step['X'],
                        y=step['Y'],
                        z=step['Z'],
                        save_encoding=self._save_encoding,
                        significant_bits=summed_significant_bits,
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
                    frame_count=captured_frames if is_video else 1,
                    duration_sec=duration_sec if is_video else 0.0,
                )
                logger.info('Protocol-Writer] Added step to protocol execution record')
            except Exception as ex:
                logger.error(
                    f'[Protocol-Writer] Failed to add step to protocol execution record: {ex}'
                )
