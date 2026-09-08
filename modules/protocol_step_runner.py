# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Per-step execution logic for protocol runs.

Handles scan iteration, motion, LED control, autofocus orchestration,
and grease redistribution.  Extracted from ``sequenced_capture_runner.py``
during the protocol-decomposition refactor.

Thread ownership:
- ``scan_loop()`` and ``scan_iterate()`` run on the **protocol-executor** thread.
- ``_leds_off()`` / ``_led_on()`` queue work onto the **IO-executor** thread.
- ``_grease_redist_w_pos()`` runs on the **IO-executor** thread.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING

from lvp_logger import logger

import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.image_mode as image_mode
from modules.exceptions import AutofocusAborted
from modules.lumascope_api.illumination import (
    FIRE_AND_FORGET_TRANSITIONS,
    LedEndPolicy,
    LedTransition,
    LedTransitionCtx,
    resolve_end_state,
)
from modules.protocol_state_machine import ProtocolState
from modules.sequential_io_executor import IOTask, PROTOCOL_ENQUEUED

if TYPE_CHECKING:
    from modules.sequenced_capture_runner import SequencedCaptureRunner

from modules.kivy_utils import schedule_ui as _schedule_ui


class ProtocolStepRunner:
    """Executes individual protocol steps within a scan.

    Receives a reference to the parent ``SequencedCaptureRunner`` to
    access shared state (protocol, scope, events, executors).  This keeps
    the step-execution logic in its own file without duplicating state.
    """

    def __init__(self, parent: SequencedCaptureRunner):
        self._p = parent

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    def scan_loop(self):
        """Iterate through all protocol steps until scan completes.

        Blocks until the scan is done (all steps executed or aborted).
        Exceptions propagate to the outer run-loop so it can classify
        the failure as fatal (hardware disconnected -> abort + notify)
        or transient (everything else -> silent retry on next period).
        Catching exceptions here at the inner layer fires a
        notification at the wrong level and turns transient faults
        into protocol-halting popups.
        """
        last_maintenance_time = time.monotonic()

        # `_aborted` is a threading.Event read here and at every
        # `p._aborted.is_set()` site below WITHOUT holding `_state_lock`.
        # Intentional: Event.is_set() is atomic by contract, and the lock on
        # ProtocolThread.abort() exists only to make new-Future-with-cleared-
        # aborted publication atomic against a concurrent abort -- not for
        # readers. The worst-case race window for a reader to miss the abort
        # signal is one step-body tick.
        while self._p._scan_in_progress.is_set() and not self._p._aborted.is_set():
            # Periodic cleanup and watchdog logging for long runs
            now_mono = time.monotonic()
            if now_mono - last_maintenance_time > 60:
                last_maintenance_time = now_mono

                collected = gc.collect()
                if collected > 0:
                    logger.info(f'[Scan Watchdog] GC collected {collected} objects')

            # Run one step iteration. Exceptions propagate to the
            # outer run loop for fatal/transient classification.
            self.scan_iterate()

            # Small delay to prevent CPU throttling
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Single step iteration
    # ------------------------------------------------------------------

    def scan_iterate(self, dt: float | None = None) -> None:
        """Execute one iteration of the scan state machine."""
        p = self._p  # shorthand

        if p._aborted.is_set():
            return
        # Video encoding runs on FILE_WORKER in background -- do NOT block
        # the next step waiting for it. Frames are already captured and queued.
        if not p._scan_in_progress.is_set():
            return
        if not p._run_in_progress_event.is_set():
            return
        if p._af_future is not None and not p._af_future.done():
            return

        if p._af_future is not None and not p._af_result_consumed:
            # The autofocus run has resolved (we passed the not-done gate above).
            # Handle it exactly once. AFE restores the pre-AF Z on a non-success,
            # which leaves the stage in motion, so the polls that immediately
            # follow hit the is_moving early-return below WITHOUT clearing
            # _af_future (it is cleared only at the step transition). Without this
            # one-shot latch each of those ~1 kHz settle polls would re-enter and
            # re-handle the same resolved future.
            #
            # The autofocus thread is the sole owner of recording the fault with
            # a full traceback: it logs every non-abort exception before setting
            # it on the future, but at that altitude it has no idea which protocol
            # step or well the run belonged to. AFE has restored a usable Z, so an
            # AF fault mid-protocol is non-fatal -- the step still captures here at
            # that fallback Z and the run continues (no halt, no modal). That
            # fallback capture may be out of focus, so emit one step-correlated
            # warning that ties a possibly-blurry frame back to its step and well.
            # It is complementary to the AF thread's traceback (which it does not
            # repeat), and the one-shot latch keeps it to a single line per fault
            # rather than one per ~1 kHz settle poll. An intentional abort is not a
            # fault, so it stays at debug.
            p._af_result_consumed = True
            _af_exc = p._af_future.exception()
            if isinstance(_af_exc, AutofocusAborted):
                logger.debug(f'[PROTOCOL] Autofocus aborted at step {p._curr_step}')
            elif _af_exc is not None:
                _af_well = p._protocol.step(idx=p._curr_step).get('Well', '?')
                logger.warning(
                    f'[PROTOCOL] Autofocus failed at step {p._curr_step} '
                    f'(well {_af_well}) ({type(_af_exc).__name__}: {_af_exc}) -- '
                    f'capturing at fallback Z and continuing'
                )
            _cam_gain = p._scope.imaging.get_gain_db() if p._scope.imaging.active_cached else '?'
            _cam_exp = p._scope.imaging.get_exposure_ms() if p._scope.imaging.active_cached else '?'
            logger.info(
                f'[SCAN DIAG] AF gate passed: future.done()=True '
                f'camera_gain={_cam_gain} camera_exp={_cam_exp} step={p._curr_step}'
            )

        remaining_scans = p.remaining_scans()
        if remaining_scans <= 0:
            return

        # Check motion timeout. The timer bounds ONE continuous motion:
        # it starts when motion is first observed in flight and resets
        # whenever the stage reports idle, so time spent in an in-step
        # autofocus run never counts against a later move's budget.
        if p._scope.motion.is_moving():
            if p._motion_wait_start is None:
                p._motion_wait_start = time.monotonic()
            if time.monotonic() - p._motion_wait_start > p.MOTION_TIMEOUT_SECONDS:
                timeout_msg = (
                    f'Step {p._curr_step} timed out waiting for motion '
                    f'({p.MOTION_TIMEOUT_SECONDS}s).'
                )
                logger.error(f'[PROTOCOL] {timeout_msg} -- transitioning to ERROR state')
                from modules.notification_center import notifications

                # The timed-out move is still in flight. Halt the motor before
                # erroring out so it stops driving toward the unreachable target
                # rather than latching against a limit. Idempotent + no-ops on
                # field firmware without a STOP command.
                p._scope.motion.stop_motion()

                notifications.error(
                    'Protocol', 'Protocol Error -- Motion Timeout', timeout_msg, fatal=True
                )
                p._scan_in_progress.clear()
                try:
                    p._set_state(ProtocolState.ERROR)
                except ValueError:
                    pass
            return
        p._motion_wait_start = None

        if not p._grease_redistribution_event.is_set():
            return

        if p._aborted.is_set() or not p._scan_in_progress.is_set():
            return

        # Fetch the step row only after the early-return gates above.
        # scan_iterate polls at up to ~1 kHz while motion is in flight,
        # and protocol.step() builds a fresh pandas Series per call --
        # fetching before the is_moving gate burned that allocation on
        # every poll of every move.
        step = p._protocol.step(idx=p._curr_step)

        # AF already pushed the Z UI to best_focus_position; do not
        # overwrite with the pre-AF step['Z']. AFE.complete() being
        # True at this point means the most recent AF run finished
        # with a result that AFE has already scheduled to the UI.
        if step.get('Auto_Focus') and p._autofocus_runner.complete():
            pass
        elif p._z_ui_update_func is not None:
            _schedule_ui(lambda dt: p._z_ui_update_func(float(step['Z'])))

        # --- Pipeline timing instrumentation ---
        _t_settle = time.monotonic()
        _settle_wait_ms = (_t_settle - p._step_start_time) * 1000
        logger.debug(f'[TIMING] Step {p._curr_step} motion settle: {_settle_wait_ms:.1f}ms')

        # Camera settings (gain, exposure) and LED_ON are handled by
        # protocol_image_writer.capture() right before the actual frame grab.
        # Setting them here would duplicate the commands.
        if p._aborted.is_set() or not p._scan_in_progress.is_set():
            return

        # BF AF for fluorescence -- read from the runner snapshot so
        # mid-run UI toggles cannot produce inconsistent AF behavior
        # across steps within one scan. The value rides the run plan,
        # resolved by the caller that owns the settings.
        bf_af_for_fluor = getattr(p, '_bf_af_for_fluorescence', False)
        if (
            bf_af_for_fluor
            and step['Color'] != 'BF'
            and p._autofocus_runner.best_focus_position() is not None
        ):
            if p._update_z_pos_from_autofocus:
                new_z_pos = p._autofocus_runner.best_focus_position()
                p._protocol.modify_step_z_height(step_idx=p._curr_step, z=new_z_pos)
            logger.info(
                f'[Capture   ] Skipping AF on {step["Color"]} -- using BF result Z={p._autofocus_runner.best_focus_position()}'
            )
            step = dict(step)
            step['Auto_Focus'] = False

        if step['Auto_Focus'] and p._af_future is None:
            if p._callbacks.autofocus_in_progress:
                _schedule_ui(lambda dt: p._callbacks.autofocus_in_progress(), 0)

            af_executor_callbacks = {}
            if p._callbacks.move_position:
                af_executor_callbacks['move_position'] = p._callbacks.move_position

            if p._aborted.is_set() or not p._scan_in_progress.is_set():
                return

            p._af_future = p.autofocus_thread.run_autofocus(
                objective_id=step['Objective'],
                save_results_to_file=p._save_autofocus_data,
                results_dir=p._parent_dir,
                run_trigger_source=p._run_trigger_source,
                callbacks=af_executor_callbacks,
                led_color=step['Color'],
                led_illumination=step['Illumination'],
                camera_gain=step['Gain'],
                camera_exposure=step['Exposure'],
                # Capture immediately follows AF on the same step, so it
                # uses the same channel + illumination AF just lit (#612).
                # Tell AF to skip its off + state-restore cycle so the
                # capture inherits the LED state already established.
                keep_led_on=True,
                # AF runs inside this protocol step, which holds the LED
                # lease; hand it over so AF nests as a child rather than
                # contending for a fresh top-level lease.
                led_lease=p._led_lease,
            )
            return

        if step['Auto_Focus'] and p._af_future is not None and not p._af_future.done():
            return

        # Light the channel LED, then arm continuous Auto_Gain against the lit
        # scene. Hardware AG converges on whatever the camera is grabbing; if
        # the LED is dark when AG arms, it rails on noise and the grab is mis-
        # exposed. Lighting first lets AG settle toward the real exposure. The
        # capture_and_wait drain then waits the measured auto_gain settle frames
        # (invalidated inside set_auto_gain) before grabbing -- no separate
        # timed wait. The STEP_LIGHT illuminate makes this channel the only lit
        # one; the capture path re-asserts the same channel idempotently. Arm
        # once per step (_auto_gain_armed_step is a one-shot keyed on _curr_step).
        if step['Auto_Gain'] and p._auto_gain_armed_step != p._curr_step:
            self._apply_step_light(step)
            # Cap AG/AE exposure to this step's channel-class ceiling before
            # arming. Set on the shared settings dict so capture() inherits it.
            # step['Color'] is the layer; the per-install override map rides
            # the run plan, so a headless run honours it too.
            p._autogain_settings['max_exposure_ms'] = config_helpers.get_ag_ae_max_exposure_ms(
                step['Color'], p._ag_ae_max_exposure_ms
            )
            p._autogain_settings['min_exposure_ms'] = config_helpers.get_ag_ae_min_exposure_ms(
                step['Color']
            )
            fut = p._io_executor.protocol_put(
                IOTask(
                    # Run-internal machinery binds the impl per the
                    # dispatch contract: the public member is the
                    # external-caller surface (SDK/REST), and internal
                    # callers already inside the run's serialization
                    # use the body directly.
                    action=p._scope.imaging._apply_layer_camera_settings_impl,
                    kwargs={
                        'gain_db': step['Gain'],
                        'exposure_ms': step['Exposure'],
                        'auto_gain': True,
                        'auto_gain_settings': p._autogain_settings,
                        # A step's arm is unattended: the capture that locks
                        # it records the state and moves on -- no notice, no
                        # re-arm (the step-end disarm below is the only Off
                        # this path needs). Left at the live-view default,
                        # every protocol capture re-armed and popped a notice.
                        'resume_after_capture': False,
                    },
                ),
                return_future=True,
            )
            if fut:
                fut.result(timeout=30)
            p._auto_gain_armed_step = p._curr_step
            # Return after arming; the next tick falls through to capture, where
            # the auto_gain settle drain runs against the now-lit scene.
            return

        # Update Z position with autofocus results
        if step['Auto_Focus'] and p._update_z_pos_from_autofocus:
            new_z_pos = p._autofocus_runner.best_focus_position()
            if new_z_pos is not None:
                p._protocol.modify_step_z_height(step_idx=p._curr_step, z=new_z_pos)
            else:
                logger.warning('[Capture   ] Autofocus returned no position -- keeping current Z')

        if p._callbacks.autofocus_complete:
            _schedule_ui(lambda dt: p._callbacks.autofocus_complete(), 0)

        if step['Auto_Focus']:
            p._autofocus_count += 1

        # --- Capture ---
        if remaining_scans > 0:
            if not p._disable_saving_artifacts:
                if p._separate_folder_per_channel:
                    save_folder = p._run_dir / step['Color']
                    save_folder.mkdir(parents=True, exist_ok=True)
                else:
                    save_folder = p._run_dir

                output_format = p._image_capture_config.output_format_sequenced
                if output_format == image_mode.OUTPUT_FORMAT_HYPERSTACK:
                    output_format = 'TIFF'

                # Video encoding runs on FILE_WORKER after capture -- no gate needed

                # Whether to hold this channel lit across the step boundary is
                # the LED authority's STEP_BOUNDARY decision. The caller reads
                # the protocol and precomputes primitives into the boundary
                # ctx; the authority owns the policy (hold within a z-stack
                # group, hold across a same-color move only on the opt-in, go
                # dark across a scan boundary, hold on the final step when
                # run-end will re-light this same channel). The decision is
                # applied through the authority after the capture completes
                # (below), not threaded into the capture leaf.
                num_steps = p._protocol.num_steps()
                same_color = False
                same_zstack_group = False
                is_scan_boundary = False
                is_run_end_boundary = False
                end_policy = LedEndPolicy.OFF
                snapshot_lit = frozenset()
                if p._curr_step < num_steps - 1:
                    next_step = p._protocol.step(idx=p._curr_step + 1)
                    same_color = next_step['Color'] == step['Color']
                    # A z-stack group is one channel acquired across z slices, so
                    # group membership already implies the same color.
                    same_zstack_group = (
                        same_color
                        and step['Z-Stack Group ID'] != -1
                        and next_step['Z-Stack Group ID'] == step['Z-Stack Group ID']
                    )
                elif p.remaining_scans() <= 1:
                    # Final step of the final scan -- the run-end boundary. The
                    # authority holds this channel only if the run-end target
                    # re-lights it, so the boundary off plus the restore a few
                    # ms later is not a visible end-of-acquire flicker on a
                    # live-view-lit z-stack. Feed the SAME end-state the cleanup
                    # RUN_END uses (resolved once), so the boundary and cleanup
                    # cannot disagree about what run-end will light.
                    is_run_end_boundary = True
                    resolved_policy, snapshot_lit = resolve_end_state(
                        p._leds_state_at_end,
                        getattr(p, '_original_led_states', None),
                        p._scope.illumination.state_color2ch,
                    )
                    if resolved_policy is not None:
                        end_policy = resolved_policy
                else:
                    # Last step of a non-final scan: the inter-scan idle runs dark.
                    is_scan_boundary = True
                # An unmapped channel makes the boundary target empty. That
                # covers three states: no LED board (diff is a no-op), a
                # layer that drives no LED (nothing to hold), and a layer
                # this unit's identity lacks (the light seam already said
                # so loudly) -- in every case there is nothing to keep lit.
                boundary_ctx = LedTransitionCtx(
                    channel=p._scope.illumination.color2ch(step['Color']),
                    illumination_ma=step['Illumination'],
                    same_color=same_color,
                    same_zstack_group=same_zstack_group,
                    keep_led_across_moves=p._keep_led_between_steps,
                    is_scan_boundary=is_scan_boundary,
                    is_run_end_boundary=is_run_end_boundary,
                    end_policy=end_policy,
                    snapshot_lit=snapshot_lit,
                )

                # Illuminate the step's channel and confirm it on BEFORE the
                # grab. The capture leaf no longer drives the LED -- it is a pure
                # grab+save -- so the run-lifecycle illuminate lives here on the
                # authority, the symmetric twin of the STEP_BOUNDARY off applied
                # after capture. Idempotent on an AG step (already lit when AG
                # armed) and on a held same-color channel.
                self._apply_step_light(step)

                _t_capture_start = time.monotonic()
                completed = p._image_writer.capture(
                    save_folder=save_folder,
                    step=step,
                    output_format=output_format,
                    protocol=p._protocol,
                    scan_count=p._scan_count,
                    sum_count=step['Sum'],
                    enable_image_saving=p._enable_image_saving,
                    autogain_settings=p._autogain_settings,
                    video_as_frames=p._video_as_frames,
                    separate_folder_per_channel=p._separate_folder_per_channel,
                    curr_step=p._curr_step,
                )
                _t_capture_done = time.monotonic()
                logger.debug(
                    f'[TIMING] Step {p._curr_step} capture+save: {(_t_capture_done - _t_capture_start) * 1000:.1f}ms'
                )
                # Record the scan's first completed acquisition; the run loop
                # anchors scan 1's timelapse period on it (a scan whose every
                # capture fails keeps the run-start fallback anchor).
                if completed and p._scan_first_capture_t is None:
                    p._scan_first_capture_t = _t_capture_start
                # Drive the step-boundary LED decision through the authority,
                # but only when the capture completed with the LED left lit. A
                # failed or aborted capture has already turned the LED off as
                # cleanup, so applying a hold target here would re-light it.
                # The fatal gate closes the cross-thread race: a fatal abort
                # raised from the file-IO thread (disk floor) can land while
                # this capture is completing, and its force_off must not be
                # undone by a boundary HOLD here. Keyed on the fatal flag,
                # NOT the abort flag -- on a user Stop the boundary's common
                # case is an EXTINGUISH, and skipping that would lengthen
                # sample illumination; Stop behavior stays exactly as it was.
                if completed and not p._fatal_abort_event.is_set():
                    self.apply_led_transition(LedTransition.STEP_BOUNDARY, boundary_ctx)

            else:
                # No saving -- turn off LEDs manually (capture normally does this)
                self.leds_off()

        # Disable autogain when moving between steps. Run-internal
        # machinery binds the impl, as the arm above does: the public
        # member is a dispatcher that refuses work while a run holds the
        # camera lane, and a refusal here raised out of the step, was
        # classified transient, and retried the whole scan.
        if step['Auto_Gain']:
            fut = p._io_executor.protocol_put(
                IOTask(
                    action=p._scope.imaging._set_auto_gain_impl,
                    kwargs={
                        'state': False,
                        'settings': p._autogain_settings,
                    },
                ),
                return_future=True,
            )
            if fut:
                # 30s window: leaves headroom under Pylon USB3 stress where
                # a single io_executor task can stretch past 5s without
                # being a real failure. Cluster with leds_off / led_on
                # below and restore_camera_state in protocol_cleanup.
                fut.result(timeout=30)

        logger.debug(
            f'[TIMING] Step {p._curr_step} total: {(time.monotonic() - p._step_start_time) * 1000:.1f}ms'
        )

        num_steps = p._protocol.num_steps()
        if p._curr_step < num_steps - 1:
            with p._protocol_state_lock:
                p._curr_step = min(p._curr_step + 1, num_steps - 1)
                p._af_future = None
                # The handled-latch travels with the future pointer: the next
                # step starts a fresh AF run whose outcome must be consumed anew.
                p._af_result_consumed = False

            if p._callbacks.update_step_number:
                _schedule_ui(lambda dt: p._callbacks.update_step_number(p._curr_step + 1), 0)
            self.go_to_step(step_idx=p._curr_step)
            return

        # End of scan -- grease redistribution if needed
        if p._autofocus_count >= 100:
            self.perform_grease_redistribution()
            p._autofocus_count = 0

        p._scan_in_progress.clear()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def default_move(self, px=None, py=None, z=None):
        """Move to plate coordinates, converting to stage coordinates.

        Each axis move is submitted through ``io_executor.protocol_put``
        and the protocol thread waits on the future, so all motor I/O
        is serialized to one worker. Calling
        ``scope.motion.move_absolute`` directly from PROTOCOL_WORKER
        instead would let motor serial writes from this thread
        interleave with any io_executor-queued motor command (UI
        sliders, manual moves) mid-step.
        """
        p = self._p
        labware = p._wellplate_loader.get_plate(plate_key=p._protocol.labware())

        if (px is not None) and (py is not None):
            sx, sy = p._coordinate_transformer.plate_to_stage(
                labware=labware,
                stage_offset=p._stage_offset,
                px=px,
                py=py,
            )

            self._move_axis_through_io('X', sx)
            p._target_x_pos = sx
            if p._callbacks.move_position:
                _schedule_ui(lambda dt: p._callbacks.move_position('X'), 0)

            self._move_axis_through_io('Y', sy)
            p._target_y_pos = sy
            if p._callbacks.move_position:
                _schedule_ui(lambda dt: p._callbacks.move_position('Y'), 0)

            if z is not None:
                self._move_axis_through_io('Z', z)
                p._target_z_pos = z
                if p._callbacks.move_position:
                    _schedule_ui(lambda dt: p._callbacks.move_position('Z'), 0)

    def _move_axis_through_io(self, axis: str, position):
        """Submit a single-axis move to io_executor and wait for completion.

        Used by ``default_move``, which runs on PROTOCOL_WORKER, to keep
        motor writes off that thread. Falls back to a direct call if the
        executor isn't available (early init / standalone tests).

        Only a caller OFF the io worker may use this: the io worker cannot
        reach a task queued behind the one it is running, so enqueue-and-
        wait from the worker itself can never complete -- code already on
        the worker (the grease routine) calls the move body directly.
        """
        p = self._p
        kwargs = {
            'axis': axis,
            'position': position,
            'wait_until_complete': False,
        }
        if p._io_executor is None:
            p._scope.motion._move_absolute_impl(**kwargs)
            return
        fut = p._io_executor.protocol_put(
            IOTask(action=p._scope.motion._move_absolute_impl, kwargs=kwargs),
            return_future=True,
        )
        if fut:
            fut.result(timeout=60.0)

    def go_to_step(self, step_idx: int):
        """Move to the position for a given protocol step."""
        p = self._p
        p._step_start_time = time.monotonic()
        p._motion_wait_start = None
        if p._aborted.is_set():
            return

        if p._callbacks.go_to_step:
            p._callbacks.go_to_step(
                protocol=p._protocol,
                step_idx=step_idx,
                include_move=True,
                ignore_auto_gain=True,
            )
        else:
            step = p._protocol.step(idx=step_idx)
            self.default_move(px=step['X'], py=step['Y'], z=step['Z'])

    # ------------------------------------------------------------------
    # Grease redistribution
    # ------------------------------------------------------------------

    def perform_grease_redistribution(self):
        p = self._p
        p._grease_redistribution_event.clear()
        result = p._io_executor.protocol_put(IOTask(action=self._grease_redist_w_pos))
        if result is not PROTOCOL_ENQUEUED:
            # The grease task did not enter the queue -- io executor disabled,
            # protocol not running, or queue full all return a non-enqueued
            # value -- so the matching set() inside _grease_redist_w_pos's
            # finally will never run. Release the gate now or the next scan's
            # scan_iterate gate blocks forever waiting on a task that does not
            # exist, a silent hang the consecutive-failure cap cannot catch.
            logger.warning(
                '[PROTOCOL] Grease redistribution task was not queued '
                '(io executor unavailable or queue full); skipping it and '
                'releasing the scan gate'
            )
            p._grease_redistribution_event.set()

    def _grease_redist_w_pos(self):
        p = self._p
        axis = 'Z'
        _t_start = time.monotonic()
        try:
            # get_current_position is a cache read (a zero-serial-IO accessor);
            # safe to call directly from any thread that needs the live z position.
            z_orig = p._scope.motion.get_current_position(axis=axis)
            # This routine already RUNS on the io worker, so the moves run
            # inline: enqueueing them back onto the single worker and waiting
            # could never complete -- the worker cannot reach a task queued
            # behind the one it is running, so the wait always expired, the
            # queued Z->0 then ran out of band on unwind, and the restore to
            # z_orig never enqueued at all.
            p._scope.motion._move_absolute_impl(
                axis, 0, wait_until_complete=True, overshoot_enabled=True
            )

            if p._callbacks.move_position:
                _schedule_ui(lambda dt, a=axis: p._callbacks.move_position(a))

            p._scope.motion._move_absolute_impl(
                axis, z_orig, wait_until_complete=True, overshoot_enabled=True
            )

            if p._callbacks.move_position:
                _schedule_ui(lambda dt, a=axis: p._callbacks.move_position(a))

            elapsed = time.monotonic() - _t_start
            if elapsed > 30:
                logger.warning(
                    f'[PROTOCOL] Grease redistribution took {elapsed:.1f}s (> 30s threshold)'
                )
            else:
                logger.debug(f'[PROTOCOL] Grease redistribution completed in {elapsed:.1f}s')
        finally:
            # ALWAYS release the gate, even if a move raised. A raise propagates
            # to the io_executor task runner (which logs it with a traceback), so
            # the failure is still surfaced -- but the gate MUST be released
            # first or the next scan's scan_iterate gate blocks forever, a silent
            # hang the consecutive-failure cap cannot catch (this runs
            # fire-and-forget; nothing reaches the run loop).
            p._grease_redistribution_event.set()

    # ------------------------------------------------------------------
    # LED control
    # ------------------------------------------------------------------

    def leds_off(self):
        """Turn all LEDs off via the IO executor.

        UI update is handled by the LED observer -- no manual callback needed.
        """
        p = self._p
        fut = p._io_executor.protocol_put(
            IOTask(action=p._scope.illumination._leds_off_impl), return_future=True
        )
        if fut:
            fut.result(timeout=30)
        else:
            try:
                p._scope.illumination._leds_off_impl()
            except Exception as ex:
                logger.warning(f'[{p.LOGGER_NAME}] Direct leds_off fallback failed: {ex}')
        # LED observer handles UI sync -- no manual callback

    def apply_led_transition(self, transition: LedTransition, ctx: LedTransitionCtx) -> None:
        """Drive an LED lifecycle transition through the run's LED authority.

        Submitted on the protocol IO queue so the transition's LED commands
        serialize with the run's moves and captures -- a move must not race the
        LEDs off at a well boundary. No-op when the run holds no lease.

        Whether the submitter waits for completion is a property of the
        transition (FIRE_AND_FORGET_TRANSITIONS): the scan-idle darkening is
        enqueued without waiting, because on the failure-retry path the
        executor carrying it may be the very thing that wedged the scan and
        FIFO ordering already places the off before anything later.
        """
        p = self._p
        lease = getattr(p, '_led_lease', None)
        if lease is None:
            return
        fut = p._io_executor.protocol_put(
            IOTask(action=lease.apply, args=(transition, ctx)), return_future=True
        )
        if fut:
            if transition not in FIRE_AND_FORGET_TRANSITIONS:
                try:
                    fut.result(timeout=30)
                except CancelledError:
                    if transition is not LedTransition.RUN_END:
                        raise
                    # The protocol queue is cleared the moment
                    # protocol-mode ends, and a run's terminal RUN_END
                    # can be enqueued exactly then -- a ROUTINE end-of-
                    # run ordering, not only an overlapping abort. The
                    # cancelled task never ran, the queue's FIFO
                    # serialization is moot once it is cleared, and the
                    # still-held lease serializes against any new run
                    # (it cannot acquire until this one releases), so
                    # the end-state applies directly. A cancelled
                    # MID-RUN transition stays dead: it belongs to
                    # whatever abort cleared the queue, and lighting a
                    # channel from here during teardown would undo that
                    # abort's darkening.
                    lease.apply(transition, ctx)
        else:
            lease.apply(transition, ctx)

    def _apply_step_light(self, step) -> None:
        """Illuminate the step's channel through the authority, confirmed on.

        The LED must be lit before the camera grabs the step's frame (a dark
        grab is mis-exposed) and before continuous auto-gain arms against the
        scene. STEP_LIGHT is a confirm-on transition, so apply blocks until the
        board reports the channel on; the short settle after covers the board's
        on-to-stable lag before the grab. A same-color step that kept its channel
        lit is left untouched (the diff self-skips), so consecutive z-slices do
        not blink off->on.
        """
        p = self._p
        if p._aborted.is_set():
            # Aborting: do not re-illuminate the sample during teardown -- the
            # abort path is turning the LEDs off, and a stray on here flashes
            # the sample at cancel time.
            return
        if not p._scope.led_connected:
            # A disconnected LED board makes every step grab a dark frame; say so
            # at the capture point so a black-frame run is diagnosable.
            logger.warning(
                '[Capture   ] LED controller not available; step channel not illuminated.'
            )
            return
        channel = p._scope.illumination.color2ch(step['Color'])
        if channel is None and step['Color'] in common_utils.get_layers_with_led():
            # The board is connected (checked above) and the layer is one
            # that drives an LED, so an unresolvable name here means this
            # unit's identity has no such layer: the step will capture
            # dark, deterministically, every scan. Say so per step -- a
            # run in progress gets logs, not popups -- rather than let
            # the frames come back dark with no named cause.
            logger.error(
                f"[Capture   ] This scope has no '{step['Color']}' LED "
                f'channel; the step will capture dark.'
            )
        self.apply_led_transition(
            LedTransition.STEP_LIGHT,
            LedTransitionCtx(
                channel=channel,
                illumination_ma=step['Illumination'],
            ),
        )
        time.sleep(0.005)
