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
from typing import TYPE_CHECKING

from lvp_logger import logger

from modules.protocol_state_machine import ProtocolState
from modules.sequential_io_executor import IOTask

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

        while self._p._scan_in_progress.is_set() and not self._p._aborted.is_set():
            # Periodic cleanup and watchdog logging for long runs
            now_mono = time.monotonic()
            if now_mono - last_maintenance_time > 60:
                last_maintenance_time = now_mono

                collected = gc.collect()
                if collected > 0:
                    logger.info(f"[Scan Watchdog] GC collected {collected} objects")

            # Run one step iteration. Exceptions propagate to the
            # outer run loop for fatal/transient classification.
            self.scan_iterate()

            # Small delay to prevent CPU throttling
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Single step iteration
    # ------------------------------------------------------------------

    def scan_iterate(self, dt=None):
        """Execute one iteration of the scan state machine."""
        p = self._p  # shorthand

        if p._aborted.is_set():
            return
        # Video encoding runs on FILE_WORKER in background — do NOT block
        # the next step waiting for it. Frames are already captured and queued.
        if not p._scan_in_progress.is_set():
            return
        if not p._run_in_progress_event.is_set():
            return
        if p._af_future is not None and not p._af_future.done():
            return

        if p._af_future is not None and p._af_future.done():
            _cam_gain = p._scope.get_gain() if p._scope.camera_active else '?'
            _cam_exp = p._scope.get_exposure_time() if p._scope.camera_active else '?'
            logger.info(
                f"[SCAN DIAG] AF gate passed: future.done()=True "
                f"camera_gain={_cam_gain} camera_exp={_cam_exp} step={p._curr_step}"
            )

        remaining_scans = p.remaining_scans()
        if remaining_scans <= 0:
            return

        step = p._protocol.step(idx=p._curr_step)

        # Check motion timeout
        if p._scope.motion.is_moving():
            if time.monotonic() - p._step_start_time > p.STEP_TIMEOUT_SECONDS:
                timeout_msg = f"Step {p._curr_step} timed out waiting for motion ({p.STEP_TIMEOUT_SECONDS}s)."
                logger.error(f"[PROTOCOL] {timeout_msg} — transitioning to ERROR state")
                from modules.notification_center import notifications
                notifications.error("Protocol", "Protocol Error — Motion Timeout", timeout_msg)
                p._scan_in_progress.clear()
                try:
                    p._set_state(ProtocolState.ERROR)
                except ValueError:
                    pass
            return

        if not p._grease_redistribution_event.is_set():
            return

        if p._aborted.is_set() or not p._scan_in_progress.is_set():
            return

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
        logger.debug(f"[TIMING] Step {p._curr_step} motion settle: {_settle_wait_ms:.1f}ms")

        # Camera settings (gain, exposure) and LED_ON are handled by
        # protocol_image_writer.capture() right before the actual frame grab.
        # Setting them here would duplicate the commands.
        if p._aborted.is_set() or not p._scan_in_progress.is_set():
            return

        # BF AF for fluorescence
        bf_af_for_fluor = False
        try:
            import modules.app_context as _app_ctx
            with _app_ctx.ctx.settings_lock:
                bf_af_for_fluor = _app_ctx.ctx.settings.get('protocol', {}).get('bf_af_for_fluorescence', False)
        except Exception as e:
            logger.debug(f"[Capture   ] Could not read bf_af_for_fluorescence setting: {e}")
        if bf_af_for_fluor and step['Color'] != 'BF':
            if p._autofocus_runner.best_focus_position() is not None:
                if p._update_z_pos_from_autofocus:
                    new_z_pos = p._autofocus_runner.best_focus_position()
                    p._protocol.modify_step_z_height(step_idx=p._curr_step, z=new_z_pos)
                logger.info(f'[Capture   ] Skipping AF on {step["Color"]} — using BF result Z={p._autofocus_runner.best_focus_position()}')
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
            )
            return

        if step['Auto_Focus'] and p._af_future is not None and not p._af_future.done():
            return

        # Check if autogain has time-finished
        if step['Auto_Gain'] and time.monotonic() < p._auto_gain_deadline:
            return

        # Reset autogain deadline for next step
        p._auto_gain_deadline = time.monotonic() + p._autogain_settings['max_duration'].total_seconds()

        # Update Z position with autofocus results
        if step['Auto_Focus'] and p._update_z_pos_from_autofocus:
            new_z_pos = p._autofocus_runner.best_focus_position()
            if new_z_pos is not None:
                p._protocol.modify_step_z_height(step_idx=p._curr_step, z=new_z_pos)
            else:
                logger.warning('[Capture   ] Autofocus returned no position — keeping current Z')

        if p._callbacks.autofocus_complete:
            _schedule_ui(lambda dt: p._callbacks.autofocus_complete(), 0)

        if step["Auto_Focus"]:
            p._autofocus_count += 1

        # --- Capture ---
        if remaining_scans > 0:
            if not p._disable_saving_artifacts:
                if p._separate_folder_per_channel:
                    save_folder = p._run_dir / step["Color"]
                    save_folder.mkdir(parents=True, exist_ok=True)
                else:
                    save_folder = p._run_dir

                output_format = p._image_capture_config['output_format']['sequenced']
                if output_format == 'ImageJ Hyperstack':
                    output_format = 'TIFF'

                # Video encoding runs on FILE_WORKER after capture — no gate needed

                # Keep LED on between consecutive steps of the same channel
                # (e.g., Z-stack slices). Avoids unnecessary LED cycling.
                _keep_led = False
                num_steps = p._protocol.num_steps()
                if p._curr_step < num_steps - 1:
                    next_step = p._protocol.step(idx=p._curr_step + 1)
                    if next_step['Color'] == step['Color']:
                        _keep_led = True

                _t_capture_start = time.monotonic()
                p._image_writer.capture(
                    save_folder=save_folder,
                    step=step,
                    output_format=output_format,
                    protocol=p._protocol,
                    scan_count=p._scan_count,
                    sum_count=step["Sum"],
                    enable_image_saving=p._enable_image_saving,
                    image_capture_config=p._image_capture_config,
                    autogain_settings=p._autogain_settings,
                    video_as_frames=p._video_as_frames,
                    separate_folder_per_channel=p._separate_folder_per_channel,
                    curr_step=p._curr_step,
                    keep_led_on=_keep_led,
                )
                _t_capture_done = time.monotonic()
                logger.debug(f"[TIMING] Step {p._curr_step} capture+save: {(_t_capture_done - _t_capture_start)*1000:.1f}ms")

            else:
                # No saving — turn off LEDs manually (capture normally does this)
                self.leds_off()

        # Disable autogain when moving between steps
        if step['Auto_Gain']:
            fut = p._io_executor.protocol_put(IOTask(
                action=p._scope.set_auto_gain,
                kwargs={
                    "state": False,
                    "settings": p._autogain_settings,
                }
            ), return_future=True)
            if fut:
                fut.result(timeout=5)

        logger.debug(f"[TIMING] Step {p._curr_step} total: {(time.monotonic() - p._step_start_time)*1000:.1f}ms")

        num_steps = p._protocol.num_steps()
        if p._curr_step < num_steps - 1:
            with p._protocol_state_lock:
                p._curr_step = min(p._curr_step + 1, num_steps - 1)
                p._af_future = None

            if p._callbacks.update_step_number:
                _schedule_ui(lambda dt: p._callbacks.update_step_number(p._curr_step + 1), 0)
            self.go_to_step(step_idx=p._curr_step)
            return

        # End of scan — grease redistribution if needed
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
        ``scope.motion.move_absolute_position`` directly from PROTOCOL_WORKER
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

    def _move_axis_through_io(self, axis: str, pos, *,
                               wait_until_complete: bool = False,
                               overshoot_enabled: bool = False,
                               timeout: float = 60.0):
        """Submit a single-axis move to io_executor and wait for completion.

        Used by ``default_move`` and ``_grease_redist_w_pos`` to keep
        motor writes off PROTOCOL_WORKER. Falls back to a direct call if
        the executor isn't available (early init / standalone tests).
        """
        p = self._p
        kwargs = {
            'axis': axis,
            'pos': pos,
            'wait_until_complete': wait_until_complete,
            'overshoot_enabled': overshoot_enabled,
        }
        if p._io_executor is None:
            p._scope.motion.move_absolute_position(**kwargs)
            return
        fut = p._io_executor.protocol_put(
            IOTask(action=p._scope.motion.move_absolute_position, kwargs=kwargs),
            return_future=True,
        )
        if fut:
            fut.result(timeout=timeout)

    def go_to_step(self, step_idx: int):
        """Move to the position for a given protocol step."""
        p = self._p
        p._step_start_time = time.monotonic()
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
        p._io_executor.protocol_put(IOTask(action=self._grease_redist_w_pos))

    def _grease_redist_w_pos(self):
        p = self._p
        axis = 'Z'
        _t_start = time.monotonic()
        # get_current_position is a cache read (LAYER-L pinned that as a
        # zero-serial-IO accessor); safe to call directly from any thread
        # that needs the live z position.
        z_orig = p._scope.motion.get_current_position(axis=axis)
        self._move_axis_through_io(
            axis, 0,
            wait_until_complete=True, overshoot_enabled=True,
            timeout=120.0,
        )

        if p._callbacks.move_position:
            _schedule_ui(lambda dt, a=axis: p._callbacks.move_position(a))

        self._move_axis_through_io(
            axis, z_orig,
            wait_until_complete=True, overshoot_enabled=True,
            timeout=120.0,
        )

        if p._callbacks.move_position:
            _schedule_ui(lambda dt, a=axis: p._callbacks.move_position(a))

        elapsed = time.monotonic() - _t_start
        if elapsed > 30:
            logger.warning(f"[PROTOCOL] Grease redistribution took {elapsed:.1f}s (> 30s threshold)")
        else:
            logger.debug(f"[PROTOCOL] Grease redistribution completed in {elapsed:.1f}s")

        p._grease_redistribution_event.set()

    # ------------------------------------------------------------------
    # LED control
    # ------------------------------------------------------------------

    def leds_off(self):
        """Turn all LEDs off via the IO executor.

        UI update is handled by the LED observer — no manual callback needed.
        """
        p = self._p
        fut = p._io_executor.protocol_put(IOTask(
            action=p._scope.leds_off
        ), return_future=True)
        if fut:
            fut.result(timeout=5)
        else:
            try:
                p._scope.leds_off()
            except Exception as ex:
                logger.warning(f"[{p.LOGGER_NAME}] Direct leds_off fallback failed: {ex}")
        # LED observer handles UI sync — no manual callback

    def led_on(self, color: str, illumination: float, block: bool = True, force: bool = False):
        """Turn on a single LED channel via the IO executor.

        UI update is handled by the LED observer — no manual callback needed.
        """
        p = self._p
        if p._aborted.is_set() and not force:
            return

        fut = p._io_executor.protocol_put(IOTask(
            action=p._scope.led_on,
            kwargs={
                "channel": p._scope.color2ch(color),
                "mA": illumination,
                "block": block,
                "owner": "protocol",
            },
        ), return_future=True)
        if fut:
            fut.result(timeout=5)
        # Sleep for 5 ms to ensure that LED properly turns on before next action
        time.sleep(0.005)
        # LED observer handles UI sync — no manual callback
