# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Protocol cleanup / shutdown logic.

Restores LED, autofocus, camera state and fires completion callbacks.
Extracted from ``sequenced_capture_runner.py`` during the
protocol-decomposition refactor.
"""

from __future__ import annotations

import threading
from concurrent.futures import CancelledError
from functools import partial
from typing import TYPE_CHECKING

from lvp_logger import logger

from modules.lumascope_api.illumination import (
    LedTransition,
    LedTransitionCtx,
    resolve_end_state,
)
from modules.protocol_state_machine import ProtocolState
from modules.sequential_io_executor import (
    IOTask,
    PROTOCOL_QUEUE_WEDGED,
    SLOW_WRITE_BLOCKED_WARN_S,
)

if TYPE_CHECKING:
    from modules.lumascope_api import Lumascope
    from modules.protocol_callbacks import ProtocolCallbacks


from modules.kivy_utils import schedule_ui as _schedule_ui

# Stall budget for queueing the run-record completion task. Short: on the
# normal path the queue is draining (the put unblocks within one write), and
# cleanup must not hang behind a wedged writer for the writer's own longer
# fatal budget just to file the record.
_RECORD_COMPLETE_STALL_S = 2.0


def run_cleanup(
    *,
    # State
    get_state_fn,
    set_state_fn,
    run_lock: threading.Lock,
    scan_in_progress: threading.Event,
    # True when the run died on a fatal fault (stalled writer, dead camera,
    # disk floor) rather than finishing or being stopped by the user.
    # Required, not defaulted: every caller states which kind of end this is.
    fatal_abort: bool,
    # Saved original states
    leds_state_at_end: str,
    original_led_states: dict,
    original_autofocus_states: dict,
    saved_camera_state: dict,
    return_to_position: dict | None,
    disable_saving_artifacts: bool,
    protocol,
    protocol_execution_record,
    # Dependencies
    scope: Lumascope,
    callbacks: ProtocolCallbacks,
    # Executor functions
    apply_led_transition_fn,
    default_move_fn,
    cancel_scheduled_events_fn,
    # IO executors
    io_executor,
    autofocus_thread,
    file_io_executor,
    camera_executor,
    # Mutable flag -- set to False when done
    set_run_in_progress_fn,
    logger_name: str = 'SequencedCaptureRunner',
    # Terminal outcome the run_complete subscribers receive
    run_status: str,
):
    """Core cleanup logic -- restores state, fires callbacks, ends executors.

    Called from ``SequencedCaptureRunner._cleanup_inner()``. run_status
    ('completed', 'aborted', 'failed', 'failed_at_start') is required so
    the cleanup site states the run's true terminal outcome; it reaches
    every run_complete subscriber as the ``status`` kwarg.
    """
    # Capture the abort state BEFORE the COMPLETING transition below. Only a
    # hardware-error abort (ERROR state) clears file_io_executor's pending queue:
    # on a hardware fault the queued frames are suspect, and letting them drain
    # can pin memory and lock the next protocol-start. Every other abort path
    # (user Stop, disk-full, 3-strike camera) leaves ERROR unset, so its pending
    # writes DRAIN to disk instead of being dropped. Considered routing those
    # through is_aborted too so Stop returns control instantly; rejected -- an
    # already-captured frame must not be discarded because the user stopped the
    # run; preserving the captured data wins over the faster stop. Revisit if
    # draining a large pending queue on Stop becomes a real usability problem.
    is_aborted = get_state_fn() == ProtocolState.ERROR

    # Transition to COMPLETING (or stay in ERROR if that's how we got here)
    if get_state_fn() not in (ProtocolState.COMPLETING, ProtocolState.ERROR, ProtocolState.IDLE):
        set_state_fn(ProtocolState.COMPLETING)

    # Cleanup runs because abort already fired (or the run is finishing
    # naturally). The abort signal -- now owned by protocol_thread -- is
    # already set if this is an abort; setting it again here would be
    # redundant and was dropped in B3.
    scan_in_progress.clear()

    # Collect cleanup-step failures so a single summary notification at
    # the end tells the user what went wrong. Each except continues to
    # the next step (fault tolerance -- all six must run regardless of
    # any one failing); total silence at the end was the bug. One
    # summary popup, not six.
    cleanup_errors: list[str] = []

    try:
        cancel_scheduled_events_fn()
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error cancelling scheduled events during cleanup: {ex}')
        cleanup_errors.append(f'Cancel scheduled events: {type(ex).__name__}: {ex}')

    # --- Unwind any in-flight autofocus BEFORE restoring LEDs ---
    # The AF worker lights its own channel during setup and restores
    # LED / camera / Z state in its finally block. If the LED restore
    # below ran first, a still-unwinding AF run would re-light or
    # re-restore on top of it, and the protocol's intended end state
    # would lose the race -- worst case an AF LED left on overnight.
    # The AF Future resolves only after that finally chain finishes,
    # so waiting on it (bounded, so a wedged AF run cannot block
    # cleanup) guarantees the LED restore below runs last.
    # A run that failed during start() never dispatched anything, so a live
    # AF future here belongs to SOMEONE ELSE -- most likely the very holder
    # whose lease refusal failed this run. Aborting it would steal the
    # operation the refusal deferred to.
    if autofocus_thread is not None and run_status != 'failed_at_start':
        _af_future = autofocus_thread.current_future
        if _af_future is not None and not _af_future.done():
            autofocus_thread.abort()
            try:
                # Returns the run's exception (normally AutofocusAborted)
                # without raising it; raises TimeoutError on the bound.
                _af_future.exception(timeout=5.0)
            except TimeoutError:
                logger.warning(
                    f'[{logger_name}] Cleanup: autofocus still unwinding '
                    'after 5.0 s; its exit path restores LED/camera state '
                    'when it finishes'
                )
            except Exception as ex:
                logger.warning(
                    f'[{logger_name}] Cleanup: error waiting for autofocus '
                    f'to unwind: {type(ex).__name__}: {ex}'
                )

    # --- Restore LEDs ---
    # One authority diff sets the run's end-state: off, or back to the
    # channels that were lit before the run. The diff turns off any channel
    # not in the target set, then asserts the target -- so restoring more
    # than one pre-run channel does not flash, where the old per-channel
    # restore loop extinguished each channel as it lit the next. An empty
    # restore target (nothing was lit pre-run) collapses to off by
    # construction. apply(RUN_END) runs on the still-held lease and serializes
    # on the protocol IO queue, so the end-state off cannot race the
    # return-to-position move across the shared serial bus.
    try:
        # A fatal abort's terminal LED state is DARK regardless of the user's
        # end policy: force_off already darkened the sample at the fault
        # site, and this forced-OFF RUN_END re-asserts dark against any step
        # that raced the abort and re-lit a channel (the OFF diff serializes
        # after such a re-light on the same FIFO protocol queue, so off
        # wins). Asserting OFF -- not skipping the restore -- is the point: a
        # skipped restore would leave a raced re-light on forever. User Stop
        # keeps the configured policy.
        end_policy, snapshot_lit = resolve_end_state(
            'off' if fatal_abort else leds_state_at_end,
            original_led_states,
            scope.illumination.color2ch,
        )
        if end_policy is None:
            logger.error(f'Unsupported LEDs state at end value: {leds_state_at_end}')
        else:
            apply_led_transition_fn(
                LedTransition.RUN_END,
                LedTransitionCtx(end_policy=end_policy, snapshot_lit=snapshot_lit),
            )
    except CancelledError:
        # An overlapping abort / new-run cycle cleared the protocol queue
        # and cancelled this restore task before it ran. The superseding
        # cycle owns LED state from here; this is a normal hand-off, not a
        # failure -- surfacing it as one produced a popup per cycle when
        # the run button was clicked rapidly.
        logger.info(
            f'[{logger_name}] Cleanup: LED restore superseded by an overlapping run/abort cycle'
        )
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error restoring LED states during cleanup: {ex}')
        cleanup_errors.append(f'Restore LED states: {type(ex).__name__}: {ex}')
    logger.info(f'[{logger_name}] Cleanup: LED restore complete')

    # --- Restore layer shader / false-color (UI side) ---
    # Each protocol step calls layer_control.apply_settings() which
    # writes the OpenGL shader white_point for that layer's
    # false-color (Red tint for the Red step, Green tint for Green,
    # etc.). Without this restore the last step's shader stays
    # active and tints the live preview after protocol stop. Cluster
    # sibling of LED-state-hygiene-at-transition (#666 / #659 /
    # #617): driver LED state was already cleared above; this is
    # the sibling UI-shader-state clear. Bugs cluster -- one cleanup
    # pass covers both halves.
    try:
        if callbacks.restore_layer_shader:
            _schedule_ui(lambda dt: callbacks.restore_layer_shader(), 0)
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error restoring layer shader during cleanup: {ex}')
        cleanup_errors.append(f'Restore layer shader: {type(ex).__name__}: {ex}')

    # --- Restore autofocus states ---
    # Guard against None / empty (the common case when no AF was active
    # for this scan). Without the guard, iteration on None raised
    # `'NoneType' object is not subscriptable` and fired ERROR every
    # scan, burying real failure signal under thousands of spurious lines.
    if not original_autofocus_states:
        logger.debug('[PROTOCOL] No autofocus states to restore')
    else:
        try:
            for layer, layer_data in original_autofocus_states.items():
                if callbacks.restore_autofocus_state:
                    callbacks.restore_autofocus_state(layer=layer, value=layer_data)
                else:
                    import modules.app_context as _app_ctx
                    from modules.settings_init import settings

                    ctx = _app_ctx.ctx
                    if ctx is not None:
                        with ctx.settings_lock:
                            settings[layer]['autofocus'] = layer_data
                    else:
                        settings[layer]['autofocus'] = layer_data
                if callbacks.reset_autofocus_btns:
                    _schedule_ui(lambda dt: callbacks.reset_autofocus_btns(), 0)
        except Exception as ex:
            logger.error(f'[PROTOCOL] Error restoring autofocus states during cleanup: {ex}')
            cleanup_errors.append(f'Restore autofocus states: {type(ex).__name__}: {ex}')

    # --- Restore camera gain and exposure ---
    # PROTO-CLEAN-1: dispatch the gain/exposure SDK calls through
    # camera_executor (CAMERA_WORKER) instead of running on MainThread.
    # Pylon's set_gain / set_exposure_time take noticeable time on real
    # hardware -- running on MainThread blocked the UI for the duration
    # of protocol stop. Submit-and-wait so cleanup still serializes:
    # the next steps (return-to-position, executor end) need camera
    # state restored before they run, otherwise live preview after stop
    # could briefly use protocol gain/exposure.
    #
    # Cleanup runs while ``protocol_running`` is still set, so we use
    # ``protocol_put`` (which accepts during a running protocol) rather
    # than ``put`` (which rejects until protocol_end fires).
    #
    # That choice is necessary but not sufficient: the camera executor is
    # also DISABLED for the duration of a run and is not re-enabled until
    # the end-executors step further down this function, and protocol_put
    # refuses while disabled. So on a normal run the enqueue below returns
    # None and the direct-call branch is what actually restores state.
    try:
        if saved_camera_state:
            tag = saved_camera_state.get('tag', '?')
            fut = camera_executor.protocol_put(
                IOTask(
                    action=scope.imaging.restore_camera_state,
                    args=(saved_camera_state,),
                ),
                return_future=True,
            )
            if fut is not None:
                # Reached only when the executor is still live -- a run that
                # failed before the disable, or a caller driving cleanup
                # without a run behind it.
                logger.info(
                    f'[{logger_name}] Cleanup: restoring camera state tag={tag} (via CAMERA_WORKER)'
                )
                fut.result(timeout=30)
            else:
                # The normal path: the enqueue was refused because the camera
                # executor is disabled, so restore inline on the cleanup
                # thread. State still gets restored either way; the log says
                # which thread did it so a trace of this run is readable.
                logger.info(
                    f'[{logger_name}] Cleanup: restoring camera state '
                    f'tag={tag} (direct -- camera executor disabled)'
                )
                scope.imaging.restore_camera_state(saved_camera_state)
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error restoring camera gain/exposure during cleanup: {ex}')
        cleanup_errors.append(f'Restore camera gain/exposure: {type(ex).__name__}: {ex}')

    # --- Return to position ---
    try:
        if return_to_position is not None:
            logger.info(
                f'[{logger_name}] Cleanup: returning to position '
                f'x={return_to_position["x"]}, y={return_to_position["y"]}, z={return_to_position["z"]}'
            )
            default_move_fn(
                px=return_to_position['x'],
                py=return_to_position['y'],
                z=return_to_position['z'],
            )
            logger.info(f'[{logger_name}] Cleanup: return-to-position move issued')
    except CancelledError:
        # Same hand-off as the LED restore above: a superseding run/abort
        # cycle cancelled the queued move; the new cycle owns stage position.
        logger.info(
            f'[{logger_name}] Cleanup: return-to-position superseded by an '
            'overlapping run/abort cycle'
        )
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error returning to position during cleanup: {ex}')
        cleanup_errors.append(f'Return to position: {type(ex).__name__}: {ex}')

    # --- End executors ---
    scan_in_progress.clear()

    io_executor.protocol_end()
    # Wait for any task that was in-flight when protocol_end fired to
    # finish before we mutate scope / camera / settings state below --
    # an in-flight task on the io_executor worker may be reading the
    # same state. Bounded so a wedged task can't block cleanup
    # indefinitely; if the timeout fires we log and proceed.
    if not io_executor.wait_for_idle(timeout=2.0):
        logger.warning(
            f'[{logger_name}] Cleanup: io_executor still mid-task '
            'after 2.0 s wait; proceeding to teardown anyway'
        )
    if autofocus_thread is not None:
        # Signal any lingering AF run to unwind. abort() is a no-op when
        # the thread is idle, so this is always safe to call.
        autofocus_thread.abort()
    camera_executor.enable()
    logger.info(f'[{logger_name}] Cleanup: protocol_end called on all executors')

    io_executor.clear_protocol_pending()
    if is_aborted:
        # Drop pending writes only on an ERROR-state abort. Drain (the
        # COMPLETING-path default) writes everything queued to disk before
        # releasing memory -- correct on normal completion AND on a user Stop
        # (don't discard captured frames), but on a hardware disconnect/error the
        # frames are suspect and the user wants control back without waiting for
        # many GB to slowly drain.
        file_io_executor.clear_protocol_pending()
        logger.info(f'[{logger_name}] Cleanup: file_io_executor pending cleared (aborted)')

    # --- Complete protocol execution record ---
    # Ordering invariant: this enqueue must run AFTER the abort-path clear
    # above. Enqueued before it, the completion task itself was cancelled by
    # the clear, so an aborted run's record silently never finalized. After
    # the clear, an aborted run's queue has room and the put returns
    # immediately even when the worker is stuck mid-write.
    try:
        if not disable_saving_artifacts and protocol_execution_record is not None:
            # On a clean finish, reconcile attempted captures against rows
            # written and warn on any shortfall. On abort, pending writes were
            # dropped on purpose above, so a shortfall is expected -- skip it.
            # Blocking put: the old fire-and-forget enqueue ignored the
            # queue-full return, so a backed-up queue silently lost the
            # record completion.
            _record_put = file_io_executor.protocol_put_wait(
                IOTask(
                    action=partial(protocol_execution_record.complete, reconcile=not is_aborted)
                ),
                should_abort=lambda: False,
                stall_timeout_s=_RECORD_COMPLETE_STALL_S,
            )
            if _record_put is PROTOCOL_QUEUE_WEDGED:
                logger.error(
                    f'[{logger_name}] Cleanup: run-record completion could not '
                    f'be queued -- the file writer is stalled on '
                    f"{file_io_executor.describe_running_task()}; this run's "
                    f'execution record will not be finalized'
                )
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error completing protocol record during cleanup: {ex}')
        cleanup_errors.append(f'Complete protocol record: {type(ex).__name__}: {ex}')

    with run_lock:
        set_run_in_progress_fn(False)
        # Transition back to IDLE from COMPLETING or ERROR
        if get_state_fn() in (ProtocolState.COMPLETING, ProtocolState.ERROR):
            set_state_fn(ProtocolState.IDLE)

    # Surface a single summary if any cleanup step failed. Fault
    # tolerance ran each step regardless; the user needs to know LED
    # state, camera settings, or stage position may not be what they
    # expect.
    if cleanup_errors:
        try:
            from modules.notification_center import notifications

            err_summary = '\n'.join(f'  - {e}' for e in cleanup_errors)
            # "ended", not "completed": this summary also fires on aborted
            # runs, and claiming completion on an abort misleads the
            # post-mortem reader.
            notifications.warning(
                'Protocol',
                'Protocol cleanup issues',
                f'Protocol ended but {len(cleanup_errors)} cleanup step(s) failed:\n'
                f'{err_summary}\n'
                f'Check LED state, camera settings, and stage position.',
            )
        except Exception as ex:
            # Best-effort -- a notification failure during cleanup must
            # not prevent the completion callbacks from firing.
            logger.error(f'[PROTOCOL] Failed to surface cleanup-error notification: {ex}')

    # Surface silently-dropped captures. A full write queue discards an
    # already-grabbed frame, so a nonzero count is images the user expected
    # that are permanently absent from disk. A throttled log was the only prior
    # signal; the run-terminal summary is the reliable surface because mid-run
    # popups are suppressed. Fires on aborted runs too -- a queue-full drop
    # during capture is unintended loss, distinct from an abort's deliberate
    # drop of pending writes.
    dropped_captures = file_io_executor.protocol_dropped_count()
    if dropped_captures > 0:
        try:
            from modules.notification_center import notifications

            notifications.warning(
                'Protocol',
                'Protocol Captures Dropped',
                f'{dropped_captures} captured image(s) could not be saved because '
                'the file writer fell behind the camera. Those images are lost '
                'from this run. Reduce the capture rate (fewer channels or '
                'Z-steps, or a slower scan) or use a faster save drive.',
            )
        except Exception as ex:
            logger.error(f'[PROTOCOL] Failed to surface dropped-capture notification: {ex}')

    # Sustained-slow-write warning, demand-relative: the time this run's
    # capture loop spent blocked waiting for a write slot. An absolute MB/s
    # floor false-fires on healthy machines (PERFORMANCE_BUDGETS.md
    # protocol_write_backpressure_wait_s), so the trigger is the run's own
    # unmet demand. Surfaced at run end because mid-run non-fatal popups are
    # suppressed; the first crossing already logged from the executor.
    blocked_s = file_io_executor.protocol_backpressure_blocked_s()
    if blocked_s >= SLOW_WRITE_BLOCKED_WARN_S:
        try:
            from modules.notification_center import notifications

            notifications.warning(
                'Protocol',
                'Very Slow File Writes',
                'Very slow writes are occurring on the save disk. '
                'Please confirm your computer and storage are OK.',
            )
        except Exception as ex:
            logger.error(f'[PROTOCOL] Failed to surface slow-write notification: {ex}')

    # --- Fire completion callbacks ---
    _file_queue_active = file_io_executor.is_protocol_queue_active()
    # Log the pending-write count so a post-run read shows HOW MANY files were
    # still draining at protocol end, not just that the queue was non-empty.
    _file_queue_depth = file_io_executor.protocol_queue_size()
    logger.info(
        f'[{logger_name}] Cleanup: file queue active={_file_queue_active} '
        f'pending_writes={_file_queue_depth}'
    )
    if _file_queue_active:
        if callbacks.run_complete:
            _schedule_ui(lambda dt: callbacks.run_complete(protocol=protocol, status=run_status), 0)
        if callbacks.files_complete:
            file_io_executor.set_protocol_complete_callback(
                callback=lambda: _schedule_ui(
                    lambda dt: callbacks.files_complete(protocol=protocol), 0
                )
            )
        file_io_executor.protocol_finish_then_end()
        logger.info(
            f'[{logger_name}] Cleanup: callbacks scheduled (run_complete now, files_complete deferred)'
        )
    else:
        if callbacks.run_complete:
            _schedule_ui(lambda dt: callbacks.run_complete(protocol=protocol, status=run_status), 0)
        if callbacks.files_complete:
            _schedule_ui(lambda dt: callbacks.files_complete(protocol=protocol), 0)
        file_io_executor.protocol_finish_then_end()
        logger.info(
            f'[{logger_name}] Cleanup: callbacks scheduled (run_complete + files_complete immediate)'
        )

    # Map the footprint right after a protocol run. No-op unless the memory
    # profiler is enabled.
    from lib import memory_profile

    memory_profile.snapshot('post_protocol')
