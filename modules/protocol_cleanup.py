# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Protocol cleanup / shutdown logic.

Restores LED, autofocus, camera state and fires completion callbacks.
Extracted from ``sequenced_capture_runner.py`` during the
protocol-decomposition refactor.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from lvp_logger import logger

from modules.protocol_state_machine import ProtocolState
from modules.sequential_io_executor import IOTask

if TYPE_CHECKING:
    from modules.lumascope_api import Lumascope
    from modules.protocol_callbacks import ProtocolCallbacks


from modules.kivy_utils import schedule_ui as _schedule_ui


def run_cleanup(
    *,
    # State
    get_state_fn,
    set_state_fn,
    run_lock: threading.Lock,
    scan_in_progress: threading.Event,
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
    leds_off_fn,
    led_on_fn,
    default_move_fn,
    cancel_scheduled_events_fn,
    # IO executors
    io_executor,
    autofocus_thread,
    file_io_executor,
    camera_executor,
    # Mutable flag — set to False when done
    set_run_in_progress_fn,
    logger_name: str = 'SequencedCaptureRunner',
):
    """Core cleanup logic — restores state, fires callbacks, ends executors.

    Called from ``SequencedCaptureRunner._cleanup_inner()``.
    """
    # PF-2: capture initial state BEFORE the COMPLETING transition below so we
    # can distinguish abort (ERROR) from normal end. On abort (e.g. hardware
    # disconnect), file_io_executor's pending queue is cleared along with the
    # other executors — otherwise queued frames stay pinned in memory while
    # they slowly drain to disk, which can lock the next protocol-start.
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

    # --- Restore LEDs ---
    try:
        if leds_state_at_end == 'off':
            leds_off_fn()
        elif leds_state_at_end == 'return_to_original':
            any_restored = False
            for color, color_data in original_led_states.items():
                if color_data['enabled']:
                    led_on_fn(
                        color=color,
                        illumination=color_data['illumination_ma'],
                        block=True,
                        force=True,
                    )
                    any_restored = True
            if not any_restored:
                # "return_to_original" with no LED active pre-run is silently
                # equivalent to "off". The user-facing label says restore;
                # the only honest restore IS leds_off when nothing was on.
                leds_off_fn()
        else:
            logger.error(f'Unsupported LEDs state at end value: {leds_state_at_end}')
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
    # hardware — running on MainThread blocked the UI for the duration
    # of protocol stop. Submit-and-wait so cleanup still serializes:
    # the next steps (return-to-position, executor end) need camera
    # state restored before they run, otherwise live preview after stop
    # could briefly use protocol gain/exposure.
    #
    # Cleanup runs while ``protocol_running`` is still set, so we use
    # ``protocol_put`` (which accepts during a running protocol) rather
    # than ``put`` (which rejects until protocol_end fires).
    try:
        if saved_camera_state:
            tag = saved_camera_state.get('tag', '?')
            logger.info(
                f'[{logger_name}] Cleanup: restoring camera state tag={tag} (via CAMERA_WORKER)'
            )
            fut = camera_executor.protocol_put(
                IOTask(
                    action=scope.imaging.restore_camera_state,
                    args=(saved_camera_state,),
                ),
                return_future=True,
            )
            if fut is not None:
                fut.result(timeout=30)
            else:
                # Executor disabled / protocol already ended — fall back to
                # a direct call so state is still restored. Real-hardware
                # path normally hits the executor branch above; this branch
                # mostly covers tests / shutdown races.
                scope.imaging.restore_camera_state(saved_camera_state)
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error restoring camera gain/exposure during cleanup: {ex}')
        cleanup_errors.append(f'Restore camera gain/exposure: {type(ex).__name__}: {ex}')

    # --- Complete protocol execution record ---
    try:
        if not disable_saving_artifacts and protocol_execution_record is not None:
            file_io_executor.protocol_put(IOTask(action=protocol_execution_record.complete))
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error completing protocol record during cleanup: {ex}')
        cleanup_errors.append(f'Complete protocol record: {type(ex).__name__}: {ex}')

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
    except Exception as ex:
        logger.error(f'[PROTOCOL] Error returning to position during cleanup: {ex}')
        cleanup_errors.append(f'Return to position: {type(ex).__name__}: {ex}')

    # --- End executors ---
    scan_in_progress.clear()

    io_executor.protocol_end()
    if autofocus_thread is not None:
        # Signal any lingering AF run to unwind. abort() is a no-op when
        # the thread is idle, so this is always safe to call.
        autofocus_thread.abort()
    camera_executor.enable()
    logger.info(f'[{logger_name}] Cleanup: protocol_end called on all executors')

    io_executor.clear_protocol_pending()
    if is_aborted:
        # PF-2: drop pending writes on abort. Drain (the COMPLETING-path default)
        # would write everything queued to disk before releasing memory — fine on
        # normal completion, but on disconnect/error the user wants control back
        # without waiting for many GB of frames to slowly drain.
        file_io_executor.clear_protocol_pending()
        logger.info(f'[{logger_name}] Cleanup: file_io_executor pending cleared (aborted)')

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
            notifications.warning(
                'Protocol',
                'Protocol cleanup issues',
                f'Protocol completed but {len(cleanup_errors)} cleanup step(s) failed:\n'
                f'{err_summary}\n'
                f'Check LED state, camera settings, and stage position.',
            )
        except Exception as ex:
            # Best-effort -- a notification failure during cleanup must
            # not prevent the completion callbacks from firing.
            logger.error(f'[PROTOCOL] Failed to surface cleanup-error notification: {ex}')

    # --- Fire completion callbacks ---
    _file_queue_active = file_io_executor.is_protocol_queue_active()
    logger.info(f'[{logger_name}] Cleanup: file queue active={_file_queue_active}')
    if _file_queue_active:
        if callbacks.run_complete:
            _schedule_ui(lambda dt: callbacks.run_complete(protocol=protocol), 0)
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
            _schedule_ui(lambda dt: callbacks.run_complete(protocol=protocol), 0)
        if callbacks.files_complete:
            _schedule_ui(lambda dt: callbacks.files_complete(protocol=protocol), 0)
        file_io_executor.protocol_finish_then_end()
        logger.info(
            f'[{logger_name}] Cleanup: callbacks scheduled (run_complete + files_complete immediate)'
        )
