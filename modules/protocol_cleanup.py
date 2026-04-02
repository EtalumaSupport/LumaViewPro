# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Protocol cleanup / shutdown logic.

Restores LED, autofocus, camera state and fires completion callbacks.
Extracted from ``sequenced_capture_executor.py`` during the
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
    protocol_ended: threading.Event,
    scan_in_progress: threading.Event,
    # Saved original states
    leds_state_at_end: str,
    original_led_states: dict,
    original_autofocus_states: dict,
    original_gain: float,
    original_exposure: float,
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
    protocol_executor,
    autofocus_io_executor,
    file_io_executor,
    camera_executor,
    # Mutable flag — set to False when done
    set_run_in_progress_fn,
    logger_name: str = "SequencedCaptureExecutor",
):
    """Core cleanup logic — restores state, fires callbacks, ends executors.

    Called from ``SequencedCaptureExecutor._cleanup_inner()``.
    """
    # Transition to COMPLETING (or stay in ERROR if that's how we got here)
    if get_state_fn() not in (ProtocolState.COMPLETING, ProtocolState.ERROR, ProtocolState.IDLE):
        set_state_fn(ProtocolState.COMPLETING)

    # Signal the scan/protocol loops to stop BEFORE turning off LEDs.
    protocol_ended.set()
    scan_in_progress.clear()

    try:
        cancel_scheduled_events_fn()
    except Exception as ex:
        logger.error(f"[PROTOCOL] Error cancelling scheduled events during cleanup: {ex}")

    # --- Restore LEDs ---
    try:
        if leds_state_at_end == "off":
            leds_off_fn()
        elif leds_state_at_end == "return_to_original":
            any_restored = False
            for color, color_data in original_led_states.items():
                if color_data['enabled']:
                    led_on_fn(color=color, illumination=color_data['illumination'], block=True, force=True)
                    any_restored = True
            if not any_restored:
                leds_off_fn()
        else:
            logger.error(f"Unsupported LEDs state at end value: {leds_state_at_end}")
    except Exception as ex:
        logger.error(f"[PROTOCOL] Error restoring LED states during cleanup: {ex}")
    logger.info(f"[{logger_name}] Cleanup: LED/camera restore complete")

    # --- Restore autofocus states ---
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
                        settings[layer]["autofocus"] = layer_data
                else:
                    settings[layer]["autofocus"] = layer_data
            if callbacks.reset_autofocus_btns:
                _schedule_ui(lambda dt: callbacks.reset_autofocus_btns(), 0)
    except Exception as ex:
        logger.error(f"[PROTOCOL] Error restoring autofocus states during cleanup: {ex}")

    # --- Restore camera gain and exposure ---
    try:
        if original_gain >= 0:
            scope.set_gain(original_gain)
        if original_exposure > 0:
            scope.set_exposure_time(original_exposure)
    except Exception as ex:
        logger.error(f"[PROTOCOL] Error restoring camera gain/exposure during cleanup: {ex}")

    # --- Complete protocol execution record ---
    try:
        if not disable_saving_artifacts and protocol_execution_record is not None:
            file_io_executor.protocol_put(IOTask(
                action=protocol_execution_record.complete
            ))
    except Exception as ex:
        logger.error(f"[PROTOCOL] Error completing protocol record during cleanup: {ex}")

    # --- Return to position ---
    try:
        if return_to_position is not None:
            logger.info(
                f"[{logger_name}] Cleanup: returning to position "
                f"x={return_to_position['x']}, y={return_to_position['y']}, z={return_to_position['z']}"
            )
            default_move_fn(
                px=return_to_position['x'],
                py=return_to_position['y'],
                z=return_to_position['z'],
            )
            logger.info(f"[{logger_name}] Cleanup: return-to-position move issued")
    except Exception as ex:
        logger.error(f"[PROTOCOL] Error returning to position during cleanup: {ex}")

    # --- End executors ---
    scan_in_progress.clear()
    protocol_ended.set()

    io_executor.protocol_end()
    protocol_executor.protocol_end()
    autofocus_io_executor.protocol_end()
    camera_executor.enable()
    logger.info(f"[{logger_name}] Cleanup: protocol_end called on all executors")

    io_executor.clear_protocol_pending()
    protocol_executor.clear_protocol_pending()

    with run_lock:
        set_run_in_progress_fn(False)
        # Transition back to IDLE from COMPLETING or ERROR
        if get_state_fn() in (ProtocolState.COMPLETING, ProtocolState.ERROR):
            set_state_fn(ProtocolState.IDLE)

    # --- Fire completion callbacks ---
    _file_queue_active = file_io_executor.is_protocol_queue_active()
    logger.info(f"[{logger_name}] Cleanup: file queue active={_file_queue_active}")
    if _file_queue_active:
        if callbacks.run_complete:
            _schedule_ui(lambda dt: callbacks.run_complete(protocol=protocol), 0)
        if callbacks.files_complete:
            file_io_executor.set_protocol_complete_callback(
                callback=lambda: _schedule_ui(lambda dt: callbacks.files_complete(protocol=protocol), 0)
            )
        file_io_executor.protocol_finish_then_end()
        logger.info(f"[{logger_name}] Cleanup: callbacks scheduled (run_complete now, files_complete deferred)")
    else:
        if callbacks.run_complete:
            _schedule_ui(lambda dt: callbacks.run_complete(protocol=protocol), 0)
        if callbacks.files_complete:
            _schedule_ui(lambda dt: callbacks.files_complete(protocol=protocol), 0)
        file_io_executor.protocol_finish_then_end()
        logger.info(f"[{logger_name}] Cleanup: callbacks scheduled (run_complete + files_complete immediate)")
