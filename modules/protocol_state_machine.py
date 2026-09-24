# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Protocol execution state machine.

Pure logic -- no threading, no I/O.  Extracted from
``sequenced_capture_runner.py`` during the protocol-decomposition refactor.
"""

import enum

from lvp_logger import logger


class SequencedCaptureRunMode(enum.Enum):
    FULL_PROTOCOL = 'full_protocol'
    SINGLE_SCAN = 'single_scan'
    SINGLE_ZSTACK = 'single_zstack'
    SINGLE_AUTOFOCUS_SCAN = 'single_autofocus_scan'
    SINGLE_COMPOSITE = 'single_composite'


class ProtocolState(enum.Enum):
    """Protocol execution state machine.

    Valid transitions:
        IDLE     -> RUNNING              (run() called)
        RUNNING  -> SCANNING             (scan started)
        RUNNING  -> COMPLETING           (all scans done, cleanup starting)
        RUNNING  -> ERROR                (unrecoverable error)
        SCANNING -> RUNNING              (scan finished, back to inter-scan wait)
        SCANNING -> COMPLETING           (abort/error during scan)
        SCANNING -> ERROR                (unrecoverable error during scan)
        COMPLETING -> IDLE               (cleanup finished)
        ERROR    -> IDLE                 (cleanup finished after error)
        RUNNING  -> IDLE                 (cleanup finished from any phase)
        SCANNING -> IDLE                 (cleanup finished from any phase)

    IDLE is reachable from every state because a run that has stopped is
    idle whatever phase it stopped in: cleanup restores it in a finally
    that runs on every path out of a run, including one that raised
    before the phase ever reached COMPLETING. A table that could not
    express that would make the restore itself raise, inside the block
    whose whole purpose is to run anyway.
    """

    IDLE = 'idle'
    RUNNING = 'running'
    SCANNING = 'scanning'
    COMPLETING = 'completing'
    ERROR = 'error'


# Allowed state transitions: {from_state: {set of valid to_states}}
PROTOCOL_STATE_TRANSITIONS: dict[ProtocolState, set[ProtocolState]] = {
    ProtocolState.IDLE: {ProtocolState.RUNNING},
    ProtocolState.RUNNING: {
        ProtocolState.SCANNING,
        ProtocolState.COMPLETING,
        ProtocolState.ERROR,
        ProtocolState.IDLE,
    },
    ProtocolState.SCANNING: {
        ProtocolState.RUNNING,
        ProtocolState.COMPLETING,
        ProtocolState.ERROR,
        ProtocolState.IDLE,
    },
    ProtocolState.COMPLETING: {ProtocolState.IDLE},
    ProtocolState.ERROR: {ProtocolState.IDLE},
}


def validate_transition(
    old_state: ProtocolState,
    new_state: ProtocolState,
    logger_name: str = 'SequencedCaptureRunner',
) -> None:
    """Raise ``ValueError`` if *old_state* -> *new_state* is not allowed."""
    if old_state == new_state:
        return  # no-op
    allowed = PROTOCOL_STATE_TRANSITIONS.get(old_state, set())
    if new_state not in allowed:
        msg = (
            f'[{logger_name}] Invalid state transition: '
            f'{old_state.value} -> {new_state.value} '
            f'(allowed: {", ".join(s.value for s in allowed)})'
        )
        logger.error(msg)
        raise ValueError(msg)
    logger.debug(f'[{logger_name}] State: {old_state.value} -> {new_state.value}')
