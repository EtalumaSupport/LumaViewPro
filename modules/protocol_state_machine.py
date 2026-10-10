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
    # One autofocus at the stage's position: the Autofocus button's run.
    SINGLE_AUTOFOCUS = 'single_autofocus'
    # An autofocus at every step of a protocol, writing each focus back
    # into it.
    SINGLE_AUTOFOCUS_SCAN = 'single_autofocus_scan'
    SINGLE_COMPOSITE = 'single_composite'

    @property
    def is_autofocus(self) -> bool:
        """Whether the run only focuses: it saves no images and builds no stack."""
        return self in (
            SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
            SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
        )

    @property
    def is_one_position(self) -> bool:
        """Whether the run acts at the stage's one position.

        One autofocus, one composite, one z-stack: a few seconds to minutes
        at a scope someone is standing at, as against a scan or protocol
        that traverses the plate and may end with nobody there.
        """
        return self in (
            SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
            SequencedCaptureRunMode.SINGLE_COMPOSITE,
            SequencedCaptureRunMode.SINGLE_ZSTACK,
        )

    @property
    def leds_state_at_end(self) -> str:
        """How the run leaves the LEDs: 'return_to_original' or 'off'.

        A one-position run hands the illumination back the way it found it;
        a plate traverse forces every channel dark, since it may end with
        nobody there to turn them off.
        """
        return 'return_to_original' if self.is_one_position else 'off'

    @property
    def words(self) -> str:
        """The run's kind as a person reads it mid-sentence ('the Z-stack run')."""
        return {
            SequencedCaptureRunMode.FULL_PROTOCOL: 'protocol',
            SequencedCaptureRunMode.SINGLE_SCAN: 'scan',
            SequencedCaptureRunMode.SINGLE_ZSTACK: 'Z-stack',
            SequencedCaptureRunMode.SINGLE_AUTOFOCUS: 'autofocus',
            SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN: 'autofocus scan',
            SequencedCaptureRunMode.SINGLE_COMPOSITE: 'composite',
        }[self]


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
