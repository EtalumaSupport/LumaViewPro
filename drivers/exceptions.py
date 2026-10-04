# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Driver-layer exception classes (raised from drivers/, caught at module/API)."""


class HardwareError(Exception):
    """Hardware communication or configuration failure (motor, LED, camera)."""

    pass


class ConfigReadError(HardwareError):
    """The board's per-unit config could not be READ (no answer, or an
    unparseable payload).

    Distinct from a board that answers with an empty or minimal config:
    after a failed read the per-unit values may exist on the board but
    are unavailable, so consumers that would trust "no per-unit value
    present" need to know the difference or they silently serve another
    source's answer for a unit that has its own.
    """

    pass


class MotionInterlockError(HardwareError):
    """The board refused a motion command because a hardware interlock is open.

    Raised by a driver whose board guards motion with its own inputs (a
    lid switch, the stage's power), read on the wire with the command they
    guard. The API turns it into its refusal or its failed home, so it
    carries what the API needs to say which one happened: whether any part
    of the refused command had already moved, and whether the driver
    stopped the motors in refusing it.

    Attributes:
        reason: ``'lid_open'`` or ``'stage_unpowered'``.
        moved: Whether anything of the refused command had moved.
        stopped: Whether the driver stopped every motor in refusing it.
    """

    def __init__(self, reason: str, *, moved: bool, stopped: bool):
        super().__init__(f'motion interlock open: {reason} (moved={moved}, stopped={stopped})')
        self.reason = reason
        self.moved = moved
        self.stopped = stopped
