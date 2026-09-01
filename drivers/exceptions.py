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
