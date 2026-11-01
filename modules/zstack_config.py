# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import enum

import numpy as np

import modules.common_utils as common_utils
from modules.exceptions import ConfigurationDefectError


class ZStackReference(enum.StrEnum):
    """Where a stack sits against the Z it is built around: its top, its
    centre or its bottom at that Z. The stored ``zstack.position`` holds the
    value; the GUI shows its own label for each."""

    TOP = 'top'
    CENTER = 'center'
    BOTTOM = 'bottom'


class ZStackConfig:
    def __init__(
        self,
        range: float,
        step_size: float,
        current_z_reference: ZStackReference,
        current_z_value: float | None = None,
    ):
        self._range = range
        self._step_size = step_size
        self._current_z_reference = current_z_reference
        self._current_z_value = current_z_value

    def number_of_steps(self) -> int:
        # Both axes must be positive for the store to describe a stack.
        # Guarding step_size alone let range=0 compute floor(0/step) + 1 == 1,
        # so a store with no extent answered "one plane" and every gate that
        # asks this count read a zero-extent stack as a configured one.
        if self._step_size <= 0 or self._range <= 0:
            return 0

        # int, not the np.float64 the arithmetic yields: callers render this
        # straight into the Steps field, where a float shows as "11.0".
        return int(np.floor(self._range / self._step_size) + 1)

    def reference_slice(self) -> int:
        """The index of the slice built at the Z the stack is built around.

        The top slice for a top stack, the bottom for a bottom stack, the
        middle for a centre stack, and the lower of the two middles when a
        centre stack has an even count, since no slice is at that Z then.
        """
        n_steps = self.number_of_steps()
        if self._current_z_reference == ZStackReference.TOP:
            return n_steps - 1
        if self._current_z_reference == ZStackReference.CENTER:
            return (n_steps - 1) // 2
        return 0

    def step_positions(self) -> dict[int, float]:
        """Each slice's Z, keyed by slice index, lowest first.

        The slices are laid over the span they cover, (n - 1) x step, and
        that span is placed against the Z the stack is built around: a top
        stack ends at it, a bottom stack starts at it, a centre stack is
        centred on it. Placing the requested range instead puts no slice at
        that Z whenever the range is not a whole number of steps (599 / 200
        centred gives -299.5, -99.5, +100.5), and that Z is the focus the
        stack's reference slice must hold.
        """
        n_steps = self.number_of_steps()
        span = (n_steps - 1) * self._step_size

        if self._current_z_reference == ZStackReference.TOP:
            start_pos = self._current_z_value - span
        elif self._current_z_reference == ZStackReference.CENTER:
            start_pos = self._current_z_value - span / 2
        elif self._current_z_reference == ZStackReference.BOTTOM:
            start_pos = self._current_z_value
        else:
            # Without this branch an unmapped reference -- or the None a
            # settings dict with no stored position yields -- leaves start_pos
            # unbound and throws UnboundLocalError on the next line, naming a
            # local variable instead of the bad reference that caused it.
            raise ConfigurationDefectError(
                'zstack_reference_unchecked',
                f'Unknown Z-stack position reference: {self._current_z_reference!r}',
            )

        position_values = (np.arange(n_steps) * self._step_size + start_pos).tolist()
        max_precision = common_utils.max_decimal_precision(parameter='z')
        position_values = [round(val, max_precision) for val in position_values]
        return dict(enumerate(position_values))
