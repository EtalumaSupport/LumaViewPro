# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import numpy as np

import modules.common_utils as common_utils
from modules.exceptions import ConfigError


class ZStackConfig:
    def __init__(
        self,
        range: float,
        step_size: float,
        current_z_reference: str,
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

    def step_positions(self) -> dict[int, float]:
        n_steps = self.number_of_steps()

        if self._current_z_reference == 'top':
            start_pos = self._current_z_value - self._range
        elif self._current_z_reference == 'center':
            start_pos = self._current_z_value - self._range / 2
        elif self._current_z_reference == 'bottom':
            start_pos = self._current_z_value
        else:
            # Without this branch an unmapped reference -- or the None a
            # settings dict with no stored position yields -- leaves start_pos
            # unbound and throws UnboundLocalError on the next line, naming a
            # local variable instead of the bad reference that caused it.
            raise ConfigError(f'Unknown Z-stack position reference: {self._current_z_reference!r}')

        position_values = (np.arange(n_steps) * self._step_size + start_pos).tolist()
        max_precision = common_utils.max_decimal_precision(parameter='z')
        position_values = [round(val, max_precision) for val in position_values]
        return dict(enumerate(position_values))
