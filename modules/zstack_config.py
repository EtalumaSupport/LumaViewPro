
import numpy as np

import modules.common_utils as common_utils


class ZStackConfig:

    def __init__(
            self,
            range: float,
            step_size: float,
            current_z_reference: str,
            current_z_value: float | None = None
    ):
        self._range = range
        self._step_size = step_size
        self._current_z_reference = current_z_reference
        self._current_z_value = current_z_value


    def number_of_steps(self) -> int:
        if self._step_size == 0:
            return 0
        
        return np.floor(self._range/self._step_size)+1
    

    def step_positions(self) -> dict[int, float]:
        n_steps = self.number_of_steps()

        if self._current_z_reference == 'top':
            start_pos = self._current_z_value - self._range
        elif self._current_z_reference == 'center':
            start_pos = self._current_z_value - self._range/2
        elif self._current_z_reference == 'bottom':
            start_pos = self._current_z_value

        position_values = (np.arange(n_steps)*self._step_size + start_pos).tolist()
        max_precision = common_utils.max_decimal_precision(parameter='z')
        position_values = [round(val, max_precision) for val in position_values]
        return {index: value for index, value in enumerate(position_values)}

