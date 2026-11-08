# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A Z-stack whose range is a whole number of steps has a slice at each end.

``ZStackConfig.number_of_steps()`` is the one count the manual stack, every
protocol step's stack and REST share: the range divided by the step, rounded
down, plus one. Divided in floating point, 0.3 / 0.1 is 2.9999999999999996,
so a typed pair that is a whole number of steps lost its top slice and the
stack stopped short of the range asked for.
"""

from _lumascope.modules.zstack_config import ZStackConfig, ZStackReference

# (range um, step um, slices): whole numbers of steps that float down, two
# that float up or land exactly, and two that are not whole and round down.
_PAIRS = [
    (0.3, 0.1, 4),
    (0.7, 0.1, 8),
    (1.9, 0.1, 20),
    (0.6, 0.2, 4),
    (2.3, 0.1, 24),
    (100.0, 10.0, 11),
    (0.35, 0.1, 4),
    (599.0, 200.0, 3),
]


def test_a_whole_number_of_steps_ends_on_the_range():
    wrong = []
    for range_um, step_um, slices in _PAIRS:
        config = ZStackConfig(
            range=range_um,
            step_size=step_um,
            current_z_reference=ZStackReference.BOTTOM,
            current_z_value=1000.0,
        )
        count = config.number_of_steps()
        positions = config.step_positions()
        if count != slices or len(positions) != slices:
            wrong.append((range_um, step_um, slices, count, len(positions)))
    assert wrong == [], f'(range, step, slices wanted, count, positions): {wrong}'
