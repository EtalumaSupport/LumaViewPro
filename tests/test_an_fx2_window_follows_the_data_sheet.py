# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2's window is placed and read out as the MT9P031 documents give.

Before, Column_Start was the centred value rounded to even, which lands on
4n at about half the widths (360 at 1900, 1060 at 500), where the register
reference requires 4n + 2 under Mirror_Column, which the driver sets. And the
simulated FX2 sent zeros where the sensor sends its first and last output
rows: the LS620's test patterns show both carry sensor data.

The expected values are the documents' and the bench's, written as literals.
"""

from __future__ import annotations

import pytest

from drivers import fx2driver
from drivers.fx2driver import FX2Camera

# DS: the active array is 2592 columns from column 16, so its centre is
# column 16 + 2592 / 2.
ARRAY_CENTRE = 1312


@pytest.fixture
def sim():
    from drivers.simulated_fx2 import SimulatedFX2

    return SimulatedFX2()


@pytest.fixture
def camera(sim):
    cam = FX2Camera(connection=sim.connection)
    yield cam
    cam.disconnect()


def test_every_window_starts_on_a_mirrored_column_and_is_centred(sim, camera):
    registers = sim.device.sensor.registers
    for w in range(100, 1901, 4):
        camera.set_frame_size(w, 100)
        start = registers[fx2driver.REG_COL_START]
        # RR R0x02: Column_Start is 4n + 2 under Mirror_Column.
        assert start % 4 == 2, (w, start)
        # DS Table 8: the sensor outputs W = Column_Size + 1 = w + 2 columns.
        assert abs(start + (w + 2) / 2 - ARRAY_CENTRE) <= 2, (w, start)


@pytest.mark.parametrize(
    'w, start', [(1900, 362), (1000, 810), (500, 1062), (1896, 362), (1880, 370)]
)
def test_the_column_start_at_the_benched_widths(sim, camera, w, start):
    camera.set_frame_size(w, w)
    assert sim.device.sensor.registers[fx2driver.REG_COL_START] == start


def test_the_simulated_frame_carries_sensor_rows_where_the_parser_stores_none(sim):
    device = sim.device
    device.sensor.write(bytes([fx2driver.REG_COL_SIZE, 0, 101]))
    device.sensor.write(bytes([fx2driver.REG_ROW_SIZE, 0, 81]))
    device.leds.brightness[ord('A')] = 255
    layout = fx2driver.frame_layout(100, 80)
    body = device.frame()[len(fx2driver.FRAME_DELIM) :]
    first, last = body[: layout.skip], body[layout.needed :]
    assert any(first[:100]), 'the first output row is sensor data, not zeros'
    assert any(last[:100]), 'the last output row is sensor data, not zeros'
    stored = [
        body[layout.skip + r * layout.stride : layout.skip + (r + 1) * layout.stride]
        for r in range(80)
    ]
    assert all(row[100] == 0 for row in stored), 'each row ends in the 0 sync byte'
