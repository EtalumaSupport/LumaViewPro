# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2's gain and its bring-up are the ones the MT9P031 documents give.

Before, the gain encoder put a 4.0-4.125x request on analog 16 with the
multiplier, outside DS Table 15's 17-32, and a 42.1 dB clamp stopped it at
127x while the camera published 42.1 dB, a value it could not produce. Connect
wrote the PLL with the sensor streaming, which DS p22 calls undefined, never
reset the sensor, so whatever an earlier writer left stayed, and wrote R0x62
to 0x6000 with no reason on record.

The expected values are the documents', written as literals.
"""

from __future__ import annotations

import pytest

from drivers import fx2driver
from drivers.fx2driver import FX2Camera

# DS Table 15: analog 8-32 with no multiplier (1-4x), analog 17-32 with the
# multiplier (4.25-8x), analog 32 with the multiplier and digital 1-120
# (9-128x). Register = Digital_Gain << 8 | Analog_Multiplier << 6 | Analog_Gain.
TABLE_15 = [
    *range(8, 33),
    *(0x40 | analog for analog in range(17, 33)),
    *((digital << 8) | 0x40 | 32 for digital in range(1, 121)),
]
# 20 x log10(128).
MAX_DB = 42.1442

# DS p21 (soft reset), p22 (the PLL), p23 (entering and leaving soft standby),
# in LumaView Classic's order: Power_PLL, standby, M/N and P1, out of
# standby, Power_PLL again (standby powered it down), Use_PLL.
BRING_UP = [
    (0x0D, 0x0001),
    (0x0D, 0x0000),
    (0x10, 0x0051),
    (0x0B, 0x0002),
    (0x0B, 0x0003),
    (0x07, 0x1F82),
    (0x07, 0x1F80),
    (0x0B, 0x0001),
    (0x11, 0x1B01),
    (0x12, 0x000D),
    (0x0B, 0x0002),
    (0x0B, 0x0003),
    (0x07, 0x1F80),
    (0x07, 0x1F82),
    (0x0B, 0x0001),
    (0x10, 0x0051),
    (0x10, 0x0053),
]


def _db(reg: int) -> float:
    return fx2driver._register_to_gain_db(reg)[1]


@pytest.fixture
def sim():
    from drivers.simulated_fx2 import SimulatedFX2

    return SimulatedFX2()


@pytest.fixture
def writes(sim, monkeypatch):
    """Every sensor register write, in order, from the camera's connect on."""
    seen = []
    write = sim.connection.sensor_reg_write

    def record(reg, value):
        seen.append((reg, value))
        return write(reg, value)

    monkeypatch.setattr(sim.connection, 'sensor_reg_write', record)
    return seen


@pytest.fixture
def camera(sim, writes):
    cam = FX2Camera(connection=sim.connection)
    yield cam
    cam.disconnect()


@pytest.mark.parametrize('reg', TABLE_15)
def test_every_table_15_setting_encodes_as_itself(reg):
    assert fx2driver._gain_db_to_register(_db(reg)) == reg


@pytest.mark.parametrize('db', [x / 10 for x in range(-50, 500, 3)])
def test_every_request_encodes_as_the_nearest_table_15_setting(db):
    reg = fx2driver._gain_db_to_register(db)
    assert reg in TABLE_15
    assert abs(_db(reg) - db) == min(abs(_db(r) - db) for r in TABLE_15)


@pytest.mark.parametrize('multiple', [4.0, 4.05, 4.1])
def test_four_times_is_analog_32_without_the_multiplier(multiple):
    import math

    assert fx2driver._gain_db_to_register(20 * math.log10(multiple)) == 0x0020


def test_the_largest_gain_is_128x():
    assert fx2driver._gain_db_to_register(60.0) == 0x7860
    assert _db(0x7860) == pytest.approx(MAX_DB, abs=1e-4)


def test_the_camera_publishes_the_gain_it_can_reach(camera):
    assert camera.profile.gain.total_max_db == pytest.approx(MAX_DB, abs=1e-4)
    assert camera.gain(60.0) == pytest.approx(MAX_DB, abs=1e-4)


def test_connect_resets_the_sensor_then_sets_the_pll_in_standby(writes, camera):
    assert writes[: len(BRING_UP)] == BRING_UP


def test_connect_writes_the_documents_registers_and_leaves_r0x62_at_its_default(writes, camera):
    after = dict(writes[len(BRING_UP) :])
    assert after[0x7F] == 0x0000  # DG p7
    assert after[0x49] == 0x0000  # the black target, kept at 0 by ruling
    assert after[0x20] == 0x0040  # Row_BLC; Mirror_Column clear
    assert 0x62 not in dict(writes)


def test_the_simulated_sensor_soft_resets_to_its_power_on_window(sim):
    sensor = sim.device.sensor
    sensor.write(bytes([0x04, 0x01, 0xF5]))
    sensor.write(bytes([0x11, 0x1B, 0x01]))
    sensor.write(bytes([0x0D, 0x00, 0x01]))
    assert sensor.registers[0x04] == 0x0A1F
    assert sensor.registers[0x11] == 0x1B01  # DS p21: a soft reset keeps the PLL fields
