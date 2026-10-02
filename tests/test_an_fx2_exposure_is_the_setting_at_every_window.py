# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 exposure integrates the time it was set to, at every window.

Before, the driver turned every exposure into shutter rows with one row time,
0.1124 ms, from a 12 MHz clock model the LS620 bench refuted: the MT9P031's
row time follows the window's width (DS Table 8), and its clock is 23.14 MHz
from a 24 MHz EXTCLK. A 50 ms request integrated about 53.9 ms at 1900 wide,
36.6 ms at 1000 and 27.0 ms at 500, a window change kept the rows, and every
file recorded 50.

The expected values here are the data sheet's, computed by hand at
f_PIXCLK = 24 MHz x 27 / (2 x 14) and written as literals, so an error the
driver and the simulator share cannot pass by agreeing with itself.
"""

from __future__ import annotations

import pytest

from drivers import fx2driver
from drivers.fx2driver import FX2Camera

# tROW = 2 x tPIXCLK x max(W/2 + 450, 486), W = Column_Size + 1 = w + 4 (DS Table 8), in us.
ROW_US = {1900: 121.160, 1000: 82.272, 500: 60.667, 100: 43.383}
# SO x 2 x tPIXCLK = 426 pixel clocks (DS p31), in us.
OVERHEAD_US = 18.407


def _integration_ms(shutter_width: int, w: int) -> float:
    """tEXP = SW x tROW - SO x 2 x tPIXCLK at a ``w``-wide window, from the literals."""
    return (shutter_width * ROW_US[w] - OVERHEAD_US) / 1000.0


@pytest.fixture
def sim():
    from drivers.simulated_fx2 import SimulatedFX2

    return SimulatedFX2()


@pytest.fixture
def camera(sim):
    cam = FX2Camera(connection=sim.connection)
    yield cam
    cam.disconnect()


@pytest.mark.parametrize('w', sorted(ROW_US))
def test_the_row_time_is_the_data_sheets(w):
    assert fx2driver.row_time_s(w + 3) * 1e6 == pytest.approx(ROW_US[w], abs=0.001)


@pytest.mark.parametrize('w', sorted(ROW_US))
def test_one_shutter_row_integrates_the_row_less_the_overhead(w):
    assert fx2driver.exposure_s(1, w + 3) * 1e6 == pytest.approx(ROW_US[w] - OVERHEAD_US, abs=0.001)


@pytest.mark.parametrize('w', sorted(ROW_US))
@pytest.mark.parametrize('request_ms', [0.5, 50.0, 178.0])
def test_a_request_lands_within_half_a_row_of_itself(w, request_ms):
    rows = fx2driver.shutter_width_for(request_ms / 1000.0, w + 3)
    assert abs(_integration_ms(rows, w) - request_ms) <= ROW_US[w] / 2000.0 + 1e-6


@pytest.mark.parametrize('w, frame_ms', [(1900, 233.60), (1000, 84.58)])
def test_the_frame_time_is_the_data_sheets(w, frame_ms):
    # tFRAME = (H + VB) x tROW, H = h + 2, VB = 26 rows at its default.
    assert fx2driver.frame_time_s(w + 3, w + 1, 1) * 1000 == pytest.approx(frame_ms, abs=0.01)


def test_a_shutter_past_the_frame_stretches_it():
    # VBMIN = max(8, SW - H) + 1 takes over from VB once SW > H + 25:
    # the frame is SW + 1 rows.
    assert fx2driver.frame_time_s(1003, 1001, 2000) * 1e6 == pytest.approx(
        2001 * ROW_US[1000], rel=1e-5
    )


def test_the_sensor_runs_the_pixel_clock_the_model_reads(sim, camera):
    # connect() writes the PLL from the fields the model reads: M = 27,
    # N_divider = 1, P1_divider = 13, so 24 MHz x 27 / (2 x 14).
    registers = sim.device.sensor.registers
    m, n_divider = registers[fx2driver.REG_PLL_CFG1] >> 8, registers[fx2driver.REG_PLL_CFG1] & 0x3F
    p1_divider = registers[fx2driver.REG_PLL_CFG2] & 0x1F
    assert fx2driver.pixel_clock_hz(m, n_divider, p1_divider) == pytest.approx(
        23_142_857.14, abs=0.01
    )


def test_a_refused_shutter_write_keeps_the_exposure_in_effect(sim, camera, monkeypatch):
    camera.exposure_t(50.0)
    write = sim.connection.sensor_reg_write

    def refuse_the_shutter(reg, value):
        if reg == fx2driver.REG_EXPOSURE:
            raise OSError('usb gone')
        return write(reg, value)

    monkeypatch.setattr(sim.connection, 'sensor_reg_write', refuse_the_shutter)
    with pytest.raises(OSError):
        camera.exposure_t(100.0)
    rows = sim.device.sensor.registers[fx2driver.REG_EXPOSURE]
    assert camera.get_exposure_t() == pytest.approx(_integration_ms(rows, 1900), abs=1e-3)


def test_the_published_minimum_is_one_row_at_the_full_window(camera):
    assert camera.profile.exposure_min_us == pytest.approx(ROW_US[1900] - OVERHEAD_US, abs=0.001)


def test_a_long_exposure_integrates_the_setting_at_the_full_window(sim, camera):
    # At 2 s the rows a Column_Size of w + 1 would give integrate 1.4 ms too
    # long at the w + 3 the driver writes, far past half a row; at 50 ms the
    # two round to the same row count.
    # The table's row times are to 1 ns, which 16507 rows carry to about 10 us.
    applied_us = camera.exposure_t(2000.0)
    rows = sim.device.sensor.registers[fx2driver.REG_EXPOSURE]
    assert _integration_ms(rows, 1900) == pytest.approx(2000.0, abs=ROW_US[1900] / 2000.0)
    assert applied_us / 1000.0 == pytest.approx(2000.0, abs=ROW_US[1900] / 2000.0)


def test_a_50ms_exposure_integrates_50ms_at_the_full_window(sim, camera):
    applied_us = camera.exposure_t(50.0)
    rows = sim.device.sensor.registers[fx2driver.REG_EXPOSURE]
    assert _integration_ms(rows, 1900) == pytest.approx(50.0, abs=ROW_US[1900] / 2000.0)
    assert applied_us / 1000.0 == pytest.approx(_integration_ms(rows, 1900), abs=1e-3)


@pytest.mark.parametrize('w', [1000, 500, 100])
def test_a_window_change_keeps_the_exposure_the_setting(sim, camera, w):
    camera.exposure_t(50.0)
    assert camera.set_frame_size(w, w) == {'width': w, 'height': w}
    rows = sim.device.sensor.registers[fx2driver.REG_EXPOSURE]
    # The sensor integrates the setting at the new window, and the driver
    # reports what the sensor integrates.
    assert _integration_ms(rows, w) == pytest.approx(50.0, abs=ROW_US[w] / 2000.0)
    assert camera.get_exposure_t() == pytest.approx(_integration_ms(rows, w), abs=1e-3)


def test_the_simulated_sensor_integrates_what_the_driver_reports(sim, camera):
    camera.exposure_t(50.0)
    camera.set_frame_size(500, 500)
    assert sim.device.sensor.exposure_s() * 1000 == pytest.approx(camera.get_exposure_t(), abs=1e-6)


@pytest.mark.parametrize('w, frame_ms', [(1900, 233.60), (1000, 84.58)])
def test_the_simulated_frame_period_is_the_data_sheets(sim, camera, w, frame_ms):
    camera.exposure_t(50.0)
    camera.set_frame_size(w, w)
    assert sim.device.sensor.frame_period_s() * 1000 == pytest.approx(frame_ms, abs=0.01)
