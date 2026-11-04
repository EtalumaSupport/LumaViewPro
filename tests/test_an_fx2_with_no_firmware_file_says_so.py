# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 whose firmware file is missing from the installation says so.

The FX2 runs the firmware the driver uploads at every connect. With the hex
file missing, the lookup raised a FileNotFoundError, which bring-up
classified as an absent camera: "Camera not found. Check USB cable and
power.", on a scope whose cable was fine and whose installation was not.
The lookup now raises the camera's firmware-missing fault, and bring-up
names it.

The simulated FX2 is the bus: it answers in its bootloader, so the driver
reaches the upload and looks for the file.
"""

import pytest

from drivers import fx2driver
from drivers.exceptions import CameraFirmwareMissingError
from drivers.registry import DriverRegistry
from drivers.simulated_fx2 import SimulatedFX2Device, SimulatedFX2Transport
from modules.lumascope_api._lumascope import _camera_failure_cause


@pytest.fixture
def no_firmware_file(monkeypatch, tmp_path):
    """The driver installed in a tree that holds no firmware folder."""
    (tmp_path / 'drivers').mkdir()
    monkeypatch.setattr(fx2driver, '__file__', str(tmp_path / 'drivers' / 'fx2driver.py'))
    monkeypatch.delattr('sys.frozen', raising=False)


@pytest.fixture
def simulated_bus(monkeypatch):
    fx2driver._FX2Connection._reset_for_test()
    monkeypatch.setattr(
        fx2driver, '_platform_transport', lambda: SimulatedFX2Transport(SimulatedFX2Device())
    )
    yield
    fx2driver._FX2Connection._reset_for_test()


class _AbsentCamera:
    """Another camera driver, its camera not attached."""

    found = False


def test_the_connect_raises_the_firmware_fault_naming_the_file(no_firmware_file):
    with pytest.raises(CameraFirmwareMissingError, match=r'LumascopeClassic\.hex'):
        fx2driver._FX2Connection(SimulatedFX2Transport(SimulatedFX2Device()))


def test_bring_up_s_camera_cause_is_the_missing_firmware(no_firmware_file, simulated_bus):
    registry = DriverRegistry('camera')
    registry.register('pylon', priority=100)(_AbsentCamera)
    registry.register('fx2', priority=80)(fx2driver.FX2Camera)

    with pytest.raises(CameraFirmwareMissingError) as caught:
        registry.create('auto')

    assert _camera_failure_cause(caught.value) == 'camera_firmware_missing'
