# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope with no FX2 attached records its absent boards as not detected.

The registry tries the FX2 drivers on every scope, most of which have no
FX2, and ranks a driver that raised above one that reported found=False.
The FX2 drivers raised when no FX2 was on the bus, so on a scope whose
EL-0940 LED board was simply absent the bring-up record said the LED board
"failed to start", with the FX2's "No Lumascope FX2 device found" as its
detail (a no-hardware sim walk, 2026-10-01). Absent is found=False; an FX2
that is present and fails still raises.
"""

from unittest.mock import MagicMock

import pytest

from drivers import fx2driver
from drivers.registry import DriverFallback, DriverRegistry


@pytest.fixture
def bus(monkeypatch):
    """A USB bus the real connection code enumerates; empty unless told."""
    transport = MagicMock(name='transport')
    transport.find.return_value = None
    fx2driver._FX2Connection._reset_for_test()
    monkeypatch.setattr(fx2driver, '_platform_transport', lambda: transport)
    yield transport
    fx2driver._FX2Connection._reset_for_test()


class _AbsentBoard:
    """A driver whose board is not attached, as the EL-0940 drivers report it."""

    def __init__(self, **kwargs):
        self.found = False


class _Null:
    pass


@pytest.mark.parametrize('fx2_driver', [fx2driver.FX2LEDController, fx2driver.FX2Camera])
def test_with_no_fx2_on_the_bus_the_fallback_is_not_detected(bus, fx2_driver):
    registry = DriverRegistry('board')
    registry.register('fx2', priority=100)(fx2_driver)
    registry.register('el0940', priority=50)(_AbsentBoard)
    registry.register('null', priority=0)(_Null)

    instance, fallback = registry.create_with_fallback('auto')

    assert isinstance(instance, _Null)
    assert fallback == DriverFallback('not_detected', (fx2_driver.__name__, '_AbsentBoard'))


@pytest.mark.parametrize('fx2_driver', [fx2driver.FX2LEDController, fx2driver.FX2Camera])
def test_an_fx2_that_is_present_and_fails_still_raises(bus, monkeypatch, fx2_driver):
    # The bootloader answers and its firmware upload fails: a present FX2
    # that cannot be used is a failure, never an absence.
    bus.find.side_effect = lambda pid: None if pid == fx2driver.PID_APP else object()

    def upload_fails(self, dev, path):
        raise RuntimeError('firmware upload failed')

    monkeypatch.setattr(fx2driver._FX2Connection, '_upload_firmware', upload_fails)
    monkeypatch.setattr(fx2driver._FX2Connection, 'find_firmware_path', staticmethod(lambda: ''))

    with pytest.raises(RuntimeError, match='firmware upload failed'):
        fx2_driver()
