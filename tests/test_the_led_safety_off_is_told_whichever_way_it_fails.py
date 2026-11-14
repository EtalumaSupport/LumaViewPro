# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bring-up's LED safety off is told whichever way it fails, and leaves every LED unknown.

The off is the scope's own, written as soon as the LED board is up. A
write that does not complete, because the board did not answer or its port
failed, leaves the scope up with the LED part's record
``safety_off_failed``, one ``LedSafetyOffNotTakenError`` told, and every
address unknown, never dark, so a later off is sent to a board still
connected. An EL-0940 whose port fails is not rejected as not live.

The Classic is the simulated FX2 under the production FX2 drivers; the
EL-0940 is the field LED firmware behind the emulated port.
"""

from __future__ import annotations

import pytest
from serial import SerialException, SerialTimeoutException

from _lumascope.drivers.exceptions import BoardNoReplyError, BoardPortFailedError
from _lumascope.drivers.fx2driver import FX2LEDController
from _lumascope.drivers.sim_wire.port import EmulatedPort
from _lumascope.modules.exceptions import LedSafetyOffNotTakenError
from _lumascope.modules.scope_parts import Board
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _classic_all_off_raises(monkeypatch, fault: Exception) -> None:
    """The FX2 LED driver's first all-off raises ``fault``; later ones are written."""
    # a stand-in by design: a fault the simulator cannot yet produce; its
    # fault doors are armed once the scope is built, and the safety off is
    # written while it is built
    real = FX2LEDController.leds_off
    raised = []

    def leds_off(self):
        if not raised:
            raised.append(fault)
            raise fault
        return real(self)

    monkeypatch.setattr(FX2LEDController, 'leds_off', leds_off)


def _el0940_all_off_write_raises(monkeypatch, error: Exception) -> None:
    """The emulated port's first ``LEDS_OFF`` write raises the serial ``error``."""
    # a stand-in by design: as for the Classic, a fault the simulator's doors
    # cannot arm before the scope is built
    real = EmulatedPort.write
    raised = []

    def write(self, data):
        if not raised and bytes(data).strip() == b'LEDS_OFF':
            raised.append(error)
            raise error
        return real(self, data)

    monkeypatch.setattr(EmulatedPort, 'write', write)


@pytest.mark.parametrize(
    ('model', 'tier', 'arm', 'fault'),
    [
        pytest.param(
            'LS620',
            'fast',
            lambda mp: _classic_all_off_raises(
                mp, BoardPortFailedError('The LED board did not take the VR_I2C_WRITE write')
            ),
            BoardPortFailedError,
            id='classic-port-failed',
        ),
        pytest.param(
            'LS620',
            'fast',
            lambda mp: _classic_all_off_raises(
                mp, BoardNoReplyError('The LED board did not answer the VR_I2C_WRITE write')
            ),
            BoardNoReplyError,
            id='classic-no-reply',
        ),
        pytest.param(
            'LS850T',
            'firmware',
            lambda mp: _el0940_all_off_write_raises(mp, SerialException('the cable was pulled')),
            BoardPortFailedError,
            id='el0940-port-failed',
            marks=pytest.mark.firmware_tier,
        ),
        pytest.param(
            'LS850T',
            'firmware',
            lambda mp: _el0940_all_off_write_raises(mp, SerialTimeoutException('write timeout')),
            BoardNoReplyError,
            id='el0940-no-reply',
            marks=pytest.mark.firmware_tier,
        ),
    ],
)
def test_a_safety_off_not_taken_is_told_once_and_leaves_every_led_unknown(
    monkeypatch, centre_posts, model, tier, arm, fault
):
    arm(monkeypatch)
    session = ScopeSession.create(
        complete_settings(microscope=model, simulator_tier=tier),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        led = next(p for p in session.bring_up_record().parts if p.part == Board.LED)
        assert (led.up, led.cause) == (True, 'safety_off_failed')
        assert led.detail.startswith(f'{fault.__name__}: ')
        told = [n for n in centre_posts if n.reason == 'safety_off_failed']
        assert [n.title for n in told] == [LedSafetyOffNotTakenError.title]
        states = {r.state for r in session.scope.illumination.led_states()}
        assert states == {LedState.UNKNOWN}
    finally:
        session.shutdown()
