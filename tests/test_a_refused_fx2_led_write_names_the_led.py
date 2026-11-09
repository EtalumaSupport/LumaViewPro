# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 LED write the device refuses raises the LED board's fault, not the camera's.

The LED and the camera share the FX2's one USB write path, and that path's
fault was the camera's: an LED command on an LS620 whose write was refused
raised ``CameraWriteFailedError``, "The camera did not take the
VR_I2C_WRITE write". The LED's write now raises the faults every LED board
raises, ``BoardPortFailedError`` for a write not taken, and the device
staying on the bus leaves the LED connected for the next command. The
scope's unblockable all-off, ``force_off``, answers that fault as it answers
a board that did not answer, False, and a lit LED it could not darken is
still believed lit.
"""

from __future__ import annotations

import pytest

from _lumascope.drivers.exceptions import BoardPortFailedError
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0


@pytest.fixture
def ls620():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_a_refused_led_write_raises_the_led_board_fault(ls620):
    ls620.scope.simulation.refuse_led_write()

    with pytest.raises(BoardPortFailedError, match='LED board') as caught:
        ls620.scope.illumination.led_on(LED0, 10.0)

    assert caught.value.reason == 'board_port_failed'


def test_the_led_stays_connected_and_takes_the_next_command(ls620):
    ls620.scope.simulation.refuse_led_write()
    with pytest.raises(BoardPortFailedError):
        ls620.scope.illumination.led_on(LED0, 10.0)

    assert ls620.scope.led_connected is True
    assert ls620.scope.illumination.led_on(LED0, 10.0) > 0


def test_a_refused_force_off_answers_false_and_its_lit_led_is_still_believed_lit(ls620):
    ls620.scope.illumination.led_on(LED0, 10.0)
    ls620.scope.simulation.refuse_led_write()

    assert ls620.scope.illumination.force_off() is False

    lit = [r.state for r in ls620.scope.illumination.get_led_states() if r.name == LED0]
    assert lit == [LedState.LIT]
