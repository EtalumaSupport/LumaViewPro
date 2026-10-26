# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LED board address is lit and darkened by the engineering members.

``led_on_by_address`` / ``led_off_by_address`` are the members a bench tool
and a model this release does not know light an address through; every
other caller names the channel. The address goes on the lane a name's
address goes on: it reads back and is reported as the name's light is, a
model's absent LED is refused by its address, and each refusal names the
member called.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ArgumentRefusedError, HardwareCommandRefusedError, MissingPart
from modules.layer_record import UNRESOLVED
from modules.lumascope_api.illumination import LedReading, LedState
from tests.scope_fakes import build_scope, led_reading


@pytest.fixture
def scope():
    s = build_scope(simulate=True, register_atexit=False, sim_model='LS850T')
    yield s
    s.disconnect()


def _heard(scope) -> list[LedReading]:
    heard: list[LedReading] = []
    scope.illumination.add_led_listener(lambda *reading: heard.append(LedReading(*reading)))
    return heard


def test_an_address_lights_and_reports_as_its_channels_name_does(scope):
    """BF is address 3 on an LS850T: its light by address is its light by name."""
    illumination = scope.illumination
    heard = _heard(scope)
    by_name = illumination.led_on('BF', 50.0)
    named = (illumination.get_led_states(), list(heard))
    illumination.led_off('BF')
    heard.clear()

    by_address = illumination.led_on_by_address(3, 50.0)

    assert by_address == by_name
    assert (illumination.get_led_states(), heard) == named
    assert led_reading(scope, 3) == LedReading(3, 'BF', LedState.LIT, by_address)
    illumination.led_off_by_address(3)
    assert led_reading(scope, 3).state == LedState.DARK


def test_on_an_unresolved_identity_an_address_lights_and_reads_back_by_its_address(scope):
    scope.channel_identity = UNRESOLVED  # a model this release does not know

    commanded = scope.illumination.led_on_by_address(6, 50.0)

    assert led_reading(scope, 6) == LedReading(6, None, LedState.LIT, commanded)
    scope.illumination.led_off_by_address(6)
    assert led_reading(scope, 6).state == LedState.DARK


def test_an_address_the_model_does_not_drive_is_refused_naming_the_member(scope):
    scope.refresh_layer_identity(override_model='LS850-0')  # LED 1's channel, address 1, gone

    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.illumination.led_on_by_address(1, 50.0)

    assert (refused.value.reason, refused.value.member, refused.value.missing) == (
        'axis_absent',
        'led_on_by_address',
        MissingPart.led_address(1),
    )
    assert led_reading(scope, 1).state == LedState.DARK


@pytest.mark.parametrize('member', ['led_on_by_address', 'led_off_by_address'])
def test_an_address_off_the_board_is_refused_offering_the_boards_addresses(scope, member):
    board = tuple(scope._led_driver.available_channels())
    call = {
        'led_on_by_address': lambda: scope.illumination.led_on_by_address(99, 50.0),
        'led_off_by_address': lambda: scope.illumination.led_off_by_address(99),
    }[member]

    with pytest.raises(ArgumentRefusedError) as refused:
        call()

    assert (refused.value.reason, refused.value.argument, refused.value.offered) == (
        'led_address_unknown',
        'address',
        board,
    )
