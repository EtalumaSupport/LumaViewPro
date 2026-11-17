# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A current handed to the LED above the board's maximum is lit at the
maximum, answered, and told once; one below zero is refused.

Before, ``led_on`` and ``led_on_by_address`` refused a current above
``capabilities.led_max_ma`` while a channel's stored current above it was
set at the maximum by the channel write, so one number met two rules by the
door it came in. Now the one LED judge (``IlluminationAPI._judge_light``)
commands the maximum, and the two members a caller hands a current to report
one ``ValueClampedNotice`` once the light has landed. The refusal below zero
is pinned in ``test_a_layer_or_led_channel_the_scope_cannot_act_on_is_refused.py``.
"""

import pytest

from _lumascope.modules.exceptions import ValueClampedNotice
from tests.stock_channels import LED0


def _clamp_notices(centre_posts):
    return [post.message for post in centre_posts if post.reason == ValueClampedNotice.reason]


def _lit_ma(scope, address):
    (reading,) = [r for r in scope.illumination.led_states() if r.address == address]
    return reading.illumination_ma


def test_a_channel_lit_above_the_maximum_is_lit_at_it_and_told(sim_scope, centre_posts):
    illumination = sim_scope.illumination
    maximum = sim_scope.capabilities.led_max_ma

    assert illumination.led_on(LED0, maximum + 500) == pytest.approx(maximum)

    assert _lit_ma(sim_scope, illumination.led_address_of(LED0)) == pytest.approx(maximum)
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([(LED0, 'illumination_ma', float(maximum + 500), float(maximum))]))
    ]
    illumination.led_off(LED0)


def test_an_address_lit_above_the_maximum_is_lit_at_it_and_told(sim_scope, centre_posts):
    illumination = sim_scope.illumination
    maximum = sim_scope.capabilities.led_max_ma
    address = illumination.led_address_of(LED0)

    assert illumination.led_on_by_address(address, maximum + 1) == pytest.approx(maximum)

    assert _lit_ma(sim_scope, address) == pytest.approx(maximum)
    assert _clamp_notices(centre_posts) == [
        str(
            ValueClampedNotice(
                [(f'LED address {address}', 'illumination_ma', float(maximum + 1), float(maximum))]
            )
        )
    ]
    illumination.led_off(LED0)


def test_a_light_at_the_maximum_is_told_nothing(sim_scope, centre_posts):
    illumination = sim_scope.illumination
    maximum = sim_scope.capabilities.led_max_ma

    assert illumination.led_on(LED0, maximum) == pytest.approx(maximum)

    assert _clamp_notices(centre_posts) == []
    illumination.led_off(LED0)
