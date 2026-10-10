# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer or LED channel no scope can act on is refused as the request's.

A name that was no layer met a different answer at each member: a
ConfigError naming the model's LEDs, another naming the catalogue, a
KeyError, a ValueError calling it a channel, a silent off, a read of
"dark". ``led_on(True)`` lit Green, since ``True == 1``, and a still named
for Lumi on a scope with no Lumi was written to disk. Each member now asks
one check first -- ``ArgumentRefusedError`` ``layer_unknown`` offering
the catalogue, ``led_channel_unknown`` offering the board's numbers -- and
a layer the model lacks is absent hardware, ``axis_absent`` naming the
layer. A value that is no name and no whole number -- None, a ``bool``, a
float -- is the ``@api`` door's
(``tests/test_an_argument_of_another_type_is_refused_at_the_door.py``),
except for the two runs, whose GUI hands None when no drawer is open:
``no_layer_selected``.
"""

import numpy as np
import pytest

from modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    MissingPart,
    RefusalCause,
)
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session

CATALOGUE = ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = _session(tmp_path_factory.mktemp('c5'), 'LS850')
    yield s
    s.shutdown()


def _refused(call, reason):
    with pytest.raises(ArgumentRefusedError) as refused:
        call()
    assert refused.value.reason == reason
    assert refused.value.cause == RefusalCause.REQUEST
    return refused.value


def _lit(session):
    states = {layer: session.scope.illumination.get_led_state(layer) for layer in CATALOGUE}
    return sorted(layer for layer, state in states.items() if state and state['enabled'])


NAME_MEMBERS = {
    'led_on': lambda s, v: s.scope.illumination.led_on(v, 10.0),
    'led_off': lambda s, v: s.scope.illumination.led_off(v),
    'get_led_state': lambda s, v: s.scope.illumination.get_led_state(v),
    'color2ch': lambda s, v: s.scope.illumination.color2ch(v),
    'set_layer_acquire': lambda s, v: s.set_layer_acquire(v, 'image'),
    'set_layer_auto_gain': lambda s, v: s.set_layer_auto_gain(v, True),
    'saved_focus': lambda s, v: s.saved_focus(v),
    'save_layer_focus': lambda s, v: s.save_layer_focus(v, 0.0),
    'apply_layer_camera': lambda s, v: s.apply_layer_camera(v),
    'get_layer_configs': lambda s, v: s.get_layer_configs([v]),
    'capture': lambda s, v: s.manual_capture.capture(layer=v, false_color_on=False),
    'update_step': lambda s, v: s.update_step(s.create_empty_protocol(), 0, layer=v),
    'recording': lambda s, v: s.manual_recording.start(layer=v),
}

# A layer this model lacks, on the members that act on a layer, is absent hardware.
ABSENT_MEMBERS = (
    'set_layer_acquire',
    'set_layer_auto_gain',
    'save_layer_focus',
    'apply_layer_camera',
    'capture',
    'update_step',
    'recording',
)


@pytest.mark.parametrize('name', ['Purple', 'bf', ''])
@pytest.mark.parametrize('member', NAME_MEMBERS)
def test_a_name_that_is_no_layer_is_refused_offering_the_catalogue(session, member, name):
    refusal = _refused(lambda: NAME_MEMBERS[member](session, name), 'layer_unknown')

    assert (refusal.value, refusal.offered) == (name, CATALOGUE)
    assert _lit(session) == []


@pytest.mark.parametrize('channel', [99, -1, np.int64(99)])
def test_a_number_off_the_boards_table_lights_nothing(session, channel):
    refusal = _refused(
        lambda: session.scope.illumination.led_on(channel, 10.0), 'led_channel_unknown'
    )

    assert refusal.offered == (0, 1, 2, 3, 4, 5)
    assert _lit(session) == []
    _refused(lambda: session.scope.illumination.ch2color(channel), 'led_channel_unknown')


@pytest.mark.parametrize('member', ABSENT_MEMBERS)
def test_a_layer_the_model_lacks_is_absent_hardware(session, member):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        NAME_MEMBERS[member](session, 'Lumi')
    assert refused.value.missing == MissingPart.layer('Lumi')
    assert not session.manual_recording.is_busy


def test_a_layer_the_model_lacks_is_a_name_and_keeps_its_answers(session):
    illumination = session.scope.illumination
    assert illumination.color2ch('Lumi') is None
    assert illumination.get_led_state('Lumi') == {'enabled': False, 'illumination_ma': None}
    illumination.led_off('Lumi')


def test_a_current_off_the_boards_range_is_refused_with_its_limits(session):
    refusal = _refused(
        lambda: session.scope.illumination.led_on('BF', -1), 'illumination_out_of_range'
    )
    assert (refusal.argument, refusal.limits) == ('illumination_ma', (0, 1000))
    assert _lit(session) == []


RUNNERS = {
    'run_autofocus': lambda r, v: r.run_autofocus(layer=v),
    'run_zstack': lambda r, v: r.run_zstack(layer=v),
}


@pytest.mark.parametrize(
    'member, name, reason',
    [
        *((member, 'Purple', 'layer_unknown') for member in ('led_on', 'led_off', *RUNNERS)),
        *((member, None, 'no_layer_selected') for member in RUNNERS),
    ],
)
def test_a_name_is_refused_whatever_the_scope_is_doing(tmp_path, member, name, reason):
    """Refused at the door, so a shut scope does not answer for the name."""
    s = _session(tmp_path, 'LS850')
    runner = s.create_protocol_runner()
    s.shutdown()
    if member in RUNNERS:
        _refused(lambda: RUNNERS[member](runner, name), reason)
    else:
        _refused(lambda: NAME_MEMBERS[member](s, name), reason)


def test_a_reason_takes_limits_exactly_when_its_words_name_them():
    with pytest.raises(TypeError):
        ArgumentRefusedError('illumination_out_of_range', argument='illumination_ma', value=-1)
    with pytest.raises(TypeError):
        ArgumentRefusedError('not_a_number', argument='gain_db', value='x', limits=(0, 1))
