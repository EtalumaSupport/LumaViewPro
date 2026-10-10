# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An argument of another type than its member declares is refused at the ``@api`` door.

Before the door each member met a wrong type its own way, or not at all: a
None axis or layer was a name refused by name, ``True`` was a number some
members refused and others did not, a string slot reached the turret's
range check, a float frame side reached the camera. Now every member's
annotation is held at the call, for every caller, before the body runs:
``ArgumentTypeRefusedError`` (a ``TypeError``, reason
``wrong_argument_type``, the request's). A number is taken by what it means
and reaches the body as the Python number; the door judges the type only.
"""

import inspect
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.api_surface import api
from modules.exceptions import ArgumentRefusedError, ArgumentTypeRefusedError, RefusalCause
from modules.scope_session import ScopeSession
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session
from tests.test_run_refusal_contract import _make_single_step_protocol


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = _session(tmp_path_factory.mktemp('door'), 'LS850T')
    yield s
    s.shutdown()


def _refused(call, member, argument):
    with pytest.raises(ArgumentTypeRefusedError) as refused:
        call()
    assert refused.value.reason == 'wrong_argument_type'
    assert refused.value.cause == RefusalCause.REQUEST
    assert (refused.value.member, refused.value.argument) == (member, argument)
    return refused.value


def test_a_number_reaches_the_member_as_the_python_number_it_means(session):
    imaging = session.scope.imaging
    imaging.set_exposure_ms(np.float64(12.5))
    assert type(imaging.exposure_ms_cached) is float
    assert imaging.exposure_ms_cached == 12.5
    imaging.set_exposure_ms(20)
    assert type(imaging.exposure_ms_cached) is float
    assert imaging.exposure_ms_cached == 20.0


def test_a_setting_written_as_a_numpy_number_is_stored_as_the_python_one(session):
    session.update_settings('video.max_fps', np.int64(25))
    assert type(session.get_setting('video.max_fps')) is int
    session.update_settings('video.max_fps', np.float64(30.0))
    assert type(session.get_setting('video.max_fps')) is float


def test_a_numpy_index_reaches_a_protocol_step_as_an_int():
    """Post-processing hands a step index read from a table, an ``np.int64``."""
    protocol = _make_single_step_protocol()
    assert protocol.step(np.int64(0))['Name'] == protocol.step(0)['Name']


def test_a_bool_is_never_a_number(session):
    before = session.scope.imaging.exposure_ms_cached
    refused = _refused(
        lambda: session.scope.imaging.set_exposure_ms(True),
        'ImagingAPI.set_exposure_ms',
        'exposure_ms',
    )
    assert refused.declared == 'float'
    assert session.scope.imaging.exposure_ms_cached == before


def test_a_float_is_not_a_whole_number(session):
    _refused(
        lambda: session.scope.diagnostics.set_motor_fan_duty(50.0),
        'DiagnosticsAPI.set_motor_fan_duty',
        'duty_pct',
    )
    _refused(lambda: session.set_frame_size(math.nan, 400), 'ScopeSession.set_frame_size', 'width')


def test_none_is_taken_only_where_the_signature_says_so(session):
    refused = _refused(
        lambda: session.scope.motion.start_home(None), 'MotionAPI.start_home', 'axis'
    )
    assert refused.declared == 'str'
    _refused(lambda: session.saved_focus(None), 'ScopeSession.saved_focus', 'layer')


@pytest.mark.parametrize('member', ['run_zstack', 'run_autofocus'])
def test_a_run_given_no_layer_is_told_none_is_selected(session, member):
    """The GUI hands the open drawer's layer, None when no drawer is open:
    these two take None and answer it by name."""
    runner = session.create_protocol_runner()
    with pytest.raises(ArgumentRefusedError) as refused:
        getattr(runner, member)(layer=None)
    assert refused.value.reason == 'no_layer_selected'


def test_a_union_takes_each_of_its_types_and_nothing_else(session):
    illumination = session.scope.illumination
    channel = session.scope.illumination.color2ch('BF')
    illumination.led_off(np.int64(channel))
    illumination.led_off('BF')
    for value in (True, 2.5, None):
        refused = _refused(
            lambda value=value: illumination.led_off(value), 'IlluminationAPI.led_off', 'channel'
        )
        assert refused.declared == 'int | str'


def test_a_class_is_taken_only_as_itself(session):
    runner = session.create_protocol_runner()
    _refused(
        lambda: runner.run_single_scan(protocol=MagicMock()),
        'ProtocolRunner.run_single_scan',
        'protocol',
    )


def test_a_dictionary_a_callable_and_a_duration_are_held_to_their_kinds(session):
    _refused(
        lambda: session.scope.protocols.create_protocol(input_config=['not', 'a', 'dict']),
        'ProtocolsAPI.create_protocol',
        'input_config',
    )
    _refused(
        lambda: session.scope.illumination.add_led_listener(5),
        'IlluminationAPI.add_led_listener',
        'listener',
    )
    protocol = _make_single_step_protocol()
    before = protocol.period()
    _refused(lambda: protocol.modify_time_params(period=5), 'Protocol.modify_time_params', 'period')
    assert protocol.period() == before


def test_the_refusal_says_which_member_takes_what(session):
    refused = _refused(
        lambda: session.select_labware(['96 well microplate']),
        'ScopeSession.select_labware',
        'labware_name',
    )
    assert isinstance(refused, TypeError)
    assert str(refused) == (
        "ScopeSession.select_labware takes labware_name as str; ['96 well microplate'] is a list."
    )


def test_a_member_with_nothing_to_check_is_the_one_marked():
    class Holder:
        def read(self) -> int:
            return 1

        def take(self, n: int) -> int:
            return n

    read, take = Holder.read, Holder.take
    assert api(read) is read
    marked = api(take)
    assert marked is not take
    assert inspect.unwrap(marked) is take


def test_a_static_and_a_class_member_are_still_reached_through_the_class(tmp_path):
    """A path is a ``FilePath``: a ``pathlib.Path`` passes the door and
    reaches the member, which judges the folder."""
    with pytest.raises(ArgumentRefusedError) as refused:
        ScopeSession.load_user_settings(tmp_path)
    assert refused.value.reason == 'not_an_installation'
    _refused(
        lambda: ScopeSession.load_user_settings(7), 'ScopeSession.load_user_settings', 'source_path'
    )
    _refused(
        lambda: ScopeSession.create(settings=7, simulate=True), 'ScopeSession.create', 'settings'
    )
