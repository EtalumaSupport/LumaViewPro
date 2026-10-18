# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The channel the scope shows is one Session store, changed through the API.

``show_channel(layer)`` makes a layer the shown channel, every other LED
dark and the layer lit only if the shown channel was lit;
``light_shown_channel(lit)`` lights or darkens it. Listeners hear each
change. A light-one-darken-rest refuses before it turns anything off, so a
refused light leaves the LEDs as they were.
"""

import pytest

from modules.exceptions import ArgumentRefusedError
from modules.lumascope_api.illumination import LedState, LedTransition, LedTransitionCtx
from modules.scope_session import ScopeSession, ShownChannel
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(microscope='LS850'), simulate=True)
    yield built
    built.shutdown()


def _lit(session) -> set[str]:
    return {r.name for r in session.scope.illumination.get_led_states() if r.state is LedState.LIT}


def test_the_session_comes_up_showing_bf_dark(session):
    assert session.shown_channel == ShownChannel('BF', lit=False)


def test_showing_a_channel_keeps_it_dark_until_it_is_lit(session):
    heard = []
    session.add_shown_channel_listener(lambda layer, lit: heard.append((layer, lit)))

    assert session.show_channel('Blue') == ShownChannel('Blue', lit=False)
    assert _lit(session) == set()

    session.light_shown_channel(True)
    blue = session.scope.illumination.get_led_state('Blue')
    assert (blue.state, blue.illumination_ma) == (
        LedState.LIT,
        session.settings['Blue']['illumination_ma'],
    )
    assert heard == [('Blue', False), ('Blue', True)]


def test_a_lit_shown_channel_moves_its_light_to_the_next_layer(session):
    session.light_shown_channel(True)
    assert _lit(session) == {'BF'}

    assert session.show_channel('Green') == ShownChannel('Green', lit=True)
    assert _lit(session) == {'Green'}

    session.light_shown_channel(False)
    assert _lit(session) == set()


def test_a_refused_light_turns_nothing_off(session):
    """The diff judges its whole target before it writes: BF stays lit
    when the light that would replace it is out of the board's range."""
    session.light_shown_channel(True)
    illumination = session.scope.illumination
    with pytest.raises(ArgumentRefusedError) as refused:
        illumination.apply_transition(
            LedTransition.MANUAL_STEP,
            LedTransitionCtx(
                channel=illumination.color2ch('Blue'),
                illumination_ma=session.scope.capabilities.led_max_ma + 1,
                preview_on=True,
            ),
        )
    assert refused.value.reason == 'illumination_out_of_range'
    assert _lit(session) == {'BF'}


def test_a_runs_step_shows_its_layer_lit_only_when_its_led_is(tmp_path):
    """A run's step makes its layer the shown channel, lit as its LED reads:
    a layer with no LED (the Lumi model's Lumi) is shown dark, a BF step lit."""
    from tests.scope_fakes import home_sim_scope
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
    from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

    built = ScopeSession.create(
        complete_settings(**_settings(tmp_path, microscope='Lumi')), simulate=True
    )
    try:
        home_sim_scope(built.scope)
        heard = []
        built.add_shown_channel_listener(lambda layer, lit: heard.append((layer, lit)))
        lumi = dict(_make_single_step_protocol(color='Lumi').step(idx=0))
        bf = {**dict(_make_single_step_protocol(color='BF').step(idx=0)), 'Name': 'b'}
        bf['Step Index'] = 1

        run = built.create_protocol_runner().start_single_scan(
            protocol=_build_real_protocol([lumi, bf]), parent_dir=str(tmp_path)
        )
        assert run.result(timeout=COMPLETION_TIMEOUT).status == 'completed'

        assert heard[:2] == [('Lumi', False), ('BF', True)]
    finally:
        built.shutdown()


def test_lighting_the_shown_channel_leaves_the_camera_on_its_layer(session):
    """A still held another layer (a raw LED named it); lighting the shown
    layer afterwards puts the camera back on the shown layer's settings."""
    camera = session.scope.camera
    session.update_settings('Blue.exposure_ms', 88.0)
    session.scope.illumination.led_on('Blue', 100.0)
    session.manual_capture.capture(false_color_on=False).result(timeout=30)
    session.scope.illumination.led_off('Blue')

    session.light_shown_channel(True)

    assert camera.get_held_layer() == 'BF'
    assert round(camera.get_exposure_ms(), 3) == session.settings['BF']['exposure_ms']


def test_bring_up_again_shows_bf_dark(session):
    session.show_channel('Blue')
    session.light_shown_channel(True)

    session.configure_scope()

    assert session.shown_channel == ShownChannel('BF', lit=False)


def _run_one_blue_step(built, tmp_path):
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
    from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

    blue = dict(_make_single_step_protocol(color='Blue').step(idx=0))
    run = built.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol([blue]), parent_dir=str(tmp_path)
    )
    assert run.result(timeout=COMPLETION_TIMEOUT).status == 'completed'


def _session_showing_bf(tmp_path):
    from tests.scope_fakes import home_sim_scope
    from tests.test_a_run_needs_every_axis_position import _settings

    built = ScopeSession.create(
        complete_settings(
            **_settings(tmp_path),
            BF={'exposure_ms': 2.0, 'gain_db': 1.0, 'auto_gain': False},
        ),
        simulate=True,
    )
    home_sim_scope(built.scope)
    built.show_channel('BF')
    return built


def test_a_runs_end_shows_the_layer_shown_before_it(tmp_path):
    """BF shown before a Blue step's run: after it BF is shown again and the
    camera holds BF's stored settings, so a still is named and set for BF."""
    built = _session_showing_bf(tmp_path)
    try:
        _run_one_blue_step(built, tmp_path)

        assert built.shown_channel.layer == 'BF'
        assert built.scope.camera.get_held_layer() == 'BF'
        assert round(built.scope.camera.get_exposure_ms(), 3) == 2.0
    finally:
        built.shutdown()


def test_a_write_during_a_run_to_the_layer_shown_before_it_is_on_the_camera_after(tmp_path):
    built = _session_showing_bf(tmp_path)

    def write_while_the_step_is_lit(layer, _lit):
        if layer == 'Blue':
            built.update_settings('BF.exposure_ms', 7.0)

    built.add_shown_channel_listener(write_while_the_step_is_lit)
    try:
        _run_one_blue_step(built, tmp_path)

        assert round(built.scope.camera.get_exposure_ms(), 3) == 7.0
    finally:
        built.remove_shown_channel_listener(write_while_the_step_is_lit)
        built.shutdown()


def test_showing_another_layer_during_a_recording_is_refused_and_changes_nothing(session):
    from modules.exceptions import HardwareCommandRefusedError

    session.light_shown_channel(True)
    recording = session.manual_recording.start(false_color_on=False)
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.show_channel('Blue')
        assert (refused.value.reason, refused.value.holder) == (
            'exclusive_activity_running',
            'recording',
        )
        assert session.shown_channel == ShownChannel('BF', lit=True)
        assert _lit(session) == {'BF'}
    finally:
        recording.stop()
        recording.result(timeout=60)


def test_showing_the_shown_layer_during_a_recording_is_no_change(session):
    recording = session.manual_recording.start(false_color_on=False)
    try:
        assert session.show_channel('BF') == ShownChannel('BF', lit=False)
    finally:
        recording.stop()
        recording.result(timeout=60)


def test_putting_a_layer_on_the_camera_is_not_a_rest_route():
    """``show_channel`` is the one public way a layer's settings reach the
    camera; the Session's own apply is in-process."""
    from modules import wire_encoding
    from rest import routes as rest_routes

    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    names = {route.member.name for route in rest_routes.routes(ScopeSession, handed_out=handed_out)}

    assert 'show_channel' in names
    assert 'apply_layer_camera' not in names
