# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera holds a layer's settings from bring-up, and after going to a step.

Outside a run, the only code that put a layer's exposure, gain and auto-gain
on the camera was the GUI's layer control. A script, and the GUI itself at
start-up, ran at the camera's default until a layer control was touched
(LS850T, Pylon: 10 ms / 0 dB after bring-up and after ``go_to_step``, the
step asking for 100 ms / 10 dB), and the files recorded those values.
``ScopeSession.apply_layer_camera`` is now that apply for every caller:
bring-up applies BF, ``go_to_step`` the step's layer, the GUI the layer
whose control changed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.exceptions import CameraSettingRejected, HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# Distinct from the simulated camera's own default (10 ms, 1 dB), so a camera
# left at its default cannot pass for one that took the layer.
_BF = {'exposure_ms': 4.0, 'gain_db': 3.0, 'auto_gain': False}
_BLUE = {'exposure_ms': 100.0, 'gain_db': 10.0, 'auto_gain': False}


def _camera(session) -> tuple[float, float]:
    camera = session.scope.camera
    return round(float(camera.read_exposure_ms()), 3), round(float(camera.read_gain_db()), 3)


def _said(monkeypatch) -> list[str]:
    import modules.scope_session as scope_session

    lines: list[str] = []
    monkeypatch.setattr(scope_session.logger, 'info', lambda msg, *a, **k: lines.append(str(msg)))
    return lines


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(BF=_BF, Blue=_BLUE), simulate=True)
    home_sim_scope(built.scope)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def _two_step_protocol(session):
    """A BF step and a Blue step at the current position, from the layers' settings."""
    for layer in session.get_layer_configs():
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'
    session.settings['Blue']['acquire'] = 'image'
    protocol = session.scope.protocols.create_protocol(
        empty_config=session.get_sequenced_capture_config()
    )
    session.add_step(protocol, before_step=0)
    colors = [protocol.step(idx=i)['Color'] for i in range(protocol.num_steps())]
    return protocol, colors.index('BF'), colors.index('Blue')


class TestBringUp:
    def test_the_camera_holds_bf_once_the_session_is_up(self, session):
        assert _camera(session) == (_BF['exposure_ms'], _BF['gain_db'])

    def test_a_scope_with_no_camera_is_not_applied_and_says_so(self, monkeypatch):
        applied = MagicMock()
        stand_in = SimpleNamespace(
            scope=SimpleNamespace(camera_connected=False),
            apply_layer_camera=applied,
            _set_shown_channel=lambda shown: None,
        )
        said = _said(monkeypatch)
        ScopeSession._apply_bring_up_layer(stand_in)
        applied.assert_not_called()
        assert any('no camera, so BF is not applied' in line for line in said)

    def test_a_scope_whose_layers_are_unresolved_still_comes_up(self, monkeypatch):
        applied = MagicMock()
        stand_in = SimpleNamespace(
            scope=SimpleNamespace(camera_connected=True),
            apply_layer_camera=applied,
            _set_shown_channel=lambda shown: None,
            _layers_on_scope=lambda: set(),
        )
        said = _said(monkeypatch)
        ScopeSession._apply_bring_up_layer(stand_in)
        applied.assert_not_called()
        assert any('no BF layer' in line for line in said)


class TestGoingToAStep:
    def test_each_go_ends_on_its_own_steps_layer(self, session):
        protocol, bf_idx, blue_idx = _two_step_protocol(session)

        session.go_to_step(protocol, blue_idx)
        assert _camera(session) == (_BLUE['exposure_ms'], _BLUE['gain_db'])

        session.go_to_step(protocol, bf_idx)
        assert _camera(session) == (_BF['exposure_ms'], _BF['gain_db'])

    def test_a_started_go_returns_with_the_camera_on_its_steps_layer(self, session):
        protocol, _bf_idx, blue_idx = _two_step_protocol(session)

        going = session.start_go_to_step(protocol, blue_idx)

        assert _camera(session) == (_BLUE['exposure_ms'], _BLUE['gain_db'])
        going.result(timeout=30)

    def test_a_repeat_after_another_layer_was_shown_shows_the_steps_layer(self, session):
        protocol, _bf_idx, blue_idx = _two_step_protocol(session)
        session.go_to_step(protocol, blue_idx)
        session.show_channel('BF')

        session.go_to_step(protocol, blue_idx)

        assert session.shown_channel.layer == 'Blue'
        assert _camera(session) == (_BLUE['exposure_ms'], _BLUE['gain_db'])

    def test_going_to_another_layers_step_during_a_recording_is_refused(self, session):
        protocol, _bf_idx, blue_idx = _two_step_protocol(session)
        recording = session.manual_recording.start(false_color_on=False)
        try:
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.go_to_step(protocol, blue_idx)
            assert (refused.value.reason, refused.value.holder) == (
                'exclusive_activity_running',
                'recording',
            )
            assert session.shown_channel.layer == 'BF'
        finally:
            recording.stop()
            recording.result(timeout=60)

    def test_a_manual_step_after_an_auto_gain_step_holds_its_own_gain(self, session):
        """Going back to a manual step from an auto-gain one puts the manual
        step's gain on the camera, and the capture gate expects it (#840).

        The camera's loop owns the gain while it runs and takes back a write
        made under it, which the simulated camera models; the gain is written
        once the loop is off. Written before the loop was turned off, the
        camera stayed at the loop's value and the gate expected nothing.
        """
        session.settings['Blue']['auto_gain'] = True
        protocol, bf_idx, blue_idx = _two_step_protocol(session)

        session.go_to_step(protocol, blue_idx)
        session.go_to_step(protocol, bf_idx)

        assert _camera(session) == (_BF['exposure_ms'], _BF['gain_db'])
        assert session.scope.imaging.frame_validity.target('gain') == _BF['gain_db']

    def test_a_camera_refusal_is_raised_once_the_stage_has_arrived(self, session, monkeypatch):
        protocol, _bf_idx, blue_idx = _two_step_protocol(session)
        refused = CameraSettingRejected('gain_db', 10.0, message='x')
        # a stand-in by design: the simulated camera takes every in-range value
        monkeypatch.setattr(session.scope.camera, 'hold_layer', MagicMock(side_effect=refused))

        with pytest.raises(CameraSettingRejected) as raised:
            session.go_to_step(protocol, blue_idx)

        assert raised.value is refused
        assert not session.scope.motion.is_moving()


class TestTheMember:
    def test_a_layer_this_scope_lacks_is_refused_and_nothing_is_applied(self, session):
        before = _camera(session)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.apply_layer_camera('Lumi')
        assert refused.value.reason == 'axis_absent'
        assert _camera(session) == before

    def test_a_held_scope_refuses_the_apply(self, session):
        before = _camera(session)
        held = session.activity_claim.try_claim('diagnostic')
        try:
            with pytest.raises(HardwareCommandRefusedError):
                session.apply_layer_camera('Blue')
        finally:
            held.release()
        assert _camera(session) == before

    def test_the_auto_gain_cap_carries_the_installations_override(self, session, monkeypatch):
        session.settings['ag_ae_max_exposure_ms'] = {'fluorescence': 123.0}
        applied = MagicMock(return_value=None)
        monkeypatch.setattr(session.scope.camera, 'hold_layer', applied)

        session.apply_layer_camera('Blue')

        layer, sent = applied.call_args.args
        assert layer == 'Blue'
        assert (sent.exposure_ms, sent.gain_db, sent.auto_gain) == (100.0, 10.0, False)
        assert sent.auto_gain_settings['max_exposure_ms'] == 123.0
        assert sent.auto_gain_settings['min_exposure_ms'] == 1.0


class TestTheHeldLayer:
    """The camera's record of the layer it holds: set by the Session's apply,
    cleared by every raw camera door, carried by a camera snapshot."""

    def test_the_layer_applied_is_the_layer_held(self, session):
        session.apply_layer_camera('Blue')
        assert session.scope.camera.get_held_layer() == 'Blue'

    def test_applying_the_held_layer_again_writes_nothing(self, session):
        imaging = session.scope.imaging
        session.apply_layer_camera('Blue')
        # The capture drains the apply's writes, so the frames are valid after it.
        imaging.capture_and_wait(timeout_s=5.0)
        assert imaging.frame_validity.frames_until_valid() == 0

        session.apply_layer_camera('Blue')

        assert imaging.frame_validity.frames_until_valid() == 0

    def test_a_store_change_is_written_although_the_layer_is_held(self, session):
        session.apply_layer_camera('Blue')
        session.settings['Blue']['exposure_ms'] = 55.0

        session.apply_layer_camera('Blue')

        assert _camera(session) == (55.0, _BLUE['gain_db'])

    def test_a_raw_camera_write_leaves_no_layer_held(self, session):
        camera = session.scope.camera
        session.apply_layer_camera('Blue')

        camera.set_exposure_ms(30.0)

        assert camera.get_held_layer() is None
        session.apply_layer_camera('Blue')
        assert _camera(session) == (_BLUE['exposure_ms'], _BLUE['gain_db'])

    def test_a_restored_snapshot_holds_the_layer_it_was_taken_with(self, session):
        camera = session.scope.camera
        session.apply_layer_camera('Blue')
        snapshot = camera.save_camera_state('test')
        camera.set_gain_db(1.0)

        camera.restore_camera_state(snapshot)

        assert camera.get_held_layer() == 'Blue'

    def test_a_trimmed_restore_leaves_no_layer_held(self, session):
        camera = session.scope.camera
        session.apply_layer_camera('Blue')
        snapshot = camera.save_camera_state('test')
        del snapshot['gain_db']

        camera.restore_camera_state(snapshot)

        assert camera.get_held_layer() is None

    def test_a_refused_layer_is_not_held_and_is_written_again(self, session, monkeypatch):
        camera = session.scope.camera
        real = camera._apply_settings_impl
        # a stand-in by design: the simulated camera takes every in-range value
        monkeypatch.setattr(
            camera,
            '_apply_settings_impl',
            MagicMock(side_effect=CameraSettingRejected('gain_db', 10.0, message='x')),
        )
        with pytest.raises(CameraSettingRejected):
            session.apply_layer_camera('Blue')
        assert camera.get_held_layer() is None

        monkeypatch.setattr(camera, '_apply_settings_impl', real)
        session.apply_layer_camera('Blue')

        assert camera.get_held_layer() == 'Blue'
        assert _camera(session) == (_BLUE['exposure_ms'], _BLUE['gain_db'])


def test_no_gui_module_puts_a_layer_on_the_camera_itself():
    """The GUI applies a layer through the Session member, never the camera one."""
    from tests.ast_seams import REPO_ROOT

    callers = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / 'ui').rglob('*.py'))
        if 'camera.apply(' in path.read_text()
    ]
    assert callers == []


class TestAWriteToTheShownLayer:
    """A write to the shown layer's camera or LED setting is on the hardware
    when it returns; a hidden layer's, or one under a held scope, is stored."""

    def test_the_shown_layers_exposure_is_on_the_camera_when_the_write_returns(self, session):
        session.update_settings('BF.exposure_ms', 12.0)

        assert _camera(session) == (12.0, _BF['gain_db'])

    def test_a_hidden_layers_exposure_is_stored_only(self, session):
        session.update_settings('Blue.exposure_ms', 33.0)

        assert _camera(session) == (_BF['exposure_ms'], _BF['gain_db'])
        assert session.settings['Blue']['exposure_ms'] == 33.0

    def test_the_lit_shown_layers_illumination_is_on_its_led(self, session):
        session.light_shown_channel(True)

        session.update_settings('BF.illumination_ma', 7.0)

        assert session.scope.illumination.get_led_state('BF').illumination_ma == 7.0

    def test_a_write_while_the_scope_is_held_is_stored_only(self, session):
        held = session.activity_claim.try_claim('diagnostic')
        try:
            session.update_settings('BF.exposure_ms', 12.0)
        finally:
            held.release()

        assert session.settings['BF']['exposure_ms'] == 12.0
        assert _camera(session) == (_BF['exposure_ms'], _BF['gain_db'])

    def test_auto_gain_turned_on_for_the_shown_layer_arms_the_camera(self, session):
        session.set_layer_auto_gain('BF', True)

        assert session.scope.camera.lock_auto_gain().state is not None
