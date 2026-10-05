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

from modules.exceptions import CameraSettingRejected, ConfigError, HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# Distinct from the simulated camera's own default (10 ms, 1 dB), so a camera
# left at its default cannot pass for one that took the layer.
_BF = {'exposure_ms': 4.0, 'gain_db': 3.0, 'auto_gain': False}
_BLUE = {'exposure_ms': 100.0, 'gain_db': 10.0, 'auto_gain': False}


def _camera(session) -> tuple[float, float]:
    imaging = session.scope.imaging
    return round(float(imaging.get_exposure_ms()), 3), round(float(imaging.get_gain_db()), 3)


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

    def test_a_camera_refusal_is_raised_once_the_stage_has_arrived(self, session, monkeypatch):
        protocol, _bf_idx, blue_idx = _two_step_protocol(session)
        refused = CameraSettingRejected(
            'gain_db', 10.0, title='Camera Setting Not Applied', message='x'
        )
        monkeypatch.setattr(
            session.scope.imaging,
            'apply_layer_camera_settings',
            MagicMock(side_effect=refused),
        )

        with pytest.raises(CameraSettingRejected) as raised:
            session.go_to_step(protocol, blue_idx)

        assert raised.value is refused
        assert not session.scope.motion.is_moving()


class TestTheMember:
    def test_a_layer_this_scope_lacks_is_refused_and_nothing_is_applied(self, session):
        before = _camera(session)
        with pytest.raises(ConfigError):
            session.apply_layer_camera('Lumi')
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
        monkeypatch.setattr(session.scope.imaging, 'apply_layer_camera_settings', applied)

        session.apply_layer_camera('Blue')

        sent = applied.call_args.kwargs
        assert sent['layer'] == 'Blue'
        assert (sent['exposure_ms'], sent['gain_db'], sent['auto_gain']) == (100.0, 10.0, False)
        assert sent['auto_gain_settings']['max_exposure_ms'] == 123.0
        assert sent['auto_gain_settings']['min_exposure_ms'] == 1.0


def test_no_gui_module_puts_a_layer_on_the_camera_itself():
    """The GUI applies a layer through the Session member, never the imaging one."""
    from tests.ast_seams import REPO_ROOT

    callers = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / 'ui').rglob('*.py'))
        if 'apply_layer_camera_settings(' in path.read_text()
    ]
    assert callers == []
