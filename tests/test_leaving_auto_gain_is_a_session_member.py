# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Leaving auto-gain stores what the lock reached, through one Session member.

The API's lock decides the values a caller leaving auto-gain stores (the
achieved gain, and the achieved exposure floored to the class's usable floor,
``stored_exposure_ms``), but the store write lived in the GUI's toggle
callback, so a script or a REST caller had to re-implement it.
``ScopeSession.set_channel_auto_gain`` is now that write for every caller, and
the GUI's toggle submits it on the camera lane.

These run on a simulated session: its auto-gain loop converges the gain to
the middle of its bounds and leaves the exposure where it was, so the
exposure a layer is armed at decides what the lock reaches. The stored
exposure's per-class floor is pinned in
``tests/test_layer_control_ag_exposure_floor.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.exceptions import (
    ArgumentRefusedError,
    ArgumentTypeRefusedError,
    HardwareCommandRefusedError,
)
from modules.lumascope_api.camera import AutoGainConvergence, CameraAutoMode
from modules.scope_session import ScopeSession
from tests.stock_channels import LED0, LED1

# The values a layer is left at, so a store the member made is told apart.
_EXPOSURE_MS = 42.0
_GAIN_DB = 7.0


def _armed(session, layer='BF', exposure_ms=_EXPOSURE_MS):
    """Show *layer* at *exposure_ms* and arm its auto-gain: the simulated loop
    converges the gain to the middle of its bounds and leaves the exposure."""
    session.update_settings(f'{layer}.exposure_ms', exposure_ms)
    session.update_settings(f'{layer}.gain_db', _GAIN_DB)
    session.show_channel(layer)
    session.set_channel_auto_gain(layer, True)
    return session


def _stored(session, layer):
    return {
        'exposure_ms': session.get_setting(f'{layer}.exposure_ms'),
        'gain_db': session.get_setting(f'{layer}.gain_db'),
        'auto_gain': session.channel_auto_gain(layer),
    }


@pytest.mark.parametrize(
    'exposure_ms, state',
    [
        (5.0, AutoGainConvergence.CONVERGED),
        (0.03, AutoGainConvergence.AT_MINIMUM),  # under the transmitted floor
        (None, AutoGainConvergence.MAXED),  # at the transmitted ceiling
    ],
)
def test_turning_off_stores_what_the_lock_reached(sim_session, exposure_ms, state):
    if exposure_ms is None:
        exposure_ms = sim_session.get_setting('ag_ae_max_exposure_ms.transmitted')
    _armed(sim_session, exposure_ms=exposure_ms)

    lock = sim_session.set_channel_auto_gain('BF', False)

    assert lock.state is state
    assert lock.gain_db != _GAIN_DB, 'the loop left the gain where it was: nothing was reached'
    assert _stored(sim_session, 'BF') == {
        'exposure_ms': round(lock.stored_exposure_ms, 2),
        'gain_db': round(lock.gain_db, 1),
        'auto_gain': False,
    }


def test_the_stored_values_carry_the_settings_resolution(sim_session):
    sim_session.update_settings('protocol.autogain.min_gain_db', 2.0)
    sim_session.update_settings('protocol.autogain.max_gain_db', 2.12)
    _armed(sim_session, exposure_ms=5.0049)

    lock = sim_session.set_channel_auto_gain('BF', False)

    assert (lock.exposure_ms, lock.gain_db) == pytest.approx((5.0049, 2.06))
    assert sim_session.get_setting('BF.gain_db') == 2.1
    assert sim_session.get_setting('BF.exposure_ms') == 5.0


def test_a_failed_lock_keeps_the_values_and_still_turns_auto_gain_off(sim_session):
    _armed(sim_session)
    # No frame comes under the arm, so what the loop reached is not known.
    sim_session.scope.simulation.stall_stream(after_s=0.0, for_s=60.0)

    lock = sim_session.set_channel_auto_gain('BF', False)

    assert lock.state is AutoGainConvergence.FAILED
    assert _stored(sim_session, 'BF') == {
        'exposure_ms': _EXPOSURE_MS,
        'gain_db': _GAIN_DB,
        'auto_gain': False,
    }


def test_turning_off_with_no_arm_standing_stores_only_the_preference(sim_session):
    sim_session.update_settings('BF.exposure_ms', _EXPOSURE_MS)
    sim_session.update_settings('BF.gain_db', _GAIN_DB)
    sim_session.show_channel('BF')

    lock = sim_session.set_channel_auto_gain('BF', False)

    assert lock.state is None
    assert _stored(sim_session, 'BF') == {
        'exposure_ms': _EXPOSURE_MS,
        'gain_db': _GAIN_DB,
        'auto_gain': False,
    }


def test_turning_the_shown_layer_on_stores_it_locks_nothing_and_arms_the_camera(sim_session):
    sim_session.show_channel('BF')

    assert sim_session.set_channel_auto_gain('BF', True) is None

    assert sim_session.channel_auto_gain('BF') is True
    assert sim_session.channel_auto_gain_lock('BF') is None
    assert sim_session.scope.camera.read_auto_gain() is CameraAutoMode.CONTINUOUS


def test_turning_off_the_shown_layer_leaves_the_camera_holding_it(sim_session):
    _armed(sim_session)

    sim_session.set_channel_auto_gain('BF', False)

    camera = sim_session.scope.camera
    assert camera.get_held_layer() == 'BF'
    assert camera.read_auto_gain() is CameraAutoMode.OFF
    assert round(camera.gain_db, 1) == sim_session.get_setting('BF.gain_db')


def test_turning_off_a_hidden_layer_stores_only_its_preference_and_locks_nothing(sim_session):
    """The camera's arm is the shown layer's: its values are not the hidden
    layer's to take, and the shown layer stays armed."""
    sim_session.update_settings(f'{LED1}.exposure_ms', 55.0)
    sim_session.update_settings(f'{LED1}.gain_db', 4.0)
    _armed(sim_session)

    lock = sim_session.set_channel_auto_gain(LED1, False)

    assert lock.state is None
    assert _stored(sim_session, LED1) == {'exposure_ms': 55.0, 'gain_db': 4.0, 'auto_gain': False}
    assert sim_session.scope.camera.read_auto_gain() is CameraAutoMode.CONTINUOUS


@pytest.mark.parametrize(
    'sim_session, layer, enabled, refused, reason',
    [
        ({}, 'UV', False, ArgumentRefusedError, 'channel_unknown'),  # not a layer of this release
        (
            {'microscope': 'Lumi'},
            LED0,
            True,
            HardwareCommandRefusedError,
            'not_on_this_scope',
        ),  # no LED0 here
        ({}, 'BF', 'off', ArgumentTypeRefusedError, 'wrong_argument_type'),  # not a bool
    ],
    indirect=['sim_session'],
)
def test_a_refused_request_changes_nothing(sim_session, layer, enabled, refused, reason):
    before = sim_session.get_settings_snapshot()

    with pytest.raises(refused) as raised:
        sim_session.set_channel_auto_gain(layer, enabled)
    assert getattr(raised.value, 'reason', None) == reason

    assert sim_session.get_settings_snapshot() == before


@pytest.mark.parametrize('sim_session', [{'microscope': 'Lumi'}], indirect=True)
def test_turning_off_a_layer_this_scope_lacks_is_admitted(sim_session):
    sim_session.set_channel_auto_gain(LED0, False)

    assert sim_session.channel_auto_gain(LED0) is False


# ---------------------------------------------------------------------------
# The GUI's toggle submits the member on the camera lane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('state, enabled', [('down', True), ('normal', False)])
def test_the_toggle_submits_the_member_on_the_camera_lane_and_redraws(
    gui_ctx, monkeypatch, state, enabled
):
    import ui.layer_control as layer_control

    submits = []
    monkeypatch.setattr(
        layer_control,
        'submit_reported',
        lambda call, redraw, label, *, lane=None, stop=False: submits.append(
            SimpleNamespace(call=call, redraw=redraw, label=label, lane=lane)
        ),
    )
    monkeypatch.setattr(layer_control.gui_logger, 'toggle', MagicMock())
    session = gui_ctx.session
    session.show_channel('BF')
    session.set_channel_auto_gain('BF', not enabled)
    widget = SimpleNamespace(
        layer='BF',
        ids={'auto_gain': SimpleNamespace(state=state)},
        render_layer_values_from_settings=MagicMock(),
        show_as_shown=MagicMock(),
    )

    layer_control.LayerControl.update_auto_gain(widget)

    (submit,) = submits
    assert submit.lane is gui_ctx.camera_executor
    assert submit.label == 'AUTO_GAIN_BF'
    assert session.channel_auto_gain('BF') is (not enabled), 'it runs on the lane, not here'
    submit.call()
    assert session.channel_auto_gain('BF') is enabled
    submit.redraw()
    widget.render_layer_values_from_settings.assert_called_once_with()
    widget.show_as_shown.assert_called_once_with()
    layer_control.gui_logger.toggle.assert_called_once_with('AUTO_GAIN_BF', enabled)


@pytest.fixture
def client(tmp_path):
    """A REST client over a simulated LS850, homed, showing BF."""
    from fastapi.testclient import TestClient

    from rest.app import build_app
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850',
            BF={'exposure_ms': 2.0, 'gain_db': 1.0, 'auto_gain': False},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(session.scope)
        with TestClient(build_app(session)) as client:
            yield session, client
    finally:
        session.shutdown()


def _lock_read(client, layer):
    answer = client.get('/api/v1/channel_auto_gain_lock', params={'channel': layer})
    assert answer.status_code == 200, answer.json()
    return answer.json()


def test_over_rest_the_lock_a_turning_off_took_is_its_own_read(client):
    """``PUT channel_auto_gain`` answers the setting's read, a bool; what the
    camera reached is ``GET channel_auto_gain_lock``, the lock the turning off
    took, its gain the one stored. It records that turning off, so turning
    auto-gain on again leaves it as it was."""
    session, client = client
    assert _lock_read(client, 'BF') is None

    client.put('/api/v1/channel_auto_gain', json={'channel': 'BF', 'enabled': True})
    put = client.put('/api/v1/channel_auto_gain', json={'channel': 'BF', 'enabled': False})
    lock = _lock_read(client, 'BF')

    assert put.json() is False
    assert lock['state'] in {'CONVERGED', 'MAXED', 'AT_MINIMUM'}
    assert round(lock['gain_db'], 1) == session.get_setting('BF.gain_db') != 1.0
    client.put('/api/v1/channel_auto_gain', json={'channel': 'BF', 'enabled': True})
    assert _lock_read(client, 'BF') == lock


def test_over_rest_a_hidden_layers_turning_off_reads_a_lock_with_no_state(client):
    """A layer the camera does not hold locks nothing: its read answers a lock
    whose state is None, where before any turning off it answered null."""
    _session, client = client
    assert _lock_read(client, LED0) is None

    client.put('/api/v1/channel_auto_gain', json={'channel': LED0, 'enabled': False})

    assert _lock_read(client, LED0)['state'] is None


def test_over_rest_the_lock_of_no_layer_is_refused(client):
    _session, client = client
    answer = client.get('/api/v1/channel_auto_gain_lock', params={'channel': 'Nope'})
    assert answer.json().get('reason') == 'channel_unknown'
