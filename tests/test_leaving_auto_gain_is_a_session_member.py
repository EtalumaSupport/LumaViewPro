# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Leaving auto-gain stores what the lock reached, through one Session member.

The API's lock decides the values a caller leaving auto-gain stores (the
achieved gain, and the achieved exposure floored to the class's usable floor,
``stored_exposure_ms``), but the store write lived in the GUI's toggle
callback, so a script or a REST caller had to re-implement it.
``ScopeSession.set_layer_auto_gain`` is now that write for every caller, and
the GUI's toggle submits it on the camera lane.

The lock results are built the way the API builds them (the stored exposure
by the API's own rule), so these exercise production's decision, not a copy.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import modules.config_helpers as config_helpers
from modules.exceptions import (
    ArgumentRefusedError,
    ArgumentTypeRefusedError,
    HardwareCommandRefusedError,
)
from modules.layer_record import LayerIdentity, LayerRecord
from modules.lumascope_api.imaging import (
    AutoGainConvergence,
    AutoGainLock,
    stored_exposure_after_lock,
)
from modules.scope_session import ScopeSession

_MEMBERS = ('set_layer_auto_gain', '_layers_on_scope')


def _session(settings, lock=None, layers=('BF', 'PC', 'DF', 'Blue', 'Green', 'Red')):
    imaging = SimpleNamespace(lock_auto_gain=MagicMock(return_value=lock))
    scope = SimpleNamespace(
        imaging=imaging,
        layer_identity=LayerIdentity(
            layers=tuple(LayerRecord(i, name, name, (), None) for i, name in enumerate(layers)),
            filterset='',
            source='scopes',
            model='LS850',
        ),
    )
    session = SimpleNamespace(scope=scope, settings=settings, settings_lock=threading.Lock())
    for name in _MEMBERS:
        setattr(session, name, getattr(ScopeSession, name).__get__(session))
    return session


def _lock(layer, state, exposure_ms, gain_db, *, ceiling_ms=200.0):
    floor = config_helpers.get_ag_ae_min_exposure_ms(layer)
    stored = stored_exposure_after_lock(exposure_ms, floor) if exposure_ms is not None else None
    return AutoGainLock(state, exposure_ms, gain_db, floor, ceiling_ms, stored_exposure_ms=stored)


def _settings(layer, exposure_ms=999.0, gain_db=0.0):
    return {layer: {'exposure_ms': exposure_ms, 'gain_db': gain_db, 'auto_gain': True}}


@pytest.mark.parametrize(
    'layer, state, achieved_ms, gain, stored_ms',
    [
        ('BF', AutoGainConvergence.CONVERGED, 5.0, 2.0, 5.0),
        ('BF', AutoGainConvergence.AT_MINIMUM, 0.03, 0.0, 0.1),  # the transmitted floor
        ('PC', AutoGainConvergence.AT_MINIMUM, 0.05, 0.0, 0.1),
        ('DF', AutoGainConvergence.AT_MINIMUM, 0.05, 0.0, 0.1),
        ('Blue', AutoGainConvergence.AT_MINIMUM, 0.4, 3.0, 1.0),  # the fluorescence floor
        ('Lumi', AutoGainConvergence.AT_MINIMUM, 0.5, 0.0, 1.0),
        ('Red', AutoGainConvergence.MAXED, 200.0, 20.0, 200.0),
    ],
)
def test_turning_off_stores_what_the_lock_reached(layer, state, achieved_ms, gain, stored_ms):
    settings = _settings(layer)
    lock = _lock(layer, state, achieved_ms, gain)
    session = _session(settings, lock)

    assert session.set_layer_auto_gain(layer, False) is lock

    assert settings[layer] == {'exposure_ms': stored_ms, 'gain_db': gain, 'auto_gain': False}


def test_the_stored_values_carry_the_settings_resolution():
    settings = _settings('BF')
    session = _session(settings, _lock('BF', AutoGainConvergence.CONVERGED, 5.0049, 2.06))

    session.set_layer_auto_gain('BF', False)

    assert settings['BF']['gain_db'] == 2.1
    assert settings['BF']['exposure_ms'] == 5.0


def test_a_failed_lock_keeps_the_values_and_still_turns_auto_gain_off():
    settings = _settings('BF', exposure_ms=42.0, gain_db=7.0)
    session = _session(settings, _lock('BF', AutoGainConvergence.FAILED, None, None))

    session.set_layer_auto_gain('BF', False)

    assert settings['BF'] == {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': False}


@pytest.mark.parametrize(
    'achieved_ms, gain, expected',
    [
        # An exposure the camera never reported keeps the stored one; the
        # valid gain beside it still lands.
        (0.0, 3.0, {'exposure_ms': 42.0, 'gain_db': 3.0}),
        # And the other way round.
        (5.0, -1.0, {'exposure_ms': 5.0, 'gain_db': 7.0}),
    ],
)
def test_a_value_the_camera_did_not_report_keeps_the_stored_one(achieved_ms, gain, expected):
    settings = _settings('BF', exposure_ms=42.0, gain_db=7.0)
    session = _session(settings, _lock('BF', AutoGainConvergence.CONVERGED, achieved_ms, gain))

    session.set_layer_auto_gain('BF', False)

    assert settings['BF'] == {**expected, 'auto_gain': False}


def test_turning_off_with_no_arm_standing_stores_only_the_preference():
    settings = _settings('BF', exposure_ms=42.0, gain_db=7.0)
    session = _session(settings, AutoGainLock(state=None))

    session.set_layer_auto_gain('BF', False)

    assert settings['BF'] == {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': False}


def test_turning_on_stores_only_the_preference_and_locks_nothing():
    settings = {'BF': {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': False}}
    session = _session(settings)

    assert session.set_layer_auto_gain('BF', True) is None

    assert settings['BF'] == {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}
    session.scope.imaging.lock_auto_gain.assert_not_called()


@pytest.mark.parametrize(
    'layer, enabled, layers, refused, reason',
    [
        (
            'UV',
            False,
            ('BF',),
            ArgumentRefusedError,
            'layer_unknown',
        ),  # not a layer of this release
        ('Blue', True, ('BF', 'Green'), HardwareCommandRefusedError, 'axis_absent'),  # no Blue here
        ('BF', 'off', ('BF',), ArgumentTypeRefusedError, 'wrong_argument_type'),  # not a bool
    ],
)
def test_a_refused_request_changes_nothing(layer, enabled, layers, refused, reason):
    settings = {'BF': {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
    session = _session(settings, layers=layers)
    before = {name: dict(values) for name, values in settings.items()}

    with pytest.raises(refused) as raised:
        session.set_layer_auto_gain(layer, enabled)
    assert getattr(raised.value, 'reason', None) == reason

    assert settings == before
    session.scope.imaging.lock_auto_gain.assert_not_called()


def test_turning_off_a_layer_this_scope_lacks_is_admitted():
    settings = _settings('Blue')
    session = _session(settings, AutoGainLock(state=None), layers=('BF',))

    session.set_layer_auto_gain('Blue', False)

    assert settings['Blue']['auto_gain'] is False


# ---------------------------------------------------------------------------
# The GUI's toggle submits the member on the camera lane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('state, enabled', [('down', True), ('normal', False)])
def test_the_toggle_submits_the_member_on_the_camera_lane_and_redraws(monkeypatch, state, enabled):
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
    ctx = SimpleNamespace(session=MagicMock(), camera_executor=object())
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    widget = SimpleNamespace(
        layer='BF',
        ids={'auto_gain': SimpleNamespace(state=state)},
        render_layer_values_from_settings=MagicMock(),
        apply_settings=MagicMock(),
    )

    layer_control.LayerControl.update_auto_gain(widget)

    (submit,) = submits
    assert submit.lane is ctx.camera_executor
    assert submit.label == 'AUTO_GAIN_BF'
    ctx.session.set_layer_auto_gain.assert_not_called()  # it runs on the lane, not here
    submit.call()
    ctx.session.set_layer_auto_gain.assert_called_once_with('BF', enabled)
    submit.redraw()
    widget.render_layer_values_from_settings.assert_called_once_with()
    widget.apply_settings.assert_called_once_with()
    layer_control.gui_logger.toggle.assert_called_once_with('AUTO_GAIN_BF', enabled)
