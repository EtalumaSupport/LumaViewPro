# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope whose model has no motor board is complete without one.

An LS620 or LS560 has no focus drive, stage or turret, so no motor board:
the catalogue says so, and the scope gets the null motor driver. That is
not a board that is missing. Such a scope starts without a home and
without a "Motor Not Connected" popup, and a run's connection check does
not refuse it. A scope whose model has motors and whose board is missing
is still refused, as before.

Both simulator tiers build a model's axes from the one catalogue answer,
so the fast tier cannot give a manual scope a Z axis that hides all this.
"""

import sys

import pytest

import modules.lumascope_api._lumascope as lumascope_module
import modules.notification_center as notification_center

from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from modules.layer_record import load_scope_models, model_axes
from modules.lumascope_api import Lumascope
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

MODELS = load_scope_models()
MANUAL = sorted(model for model in MODELS if not model_axes(MODELS, model))

firmware_runs_here = sys.platform == 'darwin' or sys.platform.startswith('linux')
TIERS = [
    'fast',
    pytest.param(
        'firmware',
        marks=pytest.mark.skipif(
            not firmware_runs_here,
            reason='the firmware-backed simulator runs on macOS and Linux only',
        ),
    ),
]


def test_the_catalogue_has_manual_scopes():
    # Without one the tests below would pass on an empty parametrization.
    assert {'LS620', 'LS560'} <= set(MANUAL)


@pytest.mark.parametrize('model', sorted(MODELS))
def test_the_fast_tier_reports_every_catalogue_models_axes(model):
    scope = Lumascope(
        simulate=True,
        sim_model=model,
        sim_tier='fast',
        warn_pre_release=False,
        register_metrics=False,
        register_atexit=False,
    )
    try:
        assert set(scope.capabilities.axes) == model_axes(MODELS, model)
        if not model_axes(MODELS, model):
            assert isinstance(scope._motion_driver, NullMotionBoard)
    finally:
        scope.disconnect()


@pytest.fixture
def errors(monkeypatch):
    said = []
    monkeypatch.setattr(
        notification_center.notifications,
        'error',
        lambda category, title, message, **kwargs: said.append(title),
    )
    return said


def _session(model: str, tier: str) -> ScopeSession:
    return ScopeSession.create(
        complete_settings(simulator_tier=tier, microscope=model),
        simulate=True,
        warn_pre_release=False,
    )


@pytest.mark.parametrize('tier', TIERS)
@pytest.mark.parametrize('model', MANUAL)
def test_a_manual_scope_starts_without_a_home_and_is_admitted(model, tier, errors):
    session = _session(model, tier)
    try:
        scope = session.scope
        assert scope.motion_expected is False
        assert not scope.motor_connected
        assert scope.are_all_connected()

        homes = []
        session.start_application_session(
            home_fn=lambda axis: homes.append(axis) or True,
            turret_fn=lambda position: homes.append(('T', position)),
        )
        assert homes == []
        assert errors == []
    finally:
        session.shutdown()


@pytest.mark.parametrize('tier', TIERS)
def test_a_motorized_scope_without_its_board_is_still_disconnected(tier):
    # A board that did not come up at boot: the registry answers the null
    # driver, as it does for a manual scope, and only the model tells them apart.
    session = _session('LS850T', tier)
    try:
        scope = session.scope
        assert scope.motion_expected is True
        scope._motion_driver = NullMotionBoard()
        assert not scope.are_all_connected()
    finally:
        session.shutdown()


def test_a_real_scope_asks_the_registry_for_real_motor_drivers_only(monkeypatch):
    asked = []

    def create(name='auto', **kwargs):
        asked.append((name, kwargs))
        return NullMotionBoard()

    monkeypatch.setattr(lumascope_module.motor_registry, 'create', create)
    # No real port is opened in the suite: the LED board is answered null.
    monkeypatch.setattr(
        lumascope_module.led_registry, 'create', lambda name='auto', **kwargs: NullLEDBoard()
    )
    scope = Lumascope(
        simulate=False,
        camera_type='sim',
        warn_pre_release=False,
        register_metrics=False,
        register_atexit=False,
    )
    try:
        assert asked == [('auto', {})]
    finally:
        scope.disconnect()
