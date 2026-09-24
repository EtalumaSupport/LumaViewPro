# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope's motor board comes in two tiers, and the settings pick.

What is pinned here: the firmware tier builds the production driver against
the real firmware and reports the catalogue's axes for every model; a model
with no axes has no motor board; an emulator that does not come up raises
instead of becoming a manual scope; the setting is refused when it names no
tier and resolves to the fast tier only where no runtime is built.
"""

import pathlib
import sys

import pytest
import serial

from drivers.motorboard import MotorBoard
from drivers.null_motorboard import NullMotionBoard
from drivers.registry import DriverNotLiveError
from drivers.sim_wire.backend import SimWireBackend
from drivers.simulated_motorboard import SimulatedMotorBoard
from modules.exceptions import ConfigError
from modules.layer_record import load_scope_models, model_axes
from modules.lumascope_api import Lumascope
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )

MODELS = load_scope_models()
_AXES_OF = {'Focus': 'Z', 'XYStage': 'XY', 'Turret': 'T'}


def _catalogue_axes(model: str) -> set[str]:
    return {axis for flag, axes in _AXES_OF.items() if MODELS[model].get(flag) for axis in axes}


def _scope(**kwargs) -> Lumascope:
    return Lumascope(
        simulate=True,
        warn_pre_release=False,
        register_metrics=False,
        register_atexit=False,
        **kwargs,
    )


@pytest.mark.parametrize('model', sorted(MODELS))
def test_the_firmware_tier_reports_every_catalogue_models_axes(model):
    scope = _scope(sim_tier='firmware', sim_model=model)
    try:
        expected = _catalogue_axes(model)
        assert set(scope.capabilities.axes) == expected
        if expected:
            assert isinstance(scope._motion_driver, MotorBoard)
        else:
            assert isinstance(scope._motion_driver, NullMotionBoard)
    finally:
        scope.disconnect()


def test_the_fast_tier_is_the_default_and_the_python_stand_in():
    scope = _scope(sim_model='LS850T')
    try:
        assert isinstance(scope._motion_driver, SimulatedMotorBoard)
    finally:
        scope.disconnect()


def test_a_tier_that_is_not_one_of_the_two_is_refused():
    with pytest.raises(ValueError, match='sim_tier'):
        _scope(sim_tier='fastest', sim_model='LS850T')


def test_a_model_the_catalogue_lacks_is_refused_not_answered_axis_less():
    with pytest.raises(ConfigError, match="no model 'LS999'"):
        model_axes(MODELS, 'LS999')


def test_an_emulator_that_does_not_come_up_raises_naming_the_driver(monkeypatch):
    # The driver swallows a failed open and reports itself not connected,
    # which the registry's auto path would answer with the null driver.
    def refused_open(self, **kwargs):
        raise serial.SerialException('the emulator did not start')

    monkeypatch.setattr(SimWireBackend, 'open', refused_open)
    with pytest.raises(DriverNotLiveError, match=r"MotorBoard \('rp2040'\)"):
        _scope(sim_tier='firmware', sim_model='LS850T')


class TestTheSessionReadsTheTier:
    def test_the_template_ships_the_firmware_tier(self):
        import json

        template = json.loads(pathlib.Path('data/settings.json').read_text())
        assert template['simulator_tier'] == 'firmware'

    def test_a_setting_that_names_no_tier_is_refused(self):
        with pytest.raises(ConfigError, match="simulator_tier 'turbo'"):
            ScopeSession.create(complete_settings(simulator_tier='turbo'), simulate=True)

    def test_a_settings_dict_without_the_key_is_refused(self):
        settings = complete_settings()
        del settings['simulator_tier']
        with pytest.raises(ConfigError, match="no 'simulator_tier'"):
            ScopeSession.create(settings, simulate=True)

    def test_the_firmware_tier_builds_the_production_driver(self):
        session = ScopeSession.create(
            complete_settings(simulator_tier='firmware', microscope='LS850T'), simulate=True
        )
        try:
            assert isinstance(session.scope._motion_driver, MotorBoard)
            assert set(session.scope.capabilities.axes) == {'X', 'Y', 'Z', 'T'}
        finally:
            session.shutdown()

    def test_a_platform_with_no_runtime_runs_the_fast_tier_and_says_so(self, monkeypatch):
        import modules.scope_session as scope_session_module

        # The suite's lvp_logger is a mock; its calls are the record.
        scope_session_module.logger.warning.reset_mock()
        monkeypatch.setattr(sys, 'platform', 'win32')
        tier = ScopeSession._simulator_tier(complete_settings(simulator_tier='firmware'))
        assert tier == 'fast'
        said = [str(c.args[0]) for c in scope_session_module.logger.warning.call_args_list]
        assert any('no MicroPython runtime' in s for s in said), said

    def test_a_linux_machine_that_has_not_built_the_runtime_runs_the_fast_tier(self, monkeypatch):
        # The Linux runtime is built where it runs, not committed: a Linux
        # developer who never ran the build script must still get a scope.
        import platform

        import drivers.sim_wire.backend as backend

        monkeypatch.setattr(sys, 'platform', 'linux')
        monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
        monkeypatch.setattr(backend, '_PACKAGE', pathlib.Path('/nonexistent'))
        tier = ScopeSession._simulator_tier(complete_settings(simulator_tier='firmware'))
        assert tier == 'fast'

    def test_a_platform_with_a_runtime_keeps_the_firmware_tier(self):
        assert (
            ScopeSession._simulator_tier(complete_settings(simulator_tier='firmware')) == 'firmware'
        )
