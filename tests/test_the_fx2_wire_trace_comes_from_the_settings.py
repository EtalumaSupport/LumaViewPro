# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 LED wire trace is switched by the settings the session prepared.

The trace was switched by three private reads of the settings file at
import, one in the FX2 driver (an upward import from drivers/ into
modules/), one in the illumination API and one in the layer panel. A
driver that reads the application's settings cannot be loaded without the
application, and three reads of one file can disagree with the settings the
session actually runs on. The session now hands the prepared value to the
scope, which hands it to the LED driver when it builds it.
"""

import ast
import pathlib
from unittest.mock import MagicMock, patch

import pytest

import drivers.fx2driver as fx2driver
import modules.lumascope_api._lumascope as lumascope_module
import modules.lumascope_api.illumination as illumination_module
from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_DRIVER = pathlib.Path(__file__).resolve().parent.parent / 'drivers' / 'fx2driver.py'


def test_the_fx2_driver_imports_no_application_module():
    tree = ast.parse(_DRIVER.read_text())
    upward = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        upward += [
            (node.lineno, name)
            for name in names
            if name == 'modules' or name.startswith('modules.')
        ]
    assert not upward, f'drivers/fx2driver.py imports from modules/: {upward}'


@pytest.mark.parametrize('enabled', [True, False])
def test_the_session_hands_the_setting_to_the_led_driver(monkeypatch, tmp_path, enabled):
    asked = []

    def led_create(name='auto', **kwargs):
        asked.append(kwargs)
        return NullLEDBoard(), None

    real_camera_create = lumascope_module.camera_registry.create
    monkeypatch.setattr(lumascope_module.led_registry, 'create_with_fallback', led_create)
    monkeypatch.setattr(
        lumascope_module.motor_registry,
        'create_with_fallback',
        lambda name='auto', **kwargs: (NullMotionBoard(), None),
    )
    # No real camera in the suite: the by-name path builds the simulated one.
    monkeypatch.setattr(
        lumascope_module.camera_registry,
        'create',
        lambda name='auto', **kwargs: real_camera_create('sim', **kwargs),
    )

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), fx2_debug_wire_enabled=enabled),
        warn_pre_release=False,
    )
    try:
        assert asked == [{'debug_wire': enabled}]
    finally:
        session.shutdown()
        session.scope.disconnect()


@pytest.mark.parametrize('enabled', [True, False])
def test_the_fx2_led_driver_traces_only_when_asked(monkeypatch, enabled):
    fake = MagicMock(name='_FX2Connection_fake')
    fake.i2c_write.return_value = 1
    monkeypatch.setattr(fx2driver._FX2Connection, 'get', classmethod(lambda cls: fake))
    led = fx2driver.FX2LEDController(debug_wire=enabled)

    with patch.object(fx2driver, 'logger') as mock_logger:
        led.led_on(0, 100)
        traced = [c for c in mock_logger.info.call_args_list if '[FX2 LED diag]' in str(c.args[0])]

    assert bool(traced) is enabled


@pytest.mark.parametrize('enabled', [True, False])
def test_the_illumination_cache_check_traces_by_the_sessions_setting(tmp_path, enabled):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), fx2_debug_wire_enabled=enabled),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        with patch.object(illumination_module, '_api_log') as mock_log:
            session.scope.illumination.led_on('Blue', 50)
            traced = [
                c for c in mock_log.info.call_args_list if 'led_on cache-check' in str(c.args[0])
            ]
        assert bool(traced) is enabled
    finally:
        session.shutdown()
        session.scope.disconnect()


def test_the_support_report_scope_never_traces(monkeypatch):
    # No real port is opened in the suite: both boards are answered null.
    monkeypatch.setattr(
        lumascope_module.motor_registry,
        'create_with_fallback',
        lambda name='auto', **kwargs: (NullMotionBoard(), None),
    )
    monkeypatch.setattr(
        lumascope_module.led_registry,
        'create_with_fallback',
        lambda name='auto', **kwargs: (NullLEDBoard(), None),
    )

    scope = lumascope_module.Lumascope.create_diagnostic()
    try:
        # The diagnostic scope is built without __init__, so the flag its
        # illumination API reads is set by create_diagnostic itself.
        assert scope._fx2_debug_wire is False
    finally:
        scope.disconnect()
