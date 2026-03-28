# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for the GUI-independent scope API modules:
- modules/config_helpers.py
- modules/scope_commands.py
- modules/scope_session.py

Uses mock objects — no hardware or Kivy needed.
"""

import datetime
import sys
import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Mock out heavy dependencies before importing modules under test
# ---------------------------------------------------------------------------
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger

sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('psutil', MagicMock())

import modules.config_helpers as config_helpers
import modules.scope_commands as scope_commands
from modules.scope_session import ScopeSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_layer_settings(**overrides):
    """Build a minimal layer settings dict."""
    defaults = {
        'acquire': True,
        'video_config': {'enabled': False},
        'autofocus': False,
        'false_color': [1, 1, 1, 1],
        'ill': 50.123456,
        'gain': 1.23456,
        'auto_gain': False,
        'exp': 10.56789,
        'sum': 1,
        'focus': 0.0,
    }
    defaults.update(overrides)
    return defaults


def _make_settings(layers=None, with_stim=False):
    """Build a minimal settings dict with the standard layers."""
    from modules.common_utils import get_layers
    if layers is None:
        layers = get_layers()

    settings = {}
    for layer in layers:
        s = _make_layer_settings()
        if with_stim:
            s['stim_config'] = {
                'enabled': True,
                'illumination': 0,
                'frequency': 1,
            }
        settings[layer] = s

    settings['protocol'] = {
        'autogain': {
            'enabled': True,
            'max_duration_seconds': 30,
            'target_mean': 128,
        },
        'labware': 'test_plate',
    }
    settings['objective_id'] = '4x'
    settings['stage_offset'] = {'x': 0, 'y': 0}
    settings['live_folder'] = '/tmp'
    return settings


def _make_mock_scope(led_available=True):
    """Build a mock scope object."""
    scope = MagicMock()
    scope.led = led_available
    type(scope).led_connected = PropertyMock(return_value=bool(led_available))
    type(scope).motor_connected = PropertyMock(return_value=True)
    scope.motion = MagicMock()
    scope.motion.driver = True
    scope.leds_off = MagicMock()
    scope.led_on = MagicMock()
    scope.led_off = MagicMock()
    scope.move_absolute_position = MagicMock()
    scope.move_relative_position = MagicMock()
    scope.zhome = MagicMock()
    scope.xyhome = MagicMock()
    scope.thome = MagicMock()
    scope.get_current_position = MagicMock(return_value={'X': 1000, 'Y': 2000, 'Z': 500})
    return scope


def _make_mock_executor():
    """Build a mock SequentialIOExecutor."""
    executor = MagicMock()
    fut = Future()
    fut.set_result(None)
    executor.put = MagicMock(return_value=fut)
    return executor


# ===========================================================================
# config_helpers tests
# ===========================================================================

class TestGetLayerConfigs:
    def test_returns_all_layers(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings)
        from modules.common_utils import get_layers
        assert set(configs.keys()) == set(get_layers())

    def test_specific_layers_filter(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings, specific_layers=['BF', 'Red'])
        assert set(configs.keys()) == {'BF', 'Red'}

    def test_illumination_rounded(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings)
        from modules.common_utils import max_decimal_precision
        precision = max_decimal_precision('illumination')
        for cfg in configs.values():
            # Value should be rounded to the expected precision
            assert cfg['illumination'] == round(50.123456, precision)

    def test_gain_rounded(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings)
        from modules.common_utils import max_decimal_precision
        precision = max_decimal_precision('gain')
        for cfg in configs.values():
            assert cfg['gain'] == round(1.23456, precision)

    def test_exposure_rounded(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings)
        from modules.common_utils import max_decimal_precision
        precision = max_decimal_precision('exposure')
        for cfg in configs.values():
            assert cfg['exposure'] == round(10.56789, precision)

    def test_stim_config_none_when_absent(self):
        settings = _make_settings(with_stim=False)
        configs = config_helpers.get_layer_configs(settings)
        for cfg in configs.values():
            assert cfg['stim_config'] is None

    def test_stim_config_illumination_synced(self):
        settings = _make_settings(with_stim=True)
        configs = config_helpers.get_layer_configs(settings)
        for cfg in configs.values():
            assert cfg['stim_config']['illumination'] == cfg['illumination']

    def test_auto_gain_bool_conversion(self):
        settings = _make_settings()
        settings['BF']['auto_gain'] = 'True'
        configs = config_helpers.get_layer_configs(settings, specific_layers=['BF'])
        assert configs['BF']['auto_gain'] is True

    def test_empty_specific_layers(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings, specific_layers=[])
        assert configs == {}

    def test_config_keys(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings, specific_layers=['BF'])
        expected_keys = {
            'acquire', 'video_config', 'stim_config', 'autofocus',
            'false_color', 'illumination', 'gain', 'auto_gain',
            'exposure', 'sum', 'focus',
        }
        assert set(configs['BF'].keys()) == expected_keys


class TestGetStimConfigs:
    def test_returns_stim_layers_only(self):
        settings = _make_settings(with_stim=True)
        # Remove stim from BF to verify filtering
        del settings['BF']['stim_config']
        stim = config_helpers.get_stim_configs(settings)
        assert 'BF' not in stim
        assert 'Red' in stim

    def test_no_stim_returns_empty(self):
        settings = _make_settings(with_stim=False)
        stim = config_helpers.get_stim_configs(settings)
        assert stim == {}


class TestGetEnabledStimConfigs:
    def test_filters_disabled(self):
        settings = _make_settings(with_stim=True)
        settings['Red']['stim_config']['enabled'] = False
        enabled = config_helpers.get_enabled_stim_configs(settings)
        assert 'Red' not in enabled
        assert 'BF' in enabled


class TestGetAutoGainSettings:
    def test_converts_seconds_to_timedelta(self):
        settings = _make_settings()
        result = config_helpers.get_auto_gain_settings(settings)
        assert result['max_duration'] == datetime.timedelta(seconds=30)
        assert 'max_duration_seconds' not in result

    def test_preserves_other_keys(self):
        settings = _make_settings()
        result = config_helpers.get_auto_gain_settings(settings)
        assert result['enabled'] is True
        assert result['target_mean'] == 128

    def test_does_not_mutate_settings(self):
        settings = _make_settings()
        config_helpers.get_auto_gain_settings(settings)
        # Original should still have max_duration_seconds
        assert 'max_duration_seconds' in settings['protocol']['autogain']


class TestGetCurrentObjectiveInfo:
    def test_returns_id_and_info(self):
        settings = _make_settings()
        helper = MagicMock()
        helper.get_objective_info.return_value = {'magnification': 4, 'focal_length': 10}
        obj_id, obj = config_helpers.get_current_objective_info(settings, helper)
        assert obj_id == '4x'
        assert obj['magnification'] == 4
        helper.get_objective_info.assert_called_once_with(objective_id='4x')


class TestFindNearestStep:
    def test_returns_minus_one_for_none_protocol(self):
        assert config_helpers.find_nearest_step(0, 0, None) == -1

    def test_returns_minus_one_for_empty_protocol(self):
        proto = MagicMock()
        proto.num_steps.return_value = 0
        assert config_helpers.find_nearest_step(0, 0, proto) == -1

    def test_finds_nearest(self):
        import pandas as pd
        proto = MagicMock()
        proto.num_steps.return_value = 3
        proto.steps.return_value = pd.DataFrame({
            'X': [0, 10, 20],
            'Y': [0, 10, 20],
        })
        assert config_helpers.find_nearest_step(9, 11, proto) == 1
        assert config_helpers.find_nearest_step(0, 0, proto) == 0
        assert config_helpers.find_nearest_step(100, 100, proto) == 2


class TestFocusLog:
    def test_increments_round(self):
        result = config_helpers.focus_log([1, 2], [0.5, 0.7], focus_round=3, source_path='.')
        assert result == 4

    def test_increments_from_zero(self):
        result = config_helpers.focus_log([], [], focus_round=0, source_path='.')
        assert result == 1


class TestBlockWaitForThreads:
    def test_waits_for_all_futures(self):
        futures = []
        for i in range(3):
            f = Future()
            f.set_result(i)
            futures.append(f)
        # Should not raise
        config_helpers.block_wait_for_threads(futures)

    def test_logs_exceptions(self):
        mock_log = MagicMock()
        f = Future()
        f.set_exception(ValueError("test error"))
        with patch.object(config_helpers, 'logger', mock_log):
            config_helpers.block_wait_for_threads([f], log_loc="TEST")
        mock_log.error.assert_called()
        assert "test error" in str(mock_log.error.call_args)


class TestGetCurrentPlatePosition:
    def test_returns_zeros_when_no_driver(self):
        scope = MagicMock()
        scope.motion = None  # No motor board connected
        type(scope).motor_connected = PropertyMock(return_value=False)
        result = config_helpers.get_current_plate_position(
            scope, _make_settings(), MagicMock(), MagicMock(),
        )
        assert result == {'x': 0, 'y': 0, 'z': 0}

    def test_falls_back_on_labware_error(self):
        scope = _make_mock_scope()
        loader = MagicMock()
        loader.get_plate.side_effect = Exception("not found")
        result = config_helpers.get_current_plate_position(
            scope, _make_settings(), MagicMock(), loader,
        )
        # Should return rounded stage positions
        assert result['z'] != 0  # Z=500 from mock


class TestLogSystemMetrics:
    def test_calls_system_metrics(self):
        settings = _make_settings()
        with patch('modules.common_utils.system_metrics') as mock_metrics, \
             patch('modules.common_utils.check_disk_space') as mock_disk, \
             patch('modules.common_utils.get_extra_disks_info') as mock_extra:
            mock_metrics.return_value = {
                'cpu_percent_total': 25.0,
                'ram_available_gb': 8.0,
                'ram_percent_total': 50.0,
                'disk_free_gb': 100.0,
                'disk_used_percent': 30.0,
                'cpu_percent_python': 5.0,
                'ram_used_python_mb': 200.0,
                'ram_used_python_percent': 2.5,
            }
            mock_disk.return_value = 100000  # plenty of space
            mock_extra.return_value = None
            config_helpers.log_system_metrics(settings)
            import pathlib
            expected_path = str(pathlib.Path('/tmp').resolve())
            mock_metrics.assert_called_once_with(path=expected_path)


# ===========================================================================
# scope_commands tests
# ===========================================================================

class TestScopeCommandsLeds:
    def test_leds_off_dispatches(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.leds_off(scope, executor)
        executor.put.assert_called_once()
        task = executor.put.call_args[0][0]
        assert task.action == scope.leds_off

    def test_leds_off_with_callback(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        cb = MagicMock()
        scope_commands.leds_off(scope, executor, callback=cb)
        task = executor.put.call_args[0][0]
        assert task.callback == cb

    def test_leds_off_skips_when_no_led(self):
        scope = _make_mock_scope(led_available=False)
        executor = _make_mock_executor()
        scope_commands.leds_off(scope, executor)
        executor.put.assert_not_called()

    def test_led_on_dispatches(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.led_on(scope, executor, channel=2, illumination=100)
        task = executor.put.call_args[0][0]
        assert task.action == scope.led_on
        assert task.args == (2, 100)

    def test_led_on_with_callback(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        cb = MagicMock()
        scope_commands.led_on(scope, executor, 1, 50, callback=cb, cb_kwargs={'layer': 'Red'})
        task = executor.put.call_args[0][0]
        assert task.callback == cb
        assert task.cb_kwargs == {'layer': 'Red'}

    def test_led_on_skips_when_no_led(self):
        scope = _make_mock_scope(led_available=False)
        executor = _make_mock_executor()
        scope_commands.led_on(scope, executor, 0, 50)
        executor.put.assert_not_called()

    def test_led_off_dispatches(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.led_off(scope, executor, channel=3)
        task = executor.put.call_args[0][0]
        assert task.action == scope.led_off
        assert task.kwargs == {'channel': 3}

    def test_led_off_skips_when_no_led(self):
        scope = _make_mock_scope(led_available=False)
        executor = _make_mock_executor()
        scope_commands.led_off(scope, executor, 0)
        executor.put.assert_not_called()

    def test_led_on_sync_blocks(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.led_on_sync(scope, executor, channel=1, illumination=75)
        executor.put.assert_called_once()
        # Verify return_future=True was passed
        _, kwargs = executor.put.call_args
        assert kwargs.get('return_future') is True

    def test_led_on_sync_skips_when_no_led(self):
        scope = _make_mock_scope(led_available=False)
        executor = _make_mock_executor()
        scope_commands.led_on_sync(scope, executor, 0, 50)
        executor.put.assert_not_called()


class TestScopeCommandsMotion:
    def test_move_absolute_dispatches(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.move_absolute(scope, executor, 'Z', 5000.0)
        task = executor.put.call_args[0][0]
        assert task.action == scope.move_absolute_position
        assert task.kwargs['axis'] == 'Z'
        assert task.kwargs['pos'] == 5000.0

    def test_move_absolute_with_options(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        cb = MagicMock()
        scope_commands.move_absolute(
            scope, executor, 'X', 1000,
            wait_until_complete=True,
            overshoot_enabled=False,
            callback=cb,
            cb_kwargs={'axis': 'X'},
        )
        task = executor.put.call_args[0][0]
        assert task.kwargs['wait_until_complete'] is True
        assert task.kwargs['overshoot_enabled'] is False
        assert task.callback == cb
        assert task.cb_kwargs == {'axis': 'X'}

    def test_move_relative_dispatches(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.move_relative(scope, executor, 'Y', -500.0)
        task = executor.put.call_args[0][0]
        assert task.action == scope.move_relative_position
        assert task.kwargs['axis'] == 'Y'
        assert task.kwargs['um'] == -500.0

    def test_move_home_z(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.move_home(scope, executor, 'Z')
        task = executor.put.call_args[0][0]
        assert task.action == scope.zhome

    def test_move_home_xy(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.move_home(scope, executor, 'xy')  # lowercase should work
        task = executor.put.call_args[0][0]
        assert task.action == scope.xyhome

    def test_move_home_turret(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        scope_commands.move_home(scope, executor, 'T')
        task = executor.put.call_args[0][0]
        assert task.action == scope.thome

    def test_move_home_with_callback(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        cb = MagicMock()
        scope_commands.move_home(scope, executor, 'Z', callback=cb, cb_args=('Z',))
        task = executor.put.call_args[0][0]
        assert task.callback == cb
        assert task.cb_args == ('Z',)

    def test_move_home_unknown_axis(self):
        scope = _make_mock_scope()
        executor = _make_mock_executor()
        mock_log = MagicMock()
        with patch.object(scope_commands, 'logger', mock_log):
            scope_commands.move_home(scope, executor, 'W')
        executor.put.assert_not_called()
        mock_log.warning.assert_called()


# ===========================================================================
# ScopeSession tests
# ===========================================================================

class TestScopeSession:
    def _make_session(self, **kwargs):
        defaults = {
            'settings': _make_settings(),
            'scope': _make_mock_scope(),
            'io_executor': _make_mock_executor(),
            'camera_executor': _make_mock_executor(),
        }
        defaults.update(kwargs)
        return ScopeSession(**defaults)

    def test_init_stores_all_fields(self):
        settings = _make_settings()
        scope = _make_mock_scope()
        io = _make_mock_executor()
        cam = _make_mock_executor()
        session = ScopeSession(
            settings=settings, scope=scope,
            io_executor=io, camera_executor=cam,
            source_path='/test',
        )
        assert session.settings is settings
        assert session.scope is scope
        assert session.io_executor is io
        assert session.camera_executor is cam
        assert session.source_path == '/test'
        assert session.focus_round == 0
        assert not session.protocol_running.is_set()

    def test_get_layer_configs_delegates(self):
        session = self._make_session()
        configs = session.get_layer_configs()
        from modules.common_utils import get_layers
        assert set(configs.keys()) == set(get_layers())

    def test_get_layer_configs_with_filter(self):
        session = self._make_session()
        configs = session.get_layer_configs(specific_layers=['Red'])
        assert set(configs.keys()) == {'Red'}

    def test_get_auto_gain_settings_delegates(self):
        session = self._make_session()
        result = session.get_auto_gain_settings()
        assert 'max_duration' in result
        assert isinstance(result['max_duration'], datetime.timedelta)

    def test_get_current_objective_info_delegates(self):
        helper = MagicMock()
        helper.get_objective_info.return_value = {'magnification': 10}
        session = self._make_session(objective_helper=helper)
        obj_id, obj = session.get_current_objective_info()
        assert obj_id == '4x'
        assert obj['magnification'] == 10

    def test_leds_off_delegates(self):
        session = self._make_session()
        session.leds_off()
        session.io_executor.put.assert_called_once()

    def test_led_on_delegates(self):
        session = self._make_session()
        session.led_on(channel=2, illumination=100)
        session.io_executor.put.assert_called_once()

    def test_led_off_delegates(self):
        session = self._make_session()
        session.led_off(channel=1)
        session.io_executor.put.assert_called_once()

    def test_move_absolute_delegates(self):
        session = self._make_session()
        session.move_absolute('Z', 3000)
        session.io_executor.put.assert_called_once()
        task = session.io_executor.put.call_args[0][0]
        assert task.kwargs['axis'] == 'Z'
        assert task.kwargs['pos'] == 3000

    def test_move_relative_delegates(self):
        session = self._make_session()
        session.move_relative('X', 100)
        session.io_executor.put.assert_called_once()

    def test_move_home_delegates(self):
        session = self._make_session()
        session.move_home('Z')
        session.io_executor.put.assert_called_once()
        task = session.io_executor.put.call_args[0][0]
        assert task.action == session.scope.zhome

    def test_no_led_skips_commands(self):
        session = self._make_session(scope=_make_mock_scope(led_available=False))
        session.leds_off()
        session.led_on(0, 50)
        session.led_off(0)
        session.io_executor.put.assert_not_called()

    def test_protocol_running_event(self):
        session = self._make_session()
        assert not session.protocol_running.is_set()
        session.protocol_running.set()
        assert session.protocol_running.is_set()
        session.protocol_running.clear()
        assert not session.protocol_running.is_set()

    def test_start_executors(self):
        io = MagicMock()
        cam = MagicMock()
        session = self._make_session(io_executor=io, camera_executor=cam)
        session.start_executors()
        io.start.assert_called_once()
        cam.start.assert_called_once()

    def test_shutdown_executors(self):
        io = MagicMock()
        cam = MagicMock()
        session = self._make_session(io_executor=io, camera_executor=cam)
        session.shutdown_executors()
        io.shutdown.assert_called_once()
        cam.shutdown.assert_called_once()
