# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for the GUI-independent scope API modules:
- modules/config_helpers.py
- modules/lumascope_api.py executor-backed command API
  (scope.illumination.led_on_async, scope.move_absolute_async, etc.)
- modules/scope_session.py

Uses mock objects + Lumascope(simulate=True) -- no hardware or Kivy needed.
"""

import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Heavy deps are mocked by tests/conftest.py at module-import time.

import modules.config_helpers as config_helpers
import modules.lumascope_api as lumascope_api
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import SequentialIOExecutor


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
        'ill_ma': 50.123456,
        'gain_db': 1.23456,
        'auto_gain': False,
        'exp_ms': 10.56789,
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
    scope._led_driver = led_available
    type(scope).led_connected = PropertyMock(return_value=bool(led_available))
    type(scope).motor_connected = PropertyMock(return_value=True)
    scope._motion_driver = MagicMock()
    scope._motion_driver.driver = True
    scope.illumination.leds_off = MagicMock()
    scope.illumination.led_on = MagicMock()
    scope.illumination.led_off = MagicMock()
    scope.motion.move_absolute_position = MagicMock()
    scope.motion.move_relative_position = MagicMock()
    scope.motion.zhome = MagicMock()
    scope.motion.home = MagicMock()
    scope.motion.thome = MagicMock()
    scope.motion.get_current_position = MagicMock(return_value={'X': 1000, 'Y': 2000, 'Z': 500})
    return scope


class _RecordingExecutor(SequentialIOExecutor):
    """A real executor that also records what was submitted to it.

    The dispatch tests below assert two things, and only a real executor
    can answer both: that the right callable was bound, and that the work
    actually ran. The binding half matters because the async tiers must
    bind the private ``_impl`` and never the public name -- a task bound to
    the public member would re-enter dispatch from the worker it already
    occupies. A mock executor answers the binding half and silently passes
    the other, since nothing it is handed ever executes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.submitted = []

    def put(self, task, return_future=False):
        self.submitted.append(task)
        return super().put(task, return_future=return_future)


def _make_real_scope_with_recording_executors(led=True, motor=True):
    """Build a real `Lumascope(simulate=True)` with real executors running.

    Set led=False or motor=False to install Null* boards (mimics
    "controller not present"); the API methods early-return in that case.

    The caller owns shutdown: `scope.disconnect()` plus `shutdown()` on
    each executor, or the worker threads outlive the test.
    """
    scope = lumascope_api.Lumascope(simulate=True)
    if not led:
        from drivers.null_ledboard import NullLEDBoard

        scope._led_driver = NullLEDBoard()
    if not motor:
        from drivers.null_motorboard import NullMotionBoard

        scope._motion_driver = NullMotionBoard()
    io_ex = _RecordingExecutor(name='TEST_IO')
    cam_ex = _RecordingExecutor(name='TEST_CAMERA')
    io_ex.start()
    cam_ex.start()
    scope.register_executors(io_executor=io_ex, camera_executor=cam_ex)
    _LIVE_RIGS.append((scope, io_ex, cam_ex))
    return scope, io_ex, cam_ex


# Real executors run real worker threads, so every rig built above has to
# be torn down or the threads outlive the test that made them.
_LIVE_RIGS = []


@pytest.fixture(autouse=True)
def _shutdown_recording_rigs():
    yield
    while _LIVE_RIGS:
        scope, io_ex, cam_ex = _LIVE_RIGS.pop()
        io_ex.shutdown()
        cam_ex.shutdown()
        scope.disconnect()


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
            assert cfg['illumination_ma'] == round(50.123456, precision)

    def test_gain_rounded(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings)
        from modules.common_utils import max_decimal_precision

        precision = max_decimal_precision('gain')
        for cfg in configs.values():
            assert cfg['gain_db'] == round(1.23456, precision)

    def test_exposure_rounded(self):
        settings = _make_settings()
        configs = config_helpers.get_layer_configs(settings)
        from modules.common_utils import max_decimal_precision

        precision = max_decimal_precision('exposure')
        for cfg in configs.values():
            assert cfg['exposure_ms'] == round(10.56789, precision)

    def test_stim_config_none_when_absent(self):
        settings = _make_settings(with_stim=False)
        configs = config_helpers.get_layer_configs(settings)
        for cfg in configs.values():
            assert cfg['stim_config'] is None

    def test_stim_config_illumination_independent(self):
        """Stim illumination is independent from imaging illumination.

        The stim brightness slider controls stim_config['illumination']
        directly -- it is NOT force-synced to the layer's imaging illumination.
        Stim config key stays bare 'illumination' (pre-freeze defer per
        units audit; stim is on its own evolution track).
        """
        settings = _make_settings(with_stim=True)
        # Set stim illumination to a different value than layer illumination
        for layer in settings:
            if isinstance(settings[layer], dict) and 'stim_config' in settings[layer]:
                settings[layer]['stim_config']['illumination'] = 200
        configs = config_helpers.get_layer_configs(settings)
        for cfg in configs.values():
            assert cfg['stim_config']['illumination'] == 200
            # Layer illumination is different (50.123456 rounded)
            assert cfg['illumination_ma'] != 200

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
            'acquire',
            'video_config',
            'stim_config',
            'autofocus',
            'false_color',
            'illumination_ma',
            'gain_db',
            'auto_gain',
            'exposure_ms',
            'sum',
            'focus',
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
        proto.steps.return_value = pd.DataFrame(
            {
                'X': [0, 10, 20],
                'Y': [0, 10, 20],
            }
        )
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


class TestGetCurrentPlatePosition:
    def test_returns_zeros_when_no_driver(self):
        scope = MagicMock()
        scope._motion_driver = None  # No motor board connected
        type(scope).motor_connected = PropertyMock(return_value=False)
        result = config_helpers.get_current_plate_position(
            scope,
            _make_settings(),
            MagicMock(),
            MagicMock(),
        )
        assert result == {'x': 0, 'y': 0, 'z': 0}

    def test_falls_back_on_labware_error(self):
        scope = _make_mock_scope()
        loader = MagicMock()
        loader.get_plate.side_effect = Exception('not found')
        result = config_helpers.get_current_plate_position(
            scope,
            _make_settings(),
            MagicMock(),
            loader,
        )
        # Should return rounded stage positions
        assert result['z'] != 0  # Z=500 from mock

    def test_zonly_scope_missing_xy_does_not_raise(self):
        # A scope with no XY stage reports position without X/Y keys; the
        # plate-coordinate (labware-loaded) branch must tolerate that instead
        # of raising KeyError when authoring/modifying a step or a z-stack.
        scope = _make_mock_scope()
        scope.motion.get_current_position = MagicMock(return_value={'Z': 500})
        transformer = MagicMock()
        transformer.stage_to_plate.return_value = (0, 0)
        loader = MagicMock()  # valid labware -> success branch, not fallback
        result = config_helpers.get_current_plate_position(
            scope,
            _make_settings(),
            transformer,
            loader,
        )
        assert set(result) == {'x', 'y', 'z'}
        assert result['z'] != 0  # Z=500 preserved on a Z-only scope


class TestLogSystemMetrics:
    def test_calls_system_metrics(self):
        settings = _make_settings()
        with (
            patch('modules.common_utils.system_metrics') as mock_metrics,
            patch('modules.common_utils.check_disk_space') as mock_disk,
            patch('modules.common_utils.get_extra_disks_info') as mock_extra,
        ):
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
# Lumascope executor-backed command API tests (LAYER-A')
# ===========================================================================


class TestLumascopeLedAPI:
    def test_leds_off_async_dispatches(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.illumination.leds_off_async()
        assert len(io_ex.submitted) == 1
        task = io_ex.submitted[0]
        assert task.action == scope.illumination._leds_off_impl

    def test_leds_off_async_with_callback(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        cb = MagicMock()
        scope.illumination.leds_off_async(callback=cb)
        task = io_ex.submitted[0]
        assert task.callback == cb

    def test_leds_off_async_skips_when_no_led(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors(led=False)
        scope.illumination.leds_off_async()
        assert io_ex.submitted == []

    def test_led_on_async_dispatches(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.illumination.led_on_async(channel=2, mA=100)
        task = io_ex.submitted[0]
        assert task.action == scope.illumination._led_on_impl
        assert task.args == (2, 100)

    def test_led_on_async_with_callback(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        cb = MagicMock()
        scope.illumination.led_on_async(1, 50, callback=cb, cb_kwargs={'layer': 'Red'})
        task = io_ex.submitted[0]
        assert task.callback == cb
        assert task.cb_kwargs == {'layer': 'Red'}

    def test_led_on_async_skips_when_no_led(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors(led=False)
        scope.illumination.led_on_async(0, 50)
        assert io_ex.submitted == []

    def test_led_off_async_dispatches(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.illumination.led_off_async(channel=3)
        task = io_ex.submitted[0]
        assert task.action == scope.illumination._led_off_impl
        assert task.kwargs == {'channel': 3}

    def test_led_off_async_skips_when_no_led(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors(led=False)
        scope.illumination.led_off_async(0)
        assert io_ex.submitted == []

    def test_led_on_blocks_until_the_write_lands(self):
        # led_on absorbed the blocking tier: it submits and does not return
        # until the worker has run the body, so the state is readable the
        # moment it returns rather than eventually. Reading it here is the
        # assertion -- a dispatcher that submitted without waiting would
        # find the channel still dark.
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.illumination.led_on(channel=1, mA=75)
        assert len(io_ex.submitted) == 1
        color = scope.illumination.ch2color(1)
        assert scope.illumination.get_led_ma(color) == 75.0

    def test_led_on_skips_when_no_led(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors(led=False)
        scope.illumination.led_on(0, 50)
        assert io_ex.submitted == []

    def test_unregistered_io_executor_runs_the_body_directly(self):
        """With no executor registered there is nothing to submit to, so the
        body runs on the calling thread instead of raising. A bare
        Lumascope() in a script or an example has no executors and must
        still drive hardware -- both the blocking and the fire-and-forget
        form."""
        scope = lumascope_api.Lumascope(simulate=True)
        try:
            scope.illumination.led_on_async(channel=0, mA=30)
            color = scope.illumination.ch2color(0)
            assert scope.illumination.get_led_ma(color) == 30.0
            scope.illumination.leds_off_async()
            assert scope.illumination.get_led_ma(color) in (None, 0.0)
        finally:
            scope.disconnect()


class TestLumascopeMotionAPI:
    def test_move_absolute_async_dispatches(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.motion.move_absolute_async('Z', 5000.0)
        task = io_ex.submitted[0]
        assert task.action == scope.motion._move_absolute_position_impl
        assert task.kwargs['axis'] == 'Z'
        assert task.kwargs['pos'] == 5000.0

    def test_move_absolute_async_with_options(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        cb = MagicMock()
        scope.motion.move_absolute_async(
            'X',
            1000,
            wait_until_complete=True,
            overshoot_enabled=False,
            callback=cb,
            cb_kwargs={'axis': 'X'},
        )
        task = io_ex.submitted[0]
        assert task.kwargs['wait_until_complete'] is True
        assert task.kwargs['overshoot_enabled'] is False
        assert task.callback == cb
        assert task.cb_kwargs == {'axis': 'X'}

    def test_move_relative_async_dispatches(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.motion.move_relative_async('Y', -500.0)
        task = io_ex.submitted[0]
        assert task.action == scope.motion._move_relative_position_impl
        assert task.kwargs['axis'] == 'Y'
        assert task.kwargs['um'] == -500.0

    def test_move_home_async_z(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.motion.move_home_async('Z')
        task = io_ex.submitted[0]
        assert task.action == scope.motion._zhome_impl

    def test_move_home_async_all(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.motion.move_home_async('all')  # lowercase should work
        task = io_ex.submitted[0]
        assert task.action == scope.motion._home_impl

    def test_move_home_async_legacy_xy_alias(self):
        """Legacy 'XY' axis label still dispatches to scope.motion.home() so
        existing callers keep working during the rename window."""
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.motion.move_home_async('XY')
        task = io_ex.submitted[0]
        assert task.action == scope.motion._home_impl

    def test_move_home_async_turret(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.motion.move_home_async('T')
        task = io_ex.submitted[0]
        assert task.action == scope.motion._thome_impl

    def test_move_home_async_with_callback(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        cb = MagicMock()
        scope.motion.move_home_async('Z', callback=cb, cb_args=('Z',))
        task = io_ex.submitted[0]
        assert task.callback == cb
        assert task.cb_args == ('Z',)

    def test_move_home_async_unknown_axis(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        # move_home_async body lives on MotionAPI (motion.py) after the
        # stateful relocation; the warning is logged through that module's
        # logger, not _lumascope.py's.
        with patch('modules.lumascope_api.motion.logger') as mock_log:
            scope.motion.move_home_async('W')
        assert io_ex.submitted == []
        mock_log.warning.assert_called()


# ===========================================================================
# ScopeSession tests
# ===========================================================================


class TestScopeSession:
    def _make_session(self, **kwargs):
        """Build a ScopeSession bound to a real Lumascope(simulate=True) +
        mock executors. The session's command-method tests assert on
        `scope`'s registered mock executor (same instance as session.io_executor)
        so call-count assertions work end-to-end through the new API.
        """
        scope, io_ex, cam_ex = _make_real_scope_with_recording_executors()
        defaults = {
            'settings': _make_settings(),
            'scope': scope,
            'io_executor': io_ex,
            'camera_executor': cam_ex,
        }
        defaults.update(kwargs)
        return ScopeSession(**defaults)

    def test_create_headless_releases_camera_start_gate(self):
        # connect() leaves the camera configured but NOT grabbing (the
        # start gate); the headless factory is the whole bring-up for the
        # sessions it builds, so it must release the gate itself -- without
        # this, every headless capture times out with no error naming the
        # closed gate.
        session = ScopeSession.create_headless(settings=_make_settings())
        try:
            assert session.scope._camera_driver.is_grabbing()
        finally:
            session.shutdown_executors()

    def test_init_stores_all_fields(self):
        settings = _make_settings()
        scope, io, cam = _make_real_scope_with_recording_executors()
        session = ScopeSession(
            settings=settings,
            scope=scope,
            io_executor=io,
            camera_executor=cam,
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

    def test_capture_sync_forwards_grab_timeout(self):
        """The sync capture forwarder must pass the content-retry budget
        through to the imaging layer. Without the passthrough, a caller
        opting into dark_floor_check gets first-grab judgment with a 0 s
        retry window -- the transient-dark-frame heal the check promises
        can never run from this surface."""
        from unittest.mock import MagicMock

        session = self._make_session()
        # The imaging sync tier dissolved into capture_and_wait, and the two
        # timeouts merged: what a caller used to pass as grab_timeout_s is
        # now the one timeout_s the merged member carries.
        session.scope.imaging.capture_and_wait = MagicMock(return_value=None)
        session.capture_and_wait_sync(dark_floor_check=True, timeout_s=2.5)
        kwargs = session.scope.imaging.capture_and_wait.call_args.kwargs
        assert kwargs['timeout_s'] == 2.5
        assert kwargs['dark_floor_check'] is True

    def test_get_current_objective_info_delegates(self):
        helper = MagicMock()
        helper.get_objective_info.return_value = {'magnification': 10}
        session = self._make_session(objective_helper=helper)
        obj_id, obj = session.get_current_objective_info()
        assert obj_id == '4x'
        assert obj['magnification'] == 10

    def test_leds_off_delegates(self):
        session = self._make_session()
        session.leds_off_async()
        assert len(session.io_executor.submitted) == 1

    def test_led_on_delegates(self):
        session = self._make_session()
        session.led_on_async(channel=2, mA=100)
        assert len(session.io_executor.submitted) == 1

    def test_led_off_delegates(self):
        session = self._make_session()
        session.led_off_async(channel=1)
        assert len(session.io_executor.submitted) == 1

    def test_move_absolute_delegates(self):
        session = self._make_session()
        session.move_absolute_async('Z', 3000)
        assert len(session.io_executor.submitted) == 1
        task = session.io_executor.submitted[0]
        assert task.kwargs['axis'] == 'Z'
        assert task.kwargs['pos'] == 3000

    def test_move_relative_delegates(self):
        session = self._make_session()
        session.move_relative_async('X', 100)
        assert len(session.io_executor.submitted) == 1

    def test_move_home_delegates(self):
        session = self._make_session()
        session.move_home_async('Z')
        assert len(session.io_executor.submitted) == 1
        task = session.io_executor.submitted[0]
        assert task.action == session.scope.motion._zhome_impl

    def test_no_led_skips_commands(self):
        session = self._make_session(scope=_make_mock_scope(led_available=False))
        session.leds_off_async()
        session.led_on_async(0, 50)
        session.led_off_async(0)
        assert session.io_executor.submitted == []

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
