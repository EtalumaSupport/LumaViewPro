# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for the GUI-independent scope API modules:
- modules/config_helpers.py
- modules/lumascope_api.py executor-backed command API
  (scope.illumination.led_on, scope.motion.move_absolute, etc.)
- modules/scope_session.py

Uses mock objects + Lumascope(simulate=True) -- no hardware or Kivy needed.
"""

import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tests.settings_fixtures import complete_settings

# Heavy deps are mocked by tests/conftest.py at module-import time.

import modules.config_helpers as config_helpers
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import SequentialIOExecutor
from tests.scope_fakes import build_scope, swap_lanes


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
        'illumination_ma': 50.123456,
        'gain_db': 1.23456,
        'auto_gain': False,
        'exposure_ms': 10.56789,
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
                'illumination_ma': 0,
                'frequency': 1,
            }
        settings[layer] = s

    settings['protocol'] = {
        'autogain': {
            'enabled': True,
            'max_duration_seconds': 30,
            'target_mean': 128,
        },
        'labware': '96 well microplate',
    }
    settings['objective_id'] = '4x Oly'
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
    scope.motion.move_absolute = MagicMock()
    scope.motion.move_relative = MagicMock()
    scope.motion.home = MagicMock()
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

    def put(self, task, return_future=False, *, override=None):
        self.submitted.append(task)
        return super().put(task, return_future=return_future, override=override)


def _make_real_scope_with_recording_executors(led=True, motor=True):
    """Build a real `Lumascope(simulate=True)` with real executors running.

    Set led=False or motor=False to install Null* boards (mimics
    "controller not present"); the API methods early-return in that case.

    The caller owns shutdown: `scope.disconnect()` plus `shutdown()` on
    each executor, or the worker threads outlive the test.
    """
    from tests.scope_fakes import record_turret_answer

    scope = record_turret_answer(build_scope(simulate=True))
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
    swap_lanes(scope, io=io_ex, camera=cam_ex)
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

        The stim brightness slider controls stim_config['illumination_ma']
        directly -- it is NOT force-synced to the layer's imaging illumination.
        Both carry the _ma suffix because both are LED current in milliamps;
        they are the same unit on independent controls, not the same value.
        """
        settings = _make_settings(with_stim=True)
        # Set stim illumination to a different value than layer illumination
        for layer in settings:
            if isinstance(settings[layer], dict) and 'stim_config' in settings[layer]:
                settings[layer]['stim_config']['illumination_ma'] = 200
        configs = config_helpers.get_layer_configs(settings)
        for cfg in configs.values():
            assert cfg['stim_config']['illumination_ma'] == 200
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
        assert obj_id == '4x Oly'
        assert obj['magnification'] == 4
        helper.get_objective_info.assert_called_once_with(objective_id='4x Oly')


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
    def test_an_expected_motor_board_that_is_absent_is_refused(self):
        # The model has a motor controller and none answers: there is no
        # position, and the origin would be recorded as if it were one.
        from modules.exceptions import HardwareCommandRefusedError

        scope = MagicMock()
        type(scope).motor_connected = PropertyMock(return_value=False)
        type(scope).motion_expected = PropertyMock(return_value=True)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            config_helpers.get_current_plate_position(
                scope,
                _make_settings(),
                MagicMock(),
                MagicMock(),
            )
        assert refused.value.reason == 'not_connected'

    def test_a_manual_scope_still_answers_the_origin(self):
        # A scope with no motor controller by design: what its steps record
        # in place of a position is decided elsewhere, and until then this
        # answer is unchanged.
        scope = MagicMock()
        type(scope).motor_connected = PropertyMock(return_value=False)
        type(scope).motion_expected = PropertyMock(return_value=False)
        result = config_helpers.get_current_plate_position(
            scope,
            _make_settings(),
            MagicMock(),
            MagicMock(),
        )
        assert result == {'x': 0, 'y': 0, 'z': 0}

    def test_an_unknown_plate_is_refused_not_answered_in_stage_coordinates(self):
        from modules.exceptions import ConfigError
        from modules.labware_loader import WellPlateLoader

        settings = _make_settings()
        settings['protocol'] = {'labware': 'nonexistent'}
        transformer = MagicMock()
        with pytest.raises(ConfigError, match="unknown labware 'nonexistent'"):
            config_helpers.get_current_plate_position(
                _make_mock_scope(),
                settings,
                transformer,
                WellPlateLoader(),
            )
        transformer.stage_to_plate.assert_not_called()

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
            mock_metrics.assert_called_once_with(path=expected_path, collect_open_files=False)


# ===========================================================================
# Lumascope executor-backed command API tests (LAYER-A')
# ===========================================================================


class TestLumascopeLedAPI:
    def test_led_on_blocks_until_the_write_lands(self):
        # led_on absorbed the blocking tier: it submits and does not return
        # until the worker has run the body, so the state is readable the
        # moment it returns rather than eventually. Reading it here is the
        # assertion -- a dispatcher that submitted without waiting would
        # find the channel still dark.
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.illumination.led_on(channel=1, illumination_ma=75)
        assert len(io_ex.submitted) == 1
        color = scope.illumination.ch2color(1)
        assert scope.illumination.get_led_state(color)['illumination_ma'] == 75.0

    def test_led_on_skips_when_no_led(self):
        # Nothing is queued AND nothing is recorded as lit. The second half
        # is the one with teeth: the body's own `if not self._driver` guard
        # cannot catch a Null board (it is truthy), so without the dispatch
        # guard the command would no-op at the driver while the state cache
        # went on claiming the channel was on.
        scope, io_ex, _ = _make_real_scope_with_recording_executors(led=False)
        scope.illumination.led_on(0, 50)
        assert io_ex.submitted == []
        lit = [c for c, s in scope.illumination.get_led_states().items() if s.get('enabled')]
        assert lit == []

    def test_led_off_blocks_until_the_write_lands(self):
        scope, io_ex, _ = _make_real_scope_with_recording_executors()
        scope.illumination.led_on(channel=1, illumination_ma=75)
        scope.illumination.led_off(1)
        assert len(io_ex.submitted) == 2
        color = scope.illumination.ch2color(1)
        assert scope.illumination.get_led_state(color)['enabled'] is False

    @pytest.mark.parametrize(
        'turn_off',
        [lambda ill: ill.led_off(0), lambda ill: ill.leds_off()],
        ids=['led_off', 'leds_off'],
    )
    def test_led_off_and_leds_off_skip_when_no_led(self, turn_off):
        # No LED controller: nothing is queued and nothing reaches a board.
        scope, io_ex, _ = _make_real_scope_with_recording_executors(led=False)
        turn_off(scope.illumination)
        assert io_ex.submitted == []

    def test_unregistered_io_executor_runs_the_body_directly(self):
        """With no executor registered there is nothing to submit to, so the
        body runs on the calling thread instead of raising. A bare
        Lumascope() in a script or an example has no executors and must
        still drive hardware."""
        scope = build_scope(simulate=True)
        try:
            scope.illumination.led_on(channel=0, illumination_ma=30)
            color = scope.illumination.ch2color(0)
            assert scope.illumination.get_led_state(color)['illumination_ma'] == 30.0
            scope.illumination.leds_off()
            assert scope.illumination.get_led_state(color)['illumination_ma'] in (None, 0.0)
        finally:
            scope.disconnect()


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
        scope, _io_ex, _cam_ex = _make_real_scope_with_recording_executors()
        defaults = {
            'settings': _make_settings(),
            'scope': scope,
            'executor_bundle': MagicMock(),
        }
        defaults.update(kwargs)
        return ScopeSession(**defaults)

    def test_a_simulated_create_releases_camera_start_gate(self):
        # connect() leaves the camera configured but NOT grabbing (the
        # start gate); the headless factory is the whole bring-up for the
        # sessions it builds, so it must release the gate itself -- without
        # this, every headless capture times out with no error naming the
        # closed gate.
        session = ScopeSession.create(complete_settings(**_make_settings()), simulate=True)
        try:
            assert session.scope._camera_driver.is_grabbing()
        finally:
            session.shutdown()

    def test_init_stores_all_fields(self):
        settings = _make_settings()
        scope, io, cam = _make_real_scope_with_recording_executors()
        session = ScopeSession(
            settings=settings,
            scope=scope,
            executor_bundle=MagicMock(),
        )
        assert session.settings is settings
        assert session.scope is scope
        assert session.io_executor is io
        assert session.camera_executor is cam
        assert session.source_path == scope.source_path
        assert session.is_protocol_running is False

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

    def test_the_current_objective_is_the_runtime_states(self):
        # The answer is the runtime state's, not the settings dict's: on
        # this turret scope, the assignment of the slot in the light path.
        from tests.scope_fakes import home_sim_scope

        session = self._make_session()
        session.scope.runtime_state.set_turret_config({1: '10x Oly', 2: None, 3: None, 4: None})
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(1)
        obj_id, obj = session.scope.runtime_state.resolve_current_objective()
        assert obj_id == '10x Oly'
        assert obj == session.scope.runtime_state.get_objective_info('10x Oly')

    def test_the_current_objective_raises_when_nothing_is_known(self):
        from modules.exceptions import ObjectiveUnknownError

        session = self._make_session()
        with pytest.raises(ObjectiveUnknownError):
            session.scope.runtime_state.resolve_current_objective()

    def test_protocol_running_derives_from_the_claim(self):
        session = self._make_session()
        assert session.is_protocol_running is False
        held = session.activity_claim.try_claim('protocol')
        assert held
        assert session.is_protocol_running is True
        held.release()
        assert session.is_protocol_running is False
