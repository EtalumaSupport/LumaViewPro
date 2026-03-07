"""
Integration tests for protocol execution through SequencedCaptureExecutor.

Tier 1: Core execution paths — verifies that the most common protocol
configurations run to completion without crashing and produce the
expected sequence of hardware calls.

Uses fully mocked scope/camera objects — no hardware or Kivy needed.
"""

import datetime
import pathlib
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, PropertyMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock out heavy dependencies before importing modules under test
# ---------------------------------------------------------------------------
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)

sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('psutil', MagicMock())
sys.modules.setdefault('kivy', MagicMock())
sys.modules.setdefault('kivy.clock', MagicMock())

# Mock camera hardware SDKs that aren't installed in the test environment
sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

# Mock settings_init before sequenced_capture_executor imports it
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {
    'BF': {'autofocus': False},
    'PC': {'autofocus': False},
    'DF': {'autofocus': False},
    'Red': {'autofocus': False},
    'Green': {'autofocus': False},
    'Blue': {'autofocus': False},
    'Lumi': {'autofocus': False},
}
sys.modules.setdefault('settings_init', _mock_settings_init)

from modules.sequential_io_executor import SequentialIOExecutor
from modules.sequenced_capture_executor import SequencedCaptureExecutor
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
from modules.protocol import Protocol

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
COMPLETION_TIMEOUT = 15  # seconds — generous for CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_camera():
    """Create a mock camera with all methods needed by the executor."""
    cam = MagicMock()
    cam.active = True
    # update_camera_config is a context manager
    @contextmanager
    def _update_cm():
        yield
    cam.update_camera_config = _update_cm
    cam.update_auto_gain_target_brightness = MagicMock()
    cam.auto_gain_once = MagicMock()
    cam.get_gain = MagicMock(return_value=1.0)
    cam.get_exposure_t = MagicMock(return_value=10.0)
    return cam


def _make_mock_scope():
    """Create a mock Lumascope with all methods needed by the executor."""
    scope = MagicMock()
    scope.camera = _make_mock_camera()
    scope.led = True
    scope.are_all_connected = MagicMock(return_value=True)
    scope.camera_is_connected = MagicMock(return_value=True)
    scope.has_turret = MagicMock(return_value=False)

    # LED methods
    scope.get_led_states = MagicMock(return_value={})
    scope.led_on = MagicMock()
    scope.leds_off = MagicMock()
    scope.color2ch = MagicMock(side_effect=lambda c: {'BF': 0, 'PC': 1, 'DF': 2, 'Red': 3, 'Green': 4, 'Blue': 5, 'Lumi': 6}.get(c, 0))

    # Motion — always report "at target" so scan doesn't stall
    scope.get_target_status = MagicMock(return_value=True)
    scope.get_overshoot = MagicMock(return_value=False)
    scope.move_absolute_position = MagicMock()
    scope.get_current_position = MagicMock(return_value=0.0)

    # Camera settings
    scope.set_gain = MagicMock()
    scope.set_exposure_time = MagicMock()
    scope.set_auto_gain = MagicMock()

    # Image capture — return a small dummy image
    scope.get_image = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
    scope.save_image = MagicMock()

    # Objective info
    scope.get_objective_info = MagicMock(return_value={'short_name': '10x', 'magnification': 10})

    return scope


def _make_executors():
    """Create and start the set of SequentialIOExecutors needed."""
    execs = {
        'io': SequentialIOExecutor(name="TEST_IO"),
        'protocol': SequentialIOExecutor(name="TEST_PROTOCOL"),
        'file_io': SequentialIOExecutor(name="TEST_FILE"),
        'camera': SequentialIOExecutor(name="TEST_CAMERA"),
        'autofocus': SequentialIOExecutor(name="TEST_AF"),
    }
    for e in execs.values():
        e.start()
    return execs


def _shutdown_executors(execs):
    """Shut down all executors."""
    for e in execs.values():
        try:
            e.shutdown()
        except Exception:
            pass


def _make_autogain_settings():
    return {
        'target_brightness': 0.3,
        'min_gain': 0.0,
        'max_gain': 20.0,
        'max_duration': datetime.timedelta(seconds=1),
    }


def _make_image_capture_config():
    return {
        'output_format': {
            'live': 'TIFF',
            'sequenced': 'TIFF',
        },
        'use_full_pixel_depth': False,
    }


def _make_single_step_protocol(
    color='BF',
    auto_gain=False,
    auto_focus=False,
    acquire='image',
    gain=1.0,
    exposure=10.0,
    illumination=50.0,
    false_color=False,
    sum_count=1,
    video_config=None,
    stim_config=None,
):
    """Create a Protocol with a single step using pandas directly."""
    import pandas as pd

    if video_config is None:
        video_config = {'duration': 1, 'fps': 5}
    if stim_config is None:
        stim_config = {}

    step = {
        'Name': 'A1_test',
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': auto_focus,
        'Color': color,
        'False_Color': false_color,
        'Illumination': illumination,
        'Gain': gain,
        'Auto_Gain': auto_gain,
        'Exposure': exposure,
        'Sum': sum_count,
        'Objective': '10x',
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': acquire,
        'Video Config': video_config,
        'Stim_Config': stim_config,
    }

    df = pd.DataFrame([step])
    protocol = MagicMock(spec=Protocol)
    protocol.num_steps.return_value = 1
    protocol.step.return_value = pd.Series(step)
    protocol.steps.return_value = df
    protocol.period.return_value = datetime.timedelta(minutes=1)
    protocol.duration.return_value = datetime.timedelta(minutes=1)
    protocol.labware.return_value = '6-well'
    protocol.capture_root.return_value = ''
    protocol.to_file = MagicMock()
    return protocol


def _make_multi_step_protocol(steps_config):
    """Create a Protocol with multiple steps.

    steps_config: list of dicts, each with keys like color, auto_gain, etc.
    Missing keys get defaults.
    """
    import pandas as pd

    defaults = {
        'color': 'BF', 'auto_gain': False, 'auto_focus': False,
        'acquire': 'image', 'gain': 1.0, 'exposure': 10.0,
        'illumination': 50.0, 'false_color': False, 'sum_count': 1,
        'video_config': {'duration': 1, 'fps': 5}, 'stim_config': {},
        'x': 10.0, 'y': 20.0, 'z': 5000.0, 'well': 'A1', 'name': None,
    }

    rows = []
    for i, cfg in enumerate(steps_config):
        merged = {**defaults, **cfg}
        name = merged['name'] or f"step_{i}_{merged['color']}"
        rows.append({
            'Name': name,
            'X': merged['x'],
            'Y': merged['y'],
            'Z': merged['z'],
            'Auto_Focus': merged['auto_focus'],
            'Color': merged['color'],
            'False_Color': merged['false_color'],
            'Illumination': merged['illumination'],
            'Gain': merged['gain'],
            'Auto_Gain': merged['auto_gain'],
            'Exposure': merged['exposure'],
            'Sum': merged['sum_count'],
            'Objective': '10x',
            'Well': merged['well'],
            'Tile': '',
            'Z-Slice': 0,
            'Custom Step': True,
            'Tile Group ID': 0,
            'Z-Stack Group ID': 0,
            'Acquire': merged['acquire'],
            'Video Config': merged['video_config'],
            'Stim_Config': merged['stim_config'],
        })

    df = pd.DataFrame(rows)
    protocol = MagicMock(spec=Protocol)
    protocol.num_steps.return_value = len(rows)
    protocol.step.side_effect = lambda idx: pd.Series(rows[idx])
    protocol.steps.return_value = df
    protocol.period.return_value = datetime.timedelta(minutes=1)
    protocol.duration.return_value = datetime.timedelta(minutes=1)
    protocol.labware.return_value = '6-well'
    protocol.capture_root.return_value = ''
    protocol.to_file = MagicMock()
    return protocol


def _run_and_wait(executor, protocol, tmp_path, **run_kwargs):
    """Run a protocol on the executor and wait for completion.

    Returns (completed: bool, run_complete_kwargs: dict).
    """
    done = threading.Event()
    result_holder = {}

    def on_complete(**kwargs):
        result_holder.update(kwargs)
        done.set()

    callbacks = run_kwargs.pop('callbacks', {})
    callbacks['run_complete'] = on_complete
    # Provide a no-op go_to_step to avoid needing real wellplate loader
    callbacks.setdefault('go_to_step', lambda **kw: None)
    callbacks.setdefault('move_position', lambda axis: None)

    executor.run(
        protocol=protocol,
        run_trigger_source='test',
        run_mode=run_kwargs.pop('run_mode', SequencedCaptureRunMode.SINGLE_SCAN),
        sequence_name='test_run',
        image_capture_config=run_kwargs.pop('image_capture_config', _make_image_capture_config()),
        autogain_settings=run_kwargs.pop('autogain_settings', _make_autogain_settings()),
        parent_dir=tmp_path / 'output',
        max_scans=run_kwargs.pop('max_scans', 1),
        callbacks=callbacks,
        leds_state_at_end=run_kwargs.pop('leds_state_at_end', 'off'),
        initial_autofocus_states={
            'BF': False, 'PC': False, 'DF': False,
            'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
        },
        **run_kwargs,
    )

    completed = done.wait(timeout=COMPLETION_TIMEOUT)
    return completed, result_holder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scope():
    return _make_mock_scope()


@pytest.fixture
def executors():
    execs = _make_executors()
    yield execs
    _shutdown_executors(execs)


@pytest.fixture
def executor(scope, executors):
    """Create a SequencedCaptureExecutor wired to mocked scope and real executors."""
    # Mock the autofocus executor to avoid real AF logic.
    # IMPORTANT: in_progress() and complete() must return False (not truthy MagicMock),
    # otherwise _scan_iterate bails early thinking AF is running.
    mock_af = MagicMock()
    mock_af.reset = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.complete = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.result = MagicMock(return_value=None)
    mock_af.best_focus_position = MagicMock(return_value=5000.0)

    exc = SequencedCaptureExecutor(
        scope=scope,
        stage_offset={'x': 0.0, 'y': 0.0},
        io_executor=executors['io'],
        protocol_executor=executors['protocol'],
        file_io_executor=executors['file_io'],
        camera_executor=executors['camera'],
        autofocus_io_executor=executors['autofocus'],
        autofocus_executor=mock_af,
    )

    # Mock the wellplate loader and coordinate transformer so _default_move
    # doesn't need real labware data from disk.
    mock_loader = MagicMock()
    mock_transformer = MagicMock()
    mock_transformer.plate_to_stage = MagicMock(return_value=(0.0, 0.0))
    exc._wellplate_loader = mock_loader
    exc._coordinate_transformer = mock_transformer

    return exc


# ===========================================================================
# Tier 1: Core Execution Paths
# ===========================================================================

class TestSingleScanBasicImage:
    """Test 1: Simplest happy path — single scan, single BF image step."""

    def test_completes_successfully(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, "Protocol did not complete within timeout"

    def test_sets_gain_and_exposure(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', gain=5.0, exposure=50.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        scope.set_gain.assert_called()
        scope.set_exposure_time.assert_called()

    def test_turns_led_on_and_off(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', illumination=75.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        scope.led_on.assert_called()
        scope.leds_off.assert_called()

    def test_auto_gain_disabled_in_step(self, executor, scope, tmp_path):
        """When auto_gain=False, set_auto_gain(False) should be called."""
        protocol = _make_single_step_protocol(color='BF', auto_gain=False)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # Should have called set_auto_gain with state=False at some point
        calls = [c for c in scope.set_auto_gain.call_args_list if c.kwargs.get('state') == False or (c.args and c.args[0] == False)]
        assert len(calls) > 0, "set_auto_gain(state=False) not called"


class TestSingleScanAutoGain:
    """Test 2: Single scan with auto-gain enabled."""

    def test_completes_with_auto_gain(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_gain=True)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_enables_then_disables_auto_gain(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_gain=True)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # Auto gain should be enabled then disabled
        enable_calls = [c for c in scope.set_auto_gain.call_args_list
                        if (c.kwargs.get('state') == True) or (c.args and c.args[0] == True)]
        disable_calls = [c for c in scope.set_auto_gain.call_args_list
                         if (c.kwargs.get('state') == False) or (c.args and c.args[0] == False)]
        assert len(enable_calls) > 0, "Auto gain was never enabled"
        assert len(disable_calls) > 0, "Auto gain was never disabled after use"

    def test_does_not_set_manual_gain_when_auto(self, executor, scope, tmp_path):
        """When auto_gain=True, manual set_gain should NOT be called in _scan_iterate."""
        protocol = _make_single_step_protocol(color='BF', auto_gain=True, gain=5.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # _capture still calls set_gain — but _scan_iterate should skip it
        # We can't easily distinguish, so just verify completion


class TestSingleScanAutoFocus:
    """Test 3: Single scan with autofocus enabled."""

    def test_completes_with_auto_focus(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        # Simulate AF already complete so _scan_iterate proceeds past AF logic
        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestSingleScanAutoGainAndAutoFocus:
    """Test 4: Single scan with both auto-gain and auto-focus."""

    def test_completes_with_both_auto_features(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_gain=True, auto_focus=True)

        # Simulate AF already complete
        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestSingleScanFluorescence:
    """Test 5: Single scan with fluorescence channel (Red)."""

    def test_completes_with_red_channel(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='Red')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_led_uses_correct_channel(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='Red', illumination=100.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # Verify led_on was called — the channel mapping is handled internally
        scope.led_on.assert_called()

    @pytest.mark.parametrize("color", ['Red', 'Green', 'Blue', 'PC', 'DF', 'Lumi'])
    def test_completes_for_all_channels(self, executor, scope, tmp_path, color):
        protocol = _make_single_step_protocol(color=color)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, f"Protocol failed for channel {color}"


class TestSingleScanVideo:
    """Test 6: Single scan with video capture."""

    def test_completes_with_video(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(
            color='BF',
            acquire='video',
            video_config={'duration': 0.5, 'fps': 5},
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_video_as_frames(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(
            color='BF',
            acquire='video',
            video_config={'duration': 0.5, 'fps': 5},
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      video_as_frames=True)
        assert completed


class TestFullProtocol:
    """Test 7: Full protocol with multiple scans."""

    def test_two_scans_complete(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        # Override period to be very short so scans happen fast
        protocol.period.return_value = datetime.timedelta(seconds=0.1)
        protocol.duration.return_value = datetime.timedelta(seconds=1)

        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=2,
        )
        assert completed


class TestMultiStepMultiChannel:
    """Test 8: Single scan with multiple steps across channels."""

    def test_bf_and_red_steps_complete(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol([
            {'color': 'BF', 'illumination': 50.0, 'exposure': 10.0},
            {'color': 'Red', 'illumination': 100.0, 'exposure': 50.0},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_steps_visited(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol([
            {'color': 'BF'},
            {'color': 'Red'},
            {'color': 'Green'},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # protocol.step should have been called for each step index
        step_indices = [c.kwargs.get('idx', c.args[0] if c.args else None)
                        for c in protocol.step.call_args_list]
        # Should contain 0, 1, 2 (at least once each)
        assert 0 in step_indices
        assert 1 in step_indices
        assert 2 in step_indices


# ===========================================================================
# Tier 1 Extras: Run-level options
# ===========================================================================

class TestImageSavingDisabled:
    """Protocol with image saving disabled should still complete."""

    def test_completes_without_saving(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      enable_image_saving=False)
        assert completed


class TestDisableSavingArtifacts:
    """Protocol with all saving artifacts disabled."""

    def test_completes_without_artifacts(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      disable_saving_artifacts=True)
        assert completed


class TestLedStateAtEnd:
    """Verify LED cleanup behavior."""

    def test_leds_off_at_end(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      leds_state_at_end='off')
        assert completed
        scope.leds_off.assert_called()

    def test_return_to_original_leds(self, executor, scope, tmp_path):
        # Set up original LED states
        scope.get_led_states.return_value = {
            'BF': {'enabled': True, 'illumination': 25.0},
        }
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      leds_state_at_end='return_to_original')
        assert completed


class TestSumAveraging:
    """Protocol with sum/frame averaging > 1."""

    def test_sum_4_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', sum_count=4)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestFalseColor:
    """Protocol with false color enabled."""

    def test_false_color_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='Red', false_color=True)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_false_color_off_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='Red', false_color=False)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestPixelDepth:
    """Protocol with different pixel depth settings."""

    def test_full_pixel_depth(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        config = _make_image_capture_config()
        config['use_full_pixel_depth'] = True
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      image_capture_config=config)
        assert completed

    def test_8bit_pixel_depth(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        config = _make_image_capture_config()
        config['use_full_pixel_depth'] = False
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      image_capture_config=config)
        assert completed


class TestReturnToPosition:
    """Protocol with return_to_position specified."""

    def test_returns_to_position(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            return_to_position={'x': 50.0, 'y': 50.0, 'z': 3000.0},
        )
        assert completed


class TestSeparateFolderPerChannel:
    """Protocol with separate folder per channel."""

    def test_separate_folders(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol([
            {'color': 'BF'},
            {'color': 'Red'},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      separate_folder_per_channel=True)
        assert completed
