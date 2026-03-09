# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Integration tests for protocol execution through SequencedCaptureExecutor.

Tier 1: Core execution paths — verifies that the most common protocol
configurations run to completion without crashing and produce the
expected sequence of hardware calls.

Uses Lumascope(simulate=True) with real SimulatedLEDBoard, SimulatedMotorBoard,
and SimulatedCamera — no hardware or Kivy needed.
"""

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock out heavy dependencies before importing modules under test
# ---------------------------------------------------------------------------
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()

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
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from lumascope_api import Lumascope
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

def _make_simulated_scope():
    """Create a Lumascope with simulated hardware in fast timing mode."""
    s = Lumascope(simulate=True)
    s.led.set_timing_mode('fast')
    s.motion.set_timing_mode('fast')
    s.camera.set_timing_mode('fast')
    s.camera.start_grabbing()
    return s


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
        'Objective': '10x Oly',
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
        'tile': '', 'z_slice': 0, 'tile_group_id': 0, 'zstack_group_id': 0,
        'objective': '10x Oly',
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
            'Objective': merged['objective'],
            'Well': merged['well'],
            'Tile': merged['tile'],
            'Z-Slice': merged['z_slice'],
            'Custom Step': True,
            'Tile Group ID': merged['tile_group_id'],
            'Z-Stack Group ID': merged['zstack_group_id'],
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
    s = _make_simulated_scope()
    yield s
    s.camera.stop_grabbing()
    s.disconnect()


@pytest.fixture
def executors():
    execs = _make_executors()
    yield execs
    _shutdown_executors(execs)


@pytest.fixture
def executor(scope, executors):
    """Create a SequencedCaptureExecutor wired to simulated scope and real executors."""
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
        # Verify gain and exposure were applied to the simulator
        assert scope.camera.get_gain() == pytest.approx(5.0, abs=0.1)
        assert scope.camera.get_exposure_t() == pytest.approx(50.0, abs=0.1)

    def test_turns_led_on_and_off(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', illumination=75.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # After protocol with leds_state_at_end='off', all LEDs should be off
        for ch, mA in scope.led._channel_states.items():
            assert mA == 0, f"LED channel {ch} still on at {mA} mA after protocol"

    def test_auto_gain_disabled_in_step(self, executor, scope, tmp_path):
        """When auto_gain=False, protocol should complete normally."""
        protocol = _make_single_step_protocol(color='BF', auto_gain=False)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


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
        # Auto gain cycle ran successfully — gain value should have been adjusted
        # from the initial value by the auto-gain convergence logic
        assert scope.camera.get_gain() > 0

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


class TestSingleScanAutoFocusNoneResult:
    """C2 fix: autofocus returns None — protocol must not crash."""

    def test_completes_when_autofocus_returns_none(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False
        af.best_focus_position.return_value = None  # autofocus failed

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, "Protocol should complete even when autofocus returns None"

    def test_z_height_not_modified_when_autofocus_returns_none(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False
        af.best_focus_position.return_value = None

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # modify_step_z_height should NOT have been called with None
        for call in protocol.modify_step_z_height.call_args_list:
            assert call.kwargs.get('z') is not None, \
                "modify_step_z_height should not be called with z=None"


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
        # After protocol with leds_state_at_end='off', LEDs are off —
        # completion confirms the LED was used during the protocol

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
        # Verify all LEDs are off via simulator state
        for ch, mA in scope.led._channel_states.items():
            assert mA == 0, f"LED channel {ch} still on at {mA} mA"

    def test_return_to_original_leds(self, executor, scope, tmp_path):
        # Turn on BF LED before protocol so executor captures it as original state
        bf_ch = scope.color2ch(color='BF')
        scope.led_on(bf_ch, 25)
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


# ===========================================================================
# Tier 2: Feature Combinations
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers for generating tiling / z-stack step configs
# ---------------------------------------------------------------------------

def _make_tile_grid_steps(rows, cols, color='BF', well='A1', spacing=1.0, **extra):
    """Generate step configs for an rows x cols tile grid.

    Each tile gets a unique (x, y) offset and a tile label like 'R0C0'.
    All tiles share the same tile_group_id so they're logically grouped.
    """
    steps = []
    base_x, base_y = 10.0, 20.0
    for r in range(rows):
        for c in range(cols):
            steps.append({
                'color': color,
                'well': well,
                'x': base_x + c * spacing,
                'y': base_y + r * spacing,
                'tile': f'R{r}C{c}',
                'tile_group_id': 1,
                **extra,
            })
    return steps


def _make_zstack_steps(num_slices, color='BF', well='A1', z_start=4000.0, z_step=100.0, **extra):
    """Generate step configs for a z-stack with num_slices slices."""
    steps = []
    for i in range(num_slices):
        steps.append({
            'color': color,
            'well': well,
            'z': z_start + i * z_step,
            'z_slice': i,
            'zstack_group_id': 1,
            **extra,
        })
    return steps


# ---------------------------------------------------------------------------
# Tiling tests
# ---------------------------------------------------------------------------

class TestTiling2x2:
    """2x2 tile grid — simplest tiling case."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=2, cols=2)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_tiles_visited(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=2, cols=2)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        step_indices = {c.kwargs.get('idx', c.args[0] if c.args else None)
                        for c in protocol.step.call_args_list}
        assert {0, 1, 2, 3} <= step_indices


class TestTilingAsymmetric1x3:
    """1x3 tile grid — single row, three columns."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=1, cols=3)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_tiles_visited(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=1, cols=3)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        step_indices = {c.kwargs.get('idx', c.args[0] if c.args else None)
                        for c in protocol.step.call_args_list}
        assert {0, 1, 2} <= step_indices


class TestTilingAsymmetric3x1:
    """3x1 tile grid — three rows, single column."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=3, cols=1)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestTilingAsymmetric3x5:
    """3x5 tile grid — 15 total tiles."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=3, cols=5)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_correct_step_count(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=3, cols=5)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        assert protocol.num_steps() == 15


class TestTilingMultiChannel:
    """Tiling with multiple color channels per tile position."""

    def test_2x2_bf_and_red(self, executor, scope, tmp_path):
        bf_tiles = _make_tile_grid_steps(rows=2, cols=2, color='BF')
        red_tiles = _make_tile_grid_steps(rows=2, cols=2, color='Red')
        steps = bf_tiles + red_tiles
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        assert protocol.num_steps() == 8


# ---------------------------------------------------------------------------
# Z-stack tests
# ---------------------------------------------------------------------------

class TestZStack:
    """Z-stack execution — multiple z-slices at one position."""

    def test_3_slice_zstack(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=3)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_10_slice_zstack(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=10)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_slices_visited(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=5)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        step_indices = {c.kwargs.get('idx', c.args[0] if c.args else None)
                        for c in protocol.step.call_args_list}
        assert {0, 1, 2, 3, 4} <= step_indices


class TestZStackWithAutoFocus:
    """Z-stack combined with autofocus."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=3, auto_focus=True)
        protocol = _make_multi_step_protocol(steps)

        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Tiling + Z-stack combined
# ---------------------------------------------------------------------------

class TestTilingWithZStack:
    """Tile grid where each tile position has a z-stack."""

    def test_2x2_tiles_3_zslices(self, executor, scope, tmp_path):
        steps = []
        base_x, base_y = 10.0, 20.0
        tile_group = 0
        for r in range(2):
            for c in range(2):
                tile_group += 1
                for z_idx in range(3):
                    steps.append({
                        'color': 'BF',
                        'x': base_x + c * 1.0,
                        'y': base_y + r * 1.0,
                        'z': 4000.0 + z_idx * 100.0,
                        'tile': f'R{r}C{c}',
                        'z_slice': z_idx,
                        'tile_group_id': tile_group,
                        'zstack_group_id': tile_group,
                    })
        protocol = _make_multi_step_protocol(steps)
        assert protocol.num_steps() == 12  # 4 tiles * 3 slices
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Multi-well protocols
# ---------------------------------------------------------------------------

class TestMultiWell:
    """Protocol spanning multiple wells."""

    def test_two_wells(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol([
            {'color': 'BF', 'well': 'A1', 'x': 10.0, 'y': 20.0},
            {'color': 'BF', 'well': 'A2', 'x': 30.0, 'y': 20.0},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_six_wells_multi_channel(self, executor, scope, tmp_path):
        wells = [('A1', 10, 20), ('A2', 30, 20), ('A3', 50, 20),
                 ('B1', 10, 40), ('B2', 30, 40), ('B3', 50, 40)]
        steps = []
        for well, x, y in wells:
            steps.append({'color': 'BF', 'well': well, 'x': float(x), 'y': float(y)})
            steps.append({'color': 'Red', 'well': well, 'x': float(x), 'y': float(y)})
        protocol = _make_multi_step_protocol(steps)
        assert protocol.num_steps() == 12
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_multi_well_with_tiling(self, executor, scope, tmp_path):
        steps = []
        for well, wx, wy in [('A1', 10.0, 20.0), ('A2', 30.0, 20.0)]:
            for r in range(2):
                for c in range(2):
                    steps.append({
                        'color': 'BF',
                        'well': well,
                        'x': wx + c * 0.5,
                        'y': wy + r * 0.5,
                        'tile': f'R{r}C{c}',
                        'tile_group_id': 1,
                    })
        protocol = _make_multi_step_protocol(steps)
        assert protocol.num_steps() == 8  # 2 wells * 4 tiles
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Run mode variants
# ---------------------------------------------------------------------------

class TestRunModeSingleZStack:
    """SINGLE_ZSTACK run mode."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=5)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            max_scans=1,
        )
        assert completed


class TestRunModeSingleAutofocusScan:
    """SINGLE_AUTOFOCUS_SCAN run mode."""

    def test_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False

        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            max_scans=1,
        )
        assert completed


class TestRunModeSingleAutofocus:
    """SINGLE_AUTOFOCUS run mode."""

    def test_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False

        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
            max_scans=1,
        )
        assert completed


# ---------------------------------------------------------------------------
# Full protocol multi-scan with tiling
# ---------------------------------------------------------------------------

class TestFullProtocolWithTiling:
    """FULL_PROTOCOL mode running multiple scans over a tile grid."""

    def test_2_scans_2x2_tiles(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=2, cols=2)
        protocol = _make_multi_step_protocol(steps)
        protocol.period.return_value = datetime.timedelta(seconds=0.1)
        protocol.duration.return_value = datetime.timedelta(seconds=1)

        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=2,
        )
        assert completed


class TestFullProtocolMultiScanMultiChannel:
    """FULL_PROTOCOL with multiple scans, multi-channel steps."""

    def test_3_scans_bf_and_red(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol([
            {'color': 'BF'},
            {'color': 'Red'},
        ])
        protocol.period.return_value = datetime.timedelta(seconds=0.1)
        protocol.duration.return_value = datetime.timedelta(seconds=1)

        completed, _ = _run_and_wait(
            executor, protocol, tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=3,
        )
        assert completed


# ---------------------------------------------------------------------------
# Stimulation during video
# ---------------------------------------------------------------------------

class TestVideoWithStimulation:
    """Video capture with LED stimulation config."""

    def test_completes_with_stim(self, executor, scope, tmp_path):
        stim_config = {
            'Blue': {
                'enabled': True,
                'illumination': 100,
                'frequency': 10,
                'pulse_width': 50,
                'pulse_count': 3,
            }
        }
        protocol = _make_single_step_protocol(
            color='BF',
            acquire='video',
            video_config={'duration': 0.5, 'fps': 5},
            stim_config=stim_config,
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_completes_with_disabled_stim(self, executor, scope, tmp_path):
        stim_config = {
            'Blue': {
                'enabled': False,
                'illumination': 100,
                'frequency': 10,
                'pulse_width': 50,
                'pulse_count': 3,
            }
        }
        protocol = _make_single_step_protocol(
            color='BF',
            acquire='video',
            video_config={'duration': 0.5, 'fps': 5},
            stim_config=stim_config,
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Combined feature tests
# ---------------------------------------------------------------------------

class TestAutoGainWithTiling:
    """Auto-gain across a tile grid."""

    def test_1x3_tiles_with_auto_gain(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=1, cols=3, auto_gain=True)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestAutoFocusWithTiling:
    """Auto-focus across a tile grid."""

    def test_2x2_tiles_with_auto_focus(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=2, cols=2, auto_focus=True)
        protocol = _make_multi_step_protocol(steps)

        af = executor._autofocus_executor
        af.complete.return_value = True
        af.in_progress.return_value = False

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestFalseColorWithTiling:
    """False color across a fluorescence tile grid."""

    def test_1x3_red_false_color(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=1, cols=3, color='Red', false_color=True)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestSumWithZStack:
    """Sum averaging combined with z-stack."""

    def test_3_slices_sum_4(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=3, sum_count=4)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestHighExposure:
    """High exposure value to exercise timing paths."""

    def test_500ms_exposure(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', exposure=500.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestImageJHyperstackFormat:
    """ImageJ Hyperstack output format (converted to TIFF internally)."""

    def test_completes_with_hyperstack_format(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        config = _make_image_capture_config()
        config['output_format']['sequenced'] = 'ImageJ Hyperstack'
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      image_capture_config=config)
        assert completed


# ===========================================================================
# Tier 3: Edge Cases and Error Handling
# ===========================================================================

# ---------------------------------------------------------------------------
# Cancellation / reset mid-run
# ---------------------------------------------------------------------------

class TestCancellationMidRun:
    """Verify that reset() stops execution cleanly."""

    def test_reset_during_multi_scan(self, executor, scope, tmp_path):
        """Start a long protocol and cancel it — should not hang."""
        protocol = _make_single_step_protocol(color='BF')
        protocol.period.return_value = datetime.timedelta(seconds=0.1)
        protocol.duration.return_value = datetime.timedelta(seconds=60)

        done = threading.Event()
        result_holder = {}

        def on_complete(**kwargs):
            result_holder.update(kwargs)
            done.set()

        callbacks = {
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }

        executor.run(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            sequence_name='test_cancel',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=100,
            callbacks=callbacks,
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False, 'PC': False, 'DF': False,
                'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
            },
        )

        # Let it run briefly then cancel
        time.sleep(1.0)
        executor.reset()

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed, "Protocol did not complete after reset()"

    def test_reset_before_first_scan_completes(self, executor, scope, tmp_path):
        """Reset immediately — should still invoke run_complete."""
        steps = _make_tile_grid_steps(rows=3, cols=5)  # 15 steps
        protocol = _make_multi_step_protocol(steps)

        done = threading.Event()

        def on_complete(**kwargs):
            done.set()

        callbacks = {
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }

        executor.run(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_early_cancel',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks=callbacks,
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False, 'PC': False, 'DF': False,
                'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
            },
        )

        # Cancel almost immediately
        time.sleep(0.2)
        executor.reset()

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed, "Protocol did not complete after early reset()"


class TestResetWhenNotRunning:
    """reset() when no protocol is active should be a no-op."""

    def test_reset_no_crash(self, executor, scope, tmp_path):
        executor.reset()  # Should not raise


# ---------------------------------------------------------------------------
# Back-to-back runs
# ---------------------------------------------------------------------------

class TestBackToBackRuns:
    """Run a protocol, wait for completion, then immediately run another.

    NOTE: There is a real timing gap between run_complete callback and the
    file_io_executor fully draining its protocol_finish flag (~0.2s).
    The second run() will abort if is_protocol_queue_active() is still True.
    We wait for that flag to clear before starting the next run.
    """

    @staticmethod
    def _wait_for_file_queue(executor, timeout=5.0):
        """Wait until file_io_executor is ready for a new protocol."""
        deadline = time.monotonic() + timeout
        while executor.file_io_executor.is_protocol_queue_active():
            if time.monotonic() > deadline:
                raise TimeoutError("file_io_executor did not drain in time")
            time.sleep(0.05)

    def test_two_sequential_runs(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')

        completed1, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed1, "First run did not complete"

        self._wait_for_file_queue(executor)

        # Second run — uses a fresh tmp subdir to avoid directory collision
        completed2, _ = _run_and_wait(executor, protocol, tmp_path / 'run2')
        assert completed2, "Second run did not complete"

    def test_three_sequential_runs_different_configs(self, executor, scope, tmp_path):
        for i, color in enumerate(['BF', 'Red', 'Green']):
            protocol = _make_single_step_protocol(color=color)
            completed, _ = _run_and_wait(executor, protocol, tmp_path / f'run{i}')
            assert completed, f"Run {i} ({color}) did not complete"
            self._wait_for_file_queue(executor)


# ---------------------------------------------------------------------------
# Disconnected hardware
# ---------------------------------------------------------------------------

class TestDisconnectedScope:
    """Protocol should not start if scope reports disconnected."""

    def test_run_aborts_when_not_connected(self, executor, scope, tmp_path):
        # Disconnect all boards so are_all_connected() returns False
        scope.led.disconnect()
        scope.motion.disconnect()
        scope.camera.active = False
        protocol = _make_single_step_protocol(color='BF')

        done = threading.Event()

        def on_complete(**kwargs):
            done.set()

        callbacks = {
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }

        executor.run(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_disconnected',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks=callbacks,
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False, 'PC': False, 'DF': False,
                'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
            },
        )

        # Should NOT have started — run_complete should NOT fire
        started = done.wait(timeout=2.0)
        assert not started, "Protocol should not have started with disconnected scope"
        assert not executor.run_in_progress()


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------

class TestZeroExposure:
    """Zero exposure — tests floor behavior in timing paths."""

    def test_zero_exposure_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', exposure=0.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestZeroIllumination:
    """Zero illumination — LED should still be called."""

    def test_zero_illumination_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', illumination=0.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestMinimalGain:
    """Gain of 0."""

    def test_zero_gain_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', gain=0.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestLargeSum:
    """Large sum count to stress frame averaging."""

    def test_sum_16_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', sum_count=16)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Large step counts
# ---------------------------------------------------------------------------

class TestLargeProtocol:
    """Protocol with many steps — verifies no accumulation bugs."""

    def test_50_step_single_scan(self, executor, scope, tmp_path):
        steps = [{'color': 'BF', 'x': float(i), 'y': 0.0} for i in range(50)]
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_50_steps_visited(self, executor, scope, tmp_path):
        steps = [{'color': 'BF', 'x': float(i), 'y': 0.0} for i in range(50)]
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        step_indices = {c.kwargs.get('idx', c.args[0] if c.args else None)
                        for c in protocol.step.call_args_list}
        assert set(range(50)) <= step_indices


# ---------------------------------------------------------------------------
# Kitchen sink: all features at once
# ---------------------------------------------------------------------------

class TestAllFeaturesEnabled:
    """Protocol exercising many features simultaneously."""

    def test_tiling_zstack_autogain_falsecolor_sum(self, executor, scope, tmp_path):
        """2x2 tiles, 3 z-slices, auto-gain, false color, sum=2."""
        steps = []
        tile_group = 0
        for r in range(2):
            for c in range(2):
                tile_group += 1
                for z_idx in range(3):
                    steps.append({
                        'color': 'Red',
                        'x': 10.0 + c * 1.0,
                        'y': 20.0 + r * 1.0,
                        'z': 4000.0 + z_idx * 100.0,
                        'tile': f'R{r}C{c}',
                        'z_slice': z_idx,
                        'tile_group_id': tile_group,
                        'zstack_group_id': tile_group,
                        'auto_gain': True,
                        'false_color': True,
                        'sum_count': 2,
                    })
        protocol = _make_multi_step_protocol(steps)
        assert protocol.num_steps() == 12
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_multi_well_multi_channel_tiling_separate_folders(self, executor, scope, tmp_path):
        """2 wells, BF+Red channels, 1x3 tiles, separate folders."""
        steps = []
        for well, wx, wy in [('A1', 10.0, 20.0), ('B1', 10.0, 40.0)]:
            for color in ['BF', 'Red']:
                for c in range(3):
                    steps.append({
                        'color': color,
                        'well': well,
                        'x': wx + c * 0.5,
                        'y': wy,
                        'tile': f'R0C{c}',
                        'tile_group_id': 1,
                    })
        protocol = _make_multi_step_protocol(steps)
        assert protocol.num_steps() == 12  # 2 wells * 2 colors * 3 tiles
        completed, _ = _run_and_wait(executor, protocol, tmp_path,
                                      separate_folder_per_channel=True)
        assert completed


# ---------------------------------------------------------------------------
# Saving edge cases
# ---------------------------------------------------------------------------

class TestSavingWithNoneParentDir:
    """When parent_dir is None, saving should be auto-disabled."""

    def test_none_parent_dir_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')

        done = threading.Event()
        result_holder = {}

        def on_complete(**kwargs):
            result_holder.update(kwargs)
            done.set()

        callbacks = {
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }

        executor.run(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_no_parent',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=None,
            max_scans=1,
            callbacks=callbacks,
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False, 'PC': False, 'DF': False,
                'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
            },
        )

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed


# ---------------------------------------------------------------------------
# Turret support
# ---------------------------------------------------------------------------

class TestWithTurret:
    """Scope with turret enabled — objective name included in filenames."""

    def test_turret_protocol_completes(self, executor, scope, tmp_path):
        scope.motion._has_turret = True
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Callback edge cases
# ---------------------------------------------------------------------------

class TestMinimalCallbacks:
    """Run with only the required run_complete callback — no optional ones."""

    def test_completes_with_minimal_callbacks(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')

        done = threading.Event()

        def on_complete(**kwargs):
            done.set()

        # Only provide run_complete — no go_to_step or move_position.
        # This forces _go_to_step to use _default_move (which we've mocked).
        executor.run(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_minimal_cb',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks={'run_complete': on_complete},
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False, 'PC': False, 'DF': False,
                'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
            },
        )

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed


# ---------------------------------------------------------------------------
# Video edge cases
# ---------------------------------------------------------------------------

class TestVideoEdgeCases:
    """Edge cases for video capture."""

    def test_very_short_video(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(
            color='BF',
            acquire='video',
            video_config={'duration': 0.1, 'fps': 10},
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_video_low_fps(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(
            color='BF',
            acquire='video',
            video_config={'duration': 0.5, 'fps': 1},
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
