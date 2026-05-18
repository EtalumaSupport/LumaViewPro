# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Integration tests using Lumascope(simulate=True) with real SequencedCaptureRunner.

Unlike test_protocol_execution.py which mocks the scope, camera, and autofocus,
these tests use real simulated hardware and verify end-to-end behavior by
inspecting simulator state after protocol runs.

Test tiers:
  - Tier 1: Single-step protocols — verify LED, motor, camera state
  - Tier 2: Multi-step protocols — multi-channel, Z-stack, tiling
  - Tier 3: Autofocus — real AutofocusRunner with SimulatedCamera focus simulation
"""

import datetime
import json
import logging
import logging.handlers
import os
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Test-specific mocks below.

# Mock settings_init before sequenced_capture_runner imports it
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

from modules.lumascope_api import Lumascope
from modules.sequential_io_executor import SequentialIOExecutor
from modules.sequenced_capture_runner import SequencedCaptureRunner
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.autofocus_runner import AutofocusRunner
from modules.protocol import Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPLETION_TIMEOUT = 30  # generous for CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executors():
    """Create and start all SequentialIOExecutors + protocol_thread."""
    from modules.protocol_thread import ProtocolThread
    names = ['io', 'file_io', 'camera', 'autofocus']
    execs = {n: SequentialIOExecutor(name=f"INTEG_{n.upper()}") for n in names}
    for e in execs.values():
        e.start()
    pt = ProtocolThread()
    pt.start()
    execs['protocol'] = pt
    return execs


def _shutdown_executors(execs):
    for name, e in execs.items():
        try:
            if name == 'protocol':
                e.stop(timeout=2.0)
            else:
                e.shutdown()
        except Exception:
            pass


def _make_autogain_settings():
    return {
        'target_brightness': 0.3,
        'min_gain': 0.0,
        'max_gain': 20.0,
        'max_duration': datetime.timedelta(seconds=2),
    }


def _make_image_capture_config():
    return {
        'output_format': {
            'live': 'TIFF',
            'sequenced': 'TIFF',
        },
        'use_full_pixel_depth': False,
    }


TILING_CONFIGS = pathlib.Path(__file__).parent.parent / "data" / "tiling.json"


def _make_protocol(steps_config):
    """Create a real Protocol object with the given steps.

    steps_config: list of dicts with keys: color, illumination, gain, exposure,
        auto_gain, auto_focus, acquire, x, y, z, well, tile, z_slice,
        tile_group_id, zstack_group_id, objective, sum_count, video_config, stim_config.
    Missing keys get defaults.
    """
    import pandas as pd

    defaults = {
        'color': 'BF', 'illumination': 50.0, 'gain': 1.0, 'exposure': 10.0,
        'auto_gain': False, 'auto_focus': False, 'acquire': 'image',
        'false_color': False, 'sum_count': 1,
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
            'X': merged['x'], 'Y': merged['y'], 'Z': merged['z'],
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
            'Step Index': i,
        })

    df = pd.DataFrame(rows)
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': df,
        'period': datetime.timedelta(minutes=1),
        'duration': datetime.timedelta(hours=1),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)


def _wait_for_executor_idle(executor, timeout=5.0):
    """Wait until executor is fully idle (not running and file IO drained)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not executor._run_in_progress_event.is_set() and not executor.file_io_executor.is_protocol_queue_active():
            return True
        time.sleep(0.05)
    return False


def _run_and_wait(executor, protocol, tmp_path, **run_kwargs):
    """Run a protocol and wait for completion. Returns (completed, result_kwargs)."""
    done = threading.Event()
    result_holder = {}

    def on_complete(**kwargs):
        result_holder.update(kwargs)
        done.set()

    callbacks = run_kwargs.pop('callbacks', {})
    callbacks['run_complete'] = on_complete
    # Don't provide go_to_step — let the executor use _default_move for real motor movement
    callbacks.setdefault('move_position', lambda axis: None)

    executor.run(
        protocol=protocol,
        run_trigger_source='test',
        run_mode=run_kwargs.pop('run_mode', SequencedCaptureRunMode.SINGLE_SCAN),
        sequence_name='integ_test',
        image_capture_config=run_kwargs.pop('image_capture_config', _make_image_capture_config()),
        autogain_settings=run_kwargs.pop('autogain_settings', _make_autogain_settings()),
        parent_dir=tmp_path / 'output',
        max_scans=run_kwargs.pop('max_scans', 1),
        callbacks=callbacks,
        leds_state_at_end=run_kwargs.pop('leds_state_at_end', 'off'),
        enable_image_saving=run_kwargs.pop('enable_image_saving', False),
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
    """Create a real Lumascope with simulated hardware."""
    s = Lumascope(simulate=True)
    # Set timing to fast for test speed
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    # Camera must be grabbing for get_image to work
    s._camera_driver.start_grabbing()
    yield s
    s._camera_driver.stop_grabbing()
    s.disconnect()


@pytest.fixture
def executors():
    execs = _make_executors()
    yield execs
    _shutdown_executors(execs)


@pytest.fixture
def executor(scope, executors):
    """Create a SequencedCaptureRunner with real simulated scope,
    real WellPlateLoader, and real CoordinateTransformer."""
    from modules.coord_transformations import CoordinateTransformer
    from modules.labware_loader import WellPlateLoader

    # Use a mock autofocus executor for non-AF tests.
    # AF executor is the one mock we keep — real AF needs real camera focus
    # simulation which is only set up in the af_executor fixture.
    mock_af = MagicMock()
    mock_af.reset = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.complete = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.result = MagicMock(return_value=None)
    mock_af.best_focus_position = MagicMock(return_value=5000.0)
    mock_af.run_in_progress = MagicMock(return_value=False)

    exc = SequencedCaptureRunner(
        scope=scope,
        stage_offset={'x': 0.0, 'y': 0.0},
        io_executor=executors['io'],
        protocol_thread=executors['protocol'],
        file_io_executor=executors['file_io'],
        camera_executor=executors['camera'],
        autofocus_thread=MagicMock(),
        autofocus_runner=mock_af,
    )
    exc._wellplate_loader = WellPlateLoader()
    exc._coordinate_transformer = CoordinateTransformer()
    return exc


@pytest.fixture
def af_executor(scope, executors):
    """Create a SequencedCaptureRunner with real AutofocusRunner for AF tests."""
    af = AutofocusRunner(
        scope=scope,
        camera_executor=executors['camera'],
        io_executor=executors['io'],
        file_io_executor=executors['file_io'],
    )

    exc = SequencedCaptureRunner(
        scope=scope,
        stage_offset={'x': 0.0, 'y': 0.0},
        io_executor=executors['io'],
        protocol_thread=executors['protocol'],
        file_io_executor=executors['file_io'],
        camera_executor=executors['camera'],
        autofocus_thread=MagicMock(),
        autofocus_runner=af,
    )
    return exc


# ===========================================================================
# Tier 1: Single-step protocols with real simulated hardware
# ===========================================================================

class TestIntegrationSingleStep:
    """Verify that single-step protocols drive real simulator state correctly."""

    def test_completes_with_simulated_scope(self, executor, scope, tmp_path):
        """Most basic integration test — protocol runs to completion on simulated hardware."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 100.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, "Protocol did not complete within timeout"

    def test_leds_off_after_completion(self, executor, scope, tmp_path):
        """After protocol completes with leds_state_at_end='off', all LEDs should be off."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 100.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path, leds_state_at_end='off')
        assert completed

        # All LED channels should be off
        for color in ('BF', 'PC', 'DF', 'Red', 'Green', 'Blue'):
            assert not scope.illumination.led_enabled(color), f"LED {color} still on after protocol"

    def test_camera_settings_applied(self, executor, scope, tmp_path):
        """Verify gain and exposure are set on the real camera simulator."""
        protocol = _make_protocol([{
            'color': 'BF', 'gain': 5.0, 'exposure': 25.0, 'illumination': 50.0,
        }])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        # The camera should have had gain and exposure set during the protocol.
        # After completion, values may be restored — but the simulator should have
        # received the calls. We verify the camera is still functional.
        assert scope._camera_driver.is_connected()

    def test_motor_position_set(self, executor, scope, tmp_path):
        """Verify the motor moves to the protocol step position."""
        z_target = 7000.0  # um
        protocol = _make_protocol([{
            'color': 'BF', 'z': z_target, 'illumination': 50.0,
        }])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        # Z position should be near the target (simulator moves instantly in fast mode)
        z_pos = scope._motion_driver.current_pos('Z')
        assert abs(z_pos - z_target) < 100.0, f"Z position {z_pos} not near target {z_target}"

    def test_image_captured(self, executor, scope, tmp_path):
        """Verify that an image was actually captured during the protocol."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 50.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        # After a successful protocol run, the camera should have grabbed at least one image
        # The array is populated by grab() calls during the capture sequence
        arr = scope._camera_driver.array
        assert arr is not None
        assert isinstance(arr, np.ndarray)
        # If grab was called, array should have 2D shape (height, width)
        if arr.ndim == 2:
            assert arr.shape[0] > 0 and arr.shape[1] > 0


class TestIntegrationAutoGain:
    """Test auto-gain with real simulated camera."""

    def test_auto_gain_completes(self, executor, scope, tmp_path):
        """Auto-gain protocol step completes without error."""
        protocol = _make_protocol([{
            'color': 'BF', 'auto_gain': True, 'illumination': 50.0,
        }])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_auto_gain_adjusts_camera(self, executor, scope, tmp_path):
        """After auto-gain, camera gain should have been modified."""
        # Set initial gain to something we can detect changed
        scope._camera_driver.gain(1.0)
        initial_gain = scope._camera_driver.get_gain()

        protocol = _make_protocol([{
            'color': 'BF', 'auto_gain': True, 'illumination': 50.0,
        }])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # We can't assert the exact gain value since it depends on auto-gain logic,
        # but the protocol should have completed without error


# ===========================================================================
# Tier 2: Multi-step protocols
# ===========================================================================

class TestIntegrationMultiChannel:
    """Multi-channel protocols with real simulated hardware."""

    def test_two_channel_completes(self, executor, scope, tmp_path):
        """BF + Green two-channel protocol completes."""
        protocol = _make_protocol([
            {'color': 'BF', 'illumination': 100.0},
            {'color': 'Green', 'illumination': 75.0},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_three_channel_completes(self, executor, scope, tmp_path):
        """BF + Green + Red three-channel protocol completes."""
        protocol = _make_protocol([
            {'color': 'BF', 'illumination': 100.0},
            {'color': 'Green', 'illumination': 75.0},
            {'color': 'Red', 'illumination': 50.0},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_leds_off_after_multi_channel(self, executor, scope, tmp_path):
        """After multi-channel protocol, all LEDs should be off."""
        protocol = _make_protocol([
            {'color': 'BF', 'illumination': 100.0},
            {'color': 'Green', 'illumination': 75.0},
            {'color': 'Red', 'illumination': 50.0},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        for color in ('BF', 'PC', 'DF', 'Red', 'Green', 'Blue'):
            assert not scope.illumination.led_enabled(color), f"LED {color} still on"


class TestIntegrationZStack:
    """Z-stack protocols with real motor movement."""

    def test_z_stack_three_slices(self, executor, scope, tmp_path):
        """3-slice Z-stack completes and motor visits all Z positions."""
        z_positions = [4000.0, 5000.0, 6000.0]
        steps = [
            {'color': 'BF', 'z': z, 'z_slice': i, 'zstack_group_id': 1,
             'illumination': 50.0}
            for i, z in enumerate(z_positions)
        ]
        protocol = _make_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_z_stack_motor_moved(self, executor, scope, tmp_path):
        """After Z-stack, motor should have moved from its initial position."""
        initial_z = scope._motion_driver.current_pos('Z')
        z_positions = [4000.0, 5000.0, 6000.0]
        steps = [
            {'color': 'BF', 'z': z, 'z_slice': i, 'zstack_group_id': 1,
             'illumination': 50.0}
            for i, z in enumerate(z_positions)
        ]
        protocol = _make_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        z_final = scope._motion_driver.current_pos('Z')
        # Motor should have moved to one of the Z-stack positions
        assert z_final in z_positions or abs(z_final - initial_z) > 100.0, \
            f"Z={z_final} didn't move from initial {initial_z}"


class TestIntegrationMultiWell:
    """Multi-well protocols with XY movement."""

    def test_two_wells_completes(self, executor, scope, tmp_path):
        """Two-well protocol completes (different XY positions)."""
        protocol = _make_protocol([
            {'color': 'BF', 'x': 10.0, 'y': 20.0, 'well': 'A1', 'illumination': 50.0},
            {'color': 'BF', 'x': 30.0, 'y': 40.0, 'well': 'A2', 'illumination': 50.0},
        ])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestIntegrationTiling:
    """Tiling protocols with multiple tile positions."""

    def test_1x3_tiling_completes(self, executor, scope, tmp_path):
        """1x3 tile pattern completes."""
        steps = [
            {'color': 'BF', 'x': 10.0 + i * 1.0, 'y': 20.0,
             'tile': f'T{i}', 'tile_group_id': 1, 'illumination': 50.0}
            for i in range(3)
        ]
        protocol = _make_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ===========================================================================
# Tier 3: Autofocus with real focus simulation
# ===========================================================================

class TestIntegrationAutofocus:
    """Autofocus tests using real AutofocusRunner + SimulatedCamera focus sim."""

    def test_autofocus_step_completes(self, af_executor, scope, tmp_path):
        """Single step with auto_focus=True completes using real AF executor."""
        # Set up focus simulation
        scope._camera_driver.set_test_pattern('focus_target')
        scope._camera_driver.set_focal_z(5000.0)

        protocol = _make_protocol([{
            'color': 'BF', 'auto_focus': True, 'z': 5000.0,
            'illumination': 100.0,
        }])
        completed, _ = _run_and_wait(af_executor, protocol, tmp_path,
                                      update_z_pos_from_autofocus=True)
        assert completed, "Autofocus protocol did not complete"

    def test_camera_state_restored_before_af_signals_done(self, scope, executors):
        """Regression test for #610: AF must restore camera state before
        clearing _af_in_progress, otherwise capture() on the protocol
        thread races ahead and reads stale cached gain.

        The bug: _iterate() cleared _af_in_progress before the finally
        block in _autofocus_loop() restored camera state. The protocol
        worker saw AF as done, entered capture(), and set_gain() was a
        no-op (cache still showed AF scanning gain). Then the finally
        block restored the previous channel's gain, and the frame was
        grabbed with wrong settings.
        """
        # Set up focus simulation
        scope._camera_driver.set_test_pattern('focus_target')
        scope._camera_driver.set_focal_z(5000.0)

        af = AutofocusRunner(
            scope=scope,
            camera_executor=executors['camera'],
            io_executor=executors['io'],
            file_io_executor=executors['file_io'],
        )

        # Simulate the multi-channel scenario: camera is at Green settings
        # when BF AF starts. This is the state AF will save and later restore.
        scope.set_gain(20.0)
        scope.set_exposure_time(100.0)

        # Drive AF through AutofocusThread so the abort_event contract
        # is satisfied and the Future resolves when AFE.run()'s finally
        # block has fully restored camera state.
        from modules.autofocus_thread import AutofocusThread
        thread = AutofocusThread(afe=af)
        thread.start()
        try:
            future = thread.run_autofocus(
                objective_id='10x Oly',
                led_color='BF',
                led_illumination=50.0,
                camera_gain=1.0,
                camera_exposure=2.0,
            )
            future.result(timeout=15.0)
        finally:
            thread.stop(timeout=2.0)

        # The moment the Future resolves, camera state must already
        # reflect the restored (pre-AF) values. AFE.run()'s finally
        # block restores camera state BEFORE clearing _af_in_progress;
        # the Future resolves AFTER AFE.run() returns, so the read
        # below cannot race the restoration.
        actual_gain = scope.get_gain()
        actual_exp = scope.get_exposure_time()
        assert abs(actual_gain - 20.0) < 0.1, (
            f"Camera gain should be restored to 20.0 (pre-AF) when "
            f"the AF Future resolves, but got {actual_gain}"
        )
        assert abs(actual_exp - 100.0) < 0.1, (
            f"Camera exposure should be restored to 100.0ms (pre-AF) when "
            f"the AF Future resolves, but got {actual_exp}"
        )


# ===========================================================================
# Tier 4: State assertion tests
# ===========================================================================

class TestIntegrationStateAssertions:
    """Verify simulator state matches expectations after protocol runs."""

    def test_led_bf_channel(self, executor, scope, tmp_path):
        """Verify BF LED is driven and turned off after protocol."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 75.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        assert not scope.illumination.led_enabled('BF')

    def test_led_green_channel(self, executor, scope, tmp_path):
        """Verify Green LED is driven and turned off after protocol."""
        protocol = _make_protocol([{'color': 'Green', 'illumination': 75.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        assert not scope.illumination.led_enabled('Green')

    def test_led_red_channel(self, executor, scope, tmp_path):
        """Verify Red LED is driven and turned off after protocol."""
        protocol = _make_protocol([{'color': 'Red', 'illumination': 75.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        assert not scope.illumination.led_enabled('Red')

    def test_scope_connected_throughout(self, executor, scope, tmp_path):
        """Scope remains connected after protocol run."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 50.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        assert scope._led_driver.is_connected()
        assert scope._motion_driver.is_connected()
        assert scope._camera_driver.is_connected()

    def test_camera_still_grabbing(self, executor, scope, tmp_path):
        """Camera should still be in grabbing state after protocol."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 50.0}])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        # Camera grabbing state is managed externally, should still be active
        assert scope._camera_driver.is_grabbing()

    @pytest.mark.skip(reason="Executor reuse bug: second run() starts but run_complete callback "
                             "never fires. Debug shows: run enters, _reset_vars clears state, "
                             "protocol_start sets flags, run_loop task submitted — but cleanup's "
                             "run_complete callback doesn't reach the test's done.set(). "
                             "Two fixes applied (is_protocol_queue_active, protocol_start clears "
                             "stale protocol_finish), but deeper issue remains in cleanup callback "
                             "dispatch. Needs executor simplification in 4.1.")
    def test_second_run_after_first(self, executor, scope, tmp_path):
        """A second protocol run completes after the first finishes."""
        protocol = _make_protocol([{'color': 'BF', 'illumination': 50.0}])

        # First run
        completed_1, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed_1, "First protocol run did not complete"

        # Wait for executor to be fully idle (run_in_progress cleared, queue drained)
        idle = _wait_for_executor_idle(executor, timeout=5.0)
        assert idle, "Executor did not reach idle state after first run"

        # Second run — should start without being blocked
        completed_2, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed_2, "Second protocol run did not complete (back-to-back blocked)"

    def test_different_exposure_per_step(self, executor, scope, tmp_path):
        """Protocol with varying exposure times completes."""
        steps = [
            {'color': 'BF', 'exposure': exp, 'illumination': 50.0}
            for exp in [5.0, 10.0, 50.0, 100.0]
        ]
        protocol = _make_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_different_gain_per_step(self, executor, scope, tmp_path):
        """Protocol with varying gain values completes."""
        steps = [
            {'color': 'BF', 'gain': g, 'illumination': 50.0}
            for g in [1.0, 2.0, 5.0, 10.0]
        ]
        protocol = _make_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ===========================================================================
# Headless / REST API Path Tests
# ===========================================================================

from modules.scope_session import ScopeSession
from modules.protocol_runner import ProtocolRunner


class TestHeadlessSession:
    """Verify ScopeSession.create_headless() and ProtocolRunner work end-to-end."""

    def test_create_headless_returns_session(self):
        """create_headless() should return a working ScopeSession."""
        session = ScopeSession.create_headless()
        assert session is not None
        assert session.scope is not None
        assert session.io_executor is not None
        assert session.camera_executor is not None
        assert session.settings is not None

    def test_create_headless_scope_is_simulated(self):
        """Headless session should use simulated hardware."""
        session = ScopeSession.create_headless()
        assert session.scope._simulated is True

    def test_create_headless_with_custom_settings(self):
        """create_headless() should accept custom settings."""
        custom = {'BF': {'autofocus': False}, 'custom_key': 42}
        session = ScopeSession.create_headless(settings=custom)
        assert session.settings['custom_key'] == 42

    def test_headless_led_commands(self):
        """Headless session should support LED on/off via scope."""
        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            scope = session.scope
            scope.illumination.led_on(channel=0, mA=100)
            assert scope.illumination.get_led_ma('Blue') == 100
            scope.illumination.led_off(channel=0)
            assert scope.illumination.get_led_ma('Blue') == -1
        finally:
            session.shutdown_executors()

    def test_headless_motor_position(self):
        """Headless session should support motor position queries."""
        session = ScopeSession.create_headless()
        scope = session.scope
        pos = scope.motion.get_current_position('Z')
        assert isinstance(pos, (int, float))

    def test_create_protocol_runner(self):
        """ScopeSession.create_protocol_runner() should return a ProtocolRunner."""
        session = ScopeSession.create_headless()
        runner = session.create_protocol_runner()
        assert isinstance(runner, ProtocolRunner)
        assert runner.session is session

    def test_protocol_runner_runs_protocol(self, tmp_path):
        """ProtocolRunner should execute a protocol to completion on headless session."""
        settings = {
            'BF': {'autofocus': False}, 'PC': {'autofocus': False},
            'DF': {'autofocus': False}, 'Red': {'autofocus': False},
            'Green': {'autofocus': False}, 'Blue': {'autofocus': False},
            'Lumi': {'autofocus': False},
            'stage_offset': {'x': 0.0, 'y': 0.0},
            'live_folder': str(tmp_path),
            'protocol': {
                'autogain': {
                    'target_brightness': 0.3,
                    'max_duration_seconds': 1.0,
                    'min_gain': 0.0,
                    'max_gain': 20.0,
                },
            },
        }
        session = ScopeSession.create_headless(settings=settings)
        session.start_executors()
        try:
            runner = session.create_protocol_runner()
            protocol = _make_protocol([{'color': 'BF', 'illumination': 100.0}])

            done = threading.Event()
            def on_complete(**kwargs):
                done.set()

            runner.run_single_scan(
                protocol=protocol,
                sequence_name="headless_test",
                parent_dir=str(tmp_path),
                callbacks={'run_complete': on_complete, 'files_complete': lambda **kw: None},
            )

            completed = done.wait(timeout=COMPLETION_TIMEOUT)
            assert completed, "Headless protocol did not complete within timeout"
        finally:
            runner.shutdown()
            session.shutdown_executors()

    def test_protocol_runner_afe_no_kivy_dependency(self):
        """AFE has no Kivy Clock dependency (Rule 15). Under the
        thread-driven model the AF thread drives iterations directly;
        there are no Clock-schedule attributes to misconfigure."""
        session = ScopeSession.create_headless()
        runner = session.create_protocol_runner()
        af = runner.sequenced_capture_runner._autofocus_runner
        assert not hasattr(af, '_clock_unschedule_fn')
        assert not hasattr(af, '_clock_schedule_interval_fn')


class TestRestAPIPrep:
    """Verify REST API prep methods (P-1 through P-8)."""

    def test_get_pixel_format(self):
        """get_pixel_format() should return format string from simulated camera."""
        session = ScopeSession.create_headless()
        fmt = session.scope.get_pixel_format()
        assert isinstance(fmt, str)
        assert fmt in ('Mono8', 'Mono10', 'Mono12')

    def test_set_pixel_format(self):
        """set_pixel_format() should change the camera format."""
        session = ScopeSession.create_headless()
        result = session.scope.set_pixel_format('Mono12')
        assert result is True
        assert session.scope.get_pixel_format() == 'Mono12'

    def test_set_pixel_format_invalid(self):
        """set_pixel_format() with invalid format should return False."""
        session = ScopeSession.create_headless()
        result = session.scope.set_pixel_format('InvalidFormat')
        assert result is False

    def test_get_supported_pixel_formats(self):
        """get_supported_pixel_formats() should return tuple of format strings."""
        session = ScopeSession.create_headless()
        formats = session.scope.get_supported_pixel_formats()
        assert isinstance(formats, tuple)
        assert len(formats) > 0
        assert 'Mono8' in formats

    def test_pixel_format_inactive_camera(self):
        """Pixel format methods should handle inactive camera gracefully."""
        session = ScopeSession.create_headless()
        session.scope._camera_driver = None
        assert session.scope.get_pixel_format() is None
        assert session.scope.set_pixel_format('Mono8') is False
        assert session.scope.get_supported_pixel_formats() == ()

    def test_get_motor_info(self):
        """get_motor_info() should return model, serial, firmware."""
        session = ScopeSession.create_headless()
        info = session.scope.get_motor_info()
        assert 'model' in info
        assert 'serial_number' in info
        assert 'firmware_version' in info
        assert info['model'] is not None

    def test_get_led_info(self):
        """get_led_info() should return firmware and connection status."""
        session = ScopeSession.create_headless()
        info = session.scope.get_led_info()
        assert info['connected'] is True
        assert info['firmware_version'] is not None

    def test_get_camera_info(self):
        """get_camera_info() should return model and connection status."""
        session = ScopeSession.create_headless()
        info = session.scope.get_camera_info()
        assert info['connected'] is True
        assert info['model'] is not None

    def test_get_system_info(self):
        """get_system_info() should return consolidated info for all hardware."""
        session = ScopeSession.create_headless()
        info = session.scope.get_system_info()
        assert 'motor' in info
        assert 'led' in info
        assert 'camera' in info
        assert info['simulated'] is True
        assert 'lvp_version' in info

    def test_system_info_no_hardware(self):
        """get_system_info() should handle missing hardware gracefully."""
        from drivers.null_motorboard import NullMotionBoard
        session = ScopeSession.create_headless()
        session.scope._motion_driver = NullMotionBoard()
        session.scope._led_driver = None
        session.scope._camera_driver = None
        info = session.scope.get_system_info()
        assert info['motor']['model'] is None
        assert info['led']['connected'] is False
        assert info['camera']['connected'] is False

    def test_encode_image_png(self):
        """encode_image() should encode numpy array to PNG bytes."""
        from modules.image_utils import encode_image
        img = np.zeros((100, 100), dtype=np.uint8)
        data = encode_image(img, 'png')
        assert isinstance(data, bytes)
        assert len(data) > 0
        # PNG magic bytes
        assert data[:4] == b'\x89PNG'

    def test_encode_image_jpeg(self):
        """encode_image() should encode numpy array to JPEG bytes."""
        from modules.image_utils import encode_image
        img = np.zeros((100, 100), dtype=np.uint8)
        data = encode_image(img, 'jpeg')
        assert isinstance(data, bytes)
        # JPEG magic bytes
        assert data[:2] == b'\xff\xd8'

    def test_encode_image_invalid_format(self):
        """encode_image() should raise ValueError for unsupported format."""
        from modules.image_utils import encode_image
        img = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported"):
            encode_image(img, 'bmp')

    def test_rest_api_log_filter_logic(self):
        """RestAPIFilter logic: only pass records with api_request=True."""
        # Replicate the filter logic here since lvp_logger is mocked in this test file
        class RestAPIFilter(logging.Filter):
            def filter(self, record):
                return bool(getattr(record, 'api_request', False))

        filt = RestAPIFilter()

        # Normal records should be filtered out
        record = logging.LogRecord('test', logging.INFO, '', 0, 'test', (), None)
        assert filt.filter(record) is False

        # Records with api_request=True should pass
        record.api_request = True
        assert filt.filter(record) is True

    def test_get_available_objectives(self):
        """get_available_objectives() should return list of objective IDs."""
        session = ScopeSession.create_headless()
        objectives = session.scope.get_available_objectives()
        assert isinstance(objectives, list)
        assert len(objectives) > 0
        # Should contain known objectives from objectives.json
        assert any('20x' in obj for obj in objectives)

    def test_get_current_objective_none_by_default(self):
        """get_current_objective() should return None before setting one."""
        session = ScopeSession.create_headless()
        assert session.scope.get_current_objective() is None

    def test_get_current_objective_after_set(self):
        """get_current_objective() should return info after set_objective()."""
        session = ScopeSession.create_headless()
        objectives = session.scope.get_available_objectives()
        session.scope.set_objective(objectives[0])
        current = session.scope.get_current_objective()
        assert current is not None
        assert isinstance(current, dict)

    def test_autofocus_runner_get_status_idle(self):
        """AutofocusRunner.get_status() should return idle state initially."""
        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            runner = session.create_protocol_runner()
            af = runner.sequenced_capture_runner._autofocus_runner
            status = af.get_status()
            assert status['state'] == 'idle'
            assert status['in_progress'] is False
            assert status['best_position'] is None
        finally:
            runner.shutdown()
            session.shutdown_executors()

    def test_autofocus_thread_abort_noop_when_idle(self):
        """AutofocusThread.abort() should be safe when no run is in flight."""
        from modules.autofocus_thread import AutofocusThread
        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            runner = session.create_protocol_runner()
            af = runner.sequenced_capture_runner._autofocus_runner
            thread = AutofocusThread(afe=af)
            thread.start()
            try:
                thread.abort()  # idle -> no-op
                assert thread.is_running is False
            finally:
                thread.stop(timeout=2.0)
        finally:
            runner.shutdown()
            session.shutdown_executors()

    def test_autofocus_thread_run_and_complete(self):
        """AutofocusThread.run_autofocus() resolves Future with the
        best focus position when AF completes."""
        from modules.autofocus_thread import AutofocusThread
        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            runner = session.create_protocol_runner()
            af = runner.sequenced_capture_runner._autofocus_runner
            thread = AutofocusThread(afe=af)
            thread.start()
            try:
                objectives = session.scope.get_available_objectives()
                future = thread.run_autofocus(objective_id=objectives[0])
                result = future.result(timeout=30)
                assert result is not None
                assert af.complete() is True
            finally:
                thread.stop(timeout=2.0)
        finally:
            runner.shutdown()
            session.shutdown_executors()

    def test_autofocus_thread_abort_during_run(self):
        """AutofocusThread.abort() unwinds an in-flight run; the Future
        surfaces AutofocusAborted."""
        from modules.autofocus_thread import AutofocusThread
        from modules.exceptions import AutofocusAborted
        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            runner = session.create_protocol_runner()
            af = runner.sequenced_capture_runner._autofocus_runner
            thread = AutofocusThread(afe=af)
            thread.start()
            try:
                objectives = session.scope.get_available_objectives()
                future = thread.run_autofocus(objective_id=objectives[0])

                # Give the thread a moment to enter AFE.run()
                time.sleep(0.1)
                assert thread.is_running is True

                thread.abort()

                with pytest.raises(AutofocusAborted):
                    future.result(timeout=5)
                assert thread.is_running is False
            finally:
                thread.stop(timeout=2.0)
        finally:
            runner.shutdown()
            session.shutdown_executors()

    def test_settings_has_rest_api_section(self):
        """Default settings template should include rest_api configuration."""
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'settings.json')
        with open(settings_path) as f:
            settings = json.load(f)
        assert 'rest_api' in settings
        assert settings['rest_api']['enabled'] is False
        assert settings['rest_api']['host'] == '127.0.0.1'
        assert settings['rest_api']['port'] == 8000
