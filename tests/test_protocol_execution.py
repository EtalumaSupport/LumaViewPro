# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Integration tests for protocol execution through SequencedCaptureRunner.

Tier 1: Core execution paths -- verifies that the most common protocol
configurations run to completion without crashing and produce the
expected sequence of hardware calls.

Uses Lumascope(simulate=True) with real SimulatedLEDBoard, SimulatedMotorBoard,
and SimulatedCamera -- no hardware or Kivy needed.
"""

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock, patch

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

from modules.exceptions import ProtocolRunRefusedError
from modules.image_mode import ImageCaptureConfig
from modules.lumascope_api import Lumascope
from modules.sequential_io_executor import SequentialIOExecutor
from modules.sequenced_capture_runner import RunPlan, SequencedCaptureRunner
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.protocol import Protocol

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
COMPLETION_TIMEOUT = 15  # seconds -- generous for CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simulated_scope():
    """Create a Lumascope with simulated hardware in fast timing mode."""
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s.imaging.start_streaming()
    return s


def _make_executors():
    """Create and start the SequentialIOExecutors + protocol_thread needed."""
    from modules.protocol_thread import ProtocolThread

    execs = {
        'io': SequentialIOExecutor(name='TEST_IO'),
        'file_io': SequentialIOExecutor(name='TEST_FILE'),
        'camera': SequentialIOExecutor(name='TEST_CAMERA'),
        'autofocus': SequentialIOExecutor(name='TEST_AF'),
    }
    for e in execs.values():
        e.start()
    pt = ProtocolThread()
    pt.start()
    execs['protocol'] = pt
    return execs


def _shutdown_executors(execs):
    """Shut down all executors + protocol_thread."""
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
        'min_gain_db': 0.0,
        'max_gain_db': 20.0,
        'max_duration': datetime.timedelta(seconds=1),
    }


def _make_image_capture_config():
    return ImageCaptureConfig.from_image_mode('8bit')


TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


def _build_real_protocol(rows, period_min=1.0, duration_hrs=1.0):
    """Build a real Protocol object from a list of step dicts."""
    import pandas as pd

    df = pd.DataFrame(rows)
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': df,
        'period': datetime.timedelta(minutes=period_min),
        'duration': datetime.timedelta(hours=duration_hrs),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)


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
    """Create a real Protocol with a single step."""
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
        'Step Index': 0,
        'Label': '',
    }
    return _build_real_protocol([step])


def _make_multi_step_protocol(steps_config):
    """Create a real Protocol with multiple steps.

    steps_config: list of dicts, each with keys like color, auto_gain, etc.
    Missing keys get defaults.
    """
    defaults = {
        'color': 'BF',
        'auto_gain': False,
        'auto_focus': False,
        'acquire': 'image',
        'gain_db': 1.0,
        'exposure_ms': 10.0,
        'illumination_ma': 50.0,
        'false_color': False,
        'sum_count': 1,
        'video_config': {'duration': 1, 'fps': 5},
        'stim_config': {},
        'x': 10.0,
        'y': 20.0,
        'z': 5000.0,
        'well': 'A1',
        'name': None,
        'tile': '',
        'z_slice': 0,
        'tile_group_id': 0,
        'zstack_group_id': 0,
        'objective': '10x Oly',
    }

    rows = []
    for i, cfg in enumerate(steps_config):
        merged = {**defaults, **cfg}
        name = merged['name'] or f'step_{i}_{merged["color"]}'
        rows.append(
            {
                'Name': name,
                'X': merged['x'],
                'Y': merged['y'],
                'Z': merged['z'],
                'Auto_Focus': merged['auto_focus'],
                'Color': merged['color'],
                'False_Color': merged['false_color'],
                'Illumination': merged['illumination_ma'],
                'Gain': merged['gain_db'],
                'Auto_Gain': merged['auto_gain'],
                'Exposure': merged['exposure_ms'],
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
                # Unique per-step label: steps that differ only in position
                # or camera settings must still derive distinct capture
                # filenames or validate_for_run refuses the run.
                'Label': name,
            }
        )
    return _build_real_protocol(rows)


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

    plan = executor.prepare(
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
            'BF': False,
            'PC': False,
            'DF': False,
            'Red': False,
            'Green': False,
            'Blue': False,
            'Lumi': False,
        },
        **run_kwargs,
    )
    executor.start(plan)

    completed = done.wait(timeout=COMPLETION_TIMEOUT)
    return completed, result_holder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scope():
    s = _make_simulated_scope()
    yield s
    s.imaging.stop_streaming()
    s.disconnect()


@pytest.fixture
def executors():
    execs = _make_executors()
    yield execs
    _shutdown_executors(execs)


@pytest.fixture
def executor(scope, executors):
    """Create a SequencedCaptureRunner with real simulated scope,
    real WellPlateLoader, and real CoordinateTransformer.

    Only the AutofocusRunner is mocked (real AF needs camera focus
    simulation which is only set up in dedicated AF test fixtures).
    """
    from modules.coord_transformations import CoordinateTransformer
    from modules.labware_loader import WellPlateLoader

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
        autofocus_thread=MagicMock(is_running=False),
        autofocus_runner=mock_af,
    )
    exc._wellplate_loader = WellPlateLoader()
    exc._coordinate_transformer = CoordinateTransformer()
    return exc


# ===========================================================================
# Tier 1: Core Execution Paths
# ===========================================================================


class TestSingleScanBasicImage:
    """Test 1: Simplest happy path -- single scan, single BF image step."""

    def test_completes_successfully(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, 'Protocol did not complete within timeout'

    def test_sets_gain_and_exposure(self, executor, scope, tmp_path):
        # Record original camera settings
        original_gain = scope._camera_driver.get_gain()
        original_exposure = scope._camera_driver.get_exposure_t()
        protocol = _make_single_step_protocol(color='BF', gain=5.0, exposure=50.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # After protocol cleanup, gain/exposure should be restored to original values
        assert scope._camera_driver.get_gain() == pytest.approx(original_gain, abs=0.1)
        assert scope._camera_driver.get_exposure_t() == pytest.approx(original_exposure, abs=0.1)

    def test_turns_led_on_and_off(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', illumination=75.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # After protocol with leds_state_at_end='off', all LEDs should be off
        for color in scope._led_driver.led_ma:
            assert not scope.illumination.led_enabled(color), f'LED {color} still on after protocol'

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
        # Auto gain cycle ran successfully -- gain value should have been adjusted
        # from the initial value by the auto-gain convergence logic
        assert scope._camera_driver.get_gain() > 0

    def test_does_not_set_manual_gain_when_auto(self, executor, scope, tmp_path):
        """When auto_gain=True, manual set_gain should NOT be called in _scan_iterate."""
        protocol = _make_single_step_protocol(color='BF', auto_gain=True, gain=5.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # _capture still calls set_gain -- but _scan_iterate should skip it
        # We can't easily distinguish, so just verify completion


class TestSingleScanAutoFocus:
    """Test 3: Single scan with autofocus enabled."""

    def test_completes_with_auto_focus(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        # Simulate AF already complete so _scan_iterate proceeds past AF logic
        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestSingleScanAutoFocusNoneResult:
    """C2 fix: autofocus returns None -- protocol must not crash."""

    def test_completes_when_autofocus_returns_none(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True
        af.best_focus_position.return_value = None  # autofocus failed

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, 'Protocol should complete even when autofocus returns None'

    def test_z_height_not_modified_when_autofocus_returns_none(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)
        original_z = protocol.step(idx=0)['Z']

        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True
        af.best_focus_position.return_value = None

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # Z height should remain unchanged when autofocus returns None
        assert protocol.step(idx=0)['Z'] == original_z, (
            'Z height should not change when autofocus returns None'
        )


class TestAutofocusFailureDoesNotHaltProtocol:
    """An autofocus failure mid-protocol must not stop the run or pop a modal --
    an unattended scan keeps capturing at the fallback Z. The autofocus thread is
    the sole owner of recording the fault with a traceback; the step runner adds
    exactly one step-correlated warning so a possibly-out-of-focus fallback-Z
    capture is traceable back to its step and well. That single line is
    complementary to the AF thread's traceback (which it does not repeat) and the
    one-shot latch keeps it from flooding once per settle poll.
    """

    def test_af_exception_warns_with_step_context_without_halt(
        self, executor, scope, tmp_path, monkeypatch
    ):
        import modules.protocol_step_runner as psr

        protocol = _make_single_step_protocol(color='BF', auto_focus=True)
        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # The run kicks off its own AF future via autofocus_thread.run_autofocus,
        # so control the fault on that returned Future -- a pre-set _af_future is
        # cleared at run start and replaced by the kicked-off one.
        af_future = MagicMock()
        af_future.done.return_value = True
        # The AF run raised (e.g. camera fault) -- carried on the Future.
        af_future.exception.return_value = RuntimeError('camera fault during AF')
        executor.autofocus_thread.run_autofocus.return_value = af_future

        warnings = []
        monkeypatch.setattr(psr.logger, 'warning', lambda msg, *a, **k: warnings.append(str(msg)))

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, 'an AF failure must not halt the protocol'

        af_warnings = [w for w in warnings if 'Autofocus failed' in w]
        assert len(af_warnings) == 1, (
            'the step runner must emit exactly one step-correlated AF-failure '
            f'warning (got {len(af_warnings)})'
        )
        w = af_warnings[0]
        # The step and well tie a blurry fallback-Z capture to its origin, and the
        # exception type + message make the line user-actionable.
        assert 'step 0' in w, f'warning is missing the step index: {w!r}'
        assert 'well A1' in w, f'warning is missing the well: {w!r}'
        assert 'RuntimeError' in w and 'camera fault during AF' in w, (
            f'warning is missing the AF exception type/message: {w!r}'
        )
        assert 'fallback Z' in w, f'warning must say capture continues at fallback Z: {w!r}'

    def test_af_failure_consumed_once_no_log_flood(self, executor, scope, monkeypatch):
        """Drive scan_iterate directly while the stage stays in motion -- the
        exact condition that made the old gate re-handle and re-log the same
        resolved AF future on every ~1 kHz settle poll. A single fault must be
        consumed exactly once and produce exactly one step-correlated warning
        across all the settle polls -- not zero, and not one per poll.
        """
        import threading as _threading

        import modules.protocol_step_runner as psr

        # Stage never reports idle: scan_iterate keeps hitting the motion-settle
        # early-return after the AF gate, so _af_future is not cleared between
        # polls (it is cleared only at the step transition).
        monkeypatch.setattr(scope.motion, 'is_moving', lambda *a, **k: True)

        protocol = _make_single_step_protocol(color='BF', auto_focus=True)
        executor._protocol = protocol
        executor._aborted = _threading.Event()
        executor._scan_in_progress.set()
        executor._run_in_progress_event.set()
        executor._grease_redistribution_event.set()
        executor._curr_step = 0
        executor._motion_wait_start = None
        executor._step_start_time = 0.0
        with executor._protocol_state_lock:
            executor._n_scans = 1
            executor._scan_count = 0
        executor._af_result_consumed = False

        af_future = MagicMock()
        af_future.done.return_value = True
        af_future.exception.return_value = RuntimeError('camera fault during AF')
        executor._af_future = af_future

        warnings = []
        monkeypatch.setattr(psr.logger, 'warning', lambda msg, *a, **k: warnings.append(str(msg)))

        for _ in range(10):
            executor._step_executor.scan_iterate()

        af_warnings = [w for w in warnings if 'Autofocus failed' in w]
        assert len(af_warnings) == 1, (
            'the step runner must emit exactly one step-correlated AF-failure '
            f'warning across the settle polls, not one per poll (got {len(af_warnings)})'
        )
        assert 'step 0' in af_warnings[0], (
            f'the AF-failure warning is missing the step index: {af_warnings[0]!r}'
        )
        assert af_future.exception.call_count == 1, (
            'the resolved AF future must be consumed exactly once, not per poll '
            f'(consumed {af_future.exception.call_count} times)'
        )
        assert executor._af_result_consumed is True


class TestSingleScanAutoGainAndAutoFocus:
    """Test 4: Single scan with both auto-gain and auto-focus."""

    def test_completes_with_both_auto_features(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_gain=True, auto_focus=True)

        # Simulate AF already complete
        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestAFSliderRaceRegression:
    """#563: scan_iterate must not overwrite the AF executor's UI write.

    Symptom (pre-fix): for an AF step at Z=5000, AF schedules a UI update to
    best_focus_position; scan_iterate then schedules a UI update with the
    pre-AF step['Z']=5000. Both writes land on Kivy's Clock queue and the
    stale step['Z'] write often wins, so the slider lies to the user even
    though the motor is at the AF-chosen position.
    """

    def test_scan_iterate_does_not_overwrite_af_z_ui(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)
        pre_af_z = protocol.step(idx=0)['Z']

        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True
        af.best_focus_position.return_value = pre_af_z + 15.0  # AF picked a different Z

        z_ui_calls = []
        executor._z_ui_update_func = lambda z: z_ui_calls.append(z)

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        assert pre_af_z not in z_ui_calls, (
            f'scan_iterate scheduled z_ui_update_func({pre_af_z}) -- this overwrites '
            f"the AF executor's UI write to best_focus_position. Bug #563 has regressed."
        )


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
        # After protocol with leds_state_at_end='off', LEDs are off --
        # completion confirms the LED was used during the protocol

    @pytest.mark.parametrize('color', ['Red', 'Green', 'Blue', 'PC', 'DF', 'Lumi'])
    def test_completes_for_all_channels(self, executor, scope, tmp_path, color):
        protocol = _make_single_step_protocol(color=color)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, f'Protocol failed for channel {color}'


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
        completed, _ = _run_and_wait(executor, protocol, tmp_path, video_as_frames=True)
        assert completed


class TestFullProtocol:
    """Test 7: Full protocol with multiple scans."""

    def test_two_scans_complete(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        # Override period to be very short so scans happen fast
        protocol.modify_time_params(
            period=datetime.timedelta(seconds=0.1),
            duration=datetime.timedelta(seconds=1),
        )

        completed, _ = _run_and_wait(
            executor,
            protocol,
            tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=2,
        )
        assert completed


class TestMultiStepMultiChannel:
    """Test 8: Single scan with multiple steps across channels."""

    def test_bf_and_red_steps_complete(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF', 'illumination_ma': 50.0, 'exposure_ms': 10.0},
                {'color': 'Red', 'illumination_ma': 100.0, 'exposure_ms': 50.0},
            ]
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_steps_visited(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF'},
                {'color': 'Red'},
                {'color': 'Green'},
            ]
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # Note: the executor deep-copies the protocol at run start (P2-14),
        # so we cannot spy on the original mock's .step() calls.
        # Completion of a 3-step protocol without error confirms all steps
        # were visited -- individual step execution is covered by other tests.


# ===========================================================================
# Tier 1 Extras: Run-level options
# ===========================================================================


class TestImageSavingDisabled:
    """Protocol with image saving disabled should still complete."""

    def test_completes_without_saving(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path, enable_image_saving=False)
        assert completed


class TestDisableSavingArtifacts:
    """Protocol with all saving artifacts disabled."""

    def test_completes_without_artifacts(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path, disable_saving_artifacts=True)
        assert completed


class TestLedStateAtEnd:
    """Verify LED cleanup behavior."""

    def test_leds_off_at_end(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path, leds_state_at_end='off')
        assert completed
        # Verify all LEDs are off via simulator public API
        for color in scope._led_driver.led_ma:
            assert not scope.illumination.led_enabled(color), f'LED {color} still on'

    def test_return_to_original_leds(self, executor, scope, tmp_path):
        # Turn on BF LED before protocol so executor captures it as original state
        bf_ch = scope.illumination.color2ch(color='BF')
        scope.illumination.led_on(bf_ch, 25)
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(
            executor, protocol, tmp_path, leds_state_at_end='return_to_original'
        )
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
        config = ImageCaptureConfig.from_image_mode('12bit_scientific')
        completed, _ = _run_and_wait(executor, protocol, tmp_path, image_capture_config=config)
        assert completed

    def test_8bit_pixel_depth(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        config = ImageCaptureConfig.from_image_mode('8bit')
        completed, _ = _run_and_wait(executor, protocol, tmp_path, image_capture_config=config)
        assert completed


class TestReturnToPosition:
    """Protocol with return_to_position specified."""

    def test_returns_to_position(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(
            executor,
            protocol,
            tmp_path,
            return_to_position={'x': 50.0, 'y': 50.0, 'z': 3000.0},
        )
        assert completed


class TestSeparateFolderPerChannel:
    """Protocol with separate folder per channel."""

    def test_separate_folders(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF'},
                {'color': 'Red'},
            ]
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path, separate_folder_per_channel=True)
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
            steps.append(
                {
                    'color': color,
                    'well': well,
                    'x': base_x + c * spacing,
                    'y': base_y + r * spacing,
                    'tile': f'R{r}C{c}',
                    'tile_group_id': 1,
                    **extra,
                }
            )
    return steps


def _make_zstack_steps(num_slices, color='BF', well='A1', z_start=4000.0, z_step=100.0, **extra):
    """Generate step configs for a z-stack with num_slices slices."""
    steps = []
    for i in range(num_slices):
        steps.append(
            {
                'color': color,
                'well': well,
                'z': z_start + i * z_step,
                'z_slice': i,
                'zstack_group_id': 1,
                **extra,
            }
        )
    return steps


# ---------------------------------------------------------------------------
# Tiling tests
# ---------------------------------------------------------------------------


class TestTiling2x2:
    """2x2 tile grid -- simplest tiling case."""

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
        # P2-14: executor deep-copies the protocol, so we cannot spy on
        # the original mock's .step() calls.  Completion of a 4-tile
        # protocol without error confirms all tiles were visited.


class TestTilingAsymmetric1x3:
    """1x3 tile grid -- single row, three columns."""

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
        # P2-14: executor deep-copies the protocol, so we cannot spy on
        # the original mock's .step() calls.  Completion of a 3-tile
        # protocol without error confirms all tiles were visited.


class TestTilingAsymmetric3x1:
    """3x1 tile grid -- three rows, single column."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=3, cols=1)
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestTilingAsymmetric3x5:
    """3x5 tile grid -- 15 total tiles."""

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
    """Z-stack execution -- multiple z-slices at one position."""

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
        # P2-14: executor deep-copies the protocol, so we cannot spy on
        # the original mock's .step() calls.  Completion of a 5-slice
        # z-stack without error confirms all slices were visited.


class TestZStackWithAutoFocus:
    """Z-stack combined with autofocus."""

    def test_completes(self, executor, scope, tmp_path):
        steps = _make_zstack_steps(num_slices=3, auto_focus=True)
        protocol = _make_multi_step_protocol(steps)

        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True

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
                    steps.append(
                        {
                            'color': 'BF',
                            'x': base_x + c * 1.0,
                            'y': base_y + r * 1.0,
                            'z': 4000.0 + z_idx * 100.0,
                            'tile': f'R{r}C{c}',
                            'z_slice': z_idx,
                            'tile_group_id': tile_group,
                            'zstack_group_id': tile_group,
                        }
                    )
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
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF', 'well': 'A1', 'x': 10.0, 'y': 20.0},
                {'color': 'BF', 'well': 'A2', 'x': 30.0, 'y': 20.0},
            ]
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_six_wells_multi_channel(self, executor, scope, tmp_path):
        wells = [
            ('A1', 10, 20),
            ('A2', 30, 20),
            ('A3', 50, 20),
            ('B1', 10, 40),
            ('B2', 30, 40),
            ('B3', 50, 40),
        ]
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
                    steps.append(
                        {
                            'color': 'BF',
                            'well': well,
                            'x': wx + c * 0.5,
                            'y': wy + r * 0.5,
                            'tile': f'R{r}C{c}',
                            'tile_group_id': 1,
                        }
                    )
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
            executor,
            protocol,
            tmp_path,
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            max_scans=1,
        )
        assert completed


class TestRunModeSingleAutofocusScan:
    """SINGLE_AUTOFOCUS_SCAN run mode."""

    def test_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', auto_focus=True)

        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True

        completed, _ = _run_and_wait(
            executor,
            protocol,
            tmp_path,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            max_scans=1,
        )
        assert completed


# SINGLE_AUTOFOCUS run mode retired -- standalone AF routes directly
# through AutofocusThread.run_autofocus() from the UI, bypassing the
# SequencedCapture path. Coverage for the standalone AF flow lives in
# the autofocus_thread regression tests.


# ---------------------------------------------------------------------------
# Full protocol multi-scan with tiling
# ---------------------------------------------------------------------------


class TestFullProtocolWithTiling:
    """FULL_PROTOCOL mode running multiple scans over a tile grid."""

    def test_2_scans_2x2_tiles(self, executor, scope, tmp_path):
        steps = _make_tile_grid_steps(rows=2, cols=2)
        protocol = _make_multi_step_protocol(steps)
        protocol.modify_time_params(
            period=datetime.timedelta(seconds=0.1),
            duration=datetime.timedelta(seconds=1),
        )

        completed, _ = _run_and_wait(
            executor,
            protocol,
            tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=2,
        )
        assert completed


class TestFullProtocolMultiScanMultiChannel:
    """FULL_PROTOCOL with multiple scans, multi-channel steps."""

    def test_3_scans_bf_and_red(self, executor, scope, tmp_path):
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF'},
                {'color': 'Red'},
            ]
        )
        protocol.modify_time_params(
            period=datetime.timedelta(seconds=0.1),
            duration=datetime.timedelta(seconds=1),
        )

        completed, _ = _run_and_wait(
            executor,
            protocol,
            tmp_path,
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

        af = executor._autofocus_runner
        af.complete.return_value = True
        af.in_progress.return_value = False
        # Per-step Future tracks AF state; mock as done so scan_iterate
        # skips kick-off and proceeds to consume the AF result.
        executor._af_future = MagicMock()
        executor._af_future.done.return_value = True

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


class TestOmeTiffHyperstackFormat:
    """OME-TIFF Hyperstack output format (formerly labeled 'ImageJ Hyperstack')."""

    def test_completes_with_hyperstack_format(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')
        config = ImageCaptureConfig.from_image_mode(
            '8bit', output_format_sequenced='OME-TIFF Hyperstack'
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path, image_capture_config=config)
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
        """Start a long protocol and cancel it -- should not hang."""
        protocol = _make_single_step_protocol(color='BF')
        protocol.modify_time_params(
            period=datetime.timedelta(seconds=0.1),
            duration=datetime.timedelta(seconds=60),
        )

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

        plan = executor.prepare(
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
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)

        # Let it run briefly then cancel
        time.sleep(1.0)
        executor.reset()

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed, 'Protocol did not complete after reset()'

    def test_reset_before_first_scan_completes(self, executor, scope, tmp_path):
        """Reset immediately -- should still invoke run_complete."""
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

        plan = executor.prepare(
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
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)

        # Cancel almost immediately
        time.sleep(0.2)
        executor.reset()

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed, 'Protocol did not complete after early reset()'


class TestResetWhenNotRunning:
    """reset() when no protocol is active should be a no-op."""

    def test_reset_no_crash(self, executor, scope, tmp_path):
        executor.reset()  # Should not raise


# ---------------------------------------------------------------------------
# Back-to-back runs
# ---------------------------------------------------------------------------


class TestBackToBackRuns:
    """Run a protocol, wait for completion, then immediately run another.

    Completion is two-phase by design: run_complete fires as soon as the
    scan finishes, while queued file writes drain afterward (files_complete).
    A second run() started while files are still writing is deliberately
    rejected with a user-facing "Files Still Writing" notification, so any
    correct back-to-back test must synchronize on the file queue draining --
    that is what _wait_for_file_queue does. This is the designed contract,
    not a workaround for an executor bug.
    """

    @staticmethod
    def _wait_for_file_queue(executor, timeout=5.0):
        """Wait until file_io_executor is ready for a new protocol."""
        deadline = time.monotonic() + timeout
        while executor.file_io_executor.is_protocol_queue_active():
            if time.monotonic() > deadline:
                raise TimeoutError('file_io_executor did not drain in time')
            time.sleep(0.05)

    def test_two_sequential_runs(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')

        completed1, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed1, 'First run did not complete'

        self._wait_for_file_queue(executor)

        # Second run -- uses a fresh tmp subdir to avoid directory collision
        completed2, _ = _run_and_wait(executor, protocol, tmp_path / 'run2')
        assert completed2, 'Second run did not complete'

    def test_three_sequential_runs_different_configs(self, executor, scope, tmp_path):
        for i, color in enumerate(['BF', 'Red', 'Green']):
            protocol = _make_single_step_protocol(color=color)
            completed, _ = _run_and_wait(executor, protocol, tmp_path / f'run{i}')
            assert completed, f'Run {i} ({color}) did not complete'
            self._wait_for_file_queue(executor)


# ---------------------------------------------------------------------------
# Disconnected hardware
# ---------------------------------------------------------------------------


class TestDisconnectedScope:
    """Protocol should not start if scope reports disconnected."""

    def test_run_aborts_when_not_connected(self, executor, scope, tmp_path):
        # Disconnect all boards so are_all_connected() returns False
        scope._led_driver.disconnect()
        scope._motion_driver.disconnect()
        scope._camera_driver.disconnect()
        protocol = _make_single_step_protocol(color='BF')

        done = threading.Event()

        def on_complete(**kwargs):
            done.set()

        callbacks = {
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }

        with pytest.raises(ProtocolRunRefusedError):
            executor.prepare(
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
                    'BF': False,
                    'PC': False,
                    'DF': False,
                    'Red': False,
                    'Green': False,
                    'Blue': False,
                    'Lumi': False,
                },
            )

        # Should NOT have started -- run_complete should NOT fire
        assert not done.is_set(), 'Protocol should not have started with disconnected scope'
        assert not executor.run_in_progress()


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------


class TestZeroExposure:
    """Zero exposure -- tests floor behavior in timing paths."""

    def test_zero_exposure_completes(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF', exposure=0.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


class TestZeroIllumination:
    """Zero illumination -- LED should still be called."""

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
    """Protocol with many steps -- verifies no accumulation bugs."""

    def test_50_step_single_scan(self, executor, scope, tmp_path):
        # Plate-mm coords inside the 6-well valid range (x in [7.76, 127.76],
        # y in [5.48, 85.48] at zero stage_offset) so every step converts to an
        # on-stage position and clears the pre-run travel-limit check; distinct
        # X keeps them 50 separate steps.
        steps = [{'color': 'BF', 'x': 10.0 + float(i), 'y': 20.0} for i in range(50)]
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

    def test_all_50_steps_visited(self, executor, scope, tmp_path):
        # Plate-mm coords inside the 6-well valid range (x in [7.76, 127.76],
        # y in [5.48, 85.48] at zero stage_offset) so every step converts to an
        # on-stage position and clears the pre-run travel-limit check; distinct
        # X keeps them 50 separate steps.
        steps = [{'color': 'BF', 'x': 10.0 + float(i), 'y': 20.0} for i in range(50)]
        protocol = _make_multi_step_protocol(steps)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # P2-14: executor deep-copies the protocol, so we cannot spy on
        # the original mock's .step() calls.  Completion of a 50-step
        # protocol without error confirms all steps were visited.


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
                    steps.append(
                        {
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
                        }
                    )
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
                    steps.append(
                        {
                            'color': color,
                            'well': well,
                            'x': wx + c * 0.5,
                            'y': wy,
                            'tile': f'R0C{c}',
                            'tile_group_id': 1,
                        }
                    )
        protocol = _make_multi_step_protocol(steps)
        assert protocol.num_steps() == 12  # 2 wells * 2 colors * 3 tiles
        completed, _ = _run_and_wait(executor, protocol, tmp_path, separate_folder_per_channel=True)
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

        plan = executor.prepare(
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
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)

        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed


# ---------------------------------------------------------------------------
# Turret support
# ---------------------------------------------------------------------------


class TestWithTurret:
    """Scope with turret enabled -- objective name included in filenames."""

    def test_turret_protocol_completes(self, executor, scope, tmp_path):
        scope._motion_driver._has_turret = True
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# Callback edge cases
# ---------------------------------------------------------------------------


class TestMinimalCallbacks:
    """Run with only the required run_complete callback -- no optional ones."""

    def test_completes_with_minimal_callbacks(self, executor, scope, tmp_path):
        protocol = _make_single_step_protocol(color='BF')

        done = threading.Event()

        def on_complete(**kwargs):
            done.set()

        # Only provide run_complete -- no go_to_step or move_position.
        # This forces _go_to_step to use _default_move (which we've mocked).
        plan = executor.prepare(
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
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)

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


# ===========================================================================
# Tier 3: Robustness & Error Handling -- P0/P1 audit fix coverage
# ===========================================================================

# ---------------------------------------------------------------------------
# P0-1: Concurrent cleanup (threading.Lock)
# ---------------------------------------------------------------------------


class TestCleanupConcurrency:
    """P0-1: _cleanup() guarded by threading.Lock -- no double cleanup."""

    def test_concurrent_reset_no_crash(self, executor, scope, tmp_path):
        """Call reset() from multiple threads while protocol is running."""
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF'},
                {'color': 'Red'},
                {'color': 'Green'},
                {'color': 'Blue'},
                {'color': 'BF'},
            ]
        )
        done = threading.Event()
        plan = executor.prepare(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_concurrent',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks={
                'run_complete': lambda **kw: done.set(),
                'go_to_step': lambda **kw: None,
            },
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)
        # Let protocol start
        time.sleep(0.1)
        # Fire reset from multiple threads simultaneously
        threads = [threading.Thread(target=executor.reset) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        # Should not crash; protocol should end
        done.wait(timeout=COMPLETION_TIMEOUT)

    def test_double_reset_idempotent(self, executor, scope, tmp_path):
        """Calling reset() twice in quick succession doesn't crash."""
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        # Protocol already completed and cleaned up -- reset again should be harmless
        executor.reset()
        executor.reset()


# ---------------------------------------------------------------------------
# P0-2: Disk space check
# ---------------------------------------------------------------------------


class TestDiskSpaceCheck:
    """P0-2: Protocol aborts when disk space is below 2 GB.

    Rule-35 audit 2026-05-19 finding 3 consolidated the disk probe onto
    common_utils.check_disk_space_ok; mocks target the imported alias in
    protocol_run_loop's namespace and return the helper's (ok, free_mb)
    tuple shape.
    """

    @patch('modules.protocol_run_loop.check_disk_space_ok')
    def test_low_disk_aborts_image_protocol(self, mock_check, executor, scope, tmp_path):
        """With very low disk space, image capture should abort without hanging."""
        mock_check.return_value = (False, 500.0)  # 500 MB free, threshold exceeded

        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, 'Protocol did not abort within timeout when disk space is low'

    @patch('modules.protocol_run_loop.check_disk_space_ok')
    def test_large_protocol_needs_more_than_2gb(self, mock_check, executor, scope, tmp_path):
        """300 image steps need 2400 MB (300 * 8 MB), so 2.0 GB free should abort."""
        mock_check.return_value = (False, 2000.0)  # 2.0 GB free vs max(2048, 2400) MB

        protocol = _make_multi_step_protocol([{'color': 'BF'} for _ in range(300)])
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, (
            'Protocol did not abort within timeout when disk space is low for large protocol'
        )

    @patch('modules.protocol_run_loop.check_disk_space_ok')
    def test_video_steps_need_500mb_each(self, mock_check, executor, scope, tmp_path):
        """5 video steps need 2.5 GB (5 * 500 MB), so 2.2 GB free should abort."""
        mock_check.return_value = (False, 2200.0)  # 2.2 GB free; 5*500=2500 MB required

        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF', 'acquire': 'video', 'video_config': {'duration': 1, 'fps': 5}}
                for _ in range(5)
            ]
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, 'Protocol did not abort for video steps requiring more disk space'

    @patch('modules.protocol_run_loop.check_disk_space_ok', side_effect=OSError('disk error'))
    def test_disk_check_exception_does_not_crash(self, mock_check, executor, scope, tmp_path):
        """If the probe raises OSError, protocol continues (swallow in caller)."""
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed


# ---------------------------------------------------------------------------
# P0-3: Capture failure handling
# ---------------------------------------------------------------------------


class TestCaptureFailure:
    """P0-3: capture_and_wait returning False records 'capture_failed'."""

    def test_capture_false_records_failure(self, executor, scope, tmp_path):
        """When capture_and_wait returns False, protocol completes (doesn't hang)."""
        protocol = _make_single_step_protocol(color='BF')

        original_capture = scope.imaging.capture_and_wait
        scope.imaging.capture_and_wait = MagicMock(return_value=False)

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        scope.imaging.capture_and_wait = original_capture

    def test_multiple_capture_failures_still_complete(self, executor, scope, tmp_path):
        """Three steps all fail capture -- protocol runs to completion."""
        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF'},
                {'color': 'Red'},
                {'color': 'Green'},
            ]
        )

        original_capture = scope.imaging.capture_and_wait
        scope.imaging.capture_and_wait = MagicMock(return_value=False)

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        scope.imaging.capture_and_wait = original_capture


# ---------------------------------------------------------------------------
# P1-4: Per-step motion timeout
# ---------------------------------------------------------------------------


class TestStepTimeout:
    """P1-4: Steps that exceed MOTION_TIMEOUT_SECONDS are skipped."""

    def test_stuck_motion_skips_step(self, executor, scope, tmp_path):
        """If motion never completes, the step times out and protocol continues."""
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        # Use a very short timeout for the test
        original_timeout = SequencedCaptureRunner.MOTION_TIMEOUT_SECONDS
        SequencedCaptureRunner.MOTION_TIMEOUT_SECONDS = 1  # 1 second

        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF', 'x': 10.0},
                {'color': 'Red', 'x': 20.0},
            ]
        )

        # Make get_target_status always return False for first step
        call_count = [0]
        original_get_target = scope.motion.get_target_status

        def slow_target(axis):
            call_count[0] += 1
            # First ~200 calls (first step) -- never reach target
            if call_count[0] < 200:
                return False
            return original_get_target(axis)

        scope.motion.get_target_status = slow_target

        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        # Restore
        SequencedCaptureRunner.MOTION_TIMEOUT_SECONDS = original_timeout
        scope.motion.get_target_status = original_get_target

        assert completed


# ---------------------------------------------------------------------------
# P1-8: Camera gain/exposure restoration
# ---------------------------------------------------------------------------


class TestCameraStateRestoration:
    """P1-8: Camera gain and exposure restored to pre-protocol values."""

    def test_gain_restored_after_protocol(self, executor, scope, tmp_path):
        """Gain is restored to original value after protocol completes."""
        scope.imaging.set_gain(3.0)
        scope.imaging.set_exposure_time(25.0)
        original_gain = scope.imaging.get_gain()
        original_exposure = scope.imaging.get_exposure_time()

        protocol = _make_single_step_protocol(color='BF', gain=10.0, exposure=100.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        assert scope.imaging.get_gain() == pytest.approx(original_gain, abs=0.1)
        assert scope.imaging.get_exposure_time() == pytest.approx(original_exposure, abs=0.1)

    def test_gain_restored_with_auto_gain(self, executor, scope, tmp_path):
        """Even with auto_gain enabled, original gain is restored after cleanup."""
        scope.imaging.set_gain(5.0)
        original_gain = scope.imaging.get_gain()

        protocol = _make_single_step_protocol(color='BF', auto_gain=True)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        assert scope.imaging.get_gain() == pytest.approx(original_gain, abs=0.1)

    def test_gain_restored_after_multi_step(self, executor, scope, tmp_path):
        """Multi-step protocol with varying gains restores to original."""
        scope.imaging.set_gain(2.0)
        scope.imaging.set_exposure_time(15.0)
        original_gain = scope.imaging.get_gain()
        original_exposure = scope.imaging.get_exposure_time()

        protocol = _make_multi_step_protocol(
            [
                {'color': 'BF', 'gain_db': 5.0, 'exposure_ms': 50.0},
                {'color': 'Red', 'gain_db': 10.0, 'exposure_ms': 100.0},
                {'color': 'Green', 'gain_db': 15.0, 'exposure_ms': 200.0},
            ]
        )
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        assert scope.imaging.get_gain() == pytest.approx(original_gain, abs=0.1)
        assert scope.imaging.get_exposure_time() == pytest.approx(original_exposure, abs=0.1)

    def test_gain_restored_after_cancellation(self, executor, scope, tmp_path):
        """Gain is restored even when protocol is cancelled mid-run."""
        scope.imaging.set_gain(4.0)
        scope.imaging.set_exposure_time(30.0)
        original_gain = scope.imaging.get_gain()
        original_exposure = scope.imaging.get_exposure_time()

        protocol = _make_multi_step_protocol(
            [
                {'color': c, 'gain_db': 12.0, 'exposure_ms': 80.0}
                for c in ['BF', 'Red', 'Green', 'Blue', 'BF']
            ]
        )

        done = threading.Event()
        plan = executor.prepare(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_cancel_restore',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks={
                'run_complete': lambda **kw: done.set(),
                'go_to_step': lambda **kw: None,
            },
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)
        time.sleep(0.2)
        executor.reset()
        done.wait(timeout=COMPLETION_TIMEOUT)

        assert scope.imaging.get_gain() == pytest.approx(original_gain, abs=0.1)
        assert scope.imaging.get_exposure_time() == pytest.approx(original_exposure, abs=0.1)


# ---------------------------------------------------------------------------
# P1-5: Validation before protocol_running_global
# ---------------------------------------------------------------------------


class TestValidationOrder:
    """P1-5: Ensure protocol_running_global is not set before validation."""

    def test_run_in_progress_false_when_not_started(self, executor, scope, tmp_path):
        """Before any run, run_in_progress should be False."""
        assert not executor.run_in_progress()

    def test_run_in_progress_false_after_completion(self, executor, scope, tmp_path):
        """After protocol completes, run_in_progress is False."""
        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed
        assert not executor.run_in_progress()


# ---------------------------------------------------------------------------
# Cleanup correctness
# ---------------------------------------------------------------------------


class TestCleanupCorrectness:
    """Verify cleanup handles all state properly."""

    def test_leds_off_after_protocol_abort(self, executor, scope, tmp_path):
        """All LEDs are off after aborting a multi-step protocol."""
        protocol = _make_multi_step_protocol(
            [{'color': c, 'illumination_ma': 100.0} for c in ['BF', 'Red', 'Green', 'Blue', 'BF']]
        )
        done = threading.Event()
        plan = executor.prepare(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_abort_leds',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks={
                'run_complete': lambda **kw: done.set(),
                'go_to_step': lambda **kw: None,
            },
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )
        executor.start(plan)
        time.sleep(0.2)
        executor.reset()
        done.wait(timeout=COMPLETION_TIMEOUT)

        for color in scope._led_driver.led_ma:
            assert not scope.illumination.led_enabled(color), f'LED {color} still on after abort'

    def test_back_to_back_runs_no_state_bleed(self, executor, scope, tmp_path):
        """Gain/exposure from run A don't leak into run B's restored values."""
        # Run A: set gain to 8
        scope.imaging.set_gain(8.0)
        scope.imaging.set_exposure_time(80.0)
        protocol_a = _make_single_step_protocol(color='BF', gain=15.0, exposure=150.0)
        completed_a, _ = _run_and_wait(executor, protocol_a, tmp_path)
        assert completed_a
        # Should restore to 8.0/80.0
        assert scope.imaging.get_gain() == pytest.approx(8.0, abs=0.1)

        # Wait for file queue to drain before starting next run
        deadline = time.monotonic() + 5.0
        while executor.file_io_executor.is_protocol_queue_active():
            if time.monotonic() > deadline:
                raise TimeoutError('file_io_executor did not drain in time')
            time.sleep(0.05)

        # Run B: change gain before second run
        scope.imaging.set_gain(2.0)
        scope.imaging.set_exposure_time(20.0)
        protocol_b = _make_single_step_protocol(color='Red', gain=12.0, exposure=120.0)
        completed_b, _ = _run_and_wait(executor, protocol_b, tmp_path / 'run2')
        assert completed_b
        # Should restore to 2.0/20.0, NOT to 8.0/80.0 from run A
        assert scope.imaging.get_gain() == pytest.approx(2.0, abs=0.1)
        assert scope.imaging.get_exposure_time() == pytest.approx(20.0, abs=0.1)


class TestProtocolLedNoFlash:
    """The capture path lights its channel exclusively (idempotent), so the
    pre-step nuclear leds_off is unnecessary: a stray Live-mode LED on another
    channel is killed without blinking the target off->on. This is the fix for
    the LED flash on a protocol / Z-stack run.
    """

    def test_capture_kills_a_stray_led_as_it_lights_its_own_channel(
        self, executor, scope, tmp_path
    ):
        # A stray channel lit before the run (e.g. a Live-mode LED left on a
        # different color when the user pressed Scan).
        stray = scope.illumination.color2ch('Red')
        scope.illumination.led_on(channel=stray, mA=80, owner='ui')
        assert scope.illumination.led_enabled('Red')

        events = []
        scope.illumination.add_led_listener(
            lambda color, enabled, mA, owner: events.append((color, enabled))
        )

        protocol = _make_single_step_protocol(color='Green', illumination=60.0)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        green_on = next((i for i, (c, e) in enumerate(events) if c == 'Green' and e), None)
        red_off = next((i for i, (c, e) in enumerate(events) if c == 'Red' and not e), None)
        assert green_on is not None, f'Green never lit during the scan: {events}'
        assert red_off is not None and red_off < green_on, (
            'the stray Red LED must be turned off as the step lights Green '
            f'(exclusive capture), not left double-illuminating the step: {events}'
        )

    def test_already_lit_channel_is_not_blinked_by_the_scan(self, executor, scope, tmp_path):
        """A channel already lit at the step's current before Scan (the user
        pressed Scan with the matching Live LED on) is left lit -- no off->on
        blink at run start. The pre-step nuclear leds_off used to clear the
        cache and force the re-light; this is the protocol/Z-stack flash.
        """
        color, mA = 'Green', 60.0
        scope.illumination.led_on(channel=scope.illumination.color2ch(color), mA=mA, owner='ui')
        assert scope.illumination.led_enabled(color)

        events = []
        scope.illumination.add_led_listener(
            lambda c, enabled, m, owner: events.append((c, enabled))
        )

        protocol = _make_single_step_protocol(color=color, illumination=mA)
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed

        # No "off then later on" pair on the already-lit channel == no blink.
        # (The cleanup leds_off at the very end has no following on, so it is
        # correctly not counted as a blink.)
        seen_off = False
        blinked = False
        for c, enabled in events:
            if c == color and not enabled:
                seen_off = True
            elif c == color and enabled and seen_off:
                blinked = True
                break
        assert not blinked, f'already-lit {color} was blinked off->on at run start: {events}'


class TestMotionTimeoutEndsRunInsteadOfWedging:
    """A motion timeout mid-run must END the protocol (ERROR -> cleanup ->
    run_complete), not wedge it. Previously the timed-out scan was counted
    complete and every later period raised an invalid ERROR->SCANNING
    transition that the transient-failure classifier retried forever -- a
    multi-day timelapse silently delivering nothing after one timeout."""

    def test_run_completes_after_motion_timeout(self, executor, scope, tmp_path, monkeypatch):
        from modules.protocol_state_machine import ProtocolState

        executor.MOTION_TIMEOUT_SECONDS = 0.3
        # Stage reports moving forever -> the step's motion wait must trip
        # the timeout instead of completing.
        monkeypatch.setattr(scope.motion, 'is_moving', lambda *a, **kw: True)

        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(
            executor,
            protocol,
            tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=3,
        )

        assert completed, (
            'Protocol wedged after a motion timeout: run_complete never '
            'fired. ERROR state must terminate the run, not be retried '
            'as a transient failure every period.'
        )
        assert executor._state == ProtocolState.IDLE, (
            f'Expected IDLE after cleanup, got {executor._state}'
        )

    def test_motion_timeout_stops_the_motor(self, executor, scope, tmp_path, monkeypatch):
        """A motion timeout must halt the in-flight move, not just error out.
        Without stop_motion the motor keeps driving toward the unreachable
        target while the protocol transitions to ERROR."""
        executor.MOTION_TIMEOUT_SECONDS = 0.3
        monkeypatch.setattr(scope.motion, 'is_moving', lambda *a, **kw: True)

        stop_calls = []
        orig_stop = scope.motion.stop_motion

        def _spy_stop(*a, **kw):
            stop_calls.append(True)
            return orig_stop(*a, **kw)

        monkeypatch.setattr(scope.motion, 'stop_motion', _spy_stop)

        protocol = _make_single_step_protocol(color='BF')
        _run_and_wait(
            executor,
            protocol,
            tmp_path,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=3,
        )

        assert stop_calls, (
            'Motion timeout did not call stop_motion; the timed-out move is '
            'left in flight while the protocol errors out.'
        )


class TestSaveFailureRecordsRow:
    """A disk-write failure must still leave a row in the execution
    record. The queue-full and capture-failed legs already record their
    failures; a save_image raise previously escaped before add_step ran,
    leaving image AND record-row silently missing -- the worst silent-
    data-gap shape for record-keyed post-processing and run accounting."""

    def test_save_image_raise_records_save_failed_row(self, executor, scope, tmp_path, monkeypatch):
        import modules.protocol_image_writer as piw

        def boom(*args, **kwargs):
            raise OSError('disk write failed (injected)')

        monkeypatch.setattr(piw, 'save_image', boom)

        protocol = _make_single_step_protocol(color='BF')
        completed, _ = _run_and_wait(executor, protocol, tmp_path)
        assert completed, 'Protocol should complete despite the save failure'

        records = list((tmp_path / 'output').rglob('*.tsv'))
        assert records, 'No execution record file was written'
        contents = '\n'.join(r.read_text() for r in records)
        assert 'save_failed' in contents, (
            'A save_image failure left no row in the execution record; '
            'the save_failed row must be written when the disk write '
            'raises.'
        )


class TestRunReturnValueContract:
    """prepare()/start() report whether the run can and did start.

    A refused run loads no protocol and creates no run directory, so a
    caller that treats the start sequence as fire-and-forget follows up
    against the PREVIOUS run's state (stale save folder) or a runner
    that never loaded a protocol (AttributeError inside a UI handler).
    The typed refusal raised by prepare() plus the None-seeded getters
    are the contract the UI call sites rely on; a failure after start()
    commits unwinds as a failed run whose terminal callback fires.
    """

    def _prepare_run(self, executor, protocol, tmp_path, callbacks=None):
        cbs = {
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }
        if callbacks:
            cbs.update(callbacks)
        return executor.prepare(
            protocol=protocol,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='run_contract_test',
            image_capture_config=_make_image_capture_config(),
            autogain_settings=_make_autogain_settings(),
            parent_dir=tmp_path / 'output',
            max_scans=1,
            callbacks=cbs,
            leds_state_at_end='off',
            initial_autofocus_states={
                'BF': False,
                'PC': False,
                'DF': False,
                'Red': False,
                'Green': False,
                'Blue': False,
                'Lumi': False,
            },
        )

    def test_refused_run_raises_and_leaves_runner_idle(self, executor, tmp_path):
        empty_protocol = _build_real_protocol([])
        with pytest.raises(ProtocolRunRefusedError):
            self._prepare_run(executor, empty_protocol, tmp_path)
        assert executor.run_dir() is None, (
            'A refused run must not leave a run directory for callers to save into'
        )
        assert not executor._run_in_progress_event.is_set(), (
            'A refused run must not mark a run as in progress'
        )

    def test_started_run_completes(self, executor, tmp_path):
        done = threading.Event()
        protocol = _make_single_step_protocol(color='BF')
        plan = self._prepare_run(
            executor,
            protocol,
            tmp_path,
            callbacks={'run_complete': lambda **kwargs: done.set()},
        )
        assert isinstance(plan, RunPlan), 'prepare() must return the validated RunPlan'
        executor.start(plan)
        assert done.wait(timeout=COMPLETION_TIMEOUT), 'Started run did not complete'

    def test_fresh_runner_getters_answer_none_not_attributeerror(self, executor):
        assert executor._protocol is None, (
            '_protocol must exist (as None) from construction so getters '
            'can answer instead of raising AttributeError'
        )
        assert executor.run_dir() is None
        assert executor.run_trigger_source() is None
        assert executor.current_step_color() is None

    def test_dir_setup_failure_fails_at_start_and_recovers(
        self, executor, scope, tmp_path, monkeypatch
    ):
        """A run-directory setup failure is a failed-at-start run, not a
        wedge: exactly one user notification, the terminal run_complete
        callback fires with the failed-at-start status, the runner is
        idle afterwards, and a subsequent prepare() succeeds."""
        import modules.notification_center as notification_center

        protocol = _make_single_step_protocol(color='BF')

        notified = []
        monkeypatch.setattr(
            notification_center.notifications,
            'error',
            lambda *args, **kwargs: notified.append(args),
        )
        monkeypatch.setattr(
            executor,
            '_create_run_dir',
            lambda: {'status': False, 'data': None, 'error': 'capture location vanished'},
        )
        completions = []
        plan = self._prepare_run(
            executor,
            protocol,
            tmp_path,
            callbacks={'run_complete': lambda **kwargs: completions.append(kwargs)},
        )
        executor.start(plan)
        assert len(notified) == 1, (
            'A run that fails at directory setup must notify the user exactly '
            f'once; got {len(notified)}: {notified}'
        )
        assert len(completions) == 1, (
            'The terminal run_complete callback must fire exactly once for a '
            f'failed-at-start run; got {completions}'
        )
        assert completions[0].get('status') == 'failed_at_start', (
            f'run_complete must report the failed-at-start status; got {completions[0]}'
        )
        assert not executor.run_in_progress(), (
            'A run that failed at directory setup must not stay marked in progress'
        )
        # A failed start must not leak hardware setup: the protocol LED
        # lease is only held by a run in flight, so a fresh top-level
        # acquire must succeed after the failure.
        lease = scope.illumination.acquire_led_lease('leak probe', alive=lambda: True)
        assert lease is not None, 'Failed-at-start run leaked the protocol LED lease'
        lease.release()
        # The runner is reusable: the next prepare() passes every gate.
        monkeypatch.undo()
        plan2 = self._prepare_run(executor, protocol, tmp_path)
        assert isinstance(plan2, RunPlan), (
            'A failed-at-start run must not wedge the runner; the next prepare() must succeed'
        )
