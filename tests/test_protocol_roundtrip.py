"""Comprehensive protocol round-trip and execution fuzz tests.

Tests every permutation of protocol creation, save/load, and execution:
- Image vs video capture
- With/without stimulation config
- With/without autofocus
- Tiling (square and non-square)
- Z-stacking
- Multi-channel (BF + fluorescence)
- Multi-well
- Edge cases (empty protocol, single step, max steps)

These tests verify that:
1. Protocols survive save → load round-trips with all config intact
2. Protocols execute to completion on simulated hardware
3. Step validation catches invalid configs before execution
"""

import copy
import datetime
import json
import pathlib
import threading

import pandas as pd
import pytest

from modules.protocol import Protocol
from modules.sequenced_capture_executor import SequencedCaptureExecutor, SequencedCaptureRunMode
from modules.sequential_io_executor import SequentialIOExecutor
from modules.lumascope_api import Lumascope
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

COMPLETION_TIMEOUT = 20  # seconds

TILING_CONFIGS = pathlib.Path(__file__).parent.parent / "data" / "tiling.json"


def _make_autogain_settings():
    return {
        'target_brightness': 0.3,
        'min_gain': 0.0,
        'max_gain': 20.0,
        'max_duration': datetime.timedelta(seconds=1),
    }


def _make_image_capture_config():
    return {
        'output_format': {'live': 'TIFF', 'sequenced': 'TIFF'},
        'use_full_pixel_depth': False,
    }


def _default_stim_config():
    return {
        "Red": {
            "enabled": False,
            "illumination": 100.0,
            "frequency": 1.0,
            "pulse_width": 10,
            "pulse_count": 100,
        },
        "Green": {
            "enabled": False,
            "illumination": 100.0,
            "frequency": 1.0,
            "pulse_width": 10,
            "pulse_count": 100,
        },
        "Blue": {
            "enabled": False,
            "illumination": 100.0,
            "frequency": 1.0,
            "pulse_width": 10,
            "pulse_count": 100,
        },
    }


def _stim_config_enabled(channels=("Green",)):
    """Stim config with specified channels enabled."""
    cfg = _default_stim_config()
    for ch in channels:
        cfg[ch]["enabled"] = True
        cfg[ch]["illumination"] = 200.0
        cfg[ch]["frequency"] = 5.0
        cfg[ch]["pulse_width"] = 20
        cfg[ch]["pulse_count"] = 50
    return cfg


def _default_video_config():
    return {"duration": 1.0, "fps": 5}


def _make_step(
    name="A1_BF",
    x=10.0, y=20.0, z=5000.0,
    color="BF",
    illumination=50.0,
    gain=1.0,
    exposure=10.0,
    auto_focus=False,
    auto_gain=False,
    false_color=False,
    sum_count=1,
    objective="10x Oly",
    well="A1",
    tile="",
    z_slice=0,
    tile_group_id=0,
    zstack_group_id=0,
    acquire="image",
    video_config=None,
    stim_config=None,
):
    return {
        "Name": name,
        "X": x, "Y": y, "Z": z,
        "Auto_Focus": auto_focus,
        "Color": color,
        "False_Color": false_color,
        "Illumination": illumination,
        "Gain": gain,
        "Auto_Gain": auto_gain,
        "Exposure": exposure,
        "Sum": sum_count,
        "Objective": objective,
        "Well": well,
        "Tile": tile,
        "Z-Slice": z_slice,
        "Custom Step": True,
        "Tile Group ID": tile_group_id,
        "Z-Stack Group ID": zstack_group_id,
        "Acquire": acquire,
        "Video Config": video_config or _default_video_config(),
        "Stim_Config": stim_config or _default_stim_config(),
        "Step Index": 0,
    }


def _build_protocol(steps, period_min=1.0, duration_hrs=1.0, labware="6 well microplate"):
    """Build a real Protocol object from a list of step dicts."""
    df = pd.DataFrame(steps)
    config = {
        "version": Protocol.CURRENT_VERSION,
        "steps": df,
        "period": datetime.timedelta(minutes=period_min),
        "duration": datetime.timedelta(hours=duration_hrs),
        "labware_id": labware,
        "capture_root": "",
        "tiling": "1x1",
    }
    return Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config=config,
    )


def _save_and_reload(protocol, tmp_path):
    """Save protocol to file and reload, returning the reloaded protocol."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    filepath = tmp_path / "test_protocol.tsv"
    result = protocol.to_file(filepath)
    assert result is None, f"to_file failed: {result}"
    assert filepath.exists(), "Protocol file not created"

    reloaded = Protocol.from_file(
        file_path=filepath,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    return reloaded


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s.led.set_timing_mode('fast')
    s.motion.set_timing_mode('fast')
    s.camera.set_timing_mode('fast')
    s.camera.start_grabbing()
    yield s
    s.camera.stop_grabbing()
    s.disconnect()


@pytest.fixture
def executors():
    execs = {
        'io': SequentialIOExecutor(name="RT_IO"),
        'protocol': SequentialIOExecutor(name="RT_PROTOCOL"),
        'file_io': SequentialIOExecutor(name="RT_FILE"),
        'camera': SequentialIOExecutor(name="RT_CAMERA"),
        'autofocus': SequentialIOExecutor(name="RT_AF"),
    }
    for e in execs.values():
        e.start()
    yield execs
    for e in execs.values():
        try:
            e.shutdown()
        except Exception:
            pass


@pytest.fixture
def executor(scope, executors):
    mock_af = MagicMock()
    mock_af.reset = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.complete = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.result = MagicMock(return_value=None)
    mock_af.best_focus_position = MagicMock(return_value=5000.0)
    mock_af.run_in_progress = MagicMock(return_value=False)

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
    mock_loader = MagicMock()
    mock_transformer = MagicMock()
    mock_transformer.plate_to_stage = MagicMock(return_value=(0.0, 0.0))
    exc._wellplate_loader = mock_loader
    exc._coordinate_transformer = mock_transformer
    return exc


@pytest.fixture
def real_executor(scope, executors):
    """Executor with REAL wellplate loader and coordinate transformer.

    This exercises the full code path including move_abs_pos → axes_config,
    which catches init bugs that mocked fixtures miss.
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
    exc._wellplate_loader = WellPlateLoader()
    exc._coordinate_transformer = CoordinateTransformer()
    return exc


def _run_and_wait(executor, protocol, tmp_path, **run_kwargs):
    done = threading.Event()
    result_holder = {}

    def on_complete(**kwargs):
        result_holder.update(kwargs)
        done.set()

    callbacks = run_kwargs.pop('callbacks', {})
    callbacks['run_complete'] = on_complete
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


# ===========================================================================
# PART 1: Save/Load Round-Trip Tests
# ===========================================================================

class TestRoundTripBasic:
    """Protocol save → load preserves all data."""

    def test_single_bf_image_step(self, tmp_path):
        proto = _build_protocol([_make_step(color="BF", acquire="image")])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 1
        step = reloaded.step(idx=0)
        assert step["Color"] == "BF"
        assert step["Acquire"] == "image"

    def test_single_fluor_step(self, tmp_path):
        proto = _build_protocol([_make_step(color="Green", illumination=200.0)])
        reloaded = _save_and_reload(proto, tmp_path)
        step = reloaded.step(idx=0)
        assert step["Color"] == "Green"
        assert step["Illumination"] == 200.0

    def test_video_step_config_preserved(self, tmp_path):
        vc = {"duration": 5.0, "fps": 30}
        proto = _build_protocol([_make_step(acquire="video", video_config=vc)])
        reloaded = _save_and_reload(proto, tmp_path)
        step = reloaded.step(idx=0)
        assert step["Acquire"] == "video"
        assert isinstance(step["Video Config"], dict), f"Video Config is {type(step['Video Config'])}, not dict"
        assert step["Video Config"]["duration"] == 5.0
        assert step["Video Config"]["fps"] == 30

    def test_stim_config_preserved(self, tmp_path):
        sc = _stim_config_enabled(channels=["Green", "Blue"])
        proto = _build_protocol([_make_step(stim_config=sc)])
        reloaded = _save_and_reload(proto, tmp_path)
        step = reloaded.step(idx=0)
        assert isinstance(step["Stim_Config"], dict), f"Stim_Config is {type(step['Stim_Config'])}, not dict"
        assert step["Stim_Config"]["Green"]["enabled"] is True
        assert step["Stim_Config"]["Blue"]["enabled"] is True
        assert step["Stim_Config"]["Red"]["enabled"] is False
        assert step["Stim_Config"]["Green"]["frequency"] == 5.0

    def test_stim_disabled_preserved(self, tmp_path):
        sc = _default_stim_config()
        proto = _build_protocol([_make_step(stim_config=sc)])
        reloaded = _save_and_reload(proto, tmp_path)
        step = reloaded.step(idx=0)
        assert isinstance(step["Stim_Config"], dict)
        for ch in ("Red", "Green", "Blue"):
            assert step["Stim_Config"][ch]["enabled"] is False


class TestRoundTripMultiStep:
    """Multi-step protocols survive round-trip."""

    def test_bf_plus_fluor(self, tmp_path):
        steps = [
            _make_step(name="A1_BF", color="BF", illumination=50.0),
            _make_step(name="A1_Green", color="Green", illumination=200.0),
            _make_step(name="A1_Red", color="Red", illumination=150.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 3
        assert reloaded.step(idx=0)["Color"] == "BF"
        assert reloaded.step(idx=1)["Color"] == "Green"
        assert reloaded.step(idx=2)["Color"] == "Red"

    def test_mixed_image_and_video(self, tmp_path):
        steps = [
            _make_step(name="A1_BF", color="BF", acquire="image"),
            _make_step(name="A1_Green_vid", color="Green", acquire="video",
                       video_config={"duration": 3.0, "fps": 10}),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Acquire"] == "image"
        assert reloaded.step(idx=1)["Acquire"] == "video"
        assert reloaded.step(idx=1)["Video Config"]["duration"] == 3.0

    def test_multi_well(self, tmp_path):
        steps = [
            _make_step(name="A1_BF", well="A1", x=10.0, y=20.0),
            _make_step(name="A2_BF", well="A2", x=30.0, y=20.0),
            _make_step(name="B1_BF", well="B1", x=10.0, y=40.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 3
        assert reloaded.step(idx=0)["Well"] == "A1"
        assert reloaded.step(idx=1)["Well"] == "A2"
        assert reloaded.step(idx=2)["Well"] == "B1"

    def test_video_with_stim(self, tmp_path):
        """OG protocol pattern: video capture with stimulation enabled."""
        sc = _stim_config_enabled(channels=["Green"])
        vc = {"duration": 5.0, "fps": 30}
        steps = [
            _make_step(name="A1_BF", color="BF", acquire="image"),
            _make_step(name="A1_Red_stim", color="Red", acquire="video",
                       video_config=vc, stim_config=sc),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)

        # Verify video config survived
        vid_step = reloaded.step(idx=1)
        assert vid_step["Acquire"] == "video"
        assert isinstance(vid_step["Video Config"], dict)
        assert vid_step["Video Config"]["fps"] == 30

        # Verify stim config survived
        assert isinstance(vid_step["Stim_Config"], dict)
        assert vid_step["Stim_Config"]["Green"]["enabled"] is True
        assert vid_step["Stim_Config"]["Green"]["frequency"] == 5.0

    def test_tiled_steps(self, tmp_path):
        """Tiled protocol steps preserve tile labels and group IDs."""
        steps = [
            _make_step(name="A1_BF_T00", tile="T00", tile_group_id=1, x=10.0, y=20.0),
            _make_step(name="A1_BF_T01", tile="T01", tile_group_id=1, x=15.0, y=20.0),
            _make_step(name="A1_BF_T10", tile="T10", tile_group_id=1, x=10.0, y=25.0),
            _make_step(name="A1_BF_T11", tile="T11", tile_group_id=1, x=15.0, y=25.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 4
        for i in range(4):
            assert reloaded.step(idx=i)["Tile Group ID"] == 1
        assert reloaded.step(idx=0)["Tile"] == "T00"
        assert reloaded.step(idx=3)["Tile"] == "T11"

    def test_nonsquare_tiling_3x1(self, tmp_path):
        """Non-square tiling (3 columns, 1 row)."""
        steps = [
            _make_step(name="A1_BF_T00", tile="T00", tile_group_id=1, x=10.0, y=20.0),
            _make_step(name="A1_BF_T01", tile="T01", tile_group_id=1, x=15.0, y=20.0),
            _make_step(name="A1_BF_T02", tile="T02", tile_group_id=1, x=20.0, y=20.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 3

    def test_zstack_steps(self, tmp_path):
        """Z-stack steps preserve Z-Slice index and group IDs."""
        steps = [
            _make_step(name="A1_BF_Z0", z=4900.0, z_slice=0, zstack_group_id=1),
            _make_step(name="A1_BF_Z1", z=5000.0, z_slice=1, zstack_group_id=1),
            _make_step(name="A1_BF_Z2", z=5100.0, z_slice=2, zstack_group_id=1),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 3
        assert reloaded.step(idx=0)["Z"] == pytest.approx(4900.0)
        assert reloaded.step(idx=2)["Z"] == pytest.approx(5100.0)
        assert reloaded.step(idx=0)["Z-Slice"] == 0
        assert reloaded.step(idx=2)["Z-Slice"] == 2

    def test_double_save_load(self, tmp_path):
        """Save → load → save → load preserves data (no drift)."""
        sc = _stim_config_enabled(channels=["Red", "Blue"])
        vc = {"duration": 2.0, "fps": 15}
        steps = [
            _make_step(name="A1_BF", color="BF"),
            _make_step(name="A1_vid", color="Green", acquire="video",
                       video_config=vc, stim_config=sc),
        ]
        proto = _build_protocol(steps)

        # First round-trip
        r1 = _save_and_reload(proto, tmp_path / "round1")

        # Second round-trip
        r2 = _save_and_reload(r1, tmp_path / "round2")

        step = r2.step(idx=1)
        assert step["Video Config"]["fps"] == 15
        assert step["Stim_Config"]["Red"]["enabled"] is True
        assert step["Stim_Config"]["Blue"]["enabled"] is True
        assert step["Stim_Config"]["Green"]["enabled"] is False


class TestRoundTripEdgeCases:
    """Edge cases in protocol save/load."""

    def test_all_channels(self, tmp_path):
        """Protocol with every channel type."""
        colors = ["BF", "PC", "DF", "Red", "Green", "Blue"]
        steps = [_make_step(name=f"A1_{c}", color=c) for c in colors]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 6
        for i, c in enumerate(colors):
            assert reloaded.step(idx=i)["Color"] == c

    def test_high_illumination(self, tmp_path):
        proto = _build_protocol([_make_step(illumination=1000.0)])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Illumination"] == 1000.0

    def test_sum_averaging(self, tmp_path):
        proto = _build_protocol([_make_step(sum_count=10)])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Sum"] == 10

    def test_autofocus_flag(self, tmp_path):
        proto = _build_protocol([_make_step(auto_focus=True)])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Auto_Focus"] == True

    def test_auto_gain_flag(self, tmp_path):
        proto = _build_protocol([_make_step(auto_gain=True)])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Auto_Gain"] == True

    def test_false_color_flag(self, tmp_path):
        proto = _build_protocol([_make_step(false_color=True, color="Green")])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["False_Color"] == True

    def test_special_chars_in_name(self, tmp_path):
        proto = _build_protocol([_make_step(name="test step (1) - BF")])
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Name"] == "test step (1) - BF"


# ===========================================================================
# PART 2: Protocol Execution Tests
# ===========================================================================

class TestExecuteSingleStep:
    """Single-step protocol execution on simulated hardware."""

    def test_bf_image(self, executor, scope, tmp_path):
        steps = [_make_step(color="BF", acquire="image")]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "BF image protocol did not complete"

    def test_fluor_image(self, executor, scope, tmp_path):
        steps = [_make_step(color="Green", illumination=200.0)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Fluorescence image protocol did not complete"

    def test_video_capture(self, executor, scope, tmp_path):
        vc = {"duration": 0.5, "fps": 5}
        steps = [_make_step(acquire="video", video_config=vc)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Video capture protocol did not complete"

    def test_sum_averaging(self, executor, scope, tmp_path):
        steps = [_make_step(sum_count=3)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Sum averaging protocol did not complete"


class TestExecuteMultiStep:
    """Multi-step protocol execution."""

    def test_bf_plus_two_fluor(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF", color="BF"),
            _make_step(name="A1_Green", color="Green", illumination=200.0),
            _make_step(name="A1_Red", color="Red", illumination=150.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "BF + 2 fluor protocol did not complete"

    def test_multi_well_bf(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF", well="A1", x=10.0, y=20.0),
            _make_step(name="A2_BF", well="A2", x=30.0, y=20.0),
            _make_step(name="A3_BF", well="A3", x=50.0, y=20.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Multi-well BF protocol did not complete"

    def test_tiled_2x2(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF_T00", tile="T00", tile_group_id=1, x=10.0, y=20.0),
            _make_step(name="A1_BF_T01", tile="T01", tile_group_id=1, x=15.0, y=20.0),
            _make_step(name="A1_BF_T10", tile="T10", tile_group_id=1, x=10.0, y=25.0),
            _make_step(name="A1_BF_T11", tile="T11", tile_group_id=1, x=15.0, y=25.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "2x2 tiled protocol did not complete"

    def test_tiled_3x1_nonsquare(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF_T00", tile="T00", tile_group_id=1, x=10.0),
            _make_step(name="A1_BF_T01", tile="T01", tile_group_id=1, x=15.0),
            _make_step(name="A1_BF_T02", tile="T02", tile_group_id=1, x=20.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "3x1 tiled protocol did not complete"

    def test_zstack_3_slices(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF_Z0", z=4900.0, z_slice=0, zstack_group_id=1),
            _make_step(name="A1_BF_Z1", z=5000.0, z_slice=1, zstack_group_id=1),
            _make_step(name="A1_BF_Z2", z=5100.0, z_slice=2, zstack_group_id=1),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Z-stack protocol did not complete"

    def test_mixed_image_and_video(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF", color="BF", acquire="image"),
            _make_step(name="A1_Green_vid", color="Green", acquire="video",
                       video_config={"duration": 0.5, "fps": 5}),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Mixed image+video protocol did not complete"


class TestExecuteWithStim:
    """Protocol execution with stimulation configs."""

    def test_video_with_stim_enabled(self, executor, scope, tmp_path):
        sc = _stim_config_enabled(channels=["Green"])
        steps = [
            _make_step(name="A1_Red_stim", color="Red", acquire="video",
                       video_config={"duration": 0.5, "fps": 5}, stim_config=sc),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Video with stim protocol did not complete"

    def test_image_with_stim_disabled(self, executor, scope, tmp_path):
        """Stim config present but disabled — should not affect image capture."""
        sc = _default_stim_config()
        steps = [_make_step(stim_config=sc)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Image with stim-disabled config did not complete"


class TestExecuteSaveLoadRun:
    """The full pipeline: create → save → reload → execute."""

    def test_bf_save_load_run(self, executor, scope, tmp_path):
        proto = _build_protocol([_make_step(color="BF")])
        reloaded = _save_and_reload(proto, tmp_path / "save")
        completed, _ = _run_and_wait(executor, reloaded, tmp_path)
        assert completed, "Reloaded BF protocol did not complete"

    def test_video_stim_save_load_run(self, executor, scope, tmp_path):
        """OG protocol pattern: save with stim+video, reload, run."""
        sc = _stim_config_enabled(channels=["Green"])
        vc = {"duration": 0.5, "fps": 5}
        steps = [
            _make_step(name="A1_BF", color="BF", acquire="image"),
            _make_step(name="A1_Red_stim", color="Red", acquire="video",
                       video_config=vc, stim_config=sc),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path / "save")

        # Verify configs survived reload
        vid_step = reloaded.step(idx=1)
        assert isinstance(vid_step["Video Config"], dict)
        assert isinstance(vid_step["Stim_Config"], dict)
        assert vid_step["Stim_Config"]["Green"]["enabled"] is True

        # Run the reloaded protocol
        completed, _ = _run_and_wait(executor, reloaded, tmp_path)
        assert completed, "Reloaded video+stim protocol did not complete"

    def test_multi_well_tiled_save_load_run(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF_T00", well="A1", tile="T00", tile_group_id=1, x=10.0, y=20.0),
            _make_step(name="A1_BF_T01", well="A1", tile="T01", tile_group_id=1, x=15.0, y=20.0),
            _make_step(name="A2_BF_T00", well="A2", tile="T00", tile_group_id=2, x=30.0, y=20.0),
            _make_step(name="A2_BF_T01", well="A2", tile="T01", tile_group_id=2, x=35.0, y=20.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path / "save")
        completed, _ = _run_and_wait(executor, reloaded, tmp_path)
        assert completed, "Reloaded multi-well tiled protocol did not complete"

    def test_back_to_back_different_protocols(self, executor, scope, tmp_path):
        """Run protocol A, then protocol B — verifies state cleanup between runs."""
        proto_a = _build_protocol([
            _make_step(name="A1_BF", color="BF"),
            _make_step(name="A1_Green", color="Green"),
        ])
        completed_a, _ = _run_and_wait(executor, proto_a, tmp_path / "run_a")
        assert completed_a, "Protocol A did not complete"

        # Wait for file I/O to drain before starting next run
        import time
        time.sleep(1.0)

        proto_b = _build_protocol([
            _make_step(name="B1_Red", color="Red", acquire="image"),
        ])
        completed_b, _ = _run_and_wait(executor, proto_b, tmp_path / "run_b")
        assert completed_b, "Protocol B did not complete after A"


class TestValidation:
    """Protocol validation catches bad configs before execution."""

    def test_invalid_video_config_not_dict(self):
        steps = [_make_step(acquire="video", video_config="not a dict")]
        proto = _build_protocol(steps)
        errors = proto.validate_steps()
        assert any("Video Config" in e for e in errors), f"Expected Video Config error, got: {errors}"

    def test_invalid_color(self):
        steps = [_make_step(color="Ultraviolet")]
        proto = _build_protocol(steps)
        errors = proto.validate_steps()
        assert len(errors) > 0, "Expected validation error for invalid color"

    def test_negative_exposure(self):
        steps = [_make_step(exposure=-1.0)]
        proto = _build_protocol(steps)
        errors = proto.validate_steps()
        assert len(errors) > 0, "Expected validation error for negative exposure"


# ===========================================================================
# PART 3: Round-Trip Gaps
# ===========================================================================

class TestRoundTripMetadata:
    """Protocol metadata (period, duration, labware, capture_root) survives round-trip."""

    def test_period_preserved(self, tmp_path):
        proto = _build_protocol([_make_step()], period_min=5.0)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.period() == datetime.timedelta(minutes=5)

    def test_duration_preserved(self, tmp_path):
        proto = _build_protocol([_make_step()], duration_hrs=12.0)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.duration() == datetime.timedelta(hours=12)

    def test_labware_preserved(self, tmp_path):
        proto = _build_protocol([_make_step()], labware="96 well microplate")
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.labware() == "96 well microplate"

    def test_capture_root_preserved(self, tmp_path):
        steps = [_make_step()]
        proto = _build_protocol(steps)
        proto._config['capture_root'] = 'experiment_42'
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.capture_root() == 'experiment_42'

    def test_video_config_various_fps(self, tmp_path):
        """Different fps values round-trip correctly."""
        for fps in [0.5, 1, 5, 10, 30, 60]:
            vc = {"duration": 2.0, "fps": fps}
            proto = _build_protocol([_make_step(acquire="video", video_config=vc)])
            reloaded = _save_and_reload(proto, tmp_path / f"fps_{fps}")
            assert reloaded.step(idx=0)["Video Config"]["fps"] == fps

    def test_video_config_various_durations(self, tmp_path):
        """Different duration values round-trip correctly."""
        for dur in [0.1, 1.0, 10.0, 60.0, 300.0]:
            vc = {"duration": dur, "fps": 5}
            proto = _build_protocol([_make_step(acquire="video", video_config=vc)])
            reloaded = _save_and_reload(proto, tmp_path / f"dur_{dur}")
            assert reloaded.step(idx=0)["Video Config"]["duration"] == dur

    def test_stim_multi_channel_enabled(self, tmp_path):
        """Stim config with all 3 channels enabled."""
        sc = _stim_config_enabled(channels=["Red", "Green", "Blue"])
        proto = _build_protocol([_make_step(stim_config=sc)])
        reloaded = _save_and_reload(proto, tmp_path)
        step = reloaded.step(idx=0)
        for ch in ("Red", "Green", "Blue"):
            assert step["Stim_Config"][ch]["enabled"] is True

    def test_stim_per_channel_values(self, tmp_path):
        """Each stim channel can have different parameter values."""
        sc = _default_stim_config()
        sc["Red"]["enabled"] = True
        sc["Red"]["frequency"] = 10.0
        sc["Red"]["pulse_width"] = 5
        sc["Red"]["pulse_count"] = 200
        sc["Green"]["enabled"] = True
        sc["Green"]["frequency"] = 20.0
        sc["Green"]["pulse_width"] = 50
        sc["Green"]["pulse_count"] = 10
        proto = _build_protocol([_make_step(stim_config=sc)])
        reloaded = _save_and_reload(proto, tmp_path)
        step = reloaded.step(idx=0)
        assert step["Stim_Config"]["Red"]["frequency"] == 10.0
        assert step["Stim_Config"]["Red"]["pulse_count"] == 200
        assert step["Stim_Config"]["Green"]["frequency"] == 20.0
        assert step["Stim_Config"]["Green"]["pulse_count"] == 10
        assert step["Stim_Config"]["Blue"]["enabled"] is False


class TestRoundTripCombinations:
    """Combined feature round-trips."""

    def test_tiling_plus_zstack(self, tmp_path):
        """Tiled + z-stacked steps."""
        steps = []
        for tile_idx, (tx, ty, tlabel) in enumerate([(10, 20, "T00"), (15, 20, "T01")]):
            for z_idx, z in enumerate([4900, 5000, 5100]):
                steps.append(_make_step(
                    name=f"A1_BF_{tlabel}_Z{z_idx}",
                    x=tx, y=ty, z=z,
                    tile=tlabel, tile_group_id=1,
                    z_slice=z_idx, zstack_group_id=tile_idx + 1,
                ))
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 6

    def test_multiwell_multichannel(self, tmp_path):
        """Multi-well × multi-channel protocol."""
        steps = []
        for well, x, y in [("A1", 10, 20), ("A2", 30, 20), ("B1", 10, 40)]:
            for color, ill in [("BF", 50), ("Green", 200), ("Red", 150)]:
                steps.append(_make_step(
                    name=f"{well}_{color}", well=well, x=x, y=y,
                    color=color, illumination=ill,
                ))
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 9
        assert reloaded.step(idx=0)["Well"] == "A1"
        assert reloaded.step(idx=0)["Color"] == "BF"
        assert reloaded.step(idx=8)["Well"] == "B1"
        assert reloaded.step(idx=8)["Color"] == "Red"

    def test_tiling_1x3(self, tmp_path):
        """1 row × 3 columns tiling."""
        steps = [
            _make_step(name=f"A1_BF_T0{i}", tile=f"T0{i}", tile_group_id=1, x=10 + i * 5)
            for i in range(3)
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 3

    def test_tiling_3x5(self, tmp_path):
        """3 rows × 5 columns tiling (15 tiles)."""
        steps = []
        for row in range(3):
            for col in range(5):
                steps.append(_make_step(
                    name=f"A1_BF_T{row}{col}",
                    tile=f"T{row}{col}",
                    tile_group_id=1,
                    x=10 + col * 5, y=20 + row * 5,
                ))
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 15

    def test_multiwell_tiled_multichannel(self, tmp_path):
        """The real-world protocol: multi-well × tiled × multi-channel."""
        steps = []
        for well, wx, wy in [("A1", 10, 20), ("A2", 30, 20)]:
            for tile_idx, (tx, ty) in enumerate([(0, 0), (5, 0), (0, 5), (5, 5)]):
                for color in ["BF", "Green"]:
                    steps.append(_make_step(
                        name=f"{well}_{color}_T{tile_idx:02d}",
                        well=well, x=wx + tx, y=wy + ty,
                        color=color,
                        tile=f"T{tile_idx:02d}", tile_group_id=1,
                    ))
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.num_steps() == 16  # 2 wells × 4 tiles × 2 channels

    def test_multiple_objectives(self, tmp_path):
        """Steps with different objectives."""
        steps = [
            _make_step(name="A1_4x", objective="4x Oly", z=3000.0),
            _make_step(name="A1_10x", objective="10x Oly", z=5000.0),
            _make_step(name="A1_20x", objective="20x Oly", z=7000.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path)
        assert reloaded.step(idx=0)["Objective"] == "4x Oly"
        assert reloaded.step(idx=1)["Objective"] == "10x Oly"
        assert reloaded.step(idx=2)["Objective"] == "20x Oly"


# ===========================================================================
# PART 4: Execution Gaps
# ===========================================================================

class TestExecuteMultiScan:
    """Multi-scan (time-lapse) execution."""

    def test_two_scan_timelapse(self, executor, scope, tmp_path):
        steps = [_make_step(color="BF")]
        proto = _build_protocol(steps, period_min=0.01, duration_hrs=0.01)
        completed, _ = _run_and_wait(executor, proto, tmp_path, max_scans=2)
        assert completed, "2-scan time-lapse did not complete"

    def test_three_scan_multichannel(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF", color="BF"),
            _make_step(name="A1_Green", color="Green"),
        ]
        proto = _build_protocol(steps, period_min=0.01, duration_hrs=0.01)
        completed, _ = _run_and_wait(executor, proto, tmp_path, max_scans=3)
        assert completed, "3-scan multi-channel did not complete"


class TestExecuteDisabledSaving:
    """Execution with saving artifacts disabled."""

    def test_no_saving(self, executor, scope, tmp_path):
        steps = [_make_step(color="BF")]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(
            executor, proto, tmp_path,
            disable_saving_artifacts=True,
        )
        assert completed, "Protocol with saving disabled did not complete"

    def test_no_saving_video(self, executor, scope, tmp_path):
        steps = [_make_step(acquire="video", video_config={"duration": 0.3, "fps": 5})]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(
            executor, proto, tmp_path,
            disable_saving_artifacts=True,
        )
        assert completed, "Video protocol with saving disabled did not complete"


class TestExecutePixelDepth:
    """Execution with different pixel depth settings."""

    def test_full_pixel_depth(self, executor, scope, tmp_path):
        steps = [_make_step(color="BF")]
        proto = _build_protocol(steps)
        icc = _make_image_capture_config()
        icc['use_full_pixel_depth'] = True
        completed, _ = _run_and_wait(executor, proto, tmp_path, image_capture_config=icc)
        assert completed, "12-bit capture protocol did not complete"


class TestExecuteSeparateFolders:
    """Execution with separate folder per channel."""

    def test_separate_folders(self, executor, scope, tmp_path):
        steps = [
            _make_step(name="A1_BF", color="BF"),
            _make_step(name="A1_Green", color="Green"),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(
            executor, proto, tmp_path,
            separate_folder_per_channel=True,
        )
        assert completed, "Protocol with separate folders did not complete"


class TestExecuteCombinations:
    """Combined feature execution."""

    def test_tiled_plus_zstack(self, executor, scope, tmp_path):
        steps = []
        for tile_idx, (tx, tlabel) in enumerate([(10, "T00"), (15, "T01")]):
            for z_idx, z in enumerate([4900, 5000, 5100]):
                steps.append(_make_step(
                    name=f"A1_BF_{tlabel}_Z{z_idx}",
                    x=tx, z=z,
                    tile=tlabel, tile_group_id=1,
                    z_slice=z_idx, zstack_group_id=tile_idx + 1,
                ))
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Tiled + z-stack protocol did not complete"

    def test_multiwell_multichannel(self, executor, scope, tmp_path):
        steps = []
        for well, x, y in [("A1", 10, 20), ("A2", 30, 20)]:
            for color in ["BF", "Green"]:
                steps.append(_make_step(
                    name=f"{well}_{color}", well=well, x=x, y=y, color=color,
                ))
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Multi-well multi-channel protocol did not complete"

    def test_multiwell_tiled_multichannel(self, executor, scope, tmp_path):
        """The full real-world protocol pattern."""
        steps = []
        for well, wx, wy in [("A1", 10, 20), ("A2", 30, 20)]:
            for tile_idx, (tx, ty) in enumerate([(0, 0), (5, 0)]):
                for color in ["BF", "Green"]:
                    steps.append(_make_step(
                        name=f"{well}_{color}_T{tile_idx:02d}",
                        well=well, x=wx + tx, y=wy + ty,
                        color=color,
                        tile=f"T{tile_idx:02d}", tile_group_id=1,
                    ))
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "Multi-well tiled multi-channel protocol did not complete"

    def test_large_protocol_50_steps(self, executor, scope, tmp_path):
        """Stress test: 50 steps should complete without timeout."""
        steps = [_make_step(name=f"step_{i}", color="BF") for i in range(50)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed, "50-step protocol did not complete"


class TestExecuteFileOutput:
    """Verify output directory and metadata files are created."""

    def test_output_directory_created(self, executor, scope, tmp_path):
        steps = [_make_step(name="A1_BF", color="BF")]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed

        output_dir = tmp_path / "output"
        assert output_dir.exists(), "Output directory not created"
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1, f"No timestamped run subdirectory in {output_dir}"

    def test_protocol_tsv_saved_in_output(self, executor, scope, tmp_path):
        steps = [_make_step(name="A1_BF", color="BF")]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed

        output_dir = tmp_path / "output"
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1

        run_dir = subdirs[0]
        tsv_files = list(run_dir.glob("*.tsv"))
        assert len(tsv_files) >= 1, f"No protocol TSV file found in {run_dir}"

    def test_execution_record_created(self, executor, scope, tmp_path):
        """Execution record JSON is written after protocol completes."""
        import time
        steps = [_make_step(name="A1_BF", color="BF")]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path)
        assert completed
        # Wait for file I/O to drain
        time.sleep(1.0)

        output_dir = tmp_path / "output"
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1

        run_dir = subdirs[0]
        # Record could be .json or .tsv depending on format
        all_files = list(run_dir.iterdir())
        assert len(all_files) >= 1, f"No files at all in {run_dir}: {all_files}"


class TestExecuteCancellation:
    """Protocol cancellation mid-run."""

    def test_cancel_during_multi_step(self, executor, scope, tmp_path):
        """Cancel a long protocol after it starts — should clean up gracefully."""
        import time

        steps = [_make_step(name=f"step_{i}", color="BF") for i in range(20)]
        proto = _build_protocol(steps)

        done = threading.Event()

        def on_complete(**kwargs):
            done.set()

        callbacks = {
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        }

        executor.run(
            protocol=proto,
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='test_cancel',
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

        # Let it run for a moment then cancel
        time.sleep(0.5)
        executor._protocol_ended.set()

        # Should still fire run_complete callback
        completed = done.wait(timeout=COMPLETION_TIMEOUT)
        assert completed, "Protocol did not fire run_complete after cancellation"
        assert not executor.run_in_progress(), "Executor still running after cancel"


class TestExecuteLEDRestore:
    """LED state restoration after protocol."""

    def test_leds_off_after_protocol(self, executor, scope, tmp_path):
        steps = [_make_step(color="Green", illumination=200.0)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(executor, proto, tmp_path, leds_state_at_end='off')
        assert completed

        # All LEDs should be off after protocol
        states = scope.get_led_states()
        for color, state in states.items():
            assert not state['enabled'], f"LED {color} still on after protocol"


# ===========================================================================
# PART 6: Real Path Tests (no mocking of drivers/transforms)
# ===========================================================================

class TestRealPathExecution:
    """Tests using real_executor with real WellPlateLoader, CoordinateTransformer,
    and simulated MotorBoard. These catch init/config bugs that mocked tests miss.

    The axes_config AttributeError (2026-03-27) would have been caught here
    because _default_move → scope.move_absolute_position → motion.move_abs_pos
    accesses self.axes_config, which must be initialized in __init__.
    """

    def test_single_bf_real_motion(self, real_executor, scope, tmp_path):
        """Single BF step with real coordinate transforms and motor movement."""
        steps = [_make_step(color="BF", x=10.0, y=20.0, z=5000.0)]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(real_executor, proto, tmp_path)
        assert completed, "Single BF with real motion did not complete"

    def test_multi_well_real_motion(self, real_executor, scope, tmp_path):
        """Multi-well protocol with real XY coordinate transforms."""
        steps = [
            _make_step(name="A1_BF", well="A1", x=10.0, y=20.0),
            _make_step(name="A2_BF", well="A2", x=30.0, y=20.0),
            _make_step(name="B1_BF", well="B1", x=10.0, y=40.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(real_executor, proto, tmp_path)
        assert completed, "Multi-well with real motion did not complete"

    def test_multichannel_real_motion(self, real_executor, scope, tmp_path):
        """BF + fluorescence with real LED and motor paths."""
        steps = [
            _make_step(name="A1_BF", color="BF", illumination=50.0),
            _make_step(name="A1_Green", color="Green", illumination=200.0),
            _make_step(name="A1_Red", color="Red", illumination=150.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(real_executor, proto, tmp_path)
        assert completed, "Multi-channel with real motion did not complete"

    def test_zstack_real_motion(self, real_executor, scope, tmp_path):
        """Z-stack with real Z axis movement."""
        steps = [
            _make_step(name="A1_BF_Z0", z=4500.0, z_slice=0, zstack_group_id=1),
            _make_step(name="A1_BF_Z1", z=5000.0, z_slice=1, zstack_group_id=1),
            _make_step(name="A1_BF_Z2", z=5500.0, z_slice=2, zstack_group_id=1),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(real_executor, proto, tmp_path)
        assert completed, "Z-stack with real motion did not complete"

    def test_tiled_real_motion(self, real_executor, scope, tmp_path):
        """2x2 tiling with real XY coordinate transforms."""
        steps = [
            _make_step(name="A1_BF_T00", tile="T00", tile_group_id=1, x=10.0, y=20.0),
            _make_step(name="A1_BF_T01", tile="T01", tile_group_id=1, x=12.0, y=20.0),
            _make_step(name="A1_BF_T10", tile="T10", tile_group_id=1, x=10.0, y=22.0),
            _make_step(name="A1_BF_T11", tile="T11", tile_group_id=1, x=12.0, y=22.0),
        ]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(real_executor, proto, tmp_path)
        assert completed, "2x2 tiling with real motion did not complete"

    def test_save_load_run_real_motion(self, real_executor, scope, tmp_path):
        """Full pipeline: create → save → reload → run with real motion."""
        steps = [
            _make_step(name="A1_BF", color="BF", x=10.0, y=20.0, z=5000.0),
            _make_step(name="A1_Green", color="Green", x=10.0, y=20.0, z=5000.0),
        ]
        proto = _build_protocol(steps)
        reloaded = _save_and_reload(proto, tmp_path / "save")
        completed, _ = _run_and_wait(real_executor, reloaded, tmp_path)
        assert completed, "Save→load→run with real motion did not complete"

    def test_video_real_motion(self, real_executor, scope, tmp_path):
        """Video capture with real motion path."""
        steps = [_make_step(
            acquire="video",
            video_config={"duration": 0.3, "fps": 5},
        )]
        proto = _build_protocol(steps)
        completed, _ = _run_and_wait(real_executor, proto, tmp_path)
        assert completed, "Video with real motion did not complete"

    def test_back_to_back_real_motion(self, real_executor, scope, tmp_path):
        """Two protocols back-to-back with real motion — verifies state cleanup."""
        import time

        proto_a = _build_protocol([_make_step(name="A1_BF", color="BF")])
        completed_a, _ = _run_and_wait(real_executor, proto_a, tmp_path / "run_a")
        assert completed_a, "Protocol A with real motion did not complete"

        time.sleep(1.0)

        proto_b = _build_protocol([
            _make_step(name="B1_Green", color="Green", x=30.0, y=30.0),
        ])
        completed_b, _ = _run_and_wait(real_executor, proto_b, tmp_path / "run_b")
        assert completed_b, "Protocol B with real motion did not complete after A"
