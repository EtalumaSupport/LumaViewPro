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
