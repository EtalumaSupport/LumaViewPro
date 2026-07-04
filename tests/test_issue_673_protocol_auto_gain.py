"""Regression test for #673 -- Auto_Gain protocol steps must fire
``set_auto_gain(state=True, ...)``.

Before the fix at LVP `<commit>` (commit message references #673), the
`else` branch of `protocol_image_writer.capture()`'s
`if not step['Auto_Gain']:` block did nothing except log a warning:

    else:
        logger.warning(f"[CAPTURE DIAG] SKIPPING camera settings -- Auto_Gain is truthy: {_ag!r}")

Combined with the gate at `ui/layer_control.py::apply_settings`
(`if not protocol_running_global.is_set():`) which short-circuits the
only other call site for `apply_layer_camera_settings`, no code path
ever fired `set_auto_gain(state=True, ...)` during a protocol scan.
The result: every Auto_Gain protocol step inherited the previous
step's / live-mode's gain + exposure (frequently overexposed).

Fix: the `else` branch now routes through
`apply_layer_camera_settings(auto_gain=True, auto_gain_settings=...)`,
which is the same path live-mode AE/AG uses. The runner's existing
`_auto_gain_deadline` loop in `scan_iterate` handles convergence wait.

This test runs a single-step protocol with `Auto_Gain=True` on the
simulator and asserts that the api-log captures the
`apply_layer_camera_settings ... auto_gain=True` event within the
protocol's `save_camera_state -> restore_camera_state` window.
"""

from __future__ import annotations

import datetime
import logging
import sys
import threading
from unittest.mock import MagicMock

import pytest

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
from modules.protocol import Protocol
from modules.sequenced_capture_runner import (
    SequencedCaptureRunner,
    SequencedCaptureRunMode,
)
from modules.sequential_io_executor import SequentialIOExecutor


def _build_single_step_ag_protocol(color='BF', auto_gain=True):
    """Build a one-step protocol with Auto_Gain configurable.

    Plate coords in mm; stage_offset zero; 6-well labware so the
    coordinate transformer accepts the position.
    """
    import pandas as pd
    import pathlib

    tiling_path = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'

    step = {
        'Name': f'A1_{color}_AG',
        'X': 24.55,
        'Y': 24.0,
        'Z': 6247.4,
        'Auto_Focus': False,
        'Color': color,
        'False_Color': color != 'BF',
        'Illumination': 50.0,
        'Gain': 5.0,
        'Auto_Gain': auto_gain,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': '4x Oly',
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': -1,
        'Custom Step': False,
        'Tile Group ID': -1,
        'Z-Stack Group ID': -1,
        'Acquire': 'image',
        'Video Config': {'duration': 5, 'fps': 30},
        'Stim_Config': {},
        'Step Index': 0,
    }

    df = pd.DataFrame([step])
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': df,
        'period': datetime.timedelta(minutes=20.0),
        'duration': datetime.timedelta(hours=1.0),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=tiling_path, config=config)


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_protocol_execution.py shape)
# ---------------------------------------------------------------------------


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s.imaging.start_streaming()
    yield s
    s.imaging.stop_streaming()
    s.disconnect()


@pytest.fixture
def executors():
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
    yield execs
    for name, e in execs.items():
        try:
            if name == 'protocol':
                e.stop(timeout=2.0)
            else:
                e.shutdown()
        except Exception:
            pass


@pytest.fixture
def executor(scope, executors):
    from modules.coord_transformations import CoordinateTransformer
    from modules.labware_loader import WellPlateLoader

    mock_af = MagicMock()
    mock_af.reset = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.complete = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.result = MagicMock(return_value=None)
    mock_af.best_focus_position = MagicMock(return_value=6247.4)
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


class _ApiLogCapture(logging.Handler):
    """Captures records from the 'LVP.api' logger."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[tuple[float, str]] = []
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self.records.append((record.created, record.getMessage()))


def _run_protocol(executor, protocol, tmp_path):
    done = threading.Event()
    result_holder: dict = {}

    def on_complete(**kwargs):
        result_holder.update(kwargs)
        done.set()

    callbacks = {
        'run_complete': on_complete,
    }

    plan = executor.prepare(
        protocol=protocol,
        run_trigger_source='test',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='issue_673_repro',
        image_capture_config={
            'output_format': {'live': 'TIFF', 'sequenced': 'TIFF'},
            'capture_depth': 8,
        },
        autogain_settings={
            'target_brightness': 0.3,
            'min_gain_db': 0.0,
            'max_gain_db': 20.0,
            'max_duration': datetime.timedelta(seconds=1),
        },
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

    completed = done.wait(timeout=30)
    return completed, result_holder


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


class TestProtocolAutoGainFires:
    """When a protocol step has Auto_Gain=True, the capture path must
    fire `set_auto_gain(state=True, ...)` (via apply_layer_camera_settings).
    The pre-fix codebase silently SKIPPED the camera settings on the
    Auto_Gain branch, leaving the camera at inherited gain/exposure.
    """

    def test_auto_gain_step_fires_set_auto_gain(self, executor, tmp_path):
        protocol = _build_single_step_ag_protocol(color='BF', auto_gain=True)

        capture = _ApiLogCapture()
        api_logger = logging.getLogger('LVP.api')
        prev_level = api_logger.level
        api_logger.setLevel(logging.INFO)
        api_logger.addHandler(capture)
        try:
            completed, _ = _run_protocol(executor, protocol, tmp_path)
        finally:
            api_logger.removeHandler(capture)
            api_logger.setLevel(prev_level)

        assert completed, 'Protocol did not complete within timeout'

        # Find save_camera_state(protocol) and restore_camera_state(protocol);
        # the protocol's execution window lives between them. Within that
        # window we expect at least one `apply_layer_camera_settings ...
        # auto_gain=True` event for the single Auto_Gain step.
        save_idx = None
        restore_idx = None
        for i, (_, msg) in enumerate(capture.records):
            if save_idx is None and 'save_camera_state' in msg and 'tag=protocol' in msg:
                save_idx = i
            elif save_idx is not None and 'restore_camera_state' in msg and 'tag=protocol' in msg:
                restore_idx = i
                break

        assert save_idx is not None, (
            'Did not see save_camera_state(protocol) -- protocol did '
            'not enter its run window. Records sample:\n'
            + '\n'.join(f'  {m}' for _, m in capture.records[:40])
        )
        assert restore_idx is not None, (
            'Did not see restore_camera_state(protocol) -- protocol did not complete cleanly.'
        )

        auto_gain_true_events = [
            i
            for i in range(save_idx, restore_idx)
            if 'apply_layer_camera_settings' in capture.records[i][1]
            and 'auto_gain=True' in capture.records[i][1]
        ]

        assert len(auto_gain_true_events) >= 1, (
            f'#673 reproduced: protocol step with Auto_Gain=True did NOT '
            f'fire `apply_layer_camera_settings ... auto_gain=True` '
            f'within the protocol window (save_idx={save_idx}, '
            f'restore_idx={restore_idx}). This means set_auto_gain '
            f'never fired and the camera held inherited gain/exposure.\n'
            f'In-window events:\n'
            + '\n'.join(f'  [{i}] {capture.records[i][1]}' for i in range(save_idx, restore_idx))
        )

    def test_led_lit_before_auto_gain_armed(self, executor, tmp_path):
        """The channel LED must be lit BEFORE AG is armed, so hardware AG
        settles against the lit scene rather than a dark frame (the #673 root
        cause: arming AG dark rails gain/exposure on noise). Within the
        protocol window the led_on event must precede the
        apply_layer_camera_settings ... auto_gain=True event.
        """
        protocol = _build_single_step_ag_protocol(color='BF', auto_gain=True)

        capture = _ApiLogCapture()
        api_logger = logging.getLogger('LVP.api')
        prev_level = api_logger.level
        api_logger.setLevel(logging.INFO)
        api_logger.addHandler(capture)
        try:
            completed, _ = _run_protocol(executor, protocol, tmp_path)
        finally:
            api_logger.removeHandler(capture)
            api_logger.setLevel(prev_level)

        assert completed, 'Protocol did not complete within timeout'

        save_idx = None
        restore_idx = None
        for i, (_, msg) in enumerate(capture.records):
            if save_idx is None and 'save_camera_state' in msg and 'tag=protocol' in msg:
                save_idx = i
            elif save_idx is not None and 'restore_camera_state' in msg and 'tag=protocol' in msg:
                restore_idx = i
                break
        assert save_idx is not None and restore_idx is not None

        led_on_idx = next(
            (i for i in range(save_idx, restore_idx) if 'led_on ch=' in capture.records[i][1]),
            None,
        )
        ag_arm_idx = next(
            (
                i
                for i in range(save_idx, restore_idx)
                if 'apply_layer_camera_settings' in capture.records[i][1]
                and 'auto_gain=True' in capture.records[i][1]
            ),
            None,
        )
        assert led_on_idx is not None, 'No led_on event in the protocol window.'
        assert ag_arm_idx is not None, 'No auto_gain=True arm event in the protocol window.'
        assert led_on_idx < ag_arm_idx, (
            f'LED must be lit before AG is armed: led_on at index {led_on_idx}, '
            f'AG arm at {ag_arm_idx}. Arming AG against a dark frame rails on '
            f'noise and the grab is mis-exposed.'
        )

    def test_no_auto_gain_step_does_not_enable_auto_gain(self, executor, tmp_path):
        """Symmetry check: Auto_Gain=False steps must NOT enable AG. The fix
        must not regress the static-gain path.
        """
        protocol = _build_single_step_ag_protocol(color='BF', auto_gain=False)

        capture = _ApiLogCapture()
        api_logger = logging.getLogger('LVP.api')
        prev_level = api_logger.level
        api_logger.setLevel(logging.INFO)
        api_logger.addHandler(capture)
        try:
            completed, _ = _run_protocol(executor, protocol, tmp_path)
        finally:
            api_logger.removeHandler(capture)
            api_logger.setLevel(prev_level)

        assert completed

        save_idx = None
        restore_idx = None
        for i, (_, msg) in enumerate(capture.records):
            if save_idx is None and 'save_camera_state' in msg and 'tag=protocol' in msg:
                save_idx = i
            elif save_idx is not None and 'restore_camera_state' in msg and 'tag=protocol' in msg:
                restore_idx = i
                break

        assert save_idx is not None and restore_idx is not None

        # Within the protocol window, no apply_layer_camera_settings with
        # auto_gain=True should fire.
        bad_events = [
            capture.records[i][1]
            for i in range(save_idx, restore_idx)
            if 'apply_layer_camera_settings' in capture.records[i][1]
            and 'auto_gain=True' in capture.records[i][1]
        ]
        assert not bad_events, (
            f'Static-gain (Auto_Gain=False) step erroneously enabled AG: {bad_events}'
        )
