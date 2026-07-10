"""Regression test for #671 -- LED-state-hygiene at the transition
from a TSV-loaded last step to a `Protocol.insert_step()`-added step.

User repro (the beta tester, beta12, 2026-05-19):
  1. Load an 8-step TSV (A1+A2 well, 4 channels each: BF, Blue, Green,
     Red, in that order).
  2. Use the UI to add a 4-channel location at a 3rd X/Y. The UI
     loops calling `protocol.insert_step()` for each channel.
  3. Run the protocol.

Observed in `Firmware/build/issue671_2026-05-19/api.log`:

    07:33:17.895  led_on ch=2  mA=350.0  owner='protocol'   <-- A2 Red
    07:33:19.090  move_abs X=19856.9um                       <-- move WITHOUT leds_off!
    07:33:19.150  move_abs Y=57899.3um
    07:33:19.231  move_abs Z=6247.4um wait
    07:33:20.920  leds_off                                   <-- delayed, AFTER move
    07:33:20.952  move_abs X=19856.9um                       <-- DUPLICATE move
    07:33:21.001  move_abs Y=57899.3um
    07:33:21.073  move_abs Z=6247.4um wait
    07:33:21.109  set_exposure 1.0ms
    07:33:21.171  led_on ch=0  mA=250.0  owner='protocol'    <-- step 9 (added BF)

The Red LED stays lit during the well-to-well move from A2 to the
3rd location. Two structural anomalies:
  (a) `leds_off` should fire BEFORE the first move_abs at the
      transition (canonical convention; intra-TSV transitions follow
      this correctly via `protocol_image_writer.capture()`).
  (b) Two `move_abs` sets target identical X/Y/Z, separated by
      `leds_off`.

The test builds a protocol mirroring the repro (8 TSV-equivalent
steps + 4 `insert_step`-added steps at a 3rd X/Y), runs it on the
simulator with `default_move` firing real motion calls, and asserts
on the api-log event sequence at the step-8 -> step-9 boundary.

This test is expected to FAIL until #671 is fixed.
"""

from __future__ import annotations

import datetime
import logging
import sys
import threading
from unittest.mock import MagicMock

import pytest

# Mirror conftest pattern (heavy deps already mocked there).
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

from modules.image_mode import ImageCaptureConfig
from modules.lumascope_api import Lumascope
from modules.protocol import Protocol
from modules.sequenced_capture_runner import (
    SequencedCaptureRunner,
    SequencedCaptureRunMode,
)
from modules.sequential_io_executor import SequentialIOExecutor


# A1 / A2 / added-location PLATE coordinates in mm (from RedStaysOn.tsv).
# Stage coords appear in api.log as `move_abs X=NNNNN.0um` after the
# CoordinateTransformer applies the labware + stage_offset transform.
A1_X, A1_Y, A1_Z = 24.55, 24.0, 6247.3684
A2_X, A2_Y, A2_Z = 63.75, 24.0, 6247.3684
ADDED_X, ADDED_Y, ADDED_Z = 100.0, 60.0, 6247.3684

CHANNEL_ORDER = ['BF', 'Blue', 'Green', 'Red']
CHANNEL_EXPOSURE = {'BF': 0.1, 'Blue': 1.0, 'Green': 6.76, 'Red': 600.0}
CHANNEL_ILLUMINATION = {'BF': 5.0, 'Blue': 250.0, 'Green': 250.0, 'Red': 350.0}
CHANNEL_GAIN = {'BF': 14.4, 'Blue': 20.0, 'Green': 20.0, 'Red': 20.0}


def _step_dict(name, x, y, z, color, idx):
    return {
        'Name': name,
        'X': x,
        'Y': y,
        'Z': z,
        'Auto_Focus': False,
        'Color': color,
        'False_Color': color != 'BF',
        'Illumination': CHANNEL_ILLUMINATION[color],
        'Gain': CHANNEL_GAIN[color],
        'Auto_Gain': False,
        'Exposure': CHANNEL_EXPOSURE[color],
        'Sum': 1,
        'Objective': '4x Oly',
        'Well': name.split('_')[0] if '_' in name else 'A1',
        'Tile': '',
        'Z-Slice': -1,
        'Custom Step': False,
        'Tile Group ID': -1,
        'Z-Stack Group ID': -1,
        'Acquire': 'image',
        'Video Config': {'duration': 5, 'fps': 30},
        'Stim_Config': {},
        'Step Index': idx,
        'Label': '',
    }


def _build_tsv_only_protocol():
    """Build the 8-step TSV (no user-added location yet)."""
    import pandas as pd

    tiling_configs = pytest.importorskip('modules.tiling_config')  # noqa: F841
    import pathlib

    tiling_path = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'

    rows = []
    idx = 0
    for well_x, well_y, well_name in [
        (A1_X, A1_Y, 'A1'),
        (A2_X, A2_Y, 'A2'),
    ]:
        for color in CHANNEL_ORDER:
            rows.append(_step_dict(f'{well_name}_{color}', well_x, well_y, A1_Z, color, idx))
            idx += 1
    df = pd.DataFrame(rows)
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': df,
        'period': datetime.timedelta(minutes=20.0),
        'duration': datetime.timedelta(hours=48.0),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=tiling_path, config=config)


def _add_3rd_location_via_insert_step(protocol):
    """Mimic the UI add-location flow: 4 insert_step calls (one per
    enabled channel) at the 3rd X/Y. Each call appends after the
    current last step.

    Mirrors `ui/protocol_settings.py::insert_step_ex` -- which loops
    over selected channels and calls `protocol.insert_step()` per
    channel.
    """
    for color in CHANNEL_ORDER:
        protocol.insert_step(
            step_name=f'added_{color}',
            layer=color,
            layer_config={
                'autofocus': False,
                'false_color': color != 'BF',
                'illumination_ma': CHANNEL_ILLUMINATION[color],
                'gain_db': CHANNEL_GAIN[color],
                'auto_gain': False,
                'exposure_ms': CHANNEL_EXPOSURE[color],
                'sum': 1,
                'acquire': 'image',
                'video_config': {'duration': 5, 'fps': 30},
            },
            plate_position={'x': ADDED_X, 'y': ADDED_Y, 'z': ADDED_Z},
            objective_id='4x Oly',
            stim_configs={},
            before_step=None,
            after_step=protocol.num_steps() - 1,
        )


# ---------------------------------------------------------------------------
# Test fixtures
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
    mock_af.best_focus_position = MagicMock(return_value=A1_Z)
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
    """Captures records from the 'LVP.api' logger so tests can assert
    on event ordering between move_abs / leds_off / led_on.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[tuple[float, str]] = []
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self.records.append((record.created, record.getMessage()))


def _run_protocol(executor, protocol, tmp_path):
    """Run protocol with default_move firing (no no-op go_to_step)."""
    done = threading.Event()
    result_holder: dict = {}

    def on_complete(**kwargs):
        result_holder.update(kwargs)
        done.set()

    callbacks = {
        'run_complete': on_complete,
        # Deliberately DO NOT set 'go_to_step' so the runner falls
        # through to default_move(), which fires real move_abs calls.
    }

    plan = executor.prepare(
        protocol=protocol,
        run_trigger_source='test',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='issue_671_repro',
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
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
# The repro test
# ---------------------------------------------------------------------------


class TestAddedLocationLedOrdering:
    """At the boundary between the last TSV step (A2 Red) and the
    first user-added step (added BF), the canonical convention
    requires the LED-off BEFORE the first `move_abs` of the next
    step, so the previous step's LED is not lit during well-to-well
    motion. The bug is that the convention is violated specifically
    at this transition.

    The invariant is the ordering (off precedes move), not the form
    of the off: the boundary off may be the nuclear `leds_off` or the
    LED authority's per-channel `led_off ch=N` diff. Either satisfies
    the contract, so routing the boundary through the authority must
    not falsely reproduce the bug.
    """

    def test_leds_off_precedes_move_at_added_location_boundary(self, executor, tmp_path):
        protocol = _build_tsv_only_protocol()
        _add_3rd_location_via_insert_step(protocol)
        assert protocol.num_steps() == 12, (
            f'setup error: expected 12 steps (8 TSV + 4 added), got {protocol.num_steps()}'
        )

        # Sanity-check the added steps are at the 3rd X/Y.
        for i in range(8, 12):
            step = protocol.step(idx=i)
            assert step['X'] == pytest.approx(ADDED_X)
            assert step['Y'] == pytest.approx(ADDED_Y)

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

        # Find the api-log index for the A2_Red led_on (step 8).
        # The protocol has THREE Red led_ons -- A1_Red (step 4),
        # A2_Red (step 8), ADDED_Red (step 12). #671 is about the
        # transition AFTER step 8 (the last TSV step), not after
        # step 12 (the last added step), so we want index [1] in
        # the list of Red led_ons (0=A1, 1=A2, 2=ADDED).
        all_red_led_ons = [
            i
            for i, (_, msg) in enumerate(capture.records)
            if 'led_on ch=2' in msg and 'mA=350' in msg and "owner='protocol'" in msg
        ]
        assert len(all_red_led_ons) >= 3, (
            f'Expected >=3 Red led_on calls (A1 + A2 + ADDED); saw '
            f'{len(all_red_led_ons)}. Records sample:\n'
            + '\n'.join(f'  {m}' for _, m in capture.records[:50])
        )
        last_a2_red_led_on = all_red_led_ons[1]  # A2_Red at step 8

        first_move_after_a2_red = None
        first_boundary_off_after_a2_red = None
        for i in range(last_a2_red_led_on + 1, len(capture.records)):
            msg = capture.records[i][1]
            if first_move_after_a2_red is None and msg.startswith('move_abs '):
                first_move_after_a2_red = i
            # The boundary off is either the nuclear leds_off or the LED
            # authority's per-channel led_off diff; match either form so the
            # invariant under test is the ordering, not the emission shape.
            if first_boundary_off_after_a2_red is None and (
                msg.strip() == 'leds_off' or msg.startswith('led_off ch=')
            ):
                first_boundary_off_after_a2_red = i
            if first_move_after_a2_red is not None and first_boundary_off_after_a2_red is not None:
                break

        assert first_move_after_a2_red is not None, (
            'Did not see any move_abs after A2 Red led_on. '
            'Did the protocol abort before transitioning to step 9?'
        )
        assert first_boundary_off_after_a2_red is not None, (
            'Did not see the boundary LED-off after A2 Red led_on.'
        )

        # THE ASSERTION: the boundary LED-off must precede the first move_abs
        # at the TSV->added-location boundary. With the bug unfixed, the move
        # fires first (Red LED stays on during the move).
        assert first_boundary_off_after_a2_red < first_move_after_a2_red, (
            f'#671 reproduced: the boundary LED-off (idx '
            f'{first_boundary_off_after_a2_red}) fired AFTER first move_abs '
            f'following A2 Red (idx {first_move_after_a2_red}). At the '
            f'TSV->added-location boundary, the LED-off must precede the '
            f"move so the previous step's LED isn't lit during "
            f'well-to-well motion.\n'
            f'Surrounding events:\n'
            + '\n'.join(
                f'  [{i}] {capture.records[i][1]}'
                for i in range(
                    max(0, last_a2_red_led_on),
                    min(
                        len(capture.records),
                        max(first_move_after_a2_red, first_boundary_off_after_a2_red) + 5,
                    ),
                )
            )
        )
