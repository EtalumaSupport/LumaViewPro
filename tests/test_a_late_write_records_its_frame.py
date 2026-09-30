# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved still records the scope as it was when its frame was taken.

A run's saves go to the file writer and land later -- behind a backlog, a
slow disk, a Stop, an error abort. The file's record used to be built when
the write ran, from the scope as it was THEN: the gain of whatever step came
next, no LED current once the run had turned the light off, the well the
stage had moved on to, and the write's own clock. The run still reported
``completed``, and nothing in the file could tell.

The record is now taken with the frame and carried to the write, so these
tests hold every write back until the run has moved on, and read the files.

The well is named from the step's position on the plate the protocol is
written for -- the plate the run moved against -- not the plate the scope
happens to have selected: a 6-well protocol run on a scope set to a 96-well
plate names 6-well wells. And it is not the step's Well field, which is empty
on an inserted step and keeps its old value when a step is moved.
"""

import datetime
import pathlib
import time
from types import SimpleNamespace

import pandas as pd
import pytest

import modules.protocol_image_writer as protocol_image_writer
from modules.image_utils import read_postproc_input_metadata
from modules.protocol import Protocol
from tests.frame_records import plate
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_manual_capture_member import _capture, _open_session, _settings


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

WRITE_HELD_S = 1.0
FILES_TIMEOUT_S = 30.0


def _step(name, index, *, x, gain, sum_count=1, autofocus=False):
    # A protocol file is named from the step's Well field, so each step's is
    # distinct -- and deliberately NOT the well its position lies in: the field
    # goes stale when a step is moved, so the file's well must not come from it.
    return {
        'Name': name,
        'X': x,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': autofocus,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': gain,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': sum_count,
        'Objective': '10x Oly',
        'Well': name,
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': -1,
        'Z-Stack Group ID': -1,
        'Acquire': 'image',
        'Video Config': {'duration': 1.0, 'fps': 5},
        'Stim_Config': {},
        'Step Index': index,
        'Auto_Named': False,
        'Label': '',
    }


def _protocol(steps):
    return Protocol(
        tiling_configs_file_loc=REPO_ROOT / 'data' / 'tiling.json',
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(steps),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(hours=1.0),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _run_with_every_write_held(tmp_path, monkeypatch, steps, on_capture=None):
    """Run ``steps`` headless with each save held on the file writer.

    Returns ``{step name: (metadata read back, when its write started)}``.
    """
    run_parent = tmp_path / 'runs'
    write_started = {}
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        real_save = protocol_image_writer.save_image

        def _held(scope, **kwargs):
            time.sleep(WRITE_HELD_S)
            write_started[kwargs['append']] = datetime.datetime.now()
            return real_save(scope, **kwargs)

        monkeypatch.setattr(protocol_image_writer, 'save_image', _held)
        if on_capture is not None:
            real_capture = protocol_image_writer.ProtocolImageWriter.capture

            def _observed(writer, *args, **kwargs):
                on_capture(writer, kwargs['step'])
                return real_capture(writer, *args, **kwargs)

            monkeypatch.setattr(protocol_image_writer.ProtocolImageWriter, 'capture', _observed)

        outcome = runner.run_single_scan(
            protocol=_protocol(steps),
            parent_dir=str(run_parent),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )
        result = outcome.wait(timeout_s=120.0)
        assert result is not None and result.status == 'completed', result

        deadline = time.monotonic() + FILES_TIMEOUT_S
        while len(list(run_parent.rglob('*.tiff'))) < len(steps) and time.monotonic() < deadline:
            time.sleep(0.05)

    files = {}
    for step in steps:
        [path] = [p for p in run_parent.rglob('*.tiff') if step['Name'] in p.name]
        [started] = [t for append, t in write_started.items() if step['Name'] in append]
        files[step['Name']] = (read_postproc_input_metadata(path) or {}, started)
    return files


class TestALateWriteRecordsItsFrame:
    def test_each_file_records_its_own_steps_capture(self, tmp_path, monkeypatch):
        """Two steps, their writes held until the run has moved past both:
        each file records its own gain, LED current, well and grab time."""
        files = _run_with_every_write_held(
            tmp_path,
            monkeypatch,
            [
                _step('B3', 0, x=20.0, gain=1.0),
                _step('B1', 1, x=60.0, gain=20.0),
            ],
        )

        for name, gain, well in (('B3', 1.0, 'A1'), ('B1', 20.0, 'A2')):
            metadata, write_started = files[name]
            assert metadata.get('gain_db') == gain, (
                f'{name} records gain {metadata.get("gain_db")}, not the {gain} dB it was '
                'captured at -- the record was read when the write ran'
            )
            assert metadata.get('illumination_ma') == 50.0, (
                f'{name} records LED current {metadata.get("illumination_ma")}, not the '
                '50 mA it was lit at -- by write time the run had turned the LED off'
            )
            assert metadata.get('well_label') == well, (
                f'{name} records well {metadata.get("well_label")!r}, not {well!r}: the step '
                "position on the protocol's 6-well plate"
            )
            recorded = datetime.datetime.fromisoformat(metadata['timestamp_iso'])
            assert recorded < write_started, (
                f'{name} is stamped {recorded}, after its write started at {write_started} '
                '-- the timestamp is the write, not the grab'
            )

    def test_a_summed_file_says_how_many_frames_it_sums(self, tmp_path, monkeypatch):
        """The exposure a summed file records is per frame; the file says how
        many frames were summed, so the integration can be recovered."""
        files = _run_with_every_write_held(
            tmp_path, monkeypatch, [_step('C1', 0, x=20.0, gain=1.0, sum_count=3)]
        )

        metadata, _ = files['C1']
        assert metadata.get('frames_summed') == 3, metadata.get('frames_summed')
        assert metadata.get('exposure_time_ms') == 10.0, metadata.get('exposure_time_ms')

    def test_an_autofocused_file_records_the_z_it_was_taken_at(self, tmp_path, monkeypatch):
        """Autofocus leaves the stage at its best focus and the frame is taken
        there; the step row the capture was handed is the one read before the
        sweep, so its Z is the planned height, not the height of the frame."""
        stage_z_at_capture = {}

        def _note_stage_z(writer, step):
            stage_z_at_capture[step['Name']] = writer._scope.motion.get_target_position()['Z']

        files = _run_with_every_write_held(
            tmp_path,
            monkeypatch,
            [_step('C2', 0, x=20.0, gain=1.0, autofocus=True)],
            on_capture=_note_stage_z,
        )

        metadata, _ = files['C2']
        assert stage_z_at_capture['C2'] != 5000.0, (
            'autofocus left the stage at the planned height, so this run cannot tell the '
            "frame's Z from the step's"
        )
        assert metadata.get('z_pos_um') == pytest.approx(stage_z_at_capture['C2'], abs=1e-4), (
            f'the file records Z {metadata.get("z_pos_um")}, but the frame was taken at '
            f'{stage_z_at_capture["C2"]}'
        )


class TestAStepWithNoPlatePositionNamesNoWell:
    """A step with no plate position has no well to name. Once it has been
    through the protocol's table its missing X/Y are NaN, not None; either way
    the file names no well -- and the write does not fail over it."""

    @pytest.mark.parametrize('missing', [None, float('nan')])
    def test_no_well_and_no_failure(self, missing):
        writer = SimpleNamespace(_labware=plate('6 well microplate'))
        step = {'X': missing, 'Y': missing}

        assert protocol_image_writer.ProtocolImageWriter._well_label(writer, step) is None

    def test_a_placed_step_names_the_well_it_lies_in(self):
        writer = SimpleNamespace(_labware=plate('6 well microplate'))

        well = protocol_image_writer.ProtocolImageWriter._well_label(writer, {'X': 60.0, 'Y': 20.0})

        assert well == 'A2'


class TestAManualStillRecordsItsFrame:
    """The manual still hands the save the same record a run's frame does:
    the LED current it was lit at and the well the stage was over."""

    def test_the_lit_current_and_the_well_reach_the_file(self, tmp_path):
        with _open_session(_settings(tmp_path)) as session:
            session.scope.illumination.led_on('BF', 40.0)
            well = session.scope.runtime_state.get_well_label()
            (path,) = _capture(session)

        metadata = read_postproc_input_metadata(path)
        assert metadata.get('illumination_ma') == 40.0, metadata.get('illumination_ma')
        assert well and metadata.get('well_label') == well, (well, metadata.get('well_label'))
        assert metadata.get('frames_summed') == 1
