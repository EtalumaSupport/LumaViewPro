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

The well is named from the position the file records, on the plate the
protocol is written for -- the plate the run moved against -- not the plate
the scope happens to have selected: a 6-well protocol run on a scope set to a
96-well plate names 6-well wells. It is not the step's Well field, which is
empty on an inserted step and keeps its old value when a step is moved, nor
the step's planned X/Y, which a scope with no XY stage never reaches.
"""

import datetime
import pathlib
import threading
import time

import pandas as pd
import pytest

import modules.protocol_image_writer as protocol_image_writer
from modules.image_utils import read_postproc_input_metadata
from modules.run_events import RunEvents
from modules.protocol import Protocol
from modules.recording_frames import FrameFact
from tests.frame_records import plate
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_manual_capture_member import (
    _capture,
    _open_session,
    _settings,
    over_the_first_well,
)


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


def _run_with_every_write_held(tmp_path, monkeypatch, steps, on_capture=None, microscope=None):
    """Run ``steps`` headless with each save held on the file writer, on
    ``microscope`` when named, else the settings' own model.

    Returns ``{step name: (metadata read back, when its write started)}``.
    """
    run_parent = tmp_path / 'runs'
    write_started = {}
    settings = headless_settings(tmp_path)
    if microscope is not None:
        settings['microscope'] = microscope
    with open_composite_session(settings) as (_session, runner):
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

        # The run's files are read once the run says they are written: a
        # file that merely exists may still be mid-write, and one read in
        # its first few hundred bytes has no metadata at all.
        files_written = threading.Event()
        outcome = runner.run_single_scan(
            protocol=_protocol(steps),
            parent_dir=str(run_parent),
            events=RunEvents(files_written=lambda *_written: files_written.set()),
        )
        result = outcome.wait(timeout_s=120.0)
        assert result is not None and result.status == 'completed', result
        assert files_written.wait(FILES_TIMEOUT_S), "the run's files were never written"

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
        sweep, so its Z is the planned height, not the height of the frame.

        The height the frame was taken at is where the stage reports it is,
        not where it was commanded: a focus found between two motor steps is
        reached as the nearest step, so the two differ by a fraction of a
        step (14 of 200 loaded runs, the axis idle in every one).
        """
        stage_z_at_capture = {}

        def _note_stage_z(writer, step):
            reported = writer._scope.motion.axis_positions()['Z']
            stage_z_at_capture[step['Name']] = reported.position

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


class TestAFileNamesTheWellAtThePositionItRecords:
    """The well is read from the frame's position, the one the file records,
    so a file cannot name a well while stating no position."""

    def test_an_unknown_position_names_no_well(self):
        fact = FrameFact(plate_x_mm=None, plate_y_mm=None, z_um=5000.0, moving=False, channel='BF')

        assert fact.well_label(plate('6 well microplate')) is None

    def test_a_known_position_names_the_well_it_lies_in(self):
        fact = FrameFact(plate_x_mm=60.0, plate_y_mm=20.0, z_um=5000.0, moving=False, channel='BF')

        assert fact.well_label(plate('6 well microplate')) == 'A2'

    def test_a_run_on_a_scope_with_no_xy_stage_names_no_well(self, tmp_path, monkeypatch):
        """An LS820 has no X or Y, so it never reaches the well a step plans:
        its file states no position, and names no well either."""
        files = _run_with_every_write_held(
            tmp_path, monkeypatch, [_step('B3', 0, x=20.0, gain=1.0)], microscope='LS820'
        )

        metadata, _ = files['B3']
        assert metadata.get('plate_pos_mm') is None, metadata.get('plate_pos_mm')
        assert not metadata.get('well_label'), (
            f'the file states no position but names well {metadata.get("well_label")!r}, '
            "the well under the step's planned X/Y"
        )


class TestAManualStillRecordsItsFrame:
    """The manual still hands the save the same record a run's frame does:
    the LED current it was lit at and the well the stage was over."""

    def test_the_lit_current_and_the_well_reach_the_file(self, tmp_path):
        with _open_session(_settings(tmp_path)) as session:
            well = over_the_first_well(session)
            session.scope.illumination.led_on('BF', 40.0)
            (path,) = _capture(session)

        metadata = read_postproc_input_metadata(path)
        assert metadata.get('illumination_ma') == 40.0, metadata.get('illumination_ma')
        assert metadata.get('well_label') == well, (well, metadata.get('well_label'))
        assert metadata.get('frames_summed') == 1
