# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run stopped partway through a scan stacks each well's complete scans.

A protocol run's hyperstack holds a well's scans as its T axis, and a stop
can cut only the last scan short. That scan's images, when they are fewer
than the protocol plans for the well there, are left out of its stack,
and the build's answer names the well and how many it left out. A well
whose only scan was cut short gets no stack and is named. A gap in any
earlier scan, or a lost frame in a scan that ran whole, still refuses the
well. The plan is the run's own protocol, so a well that reached only some
of its channels is never stacked as if those were all.

Pinned through ``StackBuilder.load_folder``, the build a run's
``PostProcessingAPI.run_hyperstack`` puts on its lane, over a run folder
written by the production protocol and run record.
"""

from __future__ import annotations

import datetime
import pathlib

import pytest
import tifffile

import _lumascope.modules.config_helpers as config_helpers
from _lumascope.modules.channel_record import release_colors
from _lumascope.modules.exceptions import PostProcessingFailedError
from _lumascope.modules.protocol_execution_record import ProtocolExecutionRecord
from _lumascope.modules.stack_builder import StackBuilder
from tests.installation_fixtures import shipped_grids
from tests.stock_channels import LED0, LED1, LED2
from tests.test_adding_a_step_is_an_api_capability import (  # noqa: F401 -- pytest fixture
    session,
    turret_in_a_known_slot,
)
from tests.test_an_add_is_one_channel_group import _added, _write_tiff
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture

FOUR = ('BF', LED0, LED1, LED2)


def _four_channel_protocol(session, *, tiling='1x1', use_zstacking=False):
    """A New protocol taking BF and three fluorescence channels at every well."""
    for layer in config_helpers.get_channel_configs(session.settings):
        session.set_channel_settings(layer, acquire='image' if layer in FOUR else None)
        session.set_channel_settings(layer, autofocus=False)
    if use_zstacking:
        session.update_settings('zstack.range', 10.0)
        session.update_settings('zstack.step_size', 5.0)
        session.update_settings('zstack.position', 'center')
    return session.new_protocol(tiling=tiling, use_zstacking=use_zstacking)


def _wells(protocol, count=2):
    wells = list(dict.fromkeys(protocol.steps()['Well']))
    return wells[:count]


def _steps_of(protocol, well, *, tile=None):
    steps = protocol.steps().reset_index(drop=True)
    at = steps['Well'] == well
    if tile is not None:
        at &= steps['Tile'] == tile
    return list(steps.index[at])


def _run_folder(tmp_path, protocol, captured, *, failed=(), gone=()):
    """The folder a run of *protocol* leaves: its protocol, its record, an image per capture.

    *captured* is each (step index, scan) the run saved, in run order;
    *failed* each one the run recorded as a failed capture; *gone* each
    saved one whose file has since left the folder.
    """
    folder = tmp_path / 'run'
    folder.mkdir()
    protocol.to_file(folder / 'run.tsv')
    record = ProtocolExecutionRecord(
        protocol_file_loc=pathlib.Path('run.tsv'),
        outfile=folder / ProtocolExecutionRecord.DEFAULT_FILENAME,
    )
    start = datetime.datetime(2026, 11, 22, 12, 0, 0)
    for n, (step_index, scan) in enumerate(captured):
        step = protocol.step_row(step_index=step_index)
        name = pathlib.Path(f'{step["Name"]}_{step_index:03}_{scan:04}.tiff')
        _write_tiff(folder / name, step['channel'])
        record.add_step(
            capture_result_file_name=name,
            step_name=step['Name'],
            step_index=step_index,
            scan_count=scan,
            timestamp=start + datetime.timedelta(seconds=n),
        )
        if (step_index, scan) in gone:
            (folder / name).unlink()
    for step_index, scan in failed:
        step = protocol.step_row(step_index=step_index)
        record.add_step(
            capture_result_file_name='capture_failed',
            step_name=step['Name'],
            step_index=step_index,
            scan_count=scan,
            timestamp=start + datetime.timedelta(hours=1),
            frame_count=0,
        )
    record.complete()
    return folder


def _build(folder):
    return StackBuilder(has_turret=False).load_folder(
        path=folder,
        tiling_grids=shipped_grids(),
        channel_colors=release_colors(),
        save_encoding='right_aligned',
    )


def _scans(steps, scans):
    return [(step, scan) for scan in scans for step in steps]


def _stack_shapes(artifact_paths):
    shapes = []
    for path in artifact_paths:
        with tifffile.TiffFile(path) as tif:
            shapes.append(tif.series[0].shape)
    return shapes


def test_a_well_cut_short_in_its_last_scan_stacks_the_scans_before_it(session, tmp_path):
    protocol = _four_channel_protocol(session)
    first, cut = _wells(protocol)
    whole, partial = _steps_of(protocol, first), _steps_of(protocol, cut)
    captured = _scans(whole, range(4)) + _scans(partial, range(3)) + _scans(partial[:3], [3])

    answer = _build(_run_folder(tmp_path, protocol, captured))

    assert answer['new_count'] == 2
    assert sorted(_stack_shapes(answer['artifact_paths'])) == [(3, 4, 8, 8), (4, 4, 8, 8)]
    assert f'well {cut} (3 image(s))' in answer['message']
    assert 'ended partway through its last scan' in answer['message']


def test_a_well_cut_short_in_its_only_scan_gets_no_stack_and_is_named(session, tmp_path):
    protocol = _four_channel_protocol(session)
    first, cut = _wells(protocol)
    captured = _scans(_steps_of(protocol, first), [0]) + _scans(_steps_of(protocol, cut)[:2], [0])

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, captured))

    fault = raised.value
    assert fault.reason == 'post_processing_incomplete'
    assert len(fault.produced_paths) == 1
    assert [cause for _, cause in fault.errors] == [
        f'Well {cut} has no complete scan: the run ended partway through its only scan. '
        'Its 2 image(s) are saved.'
    ]


def test_a_well_cut_after_its_first_image_is_named_not_skipped(session, tmp_path):
    protocol = _four_channel_protocol(session)
    (cut,) = _wells(protocol, count=1)
    captured = _scans(_steps_of(protocol, cut)[:1], [0])

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, captured))

    # The build made nothing: a failure, never the "no data" refusal a
    # folder with nothing to stack gets.
    fault = raised.value
    assert fault.reason == 'post_processing_failed'
    assert fault.produced_paths == ()
    assert f'Well {cut} has no complete scan' in str(fault)


def test_a_gap_in_an_earlier_scan_still_refuses_the_well(session, tmp_path):
    protocol = _four_channel_protocol(session)
    (well,) = _wells(protocol, count=1)
    steps = _steps_of(protocol, well)
    captured = _scans(steps, [0]) + _scans(steps[:3], [1]) + _scans(steps, [2, 3])

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, captured))

    assert 'not all captured at the same z-slices and scan counts' in str(raised.value)
    assert 'ended partway' not in str(raised.value)


def test_a_protocol_whose_channels_are_not_aligned_is_still_refused(session, tmp_path):
    protocol = _four_channel_protocol(session, use_zstacking=True)
    (well,) = _wells(protocol, count=1)
    # One channel keeps only its middle slice: the protocol itself plans a
    # grid with holes, and every scan holds all of its plan.
    steps = protocol.steps().reset_index(drop=True)
    trimmed = steps.index[
        (steps['Well'] == well) & (steps['channel'] == LED2) & (steps['Z-Slice'] != 1)
    ]
    for step_index in sorted(trimmed, reverse=True):
        session.delete_step(protocol, step_index)
    captured = _scans(_steps_of(protocol, well), range(2))

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, captured))

    assert 'not all captured at the same z-slices and scan counts' in str(raised.value)
    assert 'ended partway' not in str(raised.value)


def test_a_failed_capture_in_the_cut_scan_is_left_out_with_it(session, tmp_path):
    protocol = _four_channel_protocol(session)
    (well,) = _wells(protocol, count=1)
    steps = _steps_of(protocol, well)
    captured = _scans(steps, range(3)) + _scans(steps[:2], [3])

    answer = _build(_run_folder(tmp_path, protocol, captured, failed=[(steps[2], 3)]))

    assert _stack_shapes(answer['artifact_paths']) == [(3, 4, 8, 8)]
    assert f'well {well} (3 image(s))' in answer['message']


def test_a_failed_capture_in_a_last_scan_that_ran_whole_still_refuses_the_well(session, tmp_path):
    protocol = _four_channel_protocol(session)
    (well,) = _wells(protocol, count=1)
    steps = _steps_of(protocol, well)
    captured = _scans(steps, range(3)) + _scans(steps[:3], [3])

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, captured, failed=[(steps[3], 3)]))

    assert 'its capture failed during the run' in str(raised.value)
    assert 'ended partway' not in str(raised.value)


def test_a_frame_lost_from_an_earlier_scan_still_refuses_the_well(session, tmp_path):
    protocol = _four_channel_protocol(session)
    (well,) = _wells(protocol, count=1)
    steps = _steps_of(protocol, well)
    captured = _scans(steps, range(3)) + _scans(steps[:2], [3])

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, captured, gone=[(steps[1], 1)]))

    assert 'frame(s) the run recorded are missing' in str(raised.value)


def test_a_well_the_stop_never_reached_in_the_last_scan_stacks_quietly(session, tmp_path):
    protocol = _four_channel_protocol(session)
    unreached, reached = _wells(protocol)
    captured = _scans(_steps_of(protocol, unreached), range(3)) + _scans(
        _steps_of(protocol, reached), range(4)
    )

    answer = _build(_run_folder(tmp_path, protocol, captured))

    assert sorted(_stack_shapes(answer['artifact_paths'])) == [(3, 4, 8, 8), (4, 4, 8, 8)]
    assert answer['message'] == 'Success.'


@pytest.mark.usefixtures('turret_in_a_known_slot')
def test_a_custom_step_cut_short_in_its_only_scan_is_named_by_its_label(scope, tmp_path):
    protocol = _added(scope, adds=(FOUR,))
    first = protocol.step_row(step_index=0)
    # The label names the Add's every step; the step's own name, one channel.
    assert first['Label'] != first['Name']

    with pytest.raises(PostProcessingFailedError) as raised:
        _build(_run_folder(tmp_path, protocol, [(0, 0), (1, 0)]))

    assert f'Step {first["Label"]} has no complete scan' in str(raised.value)


def test_a_tile_cut_partway_through_its_z_stack_leaves_out_only_that_tile(session, tmp_path):
    protocol = _four_channel_protocol(session, tiling='2x2', use_zstacking=True)
    (well,) = _wells(protocol, count=1)
    tiles = list(dict.fromkeys(protocol.steps()['Tile'][protocol.steps()['Well'] == well]))
    reached, cut = tiles[0], tiles[1]
    captured = (
        _scans(_steps_of(protocol, well), range(3))
        + _scans(_steps_of(protocol, well, tile=reached), [3])
        + _scans(_steps_of(protocol, well, tile=cut)[:5], [3])
    )

    answer = _build(_run_folder(tmp_path, protocol, captured))

    shapes = _stack_shapes(answer['artifact_paths'])
    assert sorted(shape[0] for shape in shapes) == [3, 3, 3, 4]
    assert all(shape[1:3] == (3, 4) for shape in shapes)
    assert f'well {well} (5 image(s))' in answer['message']
