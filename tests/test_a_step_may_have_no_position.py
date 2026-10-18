# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step may have no position on an axis, and is refused where the scope has that axis.

A scope records no position on an axis it lacks. Such a step is blank in the
``.tsv``, NaN in ``Protocol.steps()``, None in ``Protocol.step()``. On a scope
that has the axis the step says nothing about where to go, so the run, Go To
Step and the tile and z-stack builds refuse it by name
(``step_position_missing``). On a scope without X/Y, steps with no X/Y whose
wells differ would be one field saved under several names, so the run
refuses them (``positions_unreachable``).

Driven through the session on the simulated scopes; the blank file is made as
a user's would be, saved on a stage scope and blanked as a scope without that
axis records it.
"""

from __future__ import annotations

import csv

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolRunRefusedError
from modules.protocol import ProtocolFormatError
from tests.test_a_command_for_absent_motion_hardware_is_refused import make_session  # noqa: F401


def _saved_steps(session, path, count):
    """Save a protocol of ``count`` steps added on ``session`` at different X."""
    protocol = session.create_empty_protocol()
    session.set_layer_acquire('BF', 'image')
    for i in range(count):
        session.scope.motion.move_absolute('X', 10000.0 + 5000.0 * i)
        session.add_step(protocol)
    protocol.to_file(path)
    return path


def _rewrite_steps(path, **columns):
    """Rewrite the step rows' ``columns`` (a value per step, or one for all)."""
    with open(path, newline='') as fp:
        rows = list(csv.reader(fp, delimiter='\t'))
    header = next(i for i, row in enumerate(rows) if row[:1] == ['Name'])
    steps = rows[header + 1 :]
    for column, values in columns.items():
        at = rows[header].index(column)
        for i, row in enumerate(steps):
            row[at] = values[i] if isinstance(values, list) else values
    with open(path, 'w', newline='') as fp:
        csv.writer(fp, delimiter='\t', lineterminator='\n').writerows(rows)
    return path


def _blank_xy_file(make_session, tmp_path, count=2, **columns):
    stage = make_session('LS850')
    path = _saved_steps(stage, tmp_path / 'recorded.tsv', count)
    return _rewrite_steps(path, X='', Y='', **columns)


# --- The representation -----------------------------------------------------------------


def test_a_blank_position_loads_as_no_position_and_saves_blank(make_session, tmp_path):
    session = make_session('LS820')
    path = _blank_xy_file(make_session, tmp_path)

    protocol = session.load_protocol(path)
    protocol.to_file(tmp_path / 'again.tsv')
    reloaded = session.load_protocol(tmp_path / 'again.tsv')

    assert (protocol.step(idx=0)['X'], protocol.step(idx=0)['Y']) == (None, None)
    assert protocol.steps()['X'].isna().all()
    assert (reloaded.step(idx=1)['X'], reloaded.step(idx=1)['Y']) == (None, None)
    assert reloaded.step(idx=1)['Z'] is not None


def test_post_processing_reads_a_blank_position_as_no_position(make_session, tmp_path):
    from modules.protocol import Protocol
    from tests.test_protocol_roundtrip import TILING_CONFIGS

    path = _blank_xy_file(make_session, tmp_path)

    record = Protocol.from_file(
        file_path=path, tiling_configs_file_loc=TILING_CONFIGS, runnable=False
    )

    assert (record.step(idx=0)['X'], record.step(idx=0)['Y']) == (None, None)


def test_one_of_x_and_y_without_the_other_is_refused_on_load(make_session, tmp_path):
    session = make_session('LS850')
    path = _rewrite_steps(_saved_steps(session, tmp_path / 'half.tsv', 1), Y='')

    with pytest.raises(ProtocolFormatError, match='one of X and Y without the other'):
        session.load_protocol(path)


def test_ordering_the_steps_keeps_those_without_a_position(make_session, tmp_path):
    session = make_session('LS820')
    path = _blank_xy_file(make_session, tmp_path, count=3)
    protocol = session.load_protocol(path)

    protocol.optimize_step_ordering()

    assert protocol.num_steps() == 3


def test_a_step_without_x_and_y_is_never_the_nearest(make_session, tmp_path):
    stage = make_session('LS850')
    path = _rewrite_steps(
        _saved_steps(stage, tmp_path / 'mixed.tsv', 2), X=['', '50.0'], Y=['', '40.0']
    )
    protocol = stage.load_protocol(path)

    assert config_helpers.find_nearest_step(x=0.0, y=0.0, protocol=protocol) == 1


# --- Refused on a scope that has the axis ---------------------------------------------


def _refused(call) -> ProtocolRunRefusedError:
    with pytest.raises(ProtocolRunRefusedError) as refused:
        call()
    return refused.value


def test_a_step_without_x_and_y_is_refused_by_every_door_on_a_stage_scope(make_session, tmp_path):
    path = _blank_xy_file(make_session, tmp_path)
    stage = make_session('LS850')
    protocol = stage.load_protocol(path)

    doors = {
        'run': lambda: stage.create_protocol_runner().start_single_scan(protocol),
        'go to step': lambda: stage.scope.protocols.step_targets(protocol, 0),
        'tile': lambda: stage.scope.protocols.apply_tiling(
            protocol,
            '2x2',
            frame_dimensions={'width': 1900, 'height': 1900},
            binning_size=1,
            overlap_percent=10,
        ),
        'stack': lambda: stage.scope.protocols.apply_zstacking(
            protocol, range_um=10, step_size_um=5, z_reference='center'
        ),
    }
    reasons = {door: _refused(call).reason for door, call in doors.items()}

    assert reasons == dict.fromkeys(doors, 'step_position_missing')
    assert protocol.num_steps() == 2


def test_the_refusal_names_the_steps_and_the_axes(make_session, tmp_path):
    path = _blank_xy_file(make_session, tmp_path)
    stage = make_session('LS850')
    protocol = stage.load_protocol(path)

    refusal = _refused(lambda: stage.scope.protocols.step_targets(protocol, 1))

    assert protocol.step(idx=1)['Name'] in str(refusal)
    assert 'no X, Y position' in str(refusal)


# --- On a scope without X/Y -----------------------------------------------------------


def test_steps_without_x_and_y_at_one_place_run_on_a_scope_without_a_stage(make_session, tmp_path):
    path = _blank_xy_file(make_session, tmp_path, Well='')
    session = make_session('LS820')
    protocol = session.load_protocol(path)

    outcome = (
        session.create_protocol_runner()
        .start_single_scan(protocol, parent_dir=str(tmp_path / 'run'), enable_image_saving=False)
        .result(timeout=120)
    )

    assert outcome.status == 'completed', outcome


def test_steps_without_x_and_y_naming_different_wells_are_refused_without_a_stage(
    make_session, tmp_path
):
    path = _blank_xy_file(make_session, tmp_path, count=4, Well=['A1', 'A2', 'A3', 'A4'])
    session = make_session('LS820')
    protocol = session.load_protocol(path)

    refusal = _refused(lambda: session.create_protocol_runner().start_single_scan(protocol))

    assert refusal.reason == 'positions_unreachable'
    assert 'name different wells or tiles' in str(refusal)
