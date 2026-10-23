# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step may have no position on an axis, and is refused where the scope has that axis.

A scope records no position on an axis it lacks. Such a step is blank in the
``.tsv``, NaN in ``Protocol.steps()``, None in ``Protocol.step_row()``. On a scope
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
    session.set_channel_acquire('BF', 'image')
    for i in range(count):
        session.scope.motion.move_xy(x_um=10000.0 + 5000.0 * i)
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

    assert (protocol.step_row(step_index=0)['X'], protocol.step_row(step_index=0)['Y']) == (
        None,
        None,
    )
    assert protocol.steps()['X'].isna().all()
    assert (reloaded.step_row(step_index=1)['X'], reloaded.step_row(step_index=1)['Y']) == (
        None,
        None,
    )
    assert reloaded.step_row(step_index=1)['Z'] is not None


def test_post_processing_reads_a_blank_position_as_no_position(make_session, tmp_path):
    from modules.protocol import Protocol
    from tests.test_protocol_roundtrip import TILING_CONFIGS

    path = _blank_xy_file(make_session, tmp_path)

    record = Protocol.from_file(
        file_path=path, tiling_configs_file_loc=TILING_CONFIGS, runnable=False
    )

    assert (record.step_row(step_index=0)['X'], record.step_row(step_index=0)['Y']) == (None, None)


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

    assert protocol.step_row(step_index=1)['Name'] in str(refusal)
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


def test_a_step_at_a_position_and_one_named_for_another_well_are_refused_without_a_stage(
    make_session, tmp_path
):
    stage = make_session('LS850')
    path = _rewrite_steps(
        _saved_steps(stage, tmp_path / 'mixed.tsv', 2),
        X=['14.38', ''],
        Y=['11.24', ''],
        Well=['A1', 'A2'],
    )
    session = make_session('LS820')
    protocol = session.load_protocol(path)

    refusal = _refused(lambda: session.create_protocol_runner().start_single_scan(protocol))

    assert refusal.reason == 'positions_unreachable'
    assert 'name different wells or tiles' in str(refusal)


def test_a_configuration_with_x_and_no_y_is_refused_naming_the_step(make_session):
    session = make_session('LS850', acquiring=('BF',))
    config = session.get_sequenced_capture_config()
    config['positions'] = [{'x': 10.0, 'y': None, 'z': 5000.0, 'name': 'A1'}]

    with pytest.raises(ProtocolFormatError, match='one of X and Y without the other'):
        session.scope.protocols.create_protocol(input_config=config)


# --- What a scope without an axis records -----------------------------------------------


@pytest.mark.parametrize(
    # parametrize-justified: each model is its own scope and session
    ('model', 'expected'),
    [('LS620', (None, None, None)), ('LS820', (None, None, 'measured'))],
)
def test_the_plate_read_and_the_steps_record_nothing_for_an_absent_axis(
    make_session, model, expected
):
    session = make_session(model, acquiring=('BF',))

    here = session.get_current_plate_position()
    added = session.create_empty_protocol()
    session.add_step(added)
    new = session.new_protocol()

    def shape(x, y, z):
        return (x, y, None if z is None else 'measured')

    assert shape(here.x_mm, here.y_mm, here.z_um) == expected
    step = added.step_row(step_index=0)
    assert shape(step['X'], step['Y'], step['Z']) == expected
    first = new.step_row(step_index=0)
    assert shape(first['X'], first['Y'], first['Z']) == expected
    assert first['Well'] == 'A1'


@pytest.mark.parametrize('model', ['LS620', 'LS820'])  # parametrize-justified: one scope each
def test_a_composite_runs_on_a_scope_without_a_stage(make_session, tmp_path, model):
    session = make_session(model)
    # One transmitted and one fluorescence channel: two planes of a composite.
    for layer in ('BF', 'Blue'):
        session.set_channel_acquire(layer, 'image')

    outcome = session.create_protocol_runner().run_composite(
        parent_dir=str(tmp_path / 'composite'), timeout_s=120
    )

    assert outcome.status == 'completed', outcome


def test_channels_added_together_without_a_stage_are_one_composite(make_session, tmp_path):
    # #772's case on a scope whose steps carry no X/Y.
    session = make_session('LS820')
    # One transmitted and one fluorescence channel: two planes of a composite.
    for layer in ('BF', 'Blue'):
        session.set_channel_acquire(layer, 'image')
    protocol = session.create_empty_protocol()
    session.add_step(protocol)

    pending = session.create_protocol_runner().start_single_scan(
        protocol, parent_dir=str(tmp_path / 'runs')
    )
    assert pending.result(timeout=120).status == 'completed'
    pending.files.result(timeout=120)
    run_folder = next((tmp_path / 'runs').iterdir())

    built = session.post_processing.composite(run_folder)

    assert built.new_count == 1, built.message


def test_update_step_keeps_a_loaded_steps_x_and_y_without_a_stage(make_session, tmp_path):
    stage = make_session('LS850')
    path = _saved_steps(stage, tmp_path / 'from_a_stage.tsv', 1)
    session = make_session('LS820', acquiring=('BF',))
    protocol = session.load_protocol(path)
    before = (protocol.step_row(step_index=0)['X'], protocol.step_row(step_index=0)['Y'])

    session.update_step(protocol, 0, channel='BF')

    assert (protocol.step_row(step_index=0)['X'], protocol.step_row(step_index=0)['Y']) == before
    assert None not in before


def test_writes_for_an_axis_the_scope_lacks_are_refused(make_session):
    from modules.exceptions import HardwareCommandRefusedError

    no_stage = make_session('LS820', acquiring=('BF',))
    manual = make_session('LS620', acquiring=('BF',))
    protocol = manual.create_empty_protocol()
    manual.add_step(protocol)

    bookmark_before = dict(no_stage.settings['bookmark'])
    tiling = _refused(lambda: no_stage.new_protocol(tiling='3x3'))
    step_z = _refused(lambda: manual.scope.protocols.set_step_z(protocol, 0, 123.0))
    with pytest.raises(HardwareCommandRefusedError) as bookmark:
        no_stage.save_bookmark(('X',))

    assert (tiling.reason, step_z.reason, bookmark.value.reason) == (
        'positions_unreachable',
        'positions_unreachable',
        'axis_absent',
    )
    assert protocol.step_row(step_index=0)['Z'] is None
    assert no_stage.settings['bookmark'] == bookmark_before


def test_an_autofocus_and_a_z_stack_run_on_a_scope_without_a_stage(make_session, tmp_path):
    session = make_session(
        'LS820',
        acquiring=('BF',),
        zstack={'range': 20.0, 'step_size': 10.0, 'position': 'Current Position at Center'},
    )
    # Mid-travel: homed Z sits at 0, where a stack's lower slices fall outside.
    session.scope.motion.move_z(5000.0)
    runner = session.create_protocol_runner()

    focused = runner.run_autofocus('BF', parent_dir=str(tmp_path / 'af'), timeout_s=180)
    stacked = runner.run_zstack('BF', parent_dir=str(tmp_path / 'zs'), timeout_s=180)

    assert (focused.status, stacked.status) == ('completed', 'completed'), (focused, stacked)


# --- A scope without an XY stage has one plate ------------------------------------------


def test_a_plate_with_wells_is_refused_where_it_is_selected_without_a_stage(make_session):
    from modules.exceptions import HardwareCommandRefusedError, MissingPart
    from modules.labware_loader import CENTER_PLATE

    session = make_session('LS820')

    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.select_labware('96 well microplate')

    assert (refused.value.reason, refused.value.missing) == ('axis_absent', MissingPart.XY_STAGE)
    assert session.settings['protocol']['labware'] == CENTER_PLATE
