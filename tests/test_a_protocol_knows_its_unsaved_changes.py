# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol answers whether it has changed since it was built, loaded or saved, and which file is its own.

``Protocol.has_unsaved_changes`` and ``Protocol.file_path`` are the API's
answer, so the GUI's New, Load and close ask from the same answer a REST
client reads. A protocol arrives unchanged: the build, the session's load
and the session's save each record it where they return whole, after every
edit they make on the way (the tiles, the stack, the turret slots, the
plate). Every writer of the protocol is then an edit, whichever member it
is; a save is not undone by the channel controls, which the file's Layer
Settings block holds and the record leaves out. A run's copy has no record.
"""

from __future__ import annotations

import datetime

from _lumascope.modules.protocol import Protocol
from tests.guards.test_the_gui_edits_protocols_through_the_api import WRITERS
from tests.scope_fakes import home_sim_scope
from tests.test_a_built_protocol_is_ordered_by_place_then_objective_and_slice import _protocol_file
from tests.test_a_command_for_absent_motion_hardware_is_refused import make_session  # noqa: F401
from tests.test_a_step_records_its_turret_slot import _as_an_old_file


def _ready(session):
    """``session`` homed, with Brightfield acquiring: one New and Add build on."""
    home_sim_scope(session.scope)
    session.set_channel_settings('BF', acquire='image')
    return session


def _arrived_unchanged(protocol, file_path):
    assert (protocol.has_unsaved_changes, protocol.file_path) == (False, file_path)


class TestAProtocolArrivesUnchanged:
    def test_a_load(self, make_session, tmp_path):
        session = make_session('LS820')
        path = session.save_protocol(session.create_empty_protocol(), tmp_path / 'p.tsv')

        _arrived_unchanged(session.load_protocol(path), path)

    def test_a_load_whose_turret_slots_are_filled_at_the_load(self, make_session, tmp_path):
        """A v8 file has no slot column; the load writes each step's slot."""
        session = _ready(make_session('LS850T'))
        protocol = session.create_empty_protocol()
        session.add_step(protocol)
        path = _as_an_old_file(session, protocol, tmp_path)

        loaded = session.load_protocol(path)

        assert list(loaded.steps()['Turret Slot']) == list(protocol.steps()['Turret Slot'])
        _arrived_unchanged(loaded, path)

    def test_a_load_by_a_relative_path_is_its_absolute_file(
        self, make_session, tmp_path, monkeypatch
    ):
        session = make_session('LS820')
        session.save_protocol(session.create_empty_protocol(), tmp_path / 'p.tsv')
        monkeypatch.chdir(tmp_path)

        _arrived_unchanged(session.load_protocol('p.tsv'), tmp_path / 'p.tsv')

    def test_a_tiled_new_on_a_turret_scope(self, make_session):
        session = _ready(make_session('LS850T'))

        protocol = session.new_protocol(tiling='3x3')

        assert protocol.num_steps() > 1
        _arrived_unchanged(protocol, None)

    def test_a_z_stacked_new(self, make_session):
        session = _ready(make_session('LS850'))
        session.update_settings('zstack.range', 20.0)
        session.update_settings('zstack.step_size', 5.0)

        protocol = session.new_protocol(use_zstacking=True)

        assert protocol.num_steps() > 1
        _arrived_unchanged(protocol, None)

    def test_an_empty_protocol(self, make_session):
        _arrived_unchanged(make_session('LS820').create_empty_protocol(), None)

    def test_a_save_after_an_edit(self, make_session, tmp_path):
        session = make_session('LS820')
        protocol = session.create_empty_protocol()
        session.set_protocol_capture_root(protocol, 'edited')
        assert protocol.has_unsaved_changes

        path = session.save_protocol(protocol, tmp_path / 'saved.tsv')

        _arrived_unchanged(protocol, path)


def _shift_every_z(session, protocol):
    """A scan's copy of ``protocol`` with every Z moved, for ``adopt_focus_from``."""
    scanned = protocol.copy_for_execution()
    for idx in range(scanned.num_steps()):
        scanned.modify_step_z_height(step_index=idx, z=scanned.step_row(step_index=idx)['Z'] + 7)
    protocol.adopt_focus_from(scanned)


def _other_turret_slot(session, protocol):
    slot = int(protocol.steps()['Turret Slot'].iloc[0])
    protocol.set_turret_slot([0], 2 if slot == 1 else 1)


def _stack_as_stored(session, protocol):
    # The stored stack stays written for the edits after this one on the
    # shared session; of those only ``insert_step`` reads the store, and it
    # adds without stacking.
    session.update_settings('zstack.range', 10.0)
    session.update_settings('zstack.step_size', 5.0)
    session.update_settings('zstack.position', 'bottom')
    session.apply_zstacking(protocol)


# One edit per writer of the protocol, each one that changes what a save
# writes: the guard's WRITERS, so a writer added there without a case here
# fails the walk below. ``record_saved`` records; it is not an edit.
_EDITS = {
    'adopt_focus_from': _shift_every_z,
    'apply_focus_all_channel_steps': lambda s, p: p.apply_focus_all_channel_steps('BF', 1234.0),
    'apply_tiling': lambda s, p: s.apply_tiling(p, '2x2'),
    'apply_zstack_group_focus': lambda s, p: p.apply_zstack_group_focus(0, 1234.0),
    'apply_zstacking': _stack_as_stored,
    'delete_step': lambda s, p: s.delete_step(p, 0),
    'insert_step': lambda s, p: s.add_step(p),
    'modify_autofocus': lambda s, p: p.modify_autofocus(
        step_index=0, enabled=not p.step_row(step_index=0)['Auto_Focus']
    ),
    'modify_autofocus_all_steps': lambda s, p: p.modify_autofocus_all_steps(
        enabled=not p.step_row(step_index=0)['Auto_Focus']
    ),
    'modify_capture_root': lambda s, p: s.set_protocol_capture_root(p, 'another root'),
    'modify_labware': lambda s, p: s.set_protocol_labware(p, '6 well microplate'),
    'modify_name': lambda s, p: s.rename_step(p, 0, 'renamed'),
    'modify_step': lambda s, p: s.update_step(p, 0, channel='BF', label='updated'),
    'modify_step_z_height': lambda s, p: p.modify_step_z_height(step_index=0, z=1234.0),
    'modify_time_params': lambda s, p: p.modify_time_params(
        period=datetime.timedelta(minutes=3), duration=datetime.timedelta(hours=5)
    ),
    'optimize_step_ordering': lambda s, p: p.optimize_step_ordering(),
    'set_turret_slot': _other_turret_slot,
}


def _unordered_file(session, tmp_path):
    """A protocol file whose steps are out of travel order, two objectives, every step placed."""
    protocol = session.load_protocol(_protocol_file(session, tmp_path / 'unordered.tsv'))
    # The two steps with no X/Y: a grid cannot be laid around them.
    for idx in (6, 1):
        session.delete_step(protocol, idx)
    return session.save_protocol(protocol, tmp_path / 'base.tsv')


def test_every_writer_of_the_protocol_is_an_unsaved_change(make_session, tmp_path):
    """Each writer's edit, on a protocol loaded unordered on a turret scope; then a save undoes it."""
    assert set(_EDITS) == WRITERS - {'record_saved'}
    session = _ready(make_session('LS850T'))
    path = _unordered_file(session, tmp_path)
    unnoticed, still_unsaved = [], []
    for name, edit in _EDITS.items():
        protocol = session.load_protocol(path)
        edit(session, protocol)
        if not protocol.has_unsaved_changes:
            unnoticed.append(name)
        session.save_protocol(protocol, tmp_path / f'{name}.tsv')
        if protocol.has_unsaved_changes:
            still_unsaved.append(name)

    assert (unnoticed, still_unsaved) == ([], [])


def test_the_channel_controls_are_not_a_change(make_session, tmp_path):
    """The Layer Settings block is the session's controls at the save, not the protocol."""
    session = _ready(make_session('LS820'))
    protocol = session.new_protocol()
    session.save_protocol(protocol, tmp_path / 'p.tsv')

    session.set_channel_settings('BF', gain_db=7.0)

    assert protocol.has_unsaved_changes is False


def test_a_runs_write_of_its_copy_is_not_a_save(make_session, tmp_path):
    """A run's copy is not the person's protocol: writing it records nothing."""
    session = make_session('LS820')
    protocol = session.create_empty_protocol()
    copy = protocol.copy_for_execution()

    copy.to_file(tmp_path / 'run.tsv')

    assert (copy.has_unsaved_changes, copy.file_path) == (True, None)
    assert protocol.file_path is None


def test_a_rest_client_reads_the_same_answers(make_session, tmp_path):
    from fastapi.testclient import TestClient

    from _lumascope.rest.app import build_app

    session = make_session('LS820')
    session.save_protocol(session.create_empty_protocol(), tmp_path / 'p.tsv')
    with TestClient(build_app(session)) as client:
        handle = client.post('/api/v1/load_protocol', json={'file_path': 'p.tsv'}).json()['handle']
        route = f'/api/v1/handles/Protocol/{handle}'
        before = (
            client.get(f'{route}/has_unsaved_changes').json(),
            client.get(f'{route}/file_path').json(),
        )
        client.post(
            '/api/v1/set_protocol_capture_root', json={'protocol': handle, 'root': 'edited'}
        )
        after = client.get(f'{route}/has_unsaved_changes').json()

    assert (before[0], before[1]['host_path']) == (False, str(tmp_path / 'p.tsv'))
    assert after is True


def test_a_protocol_no_build_load_or_save_returned_has_no_record(tmp_path):
    """The post-processing reader's protocol: never the person's, so never unchanged."""
    from tests.protocol_fixtures import protocol_from_file
    from tests.test_protocol_roundtrip import _build_protocol, _make_step

    path = tmp_path / 'p.tsv'
    _build_protocol([_make_step(name='A1_BF', label='a')]).to_file(path)

    read = protocol_from_file(file_path=path)

    assert isinstance(read, Protocol)
    assert (read.has_unsaved_changes, read.file_path) == (True, None)
