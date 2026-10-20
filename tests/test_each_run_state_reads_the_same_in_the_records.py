# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each run state reads the same in the records as in the reads they replace.

The Session answered its run state through ten reads, one fact each; the
records answer it now: ``controls_lock`` which of a person's controls are
locked and by what, ``live_work`` what the session is doing, and the scope's
capability whether it has a stage to move. Before the ten reads retire, each
is shown to answer what its record answers in every state the Session can be
in: idle, a run starting (its claim granted, the engine not yet running), a
run, a run unwinding after its release, a diagnostic, a run lent a
diagnostic's claim, a home, a live recording, and a finished run's files
still writing. The record's answer in each state is stated, so the states
stay pinned once the reads are gone.
"""

import contextlib
import threading

import pytest

from modules import live_work
from modules.protocol_image_writer import RunWriteBatch
from modules.protocol_state_machine import ProtocolState
from tests.protocol_drives import run_identity

COMPLETION_TIMEOUT = 30


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        yield session
    finally:
        session.shutdown()


@contextlib.contextmanager
def _idle(session, monkeypatch):
    yield


@contextlib.contextmanager
def _a_run_starting(session, monkeypatch):
    held = session.activity_claim.try_claim('protocol', run=run_identity())
    try:
        yield
    finally:
        held.release()


@contextlib.contextmanager
def _a_run(session, monkeypatch):
    held = session.activity_claim.try_claim('protocol', run=run_identity())
    # The engine's phase, set as start() sets it once the claim is taken;
    # a run driven to this edge on the simulator would end before it is read.
    monkeypatch.setattr(session.sequenced_capture_runner, '_state', ProtocolState.RUNNING)
    try:
        yield
    finally:
        monkeypatch.setattr(session.sequenced_capture_runner, '_state', ProtocolState.IDLE)
        held.release()


@contextlib.contextmanager
def _a_run_unwinding(session, monkeypatch):
    # After the run's release and before its cleanup ends.
    monkeypatch.setattr(session.sequenced_capture_runner, '_state', ProtocolState.COMPLETING)
    try:
        yield
    finally:
        monkeypatch.setattr(session.sequenced_capture_runner, '_state', ProtocolState.IDLE)


@contextlib.contextmanager
def _a_diagnostic(session, monkeypatch):
    with session.diagnostic_claim():
        yield


@contextlib.contextmanager
def _a_run_lent_a_diagnostics_claim(session, monkeypatch):
    with session.diagnostic_claim() as held:
        run = held.lend().try_claim('protocol', run=run_identity('api_autofocus'))
        monkeypatch.setattr(session.sequenced_capture_runner, '_state', ProtocolState.RUNNING)
        try:
            yield
        finally:
            monkeypatch.setattr(session.sequenced_capture_runner, '_state', ProtocolState.IDLE)
            run.release()


@contextlib.contextmanager
def _a_home(session, monkeypatch):
    held = session.activity_claim.try_claim('home')
    try:
        yield
    finally:
        held.release()


@contextlib.contextmanager
def _a_live_recording(session, monkeypatch):
    operation = session.manual_recording.start()
    try:
        yield
    finally:
        operation.stop()
        operation.result(timeout=COMPLETION_TIMEOUT)


@contextlib.contextmanager
def _a_finished_runs_files_writing(session, monkeypatch):
    # The run's batch, closed with two of its writes still to land: the
    # first held on the file lane until the state has been read.
    landing = threading.Event()
    batch = RunWriteBatch(session.file_io_executor)
    batch.submit(lambda: landing.wait(COMPLETION_TIMEOUT), {}, what='an image', pace_until=None)
    batch.submit(lambda: None, {}, what='an image', pace_until=None)
    monkeypatch.setattr(session.sequenced_capture_runner, '_write_batch', batch)
    batch.close()
    try:
        yield
    finally:
        landing.set()
        assert batch.wait_complete(COMPLETION_TIMEOUT)


# Per state: the records' answer -- live_work's kinds, the run files' count
# left, the lock's run sentence and its recording -- and the ten reads'.
STATES = [
    (
        _idle,
        ([], None, None, False),
        {
            'is_protocol_running': False,
            'run_in_progress': False,
            'exclusive_activity': None,
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': False,
            'run_lockout_named': None,
            'recording_active': False,
            'controls_locked': False,
            'motion_enabled': True,
        },
    ),
    (
        _a_run_starting,
        ([live_work.PROTOCOL], None, 'The scan run is in progress.', False),
        {
            'is_protocol_running': True,
            'run_in_progress': False,
            'exclusive_activity': 'protocol',
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': True,
            'run_lockout_named': 'The scan run is in progress.',
            'recording_active': False,
            'controls_locked': True,
            'motion_enabled': False,
        },
    ),
    (
        _a_run,
        ([live_work.PROTOCOL], None, 'The scan run is in progress.', False),
        {
            'is_protocol_running': True,
            'run_in_progress': True,
            'exclusive_activity': 'protocol',
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': True,
            'run_lockout_named': 'The scan run is in progress.',
            'recording_active': False,
            'controls_locked': True,
            'motion_enabled': False,
        },
    ),
    (
        _a_run_unwinding,
        ([live_work.PROTOCOL], None, None, False),
        {
            'is_protocol_running': False,
            'run_in_progress': True,
            'exclusive_activity': None,
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': False,
            'run_lockout_named': None,
            'recording_active': False,
            'controls_locked': False,
            'motion_enabled': True,
        },
    ),
    (
        _a_diagnostic,
        ([live_work.DIAGNOSTIC], None, 'A diagnostic is in progress.', False),
        {
            'is_protocol_running': False,
            'run_in_progress': False,
            'exclusive_activity': 'diagnostic',
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': True,
            'run_lockout_named': 'A diagnostic is in progress.',
            'recording_active': False,
            'controls_locked': True,
            'motion_enabled': False,
        },
    ),
    (
        _a_run_lent_a_diagnostics_claim,
        ([live_work.DIAGNOSTIC, live_work.PROTOCOL], None, 'A diagnostic is in progress.', False),
        {
            'is_protocol_running': True,
            'run_in_progress': True,
            'exclusive_activity': 'diagnostic',
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': True,
            'run_lockout_named': 'A diagnostic is in progress.',
            'recording_active': False,
            'controls_locked': True,
            'motion_enabled': False,
        },
    ),
    (
        _a_home,
        ([live_work.HOME], None, 'A home is in progress.', False),
        {
            'is_protocol_running': False,
            'run_in_progress': False,
            'exclusive_activity': 'home',
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': True,
            'run_lockout_named': 'A home is in progress.',
            'recording_active': False,
            'controls_locked': True,
            'motion_enabled': False,
        },
    ),
    (
        _a_live_recording,
        ([live_work.RECORDING], None, None, True),
        {
            'is_protocol_running': False,
            'run_in_progress': False,
            'exclusive_activity': 'recording',
            'protocol_files_draining': False,
            'protocol_files_pending': 0,
            'run_lockout': False,
            'run_lockout_named': None,
            'recording_active': True,
            'controls_locked': True,
            'motion_enabled': True,
        },
    ),
    (
        _a_finished_runs_files_writing,
        ([live_work.RUN_FILES], 2, "A protocol's files are still being written.", False),
        {
            'is_protocol_running': False,
            'run_in_progress': False,
            'exclusive_activity': None,
            'protocol_files_draining': True,
            'protocol_files_pending': 2,
            'run_lockout': True,
            'run_lockout_named': "A protocol's files are still being written.",
            'recording_active': False,
            'controls_locked': True,
            'motion_enabled': False,
        },
    ),
]


def _records(session):
    work = session.live_work.work
    files = [item.left for item in work if item.kind == live_work.RUN_FILES]
    lock = session.controls_lock
    return (
        [item.kind for item in work],
        (files[0] if files else None),
        lock.run_named,
        lock.recording,
    )


def _answered_by_the_records(session):
    """Each read, answered from the records a client reads."""
    work = session.live_work
    lock = session.controls_lock
    holders = [item.kind for item in work.work if item.kind in live_work.SCOPE_HOLDERS]
    files = [item.left for item in work.work if item.kind == live_work.RUN_FILES]
    return {
        'is_protocol_running': work.has(live_work.PROTOCOL),
        'run_in_progress': work.has(live_work.PROTOCOL),
        'exclusive_activity': holders[0] if holders else None,
        'protocol_files_draining': bool(files),
        'protocol_files_pending': files[0] if files else 0,
        'run_lockout': lock.run_lockout,
        'run_lockout_named': lock.run_named,
        'recording_active': lock.recording,
        'controls_locked': lock.controls_locked,
        'motion_enabled': session.scope.capabilities.has_xy_stage and not lock.run_lockout,
    }


# Where live_work is wider than a read: it names a run from its claim's
# grant to its cleanup's end, where the claim's reads begin and end with
# the claim and the engine's with its phase.
WIDER = {
    (_a_run_starting, 'run_in_progress'): True,
    (_a_run_unwinding, 'is_protocol_running'): True,
    (_a_run_unwinding, 'exclusive_activity'): 'protocol',
}


def test_each_state_reads_the_same_in_the_records(sim_session, monkeypatch):
    wrong = []
    for setup, records, reads in STATES:
        with setup(sim_session, monkeypatch):
            read_records = _records(sim_session)
            read = {name: getattr(sim_session, name) for name in reads}
            answered = _answered_by_the_records(sim_session)
        expected = {name: WIDER.get((setup, name), value) for name, value in reads.items()}
        if read_records != records:
            wrong.append(f'{setup.__name__}: the records read {read_records}, not {records}')
        if read != reads:
            wrong.append(f'{setup.__name__}: the reads read {read}, not {reads}')
        if answered != expected:
            wrong.append(f'{setup.__name__}: the records answer {answered}, not {expected}')
    assert not wrong, '\n'.join(wrong)
