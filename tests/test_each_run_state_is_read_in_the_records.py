# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each run state is read in the records, and the reads they replaced are gone.

The Session answers its run state in two records: ``controls_lock`` which of
a person's controls are locked and by what, and ``live_work`` what the
session is doing; a client on the wire reads both in ``status``. Each state
the Session can be in is read in them here: idle, a run starting (its claim
granted, the engine not yet running), a run, a run unwinding after its
release, a diagnostic, a run lent a diagnostic's claim, a home, a live
recording, and a finished run's files still writing. Before the ten one-fact
reads that once answered these retired, each was shown to answer what the
records answer in every one of these states; none is reached on the wire
now, and neither is ``live_work`` but through ``status``.
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
        run = held.lend().try_claim('protocol', run=run_identity())
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
# left, and the lock's run sentence and its recording. live_work names a run
# from its claim's grant to its cleanup's end.
STATES = [
    (_idle, ([], None, None, False)),
    (_a_run_starting, ([live_work.WorkKind.PROTOCOL], None, 'The scan run is in progress.', False)),
    (_a_run, ([live_work.WorkKind.PROTOCOL], None, 'The scan run is in progress.', False)),
    (_a_run_unwinding, ([live_work.WorkKind.PROTOCOL], None, None, False)),
    (_a_diagnostic, ([live_work.WorkKind.DIAGNOSTIC], None, 'A diagnostic is in progress.', False)),
    (
        _a_run_lent_a_diagnostics_claim,
        (
            [live_work.WorkKind.DIAGNOSTIC, live_work.WorkKind.PROTOCOL],
            None,
            'A diagnostic is in progress.',
            False,
        ),
    ),
    (_a_home, ([live_work.WorkKind.HOME], None, 'A home is in progress.', False)),
    (_a_live_recording, ([live_work.WorkKind.RECORDING], None, None, True)),
    (
        _a_finished_runs_files_writing,
        ([live_work.WorkKind.RUN_FILES], 2, "A protocol's files are still being written.", False),
    ),
]

RETIRED_READS = (
    'is_protocol_running',
    'run_in_progress',
    'exclusive_activity',
    'protocol_files_draining',
    'protocol_files_pending',
    'run_lockout',
    'run_lockout_named',
    'recording_active',
    'controls_locked',
    'motion_enabled',
    # In-process only: a wire client reads it in status.
    'live_work',
)


def _records(session):
    work = session.live_work.work
    files = [item.left for item in work if item.kind == live_work.WorkKind.RUN_FILES]
    lock = session.controls_lock
    return (
        [item.kind for item in work],
        (files[0] if files else None),
        lock.run_named,
        lock.recording,
    )


def test_each_state_is_read_in_the_records(sim_session, monkeypatch):
    wrong = []
    for setup, records in STATES:
        with setup(sim_session, monkeypatch):
            read = _records(sim_session)
        if read != records:
            wrong.append(f'{setup.__name__}: the records read {read}, not {records}')
    assert not wrong, '\n'.join(wrong)


def test_no_retired_read_is_reached_on_the_wire(sim_session):
    from fastapi.testclient import TestClient

    from rest.app import build_app

    with TestClient(build_app(sim_session)) as client:
        assert client.get('/api/v1/status').status_code == 200
        reached = {
            name: answer.status_code
            for name in RETIRED_READS
            if (answer := client.get(f'/api/v1/{name}')).status_code != 404
        }
    assert not reached, f'a retired read still answers on the wire: {reached}'
