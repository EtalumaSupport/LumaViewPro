# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build does not start while the scope's own work needs the host.

A build runs on the post-processing lane, a thread of the same process, and
shares the cores, the interpreter lock and the disk with whatever holds the
scope -- a run, a recording, a diagnostic, a home -- and with a finished
run's file writes (#812 measured a folder build halving the capture rate).
So `PostProcessingAPI` refuses a build whose turn comes while something
holds the scope (``'scope_held'``) or a run's files are still being written
(``'files_writing'``), for every caller. It is asked as the build starts, on
the lane: a build queued behind another is refused if a hold began while it
waited.

Driven through the Session on the simulated scope.
"""

import threading
import time

import pytest

from modules import live_work
from modules.exceptions import PostProcessingRefusedError, RefusalCause
from modules.sequential_io_executor import IOTask
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_run_refusal_contract import _a_closed_batch_still_writing

QUEUED_WAIT_S = 10


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def test_a_build_asked_while_the_scope_is_held_is_refused_for_its_state(session, tmp_path):
    with session.diagnostic_claim(), pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.stitch(tmp_path)

    assert refused.value.reason == 'scope_held'
    assert refused.value.cause is RefusalCause.STATE
    assert 'diagnostic' in str(refused.value).lower()


def test_a_build_asked_while_a_runs_files_are_written_is_refused(session, tmp_path):
    # A stand-in by design: a closed batch whose write never runs is the
    # simulator's only drain that stays open; the session's close waits on
    # it, so the runner gets its own batch back.
    runner = session.sequenced_capture_runner
    held, runner._write_batch = runner._write_batch, _a_closed_batch_still_writing()
    try:
        assert session.live_work.has(live_work.RUN_FILES)
        with pytest.raises(PostProcessingRefusedError) as refused:
            session.post_processing.zproject(tmp_path)
    finally:
        runner._write_batch = held

    assert refused.value.reason == 'files_writing'


def test_a_build_queued_before_a_hold_is_refused_when_its_turn_comes(session, tmp_path):
    lane = session.post_processing.lane
    release = threading.Event()
    lane.submit(IOTask(action=release.wait), 'test.blocker')
    outcome = {}

    def ask():
        try:
            session.post_processing.stitch(tmp_path)
        except Exception as e:
            outcome['raised'] = e

    caller = threading.Thread(target=ask, name='queued_build')
    caller.start()
    # The lane's queue is the one signal that the build is waiting its turn.
    deadline = time.monotonic() + QUEUED_WAIT_S
    while lane.queue_size() < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert lane.queue_size() == 1, 'the build never queued behind the blocker'

    with session.diagnostic_claim():
        release.set()
        caller.join(QUEUED_WAIT_S)

    assert not caller.is_alive()
    assert isinstance(outcome.get('raised'), PostProcessingRefusedError), outcome
    assert outcome['raised'].reason == 'scope_held'
