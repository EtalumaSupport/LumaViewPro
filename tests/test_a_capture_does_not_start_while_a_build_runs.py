# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run or a recording does not start while a build is on the post-processing lane.

A capture and a build never overlap (W4 E3, Eric 2026-10-21: "No overlap"):
the other way round, a build does not start while the scope is held
(``test_a_build_does_not_start_while_the_scope_is_busy``). A run's own
hyperstack and merge are lane work too, so the next run waits for its
predecessor's builds. Refused as ``'post_processing_running'``, a state the
caller waits out.

Driven through the Session on the simulated scope.
"""

import threading

import pytest

from modules.exceptions import RecordingRefusedError, RefusalCause
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


@pytest.fixture
def a_build_on_the_lane(session):
    """The lane running a build until the test lets it end."""
    release = threading.Event()
    session.post_processing.lane.submit(IOTask(action=release.wait), 'test.build')
    yield
    release.set()


def test_a_run_asked_while_a_build_runs_is_refused_for_its_state(session, a_build_on_the_lane):
    # The status a client polls before it asks; prepare() raises the same
    # refusal, from the one read (SequencedCaptureRunner.state_refusal).
    refused = session.status.run_refused

    assert refused is not None
    assert refused.reason == 'post_processing_running'
    assert refused.cause is RefusalCause.STATE


def test_a_recording_asked_while_a_build_runs_is_refused(session, a_build_on_the_lane):
    with pytest.raises(RecordingRefusedError) as refused:
        session.manual_recording.start()

    assert refused.value.reason == 'post_processing_running'


def test_a_build_is_over_for_a_capture_once_its_answer_is_delivered(session, tmp_path):
    # A run's files and a composite's outcome are told from the build's
    # answer, before the lane lets the build go: a run started from them is
    # not refused for the build that answered them.
    answered = threading.Event()
    lane_busy_when_answered = []

    def heard(_settled):
        lane_busy_when_answered.append(session.post_processing.has_work())
        answered.set()

    build = session.post_processing.run_hyperstack(
        tmp_path,
        save_encoding='8bit',
        channel_colors=session.channel_colors(),
        written=lambda: None,
    )
    build.add_done_callback(heard)

    assert answered.wait(10.0), 'the build was never answered'
    assert lane_busy_when_answered == [False]
