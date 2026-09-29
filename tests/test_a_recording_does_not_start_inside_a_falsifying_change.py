# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording does not start inside a change that would falsify it.

A lane refuses a turret, frame, binning or pixel-format write submitted or
dequeued while a recording holds -- but a recording writes nothing to a
lane, so it could begin while such a write was already running. On the
bench a recording started between a turret move and its Z restore: the
restore ran inside the recording and its frames claimed no objective. The
GUI's press runs the marked turret move inline inside an unmarked task, so
the lane's running task never named it; the claim counts the falsifying
writes where they run. The recording reads what its file will claim only
once it holds the claim, refuses an unknown objective as a capture does,
and the objective question a hold would refuse the answer to is not asked
until the hold ends.
"""

import ast
import logging
import threading

import pytest

from modules import settings_init
from modules.activity_claim import ActivityClaim
from modules.exceptions import ObjectiveUnknownError, RecordingRefusedError
from modules.recording_frames import resolve_recording_pixel_size
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask, SequentialIOExecutor
from modules.video_recording import RecordingConfig, VideoRecordingEngine
from tests.ast_seams import find_def
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_WAIT_S = 2.0


@pytest.fixture
def claim():
    return ActivityClaim()


def _lane(claim, name):
    ex = SequentialIOExecutor(name=name)
    ex.ask_claim(claim)
    ex.start()
    return ex


@pytest.fixture
def io_lane(claim):
    ex = _lane(claim, 'TEST_IO')
    yield ex
    ex.shutdown()


@pytest.fixture
def camera_lane(claim):
    ex = _lane(claim, 'TEST_CAMERA')
    yield ex
    ex.shutdown()


def _config(tmp_path):
    return RecordingConfig(
        fps=5.0,
        duration_s=1.0,
        width=8,
        height=6,
        bit_depth=8,
        output_dir=tmp_path,
        filename_template='frame_{index}',
        timestamp_overlay=False,
    )


def _engine(claim):
    return VideoRecordingEngine(
        write_frame=lambda *a, **k: None, claim=claim, clock=lambda: 0.0, notify=None
    )


class _Change:
    """A write that runs until released, so a recording can start inside it."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self):
        self.started.set()
        self.release.wait(_WAIT_S)


def _run_top_level(lane, *, falsifies_recording):
    change = _Change()
    fut = lane.put(
        IOTask(action=change, falsifies_recording=falsifies_recording), return_future=True
    )
    assert change.started.wait(_WAIT_S), 'the change must be running before the recording starts'
    return fut, change


def _run_inline_in_an_unmarked_task(lane):
    """The GUI's shape: an unmarked task runs the marked member inline on the lane."""
    change = _Change()

    def _gesture():
        return lane.call(IOTask(action=change, falsifies_recording=True), 'move_turret', _WAIT_S)

    fut = lane.put(IOTask(action=_gesture), return_future=True)
    assert change.started.wait(_WAIT_S), 'the change must be running before the recording starts'
    return fut, change


class TestTheRecordingStart:
    def test_a_start_inside_a_gui_turret_move_is_refused(self, claim, io_lane, tmp_path):
        fut, change = _run_inline_in_an_unmarked_task(io_lane)
        try:
            with pytest.raises(RecordingRefusedError) as refused:
                _engine(claim).start(lambda: _config(tmp_path))
            assert refused.value.reason == 'falsifying_change_in_flight'
            assert claim.holder is None, 'a refused start leaves the claim free'
        finally:
            change.release.set()
            fut.result(timeout=_WAIT_S)

    @pytest.mark.parametrize('lane_fixture', ['io_lane', 'camera_lane'])
    def test_a_start_inside_a_falsifying_task_is_refused(
        self, request, claim, tmp_path, lane_fixture
    ):
        lane = request.getfixturevalue(lane_fixture)
        fut, change = _run_top_level(lane, falsifies_recording=True)
        try:
            with pytest.raises(RecordingRefusedError) as refused:
                _engine(claim).start(lambda: _config(tmp_path))
            assert refused.value.reason == 'falsifying_change_in_flight'
            assert claim.holder is None
        finally:
            change.release.set()
            fut.result(timeout=_WAIT_S)

    def test_the_change_it_was_refused_for_runs_to_its_end(self, claim, io_lane, tmp_path):
        fut, change = _run_inline_in_an_unmarked_task(io_lane)
        with pytest.raises(RecordingRefusedError):
            _engine(claim).start(lambda: _config(tmp_path))
        change.release.set()
        assert fut.result(timeout=_WAIT_S) is None

    def test_a_start_after_the_change_ends_records(self, claim, io_lane, tmp_path):
        fut, change = _run_inline_in_an_unmarked_task(io_lane)
        change.release.set()
        fut.result(timeout=_WAIT_S)
        engine = _engine(claim)
        engine.start(lambda: _config(tmp_path))
        try:
            assert claim.owner == 'recording'
        finally:
            engine.stop('user_stop')

    def test_a_write_that_does_not_falsify_it_does_not_refuse_it(self, claim, io_lane, tmp_path):
        fut, change = _run_top_level(io_lane, falsifies_recording=False)
        engine = _engine(claim)
        try:
            engine.start(lambda: _config(tmp_path))
            assert claim.owner == 'recording'
            engine.stop('user_stop')
        finally:
            change.release.set()
            fut.result(timeout=_WAIT_S)

    def test_what_the_file_claims_is_read_under_the_claim(self, claim, tmp_path):
        seen = []

        def _build():
            seen.append(claim.owner)
            return _config(tmp_path)

        engine = _engine(claim)
        engine.start(_build)
        engine.stop('user_stop')
        assert seen == ['recording']


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    s = ScopeSession.create(
        complete_settings(microscope='LS850', objective_confirmed=False, objective_id='10x Oly'),
        simulate=True,
    )
    try:
        yield s
    finally:
        try:
            s.shutdown()
        except Exception:
            logging.getLogger(__name__).debug(
                'teardown noise is not the measurement', exc_info=True
            )


@pytest.fixture
def turret_session(monkeypatch):
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    s = ScopeSession.create(
        complete_settings(
            microscope='LS850T',
            objective_confirmed=True,
            turret_position=1,
            turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        try:
            s.shutdown()
        except Exception:
            logging.getLogger(__name__).debug(
                'teardown noise is not the measurement', exc_info=True
            )


class TestAnUnknownObjective:
    def test_it_refuses_a_recording_as_it_refuses_a_capture(self, turret_session):
        turret_session.scope.motion.move_turret(2)
        with pytest.raises(ObjectiveUnknownError) as unknown:
            resolve_recording_pixel_size(turret_session.scope)
        assert unknown.value.reason == 'slot_unassigned'

    def test_a_known_objective_gives_the_scale(self, turret_session):
        turret_session.scope.motion.move_turret(1)
        assert resolve_recording_pixel_size(turret_session.scope) > 0


class TestTheObjectiveQuestionDuringAHold:
    @pytest.mark.parametrize('kind', ['recording', 'protocol', 'diagnostic'])
    def test_it_is_not_asked_while_a_hold_would_refuse_the_answer(self, session, kind):
        assert session.objective_question() is not None, 'owed: never confirmed on this install'
        held = session.activity_claim.try_claim(kind, run_trigger_source='test')
        try:
            assert session.objective_question() is None
        finally:
            held.release()
        assert session.objective_question() is not None

    def test_the_gui_redraws_the_turret_on_every_run_state_edge(self):
        publish = find_def('lumaviewpro.py', 'publish_run_state', class_name='LumaViewProApp')
        called = [
            sub.func.attr
            for sub in ast.walk(publish)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        ]
        assert 'show_turret_state' in called, (
            'a question withheld during a hold is asked again only when the turret is redrawn'
        )

    def test_the_gui_leaves_the_decision_to_the_session(self):
        show = find_def('ui/vertical_control.py', 'show_turret_state', class_name='VerticalControl')
        source = ast.unparse(show)
        assert 'is_protocol_running' not in source, (
            "whether the question is asked during a hold is the Session's answer"
        )


class TestTheManualRecordingReadsItsClaimsUnderTheClaim:
    def test_frame_size_and_objective_are_read_while_the_recording_holds(
        self, tmp_path, monkeypatch
    ):
        from modules.manual_recording import ManualRecordingController
        from tests.test_manual_recording_controller import (
            FakeClock,
            ManualFireScheduler,
            _FakeImaging,
            _FakeScope,
            make_settings,
        )

        claim = ActivityClaim()
        read_under = {}
        frame_size = _FakeImaging.frame_size_cached.fget

        def _frame_size_read(imaging):
            read_under['frame_size'] = claim.owner
            return frame_size(imaging)

        monkeypatch.setattr(_FakeImaging, 'frame_size_cached', property(_frame_size_read))
        scope = _FakeScope()
        resolve = scope.runtime_state.resolve_current_objective

        def _objective_read():
            read_under['objective'] = claim.owner
            return resolve()

        scope.runtime_state.resolve_current_objective = _objective_read
        controller = ManualRecordingController(
            scope=scope,
            settings=make_settings(tmp_path),
            activity_claim=claim,
            scheduler=ManualFireScheduler(),
            clock=FakeClock(),
        )
        controller.start()
        controller.stop()
        assert read_under == {'frame_size': 'recording', 'objective': 'recording'}
