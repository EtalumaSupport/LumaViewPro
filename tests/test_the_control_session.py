# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One client controls the scope at a time, above the one activity slot.

A REST client and the GUI were peers with nothing between them: either
could write the hardware, the settings or the files while the other worked.
The activity claim now records a control session above its activity slot.
While one is held, work not made under the controller's taking -- or under
an activity taken for the session -- is refused at every door that asks the
claim; the activity slot keeps meaning what is running, for the holder too.
A take never lands over the operator's work: not over an activity, not over
a guarded write still in flight.
"""

import threading

import pytest

from modules.activity_claim import (
    ActivityClaim,
    ControlSessionHeldError,
    acting,
    guarded_write,
)
from modules.caller import Caller, calling
from modules.exceptions import (
    ControlSessionRefusedError,
    DiagnosticRefusedError,
    HardwareCommandRefusedError,
    RecordingRefusedError,
    SessionClosingError,
)
from modules.sequential_io_executor import IOTask, SequentialIOExecutor
from modules.wire_encoding import outcome_fields
from tests.test_a_run_says_when_its_files_are_written import held_saves  # noqa: F401 -- a fixture

_WAIT_S = 2.0
# Long enough that held work is still held when the test acts on it.
_HOLD_S = 60.0


@pytest.fixture
def claim():
    return ActivityClaim()


@pytest.fixture
def lane(claim):
    """A started lane that asks ``claim``."""
    ex = SequentialIOExecutor(name='TEST_IO')
    ex.ask_claim(claim)
    ex.start()
    yield ex
    ex.shutdown()


def _take(claim, name='plate reader', caller=Caller.REST):
    """Take ``claim``'s control session; a claim on its own has no work finishing."""
    with calling(caller):
        return claim.take_control(name, finishing=lambda: False)


def _take_on(session, name='plate reader', caller=Caller.REST):
    """Take ``session``'s control session as ``caller``; the taking its holder acts under."""
    with calling(caller):
        session_id = session.take_control(name)
    return session.activity_claim.controller(session_id)


def _run_on_lane(lane, taking):
    """Submit a task from a fresh thread acting under ``taking``; its future and whether it ran."""
    ran = threading.Event()
    box = {}

    def _submit():
        with acting(taking):
            box['fut'] = lane.put(
                IOTask(action=ran.set, silent_on_failure=True), return_future=True
            )

    t = threading.Thread(target=_submit)
    t.start()
    t.join(_WAIT_S)
    return box['fut'], ran


class TestTheTake:
    def test_a_take_records_the_session_with_its_name_and_its_taker(self, claim):
        controller = _take(claim, 'plate reader', Caller.REST)
        session = claim.control
        assert session is not None
        assert session.name == 'plate reader'
        assert session.taken_by == Caller.REST
        assert controller.session is session

    def test_a_take_and_an_end_are_announced(self):
        heard = []
        claim = ActivityClaim(on_transition=lambda: heard.append(claim.control))
        controller = _take(claim)
        claim.end_control(controller.id)
        assert heard == [controller.session, None]

    def test_a_second_take_is_refused_naming_the_session_that_holds(self, claim):
        first = _take(claim, 'plate reader')
        with pytest.raises(ControlSessionRefusedError) as refused:
            _take(claim, 'second client')
        assert refused.value.reason == 'control_session_held'
        assert refused.value.control_session == 'plate reader'
        assert claim.control is first.session

    def test_a_take_is_refused_while_an_activity_holds_the_slot(self, claim):
        held = claim.try_claim('diagnostic')
        try:
            with pytest.raises(ControlSessionRefusedError) as refused:
                _take(claim)
            assert refused.value.reason == 'exclusive_activity_running'
            assert refused.value.holder == 'diagnostic'
            assert claim.control is None
            assert held.holds, "a refused take leaves the operator's activity untouched"
        finally:
            held.release()

    def test_a_take_is_refused_while_a_guarded_write_is_in_flight(self, claim):
        claim.enter_guarded_write(None)
        try:
            with pytest.raises(ControlSessionRefusedError) as refused:
                _take(claim)
            assert refused.value.reason == 'write_in_flight'
        finally:
            claim.leave_guarded_write()
        assert _take(claim).session is claim.control, 'the take is admitted once the write returns'

    def test_a_take_is_refused_while_a_run_or_recording_that_ended_is_finishing(self, claim):
        with pytest.raises(ControlSessionRefusedError) as refused:
            claim.take_control('plate reader', finishing=lambda: True)
        assert refused.value.reason == 'work_finishing'
        assert claim.control is None

    def test_a_take_is_refused_while_the_session_closes(self, claim):
        claim.begin_closing()
        with pytest.raises(SessionClosingError):
            _take(claim)
        assert claim.control is None


class TestTheEnd:
    def test_an_end_naming_another_id_is_refused_and_the_session_stays(self, claim):
        controller = _take(claim)
        with pytest.raises(ControlSessionRefusedError) as refused:
            claim.end_control('not-the-id')
        assert refused.value.reason == 'control_session_ended'
        assert claim.control is controller.session

    def test_after_the_end_the_old_holders_lane_work_is_anyones(self, claim, lane):
        controller = _take(claim)
        claim.end_control(controller.id)
        fut, ran = _run_on_lane(lane, controller)
        fut.result(timeout=_WAIT_S)
        assert ran.is_set(), (
            'an ended control session is refused at the REST door only; in-process the old '
            "holder's work is a caller's like any other"
        )


class TestTheLanes:
    def test_a_non_holders_task_is_refused_naming_the_session(self, claim, lane):
        _take(claim, 'plate reader', Caller.REST)
        fut, ran = _run_on_lane(lane, None)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            fut.result(timeout=_WAIT_S)
        assert refused.value.reason == 'control_session_held'
        assert refused.value.control_session == 'plate reader'
        assert refused.value.holder_trigger == Caller.REST
        assert refused.value.holder is None, 'the session is not an activity'
        assert not ran.wait(0.3)

    def test_the_holders_task_runs(self, claim, lane):
        controller = _take(claim)
        fut, ran = _run_on_lane(lane, controller)
        fut.result(timeout=_WAIT_S)
        assert ran.is_set()

    def test_the_holders_plain_task_is_refused_while_its_own_activity_holds(self, claim, lane):
        controller = _take(claim)
        with acting(controller):
            held = claim.try_claim('diagnostic')
        try:
            fut, ran = _run_on_lane(lane, controller)
            with pytest.raises(HardwareCommandRefusedError) as refused:
                fut.result(timeout=_WAIT_S)
            assert refused.value.reason == 'exclusive_activity_running'
            assert refused.value.holder == 'diagnostic'
            fut, ran = _run_on_lane(lane, held)
            fut.result(timeout=_WAIT_S)
            assert ran.is_set(), 'the activity taken for the session runs its own work'
        finally:
            held.release()

    def test_a_refusal_publishes_who_holds_on_the_wire(self, claim, lane):
        _take(claim, 'plate reader', Caller.REST)
        fut, _ = _run_on_lane(lane, None)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            fut.result(timeout=_WAIT_S)
        fields = outcome_fields(refused.value, lambda value: value)
        assert fields['reason'] == 'control_session_held'
        assert fields['control_session'] == 'plate reader'
        assert fields['holder_trigger'] == 'rest'
        assert fields['holder'] is None


class TestTheActivitySlot:
    def test_a_non_holders_take_of_the_slot_is_refused_and_the_slot_stays_free(self, claim):
        _take(claim)
        with pytest.raises(ControlSessionHeldError):
            claim.try_claim('home')
        assert claim.holder is None

    def test_the_holders_take_of_the_slot_is_an_ordinary_activity(self, claim):
        controller = _take(claim)
        with acting(controller):
            held = claim.try_claim('home')
            assert held is not None
            assert claim.try_claim('diagnostic') is None, (
                'one activity at a time, for the holder too'
            )
        held.release()

    def test_work_lent_the_holders_activity_was_taken_for_the_session(self, claim):
        controller = _take(claim)
        with acting(controller):
            held = claim.try_claim('diagnostic')
        try:
            lent = held.lend().try_claim('recording')
            assert lent.control is controller.session, (
                "work lent the activity is the session's, and the take back stops it"
            )
        finally:
            held.release()


@pytest.fixture
def sim_session(tmp_path):
    """A simulated, homed scope."""
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


class TestEachTakerNamesTheSession:
    """Each taker of the slot answers a held session with its own typed refusal."""

    @pytest.fixture
    def held(self, sim_session):
        controller = _take_on(sim_session)
        yield controller
        sim_session.activity_claim.end_control(controller.id)

    def test_a_diagnostic(self, sim_session, held):
        with pytest.raises(DiagnosticRefusedError) as refused, sim_session.diagnostic_claim():
            pass
        assert refused.value.reason == 'control_session_held'
        assert refused.value.control_session == 'plate reader'
        assert refused.value.holder_trigger == Caller.REST

    def test_a_home(self, sim_session, held):
        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_session.scope.motion.start_home('Z')
        assert refused.value.reason == 'control_session_held'
        assert refused.value.control_session == 'plate reader'
        assert sim_session.activity_claim.holder is None

    def test_the_holders_home_takes_the_scope(self, sim_session, held):
        with acting(held):
            sim_session.scope.motion.start_home('Z').result(timeout=10)

    def test_a_recording(self, sim_session, held):
        with pytest.raises(RecordingRefusedError) as refused:
            sim_session.manual_recording.start()
        assert refused.value.reason == 'control_session_held'
        assert refused.value.control_session == 'plate reader'
        assert sim_session.activity_claim.holder is None

    def test_a_change_of_the_shown_layer(self, sim_session, held):
        other = 'Red' if sim_session.shown_channel.layer == 'Green' else 'Green'
        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_session.show_channel(other)
        assert refused.value.reason == 'control_session_held'
        assert sim_session.shown_channel.layer != other


class TestGuardedWrites:
    def test_a_non_holders_write_is_refused_and_not_counted(self, claim):
        controller = _take(claim)
        with pytest.raises(ControlSessionRefusedError) as refused:
            claim.enter_guarded_write(None)
        assert refused.value.reason == 'control_session_held'
        claim.end_control(controller.id)
        assert _take(claim) is not None, 'a refused write left nothing counted'

    def test_the_holders_write_is_admitted(self, claim):
        controller = _take(claim)
        claim.enter_guarded_write(controller)
        claim.leave_guarded_write()

    def test_a_take_cannot_land_while_a_guarded_member_is_mid_write(self, claim):
        # a stand-in by design: the subject is the guarded-write decorator,
        # and this owner is only what it reads the claim from.
        class _Owner:
            activity_claim = claim
            writing = threading.Event()
            finish = threading.Event()

            @guarded_write(lambda owner: owner.activity_claim)
            def write(self):
                self.writing.set()
                self.finish.wait(_WAIT_S)

        owner = _Owner()
        writer = threading.Thread(target=owner.write)
        writer.start()
        try:
            assert owner.writing.wait(_WAIT_S)
            with pytest.raises(ControlSessionRefusedError) as refused:
                _take(claim)
            assert refused.value.reason == 'write_in_flight'
        finally:
            owner.finish.set()
            writer.join(_WAIT_S)
        assert _take(claim) is not None, 'the write returned, and the take lands'


class TestTheSessionsWrites:
    @pytest.fixture
    def held(self, sim_session):
        controller = _take_on(sim_session)
        yield controller
        if sim_session.activity_claim.control is controller.session:
            sim_session.activity_claim.end_control(controller.id)

    def test_a_non_holders_setting_is_refused_and_nothing_is_stored(self, sim_session, held):
        before = sim_session.settings['jpg_quality']
        with pytest.raises(ControlSessionRefusedError) as refused:
            sim_session.update_settings('jpg_quality', 50 if before != 50 else 60)
        assert refused.value.reason == 'control_session_held'
        assert refused.value.control_session == 'plate reader'
        assert sim_session.settings['jpg_quality'] == before

    def test_the_holders_setting_is_stored(self, sim_session, held):
        wanted = 50 if sim_session.settings['jpg_quality'] != 50 else 60
        with acting(held):
            sim_session.update_settings('jpg_quality', wanted)
        assert sim_session.settings['jpg_quality'] == wanted

    def test_a_display_preference_is_written_by_anyone(self, sim_session, held):
        wanted = not sim_session.settings['show_tooltips']
        sim_session.update_settings('show_tooltips', wanted)
        assert sim_session.settings['show_tooltips'] == wanted

    def test_a_member_that_skips_its_admission_is_refused_where_it_stores(self, sim_session, held):
        before = sim_session.settings['jpg_quality']
        with pytest.raises(ControlSessionRefusedError), sim_session.settings_lock:
            sim_session._store_setting('jpg_quality', 50 if before != 50 else 60)
        assert sim_session.settings['jpg_quality'] == before

    def test_a_non_holders_protocol_save_writes_no_file(self, sim_session, held, tmp_path):
        protocol = sim_session.create_empty_protocol()
        target = tmp_path / 'saved.tsv'
        with pytest.raises(ControlSessionRefusedError):
            sim_session.save_protocol(protocol, target)
        assert not target.exists()

    def test_a_non_holders_camera_write_changes_nothing(self, sim_session, held):
        before = sim_session.scope.camera.frame_size
        with pytest.raises(ControlSessionRefusedError):
            sim_session.scope.camera.set_frame_size(before.width // 2, before.height // 2)
        assert sim_session.scope.camera.frame_size == before

    def test_a_non_holder_cannot_give_up_a_recordings_pending_frames(self, sim_session, held):
        with pytest.raises(ControlSessionRefusedError):
            sim_session.discard_close_drain()


class TestTheSessionsMembers:
    """A client takes, acts in and ends the control session through the Session."""

    def test_the_status_names_the_session_and_refuses_no_run_while_nothing_runs(self, sim_session):
        _take_on(sim_session)
        status = sim_session.status
        assert (status.control.name, status.control.taken_by) == ('plate reader', Caller.REST)
        assert status.run_refused is None, "run_refused is the activity's answer alone"

    def test_the_holder_acts_in_the_session_by_its_id(self, sim_session):
        with calling(Caller.REST):
            session_id = sim_session.take_control('plate reader')
        wanted = 50 if sim_session.settings['jpg_quality'] != 50 else 60
        with pytest.raises(ControlSessionRefusedError):
            sim_session.update_settings('jpg_quality', wanted)
        with sim_session.acting_in_control(session_id):
            sim_session.update_settings('jpg_quality', wanted)
        assert sim_session.settings['jpg_quality'] == wanted

    def test_acting_in_an_ended_session_is_refused_before_the_block_runs(self, sim_session):
        session_id = sim_session.take_control('plate reader')
        sim_session.end_control(session_id)
        ran = []
        with (
            pytest.raises(ControlSessionRefusedError) as refused,
            sim_session.acting_in_control(session_id),
        ):
            ran.append(True)
        assert refused.value.reason == 'control_session_ended'
        assert ran == []

    def test_taking_back_control_stops_the_clients_recording_and_clears_the_session(
        self, sim_session
    ):
        with calling(Caller.REST):
            session_id = sim_session.take_control('plate reader')
        with sim_session.acting_in_control(session_id):
            sim_session.manual_recording.start()
        assert sim_session.manual_recording.is_recording

        with calling(Caller.GUI):
            sim_session.end_control(session_id)

        assert sim_session.control is None
        assert not sim_session.manual_recording.is_recording
        assert sim_session.manual_recording.end_reason == 'user_stop'

    def test_taking_back_an_ended_session_leaves_the_operators_recording_begun_since(
        self, sim_session
    ):
        session_id = sim_session.take_control('plate reader')
        ended = sim_session.end_control(session_id)
        sim_session.manual_recording.start()
        try:
            sim_session.manual_recording.stop_the_recording_taken_for(ended)
            assert sim_session.manual_recording.is_recording, (
                'the take back stops the recording taken for the session, never one begun after it'
            )
        finally:
            sim_session.manual_recording.stop()

    def test_the_gui_takes_back_control_without_naming_it_and_its_recording_stops(
        self, sim_session
    ):
        with calling(Caller.REST):
            session_id = sim_session.take_control('plate reader')
        with sim_session.acting_in_control(session_id):
            sim_session.manual_recording.start()

        with calling(Caller.GUI):
            ended = sim_session.take_back_control()

        assert ended.name == 'plate reader'
        assert sim_session.control is None
        assert not sim_session.manual_recording.is_recording
        assert sim_session.take_back_control() is None, 'with none held there is nothing to end'

    def test_the_published_record_carries_no_id_in_process_either(self, sim_session):
        import dataclasses

        sim_session.take_control('plate reader')
        fields = {f.name for f in dataclasses.fields(sim_session.control)}
        assert fields == {'name', 'taken_by', 'taken_at'}, (
            'only the taker holds the id: a plugin reading session.control could act as the holder'
        )

    def test_the_objective_question_is_put_to_the_holder_and_not_to_anyone_else(self, sim_session):
        # Never confirmed for this light path, so the question is owed.
        sim_session.settings['objective_confirmed'] = None
        assert sim_session.objective_question() is not None
        with calling(Caller.REST):
            session_id = sim_session.take_control('plate reader')

        assert sim_session.objective_question() is None, 'its answer would be refused'
        with sim_session.acting_in_control(session_id):
            assert sim_session.objective_question() is not None

    def test_the_controls_lock_names_the_session_and_nothing_runs(self, sim_session):
        _take_on(sim_session)
        lock = sim_session.controls_lock
        assert lock.run_lockout and lock.controls_locked
        assert lock.run_named is None, 'the display readers keep showing the scope'
        assert lock.named == 'plate reader has remote control.'

    def test_the_close_ends_a_held_session_and_closes(self, sim_session):
        _take_on(sim_session)
        sim_session.shutdown()
        assert sim_session.control is None
        assert sim_session.live_work.closed


class TestTheTakeBackPress:
    """The GUI's Take back control press takes control back from whatever session is held."""

    @pytest.fixture
    def press(self, sim_session, monkeypatch):
        """The press, its call run here, and every outcome it reports."""
        from types import SimpleNamespace

        import modules.app_context as app_context
        import ui.ui_helpers as ui_helpers
        from modules.notification_center import notifications
        from tests.gesture_fakes import inline_submit_reported

        reported = []
        # a stand-in by design: Kivy is stubbed in the test process, and the
        # subject is the press's call, not the lane it runs on.
        monkeypatch.setattr(app_context, 'ctx', SimpleNamespace(session=sim_session))
        monkeypatch.setattr(ui_helpers, 'submit_reported', inline_submit_reported)
        monkeypatch.setattr(
            notifications, 'report_outcome', lambda outcome, **kw: reported.append(outcome)
        )
        return ui_helpers.take_back_control, reported

    def test_a_press_ends_the_held_session_and_stops_its_recording(self, sim_session, press):
        take_back, reported = press
        with calling(Caller.REST):
            session_id = sim_session.take_control('plate reader')
        with sim_session.acting_in_control(session_id):
            sim_session.manual_recording.start()

        with calling(Caller.GUI):
            take_back()

        assert sim_session.control is None
        assert not sim_session.manual_recording.is_recording
        assert reported == []

    def test_a_press_after_the_session_ended_ends_nothing_and_says_nothing(
        self, sim_session, press
    ):
        take_back, reported = press
        sim_session.end_control(sim_session.take_control('plate reader'))

        with calling(Caller.GUI):
            take_back()

        assert sim_session.control is None
        assert reported == [], (
            "control is already back: the remote client's 'session has ended' is not the operator's"
        )

    def test_a_press_over_a_session_taken_since_the_banner_drew_ends_that_one(
        self, sim_session, press
    ):
        take_back, reported = press
        sim_session.end_control(sim_session.take_control('plate reader'))
        sim_session.take_control('second client')

        with calling(Caller.GUI):
            take_back()

        assert sim_session.control is None, 'the press asks for control back, whoever holds it'
        assert reported == []


@pytest.fixture
def run_session(tmp_path):
    """A simulated, homed scope with the stage and objective a held run names."""
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def test_taking_back_control_stops_the_clients_run_and_the_session_is_cleared_at_once(
    run_session, tmp_path
):
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
    from tests.test_a_run_records_who_started_it import _start_held

    with calling(Caller.REST):
        session_id = run_session.take_control('plate reader')
    with run_session.acting_in_control(session_id):
        run = _start_held(run_session, tmp_path)

    with calling(Caller.GUI):
        run_session.end_control(session_id)
    assert run_session.control is None, 'the session ends before its run has unwound'

    outcome = run.result(timeout=COMPLETION_TIMEOUT)
    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')
    assert outcome.message == 'Stopped by LumaViewPro.'


def test_taking_back_an_ended_session_leaves_the_operators_run_begun_since(run_session, tmp_path):
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
    from tests.test_a_run_records_who_started_it import _start_held

    session_id = run_session.take_control('plate reader')
    ended = run_session.end_control(session_id)
    with calling(Caller.GUI):
        run = _start_held(run_session, tmp_path)

    run_session.sequenced_capture_runner.stop_the_run_taken_for(ended)
    # The run stopped would end at its held move; let alone, it goes on to it.
    run_session.scope.simulation.clear()

    outcome = run.result(timeout=COMPLETION_TIMEOUT)
    assert outcome.status == 'completed', (
        'the take back stops the run taken for the session, never one begun after it'
    )


class TestATakeWaitsForWhatIsStillFinishing:
    """A take is refused ``work_finishing`` while work that let the scope go still finishes.

    Each test holds one kind of that work with the slot already free, so the
    refusal is the finishing work's and not the activity's, and the take
    lands once the work is done.
    """

    def test_a_finished_runs_images(self, tmp_path, held_saves):
        from tests.test_a_run_says_when_its_files_are_written import WAIT_S, _scan
        from tests.test_composite_run_e2e import headless_settings, open_composite_session

        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            run = _scan(runner, tmp_path / 'runs', 'C1')
            assert run.result(timeout=WAIT_S) is not None
            assert session.activity_claim.holder is None
            assert not session.sequenced_capture_runner.run_live

            with pytest.raises(ControlSessionRefusedError) as refused:
                session.take_control('plate reader')
            assert refused.value.reason == 'work_finishing'

            held_saves.set()
            assert run.files.result(timeout=WAIT_S) is not None
            assert session.take_control('plate reader')

    def test_a_finished_runs_post_run_step(self, tmp_path, monkeypatch):
        import modules.image_mode as image_mode
        import modules.stack_builder as stack_builder
        from tests.test_a_run_says_when_its_files_are_written import WAIT_S, _scan
        from tests.test_composite_run_e2e import headless_settings, open_composite_session

        release = threading.Event()
        real_build = stack_builder.build_hyperstacks_for_run

        def _held_build(build):
            release.wait(WAIT_S)
            real_build(build)

        monkeypatch.setattr(stack_builder, 'build_hyperstacks_for_run', _held_build)
        try:
            with open_composite_session(headless_settings(tmp_path)) as (session, runner):
                run = _scan(
                    runner,
                    tmp_path / 'runs',
                    'C1',
                    sequenced_format=image_mode.OUTPUT_FORMAT_HYPERSTACK,
                )
                assert run.result(timeout=WAIT_S) is not None
                assert session.sequenced_capture_runner.write_batch().wait_complete(WAIT_S)
                assert session.activity_claim.holder is None

                with pytest.raises(ControlSessionRefusedError) as refused:
                    session.take_control('plate reader')
                assert refused.value.reason == 'work_finishing'

                release.set()
                assert run.files.result(timeout=WAIT_S) is not None
                assert session.take_control('plate reader')
        finally:
            release.set()

    def test_a_stopped_recordings_file(self, sim_session, monkeypatch):
        recording = sim_session.manual_recording
        closing, release = threading.Event(), threading.Event()
        done = recording.start()
        writer = recording._writer
        assert writer is not None, 'an MP4 recording, whose file is closed after the drain'
        real_close = writer.close

        def _held_close():
            closing.set()
            release.wait(_HOLD_S)
            real_close()

        # a stand-in by design: the simulator cannot yet hold a recording's
        # finish open, so its real file close -- after the drain, the claim
        # let go -- is held here.
        monkeypatch.setattr(writer, 'close', _held_close)
        try:
            recording.stop()
            assert closing.wait(_HOLD_S), 'the recording never reached its file close'
            assert sim_session.activity_claim.holder is None

            with pytest.raises(ControlSessionRefusedError) as refused:
                sim_session.take_control('plate reader')
            assert refused.value.reason == 'work_finishing'
        finally:
            release.set()
        done.result(timeout=_HOLD_S)
        assert sim_session.take_control('plate reader')

    def test_a_finished_runs_video(self, sim_session, monkeypatch):
        from unittest.mock import MagicMock

        # a stand-in by design: the simulator cannot yet hold a run's video
        # drain open past the run (the live-work tests' gap, the same one).
        writer = MagicMock(video_busy=True, video_pending_writes=4)
        monkeypatch.setattr(sim_session.sequenced_capture_runner, '_image_writer', writer)

        with pytest.raises(ControlSessionRefusedError) as refused:
            sim_session.take_control('plate reader')
        assert refused.value.reason == 'work_finishing'

        writer.video_busy = False
        assert sim_session.take_control('plate reader')


class TestTheRestDoor:
    """A request carrying the control id in its header acts in that session."""

    @pytest.fixture
    def client(self, sim_session):
        from fastapi.testclient import TestClient

        from rest.app import build_app

        with TestClient(build_app(sim_session)) as client:
            yield client

    def test_a_request_naming_the_session_writes_and_one_without_it_is_refused(
        self, sim_session, client
    ):
        taken = client.post('/api/v1/take_control', json={'name': 'plate reader'})
        assert taken.status_code == 200, taken.text
        session_id = taken.json()
        assert sim_session.control.taken_by == Caller.REST

        refused = client.put('/api/v1/acceleration_limit_pct', json={'val_pct': 40})
        assert refused.status_code == 409
        assert refused.json()['reason'] == 'control_session_held'
        assert refused.json()['control_session'] == 'plate reader'

        written = client.put(
            '/api/v1/acceleration_limit_pct',
            json={'val_pct': 40},
            headers={'Control-Session': session_id},
        )
        assert written.status_code == 200, written.text
        assert sim_session.acceleration_limit_pct == 40

    def test_the_status_names_the_session_and_never_its_id(self, client):
        session_id = client.post('/api/v1/take_control', json={'name': 'plate reader'}).json()
        status = client.get('/api/v1/status')
        assert status.json()['control']['name'] == 'plate reader'
        assert session_id not in status.text, (
            'the id is the only proof of holding: a client that read it could act as the holder'
        )

    def test_a_request_naming_an_ended_session_is_refused_409(self, sim_session, client):
        session_id = client.post('/api/v1/take_control', json={'name': 'plate reader'}).json()
        with calling(Caller.GUI):
            sim_session.end_control(session_id)

        refused = client.put(
            '/api/v1/acceleration_limit_pct',
            json={'val_pct': 40},
            headers={'Control-Session': session_id},
        )
        assert refused.status_code == 409
        assert refused.json()['reason'] == 'control_session_ended'
