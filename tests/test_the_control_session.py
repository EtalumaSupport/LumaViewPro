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

_WAIT_S = 2.0


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
    with calling(caller):
        return claim.take_control(name)


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
        claim.end_control(controller.session.id)
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
        claim.end_control(controller.session.id)
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
        controller = _take(sim_session.activity_claim, 'plate reader', Caller.REST)
        yield controller
        sim_session.activity_claim.end_control(controller.session.id)

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
        claim.end_control(controller.session.id)
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
        controller = _take(sim_session.activity_claim, 'plate reader', Caller.REST)
        yield controller
        if sim_session.activity_claim.control is controller.session:
            sim_session.activity_claim.end_control(controller.session.id)

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
