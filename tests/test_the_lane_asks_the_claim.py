# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""While a run or a diagnostic holds the scope, a lane runs only its work.

The activity claim was the one store of who holds the scope, and nothing
that writes read it: a diagnostic switched no executor mode, so any caller
could drive the motors, LEDs and camera in the middle of one, and a run's
fences left a door (``protocol_put``) open to anyone. Each IO and CAMERA
task now carries the taking it was made under, and the lane refuses one
that does not carry the holder's -- at submit and again when the worker
takes it off the queue -- with the one typed refusal.
"""

import threading

import pytest

from modules.activity_claim import ActivityClaim, acting, current_taking
from modules.exceptions import HardwareCommandRefusedError, Refusal
from modules.sequential_io_executor import ENQUEUED, IOTask, SequentialIOExecutor

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


def _submit_from_another_thread(lane, task):
    """Submit ``task`` from a fresh thread, which acts under no taking."""
    box = {}

    def _submit():
        box['fut'] = lane.put(task, return_future=True)

    t = threading.Thread(target=_submit)
    t.start()
    t.join(_WAIT_S)
    return box['fut']


class TestTheRefusal:
    def test_it_is_a_refusal_written_for_the_person_and_names_the_holder(self):
        refused = HardwareCommandRefusedError(
            'exclusive_activity_running', 'move_absolute', 'diagnostic'
        )
        assert isinstance(refused, Refusal)
        assert refused.title
        assert 'diagnostic' in str(refused)
        assert 'exclusive_activity_running' not in str(refused), (
            'the popup body is the message; a reason code is for REST, not the person'
        )


class TestADiagnosticHold:
    def test_a_non_holders_task_is_refused_and_never_runs(self, claim, lane):
        ran = threading.Event()
        held = claim.try_claim('diagnostic')
        try:
            fut = _submit_from_another_thread(lane, IOTask(action=ran.set, silent_on_failure=True))
            with pytest.raises(HardwareCommandRefusedError) as refused:
                fut.result(timeout=_WAIT_S)
            assert refused.value.holder == 'diagnostic'
            assert not ran.wait(0.3)
        finally:
            held.release()

    def test_the_holders_own_task_runs(self, claim, lane):
        held = claim.try_claim('diagnostic')
        try:
            with acting(held):
                fut = lane.put(IOTask(action=lambda: 'ran'), return_future=True)
            assert fut.result(timeout=_WAIT_S) == 'ran'
        finally:
            held.release()

    def test_a_fire_and_forget_refusal_is_returned_not_dropped(self, claim, lane):
        held = claim.try_claim('diagnostic')
        try:
            box = {}
            t = threading.Thread(
                target=lambda: box.setdefault(
                    'r', lane.put(IOTask(action=lambda: None, silent_on_failure=True))
                )
            )
            t.start()
            t.join(_WAIT_S)
            assert isinstance(box['r'], HardwareCommandRefusedError)
        finally:
            held.release()

    def test_work_queued_before_the_hold_began_is_refused_not_run(self, claim, lane):
        gate = threading.Event()
        ran = threading.Event()
        lane.put(IOTask(action=gate.wait, args=(_WAIT_S,)))
        queued = lane.put(IOTask(action=ran.set, silent_on_failure=True), return_future=True)
        held = claim.try_claim('diagnostic')
        try:
            gate.set()
            with pytest.raises(HardwareCommandRefusedError):
                queued.result(timeout=_WAIT_S)
            assert not ran.is_set()
        finally:
            held.release()

    def test_the_worker_acts_under_the_taking_of_the_task_it_runs(self, claim, lane):
        nested_ran = threading.Event()
        seen = {}

        def _outer():
            seen['taking'] = current_taking()
            return lane.put(IOTask(action=nested_ran.set))

        held = claim.try_claim('diagnostic')
        try:
            with acting(held):
                fut = lane.put(IOTask(action=_outer), return_future=True)
            assert fut.result(timeout=_WAIT_S) is ENQUEUED
            assert seen['taking'] is held
            assert nested_ran.wait(_WAIT_S)
        finally:
            held.release()

    def test_the_session_diagnostic_enters_its_taking(self):
        from tests.test_diagnostic_claim import _make_session

        session = _make_session()
        with session.diagnostic_claim() as held:
            assert current_taking() is held
        assert current_taking() is None


class TestOtherHolds:
    def test_a_recording_refuses_nothing_at_the_lane(self, claim, lane):
        held = claim.try_claim('recording')
        try:
            fut = _submit_from_another_thread(lane, IOTask(action=lambda: 'ran'))
            assert fut.result(timeout=_WAIT_S) == 'ran'
        finally:
            held.release()

    def test_unheld_everything_runs(self, lane):
        fut = _submit_from_another_thread(lane, IOTask(action=lambda: 'ran'))
        assert fut.result(timeout=_WAIT_S) == 'ran'

    def test_the_run_door_is_closed_to_a_non_holder(self, claim, lane):
        """``protocol_put`` admitted anyone while a run held (route table row 27)."""
        ran = threading.Event()
        held = claim.try_claim('protocol', run_trigger_source='test')
        lane.protocol_start()
        try:
            box = {}
            t = threading.Thread(
                target=lambda: box.setdefault(
                    'fut',
                    lane.protocol_put(
                        IOTask(action=ran.set, silent_on_failure=True), return_future=True
                    ),
                )
            )
            t.start()
            t.join(_WAIT_S)
            with pytest.raises(HardwareCommandRefusedError):
                box['fut'].result(timeout=_WAIT_S)
            with acting(held):
                ok = lane.protocol_put(IOTask(action=lambda: 'ran'), return_future=True)
            assert ok.result(timeout=_WAIT_S) == 'ran'
            assert not ran.is_set()
        finally:
            lane.protocol_end()
            held.release()


class TestABorrowing:
    def test_a_run_inside_a_diagnostic_writes_until_it_ends(self, claim, lane):
        held = claim.try_claim('diagnostic')
        run = held.lend().try_claim('protocol', run_trigger_source='api_autofocus')
        try:
            with acting(run):
                assert lane.put(IOTask(action=lambda: 1), return_future=True).result(_WAIT_S) == 1
            run.release()
            assert not run.holds, 'an ended borrowing still held'
            with acting(run):
                fut = lane.put(IOTask(action=lambda: 1, silent_on_failure=True), return_future=True)
            with pytest.raises(HardwareCommandRefusedError):
                fut.result(timeout=_WAIT_S)
        finally:
            held.release()

    def test_ending_a_borrowing_ends_what_it_lent(self, claim):
        held = claim.try_claim('diagnostic')
        run = held.lend().try_claim('protocol', run_trigger_source='api_autofocus')
        recording = run.lend().try_claim('recording')
        run.release()
        assert not recording.holds, 'a recording outlived the run it borrowed from'
        assert held.holds
        held.release()


class TestTheOverride:
    def test_the_named_override_passes_a_hold_and_nothing_else_can_claim_it(self, claim):
        ex = SequentialIOExecutor(name='TEST_IO')
        key = ex.ask_claim(claim)
        ex.start()
        held = claim.try_claim('diagnostic')
        try:
            fut = _submit_override(ex, key)
            assert fut.result(timeout=_WAIT_S) == 'ran'
            forged = _submit_override(ex, object())
            with pytest.raises(HardwareCommandRefusedError):
                forged.result(timeout=_WAIT_S)
        finally:
            held.release()
            ex.shutdown()


def _submit_override(ex, key):
    box = {}

    def _submit():
        box['fut'] = ex.put(
            IOTask(action=lambda: 'ran', silent_on_failure=True), return_future=True, override=key
        )

    t = threading.Thread(target=_submit)
    t.start()
    t.join(_WAIT_S)
    return box['fut']


class TestNoLaneWorkerWaitsOnALane:
    def test_a_blocking_dispatch_from_a_lane_worker_raises(self):
        io = SequentialIOExecutor(name='TEST_IO')
        camera = SequentialIOExecutor(name='TEST_CAMERA')
        io.start()
        camera.start()
        try:
            fut = camera.put(
                IOTask(
                    action=io.call,
                    args=(IOTask(action=lambda: None), 'move_absolute', 5.0),
                    silent_on_failure=True,
                ),
                return_future=True,
            )
            with pytest.raises(RuntimeError, match='lane'):
                fut.result(timeout=_WAIT_S)
        finally:
            io.shutdown()
            camera.shutdown()

    def test_the_worker_pool_is_not_a_lane(self):
        io = SequentialIOExecutor(name='TEST_IO')
        pool = SequentialIOExecutor(name='TEST_POOL', lane=False)
        io.start()
        pool.start()
        try:
            fut = pool.put(
                IOTask(action=io.call, args=(IOTask(action=lambda: 'ran'), 'move_absolute', 5.0)),
                return_future=True,
            )
            assert fut.result(timeout=_WAIT_S) == 'ran'
        finally:
            io.shutdown()
            pool.shutdown()


class TestTheSession:
    @pytest.fixture
    def sim_session(self, tmp_path):
        from modules.scope_session import ScopeSession
        from tests.scope_fakes import home_sim_scope
        from tests.settings_fixtures import complete_settings

        s = ScopeSession.create(
            complete_settings(
                live_folder=str(tmp_path),
                microscope='LS850T',
                objective_confirmed=True,
                turret_objectives={1: '10x Oly', 2: '4x Oly', 3: None, 4: None},
            ),
            simulate=True,
        )
        try:
            home_sim_scope(s.scope)
            s.scope.motion.move_turret(1)
            yield s
        finally:
            s.shutdown()

    def test_a_diagnostic_moves_and_a_bystander_is_refused(self, sim_session):
        motion = sim_session.scope.motion
        with sim_session.diagnostic_claim():
            motion.move_absolute('Z', 1000.0, wait_until_complete=True)
            box = {}

            def _bystander():
                try:
                    motion.move_absolute('Z', 2000.0)
                except HardwareCommandRefusedError as exc:
                    box['refused'] = exc

            t = threading.Thread(target=_bystander)
            t.start()
            t.join(10.0)
            assert box['refused'].holder == 'diagnostic'

    def test_a_still_taken_inside_a_diagnostic_is_its_own(self, sim_session):
        with sim_session.diagnostic_claim():
            paths = sim_session.manual_capture.capture(layer=None, false_color_on=False).result(
                timeout=30.0
            )
        assert paths

    def test_the_file_writer_is_not_recovered_under_a_hold(self, sim_session):
        with sim_session.diagnostic_claim(), pytest.raises(HardwareCommandRefusedError):
            sim_session.recover_file_writer()


class TestEveryHolderThreadActsUnderItsTaking:
    def test_the_autofocus_sweep_acts_under_its_callers_taking(self, claim):
        from modules.autofocus_thread import AutofocusThread

        seen = {}

        class _Sweep:
            def run(self, **kwargs):
                seen['taking'] = current_taking()
                return 1.0

        af = AutofocusThread(afe=_Sweep())
        af.start()
        held = claim.try_claim('protocol', run_trigger_source='test')
        try:
            with acting(held):
                fut = af.run_autofocus(run_trigger_source='test')
            assert fut.result(timeout=_WAIT_S) == 1.0
            assert seen['taking'] is held, (
                "the sweep's moves and LED writes are its run's; without its taking a lane "
                'refuses them while the run holds the scope'
            )
        finally:
            held.release()
            af.stop()


class TestTheHardwareMembersUseTheLanesDispatch:
    """Each public hardware member reaches its lane through ``call``, so each
    inherits its refusals -- including the one for a lane worker that waits."""

    @pytest.fixture
    def sim_session(self, tmp_path):
        from modules.scope_session import ScopeSession
        from tests.scope_fakes import home_sim_scope
        from tests.settings_fixtures import complete_settings

        s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            home_sim_scope(s.scope)
            yield s
        finally:
            s.shutdown()

    @pytest.mark.parametrize(
        ('lane', 'member'),
        [
            ('camera_executor', lambda scope: scope.motion.move_absolute('Z', 1000.0)),
            ('camera_executor', lambda scope: scope.illumination.leds_off()),
            ('io_executor', lambda scope: scope.imaging.set_gain_db(1.0)),
        ],
        ids=['motion', 'illumination', 'imaging'],
    )
    def test_a_member_called_from_a_lane_worker_raises(self, sim_session, lane, member):
        scope = sim_session.scope
        fut = getattr(sim_session, lane).put(
            IOTask(action=member, args=(scope,), silent_on_failure=True), return_future=True
        )
        with pytest.raises(RuntimeError, match='never waits on another lane'):
            fut.result(timeout=10.0)


class TestTheShutdownOverride:
    def test_shutdown_darkens_on_the_io_lane_while_a_diagnostic_holds(self, tmp_path, monkeypatch):
        from modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        illumination = session.scope.illumination
        threads = []
        real = illumination._leds_off_impl

        def _spy(*args, **kwargs):
            threads.append(threading.current_thread().name)
            return real(*args, **kwargs)

        monkeypatch.setattr(illumination, '_leds_off_impl', _spy)
        held = session.activity_claim.try_claim('diagnostic')
        try:
            session.shutdown()
        finally:
            held.release()
        assert 'IO_WORKER' in threads, (
            f"shutdown's LED drain did not run on the io lane under a diagnostic hold: {threads}"
        )


class TestTheCaptureButtonShowsTheRefusalsOwnWords:
    def test_title_and_body_are_the_refusals(self):
        from tests.test_capture_button_display import _shown

        refused = HardwareCommandRefusedError(
            'exclusive_activity_running', 'manual_capture.capture', 'diagnostic'
        )
        _category, title, body = _shown(refused)
        assert title == refused.title
        assert body == str(refused)


class TestRunCleanupIsTheRunsOwnWork:
    def test_cleanup_on_a_foreign_thread_acts_under_the_runs_taking(self, monkeypatch):
        """A stop pressed in the GUI or a script's reset ends the run on its own
        thread; the return moves and the LED end-state cleanup submits are the
        run's, and a lane refuses them under the run's hold without its taking."""
        from tests.test_diagnostic_claim import _make_session

        session = _make_session()
        runner = session.sequenced_capture_runner
        seen = {}
        monkeypatch.setattr(
            runner, '_cleanup_inner', lambda ending: seen.setdefault('t', current_taking())
        )
        held = session.activity_claim.try_claim('protocol', run_trigger_source='test')
        runner._held_claim = held
        try:
            t = threading.Thread(target=runner._cleanup, args=(object(),))
            t.start()
            t.join(_WAIT_S)
            assert seen['t'] is held
        finally:
            runner._held_claim = None
            held.release()


class TestTheHoldersWorkRunsOnTheLane:
    def test_a_write_under_the_taking_runs_on_the_lanes_worker(self, claim, lane):
        held = claim.try_claim('diagnostic')
        try:
            with acting(held):
                fut = lane.put(
                    IOTask(action=lambda: threading.current_thread().name), return_future=True
                )
            assert fut.result(timeout=_WAIT_S) == lane.executor_name
        finally:
            held.release()

    def test_a_member_called_on_its_own_lanes_worker_runs_inline(self, tmp_path):
        """A task already on the IO worker that moves the stage runs the move
        there; waiting on the queue it is draining would deadlock."""
        from modules.scope_session import ScopeSession
        from tests.scope_fakes import home_sim_scope
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            home_sim_scope(session.scope)
            motion = session.scope.motion
            seen = {}
            real = motion._move_absolute_impl

            def _move(*args, **kwargs):
                seen['thread'] = threading.current_thread().name
                return real(*args, **kwargs)

            motion._move_absolute_impl = _move
            fut = session.io_executor.put(
                IOTask(
                    action=motion.move_absolute,
                    args=('Z', 1500.0),
                    kwargs={'wait_until_complete': True},
                ),
                return_future=True,
            )
            fut.result(timeout=30.0)
            assert seen['thread'] == session.io_executor.executor_name
            assert motion.get_actual_position('Z') == pytest.approx(1500.0)
        finally:
            session.shutdown()


class TestTheRunsOwnWorkRunsUnderItsClaim:
    def test_a_protocol_with_autofocus_video_and_grease_completes(self, tmp_path, monkeypatch):
        from modules.protocol_step_runner import ProtocolStepRunner
        from modules.scope_session import ScopeSession
        from modules.sequenced_capture_runner import SequencedCaptureRunner
        from tests.scope_fakes import home_sim_scope
        from tests.settings_fixtures import complete_settings
        from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
        from tests.test_run_refusal_contract import (
            _build_real_protocol,
            _make_single_step_protocol,
        )

        # Grease runs after the hundredth autofocus; start one short so this
        # run's own autofocus brings it round.
        real_reset = SequencedCaptureRunner._reset_vars

        def _reset_one_short(self):
            real_reset(self)
            self._autofocus_count = 99

        monkeypatch.setattr(SequencedCaptureRunner, '_reset_vars', _reset_one_short)
        greased = []
        real_grease = ProtocolStepRunner._grease_redist_w_pos

        def _grease(self):
            greased.append(threading.current_thread().name)
            return real_grease(self)

        monkeypatch.setattr(ProtocolStepRunner, '_grease_redist_w_pos', _grease)

        session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
        try:
            home_sim_scope(session.scope)
            base = _make_single_step_protocol().step(idx=0)
            af_step = {**base, 'Name': 'A1_af', 'Label': 'A1_af', 'Auto_Focus': True}
            video_step = {
                **base,
                'Name': 'A1_video',
                'Label': 'A1_video',
                'Acquire': 'video',
                'Step Index': 1,
            }
            protocol = _build_real_protocol([af_step, video_step])
            runner = session.create_protocol_runner()
            files_written = threading.Event()
            runner.run_single_scan(
                protocol=protocol,
                sequence_name='gating',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks={
                    'run_complete': lambda **kw: None,
                    'files_complete': lambda **kw: files_written.set(),
                },
            )
            assert files_written.wait(COMPLETION_TIMEOUT), 'the run never finished its files'
            outcome = runner.wait_for_completion(timeout=COMPLETION_TIMEOUT)
            assert outcome.status == 'completed', outcome
            assert greased == [session.io_executor.executor_name], greased
        finally:
            session.shutdown()


class TestTheSupportReportUnderItsClaim:
    def test_its_writing_steps_reach_the_hardware(self, tmp_path):
        """The report's LED, fan and homing steps run under its diagnostic
        claim on a real session, and none of them is refused by the lanes."""
        from modules.scope_session import ScopeSession
        from modules.tech_support_report import TechSupportReport
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            TechSupportReport(session=session)._run_scope_steps(tmp_path, lambda pct, msg: None)
        finally:
            session.shutdown()
        homing = (tmp_path / 'motion_tests' / 'homing_test.txt').read_text()
        for axis in ('X', 'Y', 'Z'):
            block = homing.split(f'{axis} axis:')[1].split('axis:')[0]
            assert 'Home response: OK' in block, homing
        for rel in ('hardware_checks/led_leakage.txt', 'hardware_checks/fan_test.txt'):
            text = (tmp_path / rel).read_text()
            assert 'SKIPPED' not in text and 'refused' not in text.lower(), (rel, text)


class TestAnInlineCallAsksTheClaim:
    def test_a_task_running_when_a_hold_begins_is_refused_its_next_write(self, claim, lane):
        """The inline path on a lane's own worker asks the claim like the queue
        does: a task that started unheld cannot write after a hold began."""
        started = threading.Event()
        go_on = threading.Event()

        def _outer():
            started.set()
            go_on.wait(_WAIT_S)
            return lane.call(IOTask(action=lambda: 'wrote'), 'move_absolute', 5.0)

        fut = lane.put(IOTask(action=_outer, silent_on_failure=True), return_future=True)
        assert started.wait(_WAIT_S)
        held = claim.try_claim('diagnostic')
        try:
            go_on.set()
            with pytest.raises(HardwareCommandRefusedError):
                fut.result(timeout=_WAIT_S)
        finally:
            held.release()
