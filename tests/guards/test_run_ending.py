# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Every run ends once, with a reason, and says so to its subscribers.

The defect these cover: the status word was stated at nine sites and
computed at one from a flag that says only "aborted", while the sixteen
sites that know WHY a run died had nowhere to put it. A run could report
'completed' after a writer-lane fault killed it, and no caller could tell
a user Stop from a dead camera.
"""

from __future__ import annotations

import ast
import dataclasses
import threading
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from modules.run_outcome import EndingLatch, RunEnding
from tests.ast_seams import REPO_ROOT

MODULES = REPO_ROOT / 'modules'


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


class TestEndingLatch:
    def test_the_first_ending_is_the_one_kept(self):
        latch = EndingLatch()
        first = RunEnding('failed', 'motion_timeout', 'Timeout', 'the cause')
        later = RunEnding('failed', 'run_loop_crashed', 'Crash', 'the consequence')

        assert latch.set_if_unset(first) is True
        assert latch.set_if_unset(later) is False, (
            'a later ending overwrote the first; one failure cascades through '
            'several sites and the LAST of them is never the cause'
        )
        assert latch.get() is first

    def test_an_unended_run_has_no_ending(self):
        assert EndingLatch().get() is None

    def test_concurrent_writers_keep_exactly_one_ending(self):
        latch = EndingLatch()
        start = threading.Barrier(8)
        wins = []

        def race(i):
            start.wait()
            if latch.set_if_unset(RunEnding('failed', f'code_{i}', 't', 'm')):
                wins.append(i)

        threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(wins) == 1, f'{len(wins)} writers each believed they were first'
        assert latch.get().reason == f'code_{wins[0]}'

    def test_the_record_cannot_be_mutated_after_the_fact(self):
        ending = RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped by test')
        with pytest.raises(dataclasses.FrozenInstanceError):
            ending.status = 'completed'


# ---------------------------------------------------------------------------
# The census guard: every site that can end a run names its cause, and no
# site can state an ending without one.
# ---------------------------------------------------------------------------

# Every reason this codebase may record, with the site that owns it. A new
# fault site fails this test until it appears here -- which is the point:
# the vocabulary is a contract with REST and the SDK, not an ad-hoc string.
COVERAGE = {
    'led_channel_unavailable',
    'camera_failure',
    'file_writer_stalled',
    'disk_space_critical',
    'video_writer_died',
    'motion_timeout',
    'hardware_disconnected',
    'consecutive_scan_failures',
    'run_loop_crashed',
    'stopped',
    'force_reset',
    'capture_location_unusable',
    'run_dir_init_failed',
    'dispatch_refused',
    'start_failed',
    'completed',
}

ENDING_FILES = [
    'protocol_run_loop.py',
    'protocol_step_runner.py',
    'protocol_image_writer.py',
    'protocol_recording.py',
    'sequenced_capture_runner.py',
]


def _tree(name):
    return ast.parse((MODULES / name).read_text()), name


def _recorded_reasons():
    """Every reason literal handed to the funnel or to a RunEnding."""
    found = set()
    for name in ENDING_FILES:
        tree, _ = _tree(name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            label = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
            if label in ('abort_run_fatal', '_abort_run_fatal', 'RunStartError'):
                # (reason, domain, title, message) on the writer's own funnel,
                # (reason, title, message) through the runner's seam and on the
                # typed start failure -- the cause is first in all three.
                arg = node.args[0] if node.args else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
            elif label == 'RunEnding':
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    found.add(node.args[1].value)
    return found


def test_every_recorded_reason_is_in_the_vocabulary():
    recorded = _recorded_reasons()
    unknown = recorded - COVERAGE
    assert not unknown, (
        f'these reasons reach a caller but are in no vocabulary: {sorted(unknown)} -- '
        f'add them here and to the API docs, or reuse an existing code'
    )


def test_every_vocabulary_entry_has_a_producer():
    recorded = _recorded_reasons()
    # 'completed' is built from a module constant rather than at a call site,
    # so it is named here as the one entry with no literal argument.
    dead = COVERAGE - recorded - {'completed'}
    assert not dead, (
        f'these codes are documented but nothing can produce them: {sorted(dead)} -- '
        f'a code no site writes is a promise to a caller that never arrives'
    )


def test_no_cleanup_call_states_an_ending_without_one():
    """_cleanup takes exactly one argument: the ending. Never a bare call."""
    offenders = []
    for path in MODULES.glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == '_cleanup'
                and (len(node.args) != 1 or node.keywords)
            ):
                offenders.append(f'{path.name}:{node.lineno}')
    assert not offenders, (
        f'these cleanup calls do not pass exactly one ending: {offenders} -- '
        f'a defaulted or omitted ending lets a failure report itself as a '
        f'normal completion to every run_complete subscriber'
    )


def test_every_error_state_write_aborts_the_run_in_the_same_function():
    """ERROR without an abort is the wedge the deleted run-loop gate papered over.

    A step failure that set ERROR but left the loop running re-entered the
    scan path every period, the transition raised, and the run retried
    forever delivering nothing. The gate that caught that is gone; this is
    what replaces it -- the invariant itself, checked at build time.
    """
    offenders = []
    for name in ('protocol_run_loop.py', 'protocol_step_runner.py'):
        tree, _ = _tree(name)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            sets_error = False
            aborts = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    label = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
                    if label == '_set_state' and any(
                        isinstance(a, ast.Attribute) and a.attr == 'ERROR' for a in node.args
                    ):
                        sets_error = True
                    if label in ('abort_run_fatal', '_abort_run_fatal'):
                        aborts = True
            if sets_error and not aborts:
                offenders.append(f'{name}:{fn.name}')
    assert not offenders, (
        f'these functions put the run in ERROR without aborting it: {offenders} -- '
        f'the loop keeps running, the next period re-enters the scan path, and '
        f'the run retries forever delivering nothing'
    )


# ---------------------------------------------------------------------------
# The endings the runner itself records
# ---------------------------------------------------------------------------


def _stop_stub(trigger='test', signals_inline_cleanup=False):
    import modules.sequenced_capture_runner as scr

    cleaned = []
    return (
        scr,
        SimpleNamespace(
            _run_lock=threading.RLock(),
            _is_run_live=lambda: True,
            _run_trigger_source=trigger,
            _ending=EndingLatch(),
            _signal_abort_locked=lambda: signals_inline_cleanup,
            _cleanup=cleaned.append,
            LOGGER_NAME='TEST',
        ),
        cleaned,
    )


class TestTheRunnerRecordsWhoStoppedIt:
    def test_a_stop_records_its_requester(self):
        scr, stub, _cleaned = _stop_stub()
        scr.SequencedCaptureRunner.reset(stub, 'test')

        ending = stub._ending.get()
        assert (ending.status, ending.reason) == ('aborted', 'stopped')
        assert 'test' in ending.message, (
            f'the ending must name who stopped the run; got {ending.message!r}'
        )

    def test_a_stop_with_nothing_to_stop_records_nothing(self):
        """A no-op Stop ended no run, so it must leave no reason behind for
        the NEXT run to report as its own."""
        scr, stub, _ = _stop_stub()
        stub._is_run_live = lambda: False
        scr.SequencedCaptureRunner.reset(stub, 'test')
        assert stub._ending.get() is None

    def test_the_inline_cleanup_gets_the_same_record(self):
        scr, stub, cleaned = _stop_stub(signals_inline_cleanup=True)
        scr.SequencedCaptureRunner.reset(stub, 'test')
        assert len(cleaned) == 1
        assert cleaned[0] is stub._ending.get(), (
            'cleanup was handed a different object than the one recorded, so '
            'the two could disagree about how the run ended'
        )

    def test_force_reset_names_itself_and_its_reason(self):
        scr, stub, _ = _stop_stub()
        scr.SequencedCaptureRunner.force_reset(stub, 'app shutdown')

        ending = stub._ending.get()
        assert (ending.status, ending.reason) == ('aborted', 'force_reset')
        assert ending.message == 'app shutdown'


class TestAStartFailureCarriesItsCause:
    def test_a_typed_start_failure_keeps_its_code_and_sentence(self):
        import modules.sequenced_capture_runner as scr
        from modules.exceptions import RunStartError

        cleaned = []
        stub = SimpleNamespace(
            _run_dir=None, _ending=EndingLatch(), _cleanup=cleaned.append, LOGGER_NAME='TEST'
        )
        scr.SequencedCaptureRunner._fail_run_at_start(
            stub,
            RunStartError('capture_location_unusable', 'Run failed to start', 'Pick a folder.'),
        )

        ending = stub._ending.get()
        assert ending.status == 'failed_at_start'
        assert ending.reason == 'capture_location_unusable'
        assert ending.message == 'Pick a folder.'
        assert cleaned == [ending]

    def test_an_untyped_failure_does_not_put_a_traceback_in_front_of_the_user(self):
        import modules.sequenced_capture_runner as scr

        stub = SimpleNamespace(
            _run_dir=None, _ending=EndingLatch(), _cleanup=lambda e: None, LOGGER_NAME='TEST'
        )
        scr.SequencedCaptureRunner._fail_run_at_start(
            stub, RuntimeError('SerialException: device reports readiness but returned no data')
        )

        ending = stub._ending.get()
        assert ending.reason == 'start_failed'
        assert 'SerialException' not in ending.message, (
            f'a raw exception string reached a field a popup shows and a REST '
            f'handler serialises: {ending.message!r}'
        )


class TestACleanupPassThatDoesNotOwnTheRun:
    def test_settles_nothing_and_releases_nothing(self):
        """The second pass of every normal run arrives after the owner's pass
        has already released. Its releases key on runner-lifetime state, so
        acting here can hand away a claim a SUCCESSOR run has taken.
        """
        import modules.sequenced_capture_runner as scr

        touched = []
        stub = SimpleNamespace(
            _is_run_live=lambda: False,  # this pass does not own the run
            _io_executor=SimpleNamespace(end_protocol_mode=lambda: touched.append('io')),
            file_io_executor=SimpleNamespace(end_protocol_mode=lambda: touched.append('file')),
            _settle_run_outcome=lambda ending: touched.append('settled'),
            _release_scan_led_lease=lambda: touched.append('lease'),
            _release_activity_claim=lambda: touched.append('claim'),
        )

        scr.SequencedCaptureRunner._cleanup_inner(
            stub, RunEnding('failed', 'run_loop_crashed', 'Protocol Crashed', 'x')
        )

        assert touched == ['io', 'file'], (
            f'a pass that does not own the run touched more than the executors '
            f'it must always end: {touched}'
        )


# ---------------------------------------------------------------------------
# Which ending wins, and what the sample does about it
# ---------------------------------------------------------------------------


def _cleanup_stub(latched=None, forced_dark=False):
    """A runner far enough along to reach cleanup's single read."""
    fatal = threading.Event()
    if forced_dark:
        fatal.set()
    latch = EndingLatch()
    if latched is not None:
        latch.set_if_unset(latched)
    # MagicMock-backed: _cleanup_inner builds run_cleanup's whole kwarg list
    # before calling it, so every attribute it reads must exist. Only the ones
    # this test reasons about are pinned; the rest are inert.
    stub = MagicMock()
    stub._is_run_live = lambda: True
    stub._ending = latch
    stub._fatal_abort_event = fatal
    stub._image_writer = None  # no video lane to drain
    return stub


def _run_cleanup_args(monkeypatch, stub, stated):
    """Drive _cleanup_inner and hand back what it told run_cleanup."""
    import modules.sequenced_capture_runner as scr

    seen = {}

    def _fake_run_cleanup(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(scr, 'run_cleanup', _fake_run_cleanup)
    scr.SequencedCaptureRunner._cleanup_inner(stub, stated)
    return seen


COMPLETED = RunEnding('completed', 'completed', 'Protocol Complete', 'The run finished normally.')
STOPPED = RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped by test')


class TestTheLatchOutranksTheWordTheLoopStated:
    def test_a_writer_fatal_after_the_loop_said_completed_reports_failed(self, monkeypatch):
        """The writer lanes keep draining after the loop has said its word.
        A disk floor or a dead video writer hit during that drain used to be
        reported as a completed run, because cleanup took the loop's word.
        """
        stub = _cleanup_stub(
            latched=RunEnding(
                'failed', 'video_writer_died', 'Video Writer Failed', 'the lane died'
            ),
            forced_dark=True,
        )
        seen = _run_cleanup_args(monkeypatch, stub, COMPLETED)

        assert seen['ending'].status == 'failed'
        assert seen['ending'].reason == 'video_writer_died', (
            f'the run reported the word the loop had already stated instead of '
            f'the fault that killed it: {seen["ending"]}'
        )

    def test_an_unended_run_keeps_the_word_its_caller_stated(self, monkeypatch):
        stub = _cleanup_stub(latched=None)
        seen = _run_cleanup_args(monkeypatch, stub, COMPLETED)
        assert seen['ending'] is COMPLETED

    def test_a_stop_then_a_fatal_stays_a_stop_but_still_goes_dark(self, monkeypatch):
        """Two disciplines at once: the user's Stop is why the run ended
        (first-wins), and the fault that darkened the sample still forces the
        LED policy off, whatever the end-state setting says.
        """
        stub = _cleanup_stub(latched=STOPPED, forced_dark=True)
        seen = _run_cleanup_args(monkeypatch, stub, COMPLETED)

        assert (seen['ending'].status, seen['ending'].reason) == ('aborted', 'stopped')
        assert seen['forced_dark'] is True, (
            'the sample was force-darkened by a fault; cleanup must not restore '
            'the pre-run channels over it'
        )

    def test_a_stop_with_no_fault_keeps_the_configured_end_state(self, monkeypatch):
        stub = _cleanup_stub(latched=STOPPED, forced_dark=False)
        seen = _run_cleanup_args(monkeypatch, stub, STOPPED)
        assert seen['forced_dark'] is False, (
            'a user Stop is not a fault; forcing dark here would override the '
            "user's own end-state policy"
        )


class TestAMotionTimeoutEndsTheRunAsAFault:
    def test_it_aborts_through_the_funnel_naming_the_timeout(self):
        """A timed-out move is a fault the instrument imposed: it force-darkens
        and names itself, where the strike ceiling (a policy stop) does not.
        """
        import modules.protocol_step_runner as psr
        from tests.protocol_drives import protocol_step, scan_ready_runner

        runner = scan_ready_runner(protocol_step())
        runner._scope.motion.is_moving.return_value = True
        runner._motion_wait_start = 0.0  # already past the bound
        runner.MOTION_TIMEOUT_SECONDS = 0.0

        psr.ProtocolStepRunner(runner).scan_iterate()

        aborts = runner._image_writer._abort_run_fatal.call_args_list
        assert len(aborts) == 1, f'a motion timeout must end the run once; got {len(aborts)}'
        reason, _domain, title, _message = aborts[0].args
        assert reason == 'motion_timeout'
        assert 'Motion Timeout' in title


class TestTheRunLoopsOwnEndings:
    def test_a_crashed_run_loop_records_the_exception_as_the_cause(self):
        """The safety net used to hand cleanup the bare word 'failed'. The
        exception it caught is the only thing that knows why."""
        from tests.protocol_drives import protocol_step, run_loop_ready_runner

        runner = run_loop_ready_runner(protocol_step())
        loop = runner._run_loop_executor
        loop._run_loop_inner = MagicMock(side_effect=RuntimeError('the loop died here'))

        loop.run_loop()

        ending = runner._ending.get()
        assert (ending.status, ending.reason) == ('failed', 'run_loop_crashed')
        assert 'the loop died here' in ending.message
        assert runner._cleanup.call_args.args[0] is ending, (
            'cleanup was handed a different ending than the one recorded'
        )

    def test_the_disk_floor_aborts_outside_the_handler_that_swallows_probes(self, monkeypatch):
        """The abort used to sit inside the try whose except logs 'proceeding
        anyway'; a raise from the probe dropped the abort and the run walked
        into the next step on a full disk.
        """
        import modules.protocol_run_loop as prl
        from tests.protocol_drives import protocol_step, run_loop_ready_runner

        runner = run_loop_ready_runner(protocol_step())
        runner._parent_dir = '/tmp'
        # Real numbers: the estimate is compared with max() against a float, so
        # a mock here raises and the probe's own handler would (correctly)
        # proceed -- masking the case under test.
        runner._protocol.estimate_write_mb.return_value = 500.0
        runner._protocol.num_steps.return_value = 1
        # start() always sets this; the bare harness does not, and without it
        # the estimate raises and the probe's handler proceeds -- masking the
        # case under test.
        runner._video_max_fps = 30.0
        monkeypatch.setattr(prl, 'check_disk_space_ok', lambda folder, needed: (False, 12.0))

        runner._run_loop_executor.run_loop()

        aborts = runner._image_writer._abort_run_fatal.call_args_list
        assert len(aborts) >= 1, 'a disk floor below the run estimate must end the run'
        reason, _domain, title, message = aborts[0].args
        assert reason == 'disk_space_critical'
        assert title == 'Protocol Aborted'
        assert '12 MB free' in message, f'the message must name the shortfall; got {message!r}'


# ---------------------------------------------------------------------------
# The ending a caller holds
# ---------------------------------------------------------------------------


def _plain_scan_protocol(session):
    """The steps a composite would capture, for a NON-composite run.

    Borrowed from production assembly rather than hand-rolled: what makes
    this a scan is the run MODE, not which positions it visits.
    """
    import modules.config_helpers as config_helpers

    input_config = config_helpers.get_composite_capture_config_from_settings(
        session.settings,
        session.objective_helper,
        position=session.get_current_plate_position(),
    )
    return session.scope.protocols.create_protocol(input_config=input_config)


class TestTheCallerHoldsTheEndingOfTheRunItStarted:
    """The run's ending reaches the caller that asked for the run.

    Sixteen sites could end a run and none of the answers left the
    process: run_single_scan dropped what it started, and
    wait_for_completion was one bit with three meanings ("completed",
    "timed out", "never started"). A caller could not tell a user Stop
    from a dead camera from a run that was refused.
    """

    def test_a_scan_hands_back_the_ending_its_subscribers_were_told(self, tmp_path):
        from tests.test_composite_run_e2e import headless_settings, open_composite_session

        reported = {}
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            pending = runner.run_single_scan(
                _plain_scan_protocol(session),
                sequence_name='ending_scan',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks={'run_complete': lambda **kw: reported.update(kw)},
            )
            assert pending is not None, (
                'run_single_scan committed a run and handed back nothing; the '
                'caller has no way to learn how its own run ended'
            )
            settled = pending.wait(timeout_s=60)

        assert settled is not None, 'the committed run never settled its outcome'
        # The same fact through both channels. A caller that waits and a
        # subscriber that is called back must not be able to disagree about
        # the run they are both describing.
        assert settled.status == reported['status'] == 'completed', (
            f'the waiter was told {settled.status!r} and the run_complete '
            f'subscriber {reported.get("status")!r}'
        )
        assert settled.reason == reported['ending'].reason

    def test_a_completed_scan_carries_no_merge_verdict(self, tmp_path):
        # A scan has no merge, so the merge fields say nothing rather than
        # inventing a code. 'not_a_composite_run' in the field a caller
        # branches on for merge failures read as one.
        from tests.test_composite_run_e2e import headless_settings, open_composite_session

        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            pending = runner.run_single_scan(
                _plain_scan_protocol(session),
                sequence_name='no_merge_scan',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
            settled = pending.wait(timeout_s=60)

        assert settled is not None
        assert (settled.status, settled.merged, settled.merge_reason) == ('completed', False, ''), (
            f'a scan reported merged={settled.merged!r} merge_reason={settled.merge_reason!r}'
        )
        assert settled.artifact_path is None

    def test_wait_for_completion_answers_for_the_last_committed_run(self, tmp_path):
        from tests.test_composite_run_e2e import headless_settings, open_composite_session

        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            runner.run_single_scan(
                _plain_scan_protocol(session),
                sequence_name='runner_wait_scan',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            )
            settled = runner.wait_for_completion(timeout=60)

        assert settled is not None, 'wait_for_completion reported nothing for a committed run'
        assert settled.status == 'completed'


class TestWaitForCompletionWithNothingToReport:
    """The two answers that are not a run's outcome.

    Both were the same bit before: False meant "timed out" and also
    "never started", so a caller could not tell a slow run from one that
    was refused before it began.
    """

    def _runner_with(self, outcome):
        from modules.protocol_runner import ProtocolRunner

        runner = ProtocolRunner.__new__(ProtocolRunner)
        runner._last_outcome = outcome
        return runner

    def test_a_fresh_runner_answers_none_at_once(self):
        import time as _time

        from modules.protocol_runner import ProtocolRunner

        runner = ProtocolRunner.__new__(ProtocolRunner)
        runner._last_outcome = None

        t0 = _time.monotonic()
        assert runner.wait_for_completion(timeout=30) is None
        assert _time.monotonic() - t0 < 1.0, (
            'a runner that has committed no run must answer at once rather '
            'than blocking out the bound for a run that does not exist'
        )

    def test_a_live_run_answers_none_when_the_bound_expires(self):
        from modules.run_outcome import PendingRunOutcome

        runner = self._runner_with(PendingRunOutcome())

        assert runner.wait_for_completion(timeout=0.05) is None, (
            'an unsettled run must time out as None, distinct from a settled '
            'outcome that reports the run did not merge'
        )


class TestSessionShutdownDoesNotRewriteAReportedEnding:
    def test_a_composite_armed_at_shutdown_keeps_completed(self, tmp_path):
        """The run already told its subscribers it completed.

        Shutdown cuts the MERGE short, not the run: the executors go down
        without draining, so the merge can never finish and a blocked
        caller has to be released. Releasing it with 'aborted' would put
        the waiter and the run_complete subscriber in contradiction about
        a run that did, in fact, complete.
        """
        from modules.run_outcome import PendingRunOutcome, RunEnding
        from tests.test_composite_run_e2e import headless_settings, open_composite_session

        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            armed = PendingRunOutcome()
            armed.arm(RunEnding('completed', 'completed', 'Protocol Complete', 'The run finished.'))
            session.sequenced_capture_runner._run_outcome = armed

            session.shutdown()

            settled = armed.wait(timeout_s=5)

        assert settled is not None, 'shutdown left a caller blocked on a merge that cannot finish'
        assert settled.status == 'completed', (
            f'session shutdown rewrote a completed run as {settled.status!r}'
        )
        assert settled.merge_reason == 'shutdown', (
            'the shutdown is why no artifact followed, not how the run ended'
        )
