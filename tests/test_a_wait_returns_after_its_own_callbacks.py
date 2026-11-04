# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's operation is done only once the run has let go and its handlers have run.

A run's wait once returned at the release while the run's ``run_ended`` was
still to run, and its files' wait when the last write landed while the
lost-image report, the record's reconcile and ``files_written`` were still
to run: a caller that waited and then read what its own callback did read
it before the callback had done it. The run's ``Operation`` settles at the
end of its ``run_ended`` delivery, so every observer of it -- ``result()``,
``done()``, a done-callback, ``concurrent.futures.wait`` and
``as_completed`` -- sees the outcome only then; inside that delivery its
``result()`` answers at once.

Driven on the real engine with simulated hardware, headless (callbacks
delivered inline on the thread that ends the run) and under a stand-in for
the GUI's dispatcher, which delivers on its own thread a run whose events
ask for the UI thread. A run whose events do not ask -- a REST client's, a
script's -- is told on the thread that ends it, so it settles while the UI
thread is busy.
"""

import concurrent.futures as cf
import queue
import threading
import time

import pytest

from modules import kivy_utils
from modules.exceptions import RunCleanupFailedError
from modules.protocol import Protocol
from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

BOUND_S = 30.0
SLOW_S = 0.5


def _scan(runner, parent, events, *, save=True, sequence_name=None):
    kwargs = {} if sequence_name is None else {'sequence_name': sequence_name}
    return runner.start_single_scan(
        enable_image_saving=save,
        protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
        parent_dir=str(parent),
        events=events,
        **kwargs,
    )


@pytest.fixture
def scope(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        yield runner, tmp_path / 'runs'


class _DeferringDispatcher:
    """Delivers each scheduled callback later, on its own named thread, as the GUI's Clock does."""

    def __init__(self):
        self._queue = queue.Queue()
        self.thread = threading.Thread(target=self._serve, name='stand_in_ui', daemon=True)
        self.thread.start()

    def _serve(self):
        for func in iter(self._queue.get, None):
            func(0)

    def schedule(self, func, timeout):
        self._queue.put(func)

    def call(self, func):
        """Run *func* on the dispatcher's thread, outside any delivery; return what it returned or raised."""
        done = queue.Queue()

        def _run(_dt):
            try:
                done.put(('returned', func()))
            except Exception as ex:
                done.put(('raised', ex))

        self._queue.put(_run)
        return done.get(timeout=BOUND_S)

    def stop(self):
        self._queue.put(None)


@pytest.fixture
def deferred():
    dispatcher = _DeferringDispatcher()
    ScopeSession.set_ui_dispatcher(
        kivy_utils.UiDispatcher(schedule=dispatcher.schedule, thread=dispatcher.thread)
    )
    yield dispatcher
    ScopeSession.set_ui_dispatcher(None)
    dispatcher.stop()


def _slow(seen, name):
    def _callback(*_args):
        seen[f'{name}_thread'] = threading.current_thread()
        time.sleep(SLOW_S)
        seen[f'{name}_ended'] = time.monotonic()

    return _callback


class TestTheOperationIsNotDoneBeforeItsCallback:
    def test_result_returns_after_a_slow_run_ended(self, scope):
        runner, runs = scope
        seen = {}
        run = _scan(runner, runs, RunEvents(run_ended=_slow(seen, 'run_ended')))

        outcome = run.result(BOUND_S)

        assert outcome.status == 'completed', outcome
        assert 'run_ended_ended' in seen, 'result() returned while run_ended was still running'

    def test_files_result_returns_after_a_slow_files_written(self, scope):
        runner, runs = scope
        seen = {}
        run = _scan(runner, runs, RunEvents(files_written=_slow(seen, 'files_written')))

        files = run.files.result(BOUND_S)

        assert files.outcome == 'written', files
        assert 'files_written_ended' in seen, (
            "the run's files were done while files_written was still running"
        )

    def test_deferred_files_are_done_after_run_ended(self, scope, deferred):
        # Under a deferring dispatcher a run with no files_written has its
        # files told at once, while run_ended still waits its turn.
        runner, runs = scope
        seen = {}
        run = _scan(runner, runs, RunEvents(run_ended=_slow(seen, 'run_ended'), on_ui_thread=True))

        run.files.result(BOUND_S)

        assert seen.get('run_ended_thread') is deferred.thread
        assert 'run_ended_ended' in seen, (
            "the run's files were done while run_ended was still running"
        )


class TestARunThatDoesNotAskForTheUiThread:
    def test_settles_while_the_ui_thread_is_busy(self, scope, deferred):
        # A REST client's run in the GUI's process: its handlers run on the
        # thread that ends it, so its result() does not wait for the GUI.
        runner, runs = scope
        seen = {}
        released = threading.Event()
        deferred.schedule(lambda _dt: released.wait(BOUND_S), 0)
        try:
            run = _scan(runner, runs, RunEvents(run_ended=_slow(seen, 'run_ended')))

            outcome = run.result(BOUND_S)

            assert not released.is_set()
        finally:
            released.set()
        assert outcome.status == 'completed', outcome
        assert seen['run_ended_thread'] is not deferred.thread


class TestEveryObserverSeesTheOutcomeOnceTheRunLetGo:
    def test_a_done_callback_finds_the_scope_free_and_run_ended_run(self, scope):
        runner, runs = scope
        seen = {}
        probed = queue.Queue()
        run = _scan(runner, runs, RunEvents(run_ended=_slow(seen, 'run_ended')))

        run.add_done_callback(
            lambda op: probed.put(
                (
                    op.result(0).status,
                    runner.sequenced_capture_runner.run_in_progress(),
                    'run_ended_ended' in seen,
                )
            )
        )

        assert probed.get(timeout=BOUND_S) == ('completed', False, True)

    @pytest.mark.parametrize('observe', ['cf.wait', 'as_completed'])
    def test_the_standard_waits_see_it_after_run_ended(self, scope, observe):
        runner, runs = scope
        seen = {}
        run = _scan(runner, runs, RunEvents(run_ended=_slow(seen, 'run_ended')))

        if observe == 'cf.wait':
            done, _ = cf.wait([run], timeout=BOUND_S)
            assert done == {run}
        else:
            assert next(cf.as_completed([run], timeout=BOUND_S)) is run

        assert 'run_ended_ended' in seen, f'{observe} returned while run_ended was still running'
        assert not runner.sequenced_capture_runner.run_in_progress()

    def test_a_done_callback_that_waits_does_not_hold_the_next_run(self, scope):
        """A callback runs on the callback thread, never the run's own, so a
        callback that blocks leaves the run's teardown and the next run free."""
        runner, runs = scope
        released = threading.Event()
        entered = threading.Event()

        def _blocks(_op):
            entered.set()
            released.wait(BOUND_S)

        first = _scan(runner, runs, RunEvents(), save=False)
        first.add_done_callback(_blocks)
        try:
            assert entered.wait(BOUND_S), 'the done-callback never ran'
            second = _scan(runner, runs, RunEvents(), save=False, sequence_name='second')
            assert second.progress().is_live or second.done()
        finally:
            released.set()
        assert second.result(BOUND_S).status == 'completed'


def _waits_on_its_own_run(answers, run_box):
    def _callback(*_args):
        run = run_box['run']
        started = time.monotonic()
        answers['done'] = run.done()
        answers['result'] = run.result(BOUND_S)
        answers['took_s'] = time.monotonic() - started

    return _callback


def _run_box_ready(run_box, run):
    run_box['run'] = run
    run_box['ready'].set()


class TestInsideItsOwnRunEnded:
    @pytest.mark.parametrize('dispatch', ['inline', 'deferred'])
    def test_result_answers_at_once(self, scope, dispatch, request):
        if dispatch == 'deferred':
            request.getfixturevalue('deferred')
        runner, runs = scope
        answers = {}
        run_box = {'ready': threading.Event()}
        callback = _waits_on_its_own_run(answers, run_box)

        def _run_ended(*args):
            # The handler may be delivered before start() has returned the
            # operation to this test; it waits for it like any caller would.
            assert run_box['ready'].wait(BOUND_S)
            callback(*args)

        events = RunEvents(run_ended=_run_ended, on_ui_thread=dispatch == 'deferred')
        _run_box_ready(run_box, _scan(runner, runs, events))

        assert run_box['run'].result(BOUND_S).status == 'completed'
        assert answers['done'] is True, answers
        assert answers['result'].status == 'completed', answers
        assert answers['took_s'] < BOUND_S / 2, answers

    def test_result_answers_with_a_failed_start_delivered_inside_it(self, scope, monkeypatch):
        # A's run_ended starts B, which fails after it commits (its
        # protocol copy cannot be written): B's cleanup runs inline, and its
        # run_ended is delivered inside A's on A's thread. B's delivery
        # ending must not end A's. A saves nothing, so B is not refused for
        # A's files first.
        runner, runs = scope
        answers = {}
        run_box = {'ready': threading.Event()}
        nested = {}

        def _copy_refused(*_args, **_kwargs):
            raise OSError('No space left on device')

        def _b_run_ended(*args):
            nested['thread'] = threading.current_thread()
            nested['inside_a'] = 'a_thread' in nested and 'a_left' not in nested

        def _a_run_ended(*args):
            assert run_box['ready'].wait(BOUND_S)
            nested['a_thread'] = threading.current_thread()
            with monkeypatch.context() as patched:
                patched.setattr(Protocol, 'to_file', _copy_refused)
                try:
                    _scan(runner, runs, RunEvents(run_ended=_b_run_ended), sequence_name='b')
                except Exception as ex:
                    nested['b_start'] = ex
            _waits_on_its_own_run(answers, run_box)(*args)
            nested['a_left'] = True

        _run_box_ready(run_box, _scan(runner, runs, RunEvents(run_ended=_a_run_ended), save=False))

        assert run_box['run'].result(BOUND_S) is not None
        assert nested.get('inside_a') and nested['thread'] is nested['a_thread'], (
            "the failed start's run_ended was not delivered inside A's: "
            f'the shape under test is absent ({nested})'
        )
        assert answers['result'] is not None, answers
        assert answers['took_s'] < BOUND_S / 2, answers


class TestARaisingRunEnded:
    def test_still_settles_and_is_reported_once_as_itself(self, scope, monkeypatch):
        from modules.notification_center import notifications

        runner, runs = scope
        reported = []
        report = notifications.report_outcome

        def _counting(ex, *args, **kwargs):
            reported.append(ex)
            return report(ex, *args, **kwargs)

        monkeypatch.setattr(notifications, 'report_outcome', _counting)

        failure = RuntimeError('run_ended failed')

        def _raising(*_args):
            raise failure

        run = _scan(runner, runs, RunEvents(run_ended=_raising))

        assert run.result(BOUND_S).status == 'completed'
        assert reported.count(failure) == 1, reported
        assert not [ex for ex in reported if isinstance(ex, RunCleanupFailedError)], reported


class TestAWaitOnTheUiThread:
    def test_an_ended_run_answers(self, scope, deferred):
        """A wait refused on the UI thread is one for an operation not yet
        done (tests/test_operation.py); one done answers there."""
        runner, runs = scope
        run = _scan(runner, runs, RunEvents())
        run.files.result(BOUND_S)

        for wait in (run.result, run.files.result):
            how, answer = deferred.call(lambda wait=wait: wait(BOUND_S))
            assert how == 'returned' and answer is not None, (how, answer)


class TestTheFilesSettleWhenCleanupRaised:
    def test_a_cleanup_that_raised_before_the_build_still_settles_the_files(
        self, scope, monkeypatch
    ):
        import modules.sequenced_capture_runner as scr

        runner, runs = scope

        def _raises(**_kwargs):
            raise RuntimeError('cleanup raised')

        # A stand-in by design: a cleanup step that raises, which the
        # simulator cannot make fail on its own.
        monkeypatch.setattr(scr, 'run_cleanup', _raises)
        run = _scan(runner, runs, RunEvents())

        assert run.result(BOUND_S) is not None
        assert run.files.result(BOUND_S) is not None, "the run's files never settled"


def test_the_stand_in_dispatcher_delivers_on_its_own_thread(deferred):
    seen = []
    kivy_utils.schedule_ui(lambda dt: seen.append(threading.current_thread()))
    deadline = time.monotonic() + BOUND_S
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == [deferred.thread]
