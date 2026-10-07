# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's waits return only after the callbacks they name have run.

``handle.wait()`` returned at the release while the run's ``run_complete``
was still to run, and ``handle.wait_for_files()`` returned when the last
write landed while the lost-image report, the record's reconcile and
``files_complete`` were still to run. A caller that waits and then reads
what its own callback did read it before the callback had done it.

Driven on the real engine with simulated hardware, headless (callbacks
delivered inline on the thread that ends the run) and under a stand-in for
the GUI's dispatcher, which delivers on its own thread.
"""

import queue
import threading
import time

import pytest

from modules import kivy_utils
from modules.exceptions import RunCleanupFailedError, RunWaitOnUiThreadError
from modules.protocol import Protocol
from modules.scope_session import ScopeSession
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

BOUND_S = 30.0
SLOW_S = 0.5


def _scan(runner, parent, callbacks, *, save=True, sequence_name=None):
    kwargs = {} if sequence_name is None else {'sequence_name': sequence_name}
    return runner.run_single_scan(
        enable_image_saving=save,
        protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
        parent_dir=str(parent),
        callbacks=callbacks,
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
    def _callback(**_kwargs):
        seen[f'{name}_thread'] = threading.current_thread()
        time.sleep(SLOW_S)
        seen[f'{name}_ended'] = time.monotonic()

    return _callback


class TestAWaitIsNotWokenBeforeItsCallback:
    def test_wait_returns_after_a_slow_run_complete(self, scope):
        runner, runs = scope
        seen = {}
        handle = _scan(runner, runs, {'run_complete': _slow(seen, 'run_complete')})

        outcome = handle.wait(BOUND_S)

        assert outcome is not None and outcome.status == 'completed', outcome
        assert 'run_complete_ended' in seen, 'wait() returned while run_complete was still running'

    def test_wait_for_files_returns_after_a_slow_files_complete(self, scope):
        runner, runs = scope
        seen = {}
        handle = _scan(runner, runs, {'files_complete': _slow(seen, 'files_complete')})

        files = handle.wait_for_files(BOUND_S)

        assert files is not None and files.outcome == 'written', files
        assert 'files_complete_ended' in seen, (
            'wait_for_files() returned while files_complete was still running'
        )

    def test_deferred_wait_for_files_returns_after_run_complete(self, scope, deferred):
        # Under a deferring dispatcher a run with no files_complete has its
        # files told at once, while run_complete still waits its turn.
        runner, runs = scope
        seen = {}
        handle = _scan(runner, runs, {'run_complete': _slow(seen, 'run_complete')})

        files = handle.wait_for_files(BOUND_S)

        assert files is not None, 'wait_for_files() timed out'
        assert seen.get('run_complete_thread') is deferred.thread
        assert 'run_complete_ended' in seen, (
            'wait_for_files() returned while run_complete was still running'
        )


def _waits_on_its_own_run(answers, handle_box):
    def _callback(**_kwargs):
        handle = handle_box['handle']
        started = time.monotonic()
        answers['wait'] = handle.wait(BOUND_S)
        answers['wait_for_files'] = handle.wait_for_files(BOUND_S)
        answers['took_s'] = time.monotonic() - started

    return _callback


def _handle_box_ready(handle_box, handle):
    handle_box['handle'] = handle
    handle_box['ready'].set()


class TestACallbackThatWaitsOnItsOwnRun:
    @pytest.mark.parametrize('dispatch', ['inline', 'deferred'])
    def test_returns(self, scope, dispatch, request):
        if dispatch == 'deferred':
            request.getfixturevalue('deferred')
        runner, runs = scope
        answers = {}
        handle_box = {'ready': threading.Event()}
        callback = _waits_on_its_own_run(answers, handle_box)

        def _run_complete(**kwargs):
            # The callback may be delivered before start() has returned the
            # handle to this test; it waits for it like any caller would.
            assert handle_box['ready'].wait(BOUND_S)
            callback(**kwargs)

        _handle_box_ready(handle_box, _scan(runner, runs, {'run_complete': _run_complete}))

        assert handle_box['handle'].wait(BOUND_S) is not None
        assert answers['wait'] is not None and answers['wait'].status == 'completed', answers
        assert answers['wait_for_files'] is not None, answers
        assert answers['took_s'] < BOUND_S / 2, answers

    def test_returns_with_a_failed_start_delivered_inside_it(self, scope, monkeypatch):
        # A's run_complete starts B, which fails after it commits (its
        # protocol copy cannot be written): B's cleanup runs inline, and its
        # run_complete is delivered inside A's on A's thread. B's delivery
        # ending must not make A's wait wait on A's own delivery. A saves
        # nothing, so B is not refused for A's files first.
        runner, runs = scope
        answers = {}
        handle_box = {'ready': threading.Event()}
        nested = {}

        def _copy_refused(self, **kwargs):
            raise OSError('No space left on device')

        def _b_run_complete(**kwargs):
            nested['thread'] = threading.current_thread()
            nested['inside_a'] = 'a_thread' in nested and 'a_left' not in nested

        def _a_run_complete(**kwargs):
            assert handle_box['ready'].wait(BOUND_S)
            nested['a_thread'] = threading.current_thread()
            monkeypatch.setattr(Protocol, 'to_file', _copy_refused)
            try:
                _scan(runner, runs, {'run_complete': _b_run_complete}, sequence_name='b')
            except Exception as ex:
                nested['b_start'] = ex
            _waits_on_its_own_run(answers, handle_box)(**kwargs)
            nested['a_left'] = True

        _handle_box_ready(
            handle_box, _scan(runner, runs, {'run_complete': _a_run_complete}, save=False)
        )

        assert handle_box['handle'].wait(BOUND_S) is not None
        assert nested.get('inside_a') and nested['thread'] is nested['a_thread'], (
            "the failed start's run_complete was not delivered inside A's: "
            f'the shape under test is absent ({nested})'
        )
        assert answers['wait'] is not None, answers
        assert answers['wait_for_files'] is not None, answers
        assert answers['took_s'] < BOUND_S / 2, answers


class TestARaisingRunComplete:
    def test_still_releases_wait_and_is_reported_once(self, scope, monkeypatch):
        from modules.notification_center import notifications

        runner, runs = scope
        reported = []
        report = notifications.report_outcome

        def _counting(ex, *args, **kwargs):
            reported.append(ex)
            return report(ex, *args, **kwargs)

        monkeypatch.setattr(notifications, 'report_outcome', _counting)

        def _raising(**_kwargs):
            raise RuntimeError('run_complete failed')

        handle = _scan(runner, runs, {'run_complete': _raising})

        assert handle.wait(BOUND_S) is not None, 'a raising run_complete held wait() past its bound'
        failures = [ex for ex in reported if isinstance(ex, RunCleanupFailedError)]
        assert len(failures) == 1, reported


class TestAWaitOnTheUiThread:
    def test_outside_any_delivery_is_refused(self, scope, deferred):
        runner, runs = scope
        handle = _scan(runner, runs, {})
        assert handle.wait_for_files(BOUND_S) is not None

        for wait in (handle.wait, handle.wait_for_files):
            how, answer = deferred.call(lambda wait=wait: wait(BOUND_S))
            assert how == 'raised' and isinstance(answer, RunWaitOnUiThreadError), (how, answer)


def test_the_stand_in_dispatcher_delivers_on_its_own_thread(deferred):
    seen = []
    kivy_utils.schedule_ui(lambda dt: seen.append(threading.current_thread()))
    deadline = time.monotonic() + BOUND_S
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == [deferred.thread]
