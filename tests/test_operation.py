# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An ``Operation`` is a running Future whose done-callbacks run on the one callback thread.

The clauses of ``modules/operation.py``: it is running from construction,
so ``cancel()`` is False; its done-callbacks run on the callback thread,
one at a time in the order their operations settled, never on the thread
that settled it, so a callback may wait on another operation the same
thread settles next; ``stop()`` stops the operation's work while it is
under way and does nothing once it is done or when it has no stop; inside
its own ``settling`` delivery, the delivering thread's ``result()``,
``exception()`` and ``done()`` answer at once; a wait on the UI thread for
an operation not done is refused ``wait_on_ui_thread``, and one done
answers.
"""

import asyncio
import concurrent.futures as cf
import queue
import threading

import pytest

from modules import kivy_utils
from modules.exceptions import BlockingCallRefusedError
from modules.notification_center import notifications
from modules.operation import Operation
from modules.scope_session import ScopeSession
from tests.test_a_wait_returns_after_its_own_callbacks import _DeferringDispatcher

# Only a failure waits this long.
BOUND_S = 10.0


@pytest.fixture
def ui():
    stand_in = _DeferringDispatcher()
    ScopeSession.set_ui_dispatcher(
        kivy_utils.UiDispatcher(schedule=stand_in.schedule, thread=stand_in.thread)
    )
    yield stand_in
    ScopeSession.set_ui_dispatcher(None)
    stand_in.stop()


class TestItIsAFuture:
    def test_it_is_running_from_construction_so_it_cannot_be_cancelled(self):
        op = Operation()
        assert op.running()
        assert op.cancel() is False
        op.set_result(7)
        assert op.result(BOUND_S) == 7

    def test_the_standard_waits_accept_it(self):
        op = Operation()
        op.set_exception(ValueError('ended'))
        done, _ = cf.wait([op], timeout=BOUND_S)
        assert done == {op}
        assert isinstance(op.exception(BOUND_S), ValueError)

        async def _awaited():
            return await asyncio.wrap_future(other)

        other = Operation()
        other.set_result('answered')
        assert asyncio.run(_awaited()) == 'answered'

    def test_a_caller_s_expired_wait_leaves_it_running(self):
        op = Operation()
        with pytest.raises(cf.TimeoutError):
            op.result(timeout=0.01)
        assert op.running()


class TestItsDoneCallbacks:
    def test_they_run_on_the_callback_thread_never_the_settling_one(self):
        ran_on = queue.Queue()
        op = Operation()
        op.add_done_callback(lambda _op: ran_on.put(threading.current_thread()))
        op.set_result(None)
        settled_on = threading.current_thread()

        late = Operation()
        late.set_result(None)
        late.add_done_callback(lambda _op: ran_on.put(threading.current_thread()))

        first, second = ran_on.get(timeout=BOUND_S), ran_on.get(timeout=BOUND_S)
        assert first is not settled_on and second is not settled_on
        assert first is second, 'one callback thread'

    def test_a_callback_may_wait_on_an_operation_its_settler_settles_next(self):
        first, second = Operation(), Operation()
        answered = queue.Queue()
        first.add_done_callback(lambda _op: answered.put(second.result(BOUND_S)))

        settler = threading.Thread(
            target=lambda: (first.set_result(None), second.set_result('next')), daemon=True
        )
        settler.start()
        settler.join(BOUND_S)

        assert not settler.is_alive(), 'the settler was held by a callback that waits on it'
        assert answered.get(timeout=BOUND_S) == 'next'

    def test_they_run_one_at_a_time_in_the_order_their_operations_settled(self):
        order, released = [], threading.Event()
        ops = [Operation() for _ in range(3)]
        ops[0].add_done_callback(lambda _op: (released.wait(BOUND_S), order.append(0)))
        ops[1].add_done_callback(lambda _op: order.append(1))
        ops[2].add_done_callback(lambda _op: order.append(2))
        for op in ops:
            op.set_result(None)
        assert order == [], 'a later callback ran while the first was still running'
        released.set()

        fence = Operation()
        drained = threading.Event()
        fence.add_done_callback(lambda _op: drained.set())
        fence.set_result(None)
        assert drained.wait(BOUND_S)
        assert order == [0, 1, 2]

    def test_a_raising_callback_is_reported_once_and_the_next_still_runs(self, monkeypatch):
        reported = []
        report = notifications.report_outcome

        def _counting(ex, *args, **kwargs):
            reported.append(ex)
            return report(ex, *args, **kwargs)

        monkeypatch.setattr(notifications, 'report_outcome', _counting)
        failure = RuntimeError('a callback raised')
        ran = threading.Event()

        def _raising(_op):
            raise failure

        op = Operation()
        op.add_done_callback(_raising)
        op.add_done_callback(lambda _op: ran.set())
        op.set_result(None)

        assert ran.wait(BOUND_S)
        assert reported.count(failure) == 1, reported


class TestItsStop:
    def test_stop_while_under_way_stops_its_work(self):
        stopped = threading.Event()
        op = Operation(stop=stopped.set)
        op.stop()
        assert stopped.is_set()

    def test_stop_once_done_does_nothing(self):
        stopped = threading.Event()
        op = Operation(stop=stopped.set)
        op.set_result(None)
        op.stop()
        assert not stopped.is_set()

    def test_stop_on_an_operation_with_no_stop_does_nothing(self):
        op = Operation()
        op.stop()
        assert op.running()


class TestInsideItsOwnDelivery:
    def test_the_delivering_thread_reads_the_outcome_at_once_and_others_wait_for_the_end(self):
        op = Operation()
        inside, outside = {}, queue.Queue()

        with op.settling(result='outcome'):
            inside['done'] = op.done()
            inside['result'] = op.result(timeout=0)
            inside['exception'] = op.exception(timeout=0)
            reader = threading.Thread(target=lambda: outside.put(op.done()), daemon=True)
            reader.start()
            reader.join(BOUND_S)

        assert inside == {'done': True, 'result': 'outcome', 'exception': None}
        assert outside.get(timeout=BOUND_S) is False, 'another thread saw it done mid-delivery'
        assert op.done() and op.result(0) == 'outcome'

    def test_a_raised_outcome_reads_as_raised_inside(self):
        op = Operation()
        ended = ValueError('it failed')
        with op.settling(exception=ended):
            assert op.exception(timeout=0) is ended
            with pytest.raises(ValueError):
                op.result(timeout=0)
        assert op.exception(0) is ended

    def test_it_settles_even_when_the_delivery_raises(self):
        op = Operation()
        with pytest.raises(RuntimeError), op.settling(result='outcome'):
            raise RuntimeError('a handler raised')
        assert op.result(0) == 'outcome'


class TestAWaitOnTheUiThread:
    def test_one_not_done_is_refused(self, ui):
        op = Operation()
        for wait in (op.result, op.exception):
            how, answer = ui.call(lambda wait=wait: wait(BOUND_S))
            assert how == 'raised' and isinstance(answer, BlockingCallRefusedError), (how, answer)
            assert answer.reason == 'wait_on_ui_thread'
        assert op.running(), 'the refusal left the operation running'

    def test_one_done_answers(self, ui):
        op = Operation()
        op.set_result('answered')
        assert ui.call(lambda: op.result(BOUND_S)) == ('returned', 'answered')

    def test_inside_its_own_delivery_answers(self, ui):
        op = Operation()

        def _deliver():
            with op.settling(result='outcome'):
                return op.result(BOUND_S)

        assert ui.call(_deliver) == ('returned', 'outcome')


@pytest.fixture
def session():
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(microscope='LS850T'), simulate=True, warn_pre_release=False
    )
    yield s
    s.shutdown()


class TestTheSessionsClose:
    def test_it_runs_every_queued_callback_and_joins_the_thread(self, session):
        threads, released = [], threading.Event()
        op = Operation()
        op.add_done_callback(
            lambda _op: (released.wait(BOUND_S), threads.append(threading.current_thread()))
        )
        op.set_result(None)
        threading.Timer(0.2, released.set).start()

        session.shutdown()

        assert len(threads) == 1, 'shutdown returned before a queued callback ran'
        assert not threads[0].is_alive(), 'shutdown left the callback thread running'

    @pytest.mark.slow
    def test_a_callback_that_closes_the_session_during_its_close_does_not_hold_it(self, session):
        callback_returned = threading.Event()

        def _close_too(_op):
            # Asked once the close has begun, so this call waits on that close.
            while not session.activity_claim.closing:
                threading.Event().wait(0.01)
            session.shutdown()
            callback_returned.set()

        op = Operation()
        op.add_done_callback(_close_too)
        op.set_result(None)
        closer = threading.Thread(target=session.shutdown, daemon=True)
        closer.start()
        closer.join(BOUND_S)

        assert not closer.is_alive(), 'the close waited on a callback that waits on the close'
        assert callback_returned.wait(BOUND_S), "the callback's own shutdown never returned"
