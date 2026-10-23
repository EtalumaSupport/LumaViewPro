# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An ``Operation`` is a running Future whose done-callbacks run on the one callback thread.

The clauses of ``modules/operation.py``: it is running from construction,
so ``cancel()`` is False; its done-callbacks run on the callback thread,
one at a time in the order their operations settled, never on the thread
that settled it, so a callback may wait on another operation the same
thread settles next; ``stop()`` stops the operation's work while it is
under way and does nothing once it is done or when it has no stop; inside
its producer's ``settling`` delivery, the delivering thread's ``result()``,
``exception()`` and ``done()`` answer at once; a wait on the UI thread for
an operation not done is refused ``wait_on_ui_thread``, and one done
answers; a holder's ``set_result`` or ``set_exception`` is refused, so only
its producer settles it, through the base class.
"""

import asyncio
import concurrent.futures as cf
import queue
import threading

import pytest

from modules import kivy_utils
from modules.exceptions import BlockingCallRefusedError
from modules.notification_center import notifications
from modules.operation import Operation, settling
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
        cf.Future.set_result(op, 7)
        assert op.result(BOUND_S) == 7

    def test_the_standard_waits_accept_it(self):
        op = Operation()
        cf.Future.set_exception(op, ValueError('ended'))
        done, _ = cf.wait([op], timeout=BOUND_S)
        assert done == {op}
        assert isinstance(op.exception(BOUND_S), ValueError)

        async def _awaited():
            return await asyncio.wrap_future(other)

        other = Operation()
        cf.Future.set_result(other, 'answered')
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
        cf.Future.set_result(op, None)
        settled_on = threading.current_thread()

        late = Operation()
        cf.Future.set_result(late, None)
        late.add_done_callback(lambda _op: ran_on.put(threading.current_thread()))

        first, second = ran_on.get(timeout=BOUND_S), ran_on.get(timeout=BOUND_S)
        assert first is not settled_on and second is not settled_on
        assert first is second, 'one callback thread'

    def test_a_callback_may_wait_on_an_operation_its_settler_settles_next(self):
        first, second = Operation(), Operation()
        answered = queue.Queue()
        first.add_done_callback(lambda _op: answered.put(second.result(BOUND_S)))

        settler = threading.Thread(
            target=lambda: (
                cf.Future.set_result(first, None),
                cf.Future.set_result(second, 'next'),
            ),
            daemon=True,
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
            cf.Future.set_result(op, None)
        assert order == [], 'a later callback ran while the first was still running'
        released.set()

        fence = Operation()
        drained = threading.Event()
        fence.add_done_callback(lambda _op: drained.set())
        cf.Future.set_result(fence, None)
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
        cf.Future.set_result(op, None)

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
        cf.Future.set_result(op, None)
        op.stop()
        assert not stopped.is_set()

    def test_stop_on_an_operation_with_no_stop_does_nothing(self):
        op = Operation()
        op.stop()
        assert op.running()


class TestOneForSeveral:
    """``Operation.all_of``: one operation for several started together, as
    ``start_go_to_step`` hands out its X, Y and Z moves."""

    def test_it_is_done_once_every_one_is_and_raises_the_first_ones_raise(self):
        x, y, z = Operation(), Operation(), Operation()
        joined = Operation.all_of((x, y, z))

        cf.Future.set_exception(z, ValueError('z'))
        cf.Future.set_result(x, None)
        assert not joined.done()
        cf.Future.set_exception(y, ValueError('y'))

        assert str(joined.exception(BOUND_S)) == 'y'

    def test_a_done_callback_may_wait_on_it(self):
        x, y = Operation(), Operation()
        joined = Operation.all_of((x, y))
        answered = queue.Queue()
        x.add_done_callback(lambda _op: answered.put(joined.exception(BOUND_S)))

        cf.Future.set_result(x, None)
        cf.Future.set_result(y, None)

        assert answered.get(timeout=BOUND_S) is None

    def test_stop_stops_each_one_still_under_way(self):
        stopped = []
        x = Operation(stop=lambda: stopped.append('x'))
        y = Operation(stop=lambda: stopped.append('y'))
        joined = Operation.all_of((x, y))
        cf.Future.set_result(x, None)

        joined.stop()

        assert stopped == ['y']

    def test_with_none_it_is_done_at_once(self):
        assert Operation.all_of(()).result(0) is None


class TestFollowingALaneWaiter:
    """``Operation.following``: a lane task's waiter settles it, after the
    producer's release, so whoever reads the outcome finds it let go."""

    def test_the_release_runs_before_it_is_done_and_the_outcome_follows(self):
        waiter = cf.Future()
        waiter.set_running_or_notify_cancel()
        done_at_release = []
        op = Operation.following(waiter, first=lambda: done_at_release.append(op.done()))

        waiter.set_result(['a.tiff'])

        assert done_at_release == [False]
        assert op.result(0) == ['a.tiff']

    def test_a_raise_the_release_meets_still_settles_it_with_the_tasks_outcome(self):
        waiter = cf.Future()
        waiter.set_running_or_notify_cancel()

        def release():
            raise RuntimeError('release')

        op = Operation.following(waiter, first=release)
        waiter.set_exception(cf.CancelledError())

        assert isinstance(op.exception(0), cf.CancelledError)


class TestInsideItsOwnDelivery:
    def test_the_delivering_thread_reads_the_outcome_at_once_and_others_wait_for_the_end(self):
        op = Operation()
        inside, outside = {}, queue.Queue()

        with settling(op, result='outcome'):
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
        with settling(op, exception=ended):
            assert op.exception(timeout=0) is ended
            with pytest.raises(ValueError):
                op.result(timeout=0)
        assert op.exception(0) is ended

    def test_it_settles_even_when_the_delivery_raises(self):
        op = Operation()
        with pytest.raises(RuntimeError), settling(op, result='outcome'):
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
        cf.Future.set_result(op, 'answered')
        assert ui.call(lambda: op.result(BOUND_S)) == ('returned', 'answered')

    def test_inside_its_own_delivery_answers(self, ui):
        op = Operation()

        def _deliver():
            with settling(op, result='outcome'):
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
        cf.Future.set_result(op, None)
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
        cf.Future.set_result(op, None)
        closer = threading.Thread(target=session.shutdown, daemon=True)
        closer.start()
        closer.join(BOUND_S)

        assert not closer.is_alive(), 'the close waited on a callback that waits on the close'
        assert callback_returned.wait(BOUND_S), "the callback's own shutdown never returned"


def test_a_holder_cannot_settle_a_run_a_recording_or_a_move_it_was_handed(tmp_path):
    """Only its producer settles an operation: a holder's settle is refused, and the producer's outcome stands.

    A forged settle would also make the producer's own settle raise, on the
    tick, the finish thread or the motion monitor that makes it.
    """
    from modules.exceptions import OperationSettleRefusedError
    from modules.run_events import RunEvents
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings

    live = tmp_path / 'live'
    live.mkdir()
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    ending = threading.Event()
    release = threading.Event()

    def run_ended(*_args):
        ending.set()
        assert release.wait(BOUND_S)

    def refusals(op) -> list[str]:
        reasons = []
        for settle in (
            lambda: op.set_result('forged'),
            lambda: op.set_exception(RuntimeError('forged')),
        ):
            with pytest.raises(OperationSettleRefusedError) as refused:
                settle()
            reasons.append(refused.value.reason)
        return reasons

    try:
        home_sim_scope(session.scope)
        session.set_channel_acquire('BF', 'image')
        protocol = session.create_empty_protocol()
        session.add_step(protocol)
        run = session.create_protocol_runner().start_single_scan(
            protocol=protocol, enable_image_saving=False, events=RunEvents(run_ended=run_ended)
        )
        assert ending.wait(BOUND_S)
        on_the_run = refusals(run)
        release.set()
        outcome = run.result(timeout=BOUND_S)

        recording = session.manual_recording.start()
        on_the_recording = refusals(recording)
        recording.stop()
        ended_with = recording.result(timeout=BOUND_S)

        move = session.scope.motion.start_move_z_by(10.0)
        on_the_move = refusals(move)
        moved = move.result(timeout=BOUND_S)
    finally:
        release.set()
        session.shutdown()

    assert on_the_run == on_the_recording == on_the_move == ['settled_by_its_producer'] * 2
    assert outcome.status == 'completed'
    assert ended_with != 'forged'
    assert moved is None
