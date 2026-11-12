# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# Executors must be GUI-agnostic. No Kivy imports here.
# UI callbacks are dispatched via ui_queue.schedule_ui(), the process's
# one UI dispatcher, read at the moment a callback is dispatched; with
# none set (tests, headless, REST) a callback runs on the worker.

import concurrent.futures
from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError
import contextlib
import heapq
import itertools
import queue
from collections.abc import Callable, Iterator, Sequence
from _lumascope.lvp_logger import logger
from _lumascope.lib import profile_trace
from _lumascope.modules.ui_queue import schedule_ui
from _lumascope.modules.notification_center import notifications
from _lumascope.modules.activity_claim import (
    ActivityClaim,
    Taking,
    acting,
    activity_taking,
    command_refused,
    current_taking,
)
from _lumascope.modules.exceptions import (
    BlockingCallRefusedError,
    ControlSessionRefusedError,
    HardwareCommandRefusedError,
    LaneCallStillRunningError,
    Refusal,
)
import threading
import time


# IOTask priority constants. Lower value runs first. Only honored by
# priority_aware executors; FIFO executors ignore the field.
PRIORITY_HIGH = 0
PRIORITY_MED = 1
PRIORITY_LOW = 2

# Threading audit -- per-IOTask queue-wait + exec-time instrumentation.
# Opt-in via profile_trace_enabled in settings.json (same gate as
# serial_trace / motion_trace / frame_validity_trace). Zero overhead when
# disabled -- every timestamp site is guarded by
# profile_trace.ENABLE_PROFILE_TRACE.
_IOTASK_TRACE_HEADER = (
    'ts_ms,duration_ms,executor,task_name,action,queue_kind,'
    'queue_depth_at_enqueue,queue_wait_ms,exec_ms,exception'
)

# Sentinel returned from protocol_put when a fire-and-forget task (return_future
# False) DID enter the queue. Distinct from None: a no-future enqueue used to
# return None too, so a caller could not tell a successful enqueue from a
# dropped one (protocol-not-running returns None). Callers that
# gate later work on "did this task actually run?" check `is PROTOCOL_ENQUEUED`.
PROTOCOL_ENQUEUED = object()
# Sentinel returned by put() when a frame-carrying (droppable_live) task is
# dropped because too many are already in flight on the single worker.
LIVE_FRAME_DROPPED = object()
# Sentinel returned from put() when a fire-and-forget task (return_future False)
# DID enter the queue -- the default queue's counterpart to PROTOCOL_ENQUEUED,
# and for the same reason: a no-future enqueue returned None, which is also what
# a fenced or shut executor returns, so a successful submit was reported to
# its caller as a drop. Callers that must know whether the task will run check
# `is ENQUEUED`.
ENQUEUED = object()
# Refusal-episode lanes. A shut or fenced executor refuses every submit for
# as long as the state lasts -- a protocol run can refuse tens of thousands --
# so refusals are narrated per EPISODE (one line when the first task is lost,
# one summary when work is accepted again) instead of per task. The two queues
# refuse independently: during a run the default lane is fenced while the
# protocol lane accepts, so they interleave, and a single shared tracker would
# open and close an episode on every submit -- reproducing the per-task logging
# the episode framing exists to remove. One tracker per lane, closed only by a
# success on its OWN lane.
_LANE_DEFAULT = 'default'
_LANE_PROTOCOL = 'protocol'

# Max in-flight frame-carrying (droppable_live) tasks on the default queue
# before new frames are dropped (latest-wins). Bounds the live/record image
# backlog so a stalled single worker can't pin GBs of ~3.5 MB frame buffers
# (the manual-record balloon). Config / motor / save tasks are NOT droppable
# and stay unbounded. ~16 frames ~= 58 MB ceiling.
_LIVE_FRAME_MAXSIZE = 16


class _ReusableTaskWaiter:
    """Future-shim with the subset of concurrent.futures.Future API that
    LVP _sync callers + the executor cleanup path need: result(),
    set_result(), set_exception(). Wraps a threading.Event so wait
    semantics are identical.

    Reusable via reset() so a single instance can serve many sequential
    submissions from the same thread, dropping Lock-kernel-handle
    allocation pressure from O(submissions) to O(threads). Same kernel
    object class as Future's internal Condition.Lock (Semaphore on
    Windows) -- just allocated once and reused, instead of churned per
    call. This is the fix for the Windows kernel-handle leak observed
    during multi-hour protocol runs (Semaphore handles climbed
    ~78/min despite caller_futures cleanup being clean).

    Not thread-safe across CONCURRENT use of the same waiter -- the
    executor pairs each task with one waiter; the caller blocks on
    result() until the executor sets it; same-thread sequential reuse
    is safe via is_spent() / reset().
    """

    def __init__(self):
        self._event = threading.Event()
        self._result = None
        self._exception = None

    def reset(self) -> None:
        self._event.clear()
        self._result = None
        self._exception = None

    def is_spent(self) -> bool:
        """True if a result or exception has been set since the last reset.
        A spent waiter is safe to reset and hand to the next caller; an
        unspent waiter is still in-flight (set_result has not run yet)
        and must not be reused."""
        return self._event.is_set()

    def set_result(self, value) -> None:
        self._result = value
        self._event.set()

    def set_exception(self, exc) -> None:
        self._exception = exc
        self._event.set()

    def result(self, timeout: float | None = None) -> object:
        """The task's result, or its exception raised; read once, by the caller it was handed to.

        Taken out as it is handed over: the waiter is cached on its thread
        until that thread's next lane call, and a failure kept here would
        keep every frame of its traceback alive that long.
        """
        if not self._event.wait(timeout):
            from concurrent.futures import TimeoutError as _TimeoutError

            raise _TimeoutError(f'task did not complete within {timeout}s')
        exception, self._exception = self._exception, None
        result, self._result = self._result, None
        if exception is not None:
            try:
                raise exception
            finally:
                # Not left in this frame, which its own traceback holds.
                exception = None
        return result


# Which lane's worker this thread is, set by the worker as it starts. A lane
# worker never waits on another lane: that wait is one half of the pair that
# makes a deadlock. On its own lane it is already where the work belongs, so
# the work runs inline rather than waiting on the queue it is draining.
_lane_worker = threading.local()

# Which executor's worker this thread is, lane or not, set by the worker as
# it starts. Asked by a member whose own work is queued on an executor and
# waited for: called from a worker, that wait can be on the queue the worker
# is draining, or on a lane its caller is holding.
_executor_worker = threading.local()


def refuse_blocking_on_a_worker(member: str) -> None:
    """Refuse a blocking call from an executor's worker, before any wait is made.

    Raises:
        BlockingCallRefusedError: ``'blocking_on_worker'``, this thread is
            an executor's worker.
    """
    worker = getattr(_executor_worker, 'executor', None)
    if worker is not None:
        raise BlockingCallRefusedError('blocking_on_worker', member, worker=worker.executor_name)


# Set while the GUI's inline boundary runs a call on its own thread. That
# thread draws the window, so a call that waits on a lane there freezes the
# window for as long as the lane takes; it has to be submitted to the worker
# pool instead. Asked where a wait is about to be made, so a blocking member
# handed to the inline form fails at once, loudly and by name.
_inline_outcome = threading.local()


@contextlib.contextmanager
def inline_outcome() -> Iterator[None]:
    """Mark the calling thread as running a call that must not wait on a lane."""
    previous = getattr(_inline_outcome, 'active', False)
    _inline_outcome.active = True
    try:
        yield
    finally:
        _inline_outcome.active = previous


# One order every executor stamps its tasks in, at submission: a task asked
# later carries a larger number than every task asked before it, whichever
# executor each was put on. Work a task submits while it runs carries the
# running task's number, so a click that queues a task which later asks
# another lane is still asked at the click. Motion reads it to withhold
# work asked before a STOP.
_ask_order = itertools.count(1)
_asking = threading.local()


def ask_now() -> int:
    """A mark later than every ask made before this call, in the one ask order."""
    return next(_ask_order)


def current_ask() -> int:
    """The ask the calling thread's work is made at: the task it runs, else now."""
    running = getattr(_asking, 'ask', None)
    return running if running is not None else ask_now()


def asked_before(mark: int) -> bool:
    """Whether the task this thread runs was asked before ``mark``.

    False on a thread running no executor's task: its asks are made now.
    """
    ask = getattr(_asking, 'ask', None)
    return ask is not None and ask < mark


@contextlib.contextmanager
def _running_ask(ask: int | None) -> Iterator[None]:
    """Run under ``ask`` on this thread for the length of a ``with`` block; nests."""
    previous = getattr(_asking, 'ask', None)
    _asking.ask = ask
    try:
        yield
    finally:
        _asking.ask = previous


@contextlib.contextmanager
def asked_as_made() -> Iterator[None]:
    """Run with no ask of its own: work made here is asked as it is made."""
    with _running_ask(None):
        yield


@contextlib.contextmanager
def asked_at(ask: int) -> Iterator[None]:
    """Run as asked at ``ask``, a mark from ``ask_now``: work made here is asked then.

    For a thread that is no executor's worker but does one asker's work: a
    press of STOP after ``ask`` withholds every move it makes from then on.
    """
    with _running_ask(ask):
        yield


def refuse_blocking_inline(member: str) -> None:
    """Refuse a blocking call inside the inline boundary, before any wait is made.

    Raises:
        BlockingCallRefusedError: ``'blocking_inline'``.
    """
    if getattr(_inline_outcome, 'active', False):
        raise BlockingCallRefusedError('blocking_inline', member)


# Per-thread waiter cache. Each calling thread that submits with
# return_future=True gets one waiter for its lifetime; the same waiter
# is reset and reused on every subsequent submission from that thread.
# Threads that never call put(return_future=True) (e.g. protocol_put
# fire-and-forget paths) never allocate one.
_waiter_thread_local = threading.local()


def _claim_waiter() -> _ReusableTaskWaiter:
    """Return a reset, ready-to-use waiter for the calling thread.

    Reuses the thread's cached waiter when it's spent (i.e. its
    previous use has completed). Allocates a fresh waiter when the
    cached one is still in-flight (rare: same thread submitting a
    second future before the first completes -- not the normal LVP
    submit-then-result pattern).
    """
    waiter = getattr(_waiter_thread_local, 'waiter', None)
    if waiter is None or not waiter.is_spent():
        waiter = _ReusableTaskWaiter()
        _waiter_thread_local.waiter = waiter
    else:
        waiter.reset()
    return waiter


def _task_named(task: 'IOTask') -> str:
    """The task's action, by name, as the log names who made the task."""
    return getattr(task.action, '__name__', None) or repr(task.action)


def _member_of(task: 'IOTask') -> str:
    """The member an answer about ``task`` names: the one its submit named, else its action."""
    return task.member if task.member is not None else _task_named(task)


# A task's fate on its lane, decided once under the lane's caller-futures
# lock: the worker starting it, or the lane answering it as never run. The
# lock makes the two exclusive, so a task answered as not run is never run,
# and one started is never answered as not run.
_STARTED = 'started'
_UNRUN = 'unrun'


"""
IOTask
- Encapsulates a single unit of work:
    - action:         callable performing the I/O
    - args, kwargs:   parameters for action
    - callback:       optional function to call when done
    - cb_args, cb_kwargs: arguments for callback
    - pass_result:    if True, injects (result, exception) into cb_kwargs
- Usage:
    task = IOTask(
        action=grab_image,
        args=(well_id,),
        kwargs={'exposure_ms':100},
        callback=display_result,
        pass_result=True
    )
    executor.enqueue(task)
"""


SLOW_TASK_BUDGET_ATTR = 'slow_task_budget_s'


def slow_task_budget(seconds: float) -> Callable:
    """Declare, at the def site, how long this command may legitimately run.

    A command that physically takes longer than the default -- full homing, a
    turret move that parks and restores Z, an AF scan -- trips the "Slow task"
    WARNING on every SUCCESS unless something says otherwise. Saying it here
    rather than at the submission site is what makes it hold: an impl is
    reachable through several wrappers (a dispatcher, a hand-built IOTask on
    the UI thread, a protocol-queue submit), and a budget attached to one
    wrapper is silently absent from the others. That is a shipped defect, not
    a hypothetical -- a turret budget declared on the dispatcher never reached
    the GUI path, which warns on every successful move.

    The cost is a property of the command; the wrapper is just how it got
    queued. An explicit ``slow_task_threshold_sec`` still wins, for a caller
    that knows more than the command does about a particular submission.
    """

    def _declare(fn):
        setattr(fn, SLOW_TASK_BUDGET_ATTR, seconds)
        return fn

    return _declare


class IOTask:
    # Default threshold beyond which a task triggers the "Slow task"
    # WARNING log line. Commands that legitimately take longer declare it at
    # their def site with @slow_task_budget; a caller may still override per
    # submission via the `slow_task_threshold_sec` __init__ kwarg.
    DEFAULT_SLOW_TASK_THRESHOLD_SEC = 5.0

    def declared_slow_task_budget(self) -> float | None:
        """What anyone actually said about this task's cost, or None.

        Per-submission value first, then whatever the command declared about
        itself. Kept separate from the resolved threshold because callers
        that must distinguish "nobody declared anything" from "the default
        applies" exist -- the stuck-worker bar raises itself only on a real
        declaration, and folding the default in would move that bar for every
        task in the system.
        """
        if self.slow_task_threshold_sec is not None:
            return self.slow_task_threshold_sec
        return getattr(self.action, SLOW_TASK_BUDGET_ATTR, None)

    def resolve_slow_task_threshold(self) -> float:
        """Seconds this task may run before "Slow task" means anything."""
        declared = self.declared_slow_task_budget()
        return declared if declared is not None else self.DEFAULT_SLOW_TASK_THRESHOLD_SEC

    def __init__(
        self,
        action,
        args=None,
        kwargs=None,
        callback=None,
        cb_args=None,
        cb_kwargs=None,
        pass_result=False,
        slow_task_threshold_sec=None,
        silent_on_failure=False,
        droppable_live: bool = False,
        priority: int = PRIORITY_MED,
        falsifies_recording: bool = False,
    ):
        self.action = action
        # Set by the API member whose write would falsify a recording's
        # file; the lane refuses such a task while a recording holds.
        self.falsifies_recording = falsifies_recording
        self.priority = priority
        # When True, _on_task_done skips the generic "Task failed"
        # notification on exception -- the caller's callback (or its
        # surrounding context) is responsible for user-facing
        # notification. Logs are unaffected. The API/caller decides
        # whether to notify, not the executor.
        self.silent_on_failure = silent_on_failure
        # When True, this task carries a live/preview/record frame that the
        # display or recording can drop when the single worker falls behind
        # (latest-wins). The executor caps in-flight droppable_live tasks so
        # a stalled worker can't pin GBs of frame buffers. Must-execute
        # tasks (config, motor, save) leave this False and are never dropped.
        self.droppable_live = droppable_live
        if args is None:
            self.args = ()
        # if it's a sequence (list, tuple, etc) but not a string
        elif isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
            self.args = tuple(args)
        else:
            self.args = (args,)

        self.kwargs = kwargs if kwargs is not None else {}
        self.callback = callback
        self.protocol = None
        self.name = ''
        # The taking the submitter acted under, stamped by the lane at submit.
        # The lane runs a task while the scope is held only if this is the
        # holder's, and its worker acts under it while the task runs.
        self.taking = None
        # Set only by the lane, for a submitter holding its override key.
        self.override = False
        # Where the task stands in the one ask order, stamped at submit.
        self.asked = None
        # The API member a submit named, for the answers that name it.
        self.member = None
        # None while queued; _STARTED or _UNRUN once the lane decides.
        self.fate = None
        # Set by the lane when this task was put on it with nothing waiting
        # and nothing running: it is next whatever the worker's timing, so a
        # hold that begins before the worker takes it lets it run, as it
        # lets the running task finish.
        self.first_in_line = False

        # Per-task slow threshold. None -> use class default at run-time
        # (allows the class default to be tuned without per-instance
        # surprises). Pass an explicit float to override (e.g. 30.0
        # for tasks expected to take up to ~30 sec under normal load).
        self.slow_task_threshold_sec = slow_task_threshold_sec

        if cb_args is None:
            self.cb_args = ()
        # if it's a sequence (list, tuple, etc) but not a string
        elif isinstance(cb_args, Sequence) and not isinstance(cb_args, (str, bytes)):
            self.cb_args = tuple(cb_args)
        else:
            self.cb_args = (cb_args,)

        self.cb_kwargs = cb_kwargs if cb_kwargs is not None else {}
        self.pass_result = pass_result

    def run(self) -> tuple:
        try:
            threading.current_thread().name = self.name
            if not callable(self.action):
                logger.warning(f'{self.name} Worker received non-callable action: {self.action!s}')
            t_start = time.monotonic()
            res = self.action(*self.args, **self.kwargs)
            elapsed = time.monotonic() - t_start
            threshold = self.resolve_slow_task_threshold()
            if elapsed > threshold:
                action_name = getattr(self.action, '__name__', str(self.action))
                logger.warning(
                    f'[IOTask    ] Slow task ({elapsed:.1f}s, threshold {threshold:.1f}s): {action_name} on {self.name}'
                )
            return res, None
        except CancelledError as e:
            # concurrent.futures.CancelledError subclasses Exception, so the
            # handler below would report a by-contract cancel as a failure.
            # The executor's task epilogue is what logs cancels, at debug.
            return None, e
        except Exception as e:
            # Returned, not logged: where the exception's flight ends decides
            # who reports it -- the executor's epilogue, or the caller that
            # waits and receives it.
            return None, e

    def set_callback(self, callback, cb_args, cb_kwargs):
        self.callback = callback
        self.cb_args = cb_args
        self.cb_kwargs = cb_kwargs if cb_kwargs is not None else {}

    def on_complete(self, result: object, exception: BaseException | None) -> None:
        if self.callback is None:
            return

        def _safe_callback(dt: float) -> None:
            # The task is done and its waiter answered; a callback's raise
            # has no caller left to reach, so it is reported here.
            try:
                self.callback(*cb_args, **cb_kwargs)
            except Exception as ex:
                from _lumascope.modules.notification_center import notifications

                notifications.report_outcome(ex, solicited=False, category='IOTask')

        if self.pass_result:
            # Only copy when we need to mutate
            cb_kwargs = dict(self.cb_kwargs)
            cb_kwargs['result'] = result
            cb_kwargs['exception'] = exception
        else:
            cb_kwargs = self.cb_kwargs

        cb_args = self.cb_args
        schedule_ui(_safe_callback)

    def set_name(self, name):
        self.name = name

    def __call__(self):
        return self.run()

    def __repr__(self):
        return f'<IOTask: Action: {self.action!s} Callback: {self.callback!s}>'


"""
SequentialIOExecutor
- Manages a FIFO queue of IOTask instances.
- Runs them on exactly ONE worker thread (sequential per hardware boundary):
  one ordered command stream per executor, never overlapping. The max_workers
  __init__ argument is retained only for call-signature back-compat and is
  ignored -- widening it would reintroduce the task-retention surface the
  single worker exists to avoid.
- For each task the worker:
    1. Calls task.run() on the worker thread.
    2. Captures (result, exception).
    3. Schedules task.on_complete(result, exception) on the main/UI thread.
- Usage:
    executor = SequentialIOExecutor()
    executor.start()
    executor.put(task)
    # ... later ...
    executor.shutdown(wait=True)
"""


class _WakesOnPut:
    """A lane queue that sets its lane's wake on every put.

    The worker waits on the wake, never inside a get, so that a switch of
    the lane's state wakes it as work does. The wake is set by the queue
    itself so no put can be made that leaves its task waiting for the next
    switch.
    """

    def __init__(self, wake: threading.Event):
        super().__init__()
        self._wake = wake

    def _put(self, item):
        super()._put(item)
        self._wake.set()


class _WakingQueue(_WakesOnPut, queue.Queue):
    pass


class _WakingPriorityQueue(_WakesOnPut, queue.PriorityQueue):
    pass


class _PriorityFifoQueue:
    """PriorityQueue wrapper hiding the tuple wrap/unwrap from callers.
    put(task) / get() -> task have the same shape as queue.Queue; the
    priority field drives ordering, a monotonic counter breaks ties
    FIFO within priority. IOTask stays non-comparable because the
    counter is reached only on an impossible (priority, counter) tie.
    """

    def __init__(self, wake: threading.Event):
        self._q = _WakingPriorityQueue(wake)
        # itertools.count() is thread-safe in CPython; per-instance
        # so two priority_aware executors don't share counter state.
        self._counter = itertools.count()

    def put(self, task):
        self._q.put((task.priority, next(self._counter), task))

    def put_nowait(self, task):
        self._q.put_nowait((task.priority, next(self._counter), task))

    def get(self, block=True, timeout=None):
        _prio, _ctr, task = self._q.get(block=block, timeout=timeout)
        return task

    def get_nowait(self):
        _prio, _ctr, task = self._q.get_nowait()
        return task

    def qsize(self):
        return self._q.qsize()

    def empty(self):
        return self._q.empty()

    def task_done(self):
        self._q.task_done()


class SequentialIOExecutor:
    def __init__(
        self,
        max_workers: int = 1,
        name: str | None = None,
        priority_aware: bool = False,
        lane: bool = True,
        *,
        closed_answer: Callable[[str], BaseException],
    ):
        # What a call this executor will not run is told once it has closed,
        # given the member's name. Stated by the executor's one closer, since
        # what "closed" means is the closer's: the scope disconnected,
        # LumaViewPro closing, the server closing. Read by a submit after the
        # close and for every task the close finds unstarted.
        self._closed_answer = closed_answer
        # priority_aware=True swaps the default queue for a priority
        # wrapper; protocol_queue stays FIFO so step ordering inside
        # a protocol is preserved.
        self.priority_aware = priority_aware
        # What the worker waits on between tasks: set by every put on either
        # queue and by every switch of the lane's state the worker's next
        # decision reads -- a run starting, ending or finishing, the close.
        self._wake = threading.Event()
        if priority_aware:
            self.queue = _PriorityFifoQueue(self._wake)
        else:
            self.queue = _WakingQueue(self._wake)
        self.protocol_queue = _WakingQueue(self._wake)
        # The lane's state -- closing, in a run, a run finishing -- switches
        # under this lock, and every decision that reads that state and then
        # puts a task on a queue, or empties one, is made under it, so no
        # task lands on a queue the state has just stopped serving.
        # Re-entrant: submit holds it across its choice of door and the
        # door it calls. Held for no callback, waiter's settling or blocking
        # get. Taken after the runner's run lock and before each queue's
        # mutex and the caller-futures lock.
        self._state_lock = threading.RLock()
        # Makes "nothing ahead" and the put one step, so two submitters
        # cannot both be put first in line.
        self._first_in_line_lock = threading.Lock()
        # Selective bound for frame-carrying (droppable_live) tasks on the
        # default queue. In-flight count guarded by its own lock; incremented
        # at put(), decremented when the worker dequeues. Must-execute tasks
        # are unaffected -- they never set droppable_live.
        self._live_inflight = 0
        self._live_dropped_count = 0
        self._live_lock = threading.Lock()
        self.protocol_running = threading.Event()
        self.protocol_finish = threading.Event()
        self.name = name
        if name is not None:
            self.executor_name = name + '_' + 'WORKER'
        else:
            self.executor_name = 'WORKER'
        # max_workers retained for signature back-compat; the worker is
        # a single thread by design (sequential per hardware boundary).
        self._max_workers = max_workers
        self._worker_thread = None
        self._running_task_lock = threading.Lock()
        self._running_task = None
        self.pending_shutdown = False
        self._caller_futures_lock = threading.Lock()
        self.caller_futures = {}
        # Monotonic alloc/pop counters paired with caller_futures. Drift
        # between the two means Future entries are leaking; on Windows
        # each leaked Future leaves a Lock/Semaphore kernel handle open.
        # Surfaced by metrics_logger as a permanent invariant.
        self._caller_futures_alloc_count = 0
        self._caller_futures_pop_count = 0

        # None, or {'cause': str, 'count': int} per lane. Opened lazily by the
        # first refused submit, never by the state change that causes it: an
        # executor nobody submits to while it is closed has cost no caller
        # anything and says nothing.
        self._refusal_episodes = {_LANE_DEFAULT: None, _LANE_PROTOCOL: None}
        self._refusal_lock = threading.Lock()

        self.last_task_done_monotonic = time.monotonic()
        # Stamped by the worker when it starts a task, cleared with
        # running_task. Tracked state (not an approximation) so a stall
        # report can say how long the in-flight task has actually run.
        self._running_task_started_monotonic = None
        # Wedge-recovery worker generation. A worker thread captures the
        # generation at entry; recovery bumps it before starting a
        # replacement, so an abandoned worker stuck inside a task exits
        # without touching executor state when its call finally returns.
        self._worker_generation = 0
        # The generation the current worker was started under; a live worker
        # of an older generation is a quarantined one, not the lane's worker.
        self._started_generation = None
        # A lane (IO, CAMERA, FILE) serves one device or store in order; its
        # worker may not wait on a lane. The worker pool is not one: its
        # teardown work waits on the lanes, one way.
        self.lane = lane
        # The activity claim this lane asks before running work, and the key
        # whose holder may submit a named override past it. None until
        # ask_claim: a lane nobody wired to a claim runs what it is given.
        self._claim = None
        self._override_key = None
        # The taking that put this lane in protocol mode. On a lane that asks
        # the claim, the protocol door admits only work under it: the run's
        # queue is the run's, and a lender whose borrowed run is live is not
        # the run.
        self._protocol_taking = None

    @property
    def running_task(self):
        with self._running_task_lock:
            return self._running_task

    def _raise_running_budget(self, inner: 'IOTask') -> None:
        """Give the running task ``inner``'s declared budget when it is larger.

        ``inner`` runs inline inside the running task, so its cost is that
        task's: a GUI task that turns the turret is as slow as the turn. A
        budget is only ever raised, never lowered.
        """
        running = self.running_task
        declared = inner.declared_slow_task_budget()
        if running is None or declared is None:
            return
        if declared > (running.declared_slow_task_budget() or 0.0):
            running.slow_task_threshold_sec = declared

    @property
    def worker_alive(self) -> bool:
        """Whether this lane's worker thread is running.

        False before ``start()`` and after the worker has exited. A
        submission to a lane with no live worker is never serviced, so
        a caller that would wait on one (a drain at teardown) reads
        this first.
        """
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @running_task.setter
    def running_task(self, value):
        with self._running_task_lock:
            self._running_task = value

    def start(self) -> None:
        # One ACTIVE worker per lane is the lane's whole contract: tasks on
        # it run one at a time, in order. A second start while the current
        # worker is alive would put a second worker on the same queue, and
        # nothing would say the ordering was gone. Wedged-queue recovery is
        # the one legitimate replacement: it quarantines the stuck worker by
        # moving the generation on first, so the live thread it leaves
        # behind is no longer this lane's worker.
        if self.worker_alive and self._started_generation == self._worker_generation:
            raise RuntimeError(
                f'{self.executor_name}: already running -- a lane is started once, '
                'by whoever built it'
            )
        # daemon=True so a hung in-flight task at app teardown cannot keep
        # the process alive. Cooperative shutdown is still preferred:
        # long-running task implementations may close over the executor
        # and poll `executor.pending_shutdown` to bail early (the pattern
        # used by protocol_thread.aborted.is_set() in scan_loop).
        self._started_generation = self._worker_generation
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name=self.executor_name,
            daemon=True,
        )
        self._worker_thread.start()

    def ask_claim(self, claim: ActivityClaim) -> object:
        """Make this lane ask ``claim`` before it runs work; returns the override key.

        While a run, a diagnostic or a home holds the scope, a task that was not
        made under the holder's taking is refused -- at submit and again
        when the worker takes it off the queue -- with
        ``HardwareCommandRefusedError``, whoever made it and however, unless
        it was put on the lane first in line, with nothing waiting or
        running ahead of it, which runs as a running task does; while
        a recording holds it, a task marked ``falsifies_recording`` is. The
        key is the one way past that: the composition root keeps it for the
        named overrides, so a caller holding this executor cannot mark its
        own work as one.

        A lane asks one claim for its life. A second session composed over
        the same scope would otherwise re-point the lane at its own claim,
        and the first session's run and diagnostic fences would stop being
        enforced with nothing said.

        Raises:
            RuntimeError: this lane already asks a claim.
        """
        if self._claim is not None:
            raise RuntimeError(
                f'{self.executor_name}: already asks an activity claim -- a second '
                'session over the same scope would take the lane from the first'
            )
        self._claim = claim
        self._override_key = object()
        return self._override_key

    def _claim_refusal(
        self, task: IOTask, *, door: bool = False
    ) -> HardwareCommandRefusedError | ControlSessionRefusedError | None:
        """The refusal the claim gives ``task`` right now, or None.

        With ``door``, the task is entering the protocol queue, which also
        needs the taking that raised protocol mode.
        """
        if self._claim is None or task.override:
            return None
        who = _task_named(task)
        # Work made under a taking that has ended is its activity's leftover
        # -- an autofocus unwind that outlived the run's cleanup -- and is
        # refused even when nothing holds the scope now: the activity that
        # would have wanted it is gone.
        # A control session's taking is no activity's, so it never ends as
        # one: once its session is ended, its work is anyone's.
        activity = activity_taking(task.taking)
        if activity is not None and not activity.holds:
            return HardwareCommandRefusedError('activity_ended', who)
        # Put on the lane ahead of any hold, with nothing before it: it runs
        # as the task already running does, and the holder waits for it. A
        # recording still refuses it in _admit, where a write that would
        # falsify the file is counted.
        if task.first_in_line:
            return None
        refusing = self._claim.refusing_holder(
            task.taking, falsifies_recording=task.falsifies_recording
        )
        if refusing is None:
            if not door or task.taking is self._protocol_taking:
                return None
            refusing = self._claim.holder
        return command_refused(refusing, who)

    def _admit(
        self, task: IOTask
    ) -> tuple[HardwareCommandRefusedError | ControlSessionRefusedError | None, bool]:
        """Ask the claim whether ``task`` may run now, as it is about to run.

        A write that would falsify a recording is admitted through the
        claim's count, where it runs -- the worker, or inline inside another
        task -- because the task a lane reports running may be only the one
        that called it, and a recording must not start while it runs.

        Returns:
            (the refusal or None, whether the claim counted the task; a
            counted task is paired with ``leave_falsifying_change``).
        """
        refusal = self._claim_refusal(task)
        if refusal is not None or not task.falsifies_recording or task.override:
            return refusal, False
        if self._claim is None:
            return None, False
        refusing = self._claim.enter_falsifying_change(task.taking)
        if refusing is not None:
            return command_refused(refusing, _task_named(task)), False
        return None, True

    def _is_protocol_door_holder(self) -> bool:
        """Whether this thread acts under the taking that raised protocol mode."""
        return (
            self._protocol_taking is not None
            and self.protocol_running.is_set()
            and not self.protocol_finish.is_set()
            and current_taking() is self._protocol_taking
        )

    def _stamp(self, task: IOTask, override: object | None) -> None:
        task.set_name(self.executor_name)
        task.fate = None
        task.taking = current_taking()
        task.asked = current_ask()
        task.override = override is not None and override is self._override_key

    def _refused_at_submit(self, task: IOTask, refusal, return_future: bool, waiter=None):
        """Answer a submit the claim refused, as the worker answers a task that failed.

        The waiter carries the refusal and the task's callback is told, so a
        caller that waits, one that passes a callback and one that ignores
        the return all hear it -- a raw put whose return nobody reads gets the
        lane's refusal notice rather than silence. A caller that supplied its
        own waiter gets the refusal back instead, for ``submit`` to raise:
        that waiter has already been handed on as accepted work.
        """
        logger.warning(f'[{self.executor_name}] REFUSED {_task_named(task)} -- {refusal}')
        task.protocol = False
        # The waiter is created below, after the report, so whether there
        # will be one is what return_future says.
        self._report_task_failure(
            task, refusal, owned=task.silent_on_failure and return_future, asked=True
        )
        task.on_complete(None, refusal)
        if return_future and waiter is None:
            fut = _claim_waiter()
            fut.set_exception(refusal)
            return fut
        return refusal

    def call(self, task: IOTask, member: str, timeout_s: float | None) -> object:
        """Run ``task`` on this lane and wait for its result: the blocking dispatch.

        The public hardware members come through here. Refused work raises
        ``HardwareCommandRefusedError`` to the caller rather than returning a
        value a caller could mistake for success; the task reports nothing
        itself, because the caller that waits is the one to report it.

        Called on this lane's own worker -- from a task it is running -- the
        work runs inline on that worker, under the claim the same way, and
        its declared slow-task budget is raised onto the running task, which
        is the one the lane times.

        Raises:
            BlockingCallRefusedError: ``'blocking_on_worker'``, called from
                another lane's worker, which never waits on a lane;
                ``'blocking_inline'``, called inside the GUI's inline
                boundary.
            HardwareCommandRefusedError: ``'scope_disconnected'``, the
                scope's lane closed before the task started, or before the
                call was made (another executor raises the closed answer its
                closer states); ``'exclusive_activity_running'``, the scope
                is held by an activity this call is not made under;
                ``'activity_ended'``, the call is made under an activity
                that has ended, or a run's own call was still queued when
                the run ended.
            ControlSessionRefusedError: ``'control_session_held'``, a control
                session is held and this call is not its holder's.
            LaneCallStillRunningError: ``'call_overran'``, ``timeout_s``, the
                API's own bound, ran out with the task queued or running;
                ``'lane_closed'``, the lane closed while the task ran. Either
                way it may still finish.
        """
        return self._wait_on(task, member, timeout_s, None)

    def call_overriding(
        self, task: IOTask, member: str, timeout_s: float | None, key: object | None
    ) -> object:
        """Run ``task`` on this lane whatever holds the scope, and wait for its result.

        ``call`` for a member the API admits whatever holds the scope: a read
        that has to touch the device from this lane's thread. ``key`` is the
        one ``ask_claim`` returned, which the composition root keeps; None on
        a lane that asks no claim, which refuses nothing of a holder either.
        The task skips the claim, counts no falsifying change, and during a
        run goes through the door the run's own work uses; one still queued
        when the run ends is carried back and served after it.

        Raises:
            BlockingCallRefusedError: ``'blocking_on_worker'``, called from
                another lane's worker, which never waits on a lane;
                ``'blocking_inline'``, called inside the GUI's inline
                boundary.
            HardwareCommandRefusedError: ``'scope_disconnected'``, the
                scope's lane closed before the task started, or before the
                call was made (another executor raises the closed answer its
                closer states).
            LaneCallStillRunningError: ``'call_overran'``, ``timeout_s``, the
                API's own bound, ran out with the task queued or running;
                ``'lane_closed'``, the lane closed while the task ran. Either
                way it may still finish.
        """
        return self._wait_on(task, member, timeout_s, key)

    def _wait_on(
        self, task: IOTask, member: str, timeout_s: float | None, override: object | None
    ) -> object:
        """``call``'s and ``call_overriding``'s one body: submit, or run inline, and wait."""
        worker = getattr(_lane_worker, 'executor', None)
        if worker is self:
            self._stamp(task, override)
            refusal, counted = self._admit(task)
            if refusal is not None:
                raise refusal
            self._raise_running_budget(task)
            try:
                return task.action(*task.args, **task.kwargs)
            finally:
                if counted:
                    self._claim.leave_falsifying_change()
        if worker is not None:
            # A lane worker never waits on another lane: that wait is one
            # half of the pair that makes a deadlock.
            raise BlockingCallRefusedError(
                'blocking_on_worker', member, worker=worker.executor_name
            )
        refuse_blocking_inline(member)
        fut = self.submit(task, member, override=override)
        try:
            return fut.result(timeout=timeout_s)
        except FutureTimeoutError:
            # Take the registration back under the lock the epilogue decides
            # under. Still there: no one will hand this caller the outcome,
            # so a late one is the epilogue's to log. Gone: the epilogue
            # already claimed it for this caller and is completing the
            # future, so the outcome is this caller's after all.
            with self._caller_futures_lock:
                mine = self.caller_futures.pop(task, None)
                if mine is not None:
                    self._caller_futures_pop_count += 1
            if mine is None:
                return fut.result()
            # What the call waited behind, so an overrun reads as a slow
            # device or as a long queue.
            logger.info(
                f'[{self.executor_name}] {member} overran its {timeout_s:g} s with '
                f'{self.queue.qsize()} queued, {self.protocol_queue.qsize()} on the run '
                f'queue, running {self.describe_running_task()}'
            )
            raise LaneCallStillRunningError('call_overran', member, timeout_s) from None

    def submit(
        self,
        task: IOTask,
        member: str,
        *,
        override: object | None = None,
        waiter: concurrent.futures.Future | None = None,
    ) -> '_ReusableTaskWaiter | concurrent.futures.Future':
        """Put ``task`` on this lane without waiting; returns the waiter its outcome settles.

        The submitting half of ``call``: the task is admitted, or refused, on
        the calling thread, so it is on the lane -- or the caller knows it
        never will be -- before this returns. The task reports nothing
        itself; whoever holds the waiter reports its outcome.

        ``waiter`` is a Future the caller supplies, for a caller that hands
        the outcome to another thread: the thread's own waiter is reset on
        that thread's next submit, so it cannot travel. It is settled by the
        lane on every exit -- run, refused while queued, left unrun, let go
        while running -- and must already be marked running, so no one but
        the lane can settle it. The waiter carries the answers ``call``
        raises.

        Raises:
            HardwareCommandRefusedError: ``'scope_disconnected'``, the
                scope's lane has closed (another executor raises the closed
                answer its closer states); ``'exclusive_activity_running'``,
                the scope is held by an activity this call is not made
                under; ``'activity_ended'``, the call is made under an
                activity that has ended.
            ControlSessionRefusedError: ``'control_session_held'``, a control
                session is held and this call is not its holder's.
        """
        task.silent_on_failure = True
        task.member = member
        # The run's own call goes through the door its protocol mode keeps
        # for it; anyone else's, and the run's once the mode has ended, through
        # put. The choice and the door's enqueue are one step under the state
        # lock, so the mode cannot turn between them.
        fut = None
        overriding = override is not None and override is self._override_key
        with self._state_lock:
            if self._is_protocol_door_holder() or (overriding and self.is_protocol_running()):
                fut = self.protocol_put(task, return_future=True, override=override, waiter=waiter)
            elif self.accepts_work():
                fut = self.put(task, return_future=True, override=override, waiter=waiter)
        if fut is None:
            if self.pending_shutdown:
                raise self._closed_answer(member)
            holder = self._claim.holder if self._claim is not None else None
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', member, holder.kind if holder is not None else None
            )
        if isinstance(fut, (HardwareCommandRefusedError, ControlSessionRefusedError)):
            raise fut
        return fut

    def bounded(
        self, waiter: concurrent.futures.Future, member: str, bound_s: float
    ) -> concurrent.futures.Future:
        """The caller's outcome of a submitted task: its waiter's, or past ``bound_s``, still running.

        ``call``'s bound for a task submitted without waiting: the returned
        Future takes ``waiter``'s outcome when the lane settles it within
        ``bound_s``, and otherwise ``LaneCallStillRunningError`` -- the task
        is queued or running and may still finish. ``waiter`` itself is
        settled by the lane on every exit as before, so whatever the
        submitter hangs on it (a guard released when the task ends) runs
        when the task ends, not when the bound does. An outcome that comes
        after the bound has no one waiting for it: a failure is logged and
        not shown, as ``call``'s late one is, and a result is logged.
        """
        told = concurrent.futures.Future()
        told.set_running_or_notify_cancel()
        lock = threading.Lock()

        def settle_once(settle: Callable[[], None]) -> bool:
            with lock:
                if told.done():
                    return False
                settle()
                return True

        def overran() -> None:
            settle_once(
                lambda: told.set_exception(
                    LaneCallStillRunningError('call_overran', member, bound_s)
                )
            )

        timer = threading.Timer(bound_s, overran)
        timer.daemon = True

        def outcome(settled: concurrent.futures.Future) -> None:
            timer.cancel()
            exception = settled.exception()
            if exception is None:
                result = settled.result()
                if not settle_once(lambda: told.set_result(result)):
                    logger.info(f'[{self.executor_name}] {member} finished after its {bound_s:g} s')
            elif not settle_once(lambda: told.set_exception(exception)):
                notifications.report_outcome(
                    exception, solicited=False, category=f'Task:{member}', log_only=True
                )

        timer.start()
        waiter.add_done_callback(outcome)
        return told

    def _refuse_submit(self, lane: str, cause: str, task: IOTask):
        """Narrate a refused submit at episode granularity; always returns None.

        Every state-refusal exit routes through here so a drop cannot be added
        without narration. The bare None these exits used to return is why a
        refusal was invisible to everyone: it is also what a successful
        fire-and-forget enqueue returned, so no caller could act on it and
        almost none tried.

        The task's action names the caller in the log -- the executor name
        alone identifies the lane, not who lost work.
        """
        who = _task_named(task)
        with self._refusal_lock:
            episode = self._refusal_episodes[lane]
            if episode is not None and episode['cause'] == cause:
                episode['count'] += 1
                return None
            previous = episode
            self._refusal_episodes[lane] = {'cause': cause, 'count': 1}
        # Logged outside the lock: a handler that blocks on disk must not stall
        # every other thread submitting to this executor.
        if previous is not None:
            self._log_episode_summary(previous)
        logger.warning(
            f'[{self.executor_name}] REFUSED {who} -- {cause}. Further '
            f'refusals are counted, not logged, until work is accepted again'
        )
        return None

    def _accept_submit(self, lane: str) -> None:
        """Close an open refusal episode on this lane after a successful
        enqueue, reporting what the episode cost.

        Closing on the next success rather than from enable() / protocol_start()
        / protocol end keeps the three lifecycle methods out of it: a hook is
        three sites to keep in sync, and missing one leaves an episode open
        forever. The one imperfect case is benign -- if nothing is ever
        submitted again the summary never prints, and nothing further was lost
        to report.
        """
        if self._refusal_episodes[lane] is None:
            return
        with self._refusal_lock:
            episode = self._refusal_episodes[lane]
            if episode is None:
                return
            self._refusal_episodes[lane] = None
        self._log_episode_summary(episode)

    def _log_episode_summary(self, episode: dict) -> None:
        logger.warning(
            f'[{self.executor_name}] Accepting work again -- '
            f'{episode["count"]} task(s) were refused and never ran while '
            f'{episode["cause"]}'
        )

    def accepts_work(self) -> bool:
        """Whether a task submitted to ``put`` right now would be queued.

        ``put`` drops silently -- returns None -- in two unrelated states: the
        lane was shut down, and a protocol fenced it. A shut lane has no worker
        left to drain its queue, so work accepted there would wait out its
        caller's whole timeout. A caller that must know BEFORE submitting asks
        this instead of re-deriving the two
        conditions, because a second copy of them drifts from the ones ``put``
        actually enforces, and the drift is invisible -- the task is dropped and
        the caller is told nothing. ``put`` reads them from here for the same
        reason, so there is exactly one place they are written down.

        Does NOT describe ``protocol_put``, whose fence runs the other way: it
        requires a protocol to be running and drops when none is.
        """
        if self.pending_shutdown:
            return False
        return not (self.protocol_running.is_set() and not self.protocol_finish.is_set())

    def put(
        self,
        task: IOTask,
        return_future: bool = False,
        *,
        override: object | None = None,
        waiter: concurrent.futures.Future | None = None,
    ) -> object | None:
        """Add an IOTask to the default execution queue.

        Return value reports the enqueue outcome so a caller can tell whether
        the task will actually run:

        - return_future True, enqueued: the task's waiter (await its result).
        - return_future False, enqueued: ENQUEUED.
        - lane shut down or fenced by a running protocol: None (dropped).
        - droppable_live task over the in-flight cap: LIVE_FRAME_DROPPED.
        - the scope is held and the task is not the holder's: the refusal
          (a waiter already carrying it, when return_future and no
          ``waiter`` was supplied).

        ``waiter``, with return_future, is registered in place of the
        thread's own (``submit``).

        Every non-ENQUEUED / non-waiter outcome means the task did not enter
        the queue and will never run. Success and drop returned the same
        None until the enqueued case got its own sentinel, so every successful
        fire-and-forget submit was logged and reported as dropped.
        """
        # Naming precedes every queue insertion below: once a task is on a
        # queue the worker may already be running it, and an unnamed task
        # renames its worker thread to the empty string.
        self._stamp(task, override)
        if not self.accepts_work():
            return self._refuse_submit(_LANE_DEFAULT, self._why_put_refuses(), task)
        refusal = self._claim_refusal(task)
        if refusal is not None:
            return self._refused_at_submit(task, refusal, return_future, waiter)

        # Selective backpressure: cap in-flight frame-carrying tasks so a
        # stalled single worker can't pin GBs of frame buffers (the
        # manual-record balloon). Drop the new frame -- latest-wins is the
        # live/record contract -- with accounting. Must-execute tasks (config /
        # motor / save) never set droppable_live and are unaffected.
        if task.droppable_live:
            with self._live_lock:
                if self._live_inflight >= _LIVE_FRAME_MAXSIZE:
                    self._live_dropped_count += 1
                    n = self._live_dropped_count
                    over = True
                else:
                    self._live_inflight += 1
                    over = False
            if over:
                if n == 1 or n % 30 == 0:
                    logger.warning(
                        f'[{self.executor_name}] LIVE FRAME QUEUE FULL '
                        f'(maxsize={_LIVE_FRAME_MAXSIZE}) -- dropping frame; '
                        f'total drops this run: {n}'
                    )
                return LIVE_FRAME_DROPPED

        # Push IO work item into queue. When return_future=True, hand
        # the caller a per-thread reusable waiter (was concurrent.futures
        # Future before; switched to drop Lock kernel-handle allocation
        # pressure during high-rate protocol submission).
        fut = None
        if return_future:
            fut = waiter if waiter is not None else _claim_waiter()
        if profile_trace.ENABLE_PROFILE_TRACE:
            task._t_enqueue = time.monotonic()
            task._queue_depth_at_enqueue = self.queue.qsize() + (1 if self._running_task else 0)
            task._queue_kind = 'default'
        # The gate asked again, the waiter registered and the task put, in one
        # step no switch can land inside: a run started or a close begun since
        # the first ask refuses it here, before anyone waits on it.
        with self._state_lock:
            admitted = self.accepts_work()
            if admitted:
                if fut is not None:
                    self._register_waiter(task, fut)
                with self._first_in_line_lock:
                    # A task the worker has dequeued but not yet recorded as
                    # running is missed here, so a task put in that instant is
                    # first in line beside it: erring kept costs a hold at
                    # most one task's wait.
                    task.first_in_line = self.queue.empty() and self.running_task is None
                    self.queue.put(task)
        if not admitted:
            if task.droppable_live:
                with self._live_lock:
                    self._live_inflight -= 1
            return self._refuse_submit(_LANE_DEFAULT, self._why_put_refuses(), task)
        self._accept_submit(_LANE_DEFAULT)
        return fut if return_future else ENQUEUED

    def _why_put_refuses(self) -> str:
        """The cause ``put`` names for a task its gate refused.

        ``accepts_work`` stays the single written-down copy of the gate; the
        flag is read again only to attribute the cause, not to re-derive the
        condition.
        """
        if self.pending_shutdown:
            return 'the lane is shut down'
        return 'a protocol run has this lane fenced'

    def _register_waiter(self, task: IOTask, fut) -> None:
        """Register ``fut`` as ``task``'s waiter, for the epilogue or the close to settle.

        Called under the state lock, after the door's gate is asked again
        and before the task is put: a close takes that lock to mark the lane
        closing, so a waiter registered here is on a lane still open, and
        the close's sweep answers it; a task the close beat is refused by
        the gate and registers none.
        """
        with self._caller_futures_lock:
            self.caller_futures[task] = fut
            self._caller_futures_alloc_count += 1

    def _finish_protocol_enqueue(self, task: IOTask, fut, return_future: bool):
        """Success tail of ``protocol_put``'s enqueue."""
        # Early warning that file writes are falling behind.
        self._accept_submit(_LANE_PROTOCOL)
        depth = self.protocol_queue.qsize()
        if depth > 20 and depth % 10 == 0:
            logger.warning(
                f'[{self.executor_name}] Protocol queue depth: {depth} -- '
                f'file writes may be falling behind'
            )
        # A return_future caller needs the future back to await it; a
        # fire-and-forget caller gets PROTOCOL_ENQUEUED so it can distinguish
        # this real enqueue from a dropped task -- not-running returns None.
        if return_future:
            return fut
        return PROTOCOL_ENQUEUED

    def protocol_put(
        self,
        task: IOTask,
        return_future: bool = False,
        *,
        override: object | None = None,
        waiter: concurrent.futures.Future | None = None,
    ) -> object | None:
        """Add an IOTask to the protocol execution queue.

        The protocol queue only drains while a protocol is in session (after
        protocol_start). Return value reports the enqueue outcome so a caller
        can tell whether the task will actually run:

        - return_future True, enqueued: the task's Future (await its result).
        - return_future False, enqueued: PROTOCOL_ENQUEUED.
        - protocol not running, or the lane shut down: None (task dropped).
        - the scope is held and the task is not the holder's: the refusal
          (a waiter already carrying it, when return_future).

        Every non-PROTOCOL_ENQUEUED / non-Future outcome means the task did
        not enter the queue and will never run.
        """
        self._stamp(task, override)
        if not self._run_door_open():
            return self._refuse_submit(_LANE_PROTOCOL, self._why_run_door_refuses(), task)

        refusal = self._claim_refusal(task, door=True)
        if refusal is not None:
            return self._refused_at_submit(task, refusal, return_future, waiter)

        fut = None
        if return_future:
            fut = waiter if waiter is not None else _claim_waiter()
        if profile_trace.ENABLE_PROFILE_TRACE:
            task._t_enqueue = time.monotonic()
            task._queue_depth_at_enqueue = self.protocol_queue.qsize() + (
                1 if self._running_task else 0
            )
            task._queue_kind = 'protocol'
        # As in put: the run asked again, the waiter registered and the task
        # put in one step, so a run's end or a close since the first ask
        # refuses it here rather than leaving it on a queue nobody serves.
        with self._state_lock:
            admitted = self._run_door_open()
            if admitted:
                if fut is not None:
                    self._register_waiter(task, fut)
                self.protocol_queue.put(task)
        if not admitted:
            return self._refuse_submit(_LANE_PROTOCOL, self._why_run_door_refuses(), task)
        return self._finish_protocol_enqueue(task, fut, return_future)

    def _run_door_open(self) -> bool:
        """Whether a task put through ``protocol_put`` right now would be queued.

        ``accepts_work``'s counterpart for the run's door, whose fence runs
        the other way: open only in a run. A closed lane is shut whatever its
        mode -- a run started after the close sets the mode again, and its
        work would wait on a queue no worker serves and no close answers.
        """
        return not self.pending_shutdown and self.protocol_running.is_set()

    def _why_run_door_refuses(self) -> str:
        """The cause ``protocol_put`` names for a task its gate refused."""
        if self.pending_shutdown:
            return 'the lane is shut down'
        return 'no protocol run is in session'

    def _stall_threshold_s(self, floor_s: float) -> float:
        """Seconds an in-flight task may run before it counts as stuck.

        Per-task-aware: a running task that declared its own cost (a
        whole-recording video write can legitimately run for minutes) raises
        the bar to that value, so a healthy long write is never declared a
        wedge by a flat constant. The declaration is read through the task so
        a budget stated at the command's def site counts here too -- reading
        the submission kwarg alone would see only half of them.
        """
        task = self.running_task
        declared = task.declared_slow_task_budget() if task is not None else None
        return max(floor_s, declared or 0.0)

    def _running_task_in_flight_s(self) -> float | None:
        """Age of the in-flight task in seconds, or None when the worker is
        between tasks. This -- not time-since-last-retirement -- is the
        stuck-worker signal: an executor idle for an hour before a run
        would otherwise read as wedged the moment its first write runs
        long."""
        task = self.running_task
        started = self._running_task_started_monotonic
        if task is None or started is None:
            return None
        return time.monotonic() - started

    def describe_running_task(self) -> str:
        """Name the in-flight task for a stall report: action, target file
        (the capture's base name rides in the task's ``name`` kwarg), and
        seconds in flight."""
        task = self.running_task
        if task is None:
            return 'no task in flight'
        parts = [getattr(task.action, '__name__', str(task.action))]
        kwargs = task.kwargs if isinstance(task.kwargs, dict) else {}
        file_name = kwargs.get('name')
        if file_name:
            parts.append(f"'{file_name}'")
        started = self._running_task_started_monotonic
        if started is not None:
            parts.append(f'{time.monotonic() - started:.0f}s in flight')
        return ' '.join(parts)

    def protocol_start(self, taking: Taking | None = None) -> None:
        """Put the lane in protocol mode for the activity acting under ``taking``.

        Args:
            taking: The taking the run acts under. Required on a lane that
                asks the claim: its protocol door admits only work under it.

        Raises:
            ValueError: the lane asks the claim and no taking was given --
                a door with no owner would admit anyone.
        """
        if self._claim is not None and taking is None:
            raise ValueError(
                f'{self.executor_name}: protocol_start on a lane that asks the claim needs the '
                "run's taking"
            )
        with self._state_lock:
            self._protocol_taking = taking
            # Clear stale finish flag from previous run. If protocol_finish is
            # still set (the worker hasn't ended the old run's mode yet), clear
            # it now so the worker doesn't end protocol mode during the new run
            # -- that would clear protocol_running mid-execution. The worker
            # ends a finishing run's mode under this lock, re-asking the flag.
            stale_finish = self.protocol_finish.is_set()
            self.protocol_finish.clear()
            self.protocol_running.set()
            # After the mode is set: an override call put from here on goes on
            # the run's queue itself, and one already waiting on the default
            # queue, which protocol mode does not serve, is carried there.
            self._carry_overrides_to_protocol_queue()
            self._wake.set()
        if stale_finish:
            logger.info(f'{self.name} Cleared stale protocol_finish flag')
        logger.info(f'{self.name} Protocol Started')

    def _carry_overrides_to_protocol_queue(self) -> None:
        """Move the override tasks waiting on the default queue onto the run's queue, in order.

        An override call is served whatever holds the scope, so it must wait
        on the queue the worker serves: put just before a run, behind a
        running task, it would otherwise wait out the run on a queue
        protocol mode never serves, while the run's wait for an idle lane
        waits on it. Its waiter stays registered.
        """
        q = self.queue
        inner = q._q if isinstance(q, _PriorityFifoQueue) else q
        with inner.mutex:
            moved, kept = [], []
            for item in inner.queue:
                task = item[2] if isinstance(q, _PriorityFifoQueue) else item
                (moved if task.override else kept).append(item)
            if not moved:
                return
            inner.queue.clear()
            inner.queue.extend(kept)
            if isinstance(q, _PriorityFifoQueue):
                heapq.heapify(inner.queue)
                moved.sort(key=lambda item: item[:2])
            inner.unfinished_tasks -= len(moved)
            inner.not_full.notify_all()
        for item in moved:
            task = item[2] if isinstance(q, _PriorityFifoQueue) else item
            if task.droppable_live:
                with self._live_lock:
                    self._live_inflight -= 1
            self.protocol_queue.put(task)

    def protocol_end(self) -> None:
        with self._state_lock:
            was_running = self._end_protocol_mode_locked()
            self._wake.set()
        self._log_protocol_ended(was_running)

    def _end_protocol_mode_locked(self) -> bool:
        """End protocol mode, under the state lock; returns whether it was on."""
        was_running = self.protocol_running.is_set()
        self.protocol_running.clear()
        return was_running

    def _log_protocol_ended(self, was_running: bool) -> None:
        """Say a protocol mode ended, after the state lock is released."""
        if was_running:
            logger.info(f'{self.name} Protocol Ended')

    def wait_for_idle(self, timeout: float = 1.0) -> bool:
        """Block until the worker is between tasks (running_task is
        None) or `timeout` seconds elapse.

        Used by callers from another thread that need to ensure any
        in-flight task has completed before they tear down shared
        state the task may reference -- the canonical example is the
        protocol cleanup path, which clears `protocol_running` then
        proceeds to mutate scope / camera / settings state that an
        in-flight io-executor task may be reading.

        Returns True if the worker reached idle within `timeout`,
        False if the timeout fired first. Callers that get False
        should log and proceed -- timing out is preferable to
        blocking interpreter shutdown.

        Implementation: polls `running_task is None` at 1 ms intervals.
        The poll cost is acceptable because this method is called at
        teardown / between protocols, not in any hot path.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.running_task is None:
                return True
            time.sleep(0.001)
        return False

    def protocol_finish_then_end(self) -> None:
        self.protocol_finish.set()
        self._wake.set()
        logger.info(f'{self.name} set to complete protocol then end')

    def end_protocol_mode(self) -> None:
        """Idempotent safety net for teardown paths that left this executor
        in protocol-mode.

        While ``protocol_running`` is set the worker pulls only from
        ``protocol_queue``; if a protocol aborts or tears down without the
        normal completion path ending this executor, the worker waits
        forever for run work and the normal queue (composite,
        video, z-projection, manual file ops) is never served. Setting
        ``protocol_finish`` lets the worker drain any remaining protocol
        items (so pending file writes still flush) and then exit protocol
        mode, returning to normal-queue service. A no-op when the executor
        is not in protocol-mode, so it is safe to call on every teardown.
        """
        if self.protocol_running.is_set() and not self.protocol_finish.is_set():
            logger.warning(
                f'{self.name} still in protocol-mode at teardown; '
                f'draining protocol queue then returning to normal service'
            )
            self.protocol_finish_then_end()

    def is_protocol_running(self):
        return self.protocol_running.is_set()

    def _run_loop(self):
        my_generation = self._worker_generation
        _executor_worker.executor = self
        if self.lane:
            _lane_worker.executor = self
        while True:
            if self._worker_generation != my_generation:
                return
            try:
                # Cleared before anything the pass decides on is read: a put
                # or a switch from here on sets it again, so the wait below
                # returns at once and the next pass sees what changed.
                self._wake.clear()
                task = None
                if self._claim is not None and not self.queue.empty():
                    self._refuse_queued()
                try:
                    # The mode is read, and an ended run's queue taken, under
                    # the state lock; the take from the queue comes after its
                    # release, so a switch landing between is the next pass's
                    # to see.
                    ended = None
                    with self._state_lock:
                        in_protocol = (
                            self.protocol_running.is_set() or self.protocol_finish.is_set()
                        )
                        from_run = in_protocol and not self._first_in_line_waiting()
                        if not in_protocol and not self.protocol_queue.empty():
                            ended = self._take_protocol_pending()
                    if ended is not None:
                        self._answer_protocol_pending(*ended)
                        continue
                    if from_run:
                        task = self.protocol_queue.get(block=False)
                        task.protocol = True
                    else:
                        task = self.queue.get(block=False)
                        task.protocol = False
                        # A droppable_live task has left the queue -- free its
                        # in-flight slot so the producer can enqueue the next.
                        if task.droppable_live:
                            with self._live_lock:
                                self._live_inflight -= 1
                    if profile_trace.ENABLE_PROFILE_TRACE:
                        task._t_dequeue = time.monotonic()
                    # Claim the task the moment it leaves the queue, not after
                    # the checks below. A task already dequeued but not yet
                    # recorded as running belongs to neither half of the busy
                    # predicates (queue non-empty, or a task is running), so a
                    # waiter sampling in that window is told the run's writes
                    # are finished while the last one has not started -- the
                    # stack build then reads a directory missing its final
                    # plane, and the next run's files_writing gate admits a
                    # run over a write still in flight. Claiming here covers
                    # both queues; erring busy costs a waiter at most one
                    # task's runtime, which is the honest direction.
                    self._running_task_started_monotonic = time.monotonic()
                    self.running_task = task
                except queue.Empty:
                    if self.pending_shutdown:
                        return
                    # A finishing run's mode ends once its queue is drained,
                    # asked under the lock the next run's start takes: a start
                    # that has cleared the flag is a new run, not this one.
                    ended_mode = False
                    with self._state_lock:
                        if self.protocol_finish.is_set() and self.protocol_queue.empty():
                            ended_mode = self._end_protocol_mode_locked()
                            self.protocol_finish.clear()
                    self._log_protocol_ended(ended_mode)
                    if not ended_mode:
                        self._wake.wait()
                    continue

                ended = None
                with self._state_lock:
                    if (
                        not (self.protocol_running.is_set() or self.protocol_finish.is_set())
                        and not self.protocol_queue.empty()
                    ):
                        ended = self._take_protocol_pending()
                if ended is not None:
                    self._answer_protocol_pending(*ended)

                # Asked again as it leaves the queue: a hold can begin while
                # the task waits, and the holder's restore must not be run
                # over by work queued before it.
                refusal, counted = self._admit(task)
                if refusal is not None:
                    self._answer_unrun(task, refusal, asked=True)
                    (self.protocol_queue if task.protocol else self.queue).task_done()
                    self.running_task = None
                    self._running_task_started_monotonic = None
                    continue

                # Started, or left unrun because the lane is closing: decided
                # under the lock the close's sweep reads, so the sweep never
                # answers a started task as unrun, nor this worker runs one
                # the sweep answered.
                with self._caller_futures_lock:
                    unrun = self.pending_shutdown
                    if not unrun:
                        task.fate = _STARTED
                if unrun:
                    if counted:
                        self._claim.leave_falsifying_change()
                    # Drop the claim taken at dequeue: a claim left on a
                    # stopping worker answers "busy" forever to anything
                    # still polling.
                    self.running_task = None
                    self._running_task_started_monotonic = None
                    self._answer_unrun(task, self._closed_answer(_member_of(task)), asked=False)
                    return

                run_result = None
                run_exc = None
                try:
                    with acting(task.taking), _running_ask(task.asked):
                        run_result = task.run()
                except BaseException as e:
                    run_exc = e
                finally:
                    if counted:
                        self._claim.leave_falsifying_change()

                if self._worker_generation != my_generation:
                    # Abandoned by wedge recovery while stuck inside this
                    # task. The replacement worker owns ALL executor state
                    # now; running the epilogue here would fire a stale
                    # task-failure popup, clobber the replacement's
                    # running_task and timing stamps, complete a cancelled
                    # future, and unbalance queue bookkeeping the recovery
                    # already reconciled. Exit without touching anything.
                    # The epilogue would have reported what the task raised;
                    # this line is now its only record.
                    carried = run_exc
                    if carried is None and isinstance(run_result, tuple) and len(run_result) == 2:
                        carried = run_result[1]
                    outcome = (
                        f'; it raised {type(carried).__name__}: {carried}'
                        if carried is not None
                        else ''
                    )
                    logger.warning(
                        f'[{self.executor_name}] Abandoned worker finished '
                        f'{getattr(task.action, "__name__", str(task.action))} '
                        f'after wedge recovery; exiting without epilogue{outcome}',
                        exc_info=carried
                        if carried is not None and not isinstance(carried, Refusal)
                        else None,
                    )
                    return

                if run_exc is not None:
                    self._on_task_done(task, None, run_exc)
                elif isinstance(run_result, tuple) and len(run_result) == 2:
                    self._on_task_done(task, run_result[0], run_result[1])
                else:
                    self._on_task_done(task, run_result, None)
            except Exception as e:
                # Distinct from a task raising: this is the dequeue or the
                # epilogue failing around the task, and the loop continues.
                logger.error(
                    f'[{self.executor_name}] Worker loop error outside the task '
                    f'(dispatch or epilogue); the worker continues: {e}',
                    exc_info=True,
                )
            finally:
                # A finished task's outcome is its waiter's to hold. Kept here
                # it lives on through the wait for the next task, and with a
                # failure every frame of its traceback -- a failed build's
                # tiles -- until the lane next has work, which can be hours.
                task = run_result = run_exc = None

    def _first_in_line_waiting(self) -> bool:
        """Whether the default queue's next task was put first in line.

        A run's protocol mode serves only the run's queue; the task put on
        the lane before the run took the scope, with nothing ahead of it, is
        served first, or the run's wait for an idle lane would never end.
        """
        q = self.queue
        inner = q._q if isinstance(q, _PriorityFifoQueue) else q
        with inner.mutex:
            if not inner.queue:
                return False
            head = inner.queue[0]
            task = head[2] if isinstance(q, _PriorityFifoQueue) else head
            return task.first_in_line

    def _refuse_queued(self) -> None:
        """Refuse the default-queue tasks the claim now refuses, in place.

        A hold begins while work waits: on a lane in protocol mode the
        worker serves only the run's queue, so that work would wait out the
        whole run and then run on top of its restore. Each is answered with
        the refusal now instead -- waiter, callback, notice -- and the
        tasks the claim admits keep their places. Runs on the worker, where
        a task's epilogue always runs.
        """
        refused = []
        q = self.queue
        inner = q._q if isinstance(q, _PriorityFifoQueue) else q
        with inner.mutex:
            kept = []
            for item in inner.queue:
                task = item[2] if isinstance(q, _PriorityFifoQueue) else item
                refusal = self._claim_refusal(task)
                if refusal is None:
                    kept.append(item)
                else:
                    refused.append((task, refusal))
            if not refused:
                return
            inner.queue.clear()
            inner.queue.extend(kept)
            if isinstance(q, _PriorityFifoQueue):
                heapq.heapify(inner.queue)
            inner.unfinished_tasks -= len(refused)
            inner.not_full.notify_all()
        for task, refusal in refused:
            if task.droppable_live:
                with self._live_lock:
                    self._live_inflight -= 1
            task.protocol = False
            self._answer_unrun(task, refusal, asked=True)

    def _answer_unrun(self, task: IOTask, refusal: BaseException, *, asked: bool) -> None:
        """Answer a task the lane lets go of unrun: its waiter, its report, its callback.

        The one answer for a task that will not run: refused while queued,
        left on the run's queue when the run ended, or unstarted when the
        lane closed. Its fate is decided under the lock the worker starts a
        task under, so it is answered once, and never if it was started.
        Given with no queue's mutex held: a waiter's holder or a callback may
        put work on this lane.

        Args:
            asked: Whether someone is asking for the task now; False for the
                close's answers, which nobody asked for at that moment.
        """
        with self._caller_futures_lock:
            if task.fate is not None:
                return
            task.fate = _UNRUN
            fut = self.caller_futures.pop(task, None)
            if fut is not None:
                self._caller_futures_pop_count += 1
        logger.warning(f'[{self.executor_name}] NOT RUN {_task_named(task)} -- {refusal}')
        if fut is not None:
            fut.set_exception(refusal)
        self._report_task_failure(
            task, refusal, owned=task.silent_on_failure and fut is not None, asked=asked
        )
        task.on_complete(None, refusal)

    def _report_task_failure(
        self, task: IOTask, exception: BaseException, owned: bool, *, asked: bool
    ) -> None:
        """Report a task that failed or was refused, unless it is its waiter's.

        ``owned`` is decided where the waiter's registration is popped, under
        the lock a timed-out waiter takes to pop it back: a silent task with a
        registered waiter is that waiter's, reported where its flight ends. A
        silent task with no one waiting is logged and not shown; any other
        task is logged and shown. What it is, and its words, are its own.
        ``asked`` is whether someone asks for the task at this moment: a
        refusal nobody asked for then is reported unsolicited.
        """
        action_name = getattr(task.action, '__name__', str(task.action))
        if isinstance(exception, CancelledError):
            logger.debug(f'[{self.name}] {action_name} cancelled (by-contract)')
            return
        if owned:
            return
        # Outside a run, someone asked for this task, so a refusal is an
        # answer: never filtered as a repeat, and replacing the last refusal
        # popup. A run's own task keeps the run's mute -- mid-run only a
        # fatal error may pop up; the worker marks which queue a task came
        # off, and serves only the run's queue while a run is going.
        #
        # The category carries the action, which is the dedup identity and
        # is never shown; the title is prose, so a refused stage move and an
        # unrelated camera failure seconds apart stay distinct events.
        notifications.report_outcome(
            exception,
            solicited=asked and isinstance(exception, Refusal) and not task.protocol,
            category=f'Task:{action_name}',
            log_only=task.silent_on_failure,
            untyped_title='Protocol operation failed'
            if task.protocol
            else 'Background operation failed',
        )

    def _on_task_done(self, task: IOTask, result, exception):
        # Receives (result, exception) from worker, then schedules task.on_complete.
        # The waiter's registration is popped before the outcome is reported,
        # in one acquisition with the decision of whose it is: a timed-out
        # waiter pops under the same lock, so exactly one of the two finds it.
        with self._caller_futures_lock:
            caller_fut = self.caller_futures.pop(task, None)
            if caller_fut is not None:
                self._caller_futures_pop_count += 1
        if exception is not None:
            self._report_task_failure(
                task, exception, owned=task.silent_on_failure and caller_fut is not None, asked=True
            )
        self.last_task_done_monotonic = time.monotonic()

        # Threading audit -- emit per-IOTask timing row when opt-in tracing
        # is enabled. Fields answer "which lane starved?" (queue_wait_ms per
        # lane per time bucket), "which actions are slow?" (exec_ms per
        # action name), and "does queue depth correlate with wait?"
        # (queue_depth_at_enqueue). See src/_lumascope/lib/profile_trace.py for the unified
        # profile_trace_enabled settings gate.
        if profile_trace.ENABLE_PROFILE_TRACE:
            t_enqueue = getattr(task, '_t_enqueue', None)
            t_dequeue = getattr(task, '_t_dequeue', None)
            if t_enqueue is not None and t_dequeue is not None:
                queue_wait_ms = (t_dequeue - t_enqueue) * 1000.0
                exec_ms = (self.last_task_done_monotonic - t_dequeue) * 1000.0
                profile_trace.trace(
                    'iotask_trace.csv',
                    _IOTASK_TRACE_HEADER,
                    [
                        int(time.time() * 1000),  # ts_ms
                        f'{(queue_wait_ms + exec_ms):.3f}',  # duration_ms (total)
                        self.name or '',  # executor
                        task.name or '',  # task_name (worker thread name)
                        getattr(task.action, '__name__', str(task.action))[:40],
                        getattr(task, '_queue_kind', 'default'),
                        getattr(task, '_queue_depth_at_enqueue', -1),
                        f'{queue_wait_ms:.3f}',
                        f'{exec_ms:.3f}',
                        type(exception).__name__ if exception is not None else '',
                    ],
                    recording_id=profile_trace.NO_RECORDING,
                )
        if caller_fut:
            # This future was returned to a caller - they still hold a reference
            # DON'T null internal state or it will break their .result() call
            if exception:
                caller_fut.set_exception(exception)
            else:
                caller_fut.set_result(result)
            # Only delete our local reference, not the object internals
            del caller_fut

        task.on_complete(result, exception)
        (self.protocol_queue if task.protocol else self.queue).task_done()

        self.running_task = None
        self._running_task_started_monotonic = None

    def caller_futures_stats(self) -> tuple:
        """Return (allocs, pops, live_count) for the caller_futures dict.

        Snapshot is taken under the caller_futures_lock so the three
        values are mutually consistent. Drift between allocs and pops
        indicates Future objects accumulating in the dict (handle leak
        signal on Windows; see attribute comments at __init__).
        """
        with self._caller_futures_lock:
            return (
                self._caller_futures_alloc_count,
                self._caller_futures_pop_count,
                len(self.caller_futures),
            )

    def shutdown(self, wait: bool = True) -> None:
        # The close and its taking of what is queued are one step under the
        # state lock, so a put that passed its first ask before the close
        # re-asks after it and is refused, never left on a swept queue.
        with self._state_lock:
            self.pending_shutdown = True
            was_running = self._end_protocol_mode_locked()
            queued = self._take_queued()
            pending = self._take_protocol_pending()
            self._wake.set()
        self._log_protocol_ended(was_running)
        for task in queued:
            task.protocol = False
            self._answer_unrun(task, self._closed_answer(_member_of(task)), asked=False)
        self._answer_protocol_pending(*pending)
        if self.worker_alive and wait:
            # The close woke the worker, which returns from its wait; bound
            # the join so a hung task does not block process exit
            # indefinitely.
            self._worker_thread.join(timeout=5.0)

        # The queued tasks were answered above; a waiter still registered is
        # a task's the worker started, which may still finish, or one taken
        # off a queue unstarted -- dequeued and not yet decided, or put as
        # the close began. Each is answered, never just forgotten: a waiter
        # whose registration is gone reads that as its outcome having
        # landed, and would wait for it forever.
        with self._caller_futures_lock:
            left = list(self.caller_futures.items())
            running = [(task, fut) for task, fut in left if task.fate == _STARTED]
            for task, _fut in running:
                del self.caller_futures[task]
            self._caller_futures_pop_count += len(running)
        for task, fut in running:
            fut.set_exception(LaneCallStillRunningError('lane_closed', _member_of(task)))
        for task, _fut in left:
            if task.fate != _STARTED:
                self._answer_unrun(task, self._closed_answer(_member_of(task)), asked=False)
        self.running_task = None
        self._running_task_started_monotonic = None

    def join(self, timeout=None):
        # Block until all queued tasks processed (or until timeout)
        pass

    def _take_queued(self) -> list[IOTask]:
        """Empty the default queue as the lane closes, under the state lock; returns what it held.

        For priority_aware executors the tasks come HIGH-first, the order
        the queue would have served them in, and are answered in that order.
        """
        taken = []
        while True:
            try:
                task = self.queue.get_nowait()
            except queue.Empty:
                break
            self.queue.task_done()
            if task.droppable_live:
                with self._live_lock:
                    self._live_inflight -= 1
            taken.append(task)
        return taken

    def clear_protocol_pending(self) -> None:
        """Empty the run's queue as its protocol mode ends, answering what it held.

        An override call still waiting is carried back to the default queue,
        in order, and served after the run; every other task on the run's
        queue is the run's own and is answered ``activity_ended``. As the
        lane closes, every task is answered with the closed answer instead.
        """
        with self._state_lock:
            taken = self._take_protocol_pending()
        self._answer_protocol_pending(*taken)

    def _take_protocol_pending(self) -> tuple[bool, list[IOTask]]:
        """Empty the run's queue under the state lock, carrying its overrides back.

        Returns whether the lane is closing and every task taken, for
        ``_answer_protocol_pending`` to answer once the lock is released: an
        answer settles a waiter whose done-callbacks run on this thread.
        """
        closing = self.pending_shutdown
        pq = self.protocol_queue
        with pq.mutex:
            left = list(pq.queue)
            pq.queue.clear()
            pq.unfinished_tasks -= len(left)
            pq.not_full.notify_all()
        for task in left:
            if task.override and not closing:
                task.first_in_line = False
                self.queue.put(task)
        return closing, left

    def _answer_protocol_pending(self, closing: bool, left: list[IOTask]) -> None:
        """Answer the run's tasks ``_take_protocol_pending`` took, all but the overrides it carried."""
        for task in left:
            if task.override and not closing:
                continue
            task.protocol = True
            member = _member_of(task)
            refusal = (
                self._closed_answer(member)
                if closing
                else HardwareCommandRefusedError('activity_ended', member)
            )
            self._answer_unrun(task, refusal, asked=not closing)

    def is_busy(self):
        # Returns true if tasks queued or running
        return not (self.queue.empty() and self.running_task is None)

    def queue_size(self) -> int:
        return self.queue.qsize()

    def seconds_since_last_task(self) -> float:
        return time.monotonic() - self.last_task_done_monotonic

    def in_flight_task_stalled(self, floor_s: float) -> bool:
        """True when the task the worker is running has run past the
        (per-task-aware) stall threshold.

        The difference between "draining -- keep waiting" and "wedged --
        offer recovery": a lane that is retiring tasks keeps the in-flight
        age short, and a worker between tasks is progress by definition, so
        neither reads as stalled. The one judgement of "stuck" for every
        waiter on this lane, so a run waiting for the camera lane and a
        cleanup waiting for the file lane cannot disagree about it.
        """
        in_flight_s = self._running_task_in_flight_s()
        return in_flight_s is not None and in_flight_s >= self._stall_threshold_s(floor_s)

    def replace_stuck_worker(self, answer: Callable[[str], BaseException]) -> None:
        """Abandon a worker stuck inside a task and start a replacement.

        A task stuck in a call Python cannot interrupt -- a write to an
        unresponsive drive -- holds the lane's one worker, and everything
        queued behind it waits forever. The stuck worker is quarantined:
        it is a daemon thread, and when its call finally returns it sees the
        moved-on generation and exits without touching executor state.
        Exactly one ACTIVE worker exists at all times. Nothing queued is
        discarded; the replacement serves it in order, and the owner of the
        work decides what the stuck task's loss means.

        A no-op when no task is in flight.

        Args:
            answer: What the stuck task's waiter is told, given the stuck
                task's description: stated by the one caller, which knows
                what replacing its worker means. The task may still finish.
        """
        stuck = self.running_task
        if stuck is None:
            return
        described = self.describe_running_task()
        # WARNING, not ERROR: this is the recovery a person chose, after the
        # stall was reported as a fault; the abandoned thread and its lost
        # write are worth a reader's attention, not a second failure.
        logger.warning(f'[{self.executor_name}] Replacing a worker stuck on {described}')
        # Quarantine FIRST: were the stuck task to finish before the
        # generation moves, its epilogue would run against state the
        # replacement now owns.
        self._worker_generation += 1
        with self._caller_futures_lock:
            fut = self.caller_futures.pop(stuck, None)
            if fut is not None:
                self._caller_futures_pop_count += 1
        if fut is not None:
            fut.set_exception(answer(described))
        self.start()
        # The orphan's guarded epilogue will not clear these and the
        # replacement only sets them when it dequeues something; left
        # stale, the lane would read as stuck forever.
        self.running_task = None
        self._running_task_started_monotonic = None
