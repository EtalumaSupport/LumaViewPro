# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# Executors must be GUI-agnostic. No Kivy imports here.
# UI callbacks are dispatched via _ui_dispatch(), which defaults to
# direct invocation. The GUI layer passes Clock.schedule_once as
# the ui_dispatcher parameter when constructing executors.

from concurrent.futures import CancelledError
import heapq
import itertools
import queue
from collections.abc import Callable, Sequence
from lvp_logger import logger
from lib import profile_trace
from modules.notification_center import REFUSAL_OPERATION_KEY, notifications
from modules.activity_claim import ActivityClaim, Taking, acting, current_taking
from modules.exceptions import HardwareCommandRefusedError, Refusal
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

# F-2: sentinel returned from protocol_put when a bounded protocol_queue
# is full. Distinct from None (which means "executor disabled" or
# "protocol not running"); callers that care about overflow check for
# `is PROTOCOL_QUEUE_FULL` so a frame can be marked capture_failed in
# the execution record instead of silently dropped.
PROTOCOL_QUEUE_FULL = object()
# Sentinel returned from protocol_put when a fire-and-forget task (return_future
# False) DID enter the queue. Distinct from None: a no-future enqueue used to
# return None too, so a caller could not tell a successful enqueue from a
# dropped one (disabled / protocol-not-running both return None). Callers that
# gate later work on "did this task actually run?" check `is PROTOCOL_ENQUEUED`.
PROTOCOL_ENQUEUED = object()
# Sentinel returned by put() when a frame-carrying (droppable_live) task is
# dropped because too many are already in flight on the single worker.
LIVE_FRAME_DROPPED = object()
# Sentinel returned from put() when a fire-and-forget task (return_future False)
# DID enter the queue -- the default queue's counterpart to PROTOCOL_ENQUEUED,
# and for the same reason: a no-future enqueue returned None, which is also what
# a fenced or disabled executor returns, so a successful submit was reported to
# its caller as a drop. Callers that must know whether the task will run check
# `is ENQUEUED`.
ENQUEUED = object()
# Sentinel returned from protocol_put_wait when the blocking enqueue gave up:
# the bounded queue stayed full past the caller's stall budget AND the worker
# retired nothing in that window. Distinct from PROTOCOL_QUEUE_FULL (a
# non-blocking drop signal): WEDGED means the writer is stuck, not slow, and
# the caller owns a loud user-facing abort/recovery decision.
PROTOCOL_QUEUE_WEDGED = object()

# Refusal-episode lanes. A disabled or fenced executor refuses every submit for
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

# Slot-poll interval for the blocking protocol enqueue. Short enough that an
# abort signalled mid-wait is honored promptly; long enough that a full queue
# does not busy-spin.
_BACKPRESSURE_POLL_S = 0.25

# Cumulative blocked-enqueue wait per run past which the save disk is called
# out as too slow for the run's own demand. Demand-relative on purpose: an
# absolute MB/s floor false-fires on healthy machines (a measured healthy
# bench sustains under 1.5 MB/s -- see PERFORMANCE_BUDGETS.md
# protocol_write_backpressure_wait_s), while time the capture loop spent
# waiting for a write slot is unmet demand by definition. Consumed by the
# run-end summary in protocol_cleanup; the first crossing also logs here.
SLOW_WRITE_BLOCKED_WARN_S = 30.0

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

    def result(self, timeout=None):
        if not self._event.wait(timeout):
            from concurrent.futures import TimeoutError as _TimeoutError

            raise _TimeoutError(f'task did not complete within {timeout}s')
        if self._exception is not None:
            raise self._exception
        return self._result

    def cancel(self) -> bool:
        """Best-effort cancel; signals waiting caller with CancelledError.
        Matches the Future API used by clear_pending / clear_protocol_pending
        in the executor (which wraps in try/except, so a False return
        or no-op is also acceptable).
        """
        if self._event.is_set():
            return False
        from concurrent.futures import CancelledError as _CancelledError

        self._exception = _CancelledError()
        self._event.set()
        return True


# Which lane's worker this thread is, set by the worker as it starts. A lane
# worker never waits on another lane: that wait is one half of the pair that
# makes a deadlock. On its own lane it is already where the work belongs, so
# the work runs inline rather than waiting on the queue it is draining.
_lane_worker = threading.local()


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
    ):
        self.action = action
        self.priority = priority
        self._ui_dispatch = None  # Set by executor when task is dispatched
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
            # Reports the raise, not an escape: this returns the exception to
            # the worker, which reports it to the user. The separate task
            # epilogue owns the user-facing wording, so this line must not
            # duplicate it. This is the record that names the SYMBOL -- the
            # user-facing one deliberately does not.
            action_name = getattr(self.action, '__name__', str(self.action))
            if isinstance(e, Refusal):
                # A declined request, not a fault: no traceback, and not
                # ERROR. Logged here whatever the task -- fire-and-forget,
                # waited, or run inline on a scope with no executor, where
                # no notification is ever posted and this line is the only
                # record.
                logger.warning(f'[IOTask    ] {action_name} refused ({type(e).__name__}): {e}')
            else:
                logger.error(
                    f'[IOTask    ] {action_name} raised {type(e).__name__}: {e}', exc_info=True
                )
            return None, e

    def set_callback(self, callback, cb_args, cb_kwargs):
        self.callback = callback
        self.cb_args = cb_args
        self.cb_kwargs = cb_kwargs if cb_kwargs is not None else {}

    def on_complete(self, result, exception):
        if self.callback is None:
            return

        def _safe_callback(dt):
            try:
                self.callback(*cb_args, **cb_kwargs)
            except Exception:
                logger.error(
                    f'[IOTask    ] Callback {self.callback} raised exception', exc_info=True
                )

        if self.pass_result:
            # Only copy when we need to mutate
            cb_kwargs = dict(self.cb_kwargs)
            cb_kwargs['result'] = result
            cb_kwargs['exception'] = exception
        else:
            cb_kwargs = self.cb_kwargs

        cb_args = self.cb_args
        if self._ui_dispatch is not None:
            self._ui_dispatch(_safe_callback, 0)
        else:
            _direct_dispatch(_safe_callback, 0)

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


def _direct_dispatch(func, timeout=0):
    """Default UI dispatcher: call function directly (no GUI scheduling).

    Used when no Kivy Clock is available (tests, headless, REST API).
    Matches Clock.schedule_once(func, timeout) signature.
    """
    if callable(func):
        try:
            func(0)  # Call with dummy dt=0 (same as Clock passes)
        except Exception as e:
            import logging as _log

            _log.getLogger('LVP').debug(f'_direct_dispatch error: {e}')


class _PriorityFifoQueue:
    """PriorityQueue wrapper hiding the tuple wrap/unwrap from callers.
    put(task) / get() -> task have the same shape as queue.Queue; the
    priority field drives ordering, a monotonic counter breaks ties
    FIFO within priority. IOTask stays non-comparable because the
    counter is reached only on an impossible (priority, counter) tie.
    """

    def __init__(self):
        self._q = queue.PriorityQueue()
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
        ui_dispatcher=None,
        protocol_queue_maxsize: int = 0,
        priority_aware: bool = False,
        lane: bool = True,
    ):
        # priority_aware=True swaps the default queue for a priority
        # wrapper; protocol_queue stays FIFO so step ordering inside
        # a protocol is preserved.
        self.priority_aware = priority_aware
        if priority_aware:
            self.queue = _PriorityFifoQueue()
        else:
            self.queue = queue.Queue()
        # F-2: protocol_queue_maxsize=0 keeps the historical unbounded
        # behavior; file_io_executor passes 32 so a save-thread that
        # falls behind drops new captures with a sentinel return rather
        # than letting the queue grow without bound.
        self.protocol_queue = queue.Queue(maxsize=protocol_queue_maxsize)
        self.protocol_queue_maxsize = protocol_queue_maxsize
        self._protocol_queue_dropped_count = 0
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
        self.global_callback = None
        self.pending_shutdown = False
        self._caller_futures_lock = threading.Lock()
        self.caller_futures = {}
        # Monotonic alloc/pop counters paired with caller_futures. Drift
        # between the two means Future entries are leaking; on Windows
        # each leaked Future leaves a Lock/Semaphore kernel handle open.
        # Surfaced by metrics_logger as a permanent invariant.
        self._caller_futures_alloc_count = 0
        self._caller_futures_pop_count = 0

        self.cleared_queue = False
        self.cleared_protocol_queue = False

        self._disable = False

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
        # Per-run total the blocking protocol enqueue spent waiting for a
        # queue slot -- the demand-relative slow-disk signal. Reset at
        # protocol_start alongside the drop counter.
        self._backpressure_blocked_s = 0.0
        self._slow_write_warned = False

        # Protocol completion callback support
        self._callback_lock = threading.Lock()
        self.protocol_complete_callback = None
        self.protocol_complete_cb_args = ()
        self.protocol_complete_cb_kwargs = {}
        # Persistent protocol-mode-exit listeners (the one-shot callback
        # above is consume-once and owned by run cleanup).
        self._protocol_idle_listeners = []

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

        # UI dispatcher -- executors don't import GUI frameworks.
        # GUI layer passes Clock.schedule_once; tests/headless use default.
        self._ui_dispatch = ui_dispatcher or _direct_dispatch

    @property
    def running_task(self):
        with self._running_task_lock:
            return self._running_task

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

    def disable(self) -> None:
        """Refuse new work on the default lane; the worker finishes what it holds.

        A run closes the camera lane this way and then drives the camera from
        its own thread. The tasks already queued or running are not stopped
        and not parked: they run to completion on the worker, and the run
        waits for the lane to go idle before it touches the camera. Parking
        the worker instead left a caller waiting on a queued task until the
        run ended.
        """
        self._disable = True

    def enable(self) -> None:
        self._disable = False

    def ask_claim(self, claim: ActivityClaim) -> object:
        """Make this lane ask ``claim`` before it runs work; returns the override key.

        While a run or a diagnostic holds the scope, a task that was not
        made under the holder's taking is refused -- at submit and again
        when the worker takes it off the queue -- with
        ``HardwareCommandRefusedError``, whoever made it and however. The
        key is the one way past that: the composition root keeps it for the
        named overrides, so a caller holding this executor cannot mark its
        own work as one.
        """
        self._claim = claim
        self._override_key = object()
        return self._override_key

    def _claim_refusal(
        self, task: IOTask, *, door: bool = False
    ) -> HardwareCommandRefusedError | None:
        """The refusal the claim gives ``task`` right now, or None.

        With ``door``, the task is entering the protocol queue, which also
        needs the taking that raised protocol mode.
        """
        if self._claim is None or task.override:
            return None
        who = getattr(task.action, '__name__', None) or repr(task.action)
        # Work made under a taking that has ended is its activity's leftover
        # -- an autofocus unwind that outlived the run's cleanup -- and is
        # refused even when nothing holds the scope now: the activity that
        # would have wanted it is gone.
        if task.taking is not None and not task.taking.holds:
            return HardwareCommandRefusedError('activity_ended', who)
        holder = self._claim.refusing_holder(task.taking)
        if holder is None:
            if not door or task.taking is self._protocol_taking:
                return None
            holder = self._claim.holder
        kind = holder.kind if holder is not None else None
        return HardwareCommandRefusedError('exclusive_activity_running', who, kind)

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
        task.taking = current_taking()
        task.override = override is not None and override is self._override_key

    def _refused_at_submit(self, task: IOTask, refusal, return_future: bool):
        """Answer a submit the claim refused, as the worker answers a task that failed.

        The waiter carries the refusal and the task's callback is told, so a
        caller that waits, one that passes a callback and one that ignores
        the return all hear it -- a raw put whose return nobody reads gets the
        lane's refusal notice rather than silence.
        """
        logger.warning(f'[{self.executor_name}] REFUSED {refusal.member} -- {refusal}')
        task._ui_dispatch = self._ui_dispatch
        task.protocol = False
        self._report_task_failure(task, refusal)
        task.on_complete(None, refusal)
        if return_future:
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
        work runs inline on that worker, under the claim the same way.

        Raises:
            RuntimeError: called from another lane's worker, which never
                waits on a lane.
            HardwareCommandRefusedError: the lane is closed, or the scope is
                held by an activity this call is not made under.
        """
        worker = getattr(_lane_worker, 'executor', None)
        if worker is self:
            self._stamp(task, None)
            refusal = self._claim_refusal(task)
            if refusal is not None:
                raise refusal
            return task.action(*task.args, **task.kwargs)
        if worker is not None:
            raise RuntimeError(
                f'{member}: a blocking dispatch from the {worker.executor_name} lane worker '
                f'onto {self.executor_name} -- a lane worker never waits on another lane'
            )
        task.silent_on_failure = True
        # The run's own call goes through the door its protocol mode keeps
        # for it; anyone else's, and the run's once the mode has ended, through
        # put. Asked twice: a fence or its end can land between the question
        # and the submit, and both doors answer a closed lane with None.
        fut = None
        if self._is_protocol_door_holder():
            fut = self.protocol_put(task, return_future=True)
        if fut is None:
            fut = self.put(task, return_future=True) if self.accepts_work() else None
        if fut is None:
            holder = self._claim.holder if self._claim is not None else None
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', member, holder.kind if holder is not None else None
            )
        return fut.result(timeout=timeout_s)

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
        who = getattr(task.action, '__name__', None) or repr(task.action)
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
        executor was disabled outright, and a protocol fenced it. A caller that
        must know BEFORE submitting asks this instead of re-deriving the two
        conditions, because a second copy of them drifts from the ones ``put``
        actually enforces, and the drift is invisible -- the task is dropped and
        the caller is told nothing. ``put`` reads them from here for the same
        reason, so there is exactly one place they are written down.

        Does NOT describe ``protocol_put``, whose fence runs the other way: it
        requires a protocol to be running and drops when none is.
        """
        if self._disable:
            return False
        return not (self.protocol_running.is_set() and not self.protocol_finish.is_set())

    def put(
        self, task: IOTask, return_future: bool = False, *, override: object | None = None
    ) -> object | None:
        """Add an IOTask to the default execution queue.

        Return value reports the enqueue outcome so a caller can tell whether
        the task will actually run:

        - return_future True, enqueued: the task's waiter (await its result).
        - return_future False, enqueued: ENQUEUED.
        - executor disabled or fenced by a running protocol: None (dropped).
        - droppable_live task over the in-flight cap: LIVE_FRAME_DROPPED.
        - the scope is held and the task is not the holder's: the refusal
          (a waiter already carrying it, when return_future).

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
            # accepts_work stays the single written-down copy of the gate; the
            # flag is read again only to attribute the cause, not to re-derive
            # the condition.
            return self._refuse_submit(
                _LANE_DEFAULT,
                'the executor is disabled'
                if self._disable
                else 'a protocol run has this lane fenced',
                task,
            )
        refusal = self._claim_refusal(task)
        if refusal is not None:
            return self._refused_at_submit(task, refusal, return_future)

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
        if return_future:
            fut = _claim_waiter()
            with self._caller_futures_lock:
                self.caller_futures[task] = fut
                self._caller_futures_alloc_count += 1
        else:
            fut = None
        if profile_trace.ENABLE_PROFILE_TRACE:
            task._t_enqueue = time.monotonic()
            task._queue_depth_at_enqueue = self.queue.qsize() + (1 if self._running_task else 0)
            task._queue_kind = 'default'
        self.queue.put(task)
        self._accept_submit(_LANE_DEFAULT)
        return fut if return_future else ENQUEUED

    def _claim_protocol_future(self, task: IOTask, return_future: bool):
        """Register a per-thread reusable waiter for a protocol enqueue; see
        _claim_waiter for the rationale (kernel-handle allocation pressure
        mitigation). Returns None for fire-and-forget submissions."""
        if not return_future:
            return None
        fut = _claim_waiter()
        with self._caller_futures_lock:
            self.caller_futures[task] = fut
            self._caller_futures_alloc_count += 1
        return fut

    def _discard_protocol_future(self, task: IOTask, return_future: bool) -> None:
        """Drop a claimed waiter for a task that never entered the queue, so
        the caller isn't left holding a never-completed future that pins
        memory."""
        if not return_future:
            return
        with self._caller_futures_lock:
            if self.caller_futures.pop(task, None) is not None:
                self._caller_futures_pop_count += 1

    def _finish_protocol_enqueue(self, task: IOTask, fut, return_future: bool):
        """Success tail shared by the drop-on-full and blocking enqueues."""
        # Early warning that file writes are falling behind, before the
        # bound is reached.
        self._accept_submit(_LANE_PROTOCOL)
        depth = self.protocol_queue.qsize()
        if depth > 20 and depth % 10 == 0:
            logger.warning(
                f'[{self.executor_name}] Protocol queue depth: {depth} -- '
                f'file writes may be falling behind'
            )
        # A return_future caller needs the future back to await it; a
        # fire-and-forget caller gets PROTOCOL_ENQUEUED so it can distinguish
        # this real enqueue from a dropped task -- disabled, not-running, and
        # queue-full all return a non-PROTOCOL_ENQUEUED value.
        if return_future:
            return fut
        return PROTOCOL_ENQUEUED

    def protocol_put(self, task: IOTask, return_future: bool = False) -> object | None:
        """Add an IOTask to the protocol execution queue.

        The protocol queue only drains while a protocol is in session (after
        protocol_start). Return value reports the enqueue outcome so a caller
        can tell whether the task will actually run:

        - return_future True, enqueued: the task's Future (await its result).
        - return_future False, enqueued: PROTOCOL_ENQUEUED.
        - queue full (bounded queue at cap): PROTOCOL_QUEUE_FULL.
        - executor disabled or protocol not running: None (task dropped).
        - the scope is held and the task is not the holder's: the refusal
          (a waiter already carrying it, when return_future).

        Every non-PROTOCOL_ENQUEUED / non-Future outcome means the task did
        not enter the queue and will never run. Callers whose task must
        not be droppable use protocol_put_wait instead.
        """
        self._stamp(task, None)
        if self._disable:
            return self._refuse_submit(_LANE_PROTOCOL, 'the executor is disabled', task)

        if not self.protocol_running.is_set():
            return self._refuse_submit(_LANE_PROTOCOL, 'no protocol run is in session', task)

        refusal = self._claim_refusal(task, door=True)
        if refusal is not None:
            return self._refused_at_submit(task, refusal, return_future)

        fut = self._claim_protocol_future(task, return_future)
        if profile_trace.ENABLE_PROFILE_TRACE:
            task._t_enqueue = time.monotonic()
            task._queue_depth_at_enqueue = self.protocol_queue.qsize() + (
                1 if self._running_task else 0
            )
            task._queue_kind = 'protocol'

        # Bounded queues use put_nowait so an overflowing save thread
        # surfaces a drop signal instead of blocking the caller. Unbounded
        # queues (default, backwards compat) never raise Full, so put_nowait
        # is identical to put().
        try:
            self.protocol_queue.put_nowait(task)
        except queue.Full:
            self._protocol_queue_dropped_count += 1
            depth = self.protocol_queue.qsize()
            # Throttle the warning to avoid log inflation on a sustained
            # overflow (per drop would mirror the queue depth growth we're
            # already trying to bound).
            if (
                self._protocol_queue_dropped_count == 1
                or self._protocol_queue_dropped_count % 10 == 0
            ):
                logger.warning(
                    f'[{self.executor_name}] PROTOCOL QUEUE FULL '
                    f'(maxsize={self.protocol_queue_maxsize}, depth={depth}) -- '
                    f'dropping task; total drops this run: '
                    f'{self._protocol_queue_dropped_count}'
                )
            self._discard_protocol_future(task, return_future)
            return PROTOCOL_QUEUE_FULL
        return self._finish_protocol_enqueue(task, fut, return_future)

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

    def protocol_put_wait(
        self,
        task: IOTask,
        *,
        should_abort: Callable[[], bool],
        stall_timeout_s: float,
        return_future: bool = False,
    ) -> object | None:
        """Blocking counterpart to protocol_put: wait for a queue slot
        instead of dropping the task.

        On a full bounded queue the caller blocks, pacing the producer to
        disk drain, so a submitted task is never silently dropped. Return
        value reports the outcome:

        - return_future True, enqueued: the task's Future (await its result).
        - return_future False, enqueued: PROTOCOL_ENQUEUED.
        - PROTOCOL_QUEUE_WEDGED: the queue stayed full past stall_timeout_s
          AND the in-flight task has run past its (per-task-aware) stall
          threshold -- the writer is stuck, not slow. The task did not
          enter the queue; the caller owns the user-facing consequence.
        - None: should_abort() went true while waiting (cancelled, not
          dropped), or the executor is disabled / no protocol in session.
        - the refusal, on a lane that asks the claim, when the scope is held
          and the task is not the holder's.

        A queue that is still retiring tasks never trips the wedge return;
        the wait simply continues and the run paces to the disk.

        Args:
            should_abort: zero-arg callable polled between slot attempts;
                passed down by the caller so the executor never reaches up
                for run state.
            stall_timeout_s: minimum full-queue wait before a wedge may be
                declared; also the floor for the no-retirement window.
        """
        self._stamp(task, None)
        if self._disable:
            return self._refuse_submit(_LANE_PROTOCOL, 'the executor is disabled', task)

        if not self.protocol_running.is_set():
            return self._refuse_submit(_LANE_PROTOCOL, 'no protocol run is in session', task)

        refusal = self._claim_refusal(task, door=True)
        if refusal is not None:
            return self._refused_at_submit(task, refusal, return_future)

        fut = self._claim_protocol_future(task, return_future)
        if profile_trace.ENABLE_PROFILE_TRACE:
            # Stamped once at entry so queue_wait_ms includes the blocked
            # wait -- from the producer's view that IS queue wait.
            task._t_enqueue = time.monotonic()
            task._queue_depth_at_enqueue = self.protocol_queue.qsize() + (
                1 if self._running_task else 0
            )
            task._queue_kind = 'protocol'

        waited_s = 0.0
        while True:
            try:
                self.protocol_queue.put(task, timeout=_BACKPRESSURE_POLL_S)
                break
            except queue.Full:
                waited_s += _BACKPRESSURE_POLL_S
                self._backpressure_blocked_s += _BACKPRESSURE_POLL_S
                if (
                    not self._slow_write_warned
                    and self._backpressure_blocked_s >= SLOW_WRITE_BLOCKED_WARN_S
                ):
                    self._slow_write_warned = True
                    logger.warning(
                        f'[{self.executor_name}] Capture has spent '
                        f'{self._backpressure_blocked_s:.0f}s this run waiting '
                        f'for the save disk -- writes are not keeping up with '
                        f'capture demand'
                    )
                if should_abort():
                    self._discard_protocol_future(task, return_future)
                    return None
                in_flight_s = self._running_task_in_flight_s()
                if (
                    waited_s >= stall_timeout_s
                    and in_flight_s is not None
                    and in_flight_s >= self._stall_threshold_s(stall_timeout_s)
                ):
                    logger.error(
                        f'[{self.executor_name}] PROTOCOL QUEUE WEDGED -- full '
                        f'for {waited_s:.0f}s; in flight: '
                        f'{self.describe_running_task()}'
                    )
                    self._discard_protocol_future(task, return_future)
                    return PROTOCOL_QUEUE_WEDGED
        return self._finish_protocol_enqueue(task, fut, return_future)

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
        self._protocol_taking = taking
        # Clear stale finish flag from previous run. If protocol_finish is
        # still set (dispatcher hasn't processed it yet), clear it now so
        # the dispatcher doesn't asynchronously call protocol_end() during
        # the new run -- that would clear protocol_running mid-execution.
        if self.protocol_finish.is_set():
            self.protocol_finish.clear()
            logger.info(f'{self.name} Cleared stale protocol_finish flag')
        # Reset per run so the dropped-capture count -- and the "this run" line
        # in the overflow warning -- reflect only this run, not every run since
        # the app launched. The blocked-wait total and its warning latch are
        # per-run for the same reason.
        self._protocol_queue_dropped_count = 0
        self._backpressure_blocked_s = 0.0
        self._slow_write_warned = False
        self.protocol_running.set()
        logger.info(f'{self.name} Protocol Started')

    def protocol_end(self):
        was_running = self.protocol_running.is_set()
        self.protocol_running.clear()
        # Clear completion callback when protocol ends prematurely
        self.protocol_complete_callback = None
        self.protocol_complete_cb_args = ()
        self.protocol_complete_cb_kwargs = {}
        if was_running:
            logger.info(f'{self.name} Protocol Ended')
            self._fire_protocol_idle_listeners()

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

    def protocol_finish_then_end(self):
        self.protocol_finish.set()
        logger.info(f'{self.name} set to complete protocol then end')

    def end_protocol_mode(self):
        """Idempotent safety net for teardown paths that left this executor
        in protocol-mode.

        While ``protocol_running`` is set the worker pulls only from
        ``protocol_queue``; if a protocol aborts or tears down without the
        normal completion path ending this executor, the worker blocks
        forever on ``protocol_queue.get`` and the normal queue (composite,
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

    def set_protocol_complete_callback(self, callback, cb_args=None, cb_kwargs=None):
        """Register callback to be invoked when protocol queue is fully drained."""
        with self._callback_lock:
            self.protocol_complete_callback = callback
            self.protocol_complete_cb_args = cb_args if cb_args is not None else ()
            self.protocol_complete_cb_kwargs = cb_kwargs if cb_kwargs is not None else {}

    def add_protocol_idle_listener(self, listener) -> None:
        """Register a PERSISTENT level-change listener for protocol-mode exits.

        The one-shot ``protocol_complete_callback`` slot is consume-once
        and already owned by the run-cleanup chain; a second consumer
        registering there would clobber it or be clobbered. Listeners
        added here survive across runs and fire (with no payload, on
        the transitioning thread -- the worker on the drain path) every
        time this executor leaves protocol mode, so they must re-read
        whatever level they care about rather than trust edge context.
        """
        with self._callback_lock:
            self._protocol_idle_listeners.append(listener)

    def _fire_protocol_idle_listeners(self) -> None:
        with self._callback_lock:
            listeners = list(self._protocol_idle_listeners)
        for listener in listeners:
            try:
                listener()
            except Exception:
                logger.exception(f'{self.name} protocol-idle listener failed')

    def is_protocol_queue_active(self) -> bool:
        """Returns True if protocol queue has pending tasks or a protocol task is running.

        Does NOT include protocol_finish flag -- that flag only signals the
        dispatcher to drain remaining items, and clears asynchronously on the
        next dispatch cycle (~0.2s). Including it here caused back-to-back
        protocol runs to be blocked for up to 200ms after the queue was
        already empty (the run_complete callback fires before protocol_finish
        clears).
        """
        return not self.protocol_queue.empty() or (
            self.running_task is not None and getattr(self.running_task, 'protocol', False)
        )

    def wait_for_task(self, task: IOTask, timeout: float):
        with self._caller_futures_lock:
            if task not in self.caller_futures:
                return
            fut = self.caller_futures[task]

        try:
            fut.result(timeout=timeout)
        except Exception as e:
            logger.error(f'{self.name} Worker Error: {e}')

    def _run_loop(self):
        my_generation = self._worker_generation
        if self.lane:
            _lane_worker.executor = self
        while True:
            if self._worker_generation != my_generation:
                return
            try:
                task = None
                if self._claim is not None and not self.queue.empty():
                    self._refuse_queued()
                try:
                    if self.protocol_running.is_set() or self.protocol_finish.is_set():
                        task = self.protocol_queue.get(block=True, timeout=0.2)
                        task.protocol = True
                    elif not self.protocol_queue.empty():
                        self.clear_protocol_pending()
                        continue
                    else:
                        task = self.queue.get(block=True, timeout=0.2)
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
                    if self.protocol_finish.is_set():
                        # Capture callback locals BEFORE protocol_end --
                        # protocol_end clears protocol_complete_callback for
                        # the premature-end path, so reading it after would
                        # always be None on the normal-drain path here.
                        with self._callback_lock:
                            _cb = self.protocol_complete_callback
                            _cb_args = self.protocol_complete_cb_args
                            _cb_kwargs = self.protocol_complete_cb_kwargs
                            self.protocol_complete_callback = None
                            self.protocol_complete_cb_args = ()
                            self.protocol_complete_cb_kwargs = {}
                        self.protocol_end()
                        self.protocol_finish.clear()
                        if _cb is not None:
                            self._ui_dispatch(
                                lambda dt, cb=_cb, a=_cb_args, k=_cb_kwargs: cb(*a, **k),
                                0,
                            )
                    continue

                if (
                    not (self.protocol_running.is_set() or self.protocol_finish.is_set())
                    and not self.protocol_queue.empty()
                ):
                    self.protocol_queue.queue.clear()
                    # Late enqueues discarded outside protocol mode flip
                    # is_protocol_queue_active back to False; level
                    # listeners re-read it.
                    self._fire_protocol_idle_listeners()
                if self.pending_shutdown:
                    # Drop the claim taken at dequeue: this task will never
                    # run, and a claim left on a stopping worker answers
                    # "busy" forever to anything still polling.
                    self.running_task = None
                    self._running_task_started_monotonic = None
                    return

                task._ui_dispatch = self._ui_dispatch

                # Asked again as it leaves the queue: a hold can begin while
                # the task waits, and the holder's restore must not be run
                # over by work queued before it.
                refusal = self._claim_refusal(task)
                if refusal is not None:
                    logger.warning(
                        f'[{self.executor_name}] REFUSED {refusal.member} at dequeue -- {refusal}'
                    )
                    self._on_task_done(task, None, refusal)
                    continue

                run_result = None
                run_exc = None
                try:
                    with acting(task.taking):
                        run_result = task.run()
                except BaseException as e:
                    run_exc = e

                if self._worker_generation != my_generation:
                    # Abandoned by wedge recovery while stuck inside this
                    # task. The replacement worker owns ALL executor state
                    # now; running the epilogue here would fire a stale
                    # task-failure popup, clobber the replacement's
                    # running_task and timing stamps, complete a cancelled
                    # future, and unbalance queue bookkeeping the recovery
                    # already reconciled. Exit without touching anything.
                    logger.warning(
                        f'[{self.executor_name}] Abandoned worker finished '
                        f'{getattr(task.action, "__name__", str(task.action))} '
                        f'after wedge recovery; exiting without epilogue'
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
            logger.warning(
                f'[{self.executor_name}] REFUSED {refusal.member} while queued -- {refusal}'
            )
            task._ui_dispatch = self._ui_dispatch
            task.protocol = False
            with self._caller_futures_lock:
                fut = self.caller_futures.pop(task, None)
                if fut is not None:
                    self._caller_futures_pop_count += 1
            if fut is not None:
                fut.set_exception(refusal)
            self._report_task_failure(task, refusal)
            task.on_complete(None, refusal)

    def _report_task_failure(self, task: IOTask, exception: BaseException) -> None:
        """Tell the person about a task that failed or was refused, unless its caller will."""
        if isinstance(exception, CancelledError):
            logger.debug(
                f'[{self.name}] '
                f'{getattr(task.action, "__name__", str(task.action))} '
                f'cancelled (by-contract)'
            )
        elif getattr(task, 'silent_on_failure', False):
            # Caller opted in to handle its own notification (API/caller
            # decides, not the executor). Exception is still logged at
            # ERROR via IOTask.run() and captured in the exception
            # passed to the callback. Suppress the generic "Task failed"
            # popup. Used for the protocol image-writer retry path
            # where per-failure popups would stack
            # (see protocol_image_writer.execute_step).
            pass
        elif isinstance(exception, Refusal):
            # Its own title and words, as a warning: the person asked
            # for something the scope declined, and nothing failed.
            #
            # Outside a run, someone asked for this task, so the refusal
            # is an answer: never filtered as a repeat -- a second
            # out-of-range press that showed nothing looked like a dead
            # button -- and replacing the last refusal popup, as every
            # other refusal does. A run's own task keeps the run's mute:
            # mid-run only a fatal error may pop up. The worker marks
            # which queue a task came off, and serves only the run's
            # queue while a run is going.
            action_name = getattr(task.action, '__name__', str(task.action))
            notifications.warning(
                f'Task:{action_name}',
                exception.title,
                str(exception),
                solicited=not task.protocol,
                operation_key=REFUSAL_OPERATION_KEY,
            )
        else:
            # Typed exceptions (CaptureError / ProtocolError / etc.)
            # carry a user-friendly message in str(exception); show
            # that directly. Untyped exceptions get a generic message
            # so the popup doesn't leak raw Python class names; the
            # full trace is already in the log via _run_task above.
            from modules.exceptions import (
                CaptureError,
                ConfigError,
                MoveNotCompletedError,
                ProtocolError,
            )

            try:
                from drivers.exceptions import HardwareError

                typed = (
                    CaptureError,
                    ProtocolError,
                    ConfigError,
                    HardwareError,
                    MoveNotCompletedError,
                )
            except ImportError:
                typed = (
                    CaptureError,
                    ProtocolError,
                    ConfigError,
                    MoveNotCompletedError,
                )
            action_name = getattr(task.action, '__name__', str(task.action))
            if isinstance(exception, typed) and str(exception):
                body = str(exception)
            else:
                # Says only what the queue proves. Calling it a "step"
                # and warning that one may have been skipped described
                # end-of-scan maintenance, camera restore and data saves
                # as steps, and fired 81 times in a run where nothing was
                # skipped and every frame wrote. Which queue a task came
                # off is not evidence about steps, so it no longer
                # changes what the body claims -- only the title, which
                # says where the failure happened and nothing more.
                body = 'The operation did not complete. Check the main log for details.'
            # What the user reads has to be prose. The action is a
            # Python symbol, and for a partial or a lambda str() yields
            # a repr carrying a heap address. A refusal that brought its
            # own title already said this better than a generic line.
            title = getattr(exception, 'title', None) or (
                'Protocol operation failed' if task.protocol else 'Background operation failed'
            )
            # Identity rides in the CATEGORY. Dedup keys on
            # (category, title) and no popup renders the category, so
            # distinct actions keep distinct keys -- a refused stage
            # move and an unrelated camera failure seconds apart must
            # not collapse into one event -- while a genuine repeat of
            # one action still dedups. Carrying that identity in the
            # TITLE instead is what put Python identifiers, and for a
            # partial a heap address, in front of the user; an address
            # also made the key unmatchable, silently disabling dedup.
            notifications.error(f'Task:{action_name}', title, body)

    def _on_task_done(self, task: IOTask, result, exception):
        # Receives (result, exception) from worker, then schedules task.on_complete
        if exception is not None:
            self._report_task_failure(task, exception)
        self.last_task_done_monotonic = time.monotonic()

        # Threading audit -- emit per-IOTask timing row when opt-in tracing
        # is enabled. Fields answer "which lane starved?" (queue_wait_ms per
        # lane per time bucket), "which actions are slow?" (exec_ms per
        # action name), and "does queue depth correlate with wait?"
        # (queue_depth_at_enqueue). See lib/profile_trace.py for the unified
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
        with self._caller_futures_lock:
            caller_fut = self.caller_futures.pop(task, None)
            if caller_fut is not None:
                self._caller_futures_pop_count += 1
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
        if task.protocol:
            if not self.cleared_protocol_queue:
                self.protocol_queue.task_done()
            else:
                self.clear_protocol_pending()
                self.cleared_protocol_queue = False
        else:
            if not self.cleared_queue:
                self.queue.task_done()
            else:
                self.clear_pending()
                self.cleared_queue = False

        self.running_task = None
        self._running_task_started_monotonic = None
        if self.global_callback is not None:
            self._ui_dispatch(
                lambda dt: self.global_callback(*self.global_cb_args, **self.global_cb_kwargs), 0
            )

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

    def set_done_callback(self, callback_fn, cb_args, cb_kwargs):
        # Allows to set a callback for when any IO task finishes (universal)
        self.global_callback = callback_fn
        self.global_cb_args = cb_args
        self.global_cb_kwargs = cb_kwargs

    def shutdown(self, wait: bool = True) -> None:
        self.pending_shutdown = True
        self.enable()
        self.protocol_end()
        self.clear_pending()
        self.clear_protocol_pending()
        if self.worker_alive and wait:
            # Worker polls pending_shutdown on every queue.get timeout
            # (0.2s); bound the join so a hung task does not block
            # process exit indefinitely.
            self._worker_thread.join(timeout=5.0)

        self.global_callback = None
        self.global_cb_args = None
        self.global_cb_kwargs = None

        with self._caller_futures_lock:
            self._caller_futures_pop_count += len(self.caller_futures)
            self.caller_futures.clear()
        self.running_task = None
        self._running_task_started_monotonic = None

    def join(self, timeout=None):
        # Block until all queued tasks processed (or until timeout)
        pass

    def clear_pending(self):
        """Drain the default queue, cancelling each pending task's Future.

        For priority_aware executors the drain order is HIGH-first --
        if a HIGH cancel callback must run before a MED cancel callback
        (e.g. abort signal ordering), that's the right semantic.
        """
        # Remove all tasks still in queue
        cleared_count = 0
        while True:
            try:
                task = self.queue.get_nowait()
                # Cancel future and aggressively cleanup
                with self._caller_futures_lock:
                    fut = self.caller_futures.pop(task, None)
                    if fut is not None:
                        self._caller_futures_pop_count += 1
                if fut:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                cleared_count += 1
                # Balance out get_nowait with a task_done
                self.queue.task_done()
            except queue.Empty:
                break

        self.cleared_queue = True
        if cleared_count > 0:
            logger.info(f'{self.name} Pending Queue Cleared ({cleared_count} tasks)')

    def clear_protocol_pending(self):
        cleared_count = 0
        while True:
            try:
                task = self.protocol_queue.get_nowait()
                # Cancel future and aggressively cleanup
                with self._caller_futures_lock:
                    fut = self.caller_futures.pop(task, None)
                    if fut is not None:
                        self._caller_futures_pop_count += 1
                if fut:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                cleared_count += 1
                # Balance out get_nowait with a task_done
                self.protocol_queue.task_done()
            except queue.Empty:
                break

        self.cleared_protocol_queue = True
        if cleared_count > 0:
            logger.info(f'{self.name} Pending Protocol Queue Cleared ({cleared_count} tasks)')

    def is_busy(self):
        # Returns true if tasks queued or running
        return not (self.queue.empty() and self.running_task is None)

    def queue_size(self) -> int:
        return self.queue.qsize()

    def protocol_queue_size(self) -> int:
        """Returns the number of pending protocol tasks, including any currently running task."""
        queue_count = self.protocol_queue.qsize()
        # Add 1 if there's a currently running protocol task
        if self.running_task is not None and getattr(self.running_task, 'protocol', False):
            queue_count += 1
        return queue_count

    def protocol_dropped_count(self) -> int:
        """Captures dropped this run because the bounded write queue was full.

        Each dropped task is one already-grabbed frame the writer could not
        keep up with, so it was never saved -- the run's owner reads this at
        the end to tell the user about the lost images. Reset at protocol_start.
        """
        return self._protocol_queue_dropped_count

    def protocol_backpressure_blocked_s(self) -> float:
        """Total seconds this run's blocking protocol enqueues spent waiting
        for a queue slot -- the demand-relative slow-save-disk signal. Reset
        at protocol_start."""
        return self._backpressure_blocked_s

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

    def protocol_drain_stalled(self, threshold_s: float) -> bool:
        """True when the protocol queue still gates operations but its
        in-flight task is stuck (``in_flight_task_stalled``)."""
        if not self.is_protocol_queue_active():
            return False
        return self.in_flight_task_stalled(threshold_s)

    def recover_wedged_protocol_queue(self) -> None:
        """User-invoked recovery for a wedged protocol worker: discard
        pending protocol tasks, exit protocol mode, and -- when the worker
        is still stuck inside a task (unkillable in Python) -- abandon it
        and start a replacement worker.

        The abandoned worker is a daemon thread; when its stuck call
        finally returns it sees the moved-on generation and exits without
        touching executor state. Exactly one ACTIVE worker exists at all
        times. Replacing (rather than discard-only) is what revives the
        FILE lane behind the stuck task and lets a deferred
        protocol-complete callback (files_complete) finally fire.
        """
        stuck = self.running_task
        logger.error(
            f'[{self.executor_name}] Wedged-queue recovery invoked -- '
            f'discarding {self.protocol_queue.qsize()} pending task(s); '
            f'in flight: {self.describe_running_task()}'
        )
        if stuck is not None:
            # Quarantine the stuck worker FIRST: were it to finish between
            # the queue clear and the generation bump, its epilogue would
            # run task_done bookkeeping against the already-cleared queue.
            self._worker_generation += 1
        self.clear_protocol_pending()
        self.end_protocol_mode()
        if stuck is not None:
            with self._caller_futures_lock:
                fut = self.caller_futures.pop(stuck, None)
                if fut is not None:
                    self._caller_futures_pop_count += 1
            if fut is not None:
                try:
                    fut.cancel()
                except Exception as ex:
                    logger.debug(f'[{self.executor_name}] stuck-task future cancel: {ex}')
            self.start()
            # The orphan's guarded epilogue will not clear these and the
            # replacement only sets them when it dequeues something --
            # left stale, every is_protocol_queue_active gate would keep
            # refusing forever, which is the lockout recovery exists to end.
            self.running_task = None
            self._running_task_started_monotonic = None
