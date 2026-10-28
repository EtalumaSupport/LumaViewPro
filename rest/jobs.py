# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Anything slow is a job, for every call alike.

A call runs on a thread of its own; its route waits what the client said
it will wait (``Prefer: wait=<seconds>``, RFC 7240; 2 s when it says
nothing, at most 60 s). A call still running then is answered
``202 Accepted`` with the job's address, and the client reads the job --
``GET /api/v1/jobs/<id>``, itself waiting by ``Prefer: wait`` -- for the
answer the call would have given. A running call a member hands back (an
``Operation``) is a job too.

A job is ``{id, member, status, requested, ended, progress}`` and, once it
has ended, ``result`` or ``error``: ``status`` goes ``running`` ->
``completed`` | ``failed``, ``progress`` is how far it has got -- the
member's own report, or the operation's ``progress()`` -- and ``error`` is
the problem the call ended in. Reading a failed job is 200: the read
succeeded. ``DELETE`` on a running job is its operation's stop, which only
asks; a call adopted as a job holds no operation and cannot be stopped
there. A finished job is kept until a client forgets it, or until a new
job finds it past the count or age bound.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import threading
import time
import weakref
from collections.abc import Callable

from modules.caller import Caller, calling
from modules.exceptions import ServerRefusedError
from modules.operation import Operation
from rest import problems
from rest.problems import Answer

# What a client that says nothing waits, and the most any client waits.
WAIT_DEFAULT_S = 2.0
WAIT_CAP_S = 60.0
# Seconds a client told a call is still running waits before reading again.
RETRY_AFTER_S = 1
# The most calls running at once: a call past it is refused before it runs.
LIVE_LIMIT = 256
# Finished jobs are kept until forgotten, up to this many and this old.
FINISHED_LIMIT = 1000
FINISHED_AGE = datetime.timedelta(hours=24)


@dataclasses.dataclass(frozen=True)
class Wait:
    """How long a client will wait, and the preference it said so by.

    Attributes:
        seconds: The wait applied.
        applied: The preference the wait came from, which the answer
            acknowledges (``Preference-Applied``); None when the client named
            none.
    """

    seconds: float
    applied: str | None

    @classmethod
    def of(cls, prefer: str | None) -> Wait:
        """The wait a ``Prefer`` header asks for; a preference not understood is ignored (RFC 7240).

        ``respond-async`` asks for the answer as soon as it can be given, a
        wait of none, and wins over a ``wait`` beside it.
        """
        asked: Wait | None = None
        for token in (prefer or '').split(','):
            # A preference's own parameters follow ';', and its value may be quoted.
            name, _, value = token.split(';')[0].strip().partition('=')
            name = name.strip().lower()
            value = value.strip().strip('"')
            if name == 'respond-async':
                return cls(0.0, 'respond-async')
            if name == 'wait' and value.isdigit() and asked is None:
                seconds = min(float(value), WAIT_CAP_S)
                asked = cls(seconds, f'wait={int(seconds)}')
        return asked or cls(WAIT_DEFAULT_S, None)

    def headers(self) -> dict[str, str]:
        """The ``Preference-Applied`` header, when the client asked."""
        return {'Preference-Applied': self.applied} if self.applied else {}


class Progress:
    """How far a call has got, as its member last said: the server's ``ProgressCallback``."""

    def __init__(self) -> None:
        self._last: dict[str, object] | None = None

    def __call__(self, percent: float, detail: str | None = None) -> None:
        # One assignment: a reader on another thread sees the old or the new.
        self._last = {'progress_pct': float(percent), 'detail': detail}

    @property
    def last(self) -> dict[str, object] | None:
        """``{"progress_pct", "detail"}``, or None before the member has said anything."""
        return self._last


class Job:
    """A call a client was told is still running, and the answer it ends in.

    Attributes:
        operation: The running call a member handed back, whose stop a
            ``DELETE`` asks; None for a call adopted as a job, which holds
            none.
    """

    def __init__(
        self,
        job_id: str,
        member: str,
        requested: datetime.datetime,
        answered: asyncio.Future[Answer],
        progress: Callable[[], object],
        operation: Operation | None,
    ) -> None:
        self.id = job_id
        self.member = member
        self.requested = requested
        self.answered = answered
        self._progress = progress
        self.operation = operation

    @property
    def finished(self) -> bool:
        """Whether the call has ended."""
        return self.answered.done()

    @property
    def ended(self) -> datetime.datetime | None:
        """When the call ended, as its answer says; None while it runs."""
        return self.answered.result().ended if self.finished else None

    def view(self) -> dict[str, object]:
        """The job as a client reads it, its progress read now.

        An operation's progress is read under its own locks (a run's under
        the run's), so a caller on the event loop reads a job that holds
        one through a thread.
        """
        if not self.finished:
            status = 'running'
        else:
            answer = self.answered.result()
            status = 'failed' if answer.is_problem else 'completed'
        view = {
            'id': self.id,
            'member': self.member,
            'status': status,
            'requested': self.requested.isoformat(),
            'ended': self.ended.isoformat() if self.ended else None,
            'progress': self._progress(),
        }
        if self.finished:
            answer = self.answered.result()
            view['error' if answer.is_problem else 'result'] = answer.body
        return view


class JobRegistry:
    """The calls running on their own threads, and the jobs clients were told of."""

    def __init__(self, *, ended: Callable[[Job], None]) -> None:
        """Make a registry that holds no job yet.

        Args:
            ended: Told each job once it has ended, on the server's loop.
        """
        self._ended = ended
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        # The job each running call a member handed back is, so one running
        # call is one job whichever answer or event names it.
        self._of_future: weakref.WeakKeyDictionary[Operation, Job] = weakref.WeakKeyDictionary()
        self._next = 1
        self._live_calls = 0
        # The calls' threads, each until it has ended, so the server's close can join them.
        self._threads: set[threading.Thread] = set()

    def admit(self, *, returns_job: bool) -> None:
        """Refuse a call while the live limit is held, before it runs.

        A call whose answer can carry a running call counts the running jobs
        too: refused after it ran, its job would be lost.

        Raises:
            ServerRefusedError: ``overloaded``, with ``Retry-After``.
        """
        with self._lock:
            running = sum(1 for j in self._jobs.values() if not j.finished)
            full = self._live_calls >= LIVE_LIMIT or (returns_job and running >= LIVE_LIMIT)
        if full:
            raise problems.overloaded(
                f'{LIVE_LIMIT} calls are running: ask again when one has ended.', RETRY_AFTER_S
            )

    def run(self, work: Callable[[], Answer], name: str, request_id: str) -> asyncio.Future[Answer]:
        """Start *work* on a thread of its own; the future it settles holds its answer.

        *work* answers rather than raises, so whoever reads its answer --
        the route, or a job read later -- reads the same one.
        """
        loop = asyncio.get_running_loop()
        answered: asyncio.Future[Answer] = loop.create_future()
        with self._lock:
            self._live_calls += 1

        def settle(answer: Answer) -> None:
            if not answered.cancelled():
                answered.set_result(answer)

        def run() -> None:
            try:
                # Every member a REST call reaches runs on this thread, so a
                # run it starts or a stop it makes records a REST client.
                with calling(Caller.REST):
                    answer = work()
            except BaseException as e:
                # A SystemExit on this thread ends the call, not the server: the
                # request is answered as the fault it is rather than left waiting.
                fault = RuntimeError(f'{name} ended with {type(e).__name__}')
                fault.__cause__ = e
                answer = problems.answered_by_member(fault, request_id)
            finally:
                with self._lock:
                    self._live_calls -= 1
                    self._threads.discard(threading.current_thread())
            loop.call_soon_threadsafe(settle, answer)

        thread = threading.Thread(target=run, name=f'rest {name}')
        with self._lock:
            self._threads.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._lock:
                self._live_calls -= 1
                self._threads.discard(thread)
            raise
        return answered

    def join(self, timeout_s: float) -> list[str]:
        """Wait up to *timeout_s* for every call's thread to end; the names of those still running."""
        deadline = time.monotonic() + timeout_s
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return [t.name for t in threads if t.is_alive()]

    def adopt(
        self,
        answered: asyncio.Future[Answer],
        *,
        member: str,
        requested: datetime.datetime,
        progress: Progress | None,
    ) -> Job:
        """Hold a call that outlived the client's wait as a job."""
        return self._add(
            member,
            requested,
            answered,
            lambda: progress.last if progress else None,
            operation=None,
        )

    def of_future(
        self,
        future: Operation,
        *,
        member: str,
        answer: Callable[[Operation], Answer],
        progress: Callable[[], object],
        loop: asyncio.AbstractEventLoop,
    ) -> dict[str, object]:
        """The job a running call a member handed back is, in its wire form.

        ``answer`` turns the finished operation into its answer, on the
        thread that finished it; ``progress`` reads its ``progress()`` in
        wire form. The same operation is the same job, however many times
        it is named.
        """
        with self._lock:
            known = self._of_future.get(future)
        if known is not None:
            return known.view()
        answered: asyncio.Future[Answer] = loop.create_future()

        def finished(done: Operation) -> None:
            reply = answer(done)
            loop.call_soon_threadsafe(
                lambda: None if answered.done() else answered.set_result(reply)
            )

        job = self._add(member, problems.now(), answered, progress, operation=future)
        with self._lock:
            self._of_future[future] = job
        future.add_done_callback(finished)
        return job.view()

    def _add(
        self,
        member: str,
        requested: datetime.datetime,
        answered: asyncio.Future[Answer],
        progress: Callable[[], object],
        *,
        operation: Operation | None,
    ) -> Job:
        with self._lock:
            job_id = str(self._next)
            self._next += 1
            job = Job(job_id, member, requested, answered, progress, operation)
            self._jobs[job_id] = job
            self._prune()
        # A job is made on the loop or on its call's thread, and a future
        # takes its callbacks only on its loop.
        answered.get_loop().call_soon_threadsafe(
            answered.add_done_callback, lambda _answered: self._ended(job)
        )
        return job

    def _prune(self) -> None:
        finished = sorted(
            (j for j in self._jobs.values() if j.finished),
            key=lambda j: j.ended,
        )
        oldest = problems.now() - FINISHED_AGE
        excess = len(finished) - FINISHED_LIMIT
        for i, job in enumerate(finished):
            if i < excess or job.ended < oldest:
                del self._jobs[job.id]

    def get(self, job_id: str) -> Job:
        """The job with id *job_id*.

        Raises:
            ServerRefusedError: ``not_found``.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise problems.not_found(f'No job {job_id} is held.')
        return job

    def operation(self, job_id: str) -> Operation:
        """The operation the job with id *job_id* is, for a client that passes it by its id.

        Raises:
            ServerRefusedError: ``not_found``; ``invalid_request``, the job
                is a call held as a job when it outlived its wait, which is
                no operation.
        """
        job = self.get(job_id)
        if job.operation is None:
            raise problems.invalid_request(
                f'Job {job_id} is a call held as a job when it outlived its wait: it is no '
                'operation.'
            )
        return job.operation

    def delete(self, job_id: str) -> Job | None:
        """Forget a finished job, or ask a running one's operation to stop.

        Blocks while the stop is sent. Returns None when the job was
        forgotten; the job, still held, when its stop was asked -- the
        operation ends on its own, and once ended the job is forgotten by
        the next ``delete``. A stop asked once its operation's work has
        ended, or of an operation with no stop (a still), does nothing.

        Raises:
            ServerRefusedError: ``not_found``; ``not_cancellable`` (409), the
                job is a call adopted as one, which holds no operation to
                stop.
            Exception: what the operation's stop raised -- a refusal of the
                stop (a home on a board that cannot stop one).
        """
        job = self.get(job_id)
        if job.finished:
            with self._lock:
                self._jobs.pop(job_id, None)
            return None
        if job.operation is None:
            raise ServerRefusedError(
                'not_cancellable',
                f'Job {job_id} is a call held as a job when it outlived its wait: it '
                'holds no operation to stop, and is forgotten once it has ended.',
            )
        with calling(Caller.REST):
            job.operation.stop()
        return job

    def listing(self) -> list[dict[str, object]]:
        """Every job held, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.view() for j in sorted(jobs, key=lambda j: int(j.id), reverse=True)]
