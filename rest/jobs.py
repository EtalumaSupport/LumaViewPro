# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Anything slow is a job, for every call alike.

A call runs on a thread of its own; its route waits what the client said
it will wait (``Prefer: wait=<seconds>``, RFC 7240; 2 s when it says
nothing, at most 60 s). A call still running then is answered
``202 Accepted`` with the job's address, and the client reads the job --
``GET /api/v1/jobs/<id>``, itself waiting by ``Prefer: wait`` -- for the
answer the call would have given. A running call a member hands back (a
``Future``) is a job too.

A job is ``{id, member, status, requested, ended, progress}`` and, once it
has ended, ``result`` or ``error``: ``status`` goes ``pending`` ->
``running`` -> ``completed`` | ``failed``, ``progress`` is the member's own
report of how far it has got, and ``error`` is the problem the call ended
in. Reading a failed job is 200: the read succeeded. There is no generic
cancel: each activity that can be stopped has its own member. A finished
job is kept until a client forgets it, or until the count or age bound
passes it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import threading
from collections.abc import Callable
from concurrent.futures import Future

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


def _now() -> datetime.datetime:
    return datetime.datetime.now().astimezone()


@dataclasses.dataclass(frozen=True)
class Wait:
    """How long a client will wait, and whether it said so.

    Attributes:
        seconds: The wait applied.
        asked: Whether the client named one, which the answer acknowledges
            (``Preference-Applied``).
    """

    seconds: float
    asked: bool

    @classmethod
    def of(cls, prefer: str | None) -> Wait:
        """The wait a ``Prefer`` header asks for; a preference not understood is ignored (RFC 7240)."""
        for token in (prefer or '').split(','):
            name, _, value = token.strip().partition('=')
            if name.strip().lower() == 'wait' and value.strip().isdigit():
                return cls(min(float(value.strip()), WAIT_CAP_S), True)
        return cls(WAIT_DEFAULT_S, False)

    def headers(self) -> dict[str, str]:
        """The ``Preference-Applied`` header, when the client asked."""
        return {'Preference-Applied': f'wait={int(self.seconds)}'} if self.asked else {}


class Progress:
    """How far a call has got, as its member last said: the server's ``ProgressCallback``."""

    def __init__(self) -> None:
        self._last: dict[str, object] | None = None

    def __call__(self, percent: float, detail: str | None = None) -> None:
        # One assignment: a reader on another thread sees the old or the new.
        self._last = {'percent': float(percent), 'detail': detail}

    @property
    def last(self) -> dict[str, object] | None:
        """``{"percent", "detail"}``, or None before the member has said anything."""
        return self._last


class Job:
    """A call a client was told is still running, and the answer it ends in."""

    def __init__(
        self,
        job_id: str,
        member: str,
        requested: datetime.datetime,
        answered: asyncio.Future[Answer],
        progress: Progress | None,
        started: Callable[[], bool],
    ) -> None:
        self.id = job_id
        self.member = member
        self.requested = requested
        self.answered = answered
        self.progress = progress
        self._started = started
        self.ended: datetime.datetime | None = None
        # Registered on the loop, whichever thread made the job: an asyncio
        # future takes callbacks only there.
        answered.get_loop().call_soon_threadsafe(answered.add_done_callback, self._end)

    def _end(self, _answered: asyncio.Future[Answer]) -> None:
        self.ended = _now()

    @property
    def finished(self) -> bool:
        """Whether the call has ended."""
        return self.answered.done()

    def view(self) -> dict[str, object]:
        """The job as a client reads it."""
        if not self.finished:
            status = 'running' if self._started() else 'pending'
        else:
            answer = self.answered.result()
            status = 'failed' if answer.is_problem else 'completed'
        view = {
            'id': self.id,
            'member': self.member,
            'status': status,
            'requested': self.requested.isoformat(),
            'ended': self.ended.isoformat() if self.ended else None,
            'progress': self.progress.last if self.progress else None,
        }
        if self.finished:
            answer = self.answered.result()
            view['error' if answer.is_problem else 'result'] = answer.body
        return view


class JobRegistry:
    """The calls running on their own threads, and the jobs clients were told of."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._next = 1
        self._live_calls = 0

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
            loop.call_soon_threadsafe(settle, answer)

        threading.Thread(target=run, name=f'rest {name}').start()
        return answered

    def adopt(
        self,
        answered: asyncio.Future[Answer],
        *,
        member: str,
        requested: datetime.datetime,
        progress: Progress | None,
    ) -> Job:
        """Hold a call that outlived the client's wait as a job."""
        return self._add(member, requested, answered, progress, started=lambda: True)

    def of_future(
        self,
        future: Future,
        *,
        member: str,
        answer: Callable[[Future], Answer],
        loop: asyncio.AbstractEventLoop,
    ) -> dict[str, object]:
        """The job a running call a member handed back is, in its wire form.

        ``answer`` turns the finished future into its answer, on the thread
        that finished it.
        """
        answered: asyncio.Future[Answer] = loop.create_future()

        def finished(done: Future) -> None:
            reply = answer(done)
            loop.call_soon_threadsafe(
                lambda: None if answered.done() else answered.set_result(reply)
            )

        job = self._add(
            member, _now(), answered, None, started=lambda: future.running() or future.done()
        )
        future.add_done_callback(finished)
        return job.view()

    def _add(
        self,
        member: str,
        requested: datetime.datetime,
        answered: asyncio.Future[Answer],
        progress: Progress | None,
        *,
        started: Callable[[], bool],
    ) -> Job:
        with self._lock:
            job_id = str(self._next)
            self._next += 1
            job = Job(job_id, member, requested, answered, progress, started)
            self._jobs[job_id] = job
            self._prune()
        return job

    def _prune(self) -> None:
        finished = sorted(
            (j for j in self._jobs.values() if j.finished and j.ended is not None),
            key=lambda j: j.ended,
        )
        oldest = _now() - FINISHED_AGE
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

    def forget(self, job_id: str) -> None:
        """Forget a finished job.

        Raises:
            ServerRefusedError: ``not_found``; ``job_running`` (409), the call
                has not ended -- its own member stops it.
        """
        job = self.get(job_id)
        if not job.finished:
            raise problems.ServerRefusedError(
                409,
                'job_running',
                'Job Running',
                f'Job {job_id} has not ended: it is forgotten once it has.',
            )
        with self._lock:
            self._jobs.pop(job_id, None)

    def listing(self) -> list[dict[str, object]]:
        """Every job held, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.view() for j in sorted(jobs, key=lambda j: int(j.id), reverse=True)]
