# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The terminal, caller-observable result of a run.

A caller that starts a run learns how the RUN ended and why, and -- for
a composite -- whether the MERGE succeeded and where the artifact
landed. One object carries both, because "the run aborted" and "the run
completed but merged nothing" are different answers a caller must be
able to tell apart. That answer has to survive a lifecycle with three
awkward properties:

  - Run cleanup executes TWICE on a normal run. The loop calls it with
    'completed', then the outer safety net calls it again with 'failed',
    whose early return sits inside the try so the finally runs a second
    time carrying a contradictory status.
  - On the 'completed' pass the merge has not run yet, so that pass can
    only ARM the outcome; it cannot say how the merge went.
  - Teardown paths (session shutdown, a discard of the run's unwritten
    inputs) must be able to settle an outcome the merge thread already
    owns, or a caller waits forever on a merge that can no longer finish.

Hence three states rather than two. A two-state form -- resolve straight
from cleanup -- reports 'failed' over the real merge result on every
normal run, because the second cleanup pass wins.

  PENDING --arm()--> ARMED(token) --resolve(token)--> RESOLVED
     |                    |
     +--resolve_if_pending/force_resolve--> RESOLVED

Ownership is what keeps the second cleanup pass harmless: once ARMED,
only the holder of the arming token can say how the merge went. Cleanup's
own resolver is first-wins over PENDING alone. Teardown force-resolves
from either state, because nothing will finish the merge afterward.

Every resolver returns a bool and never raises: they run inside cleanup's
finally, ahead of the activity-claim release, and a raise there would
leak the claim and refuse every future run.

This module also holds the run's ENDING: the status a run finished in,
and the reason, title and message for it, recorded by whatever ended the
run at the site that knows why. The ending is first-wins -- a run dies
once, and the first thing that killed it is the cause; everything after
it is consequence. Cleanup reads the record once, after the lanes it
waits on have drained, so a fault that lands during the drain is still
the ending rather than a word chosen before the last fact arrived.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid

from lvp_logger import logger


@dataclasses.dataclass(frozen=True)
class RunEnding:
    """How a run ended, and why, in the shape a refusal already uses.

    Attributes:
        status: One of 'completed', 'aborted', 'failed',
            'failed_at_start'. 'aborted' is an ending the user or a
            policy ceiling asked for; 'failed' is one the instrument
            imposed.
        reason: Machine-readable cause, stable enough for a caller to
            branch on ('motion_timeout', 'disk_space_critical',
            'stopped', ...).
        title: Short human sentence, suitable as a popup heading.
        message: The sentence a user reads. Never a raw exception
            string -- those go to the log, not into a field a remote
            caller serialises.
    """

    status: str
    reason: str
    title: str
    message: str


class EndingLatch:
    """One run's ending, recorded once by whoever got there first.

    Sixteen sites can end a run and several of them fire in sequence as
    one failure cascades: a motion timeout stops the loop, which trips
    the step runner, which trips cleanup. Last-write-wins would report
    the consequence and lose the cause, so the first record sticks and
    every later one is dropped.

    Held per run, never per runner: a drained writer from a previous run
    that fires late lands on a latch nothing reads any more.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ending: RunEnding | None = None

    def set_if_unset(self, ending: RunEnding) -> bool:
        """Record the ending; True when this caller's ending is the one kept."""
        with self._lock:
            if self._ending is not None:
                return False
            self._ending = ending
            return True

    def get(self) -> RunEnding | None:
        """The recorded ending, or None when nothing has ended the run yet."""
        with self._lock:
            return self._ending


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    """How a run ended, and what its merge produced.

    The first four fields are the run's ENDING, copied from the
    RunEnding whatever ended the run recorded; the next three describe
    the merge, and the last two describe the autofocus characterization
    data. A non-composite run carries merged=False and an empty
    merge_reason under whatever status it ended in, and a run that saved
    no autofocus data carries af_data_saved=False, so a caller reads one
    shape for every run kind.

    Attributes:
        status: One of 'completed', 'aborted', 'failed',
            'failed_at_start'.
        reason: Why the RUN ended ('stopped', 'motion_timeout', ...).
        title: Short human sentence, suitable as a popup heading.
        message: The sentence a user reads.
        merged: True only when an artifact was produced.
        artifact_path: Where it landed; None whenever merged is False.
        merge_reason: Why the MERGE produced no file ('merge_timeout',
            'no_run_dir', 'shutdown', ...). Empty on success and on
            every run that has no merge, so a caller can branch on
            merged and still log one field.
        af_data_saved: True only when autofocus characterization data
            was WRITTEN. False covers every run that asked for none, and
            every run that asked and got none -- a sweep that collected
            nothing, or one whose queued save an abort discarded.
        af_data_path: The characterization file that was written; None
            whenever af_data_saved is False. Never the folder: that is
            allocated before the sweep runs and exists even when nothing
            was written, which is the answer a caller must not be given.
    """

    status: str
    reason: str
    title: str
    message: str
    merged: bool
    artifact_path: str | None
    merge_reason: str
    af_data_saved: bool
    af_data_path: str | None

    @classmethod
    def from_ending(
        cls,
        ending: RunEnding,
        *,
        merged: bool,
        artifact_path: str | None,
        merge_reason: str,
        af_data_path: str | None = None,
    ) -> RunOutcome:
        """Compose the caller's answer from the run's recorded ending.

        The only constructor the settle paths use: the run's status and
        reason are never restated at a settle site, so the word a caller
        reads is always the one the thing that ended the run recorded.

        Takes the autofocus path rather than the pair, and derives the
        flag from it: "data was saved" and "here is the file" are one
        fact, and a constructor that accepted both could be handed a
        saved=True carrying nowhere to look -- the shape a remote caller
        cannot tell from a real delivery.
        """
        return cls(
            status=ending.status,
            reason=ending.reason,
            title=ending.title,
            message=ending.message,
            merged=merged,
            artifact_path=artifact_path,
            merge_reason=merge_reason,
            af_data_saved=af_data_path is not None,
            af_data_path=af_data_path,
        )


PENDING = 'pending'
ARMED = 'armed'
RESOLVED = 'resolved'


class PendingRunOutcome:
    """One run's outcome, resolved exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = PENDING
        self._token: str | None = None
        self._ending: RunEnding | None = None
        self._outcome: RunOutcome | None = None
        self._af_data_path: str | None = None
        self._settled = threading.Event()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def record_autofocus_data(self, path: str | None) -> None:
        """Record the characterization file this run wrote, if any.

        Held here rather than passed to a resolver because a run can
        settle down any of three paths -- cleanup's first-wins resolve,
        the merge thread's, or a teardown force-resolve -- and the data
        landed (or did not) regardless of which one gets there. Recording
        it once, on the object every path composes from, is what keeps
        the three answers from diverging.

        Recorded before the outcome settles, which is the same moment for
        every run kind: the run's cleanup knows what its autofocus wrote
        by the time it settles, because the write is queued ahead of
        cleanup's own blocking record write on the same sequential lane.
        A run that saved nothing records None and reads as such.
        """
        with self._lock:
            self._af_data_path = path

    def arm(self, ending: RunEnding) -> str | None:
        """Claim the right to say how the merge went.

        Takes the run's ending because arming is the moment cleanup has
        it in hand: the merge thread that resolves later says only what
        the merge produced, and composes the rest from what is stored
        here. Returns the arming token, or None when the outcome is no
        longer PENDING -- which is the caller's signal not to start a
        merge at all, because something already settled this run.
        """
        with self._lock:
            if self._state != PENDING:
                return None
            self._ending = ending
            self._token = uuid.uuid4().hex
            self._state = ARMED
            return self._token

    def resolve_if_pending(self, ending: RunEnding, merge_reason: str = '') -> bool:
        """Settle a run whose merge never started. PENDING only.

        Cleanup's resolver. Deliberately powerless over an ARMED outcome:
        the second cleanup pass of a normal run arrives carrying a
        'failed' ending and must not overwrite a merge that is running
        or already done.
        """
        with self._lock:
            if self._state != PENDING:
                return False
            self._state = RESOLVED
            self._outcome = RunOutcome.from_ending(
                ending,
                merged=False,
                artifact_path=None,
                merge_reason=merge_reason,
                af_data_path=self._af_data_path,
            )
            self._settled.set()
            return True

    def force_resolve(self, merge_reason: str, *, fallback: RunEnding) -> bool:
        """Settle from PENDING or ARMED, because nothing will finish it.

        For teardown only -- session shutdown, or a discard of the run's
        unwritten inputs. Both leave a merge unable to complete, so the
        outcome settles here rather than leaving a caller blocked on a
        result that is never coming.

        From ARMED the run already reported its own ending to the
        run-complete subscribers, so that ending stands and merge_reason
        is the only new fact; from PENDING the run never reached cleanup
        and has no ending of its own, so ``fallback`` -- the ending the
        tearing-down caller is imposing -- becomes it. Choosing between
        them inside the lock is what keeps a concurrent arm() from
        stamping the fallback over a real ending.
        """
        with self._lock:
            if self._state == RESOLVED:
                return False
            ending = self._ending if self._ending is not None else fallback
            self._state = RESOLVED
            self._outcome = RunOutcome.from_ending(
                ending,
                merged=False,
                artifact_path=None,
                merge_reason=merge_reason,
                af_data_path=self._af_data_path,
            )
            self._settled.set()
            return True

    def resolve(
        self,
        token: str,
        *,
        merged: bool,
        artifact_path: str | None,
        merge_reason: str,
    ) -> bool:
        """Record what the merge produced. Only the arming token holder may.

        States no run status: the ending stored at arm() is the run's,
        and a merge thread is in no position to revise it. Returns False
        when the token does not match -- a stale merge thread reporting
        into a run that teardown already settled, which is a no-op
        rather than an error.
        """
        with self._lock:
            if self._state != ARMED or token != self._token:
                return False
            assert self._ending is not None, 'ARMED without the ending arm() stores'
            self._state = RESOLVED
            self._outcome = RunOutcome.from_ending(
                self._ending,
                merged=merged,
                artifact_path=artifact_path,
                merge_reason=merge_reason,
                af_data_path=self._af_data_path,
            )
            self._settled.set()
            return True

    def wait(self, timeout_s: float | None) -> RunOutcome | None:
        """Block until the outcome settles; None when the bound expires.

        None means the run never reported -- distinct from a resolved
        outcome carrying merged=False, which is a run that DID report
        that no artifact was produced. timeout_s=None blocks until the
        run settles, the wait a caller with nothing else to do makes.
        """
        if not self._settled.wait(timeout=timeout_s):
            return None
        with self._lock:
            return self._outcome

    def settle_unfinished(self, merge_reason: str, *, fallback: RunEnding) -> None:
        """Force-resolve and say so, for teardown paths that must not fail.

        Wraps force_resolve so a teardown caller cannot be the reason a
        finally block raises; the log line is what makes an outcome
        settled this way visible afterward.
        """
        try:
            if self.force_resolve(merge_reason, fallback=fallback):
                logger.info(f'[RunOutcome] Run outcome settled unfinished as {merge_reason}')
        except Exception:
            logger.error('[RunOutcome] Failed to settle the run outcome', exc_info=True)
