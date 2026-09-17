# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The terminal, caller-observable result of a run's post-run merge.

A caller that starts a composite learns whether the MERGE succeeded and
where the artifact landed -- not merely that the capture finished. That
answer has to survive a lifecycle with three awkward properties:

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
class MergeOutcome:
    """What actually happened to a run's merge.

    Attributes:
        merged: True only when an artifact was produced.
        artifact_path: Where it landed; None whenever merged is False.
        reason: Machine-readable cause when not merged ('aborted',
            'failed', 'shutdown', 'inputs_discarded', 'merge_timeout',
            'no_run_dir', ...). Empty string on success, so a caller can
            branch on merged and still log one field.
    """

    merged: bool
    artifact_path: str | None
    reason: str


PENDING = 'pending'
ARMED = 'armed'
RESOLVED = 'resolved'


class RunMergeOutcome:
    """One run's merge outcome, resolved exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = PENDING
        self._token: str | None = None
        self._outcome: MergeOutcome | None = None
        self._settled = threading.Event()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def arm(self) -> str | None:
        """Claim the right to say how the merge went.

        Returns the arming token, or None when the outcome is no longer
        PENDING -- which is the caller's signal not to start a merge at
        all, because something already settled this run.
        """
        with self._lock:
            if self._state != PENDING:
                return None
            self._token = uuid.uuid4().hex
            self._state = ARMED
            return self._token

    def resolve_if_pending(self, reason: str) -> bool:
        """Settle a run whose merge never started. PENDING only.

        Cleanup's resolver. Deliberately powerless over an ARMED outcome:
        the second cleanup pass of a normal run arrives carrying 'failed'
        and must not overwrite a merge that is running or already done.
        """
        return self._settle(MergeOutcome(False, None, reason), allow_armed=False)

    def force_resolve(self, reason: str) -> bool:
        """Settle from PENDING or ARMED, because nothing will finish it.

        For teardown only -- session shutdown, or a discard of the run's
        unwritten inputs. Both leave a merge unable to complete, so the
        outcome settles here rather than leaving a caller blocked on a
        result that is never coming.
        """
        return self._settle(MergeOutcome(False, None, reason), allow_armed=True)

    def resolve(self, token: str, outcome: MergeOutcome) -> bool:
        """Record how the merge went. Only the arming token holder may.

        Returns False when the token does not match -- a stale merge
        thread reporting into a run that teardown already settled, which
        is a no-op rather than an error.
        """
        with self._lock:
            if self._state != ARMED or token != self._token:
                return False
            self._state = RESOLVED
            self._outcome = outcome
            self._settled.set()
            return True

    def _settle(self, outcome: MergeOutcome, *, allow_armed: bool) -> bool:
        with self._lock:
            if self._state == RESOLVED:
                return False
            if self._state == ARMED and not allow_armed:
                return False
            self._state = RESOLVED
            self._outcome = outcome
            self._settled.set()
            return True

    def wait(self, timeout_s: float) -> MergeOutcome | None:
        """Block until the outcome settles; None when the bound expires.

        None means the run never reported -- distinct from a resolved
        outcome carrying merged=False, which is a run that DID report
        that no artifact was produced.
        """
        if not self._settled.wait(timeout=timeout_s):
            return None
        with self._lock:
            return self._outcome

    def settle_unfinished(self, reason: str) -> None:
        """Force-resolve and say so, for teardown paths that must not fail.

        Wraps force_resolve so a teardown caller cannot be the reason a
        finally block raises; the log line is what makes an outcome
        settled this way visible afterward.
        """
        try:
            if self.force_resolve(reason):
                logger.info(f'[RunOutcome] Merge outcome settled as {reason}')
        except Exception:
            logger.error('[RunOutcome] Failed to settle merge outcome', exc_info=True)
