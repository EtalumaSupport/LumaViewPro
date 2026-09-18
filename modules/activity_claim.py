# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Atomic compare-and-claim for the session's one exclusive activity."""

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityHolder:
    """Who holds the session's one exclusive activity.

    The kind of activity and, when that activity is a run, which run --
    one immutable object published by a single attribute store, so a
    reader that sees the holder sees the run behind it and cannot pair
    a live claim with a name read an instant earlier or later. An
    activity that is not a run carries no trigger: a recording's kind
    IS the whole answer.
    """

    kind: str
    run_trigger_source: str | None = None


class ActivityClaim:
    """Arbitrates the session's one exclusive activity.

    Exactly one exclusive activity (a protocol run XOR a video
    recording) may hold the claim at a time. ``try_claim`` is atomic:
    of two concurrent claimants exactly one wins. The claim is not
    reentrant -- a second ``try_claim`` fails even for the same owner,
    so a claimant that lost track of its own state cannot silently
    stack claims.
    """

    def __init__(self, on_transition=None) -> None:
        """Args:
        on_transition: Optional zero-argument callable invoked after
            every successful claim or release. It fires OUTSIDE this
            claim's lock but ON the transitioning thread, which may
            hold engine locks of its own -- so it must only schedule
            or notify (level-read listeners re-read state when they
            run); it must not acquire engine locks or block.
        """
        self._lock = threading.Lock()
        self._holder: ActivityHolder | None = None
        self._on_transition = on_transition

    @property
    def holder(self) -> ActivityHolder | None:
        """The current holder, or None when unheld.

        One attribute read of an immutable object, so every read stays
        lock-free and a reader cannot see a half-written holder.
        """
        return self._holder

    @property
    def owner(self) -> str | None:
        """The current holder's kind, or None when unheld."""
        holder = self._holder
        return holder.kind if holder is not None else None

    def try_claim(self, owner: str, run_trigger_source: str | None = None) -> bool:
        """Atomically claim for ``owner``; False when already held.

        Args:
            owner: the kind of activity claiming.
            run_trigger_source: which run is claiming, when the
                claimant is a run. Optional because a recording has no
                trigger; a run always passes one, which its own claim
                site is what enforces.
        """
        with self._lock:
            if self._holder is not None:
                return False
            self._holder = ActivityHolder(kind=owner, run_trigger_source=run_trigger_source)
        if self._on_transition is not None:
            self._on_transition()
        return True

    def release(self, owner: str) -> None:
        """Release ``owner``'s claim.

        Raises:
            RuntimeError: ``owner`` does not hold the claim. A release
                path that runs at the wrong time must fail loudly here
                rather than silently free another activity's claim.
        """
        with self._lock:
            held_by = self._holder.kind if self._holder is not None else None
            if held_by != owner:
                raise RuntimeError(
                    f'ActivityClaim.release({owner!r}): claim is held by {held_by!r}'
                )
            self._holder = None
        if self._on_transition is not None:
            self._on_transition()
