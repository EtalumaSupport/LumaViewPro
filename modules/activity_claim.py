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


class HeldClaim:
    """The claim as held by the one caller that took it.

    A fresh object per taking, so it identifies that taking and no
    other: the next run's HeldClaim is a different object even when its
    holder description is equal. Only it releases the claim. Nothing
    else hands it out -- ``ActivityClaim.holder`` describes the holder,
    it does not return this.
    """

    __slots__ = ('_claim',)

    def __init__(self, claim: 'ActivityClaim') -> None:
        self._claim = claim

    def release(self) -> None:
        """Release the claim this taking holds.

        Raises:
            RuntimeError: this taking no longer holds the claim -- a
                release path that runs twice or late fails loudly rather
                than freeing a claim a newer activity took.
        """
        self._claim._release(self)

    def lend(self) -> 'BorrowedClaim':
        """Lend this taking to work that runs inside its activity."""
        return BorrowedClaim(self)


class _Borrowing:
    """What a borrower holds: it acts under the lender's claim, and its
    release leaves that claim held -- the lender releases at its own end."""

    __slots__ = ()

    def release(self) -> None:
        return None


class BorrowedClaim:
    """A held claim lent to work inside the holder's activity.

    A recording inside a run acts under the run's claim: its start is
    granted while the run still holds the claim, and its end cannot free
    it. The same shape as ActivityClaim to that work, so the work does
    not know whether it runs alone or inside another activity.
    """

    __slots__ = ('_lender',)

    def __init__(self, lender: HeldClaim) -> None:
        self._lender = lender

    @property
    def holder(self) -> 'ActivityHolder | None':
        """The claim's current holder, as ActivityClaim.holder answers it."""
        return self._lender._claim.holder

    def try_claim(self, owner: str, run_trigger_source: str | None = None) -> _Borrowing | None:
        """Act under the lender's claim; None once the lender no longer holds it."""
        if not self._lender._claim._is_held_by(self._lender):
            return None
        return _Borrowing()


class ActivityClaim:
    """Arbitrates the session's one exclusive activity.

    Exactly one exclusive activity (a protocol run XOR a video
    recording) may hold the claim at a time. ``try_claim`` is atomic:
    of two concurrent claimants exactly one wins. The claim is not
    reentrant -- a second ``try_claim`` fails even for the same owner,
    so a claimant that lost track of its own state cannot silently
    stack claims. Ownership is the HeldClaim the winner receives, never
    a name: anyone can spell a kind, only the taker holds the object.
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
        self._held: HeldClaim | None = None
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

    def try_claim(self, owner: str, run_trigger_source: str | None = None) -> HeldClaim | None:
        """Atomically claim for ``owner``; None when already held.

        Args:
            owner: the kind of activity claiming -- a description for
                display and refusal text, not a credential.
            run_trigger_source: which run is claiming, when the
                claimant is a run. Optional because a recording has no
                trigger; a run always passes one, which its own claim
                site is what enforces.

        Returns:
            The HeldClaim that alone can release this taking, or None
            when another activity holds the claim.
        """
        with self._lock:
            if self._holder is not None:
                return None
            held = HeldClaim(self)
            self._held = held
            self._holder = ActivityHolder(kind=owner, run_trigger_source=run_trigger_source)
        if self._on_transition is not None:
            self._on_transition()
        return held

    def _is_held_by(self, held: HeldClaim) -> bool:
        return self._held is held

    def _release(self, held: HeldClaim) -> None:
        with self._lock:
            if self._held is not held:
                held_by = self._holder.kind if self._holder is not None else None
                raise RuntimeError(
                    f'ActivityClaim: a release by a taking that does not hold the claim '
                    f'(held by {held_by!r})'
                )
            self._held = None
            self._holder = None
        if self._on_transition is not None:
            self._on_transition()
