# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Atomic compare-and-claim for the session's one exclusive activity."""

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass

# The activity kinds that hold the WHOLE scope: a run and a diagnostic both
# drive every axis, the LEDs and the camera, so while one holds the claim a
# lane runs only work made under its taking, the controls lock and the
# objective cannot change under it. A recording is not here -- focusing,
# moving, LED, gain and exposure stay open to anyone during one.
SCOPE_HOLDING_KINDS = frozenset({'protocol', 'diagnostic'})

_acting = threading.local()


def current_taking() -> 'Taking | None':
    """The taking this thread is acting under, or None."""
    return getattr(_acting, 'taking', None)


@contextlib.contextmanager
def acting(taking: 'Taking | None') -> Iterator[None]:
    """Act under ``taking`` on this thread for the length of a ``with`` block.

    A lane stamps each task with the taking its submitter acts under, and
    while the scope is held it runs only the holder's; so every thread that
    does a holder's work enters its taking here. None acts under nothing,
    which lets a caller hand on whatever taking it was given without a
    branch. Nests: the previous taking is restored on exit.
    """
    previous = current_taking()
    _acting.taking = taking
    try:
        yield
    finally:
        _acting.taking = previous


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

    @property
    def claim(self) -> 'ActivityClaim':
        """The claim this taking was taken from."""
        return self._claim

    @property
    def holds(self) -> bool:
        """Whether this taking still holds the claim.

        False once it is released, and stays False: a later taking of the
        same claim is a different object.
        """
        return self._claim._is_held_by(self)

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
    release leaves that claim held -- the lender releases at its own end.

    It answers the rest of a held claim's questions the same way, so work
    that runs under it (a run inside a diagnostic, and a recording inside
    that run) needs no branch on whether it borrowed. It holds until it is
    released or its lender stops holding, and it lends itself: what it lent
    ends when it does, so a recording inside a borrowed run cannot outlive
    the run, and a lane stops taking an ended run's work while the lender
    still holds the scope.
    """

    __slots__ = ('_ended', '_lender')

    def __init__(self, lender: 'Taking') -> None:
        self._lender = lender
        self._ended = False

    @property
    def claim(self) -> 'ActivityClaim':
        """The claim the lender took, at the root of the lending chain."""
        return self._lender.claim

    @property
    def holds(self) -> bool:
        """Whether this borrowing has not ended and its lender still holds."""
        return not self._ended and self._lender.holds

    def release(self) -> None:
        """End this borrowing and everything lent from it; the lender keeps its claim."""
        self._ended = True

    def lend(self) -> 'BorrowedClaim':
        """Lend this borrowing to work nested inside it."""
        return BorrowedClaim(self)


class BorrowedClaim:
    """A held claim lent to work inside the holder's activity.

    A recording inside a run acts under the run's claim: its start is
    granted while the run still holds the claim, and its end cannot free
    it. The same shape as ActivityClaim to that work, so the work does
    not know whether it runs alone or inside another activity.
    """

    __slots__ = ('_lender',)

    def __init__(self, lender: 'HeldClaim | _Borrowing') -> None:
        self._lender = lender

    @property
    def holder(self) -> 'ActivityHolder | None':
        """The claim's current holder, as ActivityClaim.holder answers it."""
        return self._lender.claim.holder

    @property
    def blocking_holder(self) -> 'ActivityHolder | None':
        """The holder that would refuse a taking through this borrow.

        None while the lender holds: the lender is the activity this work
        runs inside, not one in its way. Once the lender has released, the
        claim's current holder, whoever took it since.
        """
        if self._lender.holds:
            return None
        return self.holder

    def try_claim(self, owner: str, run_trigger_source: str | None = None) -> _Borrowing | None:
        """Act under the lender's taking; None once the lender no longer holds."""
        if not self._lender.holds:
            return None
        return _Borrowing(self._lender)


# What a taker holds: its own taking, or a borrowing of someone else's.
# Both answer ``holds``, ``release`` and ``lend`` the same way, so the work
# they cover is written once for either.
Taking = HeldClaim | _Borrowing


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
    def blocking_holder(self) -> ActivityHolder | None:
        """The holder that would refuse a taking: the current holder.

        The same question BorrowedClaim answers, so a caller that may hold
        either asks it once without knowing which it holds.
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

    def refusing_holder(self, taking: Taking | None) -> ActivityHolder | None:
        """The holder that refuses work made under ``taking``, or None.

        While a run or a diagnostic holds the scope, only work under its
        taking -- the HeldClaim, or a borrowing of it that has not ended --
        is the holder's. Anything else is refused, and the holder is named
        so the refusal can say who has the scope.
        """
        holder = self._holder
        if holder is None or holder.kind not in SCOPE_HOLDING_KINDS:
            return None
        if taking is not None and taking.claim is self and taking.holds:
            return None
        return holder

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
