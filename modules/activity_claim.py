# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Atomic compare-and-claim for the session's one exclusive activity."""

import threading


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
        self._owner: str | None = None
        self._on_transition = on_transition

    @property
    def owner(self) -> str | None:
        """The current holder's name, or None when unheld."""
        return self._owner

    def try_claim(self, owner: str) -> bool:
        """Atomically claim for ``owner``; False when already held."""
        with self._lock:
            if self._owner is not None:
                return False
            self._owner = owner
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
            if self._owner != owner:
                raise RuntimeError(
                    f'ActivityClaim.release({owner!r}): claim is held by {self._owner!r}'
                )
            self._owner = None
        if self._on_transition is not None:
            self._on_transition()
