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

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: str | None = None

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
