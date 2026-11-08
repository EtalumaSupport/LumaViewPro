# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether the host is asked to stay awake: exactly while the scope is in use.

A sleeping computer drops the scope's USB devices and ends the run, so
while a run, a recording, a home or a diagnostic holds the scope, or a
client (Micro-Manager, a script) holds its control session, the operating
system is asked not to sleep while idle (``host_power.hold_awake``). The
session reads that level from its activity claim; this owner keeps only
the operating system's hold, never a second "in use" flag, and re-reads the
level at every change the claim announces.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from _lumascope.drivers import host_power
from _lumascope.lvp_logger import logger
from _lumascope.modules.exceptions import KeepAwakeNotHeldNotice

_WHY = 'LumaViewPro is using the microscope'


class KeepAwake:
    """Holds the host's request to stay awake while ``in_use`` reads True.

    ``sync`` reads ``in_use`` inside its own lock and holds or releases to
    match, so of two changes racing, the one applied last is the level read
    last: a run's release can never land after a recording's newer take.
    The lock guards only the hold; nothing but the operating system's call
    runs under it, and every report is made after it is left.

    It needs no close of its own: a session's close waits until nothing
    holds the scope and ends its control session, and refuses both from its
    first step, so the last change it announces releases the hold; the
    operating system drops a hold whose process has exited.
    """

    def __init__(self, in_use: Callable[[], bool]):
        """Args:
        in_use: Whether the scope is in use now; read at every ``sync``.
        """
        self._in_use = in_use
        self._lock = threading.Lock()
        self._hold: host_power.AwakeHold | None = None
        # The reasons already told this session: each is told once, as the
        # startup home holds the scope at every launch.
        self._told: set[str] = set()

    def sync(self) -> None:
        """Hold the request while the scope is in use, and only then.

        A run-state listener: it runs on the thread that changed the claim,
        so it waits on nothing but the operating system's call. A request
        not taken is told (``KeepAwakeNotHeldNotice``) and the work goes on.
        """
        failed: host_power.AwakeNotHeld | None = None
        released: host_power.AwakeHold | None = None
        with self._lock:
            wanted = self._in_use()
            if wanted and self._hold is None:
                answer = host_power.hold_awake(_WHY, on_lost=self._lost)
                if isinstance(answer, host_power.AwakeNotHeld):
                    failed = answer
                else:
                    self._hold = answer
            elif not wanted and self._hold is not None:
                released, self._hold = self._hold, None
                released.release()
        if failed is not None:
            self._tell(failed)
        if released is not None:
            logger.info('[KeepAwake] the scope is free: the computer may sleep again')

    def _lost(self, lost: host_power.AwakeNotHeld) -> None:
        """A hold the operating system granted ended by itself."""
        self._tell(lost)

    def _tell(self, failed: host_power.AwakeNotHeld) -> None:
        with self._lock:
            if failed.reason in self._told:
                logger.info(f'[KeepAwake] the computer is not held awake: {failed.detail}')
                return
            self._told.add(failed.reason)
        from _lumascope.modules.notification_center import notifications

        notifications.report_outcome(
            KeepAwakeNotHeldNotice(failed.reason, failed.detail),
            solicited=False,
            category='Keep Awake',
        )
