# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether the host is asked to stay awake: exactly while a session is open.

A sleeping computer drops the scope's USB devices, so a session comes back
to a scope it has lost, whether a run was under way or a person was setting
up between runs; a Modern Standby PC whose screen goes off enters connected
standby and runs the camera path throttled. The session takes the operating
system's request to keep the computer and its screen awake
(``host_power.hold_awake``) when it is built and releases
it when it closes; this owner keeps only that hold, never an "is LVP open"
flag, since the session's own lifetime is the fact.
"""

from __future__ import annotations

from _lumascope.drivers import host_power
from _lumascope.lvp_logger import logger
from _lumascope.modules.exceptions import KeepAwakeNotHeldNotice

_WHY = 'LumaViewPro is using the microscope'


class KeepAwake:
    """Holds the host's request to stay awake from construction to ``release``.

    One take per session, so each way the request can fail is told at most
    once per session with nothing to remember it. The operating system drops
    a hold whose process has exited, so a session never closed holds nothing
    once its process is gone.
    """

    def __init__(self):
        """Take the request; one not taken is told (``KeepAwakeNotHeldNotice``) and LVP goes on."""
        self._hold: host_power.AwakeHold | None = None
        answer = host_power.hold_awake(_WHY, on_lost=self._tell)
        if isinstance(answer, host_power.AwakeNotHeld):
            self._tell(answer)
        else:
            self._hold = answer
            logger.info(f'[KeepAwake] the session holds the computer awake: {answer.holds}')

    def release(self) -> None:
        """Release the request, if one was taken; a second release does nothing.

        Called from the session's teardown, which must finish whatever the
        operating system answers: a release it refuses is logged and the
        request is left for the operating system to drop when the process
        exits.
        """
        hold, self._hold = self._hold, None
        if hold is None:
            return
        try:
            hold.release()
        except OSError as ex:
            logger.warning(f'[KeepAwake] the operating system did not release the request: {ex}')
            return
        logger.info('[KeepAwake] the session is closed: the computer may sleep again')

    @staticmethod
    def _tell(failed: host_power.AwakeNotHeld) -> None:
        """A request not taken, or a hold the operating system ended by itself."""
        from _lumascope.modules.notification_center import notifications

        notifications.report_outcome(
            KeepAwakeNotHeldNotice(failed.reason, failed.detail),
            solicited=False,
            category='Keep Awake',
        )
