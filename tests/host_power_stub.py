# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The host's power request, as the suite stands it in: a recording host.

The operating system's sleep request is a layer no test owns -- a test
that took a real one would hold the developer's or CI's computer awake, and
CI's Linux runner and a Mac would answer differently -- so
``tests/conftest.py install_mock_deps`` installs it (``install``): the
real module's ``hold_awake`` becomes this host's, which records which holds
are live, and everything else production reads stays the real module's.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from _lumascope.drivers import host_power as _real


class RecordingHost:
    """The holds production has taken and not released, and the next refusal.

    Attributes:
        refuse_with: When set, ``hold_awake`` answers it instead of holding.
        on_lost: What the last hold was given to call if the host ended it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: list[_real.AwakeHold] = []
        self.refuse_with: _real.AwakeNotHeld | None = None
        self.on_lost: Callable[[_real.AwakeNotHeld], None] | None = None

    @property
    def held(self) -> int:
        """How many holds are live now."""
        with self._lock:
            return len(self._live)

    def reset(self) -> None:
        """Forget every hold and refusal: each test starts on a free host."""
        with self._lock:
            self._live.clear()
            self.refuse_with = None
            self.on_lost = None

    def hold_awake(
        self, why: str, on_lost: Callable[[_real.AwakeNotHeld], None]
    ) -> _real.AwakeHold | _real.AwakeNotHeld:
        if self.refuse_with is not None:
            return self.refuse_with
        hold: _real.AwakeHold

        def release() -> None:
            with self._lock:
                self._live.remove(hold)

        hold = _real.AwakeHold(release)
        with self._lock:
            self._live.append(hold)
            self.on_lost = on_lost
        return hold


host = RecordingHost()


def install() -> None:
    """Make the real module's ``hold_awake`` this host's, for every caller.

    The one attribute production reads, patched on the real module, so no
    importer ever holds a second copy of the driver. Idempotent.
    """
    _real.hold_awake = host.hold_awake
