# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The thread that saves a session's live settings.

What a save writes, and whether it may, is the Session's
(``ScopeSession._save_settings_file``); this module holds only when it
runs. The file is replaced by ``path_utils.write_whole``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class SettingsSaver:
    """Runs ``save`` one window after the first write that wakes it.

    Every write inside the window shares that one save, so a slider dragged
    for a second is one write to the drive rather than one per step. Its
    own thread, not the session's FILE lane: a run's images and a stalled
    writer hold that lane, and a settings write must not wait behind them.

    ``stop`` makes no save of its own: the Session makes the closing save
    itself, after its work has finished, so the file holds what the
    session ended with.
    """

    WINDOW_S = 1.0

    def __init__(self, save: Callable[[], None], *, window_s: float = WINDOW_S):
        self._save = save
        self._window_s = window_s
        self._woken = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name='LVP-SettingsSaver', daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        """A write is waiting to be saved. Never blocks, so a writer may call it under its lock."""
        self._woken.set()

    def stop(self) -> None:
        """End the thread, waiting for a save in flight. Repeatable.

        Callable from the saver's own thread -- a listener told of a failed
        save may close the session -- where it returns without the wait.
        """
        self._stopped.set()
        self._woken.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join()

    def _run(self) -> None:
        while True:
            self._woken.wait()
            if self._stopped.wait(self._window_s):
                return
            # Cleared before the save: a write landing during it wakes the
            # next window instead of being taken for one already saved.
            self._woken.clear()
            self._save()
