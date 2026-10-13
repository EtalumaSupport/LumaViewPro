# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The thread that saves a session's live settings, and the whole-file write it saves with.

What a save writes, and whether it may, is the Session's
(``ScopeSession._save_settings_file``); this module holds only when it
runs and how the file is replaced.
"""

from __future__ import annotations

import contextlib
import json
import os
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


def write_whole(file: str, settings: dict) -> None:
    """Write ``settings`` to ``file`` as JSON: the whole new file, or the old one untouched.

    Written beside the file, flushed to the drive, then moved over it, so a
    crash or a power cut mid-save leaves the previous file whole: it is the
    user's only copy of their configuration. No ``encoding=``, as every
    reader of the file has none (``settings_init.read_settings_json``).

    Raises:
        OSError: the write or the replace failed; the file is unchanged and
            nothing is left beside it.
    """
    temporary = f'{file}.saving'
    try:
        with open(temporary, 'w') as f:
            json.dump(settings, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, file)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise
