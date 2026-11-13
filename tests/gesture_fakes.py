# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ui_helpers.submit_reported`` stand-ins, for tests that fake ui_helpers.

The real helper runs its task on a lane; its own contract is pinned by its
own tests. A test of a handler's decision needs only the outcome: the call
and its redraw, inline as ``run_reported`` runs them, or the call on a
thread of its own when it waits on a lane.
"""

from __future__ import annotations


def inline_submit_reported(call, redraw, label, *, stop=False, lane=None, budget_of=None) -> None:
    """A ``submit_reported`` stand-in: *call* and *redraw* here, as ``run_reported`` runs them.

    For a test of a handler's decision, where the lane the call runs on is
    not the subject; the real helper's contract is pinned by its own tests.
    """
    from ui.ui_helpers import run_reported

    run_reported(call, redraw, label)


# Only a hung call waits this long: the worker's own answer comes first.
_WORKER_WAIT_S = 120.0


def worker_submit_reported(call, redraw, label, *, stop=False, lane=None, budget_of=None) -> None:
    """A ``submit_reported`` stand-in: *call* on a thread of its own, then *redraw* here.

    For a handler whose call waits on a lane, which the inline boundary
    refuses: the call runs off this thread, as the GUI's worker runs it,
    reported as the real helper reports it, and the redraw follows once it
    has. The worker pool itself is not the subject.
    """
    import threading

    from _lumascope.modules.caller import Caller, calling
    from ui.ui_helpers import _reported

    def off_the_gui_thread():
        with calling(Caller.GUI):
            _reported(call, label)

    worker = threading.Thread(target=off_the_gui_thread, name=f'UI:{label}', daemon=True)
    worker.start()
    worker.join(_WORKER_WAIT_S)
    assert not worker.is_alive(), f'{label}: the call did not return'
    _reported(redraw, label)
