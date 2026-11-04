# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ui_helpers.submit_reported`` run inline, for tests that fake ui_helpers.

The real helper runs its task on a lane; its own contract is pinned by its
own tests. A test of a handler's decision needs only the outcome: the call
and its redraw, as ``run_reported`` runs them.
"""

from __future__ import annotations


def inline_submit_reported(call, redraw, label, *, stop=False, lane=None, budget_of=None) -> None:
    """A ``submit_reported`` stand-in: *call* and *redraw* here, as ``run_reported`` runs them.

    For a test of a handler's decision, where the lane the call runs on is
    not the subject; the real helper's contract is pinned by its own tests.
    """
    from ui.ui_helpers import run_reported

    run_reported(call, redraw, label)
