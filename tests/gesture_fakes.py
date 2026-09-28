# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ui_helpers.submit_gesture`` run inline, for tests that fake the ui_helpers module.

The real helper runs its task on the IO lane; its own contract is pinned by
``tests/test_a_gesture_is_one_lane_task.py``. A test of a gesture's caller
needs only the outcome: the axes asked once, the moves, and the caller's GUI
work only when the moves went through.
"""

from __future__ import annotations

import typing


def inline_submit_gesture(scope) -> typing.Callable[..., None]:
    """A ``submit_gesture`` stand-in that asks *scope*'s motion API and runs at once."""

    def submit_gesture(label, *, axes, then, moves, on_moved=None) -> None:
        axes = tuple(axes)
        try:
            if axes:
                scope.motion.refuse_unknown_positions(axes, recording=False, then=then)
            moves()
        except Exception:
            # The real helper hands this to the reporter; the caller's GUI
            # work does not run.
            return
        if on_moved is not None:
            on_moved()

    return submit_gesture
