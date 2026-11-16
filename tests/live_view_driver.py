# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live view's two threads driven by hand, on one clock the test sets.

``ScopeDisplay`` judges a slow frame at each live frame the UI thread draws,
by the record the display thread adds to at each frame it picks up. A
test that needs the camera, the display thread and the UI thread to be late
by known amounts drives the widget's own two seams, ``_note_pickup`` /
``_note_held`` on the display side and ``_judge_live_draw`` on the UI side,
with the frames the camera's store would have handed the display.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np

from tests.camera_fakes import delivered_frame
from ui.scope_display import ScopeDisplay


def live_view_widget() -> ScopeDisplay:
    """A real ``ScopeDisplay``, built without a window.

    The stub base has no canvas or bindings; the instance gets them first.
    """
    widget = ScopeDisplay.__new__(ScopeDisplay)
    widget.canvas = MagicMock()
    widget.bind = MagicMock()
    ScopeDisplay.__init__(widget)
    return widget


class SlowFrameLines(logging.Handler):
    """Collects the ``[SLOW FRAME]`` lines the display writes while attached."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if '[SLOW FRAME]' in message:
            self.lines.append(message)

    def __enter__(self):
        logger = logging.getLogger('LVP.ui.scope_display')
        self._level = logger.level
        logger.addHandler(self)
        logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        logger = logging.getLogger('LVP.ui.scope_display')
        logger.removeHandler(self)
        logger.setLevel(self._level)


class LiveView:
    """A widget, its clock in seconds (``time.monotonic``), and the camera's frame count.

    Args:
        widget: The ``ScopeDisplay`` driven.
        now: The clock's start, in ``time.monotonic`` seconds.
    """

    _PIXELS = np.zeros((2, 2), dtype=np.uint8)

    def __init__(self, widget: ScopeDisplay, now: float = 1000.0):
        self.widget = widget
        self.now = now
        self._stored = 0

    def pick_up(self, *, after_ms, skipped=0, settled=True, posted=True, compute=(1.0, 3.0, 0.0)):
        """The display thread picks up the camera's next frame ``after_ms`` after the last.

        The camera stored ``skipped`` frames the display never picked up
        between them, and the frame arrived when it was picked up.
        """
        self.now += after_ms / 1000.0
        self._stored += 1 + skipped
        frame = delivered_frame(
            self._PIXELS,
            seq=self._stored,
            stored=self._stored,
            arrived_ns=round(self.now * 1e9),
        )
        grab, proc, eng = compute
        self.widget._note_pickup(
            frame,
            settled=settled,
            posted_at=self.now if posted else None,
            grab_ms=grab,
            proc_ms=proc,
            eng_ms=eng,
        )

    def held(self, ms):
        """The display draws nothing for ``ms`` on purpose (a hold, the pause)."""
        self.now += ms / 1000.0
        self.widget._note_held(ms)

    def draw(self, *, after_ms=0.0):
        """The UI thread draws the waiting live frame ``after_ms`` from now."""
        self.now += after_ms / 1000.0
        self.widget._judge_live_draw(self.now)

    def frames(self, count, period_ms, *, settled=True, compute=(1.0, 3.0, 0.0)):
        """``count`` frames, each picked up ``period_ms`` after the last and drawn at once."""
        for _ in range(count):
            self.pick_up(after_ms=period_ms, settled=settled, compute=compute)
            self.draw()
