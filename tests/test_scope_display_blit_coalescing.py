# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the single-pending-blit coalescing in ScopeDisplay.

The display thread produces a full-frame byte buffer (~3.5 MB) per iteration
and marshals the blit to the main thread via Clock. Scheduling a fresh
callback per frame let those buffers pile up whenever the main thread stalled
(homing / tiling moves) -- the source of the multi-GB live-view RAM balloon.
These tests lock in that the backlog is bounded to a single pending frame:

  * N frames produced while the main thread is stalled -> exactly ONE Clock
    callback queued, and only the LATEST frame's blit runs (intermediate
    display frames dropped, never capture/save).
  * when the main thread keeps up, every frame still blits (no coalescing,
    nothing dropped).

The real ScopeDisplay is a Kivy widget needing a GL context; the coalescing
logic touches only three instance attributes + Clock, so a minimal stand-in
borrowing the real methods exercises the exact code without constructing it.
"""

import sys
import threading
from types import ModuleType
from unittest.mock import MagicMock, patch

# ui.scope_display is a Kivy widget module. The test env mocks `kivy` but not
# the submodules this module imports, and ScopeDisplay subclasses kivy's Image
# (a bare MagicMock can't be subclassed). Provide minimal stubs -- a real base
# class for Image/Widget, MagicMock for the graphics/property/metrics modules --
# so the real module (and its real coalescing methods) import for the test.


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    """Install a module exposing REAL classes (so ScopeDisplay can subclass)."""
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


# Permissive MagicMock submodules: any attribute resolves, so this never
# shadows names other test modules import (e.g. kivy.properties.ListProperty).
# setdefault avoids clobbering anything conftest already provides.
for _name in (
    'kivy.uix',
    'kivy.graphics',
    'kivy.graphics.texture',
    'kivy.metrics',
    'kivy.properties',
    'kivy.input',
    'kivy.clock',
):
    sys.modules.setdefault(_name, MagicMock())

# Image/Widget must be REAL, subclassable base classes for `class ScopeDisplay
# (Image)` -- a bare MagicMock can't be a base.
_real_base_module('kivy.uix.image', Image=_StubWidget)
_real_base_module('kivy.uix.widget', Widget=_StubWidget)

from ui.scope_display import ScopeDisplay


class _Stand:
    """Carries only the blit-coalescing state + the real methods under test."""

    _schedule_blit = ScopeDisplay._schedule_blit
    _run_pending_blit = ScopeDisplay._run_pending_blit

    def __init__(self):
        self._pending_blit = None
        self._blit_scheduled = False
        self._blit_lock = threading.Lock()


def test_backlog_bounded_to_one_pending_under_main_thread_stall():
    sd = _Stand()
    scheduled = []
    calls = []
    with patch(
        'ui.scope_display.Clock.schedule_once',
        side_effect=lambda cb, t: scheduled.append(cb),
    ):
        # Display thread produces 100 frames while the main thread is stalled
        # (the scheduled callback never runs).
        for i in range(100):
            sd._schedule_blit(lambda i=i: calls.append(i))

        # Exactly one callback queued despite 100 frames -> no backlog.
        assert len(scheduled) == 1
        assert sd._pending_blit is not None
        assert sd._blit_scheduled is True

        # Main thread catches up: the single callback runs only the LATEST.
        scheduled[0](0)
        assert calls == [99]
        assert sd._pending_blit is None
        assert sd._blit_scheduled is False

        # A subsequent frame schedules a fresh callback.
        sd._schedule_blit(lambda: calls.append('next'))
        assert len(scheduled) == 2


def test_every_frame_blits_when_main_thread_keeps_up():
    sd = _Stand()
    scheduled = []
    calls = []
    with patch(
        'ui.scope_display.Clock.schedule_once',
        side_effect=lambda cb, t: scheduled.append(cb),
    ):
        for i in range(5):
            sd._schedule_blit(lambda i=i: calls.append(i))
            scheduled[-1](0)  # main thread drains the blit immediately

        # No coalescing when the main thread keeps up: every frame displayed.
        assert calls == [0, 1, 2, 3, 4]
        assert len(scheduled) == 5
