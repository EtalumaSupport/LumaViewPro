# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A read of the buffered frame the caller already has does no work on it.

The live view polls the camera's buffer faster than a 12-bit camera
delivers, and the buffer hands back the same frame until the next one
arrives. It used to convert that frame to 8 bits (5.2 ms on a full frame)
and draw its scale bar before it compared the frame's ordinal with the one
it had shown, then threw the result away. A caller now names the frame it
has (``unless_seq``) and gets back its ordinal and no image, with nothing
drawn or converted.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _lumascope.modules import image_utils
from _lumascope.modules.image_mode import ImageMode
from _lumascope.modules.scope_display_thread import STATUS_DUPLICATE, STATUS_OK
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.ui_queue import UiDispatcher
from ui.scope_display import ScopeDisplay

# Far past any test here; the teardown's clear() lifts it.
_STALL_S = 600.0
_STOPPED_WAIT_S = 10.0


class _StreamStopped(logging.Handler):
    """Set when the simulated camera says its stream has stopped.

    The sensor loop says so at the top of its first stalled pass, after any
    frame it had in hand is stored, so from then on the buffer holds one
    frame until the stall ends.
    """

    def __init__(self):
        super().__init__()
        self.stopped = threading.Event()

    def emit(self, record):
        if 'simulated stall: frames stopped' in record.getMessage():
            self.stopped.set()


def _stall(scope):
    """Stall the stream with a frame in the buffer, the capture's own."""
    scope.imaging.get_image(force_to_8bit=False)
    watcher = _StreamStopped()
    logger = logging.getLogger('lvp_logger')
    logger.addHandler(watcher)
    try:
        scope.simulation.stall_stream(0.0, _STALL_S)
        assert watcher.stopped.wait(_STOPPED_WAIT_S), 'the simulated stream never stopped'
    finally:
        logger.removeHandler(watcher)


@pytest.fixture
def stalled_12bit(sim_scope):
    sim_scope.camera.set_image_mode(ImageMode.SCIENTIFIC)
    _stall(sim_scope)
    yield sim_scope
    sim_scope.simulation.clear()


def test_a_read_of_the_frame_the_caller_has_returns_its_ordinal_and_no_image(stalled_12bit):
    imaging = stalled_12bit.imaging
    image, seq = imaging.get_image_from_buffer(force_to_8bit=True)
    assert image is not None and seq is not None

    assert imaging.get_image_from_buffer(force_to_8bit=True, unless_seq=seq) == (None, seq)

    # Any other ordinal is not the frame the caller has: the frame comes back.
    again, again_seq = imaging.get_image_from_buffer(force_to_8bit=True, unless_seq=seq - 1)
    assert again_seq == seq
    assert again is not None and again.shape == image.shape


@pytest.fixture
def held_ui():
    queued = []
    ScopeSession.set_ui_dispatcher(
        UiDispatcher(schedule=lambda func, timeout: queued.append(func), thread=None)
    )
    yield queued
    ScopeSession.set_ui_dispatcher(None)


@pytest.fixture
def display(gui_ctx, held_ui):
    # a stand-in by design: the layer panel is a Kivy widget tree, which
    # never runs here; with no layer open the stats have nowhere to show.
    gui_ctx.image_settings = SimpleNamespace(channel_control_lookup=lambda channel: None)
    # The stub base has no canvas, bindings or layout; the instance gets
    # them first, a 100 px box for the overlays to place themselves in.
    widget = ScopeDisplay.__new__(ScopeDisplay)
    widget.canvas = MagicMock()
    widget.bind = MagicMock()
    widget.norm_image_size = (100, 100)
    widget.center_x = widget.center_y = 50
    ScopeDisplay.__init__(widget)
    return widget


def test_the_live_view_converts_nothing_for_a_frame_it_has_shown(gui_ctx, display, monkeypatch):
    scope = gui_ctx.scope
    scope.camera.set_image_mode(ImageMode.SCIENTIFIC)
    _stall(scope)
    try:
        assert display._render_one_frame(open_channel='BF') == STATUS_OK

        # The work skipped is the contract here, and nothing a caller sees
        # changes with it; the conversion's own entry point counts it, where
        # a wall-time bound would pass or fail with the host's load.
        converted = []
        convert = image_utils.convert_to_8bit

        def counting(*args, **kwargs):
            converted.append(1)
            return convert(*args, **kwargs)

        monkeypatch.setattr(image_utils, 'convert_to_8bit', counting)
        assert display._render_one_frame(open_channel='BF') == STATUS_DUPLICATE
        assert converted == [], 'the live view converted a frame it had already shown'
    finally:
        scope.simulation.clear()
