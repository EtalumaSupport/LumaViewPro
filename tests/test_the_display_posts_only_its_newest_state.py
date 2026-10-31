# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the live display posts for a held-up UI thread stays bounded, and a saved frame keeps its hold.

A field LS850T's UI thread stopped for four days with live view on. The
display thread kept posting the engineering validity dot, one Kivy callback
per frame, and the no-camera indicator once per loop while its gate waited
on the UI thread: 0.7M to 74M objects, 8.7 GB, and 10,451 indicator lines
replayed in under a second when it ran again. Each kind of update the
display posts now has one pending callback, running the newest state.

A saved protocol frame is posted from the run's thread. A live frame already
rendering when the hold began was posted after it and drawn over it; a live
blit now draws nothing while the hold is in force.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.kivy_utils import UiDispatcher, ui_backlog
from modules.lumascope_api._lumascope import Lumascope
from modules.scope_session import ScopeSession
from ui.scope_display import ScopeDisplay

_FRAMES = 30


class _Held:
    """A UI thread that is held up: callbacks queue until the test runs them."""

    def __init__(self):
        self.queue = []

    def schedule(self, func, timeout):
        self.queue.append(func)

    def run(self):
        queue, self.queue = self.queue, []
        for func in queue:
            func(0)


@pytest.fixture
def held():
    ui = _Held()
    ScopeSession.set_ui_dispatcher(UiDispatcher(schedule=ui.schedule, thread=None))
    yield ui
    ScopeSession.set_ui_dispatcher(None)


@pytest.fixture
def display(gui_ctx):
    gui_ctx.session.engineering_mode = True
    # a stand-in by design: the layer panel is a Kivy widget tree, which
    # never runs here; with no layer open the stats have nowhere to show.
    gui_ctx.image_settings = SimpleNamespace(layer_lookup=lambda layer: None)
    # The stub base has no canvas, bindings or layout; the instance gets
    # them first, a 100 px box for the overlays to place themselves in.
    widget = ScopeDisplay.__new__(ScopeDisplay)
    widget.canvas = MagicMock()
    widget.bind = MagicMock()
    widget.norm_image_size = (100, 100)
    widget.center_x = widget.center_y = 50
    ScopeDisplay.__init__(widget)
    return widget


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def display_log():
    handler = _Lines()
    logger = logging.getLogger('LVP.ui.scope_display')
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.lines
    logger.removeHandler(handler)
    logger.setLevel(level)


def _shown_frames(gui_ctx, display, count):
    """Run the real display loop until it has rendered ``count`` frames, then stop it."""
    thread = gui_ctx.scope_display_thread
    rendered = []
    enough = threading.Event()

    def on_frame(*_frame):
        rendered.append(1)
        if len(rendered) >= count:
            enough.set()

    thread.update_layer_config(open_layer='BF')
    thread.add_frame_listener(on_frame)
    thread.start(display, fps=0)
    try:
        assert enough.wait(30), f'the display rendered {len(rendered)} of {count} frames'
    finally:
        thread.stop()
        thread.remove_frame_listener(on_frame)


def test_a_held_ui_thread_holds_one_post_per_kind_and_shows_the_last_state(
    gui_ctx, display, held, display_log
):
    _shown_frames(gui_ctx, display, _FRAMES)
    # The frame, the live view's indicator, the validity dot and the
    # engineering stats: one entry each in the dispatcher's queue, however
    # many frames were rendered, and one drain asked of the UI thread.
    assert (ui_backlog(), len(held.queue)) == (4, 1), f'after {_FRAMES} frames'

    gui_ctx.scope.simulation.remove_camera()
    for _ in range(100):
        display._render_one_frame(open_layer='BF')
    assert (ui_backlog(), len(held.queue)) == (4, 1), 'after the camera left'

    held.run()
    assert display.source == './data/icons/camera_to_USB.png'
    shown = [line for line in display_log if 'no-camera indicator' in line]
    assert len(shown) == 1, display_log
    assert not [line for line in display_log if 'restored' in line], display_log


def test_a_display_started_with_no_camera_shows_the_indicator(gui_ctx, display, held):
    gui_ctx.scope.simulation.remove_camera()
    display._render_one_frame(open_layer=None)
    held.run()
    assert display.source == './data/icons/camera_to_USB.png'


def test_a_camera_that_returns_restores_the_live_view_once(
    gui_ctx, display, held, display_log, monkeypatch
):
    # a stand-in by design: the simulator's removed camera does not come
    # back (``SimulationAPI.remove_camera``), so its return is a simulator
    # gap; the presence the display reads is set here.
    present = [True]
    monkeypatch.setattr(Lumascope, 'camera_connected', property(lambda self: present[0]))

    display._render_one_frame(open_layer=None)
    held.run()
    assert display.source == ''
    assert not [line for line in display_log if 'restored' in line], display_log

    present[0] = False
    display._render_one_frame(open_layer=None)
    held.run()
    present[0] = True
    for _ in range(20):
        display._render_one_frame(open_layer=None)
    held.run()
    assert display.source == ''
    restored = [line for line in display_log if 'live view restored' in line]
    assert len(restored) == 1, display_log


def _frame(value):
    return np.full((4, 4), value, dtype=np.uint8)


def _drawn(display):
    """The bytes last put on the display's mono texture."""
    return display._mono_texture.blit_buffer.call_args.args[0]


def test_a_saved_frame_is_not_drawn_over_by_a_live_one_during_its_hold(gui_ctx, display, held):
    display._PROTOCOL_HOLD_MS = 60_000
    for value in range(50):
        display.hold_protocol_saved_image(_frame(value), 8)
    assert ui_backlog() == 1, f'{ui_backlog()} saved frames pending'

    # A live frame already rendering when the hold began is posted after
    # the saved one and runs after it.
    gen = display._current_generation()
    held.schedule(
        lambda dt: display.create_and_set_texture(
            _frame(200).tobytes(), (4, 4), generation=gen, live=True
        ),
        0,
    )
    held.run()
    assert display.frames_shown == 1
    assert _drawn(display) == _frame(49).tobytes()

    # The hold ends; the live view draws again.
    gui_ctx.scope_display_thread.bump_protocol_hold(0)
    display.create_and_set_texture(_frame(201).tobytes(), (4, 4), generation=gen, live=True)
    assert _drawn(display) == _frame(201).tobytes()


def test_a_saved_frame_and_derived_images_wait_as_one_held_frame(gui_ctx, display, held):
    # Post-processing's derived images and a run's saved frames are one state:
    # each waiting post holds a whole image, so only the newest waits.
    import ui.post_processing as post_processing

    gui_ctx.scope_display = display
    shown = []
    # a stand-in by design: the draw is Kivy's texture upload; the subject is the post.
    display.hold_derived_image = lambda image, bits: shown.append(int(image[0, 0]))
    display.hold_protocol_saved_image(_frame(1), 8)
    for value in range(50):
        post_processing.QuickEnhanceControls._queue_derived_image(None, _frame(value), 8)
    assert ui_backlog() == 1, f'{ui_backlog()} held frames pending'
    held.run()
    assert shown == [49]


def test_a_saved_frame_is_held_its_whole_time_from_its_draw(gui_ctx, display, held):
    # The hold began on the run's thread when the frame was posted, so the
    # time the post waited for a slow UI thread came out of it: here all of
    # it. Counted from the draw, the frame is held its whole time.
    display._PROTOCOL_HOLD_MS = 0
    display.hold_protocol_saved_image(_frame(7), 8)
    assert not gui_ctx.scope_display_thread.protocol_hold_in_force
    display._PROTOCOL_HOLD_MS = 60_000
    held.run()
    assert _drawn(display) == _frame(7).tobytes()
    assert gui_ctx.scope_display_thread.protocol_hold_in_force


def test_a_derived_image_drawn_is_held_its_time(gui_ctx, display):
    display._PROTOCOL_HOLD_MS = 60_000
    assert not gui_ctx.scope_display_thread.protocol_hold_in_force
    display.hold_derived_image(np.zeros((4, 4, 3), dtype=np.uint8), 8)
    assert gui_ctx.scope_display_thread.protocol_hold_in_force
