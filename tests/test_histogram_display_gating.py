# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live histogram draws the live frame, only when it is on screen, and never a frame no longer live.

The histogram is a live-image tool with no value off-screen, yet its
0.5 s Clock keeps ticking in states where the widget is not displayed;
Histogram._is_displayed holds the display conditions, and the tick skips
its work whenever the widget is off-screen.

While a run holds the scope or the picture is frozen, the tick reads no
frame (#534). A run, a camera gone and a channel left blank the bars, since
the live picture goes on (or says the camera is gone) while the bars would
be a frame from before; a freeze keeps them, the frozen frame's own.

Test approach
-------------
_is_displayed runs in isolation over a duck-typed self/ctx. The tick runs
the real widget over the simulator's session (``gui_ctx``): the conftest's
Kivy stubs make ui.histogram importable, and the widget's canvas is a
stand-in that records whether bars are on it.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest
from tests.stock_channels import LED1, LED2


REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = (REPO / 'ui' / 'histogram.py').read_text()


def _load_is_displayed():
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_is_displayed':
            ns: dict = {}
            exec(textwrap.dedent(ast.get_source_segment(SRC, node)), ns)
            return ns['_is_displayed']
    raise AssertionError('_is_displayed not found in ui/histogram.py')


_is_displayed = _load_is_displayed()


class _Toggle:
    def __init__(self, state):
        self.state = state


class _Item:
    def __init__(self, collapse):
        self.collapse = collapse


class _LayerObj:
    def __init__(self, show):
        self.show_camera_controls = show


class _ImageSettings:
    def __init__(self, drawer_state='down', collapse=False, show=True):
        self.ids = {'toggle_imagesettings': _Toggle(drawer_state)}
        self._item = _Item(collapse)
        self._layer_obj = _LayerObj(show)

    def accordion_item_lookup(self, channel):
        return self._item

    def channel_control_lookup(self, channel):
        return self._layer_obj


class _Ctx:
    def __init__(self, image_settings):
        self.image_settings = image_settings


class _Self:
    def __init__(self, channel=LED1):
        self.channel = channel


def _displayed(channel=LED1, **kw):
    return _is_displayed(_Self(channel), _Ctx(_ImageSettings(**kw)))


def test_displayed_when_drawer_open_accordion_expanded_controls_shown():
    assert _displayed(drawer_state='down', collapse=False, show=True) is True


def test_not_displayed_when_drawer_collapsed():
    assert _displayed(drawer_state='normal') is False


def test_not_displayed_when_accordion_collapsed():
    assert _displayed(collapse=True) is False


def test_not_displayed_when_camera_controls_hidden():
    # The gap this change closes: accordion expanded + drawer open +
    # preview playing, but camera controls (which host the histogram)
    # are hidden -- must not compute.
    assert _displayed(show=False) is False


def test_not_displayed_when_layer_unset():
    assert _is_displayed(_Self(channel=None), _Ctx(_ImageSettings())) is False


def test_not_displayed_when_image_settings_missing():
    assert _is_displayed(_Self(LED1), _Ctx(None)) is False


def test_histogram_calls_is_displayed_before_camera_read():
    # Structural lock: the display guard must run before the camera read,
    # so a refactor can't reintroduce off-screen computation.
    guard = SRC.index('_is_displayed(ctx)')
    read = SRC.index('buffered_image(force_to_8bit')
    assert guard < read, '_is_displayed guard must precede the camera read'


def _a_panel_with_a_recorded_histogram_schedule(monkeypatch):
    """The image-settings panel over a stub context, with what is scheduled recorded.

    Returns the module, each channel's controls, the panel and the set of
    histogram callbacks scheduled now.
    """
    from types import SimpleNamespace

    import _lumascope.modules.app_context as _app_ctx
    import _lumascope.modules.common_utils as common_utils
    import ui.image_settings as image_settings

    # a stand-in by design: Kivy is stubbed in the test process, so the
    # interval routine is a recorder of what is scheduled, each handle
    # cancelling its own; the subject is the panel, not Kivy's timer
    scheduled = set()

    def schedule(callback, _interval, _label):
        scheduled.add(callback)
        return SimpleNamespace(cancel=lambda: scheduled.discard(callback))

    monkeypatch.setattr(image_settings, 'schedule_ui_interval', schedule)
    controls = {
        channel: SimpleNamespace(
            ids={'histo_id': SimpleNamespace(histogram=lambda *a: None, blank=lambda: None)}
        )
        for channel in common_utils.get_channels()
    }
    panel = SimpleNamespace(
        ids={'toggle_imagesettings': SimpleNamespace(state='down')},
        x=0,
        tab_width=20,
        settings_width=300,
        channel_control_lookup=lambda channel: controls[channel],
        _histogram_tick=None,
    )
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=SimpleNamespace(controls_lock=SimpleNamespace(run_named=None)),
            lumaview=SimpleNamespace(width=1000),
            image_settings=panel,
        ),
    )
    return image_settings, controls, panel, scheduled


def test_the_shown_histogram_draws_again_after_the_panel_folds_and_unfolds(monkeypatch):
    """Folding the image-settings panel leaves the shown channel's histogram
    scheduled, so it draws again the moment the panel unfolds; while folded,
    _is_displayed (above) is what skips its work."""
    image_settings, controls, panel, scheduled = _a_panel_with_a_recorded_histogram_schedule(
        monkeypatch
    )
    image_settings.set_histogram_channel(active_channel=LED1)
    shown = controls[LED1].ids['histo_id'].histogram

    panel.ids['toggle_imagesettings'].state = 'normal'
    image_settings.ImageSettings.place_settings(panel)
    panel.ids['toggle_imagesettings'].state = 'down'
    image_settings.ImageSettings.place_settings(panel)

    assert panel.x == 1000 - 300
    assert scheduled == {shown}


def test_showing_another_channel_stops_the_last_ones_histogram(monkeypatch):
    """Only the shown channel's histogram is scheduled: showing another one
    cancels the last one's schedule through its handle."""
    image_settings, controls, _panel, scheduled = _a_panel_with_a_recorded_histogram_schedule(
        monkeypatch
    )
    image_settings.set_histogram_channel(active_channel=LED1)
    image_settings.set_histogram_channel(active_channel=LED2)

    assert scheduled == {controls[LED2].ids['histo_id'].histogram}


class _Canvas:
    """A stand-in by design: Kivy is stubbed in the test process. Records
    whether bars are on the widget: drawing into it puts them there, a
    clear takes them off."""

    def __init__(self):
        self.holds_bars = False

    def clear(self):
        self.holds_bars = False

    def __enter__(self):
        self.holds_bars = True
        return self

    def __exit__(self, *exc):
        return False


def _a_histogram(channel):
    from ui.histogram import Histogram

    histogram = Histogram()
    histogram.channel = channel
    histogram.bg_color = (1, 1, 1, 1)
    histogram.canvas = _Canvas()
    histogram.x, histogram.y, histogram.width, histogram.height = 0, 0, 256, 100
    return histogram


@pytest.fixture
def shown(gui_ctx, monkeypatch):
    """LED1's histogram on screen with bars drawn, over the simulator, its frame reads counted."""
    from types import SimpleNamespace

    import _lumascope.modules.common_utils as common_utils
    import ui.image_settings as image_settings

    # a stand-in by design: the Kivy tree never runs here; the panel answers
    # the histogram's display questions and hands each channel's widgets
    histograms = {channel: _a_histogram(channel) for channel in common_utils.get_channels()}
    controls = {
        channel: SimpleNamespace(
            show_camera_controls=True,
            ids={'logHistogram_id': SimpleNamespace(active=False), 'histo_id': widget},
        )
        for channel, widget in histograms.items()
    }
    gui_ctx.image_settings = SimpleNamespace(
        ids={'toggle_imagesettings': SimpleNamespace(state='down')},
        accordion_item_lookup=lambda channel: SimpleNamespace(collapse=False),
        channel_control_lookup=lambda channel: controls[channel],
        _histogram_tick=None,
    )
    gui_ctx.viewer = SimpleNamespace(black=0.0, white=1.0)
    monkeypatch.setattr(
        image_settings,
        'schedule_ui_interval',
        lambda callback, _interval, _label: SimpleNamespace(cancel=lambda: None),
    )
    gui_ctx.scope.imaging.capture_image()
    reads = []
    real_read = gui_ctx.scope.imaging.buffered_image

    def counted_read(*args, **kwargs):
        reads.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(gui_ctx.scope.imaging, 'buffered_image', counted_read)
    histograms[LED1].histogram()
    assert histograms[LED1].canvas.holds_bars
    reads.clear()
    return SimpleNamespace(
        ctx=gui_ctx, histograms=histograms, reads=reads, image_settings=image_settings
    )


def test_a_run_holding_the_scope_blanks_the_bars_and_reads_no_frame(shown):
    with shown.ctx.session.diagnostic_claim():
        shown.histograms[LED1].histogram()

        assert not shown.histograms[LED1].canvas.holds_bars
        assert shown.reads == []


def test_the_bars_return_on_the_first_tick_after_the_run(shown):
    with shown.ctx.session.diagnostic_claim():
        shown.histograms[LED1].histogram()

    shown.histograms[LED1].histogram()

    assert shown.histograms[LED1].canvas.holds_bars


def test_a_frozen_picture_keeps_its_frames_bars_and_reads_no_frame(shown):
    shown.ctx.scope_display_thread.pause()

    shown.histograms[LED1].histogram()

    assert shown.histograms[LED1].canvas.holds_bars
    assert shown.reads == []


def test_a_camera_gone_blanks_the_bars(shown):
    # a stand-in by design: the simulated camera has no unplug of its own
    # (a simulator gap); the driver's removal mark is what a USB drop sets
    shown.ctx.scope._camera_driver._mark_disconnected()

    shown.histograms[LED1].histogram()

    assert not shown.histograms[LED1].canvas.holds_bars


def test_a_channel_left_keeps_no_bars_for_its_next_showing(shown):
    shown.image_settings.set_histogram_channel(active_channel=LED1)

    shown.image_settings.set_histogram_channel(active_channel=LED2)

    assert not shown.histograms[LED1].canvas.holds_bars
