# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The view is shown in the colour of what is drawn, not a layer key.

``ShaderViewer`` shows the view in a colour: each plane's white point is
the view's white times the colour's weight on that plane, so a colour that
is not a primary shows as itself, and the record of the view's colour is
the colour's ``'#rrggbb'`` text. The colour is decided at each draw from
what was drawn (``ShaderViewer.show_drawn``): an RGB image (the bullseye, a
derived image, a composite) untinted, so its own colours show; a held mono
frame in its own channel's shown colour; a live frame in the live colour,
which a layer's drawer sets (``ChannelControl.update_shader``) to its
layer's shown colour, its display colour with false colour on, grey with it
off. A run shows each step in its channel's colour as the step lights it,
and before this a held run frame was drawn in the next step's.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import _lumascope.modules.app_context as _app_ctx
import _lumascope.modules.channel_record as channel_record
from ui.channel_control import ChannelControl
from ui.microscope_settings import MicroscopeSettings
from ui.shader import ShaderViewer
from tests.stock_channels import LED0, LED2

_BLUE = channel_record.parse_display_color('#0000ff')
_RED = channel_record.parse_display_color('#ff0000')


def _view():
    # The stub base has no canvas or Kivy properties; the instance gets them,
    # and the subject is what the viewer writes into its canvas.
    view = ShaderViewer.__new__(ShaderViewer)
    view.canvas = {}
    view.size = (10, 10)
    view.black = 0.0
    view.white = 1.0
    return view


def _white(view):
    return view.canvas['white_point'][:3]


def test_each_plane_is_the_views_white_times_the_colours_weight(caplog):
    shown = []
    with caplog.at_level(logging.DEBUG, logger='LVP.gui_interactions'):
        for color in (
            _BLUE,
            channel_record.GRAY,
            channel_record.parse_display_color('#ff00ff'),
            channel_record.parse_display_color('#ff8000'),
        ):
            view = _view()
            view.set_live_color(color)
            shown.append(view.canvas['white_point'])

    assert shown == [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 128 / 255, 0.0, 1.0),
    ]
    recorded = [r.getMessage() for r in caplog.records if 'VIEW_FALSE_COLOR' in r.getMessage()]
    assert [m.split()[-1] for m in recorded] == ['#0000ff', '#ffffff', '#ff00ff', '#ff8000']


@pytest.fixture
def channels(monkeypatch):
    """The layer panel's channels, each answering its shown colour: LED2 red."""
    # a stand-in by design: the layer panel is a Kivy widget tree, which
    # never runs here; the subject is the viewer, which asks it.
    controls = {LED2: SimpleNamespace(shown_color=lambda: _RED)}
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            image_settings=SimpleNamespace(channel_control_lookup=lambda channel: controls[channel])
        ),
    )


@pytest.mark.parametrize('colorfmt', ['rgb', 'bgr'])
def test_an_rgb_image_is_drawn_untinted_whatever_the_live_colour(channels, colorfmt):
    # parametrize-justified: the bullseye's texture and a derived image's
    view = _view()
    view.set_live_color(_BLUE)
    view.show_drawn(colorfmt)
    assert _white(view) == (1.0, 1.0, 1.0)


def test_a_held_frame_keeps_its_channels_colour_until_a_live_frame_is_drawn(channels):
    view = _view()
    view.set_live_color(_BLUE)
    view.show_drawn('luminance')
    assert _white(view) == (0.0, 0.0, 1.0)

    view.show_drawn('luminance', LED2)
    assert _white(view) == (1.0, 0.0, 0.0)
    # The run's next step shows its channel while the frame is held.
    view.set_live_color(channel_record.GRAY)
    assert _white(view) == (1.0, 0.0, 0.0)

    view.show_drawn('luminance')
    assert _white(view) == (1.0, 1.0, 1.0)


def test_a_new_live_colour_shows_at_once_over_a_live_frame(channels):
    view = _view()
    view.show_drawn('luminance')
    view.set_live_color(_BLUE)
    assert _white(view) == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ('layer', 'false_color', 'shown'),
    [
        (LED0, True, (0.0, 0.0, 1.0)),
        (LED0, False, channel_record.GRAY),
        ('BF', True, channel_record.GRAY),
    ],
)
def test_a_layers_shown_colour_is_its_colour_with_false_colour_on_and_grey_off(
    monkeypatch, layer, false_color, shown
):
    # parametrize-justified: false colour on, off, and a layer whose colour is grey
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(session=SimpleNamespace(channel_colors=channel_record.release_colors())),
    )
    drawer = SimpleNamespace(
        channel=layer, ids={'false_color': SimpleNamespace(active=false_color)}
    )

    assert ChannelControl.shown_color(drawer) == shown


def test_a_drawer_sets_the_live_colour_to_its_layers(monkeypatch):
    viewer = MagicMock()
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(viewer=viewer))
    drawer = SimpleNamespace(shown_color=lambda: _BLUE)

    ChannelControl.update_shader(drawer, dt=0)

    viewer.set_live_color.assert_called_once_with(_BLUE)


def test_the_bullseye_toggle_sets_the_display_and_leaves_the_colour_to_the_draw(monkeypatch):
    viewer = MagicMock()
    ctx = SimpleNamespace(viewer=viewer, scope_display=SimpleNamespace(use_bullseye=False))
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    button = SimpleNamespace(state='down')
    panel = SimpleNamespace(ids={'enable_bullseye_btn_id': button})

    MicroscopeSettings.update_bullseye_state(panel)
    assert ctx.scope_display.use_bullseye is True

    button.state = 'normal'
    MicroscopeSettings.update_bullseye_state(panel)
    assert ctx.scope_display.use_bullseye is False
    assert viewer.mock_calls == []
