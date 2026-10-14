# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live view is shown in a colour, not a layer key.

``ShaderViewer.update_shader`` takes a colour: each plane's white point is
the view's white times the colour's weight on that plane, so a colour that
is not a primary shows as itself, and the record of the view's colour is
the colour's ``'#rrggbb'`` text. A layer's drawer passes its layer's shown
colour (``layer_record.shown_color``): its display colour with false colour
on, grey with it off; the bullseye shows grey.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import modules.layer_record as layer_record
from modules.scope_session import ShownChannel
from ui.layer_control import LayerControl
from ui.microscope_settings import MicroscopeSettings
from ui.shader import ShaderViewer


def _view():
    # a stand-in by design: Kivy is stubbed in the test process; the subject
    # is what update_shader writes into the canvas
    return SimpleNamespace(canvas={}, size=(10, 10), black=0.0, white=1.0)


def test_each_plane_is_the_views_white_times_the_colours_weight(caplog):
    shown = []
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        for color in (
            layer_record.parse_display_color('#0000ff'),
            layer_record.GREY,
            layer_record.parse_display_color('#ff00ff'),
            layer_record.parse_display_color('#ff8000'),
        ):
            view = _view()
            ShaderViewer.update_shader(view, color)
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
def viewer(monkeypatch):
    viewer = MagicMock()
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            viewer=viewer, session=SimpleNamespace(channel_colors=layer_record.release_colors)
        ),
    )
    return viewer


@pytest.mark.parametrize(
    ('layer', 'false_color', 'shown'),
    [
        ('Blue', True, (0.0, 0.0, 1.0)),
        ('Blue', False, layer_record.GREY),
        ('BF', True, layer_record.GREY),
    ],
)
def test_a_drawer_shows_its_layers_colour_with_false_colour_on_and_grey_off(
    viewer, layer, false_color, shown
):
    # parametrize-justified: false colour on, off, and a layer whose colour is grey
    drawer = SimpleNamespace(layer=layer, ids={'false_color': SimpleNamespace(active=false_color)})

    LayerControl.update_shader(drawer, dt=0)

    viewer.update_shader.assert_called_once_with(shown)


def test_the_bullseye_shows_grey_and_leaving_it_redraws_the_shown_layer(monkeypatch):
    shown_layer = MagicMock()
    viewer = MagicMock()
    ctx = SimpleNamespace(
        viewer=viewer,
        scope_display=SimpleNamespace(use_bullseye=False),
        session=SimpleNamespace(
            shown_channel=ShownChannel('Red', lit=False),
            channel_colors=layer_record.release_colors,
        ),
        image_settings=SimpleNamespace(layer_lookup=lambda layer: {'Red': shown_layer}[layer]),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    button = SimpleNamespace(state='down')
    panel = SimpleNamespace(ids={'enable_bullseye_btn_id': button})

    MicroscopeSettings.update_bullseye_state(panel)
    assert viewer.update_shader.call_args_list == [((layer_record.GREY,),)]
    assert ctx.scope_display.use_bullseye is True

    button.state = 'normal'
    MicroscopeSettings.update_bullseye_state(panel)
    shown_layer.update_shader.assert_called_once_with(dt=0)
    assert ctx.scope_display.use_bullseye is False
