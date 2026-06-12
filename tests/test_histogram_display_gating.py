# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live histogram must compute only when it is actually on screen.

The histogram is a live-image tool with no value off-screen, yet its
0.5 s Clock keeps ticking in states where the widget is not displayed.
Beyond the paused / protocol-running guards (#534), the camera-controls
toggle can hide the histogram while the accordion is still expanded and
preview playing -- there the old code still did the camera read +
128-bin mesh build + GPU upload for nothing.

Histogram._is_displayed centralizes the real display conditions so the
expensive work is skipped whenever the widget is off-screen.

Test approach
-------------
The conftest stubs kivy with MagicMock, so ui.histogram (which imports
kivy.graphics) can't be imported. Instead we extract _is_displayed from
source and exec it in isolation -- it has no kivy dependency, only
self.layer + ctx.image_settings -- and exercise the real logic via a
duck-typed self/ctx. Plus an AST/source lock that the guard runs before
the camera read.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap


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

    def accordion_item_lookup(self, layer):
        return self._item

    def layer_lookup(self, layer):
        return self._layer_obj


class _Ctx:
    def __init__(self, image_settings):
        self.image_settings = image_settings


class _Self:
    def __init__(self, layer='Green'):
        self.layer = layer


def _displayed(layer='Green', **kw):
    return _is_displayed(_Self(layer), _Ctx(_ImageSettings(**kw)))


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
    assert _is_displayed(_Self(layer=None), _Ctx(_ImageSettings())) is False


def test_not_displayed_when_image_settings_missing():
    assert _is_displayed(_Self('Green'), _Ctx(None)) is False


def test_histogram_calls_is_displayed_before_camera_read():
    # Structural lock: the display guard must run before the camera read,
    # so a refactor can't reintroduce off-screen computation.
    guard = SRC.index('_is_displayed(ctx)')
    read = SRC.index('get_image_from_buffer(force_to_8bit')
    assert guard < read, '_is_displayed guard must precede the camera read'
