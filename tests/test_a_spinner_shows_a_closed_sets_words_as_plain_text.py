# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A spinner showing a closed set's words is handed them as plain text.

The projection methods and the trendlines are StrEnums, and the panels handed
the members themselves to their spinners. Kivy builds the strings it renders
from the type of the text it is given (``type(text)('\\n')`` when it shortens a
label), so a member's set was asked for ``'\\n'`` and the app crashed as the
post-processing panel drew: the GUI start in the walk of W5b part C's second
push. A widget is handed ``str`` of a member; a pick reads that text back, and
the member's door converts it as it does any caller's.

Each panel's method is run unbound on a stand-in: Kivy is not importable in
the test environment.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.graph_analysis import Trendline
from modules.zproject_method import ZProjectMethod

import ui.post_processing as post_processing


def test_the_projection_spinner_is_filled_with_plain_words(monkeypatch):
    ctx = MagicMock()
    ctx.session.post_processing.projection_methods.return_value = tuple(ZProjectMethod)
    monkeypatch.setattr(post_processing._app_ctx, 'ctx', ctx)
    spinner = SimpleNamespace(values=(), text='')

    post_processing.ZProjectionControls._init_ui(
        SimpleNamespace(ids={'zprojection_method_spinner': spinner})
    )

    assert spinner.values == [str(method) for method in ZProjectMethod]
    assert {type(value) for value in spinner.values} == {str}
    assert type(spinner.text) is str


def test_the_trendline_spinner_is_filled_with_plain_words():
    spinners = {
        name: SimpleNamespace(values=(), text='')
        for name in ('graphing_x_axis_spinner', 'graphing_y_axis_spinner', 'trendline_spinner')
    }
    panel = SimpleNamespace(
        ids=SimpleNamespace(**spinners),
        _x_axes=['Time'],
        _y_axes=['Count'],
        selected_x_axis='Time',
        selected_y_axis='Count',
        _trendline_kinds=(Trendline.LINEAR, Trendline.QUADRATIC),
        _trendline_kind=Trendline.LINEAR,
        _figure=object(),
        graph_widget=SimpleNamespace(show=lambda figure: None),
    )

    post_processing.GraphingControls._redraw_graph(panel)

    trendline = spinners['trendline_spinner']
    assert {type(value) for value in trendline.values} == {str}
    assert type(trendline.text) is str
