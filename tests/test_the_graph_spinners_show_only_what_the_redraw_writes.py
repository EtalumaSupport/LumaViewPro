# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The graph's spinners show only what the redraw writes.

The axis and trendline spinners carried ``text_autoupdate: True``. Kivy then
sets a spinner's text to its first value whenever new values arrive and the
text is not among them. A load always triggers that, since 'X-Axis' is never
a column. So every load picked the first X and Y columns, logged them as the
person's SELECTs, and could plot a column against itself (Eric's batch 3
Part B sim walk, 2026-10-02: 'well' against 'well'). A load now leaves the axes
unchosen until the person picks (Eric, the same walk), and
``GraphingControls._redraw_graph`` writes each spinner's text from what is
stored.
"""

from __future__ import annotations

import pathlib

_KV = pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'lumaviewpro.kv'


def _rule(name: str) -> str:
    # pin-justified: a .kv rule has no seam; reading the rule is the evidence.
    text = _KV.read_text()
    start = text.index(f'\n<{name}>:')
    end = text.index('\n<', start + 1)
    return text[start:end]


def test_no_graph_spinner_rewrites_its_own_text():
    rule = _rule('GraphingControls')
    for spinner in ('graphing_x_axis_spinner', 'graphing_y_axis_spinner', 'trendline_spinner'):
        assert f'id: {spinner}' in rule
    assert 'text_autoupdate' not in rule
