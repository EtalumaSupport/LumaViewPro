# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The cell-count range sliders commit, and record, on their own release only.

Kivy delivers ``on_touch_up`` and ``on_touch_move`` to every widget for a
touch anywhere in the window, so the six range sliders, once bound to
``on_touch_up``, recorded six slider moves and rewrote six bounds on every
button click elsewhere. ``on_release`` is dispatched by ``RangeSlider`` only
for the touch it grabbed (pinned with real Kivy in
``tests/test_the_cell_count_panel_shows_without_writing.py``).

A guard because the fact is the kv binding: the suite stands Kivy in, so no
widget is built from the kv and no public call reaches the binding; the kv
text is the one place the event kind is written.
"""

from __future__ import annotations

import re

from tests.ast_seams import REPO_ROOT

_RANGE_SLIDERS = (
    'area',
    'perimeter',
    'sphericity',
    'min_intensity',
    'mean_intensity',
    'max_intensity',
)


def test_the_range_sliders_commit_on_their_own_release_only():
    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    for name in _RANGE_SLIDERS:
        assert f'on_release: root.slider_adjustment_{name}()' in kv, name
        assert f'on_touch_up: root.slider_adjustment_{name}()' not in kv, name
    assert not re.search(r'on_touch_move: root\.slider_adjustment_', kv)
