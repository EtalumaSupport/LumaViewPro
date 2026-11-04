# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A sidebar ScrollView holds one scroll-bar colour binding, however often it moves.

Kivy 2.3.1's ScrollView.update_from_scroll binds _change_bar_color to
bar_color on every call, with no check for one already bound, and its 0.5 s
fade trigger unbinds one. A drawer opening repositions the sidebar on every
animation frame, so each opening left a binding per frame behind: about
1,600 per 12-step scan once the drawers followed a run's steps, never
released (the W1b D2 sim walk's object census, 2026-10-13). The sidebars'
ScrollView unbinds before Kivy binds, so one stays.

It runs real Kivy in a child process, because the test environment stands
Kivy's widgets in. Kivy's own ScrollView is the known positive.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

_BINDINGS_AFTER_MOVES = r"""
import os
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from ui.mod_slider import ModSliderAwareScrollView
for cls in (ScrollView, ModSliderAwareScrollView):
    view = cls(size=(100, 100))
    view.add_widget(Label(size_hint_y=None, height=500))
    for _ in range(50):
        view.update_from_scroll()
    print(sum(1 for o in view.get_property_observers('bar_color')
              if getattr(o, '__name__', '') == '_change_bar_color'))
"""


@pytest.mark.needs_display
def test_fifty_moves_leave_one_bar_colour_binding(tmp_path):
    # Kivy reads its config from and writes its logs and icons into KIVY_HOME,
    # so the child gets one of its own rather than the shared ~/.kivy.
    done = subprocess.run(
        [sys.executable, '-c', _BINDINGS_AFTER_MOVES],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, 'KIVY_HOME': str(tmp_path)},
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    kivys, sidebars = done.stdout.split()[-2:]
    assert int(kivys) > 1, "Kivy's own ScrollView no longer piles up bindings"
    assert sidebars == '1'
