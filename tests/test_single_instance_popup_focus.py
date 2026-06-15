# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: the single-instance "already running" popup must be forced
to the foreground.

Bench report 2026-06-03: when a second LumaViewPro launch loses the
instance lock, the tkinter messagebox opened behind the already-running
LVP window and got buried, so the user saw nothing and didn't know why
the second launch did nothing. The fix forces the dialog topmost +
focused before showing it.

Source-inspection (the popup path runs pre-Kivy at process start and pops
a native tkinter dialog -- not driveable in a headless test), matching the
convention of the other UI regression tests in this suite.
"""

from __future__ import annotations

import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = (REPO / 'lumaviewpro.py').read_text()


def test_already_running_popup_is_forced_topmost_and_focused():
    assert "_root.attributes('-topmost', True)" in SRC, (
        'the already-running popup must set the root topmost, or the dialog '
        'opens behind the running LVP window and gets buried.'
    )
    assert '_root.focus_force()' in SRC, (
        'the already-running popup must force focus so it surfaces above the existing window.'
    )


def test_already_running_messagebox_is_parented_to_topmost_root():
    # Parenting the messagebox to the topmost root is what makes the dialog
    # itself inherit the foreground placement (not just the hidden root).
    assert 'parent=_root' in SRC, (
        'showerror must be parented to the topmost root so the dialog inherits '
        'the foreground placement.'
    )
