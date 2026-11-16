# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The window is minimized from Kivy's minimize until the window is back.

A slow frame's line names the window's state when drawn, and the app answers
whether it is minimized (``LumaViewProApp.window_minimized``, handed to the
display). Kivy sends ``on_minimize`` on every platform, but on the Mac it never
sent ``on_restore``: the window came back, gained focus, and every slow frame
after it still said ``window=minimized``. A minimized window cannot hold
focus, so a focus gain means the window is back.
"""

from __future__ import annotations


def _app():
    from lumaviewpro import LumaViewProApp

    # a stand-in by design: the subject is the app's window handlers, which
    # Kivy calls; building the app needs a window and the whole widget tree.
    class _App:
        _window_minimized = False
        _on_window_minimize = LumaViewProApp._on_window_minimize
        _on_window_restore = LumaViewProApp._on_window_restore
        _on_window_focus = LumaViewProApp._on_window_focus
        window_minimized = LumaViewProApp.window_minimized

    return _App()


def test_a_focus_gain_after_a_minimize_is_the_window_back_without_a_restore():
    app = _app()
    app._on_window_minimize()
    app._on_window_focus(None, False)
    assert app.window_minimized()
    app._on_window_focus(None, True)
    assert not app.window_minimized()


def test_a_restore_is_the_window_back():
    app = _app()
    app._on_window_minimize()
    app._on_window_restore()
    assert not app.window_minimized()
