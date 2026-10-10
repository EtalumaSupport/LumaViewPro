# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The title's pixel readout clears when the pointer leaves the window.

The readout shows while the hover flag is set, and the flag was cleared only
by a mouse position outside the image. The window sends no mouse position
once the pointer has left it, so leaving through the window's edge froze the
readout on the last pixel it saw (seen in the sim: it kept showing until the
pointer came back). The window's cursor-leave event now clears it.
"""

import inspect
from types import SimpleNamespace

import ui.shader


def test_leaving_the_window_clears_the_hover_flag():
    viewer = SimpleNamespace(_mouse_over_image=True)
    ui.shader.ShaderViewer._on_cursor_leave(viewer, None)
    assert viewer._mouse_over_image is False


def test_the_viewer_listens_for_the_pointer_leaving_the_window():
    source = inspect.getsource(ui.shader.ShaderViewer.__init__)
    assert 'Window.bind(on_cursor_leave=self._on_cursor_leave)' in source
