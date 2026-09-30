# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A fault while composing the window title is reported, not hidden.

The status bar composes the title ten times a second on the clock. Its body
was wrapped in an except-all that logged at DEBUG, so any fault in the title
-- the FPS, the cursor readouts, the event text -- disappeared at the default
log level. The tick now draws through the GUI's boundary for a redraw no
person asked for, which reports the fault where it stops and keeps the
application running.
"""

import inspect

import ui.shader


def test_the_status_bar_tick_draws_through_the_gui_boundary():
    source = inspect.getsource(ui.shader.ShaderViewer.__init__)
    assert "draw_unasked(lambda: self._update_status_bar(dt), 'STATUS_BAR')" in source


def test_the_title_composition_catches_nothing():
    source = inspect.getsource(ui.shader.ShaderViewer._update_status_bar)
    assert 'except' not in source
