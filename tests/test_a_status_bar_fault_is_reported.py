# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A fault while composing the window title is reported, not hidden.

The status bar composes the title ten times a second on the clock. Its body
was wrapped in an except-all that logged at DEBUG, so any fault in the title
-- the FPS, the cursor readouts, the event text -- disappeared at the default
log level. The title composition now catches nothing: the status bar is
scheduled through ``schedule_ui_interval``, which reports a raising tick
and keeps the schedule running (its contract, tested in
test_a_repeating_ui_callback_runs_on_after_it_raises.py; the cadence and
the routine's use, in test_status_bar_xy_readout.py).
"""

import inspect

import ui.shader


def test_the_title_composition_catches_nothing():
    source = inspect.getsource(ui.shader.ShaderViewer._update_status_bar)
    assert 'except' not in source
