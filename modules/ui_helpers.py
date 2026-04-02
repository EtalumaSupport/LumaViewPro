# Copyright Etaluma, Inc.
"""Compatibility shim — ui_helpers has moved to ui/ui_helpers.py.

This file re-exports everything so existing imports continue to work.
New code should import from ui.ui_helpers directly.
"""
# ruff: noqa: F401
from ui.ui_helpers import (  # noqa: F401
    set_last_save_folder,
    focus_log,
    update_autofocus_selection_after_protocol,
    find_nearest_step,
    scope_leds_off,
    move_absolute_position,
    move_relative_position,
    move_home,
    set_recording_title,
    set_writing_title,
    reset_title,
    move_home_cb,
    live_histo_off,
    live_histo_reverse,
    reset_acquire_ui,
    reset_stim_ui,
    cleanup_scrollview_viewport,
)
