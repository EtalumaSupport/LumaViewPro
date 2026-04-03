# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""UI dispatch utilities for module-layer code.

Rule 15: Module-layer code must not import Kivy. This module provides
a ui_dispatch function that is set by the GUI layer at startup.
Non-GUI contexts (tests, headless, REST) get direct invocation.
"""

# Global UI dispatcher — set by lumaviewpro.py at startup to
# Clock.schedule_once. Default is direct invocation.
_ui_dispatcher = None


def set_ui_dispatcher(dispatcher):
    """Set the global UI dispatcher (called once by the GUI layer at startup).

    Args:
        dispatcher: A function with signature (func, timeout) that schedules
                    func on the GUI thread. Typically Clock.schedule_once.
    """
    global _ui_dispatcher
    _ui_dispatcher = dispatcher


def schedule_ui(func, timeout=0):
    """Schedule a function on the UI thread, or call directly if no GUI.

    Same signature as Clock.schedule_once — func receives dt argument.
    """
    if _ui_dispatcher is not None:
        _ui_dispatcher(func, timeout)
    else:
        # No GUI — call directly (tests, headless, REST API)
        if callable(func):
            try:
                func(0)
            except Exception:
                pass
