# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""UI dispatch utilities for module-layer code.

Module-layer code must not import Kivy. This module provides a
ui_dispatch function that is set by the GUI layer at startup.
Non-GUI contexts (tests, headless, REST) get direct invocation.
"""

from collections.abc import Callable

from lvp_logger import logger

# Global UI dispatcher -- set by lumaviewpro.py at startup to
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


def schedule_ui(func: Callable, timeout: float = 0) -> None:
    """Schedule a function on the UI thread, or call directly if no GUI.

    Same signature as Clock.schedule_once -- func receives dt argument.
    """
    if _ui_dispatcher is not None:
        _ui_dispatcher(func, timeout)
    else:
        # No GUI -- call directly (tests, headless, REST API)
        if callable(func):
            try:
                func(0)
            except Exception:
                # Deliberately more forgiving than the GUI branch, which
                # re-raises: a REST or headless caller must not be killed
                # by one bad UI callback. The failure still has to be
                # visible -- discarding it made a throwing protocol or
                # recording callback produce no record at all, so the run
                # looked like it had succeeded.
                logger.exception('[KivyUtils] scheduled UI callback failed')
