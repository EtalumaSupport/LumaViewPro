# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Debounce decorator for preventing rapid repeated method calls.

Use on button handlers that trigger hardware commands to prevent
double-clicks from sending duplicate commands.
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from modules import gui_logger

_F = TypeVar('_F', bound=Callable)


def debounce(delay: float) -> Callable[[_F], _F]:
    """Decorator that ignores repeated calls within *delay* seconds.

    Each decorated method gets its own per-instance window for each set of
    arguments, stored as ``self._debounce_<method_name>``.  A call with the
    same arguments before *delay* seconds have elapsed is dropped, and the
    drop is recorded: the handler's own record is inside the method, so a
    press dropped here would otherwise leave no line at all.  A call with
    different arguments is a different press -- the turret's slot 3 pressed
    just after slot 2 runs.

    Args:
        delay: Minimum interval in seconds between accepted calls.

    Example::

        class VerticalControl:
            @debounce(0.3)
            def coarse_up(self):
                io_executor.put(IOTask(...))
    """

    def decorator(func):
        attr = f'_debounce_{func.__name__}'

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            now = time.monotonic()
            key = (args, tuple(sorted(kwargs.items())))
            windows = getattr(self, attr, None)
            if windows is None:
                windows = {}
                setattr(self, attr, windows)
            if now - windows.get(key, 0.0) < delay:
                detail = f'dropped args={args}' if args else 'dropped'
                gui_logger.button(f'{type(self).__name__}.{func.__name__}', detail)
                return
            windows[key] = now
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
