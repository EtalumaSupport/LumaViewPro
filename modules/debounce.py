# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Debounce decorator for preventing rapid repeated method calls.

Use on button handlers that trigger hardware commands to prevent
double-clicks from sending duplicate commands.
"""
import time
from functools import wraps


def debounce(delay: float):
    """Decorator that ignores repeated calls within *delay* seconds.

    Each decorated method gets its own per-instance timestamp stored as
    ``self._debounce_<method_name>``.  If the method is called again
    before *delay* seconds have elapsed, the call is silently dropped.

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
            last = getattr(self, attr, 0.0)
            if now - last < delay:
                return
            setattr(self, attr, now)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
