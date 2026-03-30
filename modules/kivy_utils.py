# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Kivy threading utilities shared across protocol and autofocus modules."""

try:
    from kivy.clock import Clock
except ImportError:
    class Clock:
        @staticmethod
        def schedule_once(func, timeout): func(0)
        @staticmethod
        def schedule_interval(func, interval): pass


def schedule_ui(func, timeout=0):
    """Schedule a function on the Kivy main thread, or call directly if
    no Kivy event loop is running (e.g., in tests).
    Same signature as Clock.schedule_once — func receives dt argument.
    """
    try:
        from kivy.base import EventLoop
        if EventLoop.status == 'started':
            Clock.schedule_once(func, timeout)
            return
    except Exception:
        pass
    func(0)
