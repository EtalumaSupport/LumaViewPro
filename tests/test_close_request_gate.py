# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The window-close gate: re-entry, and a watchdog that stops itself.

Closing the app during a video drain schedules a repeating Kivy interval
that polls the drain and calls ``stop()`` when it empties. Two structural
invariants keep that from turning into a runaway:

1. The interval callback must terminate ITSELF. A Kivy interval repeats
   unless its callback returns False; unscheduling via an attribute the
   callback reads off ``self`` is not the same thing, because a second
   close overwrites that attribute and orphans the first event -- which
   then never unschedules and calls ``stop()`` on every tick forever.
2. The close handler must refuse re-entry while a close is already in
   flight. A Kivy Popup is modal only for in-canvas touch;
   ``on_request_close`` is an OS-level window event and fires again on a
   second X even while the drain-progress popup is up.

Scanned rather than executed: ``lumaviewpro.py`` builds a Kivy widget
tree at import, which ``test_lumaviewpro_app_smoke`` records as
deliberately out of reach for the suite. The invariants above are
structural, so the structure is what gets parsed.
"""

import ast
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENTRY_POINT = os.path.join(_REPO_ROOT, 'lumaviewpro.py')
_APP_CLASS = 'LumaViewProApp'
_WATCH_ATTR = '_drain_close_watch'


def _module_tree():
    with open(_ENTRY_POINT, encoding='utf-8') as handle:
        return ast.parse(handle.read(), filename=_ENTRY_POINT)


def _app_class(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == _APP_CLASS:
            return node
    pytest.fail(f'{_APP_CLASS} not found in lumaviewpro.py (precondition)')


def _method(class_node, name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f'{_APP_CLASS}.{name} not found (precondition)')


def _interval_callback(method_node):
    """The function handed to Clock.schedule_interval inside ``method_node``."""
    scheduled = None
    for node in ast.walk(method_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'schedule_interval'
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            scheduled = node.args[0].id
    if scheduled is None:
        pytest.fail(
            f'{method_node.name} no longer schedules a named interval callback (precondition)'
        )
    for node in ast.walk(method_node):
        if isinstance(node, ast.FunctionDef) and node.name == scheduled:
            return node
    pytest.fail(f'the scheduled callback {scheduled!r} is not defined in {method_node.name}')


def test_drain_close_watchdog_stops_itself():
    """The interval callback returns False on its terminal path.

    Without it the event repeats forever whenever the attribute it
    unschedules has been overwritten by a second close, and each tick
    calls stop() again.
    """
    watch = _interval_callback(_method(_app_class(_module_tree()), '_close_with_drain_progress'))

    returns_false = [
        node
        for node in ast.walk(watch)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]

    assert returns_false, (
        f'{watch.name} must return False on the tick that finishes the drain, so the '
        'Kivy interval stops itself. Unscheduling through an attribute is not '
        'sufficient: a second close overwrites it and orphans this event, which then '
        'calls stop() on every tick.'
    )


def test_close_request_refuses_re_entry_while_a_close_is_in_flight():
    """on_request_close consults the in-flight close before starting another.

    A second X while the drain-progress popup is up otherwise runs the
    whole drain-close path a second time.
    """
    handler = _method(_app_class(_module_tree()), 'on_request_close')

    reads_watch = [
        node
        for node in ast.walk(handler)
        if isinstance(node, ast.Attribute) and node.attr == _WATCH_ATTR
    ]

    assert reads_watch, (
        f'on_request_close must consult self.{_WATCH_ATTR} and refuse to start a '
        'second drain close while one is already in flight. A Kivy Popup does not '
        'block the OS-level close event, so the second X re-enters this handler.'
    )


def test_watch_attribute_has_a_class_level_default():
    """The guard reads the attribute before any close has set it.

    Declared on the class so the first read cannot raise AttributeError,
    rather than each reader defending itself with getattr.
    """
    class_node = _app_class(_module_tree())

    declared = [
        target.id
        for node in class_node.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == _WATCH_ATTR
    ]

    assert declared, (
        f'{_APP_CLASS} must declare {_WATCH_ATTR} at class level so the re-entry '
        'guard can read it before the first drain close assigns it.'
    )
