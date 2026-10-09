# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The window's close: the GUI asks and shows; the Session closes.

The window's X asks, naming what ``session.live_work`` lists, and then
runs ``session.shutdown`` on a thread of its own, so the Kivy thread keeps
drawing the progress and the Discard button stays live; the app stops when
the close returns. The GUI does not stop a run or a recording, wait for
one, or decide what is still draining: the Session's close does each, the
same for every host. Four structural invariants hold it there:

1. The close handler refuses re-entry while a close is in flight. A Kivy
   Popup is modal only for in-canvas touch; ``on_request_close`` is an
   OS-level window event and fires again on a second X.
2. The progress watch is an interval that stops itself by returning
   False, once the close's thread has ended.
3. The close runs ``shutdown`` on its own thread, not the Kivy thread.
4. Nothing in the GUI's close stops or waits for a run or recording, or
   reads the drains beneath the Session.

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
_CLOSE_ATTR = '_close_thread'


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


def _attrs(node):
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def test_the_close_watch_stops_itself():
    """The interval callback returns False on its terminal path.

    Without it the event repeats forever, calling stop() on every tick.
    """
    watch = _interval_callback(_method(_app_class(_module_tree()), '_close_the_session'))

    returns_false = [
        node
        for node in ast.walk(watch)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]

    assert returns_false, (
        f'{watch.name} must return False on the tick that finds the close ended, so '
        'the Kivy interval stops itself.'
    )


def test_close_request_refuses_re_entry_while_a_close_is_in_flight():
    """on_request_close and the close itself consult the in-flight close.

    A second X while the progress popup is up otherwise starts a second
    close and a second popup.
    """
    class_node = _app_class(_module_tree())
    for name in ('on_request_close', '_close_the_session'):
        assert _CLOSE_ATTR in _attrs(_method(class_node, name)), (
            f'{name} must consult self.{_CLOSE_ATTR} and refuse to start a second '
            'close while one is in flight.'
        )


def test_the_close_attribute_has_a_class_level_default():
    """The guard reads the attribute before any close has set it."""
    class_node = _app_class(_module_tree())

    declared = [
        target.id
        for node in class_node.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == _CLOSE_ATTR
    ]

    assert declared, f'{_APP_CLASS} must declare {_CLOSE_ATTR} at class level.'


def test_the_window_close_runs_the_sessions_close_on_its_own_thread():
    """shutdown runs on a thread the close starts, so the progress can draw."""
    close = _method(_app_class(_module_tree()), '_close_the_session')

    threads = [
        node
        for node in ast.walk(close)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'Thread'
    ]
    targets = [
        kw.value.attr
        for call in threads
        for kw in call.keywords
        if kw.arg == 'target' and isinstance(kw.value, ast.Attribute)
    ]

    assert targets == ['_shut_the_session_down'], (
        f'_close_the_session must start one thread running _shut_the_session_down: {targets}'
    )
    assert 'shutdown' in _attrs(_method(_app_class(_module_tree()), '_shut_the_session_down'))


def test_the_close_asks_from_the_sessions_live_work_and_discards_through_it():
    """What the close names, and its one escape, are the Session's."""
    class_node = _app_class(_module_tree())

    assert 'live_work' in _attrs(_method(class_node, 'on_request_close'))
    reads = _attrs(_method(class_node, '_close_the_session'))
    assert {'live_work', 'discard_close_drain'} <= reads, (
        "the close must show the session's live_work and discard through discard_close_drain"
    )


def test_the_gui_neither_ends_nor_waits_for_a_run_or_recording_at_close():
    """The Session's close stops and waits; a second copy here drifts from it."""
    class_node = _app_class(_module_tree())
    decided_here = {
        'force_reset',
        'cancel_all_protocols',
        'wait_for_run_idle',
        'is_recording',
        'is_busy',
        'video_drain_busy',
        'sequenced_capture_runner',
    }
    for name in ('on_request_close', '_close_the_session', '_prepare_the_close', 'on_stop'):
        found = _attrs(_method(class_node, name)) & decided_here
        assert not found, f'{name} still decides the close itself: {sorted(found)}'
    stops = [
        node
        for node in ast.walk(class_node)
        if isinstance(node, ast.Attribute)
        and node.attr == 'stop'
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == 'manual_recording'
    ]
    assert not stops, "the GUI stops the recording itself; the Session's close does"


def test_the_apps_stop_ends_the_loop_and_leaves_on_stop_to_run():
    """Every exit runs on_stop once.

    Kivy's App.stop() dispatches on_stop and then run() dispatches it again
    when the loop returns, so an exit made from inside the loop (Confirm
    Exit, the drain close, quit and repair) cancelled, saved and tore down
    twice, the second save after the scope was disconnected. The app's
    stop() only ends the loop, and run() is the one dispatcher.
    """
    stop = _method(_app_class(_module_tree()), 'stop')

    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
        for node in ast.walk(stop)
        if isinstance(node, ast.Call)
    ]

    assert 'stopTouchApp' in calls, f'{_APP_CLASS}.stop must end the event loop: {calls}'
    assert not {'dispatch', '_stop', 'stop', 'on_stop'} & set(calls), (
        f'{_APP_CLASS}.stop must not dispatch on_stop itself (it calls {calls}); '
        'run() dispatches it when the loop returns, and a second dispatch repeats '
        'the whole shutdown.'
    )
