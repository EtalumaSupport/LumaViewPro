# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No run starter in the GUI refuses a click: every refusal is the API's.

Contract
--------
A run-starting button hands its press to the engine through the boundary,
and whatever the engine refuses it raises, once, to the one reporter -- so a
REST or SDK caller gets the same refusal with the same words, and the GUI
decides nothing. The only branch a starter takes before the engine answers
is its own Stop: whether the run this button started is still live is the
engine's answer (``is_live_run``), and a press on it stops that run.

The other is a press whose own touch carried a refused edit
(``refused_in_this_input``): Kivy commits a focused field before the
button's ``on_release`` handler runs, the API has already refused that
edit and the boundary has shown its refusal, and starting would run the
value the person just tried to change. It refuses nothing of its own; it
is the input's half, which REST has no counterpart to. Every starter asks
it, and is bound ``on_release``: an ``on_press`` handler runs before the
field commits, so it would start on the old value and the question could
never hold.

Any other early return in a starter is a refusal the GUI made for itself:
a gate that repeats one the API already raises (the file-drain gate this
guard replaced was one), or one only the GUI enforces, which REST would
never see. Either way it fails the build.

Test approach
-------------
The Kivy UI classes cannot be instantiated headlessly (ids, _app_ctx,
worker pool), so the rule is held by AST over the starter roster, the same
way the sibling guards in test_protocol_start_refusal_ui_gate.py and
test_controls_lockout.py do it. The classifier is proven on planted
branches, so a clean roster is not a vacuous pass.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import REPO_ROOT, parse_module


# Every button that starts a run or capture from the GUI: where it lives,
# the kv handlers that reach it, and whether it is also its own run's Stop.
RUN_STARTERS = (
    # The protocol panel's Scan, Protocol and Autofocus Scan buttons share one press.
    (
        'ui/protocol_settings.py',
        'ProtocolSettings',
        '_press_panel_run',
        ('run_scan_from_ui', 'run_protocol_from_ui', 'run_autofocus_scan_from_ui'),
        True,
    ),
    ('ui/zstack.py', 'ZStack', 'run_zstack_acquire_from_ui', ('run_zstack_acquire_from_ui',), True),
    (
        'ui/vertical_control.py',
        'VerticalControl',
        'run_autofocus_from_ui',
        ('run_autofocus_from_ui',),
        True,
    ),
    (
        'ui/composite_capture.py',
        'CompositeCapture',
        'composite_capture',
        ('composite_capture',),
        True,
    ),
    # A still is one frame: there is no run of its own to stop.
    ('ui/composite_capture.py', 'CompositeCapture', 'live_capture', ('live_capture',), False),
    ('ui/main_display.py', 'MainDisplay', 'record_button', ('record_button',), True),
)


def _method_node(rel_path: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = parse_module(rel_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    return sub
    raise AssertionError(f'{rel_path}: {class_name}.{method_name} is gone -- roster drifted')


def _called_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                out.add(func.id)
            elif isinstance(func, ast.Attribute):
                out.add(func.attr)
    return out


def _returning_branches(method: ast.FunctionDef):
    """Yield (if-node, branch) for each branch of an ``if`` that returns.

    Only the starter's own body counts: a nested function (the start closure
    handed to the pool) runs on the engine's side of the boundary.
    """
    nested = {
        sub
        for node in ast.walk(method)
        if isinstance(node, (ast.FunctionDef, ast.Lambda)) and node is not method
        for sub in ast.walk(node)
    }
    for node in ast.walk(method):
        if not isinstance(node, ast.If) or node in nested:
            continue
        for branch in (node.body, node.orelse):
            if any(isinstance(stmt, ast.Return) for stmt in branch):
                yield node, branch


def _is_own_stop(node: ast.If) -> bool:
    # The own-run question: the handle the button's start returned is live,
    # or, for Record, the recording controller has one open.
    return any(
        isinstance(sub, ast.Attribute) and sub.attr in ('is_live', 'is_recording')
        for sub in ast.walk(node.test)
    )


def _is_a_refused_edits_touch(node: ast.If) -> bool:
    # The question itself, or the starters' one routine that asks it.
    return bool(
        {'refused_in_this_input', 'start_pressed_off_a_refused_edit'} & _called_names(node.test)
    )


def _refusals(method: ast.FunctionDef) -> list[int]:
    return [
        node.lineno
        for node, _branch in _returning_branches(method)
        if not (_is_own_stop(node) or _is_a_refused_edits_touch(node))
    ]


def _asks_for_a_refused_edit(method: ast.FunctionDef) -> bool:
    return any(_is_a_refused_edits_touch(node) for node, _branch in _returning_branches(method))


def test_no_run_starter_refuses_a_click():
    """The guard: a starter's only early returns are its own Stop and a refused edit's touch."""
    refusing = []
    drifted = []
    for rel_path, class_name, method_name, _entries, has_stop in RUN_STARTERS:
        method = _method_node(rel_path, class_name, method_name)
        stops = sum(1 for node, _ in _returning_branches(method) if _is_own_stop(node))
        if stops != int(has_stop):
            drifted.append(f'{class_name}.{method_name}: {stops} own-run Stop branches')
        refusing += [
            f'{rel_path}:{line} in {class_name}.{method_name}' for line in _refusals(method)
        ]

    assert not drifted, (
        'the AST shapes drifted, so a clean result would mean nothing: ' + ', '.join(drifted)
    )
    assert not refusing, (
        'a run starter refuses a click itself instead of handing the press to '
        'the engine, whose refusal every client sees: ' + ', '.join(refusing)
    )


def test_every_run_starter_starts_nothing_off_a_refused_edit():
    """Each starter asks whether its press carried a refused edit, and returns when it did."""
    silent = [
        f'{rel_path} {class_name}.{method_name}'
        for rel_path, class_name, method_name, _entries, _stop in RUN_STARTERS
        if not _asks_for_a_refused_edit(_method_node(rel_path, class_name, method_name))
    ]
    assert not silent, (
        'a run starter pressed straight off a refused edit would start on the '
        'stored value the person just tried to change: ' + ', '.join(silent)
    )


_KV_BINDING = re.compile(r'^\s*(on_press|on_release)\s*:\s*root\.(\w+)\(', re.MULTILINE)


def test_every_run_starter_is_bound_on_release():
    """Each starter's kv handler runs on release, after a focused field has committed."""
    bound: dict[str, set[str]] = {}
    for kv in sorted((REPO_ROOT / 'ui').glob('*.kv')):
        for event, handler in _KV_BINDING.findall(kv.read_text()):
            bound.setdefault(handler, set()).add(event)
    wrong = [
        f'{handler} bound {sorted(bound.get(handler, set())) or "nowhere"}'
        for _path, _class, _method, entries, _stop in RUN_STARTERS
        for handler in entries
        if bound.get(handler) != {'on_release'}
    ]
    assert not wrong, (
        'a run starter must be bound on_release alone: on_press runs before the '
        'field the person just typed in commits: ' + ', '.join(wrong)
    )


def test_the_guard_catches_a_gate_and_a_silent_return():
    """The guard's own falsifier: a gate that notifies and one that says
    nothing are both refusals the GUI made."""
    module = ast.parse(
        'class X:\n'
        '    def starter(self):\n'
        '        if files_draining():\n'
        '            show_notification_popup(title="Wait")\n'
        '            return\n'
        '        if not ready:\n'
        '            return\n'
        '        start()\n'
    )
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef))
    assert _refusals(method) == [3, 6]


def test_the_guard_leaves_the_own_run_stop_alone():
    module = ast.parse(
        'class X:\n'
        '    def starter(self):\n'
        '        run = self._run\n'
        '        if run is not None and run.is_live:\n'
        '            submit(run.stop)\n'
        '            return\n'
        '        def _start():\n'
        '            if nothing_to_do:\n'
        '                return\n'
        '        submit(_start)\n'
    )
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef))
    assert _refusals(method) == [], 'the own Stop or the pool-side closure was read as a refusal'


def test_the_guard_leaves_a_refused_edits_touch_alone_and_nothing_else_like_it():
    module = ast.parse(
        'class X:\n'
        '    def starter(self):\n'
        '        if refused_in_this_input():\n'
        '            self.draw_protocol_buttons()\n'
        '            return\n'
        '        if refused_by_the_gui():\n'
        '            return\n'
        '        if start_pressed_off_a_refused_edit("X", self.redraw):\n'
        '            return\n'
        '        start()\n'
    )
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef))
    assert _refusals(method) == [6]
    assert _asks_for_a_refused_edit(method)


def test_the_guard_catches_a_starter_that_never_asks():
    """A starter with only its own Stop, or asking inside its pool-side closure, does not ask."""
    module = ast.parse(
        'class X:\n'
        '    def starter(self):\n'
        '        if self._run is not None and self._run.is_live:\n'
        '            return\n'
        '        def _start():\n'
        '            if refused_in_this_input():\n'
        '                return\n'
        '        submit(_start)\n'
    )
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef))
    assert not _asks_for_a_refused_edit(method)
