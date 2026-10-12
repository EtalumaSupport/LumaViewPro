# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Typed text has one emitter, and it is never deferred.

A deferring wrapper around the typed-text emitter meant the apply record
reached the file BEFORE the line saying what was typed, and a freeze inside
the deferral window lost the last entry entirely. Every adopter now calls
``gui_logger.text_input`` directly, so the record lands before the entry does
anything. The fact pinned is that no call defers it, whatever the wrapper is
named: a ``gui_logger.text_input`` call is never the body of a callable handed
to the Kivy clock, and never inside a ``@debounce`` handler, which drops a
call made inside its window and the record with it.
"""

from __future__ import annotations

import ast

from tests.ast_seams import production_modules, walk_defs

# The Kivy clock's members that run a callable later.
_DEFERRING = frozenset({'schedule_once', 'schedule_interval', 'create_trigger'})
_DEBOUNCE = 'debounce'


def _is_text_input(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'text_input'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'gui_logger'
    )


def _is_text_input_reference(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'text_input'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'gui_logger'
    )


def _records_typed_text(node):
    """True when ``node`` holds a typed-text record (a call, or the emitter itself)."""
    return any(_is_text_input(n) or _is_text_input_reference(n) for n in ast.walk(node))


def _is_debounced(fn):
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, 'id', '')
        if name == _DEBOUNCE:
            return True
    return False


def deferred_typed_text(tree):
    """``(line, how)`` for every typed-text record that is deferred or droppable.

    The callable handed to the clock is read where it is written: a lambda or
    ``functools.partial`` in the call itself, or a def of that name in the same
    module (a closure, a method reached through ``self``).
    """
    defs = {}
    for qualname, fn in walk_defs(tree.body):
        defs.setdefault(qualname.rsplit('.', 1)[-1], []).append(fn)
    found = []
    for qualname, fn in walk_defs(tree.body):
        if _is_debounced(fn) and _records_typed_text(fn):
            found.append((fn.lineno, f'{qualname} records typed text under @{_DEBOUNCE}'))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _DEFERRING
            and node.args
        ):
            continue
        callable_ = node.args[0]
        if isinstance(callable_, ast.Name):
            named = callable_.id
        elif isinstance(callable_, ast.Attribute) and not _is_text_input_reference(callable_):
            named = callable_.attr
        else:
            named = None
        bodies = defs.get(named, []) if named else [callable_]
        if any(_records_typed_text(body) for body in bodies):
            found.append(
                (node.lineno, f'{node.func.attr}({ast.unparse(callable_)}) records typed text')
            )
    return found


def test_the_typed_text_record_is_never_deferred():
    """The typed-text record is written by the handler as it runs, before the
    entry acts: never handed to the clock to write later, never inside a
    handler that drops a call made too soon."""
    offenders = [
        f'{rel}:{line} {how}'
        for rel, tree in production_modules()
        for line, how in deferred_typed_text(tree)
    ]
    assert not offenders, (
        'a typed-text record is deferred or droppable; call gui_logger.text_input '
        'in the handler itself, before the entry acts, so the record lands in '
        'the order the user acted and a freeze cannot lose it:\n  ' + '\n  '.join(offenders)
    )


def test_the_deferral_scan_sees_each_form():
    """Guards the test above against passing because it matches nothing."""
    tree = ast.parse(
        'class Box:\n'
        '    def commit(self):\n'
        '        Clock.schedule_once(lambda dt: gui_logger.text_input("A", 1))\n'
        '        Clock.schedule_once(self._later)\n'
        '        Clock.create_trigger(functools.partial(gui_logger.text_input, "C", 3))\n'
        '    def _later(self, dt):\n'
        '        gui_logger.text_input("B", 2)\n'
        '    @debounce(0.3)\n'
        '    def typed(self):\n'
        '        gui_logger.text_input("D", 4)\n'
        '    def now(self):\n'
        '        gui_logger.text_input("E", 5)\n'
        '        Clock.schedule_once(self.redraw)\n'
        '    def redraw(self, dt):\n'
        '        pass\n'
    )
    assert [line for line, _how in deferred_typed_text(tree)] == [9, 3, 4, 5]
