# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every wait a caller passes through the API is checked against ``WAIT_S``.

A wait past ``threading.TIMEOUT_MAX`` raises ``OverflowError`` out of the
wait itself, and that limit is 49.7 days on Windows and about 292 years
elsewhere: ``wait_until_finished_moving`` took a caller's ``1e12`` to its
arrival event unchecked, and REST answered a 500. Fifteen members shared
the gap. Each now asks ``refuse_unless_wait_s`` (or bounds by ``WAIT_S``)
where the wait enters, itself or in a method of its own class it calls;
a new member that takes a wait and does neither is named here.
"""

import ast
import re

from tests.ast_seams import iter_package_modules

# A parameter that names a time the member waits or holds for.
_A_WAIT = re.compile(r'(^|_)timeout(_s)?$|_delay_s$|^seconds$|_for_s$')
_THE_CHECK = ('refuse_unless_wait_s', 'WAIT_S')
# The handle every run and job gives back: not marked, but every caller's.
_HANDLES = {('modules/operation.py', 'Operation')}


def _is_api(fn: ast.FunctionDef) -> bool:
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == 'api':
            return True
    return False


def _checks(fn: ast.FunctionDef) -> bool:
    return any(name in ast.unparse(fn) for name in _THE_CHECK)


def _unchecked():
    found = []
    for rel_path, tree in iter_package_modules(['modules']):
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
            handle = (rel_path, cls.name) in _HANDLES
            for fn in methods.values():
                if fn.name.startswith('_') or not (handle or _is_api(fn)):
                    continue
                waits = [a.arg for a in fn.args.args + fn.args.kwonlyargs if _A_WAIT.search(a.arg)]
                if not waits or _checks(fn):
                    continue
                called = {
                    node.func.attr
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'self'
                }
                if not any(_checks(methods[name]) for name in called if name in methods):
                    found.append(f'{rel_path}::{cls.name}.{fn.name} {waits}')
    return found


def test_every_wait_a_caller_passes_is_checked_where_it_enters():
    assert _unchecked() == []
