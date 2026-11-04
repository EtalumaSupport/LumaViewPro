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
import importlib
import re

from tests.ast_seams import MemberNames, iter_package_modules

# A parameter that names a time the member waits or holds for.
_A_WAIT = re.compile(r'(^|_)timeout(_s)?$|_delay_s$|^seconds$|_for_s$')
_THE_CHECK = MemberNames('modules/finite_number.py', {'refuse_unless_wait_s', 'WAIT_S'})
# The handle every run and job gives back: not marked, but every caller's.
_HANDLES = {('modules/operation.py', 'Operation')}
# A member whose wait is refused against a narrower range of its own, by the
# module constant holding that range; the range must end within WAIT_S.
_NARROWER = {
    'modules/lumascope_api/diagnostics.py::DiagnosticsAPI.run_grab_lifecycle_benchmark': (
        'modules.lumascope_api.diagnostics',
        '_BENCHMARK_DELAY_S',
    ),
}


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
                where = f'{rel_path}::{cls.name}.{fn.name}'
                if where in _NARROWER and _NARROWER[where][1] in ast.unparse(fn):
                    continue
                if not any(_checks(methods[name]) for name in called if name in methods):
                    found.append(f'{where} {waits}')
    return found


def test_every_wait_a_caller_passes_is_checked_where_it_enters():
    assert _unchecked() == []


def test_a_narrower_range_ends_within_the_longest_wait():
    from modules.finite_number import WAIT_S

    beyond = {
        where: getattr(importlib.import_module(module), constant)
        for where, (module, constant) in _NARROWER.items()
        if not WAIT_S[0] <= getattr(importlib.import_module(module), constant)[1] <= WAIT_S[1]
    }
    assert beyond == {}, f'a range past WAIT_S is no bound on the wait: {beyond}'
