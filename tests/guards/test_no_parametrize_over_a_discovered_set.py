# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No parametrize fans out cases that are computed rather than written.

A parametrize is for different behaviours, or for cases that each need
their own fixture or isolation. One assertion over a discovered set or a
data table (exception classes, API members, files, a register table, a dB
range) is ONE test that walks the set and names every offender in its
failure, the form of ``test_every_refusal_states_its_reason.py``. Fanned
out, it costs a collected test per item on every run: one guard's three
checks over about 120 exception classes were 359 tests.

The shape this counts is the one the source shows: a ``parametrize`` (or a
fixture's ``params``) whose cases are not written at the site, a call, a
comprehension, or a name bound to either or imported. Cases written out
as a list or tuple, at the site or in a module-level name, are not
counted. Where a computed set is fair, each case its own scope, model or
dialect, say so with ``# parametrize-justified: <why>`` on the decorator's
line or within the three lines above it.

Every site the guard found at its first run was collapsed or justified
on 2026-10-12, so it is a ban, not a pin.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, iter_package_modules

_JUSTIFICATION = 'parametrize-justified'
_JUSTIFICATION_WINDOW = 3  # lines above the decorator that may carry it

_WRITTEN = (ast.List, ast.Tuple, ast.Set, ast.Dict)
_COMPUTED_INSIDE = (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp, ast.Starred)


def _module_bindings(tree):
    """``{name: value}`` for each module-level single-name assignment."""
    bound = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            bound[node.targets[0].id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            bound[node.target.id] = node.value
    return bound


def _cases(call):
    """The cases a ``parametrize`` or ``fixture(params=)`` call takes, else None."""
    if not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr == 'parametrize':
        for keyword in call.keywords:
            if keyword.arg == 'argvalues':
                return keyword.value
        return call.args[1] if len(call.args) > 1 else None
    if call.func.attr == 'fixture':
        for keyword in call.keywords:
            if keyword.arg == 'params':
                return keyword.value
    return None


def _written(cases, bindings):
    if isinstance(cases, ast.Name) and cases.id in bindings:
        cases = bindings[cases.id]
    if not isinstance(cases, _WRITTEN):
        return False
    return not any(isinstance(node, _COMPUTED_INSIDE) for node in ast.walk(cases))


def _justified(lines, lineno):
    window = lines[max(0, lineno - 1 - _JUSTIFICATION_WINDOW) : lineno]
    return any(_JUSTIFICATION in line for line in window)


def computed_parametrize_sites():
    """``{'tests/<file>.py': ['<line>: <cases>', ...]}`` for each unjustified site."""
    found: dict[str, list[str]] = {}
    for rel_path, tree in iter_package_modules(('tests',)):
        bindings = _module_bindings(tree)
        lines = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            cases = _cases(node)
            if cases is None or _written(cases, bindings):
                continue
            if lines is None:
                lines = (REPO_ROOT / rel_path).read_text(encoding='utf-8').splitlines()
            if not _justified(lines, node.lineno):
                found.setdefault(rel_path, []).append(f'{node.lineno}: {ast.unparse(cases)}')
    return found


def test_no_parametrize_fans_out_a_computed_set():
    found = computed_parametrize_sites()
    assert not found, (
        'A parametrize fans a computed set out into one test per item. If it '
        'is one assertion over the set, make it one test that walks the set '
        'and names every offender (tests/case_walk.py); if each case is its own '
        'behaviour or needs its own fixture, write the cases out or mark the site '
        f'"# {_JUSTIFICATION}: <why>". Sites: {found}'
    )
