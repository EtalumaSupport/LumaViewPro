# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""AST seam checks: assert a function/method EXISTS on a production
module without pinning its source text.

A string pin like ``'def x(' in src`` breaks when the signature is
reformatted, wrapped, or gains a parameter; an AST lookup asserts the
seam itself (name, optionally parameter names and return annotation)
and survives any behavior-preserving refactor. Use these helpers for
"the API/driver must implement X" locks; keep behavioral assertions
for what X actually does.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


@cache
def parse_module(rel_path: str) -> ast.Module:
    """Parse a production module once per test session."""
    return ast.parse((REPO_ROOT / rel_path).read_text())


def iter_package_modules(packages):
    """Yield ``(rel_path, ast.Module)`` for every ``.py`` under ``packages``.

    The one walker for whole-package AST scans, so guards that sweep
    `modules/` + `ui/` share it instead of each hand-rolling a
    `rglob` + `ast.parse` loop. Paths are POSIX-relative to the repo
    root and sorted, so failure messages are stable across platforms.
    """
    for package in packages:
        for path in sorted((REPO_ROOT / package).rglob('*.py')):
            yield (
                path.relative_to(REPO_ROOT).as_posix(),
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path)),
            )


def production_modules():
    """Yield ``(rel_path, ast.Module)`` for every production module.

    ``iter_package_modules`` walks packages only, and the startup path
    lives in the top-level ``lumaviewpro.py`` -- the one module a
    package-only sweep would miss.
    """
    yield from iter_package_modules(('modules', 'ui'))
    yield 'lumaviewpro.py', parse_module('lumaviewpro.py')


def walk_defs(body, prefix=''):
    """Yield ``(qualname, node)`` for every def, methods and closures included."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from walk_defs(node.body, f'{prefix}{node.name}.')
        elif isinstance(node, _DEF_TYPES):
            qualname = f'{prefix}{node.name}'
            yield qualname, node
            yield from walk_defs(node.body, f'{qualname}.')


def innermost_subscript_key(node, keys):
    """The literal key in ``keys`` at the innermost slice of a subscript chain, else None.

    A fact written one level in -- ``settings['section'][n] = ...`` --
    has its key at the inner slice, so a slice-only match on the outer
    subscript would miss it.
    """
    while isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and key.value in keys:
            return key.value
        node = node.value
    return None


def writers_of_settings_keys(keys) -> set[tuple[str, str]]:
    """``(rel_path, qualname)`` for every production function that writes one of ``keys``.

    A write is a subscript store whose innermost literal key is in
    ``keys``, or an ``update_settings(key, ...)`` call naming one. The
    census a single-writer guard pins against: the guard names the one
    home of each fact and this returns everywhere that fact is written.
    """
    found = set()
    for rel_path, tree in production_modules():
        for qualname, fn in walk_defs(tree.body):
            for node in ast.walk(fn):
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                    if innermost_subscript_key(node, keys) is not None:
                        found.add((rel_path, qualname))
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'update_settings'
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in keys
                ):
                    found.add((rel_path, qualname))
    return found


def direct_call_names(fn) -> list[str]:
    """Names called in this function's OWN body; nested defs excluded.

    Without the exclusion an outer function would be credited with every
    call its closures make, and a "called exactly once" count would
    silently drift.
    """
    names = []
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return names


def find_def(rel_path: str, name: str, class_name: str | None = None):
    """Return the FunctionDef node for ``name``, or None when absent.

    With ``class_name``, only that class's subtree is searched;
    otherwise the whole module (nested and method defs included).
    """
    tree = parse_module(rel_path)
    scopes: list[ast.AST] = [tree]
    if class_name is not None:
        scopes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, _DEF_TYPES) and node.name == name:
                return node
    return None


def assert_def(
    rel_path: str,
    name: str,
    *,
    class_name: str | None = None,
    params: list[str] | None = None,
    has_params: list[str] | None = None,
    returns: str | None = None,
    msg: str = '',
) -> None:
    """Assert the function exists; optionally check its signature seam.

    Args:
        params: exact positional-arg name list (including self) when the
            full parameter list is the contract.
        has_params: parameter names that must be present (by name,
            positional or keyword-only) without constraining the rest.
        returns: the return annotation as source text (e.g. 'bool').
    """
    fn = find_def(rel_path, name, class_name)
    assert fn is not None, msg or f'{rel_path}: def {name}(...) not found'
    if params is not None:
        actual = [a.arg for a in fn.args.args]
        assert actual == list(params), (
            f'{rel_path}: {name} params {actual} != expected {list(params)}. {msg}'
        )
    if has_params is not None:
        present = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        missing = [p for p in has_params if p not in present]
        assert not missing, f'{rel_path}: {name} missing param(s) {missing}. {msg}'
    if returns is not None:
        actual_ret = ast.unparse(fn.returns) if fn.returns is not None else None
        assert actual_ret == returns, (
            f'{rel_path}: {name} return annotation {actual_ret!r} != {returns!r}. {msg}'
        )
