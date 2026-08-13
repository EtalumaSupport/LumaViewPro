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
