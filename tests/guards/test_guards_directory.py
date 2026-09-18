# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The guards directory is the declaration.

A test whose subject is the whole tree or a whole live surface -- a count
ratchet, a surface-parity check, an architecture sweep -- is a guard, and
no run near a changed file ever selects it. A guard is declared by living
here: the pre-commit hook written by tools/install_hooks.py exports the
index and runs this directory, and nothing else. Three shapes would let a
guard be declared and still not gate, and this module refuses each:

* a ratchet registered from a module outside the directory is announced at
  the end of a full run and gates nothing;
* a guard that skips or xfails itself passes by not running, because pytest
  exits 0 on an all-skipped selection and the exit code is the hook's only
  signal;
* a module that computes the repo root from its own location keeps working
  after a move, pointed at the wrong tree -- the root has one owner,
  tests.ast_seams.REPO_ROOT.

Membership is a ruling, not a derivation: no syntactic pattern separates a
guard from a unit test that globs a fixture directory, so nothing here
polices completeness. A guard written outside the directory is caught where
such misses are caught today, by the full suite.
"""

import ast

from tests.ast_seams import iter_package_modules

GUARDS = 'tests/guards/'

# Every pytest spelling that lets a selected test finish without running
# its assertions and still exit 0.
_NOT_RUN_SPELLINGS = frozenset({'skip', 'skipif', 'importorskip', 'xfail'})


def _binds_ratchet_registry(tree) -> set[str]:
    """The names a module binds to tests.ratchets, in any import spelling."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'tests':
            bound.update(a.asname or a.name for a in node.names if a.name == 'ratchets')
        elif isinstance(node, ast.ImportFrom) and node.module == 'tests.ratchets':
            bound.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            bound.update(a.asname or a.name for a in node.names if a.name == 'tests.ratchets')
    return bound


def _registers_a_ratchet(tree) -> bool:
    if not _binds_ratchet_registry(tree):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'register':
            return True
        if isinstance(func, ast.Name) and func.id == 'register':
            return True
    return False


def test_every_ratchet_is_registered_from_the_guards_directory():
    """A ratchet outside the directory is announced by a full run and gated
    by nothing; the registry is consumed by conftest, which never registers."""
    outside = [
        rel
        for rel, tree in iter_package_modules(('tests',))
        if not rel.startswith(GUARDS) and _registers_a_ratchet(tree)
    ]
    assert outside == [], (
        'These modules register a ratchet from outside tests/guards/, where no '
        f'commit runs them: {outside}. Move each into tests/guards/.'
    )


def test_no_guard_skips_itself():
    """pytest exits 0 when every selected test skipped, and the hook reads
    only the exit code, so a guard that skips is a guard that passed."""
    offenders = []
    for rel, tree in iter_package_modules((GUARDS,)):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _NOT_RUN_SPELLINGS:
                offenders.append(f'{rel}:{node.lineno} {ast.unparse(node)}')
            elif isinstance(node, ast.ImportFrom) and node.module == 'pytest':
                offenders.extend(
                    f'{rel}:{node.lineno} from pytest import {a.name}'
                    for a in node.names
                    if a.name in _NOT_RUN_SPELLINGS
                )
    assert offenders == [], (
        'A guard may not skip or xfail itself; make the empty case a pass on '
        f'the empty set instead: {offenders}'
    )


def test_no_guard_computes_its_own_repo_root():
    """The repo root has one owner. A root hand-rolled from the module's own
    location survives a move of the module and points at the wrong tree,
    silently; this is the whole class, so the name itself is refused. A
    production module's own file attribute is a different thing and is not."""
    offenders = [
        f'{rel}:{node.lineno}'
        for rel, tree in iter_package_modules((GUARDS,))
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == '__file__'
    ]
    assert offenders == [], (
        f'Use tests.ast_seams.REPO_ROOT instead of a hand-rolled root: {offenders}'
    )


def test_the_hook_runs_the_guards_directory():
    """The stage is read through the installer's symbol, not the file: the
    installer is the one writer of the hook, and what it writes is the gate."""
    from tools.install_hooks import _HOOK_SCRIPT

    lines = [ln.strip() for ln in _HOOK_SCRIPT.splitlines() if not ln.strip().startswith('#')]
    assert any(ln.startswith('python3 -m pytest') and GUARDS.rstrip('/') in ln for ln in lines), (
        'The pre-commit hook must run pytest on tests/guards; nothing else runs them per commit.'
    )
    assert any('git checkout-index' in ln for ln in lines), (
        'The gate judges the index (what will be committed), not the working tree.'
    )
