# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Name the test files that reach a set of changed production modules.

The per-commit targeted run used to be chosen by a grep for the changed
member's name. A grep finds names, not reach: a test that drives the member
through its owner's public surface, or feeds it a hand-built input, never
names it, and the push suite found seven such tests in five days, each at
the price of a second suite. This reads the import graph instead.

A test file reaches a changed module when any of these holds:

* it imports the module, at module level or inside a function;
* it imports a ``tests`` helper module that does (transitively);
* it requests a conftest fixture whose body imports the module, or whose
  body uses a name the conftest imported from it;
* it is a guard under ``tests/guards/``: a guard reads source files by
  path and imports nothing it checks, so every guard is in every set.

A changed module that an autouse fixture imports itself is reached by
every test in the suite; the answer is then the whole of ``tests/`` and the
reason is printed. (One it reaches only through a package that exports it,
or through a tests helper, runs it only where the test built something,
and the test's own imports show that.) What no import can show, a test
that hand-builds an installation tree or a settings file the changed
member reads, stays a census item for the author.

Usage, from the repository root (one test path per line on stdout, the
count and the reasons on stderr)::

    python3 tools/tests_reaching.py modules/scope_session.py ui/foo.py
    python3 tools/tests_reaching.py --commit HEAD
    python3 tools/tests_reaching.py --diff          # staged and unstaged

Pipe it to pytest: ``python3 -m pytest $(python3 tools/tests_reaching.py --diff)``.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PRODUCTION_ROOTS = ('modules/', 'ui/', 'drivers/', 'tools/', 'lumaviewpro.py')
GUARDS = 'tests/guards'


def module_name(rel_path: str) -> str:
    """The dotted name a repository-relative ``.py`` path is imported by."""
    parts = Path(rel_path).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def package_of(rel_path: str) -> str:
    """The package a repository-relative ``.py`` file's relative imports start from."""
    name = module_name(rel_path)
    if Path(rel_path).name == '__init__.py':
        return name
    return name.rpartition('.')[0]


def imported_modules(tree: ast.AST, package: str = '') -> set[str]:
    """Every module name an AST imports, at any depth.

    ``from pkg import name`` records both ``pkg`` and ``pkg.name``: the
    name may be a submodule, and recording it costs nothing when it is a
    function. A relative import is resolved against ``package``, the
    containing package of the file (``package_of``), one level up per dot
    beyond the first.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                anchor = package.split('.')[: len(package.split('.')) - node.level + 1]
                base = '.'.join(part for part in ('.'.join(anchor), base) if part)
            if base:
                found.add(base)
                found.update(f'{base}.{alias.name}' for alias in node.names)
    return found


def _reaches(imported: set[str], changed: set[str]) -> set[str]:
    """The changed modules an import set touches: the module or a child of it."""
    hit = set()
    for name in imported:
        for module in changed:
            if name == module or name.startswith(module + '.'):
                hit.add(module)
    return hit


@dataclass
class _Fixture:
    name: str
    autouse: bool
    modules: set[str] = field(default_factory=set)


def _fixtures(tree: ast.Module, package: str) -> list[_Fixture]:
    """The fixtures a conftest defines, each with the modules its body reaches.

    A body reaches a module it imports itself, and one the conftest imported
    at module level under a name the body uses (``build_scope`` is imported
    once at the top and called inside ``sim_scope``).
    """
    top_level: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level[(alias.asname or alias.name).split('.')[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                top_level[alias.asname or alias.name] = f'{node.module}.{alias.name}'
    fixtures = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        marks = [d for d in node.decorator_list if 'fixture' in ast.unparse(d)]
        if not marks:
            continue
        autouse = any(
            isinstance(d, ast.Call)
            and any(
                k.arg == 'autouse' and getattr(k.value, 'value', False) is True for k in d.keywords
            )
            for d in marks
        )
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        modules = imported_modules(node, package)
        modules.update(top_level[name] for name in used if name in top_level)
        fixtures.append(_Fixture(node.name, autouse, modules))
    return fixtures


def _requested_fixtures(tree: ast.Module) -> set[str]:
    """The fixture names a test module asks for: parameters and usefixtures marks."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.update(a.arg for a in node.args.args + node.args.kwonlyargs)
            for d in node.decorator_list:
                text = ast.unparse(d)
                if 'usefixtures' in text and isinstance(d, ast.Call):
                    names.update(c.value for c in d.args if isinstance(c, ast.Constant))
    return names


def _packages_exporting(root: Path, changed: set[str]) -> set[str]:
    """The ancestor packages whose ``__init__`` imports a changed module.

    ``from modules.lumascope_api import Lumascope`` runs the package's
    ``__init__``, which imports ``_lumascope``; a test importing the package
    reaches the module without ever naming it. Walked outward, so a package
    exporting a package that exports the module is found too.
    """
    aliases: set[str] = set()
    frontier = set(changed)
    while frontier:
        module = frontier.pop()
        parts = module.split('.')
        for depth in range(len(parts) - 1, 0, -1):
            package = '.'.join(parts[:depth])
            init = root / Path(*parts[:depth]) / '__init__.py'
            if package in aliases or package in changed or not init.exists():
                continue
            try:
                tree = ast.parse(init.read_text(encoding='utf-8'), filename=str(init))
            except SyntaxError:
                continue
            if _reaches(imported_modules(tree, package), {module}):
                aliases.add(package)
                frontier.add(package)
    return aliases


@dataclass
class Reach:
    files: list[str]
    reasons: list[str]


def reach_of(root: Path, changed_paths: list[str]) -> Reach:
    """The test files under ``root`` that reach the changed production paths."""
    edited = {module_name(p) for p in changed_paths if p.endswith('.py')}
    exporting = _packages_exporting(root, edited)
    changed = edited | exporting
    tests_dir = root / 'tests'
    trees: dict[str, ast.Module] = {}
    for path in sorted(tests_dir.rglob('*.py')):
        rel = path.relative_to(root).as_posix()
        try:
            trees[rel] = ast.parse(path.read_text(encoding='utf-8'), filename=rel)
        except SyntaxError:
            continue
    imports = {rel: imported_modules(tree, package_of(rel)) for rel, tree in trees.items()}

    # A tests helper reaches what it imports, and what the helpers it imports
    # reach; a fixed point over the tests package.
    helper_reach = {rel: _reaches(names, changed) for rel, names in imports.items()}
    by_module = {module_name(rel): rel for rel in trees}
    grew = True
    while grew:
        grew = False
        for rel, names in imports.items():
            for name in names:
                helper = by_module.get(name)
                if helper and not helper_reach[helper] <= helper_reach[rel]:
                    helper_reach[rel] |= helper_reach[helper]
                    grew = True

    fixtures: list[_Fixture] = []
    for rel, tree in trees.items():
        if Path(rel).name == 'conftest.py':
            fixtures.extend(_fixtures(tree, package_of(rel)))
    fixture_reach = {}
    autouse_direct: dict[str, set[str]] = {}
    for fixture in fixtures:
        direct = _reaches(fixture.modules, changed)
        reach = set(direct)
        for name in fixture.modules:
            # `from tests.scope_fakes import build_scope` names the helper by
            # its member; the helper's own reach is under the module's name.
            helper = by_module.get(name) or by_module.get(name.rsplit('.', 1)[0])
            if helper:
                reach |= helper_reach[helper]
        fixture_reach[fixture.name] = reach
        if fixture.autouse:
            autouse_direct[fixture.name] = direct

    reasons = []
    if exporting:
        reasons.append('reached through the package import too: ' + ', '.join(sorted(exporting)))
    # An autouse fixture that imports the edited module itself runs it in
    # every test. One that reaches it only through a package exporting it
    # (the API package, imported to disconnect the scopes a test built) or
    # through a tests helper it names runs the module only where the test
    # built something, and the test's own imports show that.
    autouse_hits = sorted(
        f'{module} is imported by the autouse fixture {name}'
        for name, direct in autouse_direct.items()
        for module in direct & edited
    )
    test_files = sorted(rel for rel in trees if Path(rel).name.startswith('test_'))
    if autouse_hits:
        reasons = ['the whole suite: ' + '; '.join(autouse_hits)]
        return Reach(test_files, reasons)

    chosen: dict[str, str] = {}
    for rel in test_files:
        if rel.startswith(GUARDS + '/'):
            chosen[rel] = 'guard'
        elif helper_reach[rel]:
            chosen[rel] = 'import'
        else:
            wanted = _requested_fixtures(trees[rel])
            if any(fixture_reach.get(name) for name in wanted):
                chosen[rel] = 'fixture'
    counts = {
        kind: sum(1 for k in chosen.values() if k == kind)
        for kind in ('import', 'fixture', 'guard')
    }
    reasons.append(
        f'{len(chosen)} test files for {sorted(changed)}: {counts["import"]} by import '
        f'(module level or inside a function, or through a tests helper), '
        f'{counts["fixture"]} through a conftest fixture, {counts["guard"]} guards (always)'
    )
    return Reach(sorted(chosen), reasons)


def _changed_from_git(root: Path, commit: str | None, diff: bool) -> list[str]:
    if commit:
        out = subprocess.run(
            ['git', '-C', str(root), 'show', '--name-only', '--format=', commit],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    else:
        out = subprocess.run(
            ['git', '-C', str(root), 'diff', '--name-only', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return [p for p in out.split() if p.startswith(PRODUCTION_ROOTS)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('paths', nargs='*', help='changed production paths, repository-relative')
    parser.add_argument('--commit', help='take the changed paths from this commit')
    parser.add_argument(
        '--diff', action='store_true', help='take them from the working tree against HEAD'
    )
    parser.add_argument(
        '--root', default='.', help='the repository root (default: the current directory)'
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    paths = list(args.paths)
    if args.commit or args.diff:
        paths += _changed_from_git(root, args.commit, args.diff)
    paths = [p for p in paths if p.startswith(PRODUCTION_ROOTS) and p.endswith('.py')]
    if not paths:
        print('no changed production .py path given', file=sys.stderr)
        return 2
    reach = reach_of(root, paths)
    for reason in reach.reasons:
        print(reason, file=sys.stderr)
    print('\n'.join(reach.files))
    return 0


if __name__ == '__main__':
    sys.exit(main())
