# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where the build's files are, and where the run was launched from, are each computed once.

The build's own files (the code, its shipped data, the FX2 hex, the git
checkout) are found through ``_lumascope.lib.build_files.build_root``, links followed;
the folder the run was launched from through ``_lumascope.modules.path_utils.launch_root``,
links not followed. Before, each driver, ``src/_lumascope/lib/profile_trace`` and the GUI's
environment walked up from its own ``__file__`` or the interpreter's path,
the FX2 search computed a fifth answer of its own, and a reader took
whichever answer was nearest. This walk refuses, in production code, a read
of ``__file__`` outside the two owners, the simulated board package's own
folder and the entry's own ``src/`` line, and a folder made from ``sys.executable`` outside ``launch_root``
and the support report's listing of the interpreter's own folder.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, iter_package_modules, parse_module, tracked_files

# (file, the def that reads it); '<module>' is a module-level read.
_FILE_READERS = {
    ('src/_lumascope/lib/build_files.py', 'build_root'),
    ('src/_lumascope/modules/path_utils.py', 'launch_root'),
    # The package's own files (its MicroPython runtime and board images),
    # which ship inside it, not the build's.
    ('src/_lumascope/drivers/sim_wire/backend.py', '<module>'),
    # The entry puts its own checkout's src/ first, before any import could
    # reach an owner, so it reads its own path.
    ('lumaviewpro.py', '<module>'),
}
_EXECUTABLE_FOLDER_READERS = {
    ('src/_lumascope/modules/path_utils.py', 'launch_root'),
    # The C runtimes beside the interpreter, which is the question asked.
    ('src/_lumascope/modules/tech_support_report.py', 'TechSupportReport._step_runtime_census'),
}
# A call that turns its argument into a folder or a path.
_PATH_CALLS = Words({'Path', 'abspath', 'dirname', 'realpath', 'split'})


def _production_modules():
    yield from iter_package_modules(
        (
            'src/_lumascope/modules',
            'ui',
            'src/_lumascope/drivers',
            'src/_lumascope/lib',
            'src/_lumascope/rest',
        )
    )
    for rel_path in tracked_files(pattern='*.py'):
        yield rel_path, parse_module(rel_path)


def _sys_names(tree: ast.Module) -> set[str]:
    """The names the module binds to ``sys``, aliases included."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name for alias in node.names if alias.name == 'sys')
    return names


def _is_executable(node: ast.AST, sys_names: set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'executable'
        and isinstance(node.value, ast.Name)
        and node.value.id in sys_names
    )


def _calls_a_path(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
    return name in _PATH_CALLS


def _reads(tree: ast.Module):
    """Yield (def qualname, lineno, '__file__' | 'executable folder') for each read."""
    sys_names = _sys_names(tree)

    def visit(node, qualname):
        for child in ast.iter_child_nodes(node):
            inner = qualname
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                inner = child.name if qualname == '<module>' else f'{qualname}.{child.name}'
            if isinstance(child, ast.Name) and child.id == '__file__':
                yield qualname, child.lineno, '__file__'
            elif _calls_a_path(child) and any(_is_executable(a, sys_names) for a in child.args):
                yield qualname, child.lineno, 'executable folder'
            yield from visit(child, inner)

    yield from visit(tree, '<module>')


def _allowed(rel_path: str, qualname: str, kind: str) -> bool:
    readers = _FILE_READERS if kind == '__file__' else _EXECUTABLE_FOLDER_READERS
    return (rel_path, qualname) in readers


def test_the_build_and_launch_folders_are_computed_only_by_their_owners():
    found = [
        f'{rel_path}:{lineno} ({qualname}) reads the {kind}'
        for rel_path, tree in _production_modules()
        for qualname, lineno, kind in _reads(tree)
        if not _allowed(rel_path, qualname, kind)
    ]
    assert found == [], (
        "the build's files are _lumascope.lib.build_files.build_root()'s, the launch folder "
        f"_lumascope.modules.path_utils.launch_root()'s: {found}"
    )


def test_the_guard_sees_every_allowed_read():
    # Each exemption must still name a real read, or a rename would leave the
    # guard exempting a site that no longer exists.
    trees = dict(_production_modules())
    seen = {
        (rel_path, qualname, kind)
        for rel_path, tree in trees.items()
        for qualname, _lineno, kind in _reads(tree)
    }
    for rel_path, qualname in _FILE_READERS:
        assert (rel_path, qualname, '__file__') in seen
    for rel_path, qualname in _EXECUTABLE_FOLDER_READERS:
        assert (rel_path, qualname, 'executable folder') in seen


def test_the_guard_refuses_a_planted_read():
    planted = ast.parse(
        'import os, pathlib\n'
        'import sys as _sys\n'
        'ROOT = pathlib.Path(__file__).parent\n'
        'class Finder:\n'
        '    def where(self):\n'
        '        return os.path.dirname(_sys.executable)\n'
        'print(_sys.executable)\n'
    )
    assert sorted((q, kind) for q, _lineno, kind in _reads(planted)) == [
        ('<module>', '__file__'),
        ('Finder.where', 'executable folder'),
    ]
