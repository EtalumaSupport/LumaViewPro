# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``lumascope`` is the API's one public module: generated, whole, and loaded by no first-party code.

A user writes ``from lumascope import X`` for every published name
(``src/lumascope/__init__.py``). This guard holds four things:

1. **Current.** The checked-in module is the generator's render
   (``tests.public_face``) of the marks, byte for byte: a name published or
   withdrawn without regenerating fails, naming the command. Run with
   ``LVP_WRITE_PUBLIC_FACE=1``, the rule writes the render instead.
2. **Whole.** Its ``__all__`` is the published set: every name the
   reference has a section for, the root ``LumascopeError``, and every
   declared constant; each is the object its defining module holds, and the
   module holds no other public name.
3. **One root.** Every published exception is a ``LumascopeError``, so
   ``except LumascopeError`` catches whatever the API raises on purpose.
4. **No cycle, and first-party code pays nothing.** No ``_lumascope``
   module imports ``lumascope``, and importing an internal module does not
   load it.

And ``python -m lumascope.rest`` names itself so in its usage, and is the one
command: ``python -m _lumascope.rest`` runs nothing.
"""

from __future__ import annotations

import ast
import difflib
import importlib
import os
import pathlib
import subprocess
import sys
import types

import pytest

from _lumascope.lib.lumascope_error import LumascopeError
from _lumascope.modules import wire_encoding
from _lumascope.modules.api_surface import api
from tests.api_reference import declared, published
from tests.ast_seams import REPO_ROOT, MemberNames, tracked_files
from tests.marked_set import Marked, marked_set, project_modules
from tests.public_face import FACE, REGENERATE, WRITE, declared_constants, face_text, homes

# One of each kind the face holds, whose absence would mean a source was
# lost: a class section, a closed set, an outcome, the root, a module's
# function, and the declared constant.
NOT_VACUOUS = MemberNames(
    'src/_lumascope',
    {
        'ScopeSession',
        'RunStatus',
        'PositionOutOfRangeError',
        'LumascopeError',
        'build_app',
        'PLUGIN_API_LEVEL',
    },
)


@pytest.fixture(scope='module')
def universe() -> dict[str, type]:
    return wire_encoding.project_classes()


@pytest.fixture(scope='module')
def marked(universe: dict[str, type]) -> dict[str, Marked]:
    return marked_set(universe)


def _homes(marked: dict[str, Marked], universe: dict[str, type]) -> dict[str, str]:
    """Each name the face should publish, with the module it is imported from."""
    sections = declared(marked, universe, wire_encoding.project_aliases())
    return homes(published(sections), declared_constants(project_modules()))


@pytest.fixture(scope='module')
def expected(marked: dict[str, Marked], universe: dict[str, type]) -> dict[str, str]:
    return _homes(marked, universe)


@pytest.fixture(scope='module')
def face() -> types.ModuleType:
    return importlib.import_module('lumascope')


def test_the_face_is_current(expected: dict[str, str]) -> None:
    rendered = face_text(expected)
    if os.environ.get(WRITE):
        FACE.write_text(rendered, encoding='utf-8', newline='\n')
        return
    committed = FACE.read_text(encoding='utf-8') if FACE.exists() else ''
    changed = list(
        difflib.unified_diff(
            committed.splitlines(), rendered.splitlines(), 'committed', 'rendered', lineterm='', n=1
        )
    )
    assert not changed, (
        f'{FACE.relative_to(REPO_ROOT)} is not the render of the marks; regenerate it:\n'
        f'  {REGENERATE}\n' + '\n'.join(changed[:60])
    )


class PlantedRecord:
    """A class marked after the face was generated."""

    @api
    def planted(self) -> None:
        """Nothing.

        Raises:
            Nothing.
        """


def test_a_class_marked_after_the_face_was_generated_is_not_current(
    universe: dict[str, type],
) -> None:
    planted = {**universe, PlantedRecord.__name__: PlantedRecord}
    rendered = face_text(_homes(marked_set(planted), planted))
    assert f'    {PlantedRecord.__name__},' in rendered
    assert rendered != FACE.read_text(encoding='utf-8')


def test_the_face_publishes_the_declared_set_and_nothing_else(
    face: types.ModuleType, expected: dict[str, str]
) -> None:
    assert expected.keys() >= NOT_VACUOUS, sorted(NOT_VACUOUS - expected.keys())
    assert sorted(face.__all__) == sorted(expected)
    wrong = [
        name
        for name, module in expected.items()
        if getattr(face, name) is not getattr(sys.modules[module], name)
    ]
    assert not wrong, f'names the face binds to another object than their module holds: {wrong}'
    extra = sorted(
        name
        for name, value in vars(face).items()
        if not name.startswith('_')
        and not isinstance(value, types.ModuleType)
        and name not in expected
    )
    assert not extra, f'public names the face holds outside __all__: {extra}'


def test_every_published_exception_is_a_lumascope_error(face: types.ModuleType) -> None:
    stray = sorted(
        name
        for name in face.__all__
        if isinstance(value := getattr(face, name), type)
        and issubclass(value, BaseException)
        and not issubclass(value, LumascopeError)
    )
    assert not stray, f'published exceptions outside LumascopeError: {stray}'


def imports_of_the_face(root: pathlib.Path, paths: list[str]) -> list[str]:
    """Each of *paths*, relative to *root*, that imports ``lumascope`` or a submodule of it, with its line."""
    found = []
    for path in paths:
        tree = ast.parse((root / path).read_text(encoding='utf-8'), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                named = [node.module or '']
            else:
                continue
            if any(n == 'lumascope' or n.startswith('lumascope.') for n in named):
                found.append(f'{path}:{node.lineno}')
    return found


def test_no_internal_module_imports_the_face() -> None:
    found = imports_of_the_face(REPO_ROOT, tracked_files(('src/_lumascope',)))
    assert not found, f'internal modules importing the public face (a load cycle): {found}'


def test_an_internal_import_of_the_face_is_found(tmp_path: pathlib.Path) -> None:
    (tmp_path / 'clean.py').write_text('import _lumascope.modules\n', encoding='utf-8')
    (tmp_path / 'planted.py').write_text('from lumascope import ScopeSession\n', encoding='utf-8')
    (tmp_path / 'deep.py').write_text('import lumascope.rest\n', encoding='utf-8')
    found = imports_of_the_face(tmp_path, ['clean.py', 'deep.py', 'planted.py'])
    assert found == ['deep.py:1', 'planted.py:1']


def test_an_internal_module_does_not_load_the_face() -> None:
    done = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys, _lumascope.modules.fx2_driver_step; print("lumascope" in sys.modules)',
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'False'


def test_the_rest_command_names_itself_in_its_usage() -> None:
    done = subprocess.run(
        [sys.executable, '-m', 'lumascope.rest', '--help'],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith('usage: python -m lumascope.rest ')


def test_the_internal_package_is_no_command() -> None:
    done = subprocess.run(
        [sys.executable, '-m', '_lumascope.rest', '--help'],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert done.returncode == 1, done.stdout
    assert "'_lumascope.rest' is a package and cannot be directly executed" in done.stderr, (
        done.stderr
    )
