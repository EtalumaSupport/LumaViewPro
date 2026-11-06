# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the published docs teach a reader to import from LumaViewPro is ``lumascope``.

``lumascope`` is the API's one public module; ``_lumascope`` is internal and
can change in any release, and ``tests``, ``ui`` and ``tools`` are not
installed for a reader at all. Every example script under
``docs/api_examples/`` and every Python fence (labelled ``python`` or
unlabelled) of the guide, the plugin tutorial and the READMEs is read by
its AST, and an import whose first segment names a module or package of
this checkout, other than ``lumascope``, is reported with its file and
line. The checkout's names are computed from the files git tracks, so a new
top-level package is covered the day it is added; the standard library and
a document's own package (the tutorial's plugin) are not the checkout's.
"""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

from tests.ast_seams import REPO_ROOT, tracked_files
from tests.guards.test_api_doc_guard import fence_info, guide_files

_PUBLIC = 'lumascope'
# The guide is the hub and the files its map names, read as the guide guard reads them.
_DOCS = (
    'README.md',
    *guide_files(),
    'docs/plugin-tutorial.md',
    'docs/api_examples/README.md',
)
_PYTHON_FENCES = ('', 'python')


def checkout_heads() -> frozenset[str]:
    """Every name a top-level import of this checkout's code begins with.

    The first folder or module of each tracked ``.py``, and under ``src/``
    the second: ``src/`` is put on the path, not imported.
    """
    heads = set()
    for rel in tracked_files():
        parts = PurePosixPath(rel).parts
        if parts[0] == 'src':
            parts = parts[1:]
        if len(parts) == 1:
            heads.add(PurePosixPath(parts[0]).stem)
        elif parts:
            heads.add(parts[0])
    return frozenset(heads)


def python_fences(text: str):
    """``(first_lineno, source)`` for each Python fence of a markdown text."""
    fence, lines, start = None, [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            if fence is None:
                fence, lines, start = fence_info(line)[0], [], lineno + 1
            else:
                if fence in _PYTHON_FENCES:
                    yield start, '\n'.join(lines)
                fence = None
            continue
        if fence is not None:
            lines.append(line)


def private_imports(label: str, source: str, start: int, heads: frozenset[str]) -> list[str]:
    """A line per import in ``source`` from the checkout that is not from ``lumascope``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module]
        else:
            continue
        for module in modules:
            head = module.split('.')[0]
            if head != _PUBLIC and head in heads:
                found.append(f'  {label}:{start + node.lineno - 1}  {module}')
    return found


def taught_sources():
    """``(label, first_lineno, source)`` for every example file and every doc fence."""
    for rel in tracked_files(('docs/api_examples',), '*.py'):
        yield rel, 1, (REPO_ROOT / rel).read_text(encoding='utf-8')
    for rel in _DOCS:
        for start, source in python_fences((REPO_ROOT / rel).read_text(encoding='utf-8')):
            yield rel, start, source


def test_the_docs_import_only_lumascope():
    heads = checkout_heads()
    failures, unparsed, read = [], [], set()
    for label, start, source in taught_sources():
        try:
            failures += private_imports(label, source, start, heads)
        except SyntaxError as exc:
            unparsed.append(f'  {label}:{start}  {exc.msg}')
        read.add(label)
    assert not unparsed, (
        'A Python fence in a published doc does not parse, so its imports go unread; '
        'label a fence that is not Python (```text).\n' + '\n'.join(unparsed)
    )
    assert not failures, (
        'A published doc or example imports from LumaViewPro other than through '
        '`lumascope`, the one public module. Import the name from `lumascope`.\n'
        + '\n'.join(failures)
    )
    assert {*guide_files(), 'docs/plugin-tutorial.md'} <= read, (
        f'the walk read no Python fence of a doc it must read: it read {sorted(read)}'
    )


def test_a_planted_private_import_is_reported_and_the_public_one_is_not():
    heads = checkout_heads()
    assert {'_lumascope', 'lumascope', 'tests', 'ui', 'tools'} <= heads, sorted(heads)
    planted = (
        'from __future__ import annotations\n'
        'import os\n'
        'from lumascope import ScopeSession\n'
        'from _lumascope.modules.scope_session import ScopeSession\n'
        'import tests.plugin_test_harness\n'
        'import hello_postproc\n'
    )
    assert private_imports('planted', planted, 10, heads) == [
        '  planted:13  _lumascope.modules.scope_session',
        '  planted:14  tests.plugin_test_harness',
    ]
