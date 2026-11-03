# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every host serves REST through ``rest.server.build_server``, the one place a uvicorn server is configured.

The GUI, ``python -m rest`` and a program serving its own session each get
the judge of ``rest_api``, the caller-thread bind and the one configuration
only by serving through it. No production module but ``rest/server.py``
calls ``uvicorn.Config`` or ``uvicorn.run``, and no published document tells
a program to: the guide's serving example calls the builder.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT as _REPO

_THE_BUILDER = 'rest/server.py'
_SKIPPED = {'tests', 'build', '.git', '.venv', 'venv', '__pycache__', 'node_modules'}
_SERVING = ('uvicorn.Config(', 'uvicorn.run(')


def _production_files():
    for path in sorted(_REPO.rglob('*.py')):
        rel = path.relative_to(_REPO)
        if not _SKIPPED.intersection(rel.parts):
            yield rel.as_posix(), path


def _serves_uvicorn(tree: ast.AST) -> list[int]:
    lines = []
    from_uvicorn = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'uvicorn':
            from_uvicorn |= {a.asname or a.name for a in node.names if a.name in ('Config', 'run')}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = ast.unparse(node.func)
            if called in ('uvicorn.Config', 'uvicorn.run') or called in from_uvicorn:
                lines.append(node.lineno)
    return lines


def test_no_production_module_but_the_builder_configures_a_uvicorn_server():
    found = {}
    for rel, path in _production_files():
        if rel == _THE_BUILDER:
            continue
        lines = _serves_uvicorn(ast.parse(path.read_text(encoding='utf-8')))
        if lines:
            found[rel] = lines
    assert found == {}
    # Not vacuous: the builder itself is found.
    assert _serves_uvicorn(ast.parse((_REPO / _THE_BUILDER).read_text(encoding='utf-8')))


def test_no_published_document_serves_without_the_builder():
    docs = [*sorted((_REPO / 'docs').rglob('*.md')), _REPO / 'README.md']
    found = {
        doc.relative_to(_REPO).as_posix(): [s for s in _SERVING if s in doc.read_text('utf-8')]
        for doc in docs
    }
    assert {doc: said for doc, said in found.items() if said} == {}
    assert 'build_server(session' in (_REPO / 'docs' / 'lumascope-skills.md').read_text('utf-8')
