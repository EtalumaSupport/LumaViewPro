# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The tiling grids a caller can choose come from the scope's data folder.

The protocol panel used to build its own tiling config in its constructor,
which runs before the session exists, so it fell back to the installation's
folder rather than the one the scope was started on. The session now answers
(`ScopeSession.tiling_config()`), from the grids its scope read, so an
integrator can list the labels `new_protocol` accepts, and nothing in `ui/`
reads tiling.json itself.
"""

from __future__ import annotations

import ast
import json
import shutil

from modules.scope_session import ScopeSession
from tests.ast_seams import REPO_ROOT, iter_package_modules
from tests.settings_fixtures import complete_settings


def test_the_session_answers_from_the_folder_its_scope_was_started_on(tmp_path):
    shutil.copytree(REPO_ROOT / 'data', tmp_path / 'data')
    tiling_file = tmp_path / 'data' / 'tiling.json'
    tiling = json.loads(tiling_file.read_text())
    tiling['data']['9x9'] = {'m': 9, 'n': 9}
    tiling['metadata']['default'] = '2x2'
    tiling_file.write_text(json.dumps(tiling))

    session = ScopeSession.create(complete_settings(), source_path=str(tmp_path), simulate=True)
    try:
        config = session.tiling_config()
    finally:
        session.shutdown()

    assert '9x9' in config.available
    assert config.default == '2x2'


def _builds_tiling_config(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
    return name == 'TilingConfig'


def _names_tiling_file(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 'tiling.json'


def test_nothing_in_ui_reads_the_tiling_file_itself():
    found = [
        f'{rel_path}:{node.lineno}'
        for rel_path, tree in iter_package_modules(('ui',))
        for node in ast.walk(tree)
        if _builds_tiling_config(node) or _names_tiling_file(node)
    ]
    assert found == [], (
        'the GUI asks the session for its tiling grids (session.tiling_config()); '
        f'it never builds a TilingConfig or names tiling.json: {found}'
    )
