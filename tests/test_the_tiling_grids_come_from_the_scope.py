# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The tiling grids a caller can choose come from the scope's data folder.

The protocol panel used to build its own tiling config in its constructor,
which runs before the session exists, so it fell back to the installation's
folder rather than the one the scope was started on. The scope now answers
(`ProtocolsAPI.tiling_config()`), an integrator can list the labels
`create_protocol` accepts, and nothing in `ui/` reads tiling.json itself.
"""

from __future__ import annotations

import ast
import json
import shutil

from tests.ast_seams import REPO_ROOT, iter_package_modules
from tests.scope_fakes import build_scope


def test_the_scope_answers_from_the_folder_it_was_started_on(tmp_path):
    shutil.copytree(REPO_ROOT / 'data', tmp_path / 'data')
    tiling_file = tmp_path / 'data' / 'tiling.json'
    tiling = json.loads(tiling_file.read_text())
    tiling['data']['9x9'] = {'m': 9, 'n': 9}
    tiling['metadata']['default'] = '2x2'
    tiling_file.write_text(json.dumps(tiling))

    scope = build_scope(simulate=True, register_atexit=False, source_path=tmp_path)
    config = scope.protocols.tiling_config()

    assert '9x9' in config.available_configs()
    assert config.default_config() == '2x2'


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
        'the GUI asks the scope for its tiling grids (scope.protocols.tiling_config()); '
        f'it never builds a TilingConfig or names tiling.json: {found}'
    )
