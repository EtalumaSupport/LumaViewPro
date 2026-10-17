# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every grab start and stop goes through the base Camera's two members.

``Camera.start_grabbing`` and ``Camera.stop_grabbing`` record whether the
grab is wanted, the fact the stream check reads to tell a stop someone
asked for from a grab that stopped on its own. A driver that defined
either member itself would start or stop a grab the record never heard
of; each driver's own body is the hook the base calls.
"""

import ast

from tests.ast_seams import REPO_ROOT

_MEMBERS = {'start_grabbing', 'stop_grabbing'}


def test_no_driver_defines_the_grab_start_or_stop_itself():
    definers = []
    for path in sorted((REPO_ROOT / 'drivers').rglob('*.py')):
        if path.name == 'camera.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in _MEMBERS:
                        definers.append(
                            f'{path.relative_to(REPO_ROOT).as_posix()} {node.name}.{item.name}'
                        )
    assert definers == [], f'define _start_grabbing / _stop_grabbing instead: {definers}'
