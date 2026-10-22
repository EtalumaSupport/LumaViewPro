# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI asks for a Z move and leaves its backlash approach to the API.

A move of Z to a place approaches from below and a move by a distance does
not; ``MotionAPI.start_move_z`` / ``start_move_z_by`` decide it.
``overshoot_enabled`` is a bench script's override, and a GUI that passed
it would be deciding how a move reaches the board.
"""

import ast

from tests.ast_seams import iter_package_modules


def test_no_gui_call_chooses_the_approach():
    offenders = [
        f'{rel}:{node.lineno}'
        for rel, tree in iter_package_modules(['ui'])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and any(kw.arg == 'overshoot_enabled' for kw in node.keywords)
    ]

    assert offenders == [], f'ui/ passes overshoot_enabled: {offenders}'
