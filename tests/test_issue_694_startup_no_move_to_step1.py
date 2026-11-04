# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for #694: startup must not drive the stage to protocol step 1.

Bug
---
On startup, after homing and turret init, complete_initialization called
go_to_step when a protocol was loaded, driving the stage (X/Y/Z + turret) to
protocol step 1 -- surprising and unwanted motion.

Fix
---
complete_initialization now stays where homing left the stage and shows the
Session's shown channel (BF in a fresh session) at its saved settings, through
that layer's apply_settings -- identical to the no-protocol path, with no
stage motion either way. A loaded protocol's steps
remain available (the table is rendered by load_protocol); the user navigates
to them explicitly.

Test approach
-------------
AST source scan -- complete_initialization is a nested on_start helper whose
behavioral exec needs a full Kivy app + scope. The structural lock catches a
re-introduction of the startup move.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LUMAVIEWPRO_SRC = REPO / 'lumaviewpro.py'


def _complete_initialization() -> ast.FunctionDef:
    tree = ast.parse(LUMAVIEWPRO_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'complete_initialization':
            return node
    raise AssertionError('complete_initialization not found in lumaviewpro.py')


def test_startup_does_not_move_to_step():
    """complete_initialization must not call go_to_step at startup."""
    src = ast.unparse(_complete_initialization())
    assert 'go_to_step' not in src, (
        'complete_initialization must not call go_to_step at startup -- the '
        'stage stays where homing left it, it does not move to protocol step 1'
    )


def test_startup_shows_the_shown_channel():
    """Startup applies the shown channel's settings (no move)."""
    src = ast.unparse(_complete_initialization())
    assert 'shown_channel' in src and 'apply_settings()' in src, (
        "startup must show the Session's shown channel through its layer's "
        'apply_settings so the camera shows a sensible default'
    )
