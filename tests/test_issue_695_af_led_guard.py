# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for #695: a live LED apply must not turn off an AF-owned channel.

Bug
---
Clicking the AF button defocuses the exposure field, whose apply chain
(exp_text -> apply_settings -> update_led_state -> set_led_state) issued a
led_off on the channel autofocus was using -- about 50 ms after autofocus
turned it on. Autofocus then scanned dark frames yet reported success (the
focus score collapsed). The ImagingAPI is_focusing flag existed and was
mirrored by the AF runner, but the live LED-apply path never consulted it.

Fix
---
LayerControl.update_led_state early-returns while scope.imaging.is_focusing
is True, so a live UI apply cannot turn off the channel autofocus is using.

Test approach
-------------
AST source scan -- behavioral exec of update_led_state needs a live Kivy
LayerControl widget + scope + camera_executor (out of scope here, matching
test_autofocus_is_focusing_wired.py). The structural lock catches a
re-removal of the guard.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(LAYER_CONTROL_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'LayerControl':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f'LayerControl.{name} not found')


def test_update_led_state_early_returns_while_focusing():
    """update_led_state must early-return when scope.imaging.is_focusing."""
    method = _method('update_led_state')
    guarded = False
    for node in ast.walk(method):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            body_returns = any(isinstance(stmt, ast.Return) for stmt in node.body)
            if 'is_focusing' in test_src and body_returns:
                guarded = True
                break
    assert guarded, (
        'update_led_state must early-return when scope.imaging.is_focusing is '
        'True, so a live UI apply cannot turn off the channel autofocus is '
        'using mid-scan (#695)'
    )
