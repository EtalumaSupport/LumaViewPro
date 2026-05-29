# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression locks for the deeper code-review hardening fixes.

Companion to test_review_prepass_hardening.py: pins the findings from the
multi-agent review of the mono-native + overlap bundle. Behavioral where
the module imports under the harness; source/AST locks where the carrier
is Kivy-bound and cannot be instantiated (window keyboard handler,
layer_control focus refresh).
"""

from __future__ import annotations

import ast
import inspect
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent


def _method_src(rel_path: str, class_name: str | None, func_name: str) -> str:
    source = (REPO / rel_path).read_text()
    tree = ast.parse(source)
    nodes = ast.walk(tree)
    if class_name is not None:
        cls = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        nodes = ast.walk(cls)
    fn = next(
        n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == func_name
    )
    return ast.unparse(fn)


class TestAltF4Keycode:
    """The Alt-F4 GUI-log branch must match Kivy's F4 keycode (285), not
    F12 (293), or a real Alt-F4 never logs and a stray Alt-F12 mislogs."""

    def test_alt_f4_branch_uses_keycode_285(self):
        src = _method_src('lumaviewpro.py', 'LumaViewProApp', '_on_window_keyboard')
        assert 'Alt-F4' in src
        assert 'key == 285' in src, 'Alt-F4 branch must test Kivy F4 keycode 285.'
        assert 'key == 293' not in src, '293 is F12, not F4.'


class TestCompositeNoDeadThresholdParam:
    """generate_composite_from_paths must not re-grow brightness_thresholds:
    it was accepted + documented as 'forwarded to build_composite' but never
    threaded -- the builder recomputes thresholds from settings, so the param
    silently did nothing."""

    def test_signature_has_no_brightness_thresholds(self):
        from modules.composite_generation import CompositeGeneration

        params = inspect.signature(
            CompositeGeneration.generate_composite_from_paths
        ).parameters
        assert 'brightness_thresholds' not in params

