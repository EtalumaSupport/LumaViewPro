# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for blank-step-name preservation on rename.

Bug
---
Repro: blank labware -> add a custom step -> click into its (blanked)
name field -> click Change (or just blur the field). The auto-assigned
custom#### name is wiped to '' so the step falls back to the default
name; two added steps then collide on the same default and the saved
protocol TSV carries no names for them.

Root cause
----------
An auto-named custom step blanks step_name_input.text so the default
name shows as a hint placeholder, not editable text. Two paths persist
that field into the protocol:
  - modify_step_ex      (the Change button)
  - step_name_validation (on_text_validate / on_focus blur)
modify_step_ex already preserved the existing name when the field was
blank; step_name_validation did not -- it wrote the empty string
straight to Protocol.modify_name, wiping the name on blur, before the
Change path even ran.

Fix
---
Both paths route the field text through
common_utils.resolve_step_rename, which returns None for a blank field
("no rename intended"). On None, step_name_validation skips the write
and modify_step_ex substitutes the existing name. A non-empty entry is
the user's rename and passes through sanitized.

Test approach
-------------
resolve_step_rename is a pure support function (raw text + a sanitize
callable; no Kivy / ctx), so its behavior is exercised directly at
runtime. The two call sites stay Kivy-bound, so AST locks assert both
route through the helper and that modify_step_ex's None-guard still
precedes the Protocol.modify_step call.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    """Return the AST node for ClassName.method_name."""
    source = PROTOCOL_SETTINGS_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in source')


class TestResolveStepRename:
    """Runtime behavior of the shared blank-field rename policy."""

    @staticmethod
    def _resolve(raw: str):
        from modules.common_utils import resolve_step_rename

        # A sanitize stub that strips whitespace -- enough to drive the
        # policy (blank-after-sanitize -> None). The real
        # sanitize_step_name also drops invalid path chars; that is not
        # what this policy gates on.
        return resolve_step_rename(raw, lambda s: s.strip())

    def test_empty_field_means_no_rename(self):
        assert self._resolve('') is None

    def test_whitespace_only_field_means_no_rename(self):
        # Sanitizes down to empty -> still "no rename intended".
        assert self._resolve('   ') is None

    def test_real_name_passes_through_sanitized(self):
        assert self._resolve('  My Step  ') == 'My Step'


class TestRenamePathsRouteThroughHelper:
    """Source-level lock: both persist paths use resolve_step_rename so the
    blank-field policy cannot diverge between them."""

    def test_step_name_validation_uses_helper_with_none_guard(self):
        method = _method_node('ProtocolSettings', 'step_name_validation')
        body_src = ast.unparse(method)
        assert 'resolve_step_rename' in body_src, (
            'step_name_validation must route the field text through '
            'resolve_step_rename so a blank field does not clobber the '
            'auto-assigned step name.'
        )
        assert 'is None' in body_src, (
            'step_name_validation must guard the resolve_step_rename result '
            'against None (blank field = no rename) before calling modify_name.'
        )

    def test_modify_step_ex_uses_helper_with_none_guard(self):
        method = _method_node('ProtocolSettings', 'modify_step_ex')
        body_src = ast.unparse(method)
        assert 'resolve_step_rename' in body_src, (
            'modify_step_ex must route the field text through '
            'resolve_step_rename so its blank-field handling matches '
            'step_name_validation.'
        )

    def test_modify_step_ex_none_guard_runs_before_modify_step(self):
        """The None-guard must run before Protocol.modify_step(...), else
        modify_step still receives an unresolved name."""
        method = _method_node('ProtocolSettings', 'modify_step_ex')
        guard_lineno = None
        modify_call_lineno = None
        for node in ast.walk(method):
            if (
                guard_lineno is None
                and isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and any(isinstance(op, ast.Is) for op in node.test.ops)
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value is None
            ):
                guard_lineno = node.lineno
            if (
                modify_call_lineno is None
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'modify_step'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == '_protocol'
            ):
                modify_call_lineno = node.lineno
        assert guard_lineno is not None, (
            'modify_step_ex must contain an `if step_name is None:` guard '
            'that preserves the existing protocol step name.'
        )
        assert modify_call_lineno is not None, (
            'self._protocol.modify_step(...) call not found in modify_step_ex'
        )
        assert guard_lineno < modify_call_lineno, (
            f'None-guard at line {guard_lineno} must run before '
            f'self._protocol.modify_step at line {modify_call_lineno}.'
        )
