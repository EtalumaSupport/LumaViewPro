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
and modify_step_ex passes label=None to Protocol.modify_step, which
keeps the step's existing Label and auto/user flag. A non-empty entry
is the user's rename and passes through sanitized.

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

    def test_modify_step_ex_resolved_rename_flows_to_label_kwarg(self):
        """The blank-field policy now lives in Protocol.modify_step itself:
        label=None means "keep the existing label and auto/user flag". The
        UI's job is to pass the resolve_step_rename result -- None included
        -- straight through as the label kwarg, so a blank field can never
        clobber the step's name."""
        method = _method_node('ProtocolSettings', 'modify_step_ex')
        resolved_var = None
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'resolve_step_rename'
                and isinstance(node.targets[0], ast.Name)
            ):
                resolved_var = node.targets[0].id
        assert resolved_var is not None, (
            'modify_step_ex must assign the resolve_step_rename result to a variable'
        )

        label_kwarg = None
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'modify_step'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == '_protocol'
            ):
                for kw in node.keywords:
                    if kw.arg == 'label':
                        label_kwarg = kw.value
        assert label_kwarg is not None, (
            'self._protocol.modify_step(...) must receive the rename via label='
        )
        assert isinstance(label_kwarg, ast.Name) and label_kwarg.id == resolved_var, (
            'the label kwarg must be the unmodified resolve_step_rename '
            'result (None = blank field = keep the existing label)'
        )
