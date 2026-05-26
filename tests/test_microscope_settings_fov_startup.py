# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test for #658: FOV fields populate at startup in
MicroscopeSettings.load_settings.

Bug
---
load_settings populated frame_width / frame_height inputs and objective
magnification but never computed the derived FOV values, so
field_of_view_width_id and field_of_view_height_id stayed blank until
the user clicked Frame Size or selected an objective (both have their
own FOV-recalc handlers).

Fix
---
Replicate the FOV computation pattern from frame_size() and
select_objective() inside load_settings, immediately after the
objective is resolved. Sibling sites unchanged.

Test approach
-------------
Source-level structural lock via AST: load_settings's body must
- call common_utils.get_field_of_view(...) and
- write to self.ids['field_of_view_width_id'].text + ...['field_of_view_height_id'].text
A behavioral exec would have to stub Kivy ids, settings JSON,
objective_helper, common_utils, the lvp_lock context manager, and
several module-scope helpers; mocking surface overwhelms signal.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
MICROSCOPE_SETTINGS_SRC = REPO / 'ui' / 'microscope_settings.py'


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    source = MICROSCOPE_SETTINGS_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in source')


class TestLoadSettingsFovStartup:
    """Source-level lock that load_settings computes + writes FOV at startup."""

    def test_load_settings_calls_get_field_of_view(self):
        """load_settings must call common_utils.get_field_of_view so the
        FOV inputs have a value at startup -- not just on user
        interaction with the frame-size / objective handlers."""
        method = _method_node('MicroscopeSettings', 'load_settings')
        body_src = ast.unparse(method)
        assert 'get_field_of_view' in body_src, (
            'load_settings must call common_utils.get_field_of_view at '
            'startup. Without this the field_of_view_*_id inputs stay '
            'blank until the user clicks Frame Size or selects an '
            'objective. See class docstring.'
        )

    def test_load_settings_writes_both_fov_input_ids(self):
        """load_settings must assign to both field_of_view_width_id.text
        and field_of_view_height_id.text. A one-axis write would only
        partially fix #658."""
        method = _method_node('MicroscopeSettings', 'load_settings')
        body_src = ast.unparse(method)
        for ids_key in ('field_of_view_width_id', 'field_of_view_height_id'):
            assert ids_key in body_src, (
                f"load_settings must assign to self.ids['{ids_key}'].text "
                f'so both axes populate at startup. Found body without '
                f'this id reference.'
            )

    def test_get_field_of_view_call_uses_binning_and_frame_size(self):
        """The FOV call must receive frame_size and binning_size so the
        startup value matches what frame_size() / select_objective()
        produce later. A bare get_field_of_view() with hardcoded
        defaults would diverge from the user-interaction handlers."""
        method = _method_node('MicroscopeSettings', 'load_settings')
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'get_field_of_view'):
                continue
            kwarg_names = {kw.arg for kw in node.keywords if kw.arg}
            required = {'frame_size', 'binning_size'}
            assert required.issubset(kwarg_names), (
                f'get_field_of_view call in load_settings must pass '
                f'frame_size= and binning_size= keyword args; got {sorted(kwarg_names)}. '
                'These match the sibling call sites in frame_size() and '
                'select_objective() so all three handlers compute the same FOV.'
            )
            return
        raise AssertionError(
            'No get_field_of_view call found in load_settings (the previous '
            'test should have caught this).'
        )
