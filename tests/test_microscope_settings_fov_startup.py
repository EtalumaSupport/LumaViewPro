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
load_settings shows the turret and objective through
VerticalControl.show_turret_state, the one display of the API's answer,
and that refreshes the FOV readout through
MicroscopeSettings.refresh_fov_labels -- the same refresher frame-size
changes use, which shows the scope's field of view
(``RuntimeState.get_field_of_view``). Its display is tested by behaviour in
test_microscope_settings.py.

Test approach
-------------
Source-level structural lock via AST on the chain: load_settings ->
show_turret_state -> refresh_fov_labels -> the scope's get_field_of_view.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent


def _method_node(rel: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse((REPO / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {rel}')


def _calls(method: ast.FunctionDef) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


class TestLoadSettingsFovStartup:
    def test_load_settings_shows_the_turret_state(self):
        method = _method_node('ui/microscope_settings.py', 'MicroscopeSettings', 'load_settings')
        assert 'show_turret_state' in _calls(method)

    def test_the_turret_display_refreshes_the_fov(self):
        method = _method_node('ui/vertical_control.py', 'VerticalControl', 'show_turret_state')
        assert 'refresh_fov_labels' in _calls(method)

    def test_the_refresher_shows_the_scopes_field_of_view(self):
        method = _method_node(
            'ui/microscope_settings.py', 'MicroscopeSettings', 'refresh_fov_labels'
        )
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'get_field_of_view'
            ):
                # The scope decides: the GUI hands it nothing to compute from.
                assert (node.args, node.keywords) == ([], [])
                return
        raise AssertionError('refresh_fov_labels shows no field of view')
