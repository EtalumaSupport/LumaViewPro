# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: show-step-locations is a persisted system setting.

The "Show step locations" toggle used to be transient -- it wrote no
setting and called ctx.stage.show_protocol_steps() directly, so it reset
to off every launch. It is now promoted to settings['show_step_locations']
and the toggle itself lives in the Advanced Settings modal, a row of
ui/setting_controls.py (its write is pinned in
test_a_setting_control_is_one_row.py); load_settings re-applies the saved
view to the stage at startup.

Guards:
  - the key ships in the tracked settings.json schema default, so the
    settings_init default-merge backfills every user's current.json;
  - the stage's view follows the stored setting, whoever wrote it;
  - load_settings re-applies the saved view to the stage at startup.
"""

from __future__ import annotations

import ast
import json
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
MICROSCOPE_SETTINGS_SRC = REPO / 'ui' / 'microscope_settings.py'


def _method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(src, child)
    raise AssertionError(f'{class_name}.{method_name} not found in {path}')


def test_show_step_locations_in_settings_schema():
    """The tracked settings.json default carries the key for the merge to backfill."""
    data = json.loads((REPO / 'data' / 'settings.json').read_text())
    assert 'show_step_locations' in data, (
        'data/settings.json must define show_step_locations so the '
        'default-merge backfills it and the persisted read does not KeyError'
    )


def test_the_stage_view_follows_the_stored_setting():
    """A change to the setting, the toggle's or a REST client's, redraws the stage's step crosses."""
    from types import SimpleNamespace

    from ui.listener_bridge import _routed

    shown = []
    # a stand-in by design: the stage is a Kivy widget; the subject is the follower's route
    ctx = SimpleNamespace(
        settings={'show_step_locations': True},
        stage=SimpleNamespace(show_protocol_steps=lambda enable: shown.append(enable)),
        motion_settings=SimpleNamespace(
            ids={'microscope_settings_id': SimpleNamespace(advanced_settings_shown=lambda: None)}
        ),
    )
    for redraw, _channel in _routed('show_step_locations'):
        redraw(ctx, 'show_step_locations')
    assert shown == [True]


def test_load_settings_restores_saved_state():
    """Startup restores the saved toggle and re-applies it to the stage view."""
    source = _method_source(MICROSCOPE_SETTINGS_SRC, 'MicroscopeSettings', 'load_settings')
    assert "settings['show_step_locations']" in source, (
        'load_settings must restore show_step_locations from the persisted setting'
    )
    assert 'show_protocol_steps' in source, (
        'load_settings must re-apply the saved show-step-locations view at startup'
    )
