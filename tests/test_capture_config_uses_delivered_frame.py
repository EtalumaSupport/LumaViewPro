# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: a capture config carries the frame size the camera
DELIVERED, never the size a user has typed but not applied.

The frame width/height fields are an editor, not a store. They commit
only through MicroscopeSettings.frame_size(), which pushes the size to
the camera and writes settings['frame'] -- and the field text -- from
the geometry the camera actually delivered. Two paths leave the typed
text standing while settings holds something else: frame_size() returns
early when no camera is connected, and the push landing never runs when
the scope slot is empty or the camera refuses. A config assembled from
the field text in that window describes a geometry no camera is at.

settings['frame'] is the authority for every consumer except the apply
handler itself, which legitimately wants the typed request.
"""

import ast
import pathlib
from unittest.mock import MagicMock

REPO = pathlib.Path(__file__).resolve().parent.parent


def _patch_ctx(monkeypatch, *, settings: dict):
    """An app context for the GUI config lane."""
    ctx = MagicMock()
    ctx.settings = settings
    ctx.engineering_mode = False

    protocol_settings = MagicMock()
    protocol_settings.ids = {
        'tiling_size_spinner': MagicMock(text='1x1'),
        'acquire_zstack_id': MagicMock(active=False),
    }
    protocol_settings.get_tiling_overlap_percent.return_value = 0.0
    ctx.motion_settings.ids = {'protocol_settings_id': protocol_settings}
    ctx.session.get_current_objective_info.return_value = ('4x', {'focal_length': 45.0})

    import modules.app_context as app_context

    monkeypatch.setattr(app_context, 'ctx', ctx)
    return ctx


def _settings():
    import json

    settings = json.loads((REPO / 'data' / 'settings.json').read_text())
    settings['frame'] = {'width': 1900, 'height': 1900}
    return settings


def test_config_carries_delivered_frame_not_typed_text(monkeypatch):
    """1024 typed and never applied; the camera is at 1900."""
    _patch_ctx(monkeypatch, settings=_settings())

    from modules.config_ui_getters import get_sequenced_capture_config_from_ui

    config = get_sequenced_capture_config_from_ui()

    assert config['frame_dimensions'] == {'width': 1900, 'height': 1900}


def test_no_modules_file_reads_the_frame_fields():
    """The frame width/height widgets are read only by the widget that owns
    them.

    A modules/ file reaching into the Kivy tree for these fields is the
    defect this pair of tests exists for: it puts the typed request where
    callers who wanted the delivered geometry could pick it up. Reading
    them from ui/microscope_settings.py is legal -- that is the class the
    fields live on, reading its own tree.
    """
    offenders = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / 'modules').glob('*.py')
        if 'frame_width_id' in path.read_text() or 'frame_height_id' in path.read_text()
    }

    assert offenders == set(), offenders


def test_the_apply_handler_reads_the_typed_fields_off_its_own_tree():
    """frame_size() applies a typed edit, so it -- and only it -- wants the
    typed value, and it takes it from self.ids rather than reaching for the
    app context."""
    from tests.ast_seams import assert_def, find_def

    assert_def(
        'ui/microscope_settings.py',
        '_typed_frame_dimensions',
        class_name='MicroscopeSettings',
        msg='the typed-field read belongs on the widget that owns the fields',
    )

    reader = find_def('ui/microscope_settings.py', '_typed_frame_dimensions', 'MicroscopeSettings')
    reached = [
        node.id for node in ast.walk(reader) if isinstance(node, ast.Name) and node.id == '_app_ctx'
    ]
    assert not reached, 'the typed-field read must not reach for the app context'
