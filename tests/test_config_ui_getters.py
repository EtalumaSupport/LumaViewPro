# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for UI-dependent config getters in modules/config_ui_getters.py.

Headless equivalents in modules/config_helpers.py are tested in
tests/test_headless_config.py. The UI variants must handle the same
failure modes (missing/invalid labware id, missing loader) without
crashing — issue #634 was caused by an inner-except missing-return
that returned implicit None and broke tuple-unpacking callers.
"""

from unittest.mock import MagicMock

import pytest


def _patch_ctx(monkeypatch, *, spinner_text: str, settings: dict, loader):
    """Build a MagicMock app context shaped like the real one."""
    ctx = MagicMock()
    spinner = MagicMock()
    spinner.text = spinner_text
    protocol_settings = MagicMock()
    protocol_settings.ids = {'labware_spinner': spinner}
    ctx.motion_settings.ids = {'protocol_settings_id': protocol_settings}
    ctx.settings = settings
    ctx.wellplate_loader = loader

    import modules.app_context as app_context
    monkeypatch.setattr(app_context, 'ctx', ctx)
    return ctx


class TestGetSelectedLabware:
    """UI variant — get_selected_labware() reads spinner with settings fallback."""

    def test_spinner_has_valid_labware(self, monkeypatch):
        loader = MagicMock()
        plate = MagicMock()
        loader.get_plate.return_value = plate
        _patch_ctx(monkeypatch,
                   spinner_text='96 well microplate',
                   settings={'protocol': {'labware': 'unused-fallback'}},
                   loader=loader)

        from modules.config_ui_getters import get_selected_labware
        labware_id, obj = get_selected_labware()
        assert labware_id == '96 well microplate'
        assert obj is plate

    def test_spinner_empty_falls_back_to_settings(self, monkeypatch):
        loader = MagicMock()
        plate = MagicMock()
        loader.get_plate.return_value = plate
        _patch_ctx(monkeypatch,
                   spinner_text='',
                   settings={'protocol': {'labware': '96 well microplate'}},
                   loader=loader)

        from modules.config_ui_getters import get_selected_labware
        labware_id, obj = get_selected_labware()
        assert labware_id == '96 well microplate'
        assert obj is plate

    def test_spinner_has_stale_default_returns_none(self, monkeypatch):
        # Issue #634 regression: KV file shipped `text: 'New'` as the spinner
        # default, so on first run before settings synced to UI the spinner
        # text was 'New' which doesn't exist in labware.json. The buggy
        # version returned implicit None instead of (None, None) — every
        # caller that did `labware_id, _ = get_selected_labware()` then hit
        # TypeError and crashed the app.
        loader = MagicMock()
        loader.get_plate.side_effect = KeyError('New')
        _patch_ctx(monkeypatch,
                   spinner_text='New',
                   settings={'protocol': {'labware': '96 well microplate'}},
                   loader=loader)

        from modules.config_ui_getters import get_selected_labware
        result = get_selected_labware()
        # Must be a 2-tuple even on failure — callers tuple-unpack.
        assert result == (None, None)

    def test_spinner_empty_and_settings_missing_returns_none(self, monkeypatch):
        loader = MagicMock()
        loader.get_plate.side_effect = KeyError('')
        _patch_ctx(monkeypatch,
                   spinner_text='',
                   settings={},
                   loader=loader)

        from modules.config_ui_getters import get_selected_labware
        result = get_selected_labware()
        assert result == (None, None)

    def test_loader_keyerror_returns_none(self, monkeypatch):
        loader = MagicMock()
        loader.get_plate.side_effect = KeyError('nonexistent plate')
        _patch_ctx(monkeypatch,
                   spinner_text='nonexistent plate',
                   settings={'protocol': {'labware': 'nonexistent plate'}},
                   loader=loader)

        from modules.config_ui_getters import get_selected_labware
        result = get_selected_labware()
        assert result == (None, None)

    def test_caller_tuple_unpack_does_not_crash_on_failure(self, monkeypatch):
        # Direct check of the crash chain from #634: every caller does
        # `labware_id, _ = get_selected_labware()`. That must not raise.
        loader = MagicMock()
        loader.get_plate.side_effect = KeyError('New')
        _patch_ctx(monkeypatch,
                   spinner_text='New',
                   settings={'protocol': {'labware': 'New'}},
                   loader=loader)

        from modules.config_ui_getters import get_selected_labware
        # Must not raise TypeError ('cannot unpack non-iterable NoneType').
        labware_id, _ = get_selected_labware()
        assert labware_id is None
