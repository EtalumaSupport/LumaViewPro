# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for UI-dependent config getters in modules/config_ui_getters.py.

Headless equivalents in modules/config_helpers.py are tested in
tests/test_headless_config.py. Per Eric's 2026-04-25 directive,
get_selected_labware() ALWAYS returns a valid (labware_id, plate) tuple
-- never None -- by falling back to the shipped default labware and then
to the first available plate. Issue #634/#632 cluster: removing None
from the contract retires the cluster of latent crash sites that
consumed it without None-checking.
"""

from unittest.mock import MagicMock


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
    """UI variant -- always returns a valid (labware_id, plate) tuple."""

    def test_spinner_has_valid_labware(self, monkeypatch):
        loader = MagicMock()
        plate = MagicMock()
        loader.get_plate.return_value = plate
        _patch_ctx(
            monkeypatch,
            spinner_text='96 well microplate',
            settings={'protocol': {'labware': 'unused-fallback'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, obj = get_selected_labware()
        assert labware_id == '96 well microplate'
        assert obj is plate

    def test_spinner_empty_falls_back_to_settings(self, monkeypatch):
        loader = MagicMock()
        plate = MagicMock()
        loader.get_plate.return_value = plate
        _patch_ctx(
            monkeypatch,
            spinner_text='',
            settings={'protocol': {'labware': '96 well microplate'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, obj = get_selected_labware()
        assert labware_id == '96 well microplate'
        assert obj is plate

    def test_spinner_has_stale_default_falls_back_to_default(self, monkeypatch):
        # Issue #634 regression: KV file used to ship `text: 'New'` as the
        # spinner default. On first run, the spinner read 'New' before
        # settings synced to UI -- and 'New' isn't a valid labware key.
        # Per Eric's 2026-04-25 directive, the function now falls back
        # cleanly to DEFAULT_LABWARE_ID rather than returning None.
        loader = MagicMock()
        default_plate = MagicMock()

        def fake_get_plate(plate_key=None):
            if plate_key == 'New':
                raise KeyError('New')
            return default_plate

        loader.get_plate.side_effect = fake_get_plate
        _patch_ctx(
            monkeypatch,
            spinner_text='New',
            settings={'protocol': {'labware': 'New'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, obj = get_selected_labware()
        # Falls back to '96 well microplate' (DEFAULT_LABWARE_ID).
        assert labware_id == '96 well microplate'
        assert obj is default_plate

    def test_spinner_empty_and_settings_missing_uses_default(self, monkeypatch):
        loader = MagicMock()
        default_plate = MagicMock()
        loader.get_plate.return_value = default_plate
        _patch_ctx(monkeypatch, spinner_text='', settings={}, loader=loader)

        from modules.config_ui_getters import get_selected_labware

        labware_id, obj = get_selected_labware()
        assert labware_id == '96 well microplate'
        assert obj is default_plate

    def test_loader_keyerror_falls_back_to_first_available(self, monkeypatch):
        # Both requested AND default missing -> fall back to first plate
        # in loader.get_plate_list().
        loader = MagicMock()
        first_plate = MagicMock()

        def fake_get_plate(plate_key=None):
            if plate_key in ('nonexistent plate', '96 well microplate'):
                raise KeyError('not found')
            return first_plate

        loader.get_plate.side_effect = fake_get_plate
        loader.get_plate_list.return_value = ['some-other-plate']
        _patch_ctx(
            monkeypatch,
            spinner_text='nonexistent plate',
            settings={'protocol': {'labware': 'nonexistent plate'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, obj = get_selected_labware()
        assert labware_id == 'some-other-plate'
        assert obj is first_plate

    def test_caller_tuple_unpack_does_not_crash_on_any_input(self, monkeypatch):
        # The original #634 crash chain was `labware_id, _ = get_selected_labware()`
        # blowing up on TypeError. With the always-valid contract, this
        # path is impossible -- labware_id is always a non-None string.
        loader = MagicMock()
        default_plate = MagicMock()

        def fake_get_plate(plate_key=None):
            if plate_key == 'New':
                raise KeyError('New')
            return default_plate

        loader.get_plate.side_effect = fake_get_plate
        _patch_ctx(
            monkeypatch,
            spinner_text='New',
            settings={'protocol': {'labware': 'New'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, _ = get_selected_labware()
        assert isinstance(labware_id, str)
        assert labware_id  # non-empty


class TestTimingAndBinningParseNotifies:
    """A failed parse of period / duration / binning notifies the user instead
    of silently running the protocol with a default value (EXC-M-8)."""

    @staticmethod
    def _patch(monkeypatch, *, period='1', duration='1', binning='1x1'):
        ctx = MagicMock()
        period_field = MagicMock()
        period_field.text = period
        dur_field = MagicMock()
        dur_field.text = duration
        protocol_settings = MagicMock()
        protocol_settings.ids = {'capture_period': period_field, 'capture_dur': dur_field}
        binning_spinner = MagicMock()
        binning_spinner.text = binning
        microscope_settings = MagicMock()
        microscope_settings.ids = {'binning_spinner': binning_spinner}
        ctx.motion_settings.ids = {
            'protocol_settings_id': protocol_settings,
            'microscope_settings_id': microscope_settings,
        }

        import modules.app_context as app_context

        monkeypatch.setattr(app_context, 'ctx', ctx)

        warnings = []
        import modules.notification_center as nc

        monkeypatch.setattr(
            nc.notifications,
            'warning',
            lambda category, title, message, **k: warnings.append((category, title, message)),
        )
        return warnings

    def test_unparseable_period_notifies(self, monkeypatch):
        warnings = self._patch(monkeypatch, period='not-a-number')
        from modules.config_ui_getters import get_protocol_time_params

        get_protocol_time_params()
        assert any('Timing' in title for _, title, _ in warnings)

    def test_unparseable_binning_notifies(self, monkeypatch):
        warnings = self._patch(monkeypatch, binning='garbage')
        from modules.config_ui_getters import get_binning_from_ui

        assert get_binning_from_ui() == 1
        assert any('Binning' in title for _, title, _ in warnings)

    def test_valid_values_do_not_notify(self, monkeypatch):
        warnings = self._patch(monkeypatch, period='5', duration='2', binning='2x2')
        from modules.config_ui_getters import get_binning_from_ui, get_protocol_time_params

        get_protocol_time_params()
        assert get_binning_from_ui() == 2
        assert warnings == []

    def test_subsecond_period_notifies_on_clamp(self, monkeypatch):
        # 0.001 min = 0.06 s, below the 1 s floor -> clamped to 1 s. The user
        # must be told instead of silently seeing 0.016667 reappear.
        warnings = self._patch(monkeypatch, period='0.001', duration='2')
        from modules.config_ui_getters import get_protocol_time_params

        get_protocol_time_params()
        assert any('Timing' in title for _, title, _ in warnings)

    def test_zero_single_scan_does_not_notify(self, monkeypatch):
        # 0 is the single-scan marker, preserved by the floor -- not a clamp.
        warnings = self._patch(monkeypatch, period='0', duration='0')
        from modules.config_ui_getters import get_protocol_time_params

        get_protocol_time_params()
        assert warnings == []
