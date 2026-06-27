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

    def test_subsecond_clamp_is_silent_in_getter(self, monkeypatch):
        # The clamp warning moved OUT of this getter (which save + run-start
        # also call, causing repeated warnings) into update_period /
        # update_duration, which fire once at the field edit. So the getter
        # itself must stay silent on a sub-second value.
        warnings = self._patch(monkeypatch, period='0.001', duration='2')
        from modules.config_ui_getters import get_protocol_time_params

        get_protocol_time_params()
        assert warnings == []

    def test_zero_single_scan_does_not_notify(self, monkeypatch):
        # 0 is the single-scan marker, preserved by the floor -- not a clamp.
        warnings = self._patch(monkeypatch, period='0', duration='0')
        from modules.config_ui_getters import get_protocol_time_params

        get_protocol_time_params()
        assert warnings == []


class TestImageCaptureConfigSharedBuilder:
    """The UI lane (get_image_capture_config_from_ui) and the settings/headless
    lane (get_image_capture_config_from_settings) must forward an IDENTICAL
    capture-config dict for the same image mode + inputs. They differ only in
    where the mode / output_format / jpg_quality come from; the dict shape and
    the capture_depth / save_encoding derivation are folded into one shared
    builder so the two paths cannot drift.
    """

    @staticmethod
    def _patch_ui_ctx(monkeypatch, *, mode, live, sequenced, jpg_quality):
        live_spinner = MagicMock()
        live_spinner.text = live
        seq_spinner = MagicMock()
        seq_spinner.text = sequenced
        microscope_settings = MagicMock()
        microscope_settings.ids = {
            'live_image_output_format_spinner': live_spinner,
            'sequenced_image_output_format_spinner': seq_spinner,
        }
        ctx = MagicMock()
        ctx.motion_settings.ids = {'microscope_settings_id': microscope_settings}
        ctx.scope_display.image_mode = mode
        ctx.settings = {'jpg_quality': jpg_quality}

        import modules.app_context as app_context

        monkeypatch.setattr(app_context, 'ctx', ctx)

    def test_ui_and_settings_lanes_produce_identical_config(self, monkeypatch):
        mode = '12bit_scientific'
        live, sequenced, jpg_quality = 'PNG', 'JPG', 55
        self._patch_ui_ctx(
            monkeypatch, mode=mode, live=live, sequenced=sequenced, jpg_quality=jpg_quality
        )

        from modules.config_ui_getters import get_image_capture_config_from_ui
        from modules.config_helpers import (
            build_image_capture_config,
            get_image_capture_config_from_settings,
        )

        ui_cfg = get_image_capture_config_from_ui()
        settings_cfg = get_image_capture_config_from_settings(
            {
                'image_output_format': {'live': live, 'sequenced': sequenced},
                'image_mode': mode,
                'jpg_quality': jpg_quality,
            }
        )
        shared = build_image_capture_config(
            output_format={'live': live, 'sequenced': sequenced},
            mode=mode,
            jpg_quality=jpg_quality,
        )

        # All three are the SAME dict for the same inputs.
        assert ui_cfg == settings_cfg == shared

    def test_derived_keys_come_only_from_the_shared_builder(self, monkeypatch):
        # Drift proof: patch the single mode-resolution the shared builder uses
        # and confirm BOTH lanes pick up the changed capture_depth /
        # save_encoding identically -- i.e. a new image-mode-derived key is a
        # single edit in the builder, not one-per-lane.
        mode = '12bit_scientific'
        self._patch_ui_ctx(monkeypatch, mode=mode, live='TIFF', sequenced='TIFF', jpg_quality=90)

        import modules.config_helpers as config_helpers
        from modules.config_ui_getters import get_image_capture_config_from_ui

        sentinel = {'capture_depth': 999, 'save_encoding': 'SENTINEL'}
        monkeypatch.setattr(config_helpers.image_mode, 'resolve_image_mode', lambda _mode: sentinel)

        ui_cfg = get_image_capture_config_from_ui()
        settings_cfg = config_helpers.get_image_capture_config_from_settings({'image_mode': mode})

        assert ui_cfg['capture_depth'] == settings_cfg['capture_depth'] == 999
        assert ui_cfg['save_encoding'] == settings_cfg['save_encoding'] == 'SENTINEL'


def test_protocol_time_clamped_detects_subsecond_per_unit():
    # The edit handlers use this to decide whether to warn, with the correct
    # unit: period is minutes, duration is hours.
    from modules import config_helpers

    assert config_helpers.protocol_time_clamped(0.001, 'minutes') is True
    assert config_helpers.protocol_time_clamped(0.0001, 'hours') is True
    # Normal values are not clamped.
    assert config_helpers.protocol_time_clamped(5, 'minutes') is False
    assert config_helpers.protocol_time_clamped(1, 'hours') is False
    # 0 is the single-scan marker, not a clamp.
    assert config_helpers.protocol_time_clamped(0, 'minutes') is False
    assert config_helpers.protocol_time_clamped(0, 'hours') is False
