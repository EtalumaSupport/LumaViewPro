# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for headless (GUI-free) config helpers in modules/config_helpers.py.

These functions must work without Kivy or any UI -- they read from
the settings dict only.
"""

import datetime
from unittest.mock import MagicMock

from modules.config_helpers import (
    get_binning_from_settings,
    get_frame_dimensions_from_settings,
    get_protocol_time_params_from_settings,
    get_image_capture_config_from_settings,
    get_selected_labware_from_settings,
    get_zstack_params_from_settings,
    get_auto_gain_settings,
    get_current_objective_info,
)


class TestGetBinningFromSettings:
    def test_reads_binning(self):
        assert get_binning_from_settings({'binning_size': 2}) == 2

    def test_defaults_to_1(self):
        assert get_binning_from_settings({}) == 1

    def test_handles_string(self):
        assert get_binning_from_settings({'binning_size': '4'}) == 4

    def test_handles_invalid(self):
        assert get_binning_from_settings({'binning_size': 'bad'}) == 1


class TestGetFrameDimensions:
    def test_reads_frame(self):
        result = get_frame_dimensions_from_settings({'frame': {'width': 800, 'height': 600}})
        assert result == {'width': 800, 'height': 600}

    def test_defaults(self):
        result = get_frame_dimensions_from_settings({})
        assert result == {'width': 1900, 'height': 1900}


class TestGetProtocolTimeParams:
    def test_reads_params(self):
        settings = {'protocol': {'period': 5, 'duration': 2}}
        result = get_protocol_time_params_from_settings(settings)
        assert result['period'] == datetime.timedelta(minutes=5)
        assert result['duration'] == datetime.timedelta(hours=2)

    def test_defaults(self):
        result = get_protocol_time_params_from_settings({})
        assert result['period'] == datetime.timedelta(minutes=1)
        assert result['duration'] == datetime.timedelta(hours=1)


class TestGetImageCaptureConfig:
    def test_reads_config(self):
        settings = {
            'image_output_format': {'live': 'PNG', 'sequenced': 'TIFF'},
            'use_full_pixel_depth': True,
        }
        result = get_image_capture_config_from_settings(settings)
        assert result['output_format']['live'] == 'PNG'
        assert result['use_full_pixel_depth'] is True

    def test_defaults(self):
        result = get_image_capture_config_from_settings({})
        assert result['output_format']['live'] == 'TIFF'
        assert result['use_full_pixel_depth'] is False


class TestGetSelectedLabware:
    """get_selected_labware_from_settings ALWAYS returns a valid plate per
    Eric's 2026-04-25 directive -- never None. Falls back to the shipped
    default, then to the first available plate, then raises only if the
    loader is genuinely empty (broken install).
    """

    def test_reads_labware(self):
        loader = MagicMock()
        plate = MagicMock()
        loader.get_plate.return_value = plate
        settings = {'protocol': {'labware': '96-well'}}
        labware_id, obj = get_selected_labware_from_settings(settings, loader)
        assert labware_id == '96-well'
        assert obj is plate

    def test_empty_settings_falls_back_to_default(self):
        # No labware in settings -> use DEFAULT_LABWARE_ID '96 well microplate'.
        loader = MagicMock()
        plate = MagicMock()
        loader.get_plate.return_value = plate
        labware_id, obj = get_selected_labware_from_settings({}, loader)
        assert labware_id == '96 well microplate'
        assert obj is plate

    def test_loader_keyerror_falls_back_to_default(self):
        # Settings has a labware id but loader doesn't recognize it ->
        # fall back to default.
        loader = MagicMock()
        default_plate = MagicMock()

        def fake_get_plate(plate_key=None):
            if plate_key == 'nonexistent':
                raise KeyError('not found')
            return default_plate

        loader.get_plate.side_effect = fake_get_plate
        settings = {'protocol': {'labware': 'nonexistent'}}
        labware_id, obj = get_selected_labware_from_settings(settings, loader)
        assert labware_id == '96 well microplate'
        assert obj is default_plate

    def test_unavailable_labware_notifies_user(self, monkeypatch):
        # The substitution changes plate geometry, so the user must be told
        # rather than have the protocol silently run on the wrong plate (EXC-M-9).
        loader = MagicMock()
        default_plate = MagicMock()

        def fake_get_plate(plate_key=None):
            if plate_key == 'nonexistent':
                raise KeyError('not found')
            return default_plate

        loader.get_plate.side_effect = fake_get_plate

        warnings = []
        import modules.notification_center as nc

        monkeypatch.setattr(
            nc.notifications,
            'warning',
            lambda category, title, message, **k: warnings.append((category, title, message)),
        )

        get_selected_labware_from_settings({'protocol': {'labware': 'nonexistent'}}, loader)
        assert any('Labware' in category for category, _, _ in warnings)

    def test_loader_keyerror_on_default_falls_back_to_first_available(self):
        # Both requested AND default missing -> fall back to first plate
        # in the loader's list.
        loader = MagicMock()
        first_plate = MagicMock()

        def fake_get_plate(plate_key=None):
            if plate_key in ('requested-key', '96 well microplate'):
                raise KeyError('not found')
            return first_plate

        loader.get_plate.side_effect = fake_get_plate
        loader.get_plate_list.return_value = ['some-other-plate']
        settings = {'protocol': {'labware': 'requested-key'}}
        labware_id, obj = get_selected_labware_from_settings(settings, loader)
        assert labware_id == 'some-other-plate'
        assert obj is first_plate

    def test_loader_completely_empty_raises_valueerror(self):
        # Genuinely-broken install: labware.json missing entirely. The
        # function must raise rather than return None -- caller can't
        # recover from a missing labware database.
        loader = MagicMock()
        loader.get_plate.side_effect = KeyError('not found')
        loader.get_plate_list.return_value = []
        settings = {'protocol': {'labware': 'anything'}}
        import pytest
        from modules.exceptions import ConfigError

        with pytest.raises(ConfigError, match='no plates registered'):
            get_selected_labware_from_settings(settings, loader)


class TestGetZstackParams:
    def test_reads_params(self):
        settings = {'protocol': {'zstack': {'range': 50, 'step_size': 5, 'z_reference': 'top'}}}
        result = get_zstack_params_from_settings(settings)
        assert result == {'range': 50.0, 'step_size': 5.0, 'z_reference': 'top'}

    def test_defaults(self):
        result = get_zstack_params_from_settings({})
        assert result['range'] == 0.0
        assert result['step_size'] == 1.0
        assert result['z_reference'] == 'center'


class TestGetAutoGainSettings:
    def test_converts_seconds_to_timedelta(self):
        settings = {'protocol': {'autogain': {'max_duration_seconds': 30, 'target_percent': 80}}}
        result = get_auto_gain_settings(settings)
        assert result['max_duration'] == datetime.timedelta(seconds=30)
        assert 'max_duration_seconds' not in result
        assert result['target_percent'] == 80


class TestGetCurrentObjectiveInfo:
    def test_reads_objective(self):
        helper = MagicMock()
        helper.get_objective_info.return_value = {'focal_length': 10}
        settings = {'objective_id': '20x Oly'}
        obj_id, info = get_current_objective_info(settings, helper)
        assert obj_id == '20x Oly'
        assert info['focal_length'] == 10
