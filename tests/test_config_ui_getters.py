# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for UI-dependent config getters in modules/config_ui_getters.py.

Headless equivalents in modules/config_helpers.py are tested in
tests/test_headless_config.py. get_selected_labware() answers the stored
plate as a (labware_id, plate) tuple, never None, or raises ConfigError
naming a plate the catalogue does not have. It never substitutes a
different plate.
"""

import datetime
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
    """UI variant -- the stored plate, or ConfigError; never another plate."""

    def test_settings_is_the_single_store(self, monkeypatch):
        # Settings owns the selection: the spinner writes through on
        # every user pick, so a stale spinner text (a settings write
        # that bypassed it, e.g. a protocol load) must NOT shadow the
        # stored value -- the divergence the spinner-first read used to
        # open between the GUI and headless paths.
        loader = MagicMock()
        plate = MagicMock()
        loader.resolve_plate_key.side_effect = lambda name: name
        loader.get_plate.return_value = plate
        _patch_ctx(
            monkeypatch,
            spinner_text='stale spinner text',
            settings={'protocol': {'labware': '96 well microplate'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, obj = get_selected_labware()
        assert labware_id == '96 well microplate'
        assert obj is plate

    def test_spinner_empty_falls_back_to_settings(self, monkeypatch):
        loader = MagicMock()
        plate = MagicMock()
        loader.resolve_plate_key.side_effect = lambda name: name
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

    def test_invalid_stored_labware_is_refused_by_name(self, monkeypatch):
        # 'New' was the old KV spinner default that leaked into settings.
        # A plate the catalogue does not have is refused, never replaced
        # by the default: the default's geometry would move every well.
        from modules.exceptions import ConfigError
        from modules.labware_loader import WellPlateLoader

        _patch_ctx(
            monkeypatch,
            spinner_text='New',
            settings={'protocol': {'labware': 'New'}},
            loader=WellPlateLoader(),
        )

        from modules.config_ui_getters import get_selected_labware

        with pytest.raises(ConfigError, match="unknown labware 'New'"):
            get_selected_labware()

    def test_spinner_empty_and_settings_missing_is_refused(self, monkeypatch):
        from modules.exceptions import ConfigError
        from modules.labware_loader import WellPlateLoader

        _patch_ctx(monkeypatch, spinner_text='', settings={}, loader=WellPlateLoader())

        from modules.config_ui_getters import get_selected_labware

        with pytest.raises(ConfigError):
            get_selected_labware()

    def test_no_first_available_plate_is_substituted(self, monkeypatch):
        from modules.exceptions import ConfigError

        loader = MagicMock()
        loader.resolve_plate_key.side_effect = ConfigError("unknown labware 'nonexistent plate'")
        loader.get_plate_list.return_value = ['some-other-plate']
        _patch_ctx(
            monkeypatch,
            spinner_text='nonexistent plate',
            settings={'protocol': {'labware': 'nonexistent plate'}},
            loader=loader,
        )

        from modules.config_ui_getters import get_selected_labware

        with pytest.raises(ConfigError, match='nonexistent plate'):
            get_selected_labware()
        loader.get_plate.assert_not_called()

    def test_caller_tuple_unpack_gets_a_string_or_a_raise(self, monkeypatch):
        # The original crash chain was `labware_id, _ = get_selected_labware()`
        # blowing up on TypeError from a None. The answer is a non-empty
        # string or a ConfigError the caller can report -- never None.
        from modules.labware_loader import WellPlateLoader

        _patch_ctx(
            monkeypatch,
            spinner_text='',
            settings={'protocol': {'labware': '6 well microplate'}},
            loader=WellPlateLoader(),
        )

        from modules.config_ui_getters import get_selected_labware

        labware_id, _ = get_selected_labware()
        assert isinstance(labware_id, str)
        assert labware_id == '6 well microplate'


class TestTimingAndBinningParseNotifies:
    """A failed parse must not silently run the protocol on a default value.

    Binning still notifies from its getter. Protocol time no longer does: it
    refuses, so a headless caller sees the same failure -- the notification
    for a bad KEYSTROKE now belongs to the field's own handler, which is the
    only place that sees the typed text.
    """

    @staticmethod
    def _patch(monkeypatch, *, period='1', duration='1', binning='1x1'):
        """A GUI whose store holds period/duration and whose spinner holds binning.

        The two time fields are set to the same values as the store: they are
        no longer read for the config, and leaving them in place keeps the
        harness honest about what a real GUI looks like.
        """
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
        ctx.settings = {
            'protocol': {'period': period, 'duration': duration},
            'binning': {'size': binning},
        }

        import modules.app_context as app_context

        monkeypatch.setattr(app_context, 'ctx', ctx)

    def test_an_unparseable_stored_period_is_refused_naming_it(self):
        # The stored schedule is read in one place, the settings lane, and a
        # value it cannot use is refused there -- never replaced by a default
        # schedule the user did not choose.
        from modules.config_helpers import get_protocol_time_params_from_settings
        from modules.protocol import ProtocolScheduleRefusedError

        with pytest.raises(ProtocolScheduleRefusedError) as err:
            get_protocol_time_params_from_settings(
                {'protocol': {'period': 'not-a-number', 'duration': '1'}}
            )
        assert 'period' in str(err.value)

    def test_an_unparseable_stored_duration_is_refused_naming_it(self):
        from modules.config_helpers import get_protocol_time_params_from_settings
        from modules.protocol import ProtocolScheduleRefusedError

        with pytest.raises(ProtocolScheduleRefusedError) as err:
            get_protocol_time_params_from_settings(
                {'protocol': {'period': '1', 'duration': 'not-a-number'}}
            )
        assert 'duration' in str(err.value)

    def test_absent_schedule_still_defaults(self, monkeypatch):
        # Absent is not corrupt: the shipped template carries both keys and the
        # default merge fills them, so an absent key means a caller built a
        # config without a schedule. That keeps working.
        self._patch(monkeypatch)
        from modules.config_helpers import get_protocol_time_params_from_settings

        params = get_protocol_time_params_from_settings({})
        assert params['period'] == datetime.timedelta(minutes=1)
        assert params['duration'] == datetime.timedelta(hours=1)


class TestGettersThatTakeTheirInputs:
    """The getter that used to reach the app context for a GUI fact.

    get_active_layer_config read which accordion drawer was open, so it could
    not answer for a caller that is not the running app.

    These build on the shipped template rather than a hand-made dict: layers
    are top-level keys and the z-stack reference is a display label, and a
    hand-made shape that gets either wrong tests a config that cannot occur.
    """

    @staticmethod
    def _ctx(monkeypatch, **overrides):
        import json
        import pathlib as _pathlib

        repo = _pathlib.Path(__file__).resolve().parent.parent
        settings = json.loads((repo / 'data' / 'settings.json').read_text())
        settings.update(overrides)

        ctx = MagicMock()
        ctx.settings = settings

        import modules.app_context as app_context

        monkeypatch.setattr(app_context, 'ctx', ctx)
        return ctx

    def test_the_layer_is_the_one_named(self, monkeypatch):
        self._ctx(monkeypatch)

        from modules.config_ui_getters import get_active_layer_config

        layer, config = get_active_layer_config('Blue')

        assert layer == 'Blue'
        assert 'exposure_ms' in config

    def test_no_layer_selected_is_refused(self, monkeypatch):
        """The GUI passes whatever the open drawer was, including nothing.

        The refusal lives in the getter rather than in each caller, so the
        three GUI starters cannot come to disagree about it.
        """
        self._ctx(monkeypatch)

        from modules.config_ui_getters import get_active_layer_config

        with pytest.raises(Exception, match='No layer currently selected'):
            get_active_layer_config(None)
