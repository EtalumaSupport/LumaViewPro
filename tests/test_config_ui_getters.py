# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for UI-dependent config getters in modules/config_ui_getters.py.

Headless equivalents in modules/config_helpers.py are tested in
tests/test_headless_config.py. get_selected_labware() answers the live
scope's selected plate as a (labware_id, plate) tuple, never None, or
raises ConfigError naming a plate the catalogue does not have. It never
substitutes a different plate.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def gui_on_a_scope(monkeypatch):
    """The GUI's context attached to a simulated LS850 bound to the template's settings.

    Yields the scope and the settings it reads; a test names its plate there.
    """
    import modules.app_context as app_context
    from tests.scope_fakes import bind_settings_like_a_session, build_scope

    scope = build_scope(simulate=True, sim_model='LS850', register_atexit=False)
    settings = bind_settings_like_a_session(scope)
    # A stand-in by design: the GUI's context, whose one read here is its live scope.
    ctx = MagicMock()
    ctx.lumaview.scope = scope
    monkeypatch.setattr(app_context, 'ctx', ctx)
    yield scope, settings
    scope.disconnect()


class TestGetSelectedLabware:
    """The GUI's selected plate is the scope's answer; a plate it lacks is refused, never replaced."""

    def test_the_answer_is_the_plate_the_scope_has_selected(self, gui_on_a_scope):
        from modules.config_ui_getters import get_selected_labware

        scope, settings = gui_on_a_scope
        settings['protocol']['labware'] = '6 well microplate'

        assert get_selected_labware() == ('6 well microplate', scope.labware['6 well microplate'])

    def test_a_stored_plate_the_catalogue_lacks_is_refused_by_name(self, gui_on_a_scope):
        # 'New' was the old KV spinner default that leaked into settings. A
        # plate the catalogue does not have is refused, never replaced by
        # another: that plate's geometry would move every well.
        from modules.config_ui_getters import get_selected_labware
        from modules.exceptions import CatalogueNameRefusedError

        _, settings = gui_on_a_scope
        settings['protocol']['labware'] = 'New'

        with pytest.raises(CatalogueNameRefusedError) as refused:
            get_selected_labware()
        assert (refused.value.reason, refused.value.value) == ('labware_unknown', 'New')


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
