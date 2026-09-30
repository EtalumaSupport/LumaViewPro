# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for headless (GUI-free) config helpers in modules/config_helpers.py.

These functions must work without Kivy or any UI -- they read from
the settings dict only.
"""

import datetime
from unittest.mock import MagicMock

import pytest

from modules.config_helpers import (
    get_binning_from_settings,
    get_frame_dimensions_from_settings,
    get_protocol_time_params_from_settings,
    get_image_capture_config_from_settings,
    get_selected_labware_from_settings,
    get_sequenced_capture_config_from_settings,
    get_zstack_params_from_settings,
    get_auto_gain_settings,
    get_current_objective_info,
)
from modules.exceptions import ConfigError
from modules.objectives_loader import ObjectiveLoader
from modules.labware_loader import WellPlateLoader
from modules.zstack_config import ZStackConfig
from tests.settings_fixtures import complete_settings


def _objective_helper_for(settings: dict):
    """The real loader -- the objective id comes from the shipped template."""
    return ObjectiveLoader()


def _wellplate_loader():
    """The real loader -- labware resolves through the catalogue the GUI lane
    uses, so a plate it does not have is refused rather than reaching the
    config as a bare empty string."""
    return WellPlateLoader()


class TestGetBinningFromSettings:
    """Reads the key the GUI writes and the template ships.

    Every case here previously hand-built a top-level ``binning_size``
    integer. Nothing writes that key and no shipped template carries it, so
    those cases described a config that cannot occur, and passing them meant
    only that the getter agreed with the test about a fiction. The factor now
    comes from the selector's stored label.
    """

    def test_reads_the_stored_selector_label(self):
        assert get_binning_from_settings({'binning': {'size': '2x2'}}) == 2
        assert get_binning_from_settings({'binning': {'size': '4x4'}}) == 4

    def test_matches_the_shipped_template(self):
        # The template is the config every install starts from, so the getter
        # answering it wrong is the whole defect this replaced.
        import json
        import pathlib

        template = json.loads(
            (pathlib.Path(__file__).resolve().parents[1] / 'data' / 'settings.json').read_text()
        )
        assert get_binning_from_settings(template) == 1
        assert 'binning_size' not in template

    def test_absent_binning_block_is_unbinned(self):
        assert get_binning_from_settings({}) == 1
        assert get_binning_from_settings({'binning': {}}) == 1

    def test_the_retired_top_level_key_is_not_consulted(self):
        # A carried-forward file could still hold it; it must not be able to
        # disagree with the label the selector writes.
        assert get_binning_from_settings({'binning_size': 4, 'binning': {'size': '2x2'}}) == 2


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
        # Two DIFFERENT real formats: the point is that each key is read from
        # settings rather than defaulted, so they only need to be
        # distinguishable. 'PNG' stood in the live slot until the config
        # learned to refuse a format this build cannot write.
        settings = {
            'image_output_format': {'live': 'OME-TIFF', 'sequenced': 'TIFF'},
            'image_mode': '12bit_scientific',
        }
        result = get_image_capture_config_from_settings(settings)
        assert result.output_format_live == 'OME-TIFF'
        assert result.capture_depth == 12

    def test_defaults(self):
        result = get_image_capture_config_from_settings({})
        assert result.output_format_live == 'TIFF'
        assert result.capture_depth == 8


class TestGetSelectedLabware:
    """get_selected_labware_from_settings answers the stored plate or
    raises ConfigError naming it -- never None, and never a different
    plate. A substituted plate has different geometry, so every well
    position computed from it would be wrong while the protocol reads as
    if it ran normally.
    """

    def test_reads_labware(self):
        loader = _wellplate_loader()
        settings = {'protocol': {'labware': '24 well microplate'}}
        labware_id, obj = get_selected_labware_from_settings(settings, loader)
        assert labware_id == '24 well microplate'
        assert obj.config == loader.get_plate('24 well microplate').config

    def test_a_retired_spelling_answers_under_the_catalogue_key(self):
        loader = _wellplate_loader()
        labware_id, _ = get_selected_labware_from_settings(
            {'protocol': {'labware': 'Center Dish'}}, loader
        )
        assert labware_id == 'Center Plate'

    def test_settings_naming_no_plate_are_refused(self):
        with pytest.raises(ConfigError, match='labware name must be a string'):
            get_selected_labware_from_settings({}, _wellplate_loader())

    def test_an_unknown_plate_is_refused_by_name(self):
        with pytest.raises(ConfigError, match="unknown labware 'nonexistent'") as refused:
            get_selected_labware_from_settings(
                {'protocol': {'labware': 'nonexistent'}}, _wellplate_loader()
            )
        # The message lists what the user can pick instead.
        assert '96 well microplate' in str(refused.value)

    def test_an_unknown_plate_posts_nothing(self, monkeypatch):
        # The raise is the report: the caller that waits on it shows it, once.
        posted = []
        import modules.notification_center as nc

        monkeypatch.setattr(nc.notifications, 'warning', lambda *a, **k: posted.append(a))
        with pytest.raises(ConfigError):
            get_selected_labware_from_settings(
                {'protocol': {'labware': 'nonexistent'}}, _wellplate_loader()
            )
        assert posted == []

    def test_an_empty_name_is_refused_not_defaulted(self):
        with pytest.raises(ConfigError, match="unknown labware ''"):
            get_selected_labware_from_settings({'protocol': {'labware': ''}}, _wellplate_loader())


class TestGetZstackParams:
    """Reads the container and leaf the GUI writes and the template ships.

    Both cases here previously hand-built ``protocol.zstack`` holding a
    ``z_reference`` token. Nothing writes that container and no template
    carries it, so the cases described a config that cannot occur and green
    meant only that the getter agreed with the test about a fiction. The
    store keeps the stack top-level under ``zstack`` and keeps the reference
    in ``position`` as the spinner's display label.
    """

    def test_reads_the_stack_the_gui_wrote(self):
        settings = complete_settings(
            zstack={'range': 50, 'step_size': 5, 'position': 'Current Position at Top'}
        )
        assert get_zstack_params_from_settings(settings) == {
            'range': 50.0,
            'step_size': 5.0,
            'z_reference': 'top',
        }

    def test_unconfigured_template_reports_no_stack(self):
        """The shipped template ships zeros, so it has no z-stack.

        Defaulting step_size to 1 made an unconfigured store report one slice
        rather than none, which is what ``number_of_steps()`` then answered.
        """
        result = get_zstack_params_from_settings(complete_settings())
        assert result['range'] == 0.0
        assert result['step_size'] == 0.0
        assert result['z_reference'] == 'center'
        assert (
            ZStackConfig(
                range=result['range'],
                step_size=result['step_size'],
                current_z_reference=result['z_reference'],
                current_z_value=0.0,
            ).number_of_steps()
            == 0
        )

    def test_absent_position_yields_none_rather_than_a_guess(self):
        """Only a hand-built dict can omit it -- the L2/SDK case.

        None rather than 'center' so the missing key stays visible; the guard
        lives where the value is consumed.
        """
        assert (
            get_zstack_params_from_settings({'zstack': {'range': 10, 'step_size': 2}})[
                'z_reference'
            ]
            is None
        )

    def test_unmapped_position_label_refuses_as_config_error(self):
        """Typed, not a bare Exception -- REST middleware can only map the
        typed error onto a response."""
        with pytest.raises(ConfigError, match='Unknown Z-stack position reference'):
            get_zstack_params_from_settings({'zstack': {'position': 'Focus at Middle'}})

    @pytest.mark.parametrize('key', ['range', 'step_size'])
    def test_unparseable_number_refuses_as_config_error(self, key):
        """A raw ValueError from float() escaped this lane untyped, so one
        corrupt value failed differently on the GUI and REST lanes."""
        with pytest.raises(ConfigError, match=f'Z-stack {key} is not a number'):
            get_zstack_params_from_settings({'zstack': {key: 'wide'}})

    def test_absent_position_refuses_loudly_where_it_is_consumed(self):
        """The whole contract: no stored position -> a named refusal.

        Without the else-branch this raised UnboundLocalError naming a local
        variable instead of the reference that caused it.
        """
        params = get_zstack_params_from_settings({'zstack': {'range': 10, 'step_size': 2}})
        config = ZStackConfig(
            range=params['range'],
            step_size=params['step_size'],
            current_z_reference=params['z_reference'],
            current_z_value=5.0,
        )
        with pytest.raises(ConfigError, match='Unknown Z-stack position reference'):
            config.step_positions()


class TestHeadlessTilingOverlap:
    def test_reads_the_top_level_key_the_template_ships(self):
        """Read from under ``protocol`` this found nothing and gave every
        headless run 0% overlap regardless of configuration."""
        settings = complete_settings(tiling_overlap_percent=25.0)
        config = get_sequenced_capture_config_from_settings(
            settings,
            objective_helper=_objective_helper_for(settings),
            wellplate_loader=_wellplate_loader(),
        )
        assert config['tiling_overlap_percent'] == 25.0


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
