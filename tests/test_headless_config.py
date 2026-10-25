# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for headless (GUI-free) config helpers in modules/config_helpers.py.

These functions must work without Kivy or any UI -- they read from
the settings dict only.
"""

import datetime

import pytest

from modules.config_helpers import (
    get_binning_from_settings,
    get_frame_dimensions_from_settings,
    get_protocol_time_params_from_settings,
    get_image_capture_config_from_settings,
    get_selected_labware_from_settings,
    get_sequenced_capture_config_from_settings,
    get_zstack_params_from_settings,
)
from modules.exceptions import CatalogueNameRefusedError, ConfigError
from modules.objectives_loader import ObjectiveLoader
from modules.labware_loader import WellPlateLoader
from modules.notification_center import Severity
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

    def test_the_retired_top_level_key_is_not_consulted(self):
        # A carried-forward file could still hold it; it must not be able to
        # disagree with the label the selector writes.
        assert get_binning_from_settings({'binning_size': 4, 'binning': {'size': '2x2'}}) == 2


class TestGetFrameDimensions:
    def test_reads_frame(self):
        result = get_frame_dimensions_from_settings({'frame': {'width': 800, 'height': 600}})
        assert result == {'width': 800, 'height': 600}


class TestGetProtocolTimeParams:
    def test_reads_params(self):
        settings = {'protocol': {'period': 5, 'duration': 2}}
        result = get_protocol_time_params_from_settings(settings)
        assert result['period'] == datetime.timedelta(minutes=5)
        assert result['duration'] == datetime.timedelta(hours=2)


class TestGetImageCaptureConfig:
    def test_reads_config(self):
        # Two DIFFERENT real formats: the point is that each key is read from
        # settings rather than defaulted, so they only need to be
        # distinguishable. 'PNG' stood in the live slot until the config
        # learned to refuse a format this build cannot write.
        settings = complete_settings(
            image_output_format={'live': 'OME-TIFF', 'sequenced': 'TIFF'},
            image_mode='12bit_scientific',
        )
        result = get_image_capture_config_from_settings(settings)
        assert result.output_format_live == 'OME-TIFF'
        assert result.capture_depth == 12


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
        assert obj == loader.get_plate('24 well microplate')

    def test_a_retired_spelling_answers_under_the_catalogue_key(self):
        loader = _wellplate_loader()
        labware_id, _ = get_selected_labware_from_settings(
            {'protocol': {'labware': 'Center Dish'}}, loader
        )
        assert labware_id == 'Center Plate'

    def test_an_unknown_plate_is_refused_by_name(self):
        with pytest.raises(
            CatalogueNameRefusedError, match=r"labware catalogue.*'nonexistent' is not one"
        ) as refused:
            get_selected_labware_from_settings(
                {'protocol': {'labware': 'nonexistent'}}, _wellplate_loader()
            )
        # The message lists what the user can pick instead.
        assert '96 well microplate' in str(refused.value)

    def test_an_unknown_plate_posts_nothing(self, centre_posts):
        # The raise is the report: the caller that waits on it shows it, once.
        with pytest.raises(ConfigError):
            get_selected_labware_from_settings(
                {'protocol': {'labware': 'nonexistent'}}, _wellplate_loader()
            )
        assert [n for n in centre_posts if n.severity == Severity.WARNING] == []

    def test_an_empty_name_is_refused_not_defaulted(self):
        with pytest.raises(CatalogueNameRefusedError, match=r"labware catalogue.*'' is not one"):
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
        settings = complete_settings(zstack={'range': 50, 'step_size': 5, 'position': 'top'})
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

    def test_a_null_reference_refuses_loudly_where_it_is_consumed(self):
        """A config built with no reference -> a named refusal where it is used.

        The store cannot hold one (the writer refuses it, the load replaces
        it), but a caller can build the config by hand. Without the
        else-branch this raised UnboundLocalError naming a local variable
        instead of the reference that caused it.
        """
        config = ZStackConfig(
            range=10.0, step_size=2.0, current_z_reference=None, current_z_value=5.0
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
            current_z=0.0,
        )
        assert config['tiling_overlap_percent'] == 25.0
