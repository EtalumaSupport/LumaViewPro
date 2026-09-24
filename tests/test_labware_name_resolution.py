# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A labware name is resolved or refused where it enters the program.

One plate has one spelling inside the program: the catalogue key. A name the
catalogue once used and has since renamed is translated at the edge, once,
and nothing past the edge ever compares an old spelling to a key. A plate
this installation does not have is refused at the API boundary, by name and
with the plates it does have, before anything is adopted -- never substituted
into the running scope. A reader that does not know the installation
(post-processing re-reading an archived run) canonicalizes and does not judge.
"""

import logging
import pathlib

import pytest

from modules import labware_loader
from modules.exceptions import ConfigError
from modules.protocol import Protocol, ProtocolFormatError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'
CATALOGUE = labware_loader.WellPlateLoader(source_path='.')

# Both spellings the catalogue's own history has retired, and their keys.
RENAMED = {
    '384 well Corning Spheroid Microplate': '384 well microplate',
    'Center Dish': 'Center Plate',
}
UNKNOWN_PLATE = 'Acme 1536 Ultra Plate'

_STEP_HEADER = (
    'Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\t'
    'Auto_Gain\tExposure\tSum\tObjective\tWell\tTile\tZ-Slice\t'
    'Custom Step\tTile Group ID\tZ-Stack Group ID\tAcquire\t'
    'Video Config\tStim_Config\tStep Index\n'
)
_STEP_ROW = (
    's0\t0\t0\t0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t40.0\t1\t'
    "20x Oly\t\t\t-1\tTrue\t-1\t-1\timage\t{'fps': 5, 'duration': 5}\t"
    "{'Blue': {'enabled': False, 'illumination': 100, 'frequency': 1, "
    "'pulse_width': 10, 'pulse_count': 1}, 'Green': {'enabled': False, "
    "'illumination': 100, 'frequency': 1, 'pulse_width': 10, "
    "'pulse_count': 1}, 'Red': {'enabled': False, 'illumination': 100, "
    "'frequency': 1, 'pulse_width': 10, 'pulse_count': 1}}\t0\n"
)


def _protocol_file(tmp_path, labware):
    tsv = tmp_path / 'named.tsv'
    tsv.write_text(
        'LumaViewPro Protocol\n'
        'Version\t5\n'
        'Period\t30.0\n'
        'Duration\t24.0\n'
        f'Labware\t{labware}\n'
        '\n'
        'Steps\n' + _STEP_HEADER + _STEP_ROW
    )
    return tsv


class TestTheFold:
    """One alias table, read through one function, usable without the catalogue."""

    @pytest.mark.parametrize('old_name, key', sorted(RENAMED.items()))
    def test_a_retired_spelling_folds_to_its_key(self, old_name, key):
        assert labware_loader.canonical_plate_name(old_name) == key

    def test_a_key_folds_to_itself(self):
        for key in CATALOGUE.get_plate_list():
            assert labware_loader.canonical_plate_name(key) == key

    def test_a_name_the_table_does_not_know_is_returned_as_given(self):
        # The fold translates; it does not judge. Judging needs the catalogue.
        assert labware_loader.canonical_plate_name(UNKNOWN_PLATE) == UNKNOWN_PLATE

    def test_the_table_covers_every_rename_in_the_catalogue_history(self):
        # 'Center Dish' shipped, was renamed, and was missing from the table for
        # three years; protocols written before the rename still carry it.
        assert CATALOGUE.is_known_plate('Center Dish')


class TestTheResolver:
    """The catalogue key, or a refusal that names the plate and the plates there are."""

    @pytest.mark.parametrize('old_name, key', sorted(RENAMED.items()))
    def test_a_retired_spelling_resolves_to_its_key(self, old_name, key):
        assert CATALOGUE.resolve_plate_key(old_name) == key

    def test_an_unknown_plate_is_refused_by_name_with_the_choices(self):
        with pytest.raises(ConfigError, match=UNKNOWN_PLATE) as refusal:
            CATALOGUE.resolve_plate_key(UNKNOWN_PLATE)
        assert '96 well microplate' in str(refusal.value)

    @pytest.mark.parametrize('not_a_name', [None, 7, ['a', 'list'], {'k': 1}])
    def test_a_non_string_is_refused_not_crashed(self, not_a_name):
        # A wire payload can decode to anything; the dict lookup would answer
        # an unhashable one with TypeError instead of a refusal.
        with pytest.raises(ConfigError, match='must be a string'):
            CATALOGUE.resolve_plate_key(not_a_name)

    def test_the_membership_test_and_the_lookup_agree_with_the_resolver(self):
        for old_name, key in RENAMED.items():
            assert CATALOGUE.is_known_plate(old_name)
            resolved = CATALOGUE.get_plate(plate_key=old_name)
            canonical = CATALOGUE.get_plate(plate_key=key)
            assert resolved.config == canonical.config
        assert not CATALOGUE.is_known_plate(UNKNOWN_PLATE)


class TestTheReader:
    """The TSV reader canonicalizes always and judges only when handed the catalogue."""

    @pytest.mark.parametrize('old_name, key', sorted(RENAMED.items()))
    def test_a_retired_spelling_loads_as_its_key(self, tmp_path, old_name, key):
        proto = Protocol.from_file(
            file_path=_protocol_file(tmp_path, old_name),
            tiling_configs_file_loc=TILING_CONFIGS,
        )
        assert proto.labware() == key

    def test_with_the_catalogue_an_unknown_plate_is_refused_before_any_object_exists(
        self, tmp_path
    ):
        with pytest.raises(ProtocolFormatError, match=UNKNOWN_PLATE):
            Protocol.from_file(
                file_path=_protocol_file(tmp_path, UNKNOWN_PLATE),
                tiling_configs_file_loc=TILING_CONFIGS,
                wellplate_loader=CATALOGUE,
            )

    def test_without_the_catalogue_an_unknown_plate_loads(self, tmp_path):
        """Post-processing re-reads a run's protocol on whatever install it is
        on, and never uses the plate. A reader that does not know the
        installation has no standing to refuse for it."""
        proto = Protocol.from_file(
            file_path=_protocol_file(tmp_path, UNKNOWN_PLATE),
            tiling_configs_file_loc=TILING_CONFIGS,
        )
        assert proto.labware() == UNKNOWN_PLATE

    def test_a_reloaded_legacy_file_is_saved_under_the_key(self, tmp_path):
        proto = Protocol.from_file(
            file_path=_protocol_file(tmp_path, 'Center Dish'),
            tiling_configs_file_loc=TILING_CONFIGS,
        )
        saved = tmp_path / 'resaved.tsv'
        proto.to_file(file_path=saved)
        assert 'Labware\tCenter Plate\n' in saved.read_text()
        assert 'Center Dish' not in saved.read_text()


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(), simulate=True)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


class TestTheApiLoad:
    """The API's load knows the installation, so it is where the refusal lives."""

    def test_an_unknown_plate_is_refused_at_the_load(self, tmp_path, session):
        with pytest.raises(ProtocolFormatError, match=UNKNOWN_PLATE):
            session.scope.protocols.load_protocol(file_path=_protocol_file(tmp_path, UNKNOWN_PLATE))

    def test_a_retired_spelling_loads_as_its_key(self, tmp_path, session):
        proto = session.scope.protocols.load_protocol(
            file_path=_protocol_file(tmp_path, 'Center Dish')
        )
        assert proto.labware() == 'Center Plate'
