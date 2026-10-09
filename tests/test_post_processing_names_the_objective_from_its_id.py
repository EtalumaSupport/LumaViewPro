# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Post-processing names a file's objective from the id the run recorded, with no catalogue.

The only thing post-processing needed from the objective catalogue was the
objective's token in an output filename, and that token is a function of the
id. Each post-processor used to build its own catalogue from the
installation's folder, so a run post-processed on another installation, or
after the catalogue was edited, dropped the objective from its names -- and a
broken catalogue file failed the whole load. Capture and post-processing now
derive the token through one function, so they cannot name one objective two
ways.
"""

import json
import pathlib

import pytest

from modules.common_utils import PostFunction
from modules.objectives_loader import ObjectiveLoader, objective_short_name
from modules.protocol_post_processor import ProtocolPostProcessor

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Each shipped objective's token as the catalogue produced it before this
# change, recorded as data: capture names must not move.
SHIPPED_TOKENS = {
    '1.25x Oly': '1.25xOly',
    '2x Oly': '2xOly',
    '2.5x Meiji': '2.5xMeiji',
    '4x Oly': '4xOly',
    '10x Oly': '10xOly',
    '10x Phase': '10xPhase',
    '20x Oly': '20xOly',
    '20x w/collar': '20xCollar',
    '20x Phase': '20xPhase',
    '40x w/collar': '40xCollar',
    '40x Phase': '40xPhase',
    '60x w/collar': '60xCollar',
    '60x Meiji': '60xMeiji',
    '100x U Plan Oly': '100xUPlanOly',
    '100x M Plan Oly': '100xMPlanOly',
    '100x Meiji': '100xMeiji',
}


class _NameOnly(ProtocolPostProcessor):
    """The smallest concrete subclass; only the naming helper is under test."""

    @staticmethod
    def _get_groups(df):
        return df

    @staticmethod
    def _filter_ignored_types(df):
        return df

    def _generate_filename(self, df, **kwargs):
        raise NotImplementedError

    def _group_algorithm(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _add_record(protocol_post_record, alg_metadata: dict, root_path):
        raise NotImplementedError


@pytest.fixture
def turret_post_processor():
    return _NameOnly(post_function=PostFunction.COMPOSITE, has_turret=True)


def test_every_shipped_objective_keeps_its_token():
    catalogue = ObjectiveLoader()
    assert set(catalogue.get_objectives_list()) == set(SHIPPED_TOKENS)
    for objective_id, token in SHIPPED_TOKENS.items():
        assert catalogue.get_objective_info(objective_id)['short_name'] == token
        assert objective_short_name(objective_id) == token


def test_a_catalogue_cannot_rename_an_objective(tmp_path):
    # Capture takes its token from the catalogue and post-processing derives
    # it from the id, so a catalogue entry naming its own short_name would
    # name one objective two ways.
    catalogue = json.loads((REPO_ROOT / 'data' / 'objectives.json').read_text())
    catalogue['4x Oly']['short_name'] = 'Custom'
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'objectives.json').write_text(json.dumps(catalogue))

    loaded = ObjectiveLoader(source_path=tmp_path)

    assert loaded.get_objective_info('4x Oly')['short_name'] == objective_short_name('4x Oly')


def test_an_id_the_catalogue_lacks_is_named(turret_post_processor):
    token = turret_post_processor._get_objective_short_name_if_has_turret('63x Zeiss oil')
    assert token == '63xZeissOil'


def test_a_step_with_no_objective_is_not_named(turret_post_processor):
    assert turret_post_processor._get_objective_short_name_if_has_turret('') is None


def test_a_missing_id_is_refused_not_coerced():
    with pytest.raises(TypeError):
        objective_short_name(None)
