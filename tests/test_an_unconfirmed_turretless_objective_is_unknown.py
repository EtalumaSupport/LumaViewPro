# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""On a scope with no turret, the objective is unknown until a person confirms it.

The owner is ``RuntimeState.resolve_current_objective``: without a turret
it answers the stored ``objective_id`` only while ``objective_confirmed``
names that id (``RuntimeState.objective_confirmed``). A stored id nobody
confirmed -- the shipped template's, one put in place of an id the
catalogue does not hold, one a turret period left -- is refused as
``ObjectiveUnknownError('none_selected')``, so no capture is stamped with
a scale for glass nobody said is mounted. Selecting an objective without a
turret is saying what is mounted, so ``select_objective`` and
``confirm_objective`` both record the confirmation as the id.

Each case is brought up through ``ScopeSession.create`` on the simulated
LS850 (no turret), as every host brings a scope up.
"""

import json
import logging
import pathlib

import pytest

from _lumascope.modules.exceptions import InstallationFileError, ObjectiveUnknownError
from _lumascope.modules.objectives_loader import OBJECTIVE_CONFIRMED_ON_A_TURRET, ObjectiveLoader
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings, installation_of_its_own

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = 'LS850'


def _handed(tmp_path, **stored) -> ScopeSession:
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope=MODEL, **stored),
        simulate=True,
    )


def _from_a_file(tmp_path, **stored) -> ScopeSession:
    # A stored file goes through the load's repairs, which a handed dict's
    # refusals do not reach: a null id in a file is replaced, in a hand refused.
    root = installation_of_its_own()
    current = root / 'data' / 'current.json'
    on_disk = json.loads(current.read_text())
    on_disk.update(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope=MODEL, **stored)
    )
    current.write_text(json.dumps(on_disk))
    return ScopeSession.create(
        ScopeSession.load_user_settings(root), source_path=root, simulate=True
    )


# parametrize-justified: each stored state is its own session bring-up.
UNCONFIRMED = {
    'the template': (_handed, {'objective_confirmed': None}),
    'a turret period left it': (
        _handed,
        {'objective_confirmed': OBJECTIVE_CONFIRMED_ON_A_TURRET, 'objective_id': '40x Phase'},
    ),
    'a confirmed id the catalogue lacks, replaced at bring-up': (
        _handed,
        {'objective_confirmed': 'Not An Objective', 'objective_id': 'Not An Objective'},
    ),
    'a null id in a file, replaced at load': (
        _from_a_file,
        {'objective_id': None, 'objective_confirmed': 'turretless'},
    ),
    'an undated confirmation of an id the catalogue lacks': (
        _handed,
        {'objective_confirmed': True, 'objective_id': 'Not An Objective'},
    ),
}


@pytest.mark.parametrize('case', UNCONFIRMED, ids=list(UNCONFIRMED))
def test_an_unconfirmed_objective_is_unknown_and_asked_for(tmp_path, case):
    bring_up, stored = UNCONFIRMED[case]
    session = bring_up(tmp_path, **stored)
    try:
        with pytest.raises(ObjectiveUnknownError) as refused:
            session.capture_settings_snapshot()
        assert refused.value.reason == 'none_selected'
        with pytest.raises(ObjectiveUnknownError) as no_objective:
            _ = session.objective_id
        assert no_objective.value.reason == 'none_selected'
        with pytest.raises(ObjectiveUnknownError) as no_scale:
            session.scope.runtime_state.get_pixel_size_um()
        assert no_scale.value.reason == 'none_selected'
        assert session.objective_question() is not None
    finally:
        session.shutdown()


# parametrize-justified: the two members are two doors to one confirmation.
@pytest.mark.parametrize('member', ['confirm_objective', 'select_objective'])
def test_saying_which_objective_is_mounted_confirms_it(tmp_path, member):
    session = _handed(tmp_path, objective_confirmed=None)
    try:
        getattr(session, member)('10x Oly')
        assert session.objective_id == '10x Oly'
        assert session.settings['objective_confirmed'] == '10x Oly'
        assert session.objective_question() is None
        assert session.capture_settings_snapshot()['objective_id'] == '10x Oly'
    finally:
        session.shutdown()


def test_an_undated_confirmation_of_a_catalogue_id_is_known_from_bring_up(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    session = _handed(tmp_path, objective_confirmed=True, objective_id='20x Oly')
    try:
        assert session.objective_id == '20x Oly'
        assert session.settings['objective_confirmed'] == '20x Oly'
        optics = [r.getMessage() for r in caplog.records if r.getMessage().startswith('[Optics')]
        assert optics and '20x Oly' in optics[0], optics
    finally:
        session.shutdown()


def test_a_light_path_confirmation_from_a_file_is_the_id_stored_beside_it(tmp_path):
    session = _from_a_file(tmp_path, objective_confirmed='turretless', objective_id='4x Oly')
    try:
        assert session.settings['objective_confirmed'] == '4x Oly'
        assert session.objective_id == '4x Oly'
        assert session.objective_question() is None
    finally:
        session.shutdown()


def test_a_catalogue_keying_an_objective_with_the_turret_word_refuses_to_load(tmp_path):
    catalogue = json.loads((REPO_ROOT / 'data' / 'objectives.json').read_text())
    catalogue[OBJECTIVE_CONFIRMED_ON_A_TURRET] = catalogue['20x Oly']
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'objectives.json').write_text(json.dumps(catalogue))

    with pytest.raises(InstallationFileError, match=repr(OBJECTIVE_CONFIRMED_ON_A_TURRET)):
        ObjectiveLoader(source_path=tmp_path)
