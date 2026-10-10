# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored objective id the catalogue lacks is replaced at bring-up, and told once per load.

Bring-up refused a stored objective_id that named no catalogue objective
with ConfigError, and the GUI's answer to that refusal is to come up on the
shipped template: every other stored setting was dropped for one id, the
plate's defect over again. On a scope with no turret the id is replaced by
the shipped objective where the catalogue is first available and joins the
load's one notice; a turreted scope never reads the stored id, so it keeps
it, and its unknown assignment stays the objective question's.
"""

import json
import logging
import pathlib
import shutil

import pytest

from modules import settings_init
from modules.exceptions import StoredSettingReplacedNotice
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings, complete_settings_without

TURRET_MODEL = 'LS850T'
TEMPLATE = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'settings.json'


def _loaded(tmp_path, edit):
    """The settings one load prepares from a current.json ``edit`` changed."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    current = json.loads(TEMPLATE.read_text())
    edit(current)
    (data_dir / 'current.json').write_text(json.dumps(current))
    settings, _ = settings_init.prepare_settings(
        logging.getLogger(__name__), str(tmp_path), fall_back_to_template=False
    )
    settings['live_folder'] = str(tmp_path)
    settings['simulator_tier'] = 'fast'
    return settings


def _notices(heard):
    return [n for n in heard if n.reason == StoredSettingReplacedNotice.reason]


@pytest.mark.parametrize('stored', ['NOT_AN_OBJECTIVE', None, 7])
def test_a_scope_with_no_turret_comes_up_on_the_shipped_objective_and_keeps_the_rest(
    tmp_path, stored
):
    settings = complete_settings(live_folder=str(tmp_path), jpg_quality=77)
    settings['objective_id'] = stored
    heard = []
    session = ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)
    try:
        shipped = session.scope.settings_template['objective_id']
        assert session.settings['objective_id'] == shipped
        assert session.scope.runtime_state.get_current_objective_id() == shipped
        assert session.settings['jpg_quality'] == 77
    finally:
        session.shutdown()
    (notice,) = _notices(heard)
    assert notice.message == str(StoredSettingReplacedNotice([('objective_id', stored, shipped)]))


def test_a_dict_with_no_objective_id_comes_up_on_the_shipped_one_and_is_told(tmp_path):
    # A hand-built dict that never carried the key, the shape a factory
    # refused by name: on a scope with no turret the key is the objective,
    # so it is filled with the shipped one and told, as a null is.
    heard = []
    session = ScopeSession.create(
        complete_settings_without('objective_id', live_folder=str(tmp_path)),
        simulate=True,
        outcome_listener=heard.append,
    )
    try:
        shipped = session.scope.settings_template['objective_id']
        assert session.settings['objective_id'] == shipped
    finally:
        session.shutdown()
    (notice,) = _notices(heard)
    assert notice.message == str(StoredSettingReplacedNotice([('objective_id', None, shipped)]))


def test_a_turreted_scope_keeps_the_stored_id_it_never_reads(tmp_path):
    settings = complete_settings(
        live_folder=str(tmp_path),
        microscope=TURRET_MODEL,
        turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None},
    )
    settings['objective_id'] = 'NOT_AN_OBJECTIVE'
    heard = []
    session = ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)
    try:
        assert session.settings['objective_id'] == 'NOT_AN_OBJECTIVE'
    finally:
        session.shutdown()
    assert _notices(heard) == []


def test_a_bad_objective_and_a_bad_binning_label_in_one_load_are_one_notice(tmp_path):
    def edit(current):
        current['objective_id'] = 'NOT_AN_OBJECTIVE'
        current['binning']['size'] = '2x4'

    heard = []
    session = ScopeSession.create(
        _loaded(tmp_path, edit), simulate=True, outcome_listener=heard.append
    )
    try:
        shipped = session.scope.settings_template
        assert session.settings['binning']['size'] == shipped['binning']['size']
        assert session.settings['objective_id'] == shipped['objective_id']
    finally:
        session.shutdown()
    (notice,) = _notices(heard)
    assert notice.message == str(
        StoredSettingReplacedNotice(
            [
                ('binning.size', '2x4', shipped['binning']['size']),
                ('objective_id', 'NOT_AN_OBJECTIVE', shipped['objective_id']),
            ]
        )
    )
