# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored plate the catalogue cannot resolve is replaced at bring-up, and told once per load.

Bring-up refused a stored labware the catalogue did not have (a null, a
number, an empty name, a plate since removed) with ConfigError, and the
GUI's answer to that refusal is to come up on the shipped template:
every other stored setting was dropped for one bad plate name. The plate
is now replaced by the shipped one where the catalogue is first
available, and the replacement joins the load's own in one notice.
"""

import json
import logging
import pathlib
import shutil

import pytest

from modules import settings_init
from modules.exceptions import ConfigError, StoredSettingReplacedNotice
from modules.lumascope_api._lumascope import Lumascope
from modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = REPO / 'data' / 'settings.json'
SHIPPED = json.loads(TEMPLATE.read_text())


def _notices(heard):
    """The replacement notices a listener heard, one per load."""
    return [n for n in heard if n.title == StoredSettingReplacedNotice.title]


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


@pytest.mark.parametrize('stored', [None, 7, '', 'A plate since removed'])
def test_the_scope_comes_up_on_the_shipped_plate_and_keeps_every_other_setting(tmp_path, stored):
    settings = complete_settings(live_folder=str(tmp_path), jpg_quality=77)
    settings['protocol']['labware'] = stored
    heard = []

    session = ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)
    try:
        assert session.settings['protocol']['labware'] == SHIPPED['protocol']['labware']
        assert session.settings['jpg_quality'] == 77
    finally:
        session.shutdown()
    (notice,) = _notices(heard)
    assert notice.message == str(
        StoredSettingReplacedNotice([('protocol.labware', stored, SHIPPED['protocol']['labware'])])
    )


def test_a_bad_plate_and_a_bad_period_in_one_load_are_one_notice(tmp_path):
    def edit(current):
        current['protocol']['labware'] = 'A plate since removed'
        current['protocol']['period'] = 0.001

    heard = []
    session = ScopeSession.create(
        _loaded(tmp_path, edit), simulate=True, outcome_listener=heard.append
    )
    session.shutdown()

    (notice,) = _notices(heard)
    assert 'protocol.labware' in notice.message
    assert 'protocol.period' in notice.message


def test_a_bring_up_that_raises_still_tells_what_its_load_replaced_and_the_fallback_tells_nothing(
    tmp_path, monkeypatch
):
    def edit(current):
        current['protocol']['labware'] = 'A plate since removed'
        current['protocol']['period'] = 0.001

    settings = _loaded(tmp_path, edit)
    real_initialize = Lumascope.initialize

    def refuse(self, config):
        raise ConfigError('a stored value bring-up cannot use')

    monkeypatch.setattr(Lumascope, 'initialize', refuse)
    heard = []
    with pytest.raises(ConfigError):
        ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)
    (notice,) = _notices(heard)
    assert 'protocol.labware' in notice.message
    assert 'protocol.period' in notice.message

    monkeypatch.setattr(Lumascope, 'initialize', real_initialize)
    fallback_heard = []
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)),
        simulate=True,
        outcome_listener=fallback_heard.append,
    )
    session.shutdown()
    assert _notices(fallback_heard) == []


def test_a_caller_that_brings_its_own_scope_up_is_told_by_the_bring_up(tmp_path):
    settings = complete_settings(live_folder=str(tmp_path))
    settings['protocol']['labware'] = 'A plate since removed'
    heard = []
    session = ScopeSession.create(
        settings, scope=build_scope(simulate=True), outcome_listener=heard.append
    )
    try:
        assert _notices(heard) == []
        session.configure_scope()
    finally:
        session.shutdown()
    (notice,) = _notices(heard)
    assert "protocol.labware 'A plate since removed'" in notice.message
