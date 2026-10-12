# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A settings dict handed to ``create`` is prepared as the file's is, by the one door.

Before the door, a dict from a caller's hand reached the session as it
was: ``create`` refused it for a key it lacked, and every reader below
hedged with a second value for the keys it might lack, dead on the
prepared file and live only on that hand dict. Now ``create`` runs the
file's own preparation on it (``settings_init.prepare_loaded``): every
template key is present at every depth, a value the writer refuses is
refused as a write is, a container of the wrong kind is refused as the
argument, and a dict the file's preparation already made usable goes
through unchanged, its stored turret assignments included.
"""

import copy
import json
import logging
import pathlib
import shutil
import threading

import pytest

from modules import settings_init
from modules.exceptions import (
    ArgumentRefusedError,
    SettingRefusedError,
    StoredSettingReplacedNotice,
)
from modules.lumascope_api._constants import ACCELERATION_PCT_MAX
from modules.scope_session import ScopeSession
from modules.settings_paths import _leaf_paths
from tests.settings_fixtures import complete_settings, shipped_template

TEMPLATE = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'settings.json'
LOGGER = logging.getLogger(__name__)


def _value_at(settings: dict, path: str):
    value = settings
    for segment in path.split('.'):
        value = value[segment]
    return value


def test_a_hand_dict_comes_up_with_every_template_key_and_is_told_nothing(tmp_path):
    given = {'live_folder': str(tmp_path), 'microscope': 'LS850', 'simulator_tier': 'fast'}
    heard = []
    session = ScopeSession.create(dict(given), simulate=True, outcome_listener=heard.append)
    try:
        template = shipped_template()
        for path, shipped in _leaf_paths(template):
            if path in given or path.startswith('turret_objectives.'):
                continue
            assert _value_at(session.settings, path) == shipped, path
        for path, value in given.items():
            assert session.settings[path] == value
        # The template's slot keys are strings in the file and ints in the store.
        assert set(session.settings['turret_objectives']) == {1, 2, 3, 4}
    finally:
        session.shutdown()
    # Filled, not replaced: the load's notice names nothing.
    assert [n for n in heard if n.reason == StoredSettingReplacedNotice.reason] == []


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        ('jpg_quality', 500),
        ('motion.acceleration_max_pct', ACCELERATION_PCT_MAX + 400),
        ('motion.acceleration_max_pct', '50'),
        ('motion.acceleration_max_pct', None),
    ],
)
def test_a_value_the_writer_refuses_is_refused_as_a_write_is_and_nothing_is_built(path, value):
    settings = complete_settings()
    block, _, leaf = path.rpartition('.')
    (settings[block] if block else settings)[leaf] = value
    before = set(threading.enumerate())
    with pytest.raises(SettingRefusedError) as refused:
        ScopeSession.create(settings, simulate=True)
    assert refused.value.path == path
    assert refused.value.reason in ('out_of_range', 'wrong_kind')
    # Only the threads this create started count: a thread another test on
    # the same worker left running may end during this one.
    assert [t.name for t in threading.enumerate() if t not in before] == []


def test_a_container_of_the_wrong_kind_is_refused_as_the_argument():
    settings = complete_settings()
    settings['motion'] = [100]
    with pytest.raises(ArgumentRefusedError) as refused:
        ScopeSession.create(settings, simulate=True)
    assert (refused.value.reason, refused.value.argument, refused.value.kind) == (
        'wrong_kind',
        'settings.motion',
        'a dict',
    )


def _prepared_from_a_file(tmp_path, edit):
    """The settings the file's preparation makes of a current.json ``edit`` changed."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    current = json.loads(TEMPLATE.read_text())
    edit(current)
    (data_dir / 'current.json').write_text(json.dumps(current))
    prepared, _rejected = settings_init.prepare_settings(
        LOGGER, str(tmp_path), fall_back_to_template=False
    )
    return prepared


def test_the_door_on_the_files_preparation_changes_nothing(tmp_path):
    def edit(current):
        current['turret_objectives'] = {'1': '4x Oly', '2': '10x Oly', '3': None, '4': '40x Oly'}
        current['jpg_quality'] = 77
        current['live_folder'] = str(tmp_path)

    prepared = _prepared_from_a_file(tmp_path, edit)
    before = copy.deepcopy(prepared)
    again = settings_init.prepare_loaded(
        LOGGER, prepared, shipped_template(), str(tmp_path), source='settings', handed=True
    )
    assert again is prepared
    assert again == before


def test_a_prepared_files_turret_assignments_survive_create(tmp_path):
    def edit(current):
        current['microscope'] = 'LS850T'
        current['simulator_tier'] = 'fast'
        current['turret_objectives'] = {'1': '4x Oly', '2': '10x Oly', '3': None, '4': '40x Oly'}
        current['live_folder'] = str(tmp_path)

    session = ScopeSession.create(_prepared_from_a_file(tmp_path, edit), simulate=True)
    try:
        assert session.settings['turret_objectives'] == {
            1: '4x Oly',
            2: '10x Oly',
            3: None,
            4: '40x Oly',
        }
    finally:
        session.shutdown()
