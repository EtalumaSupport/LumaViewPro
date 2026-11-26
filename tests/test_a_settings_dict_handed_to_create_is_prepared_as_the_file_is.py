# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A settings dict handed to ``create`` is prepared as the file's is, by the one door.

Before the door, a dict from a caller's hand reached the session as it
was: ``create`` refused it for a key it lacked, and every reader below
hedged with a second value for the keys it might lack, dead on the
prepared file and live only on that hand dict. Now ``create`` runs the
file's own preparation on it (``settings_init.prepare_loaded``): every
template key is present at every depth, a value the writer refuses is
refused as a write is, an inverted auto-gain range is refused as an arm
refuses it, a container of the wrong kind is refused as the argument,
and a dict the file's preparation already made usable goes through
unchanged, its stored turret assignments included.
"""

import copy
import json
import logging
import pathlib
import shutil
import threading

import pytest

from _lumascope.modules import settings_init
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    SettingRefusedError,
    StoredSettingReplacedNotice,
)
from _lumascope.modules.lumascope_api._constants import ACCELERATION_PCT_MAX
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.settings_paths import _leaf_paths
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
            if path == 'protocol.filepath':
                # The shipped relative path is the data root's, made absolute.
                assert shipped == './data/new_default_protocol.tsv'
                shipped = str(
                    pathlib.Path(session.scope.source_path, 'data', 'new_default_protocol.tsv')
                )
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
    ('path', 'value', 'reason'),
    [
        ('jpg_quality', 500, 'out_of_range'),
        ('motion.acceleration_max_pct', ACCELERATION_PCT_MAX + 400, 'out_of_range'),
        # A number setting given anything but a finite number.
        ('motion.acceleration_max_pct', '50', 'not_a_number'),
        ('motion.acceleration_max_pct', None, 'not_a_number'),
        ('zstack.range', 10**400, 'not_a_number'),
        ('BF.focus', 'abc', 'not_a_number'),
        # Another kind of setting given a value of another kind.
        ('show_tooltips', 'yes', 'wrong_kind'),
        # A closed set's leaf: a value outside the set is not one it offers.
        ('image_mode', '16bit', 'not_offered'),
    ],
)
def test_a_value_the_writer_refuses_is_refused_as_a_write_is_and_nothing_is_built(
    path, value, reason
):
    settings = complete_settings()
    block, _, leaf = path.rpartition('.')
    (settings[block] if block else settings)[leaf] = value
    before = set(threading.enumerate())
    with pytest.raises(SettingRefusedError) as refused:
        ScopeSession.create(settings, simulate=True)
    assert refused.value.path == path
    assert refused.value.reason == reason
    # Only the threads this create started count: a thread another test on
    # the same worker left running may end during this one.
    assert [t.name for t in threading.enumerate() if t not in before] == []


def test_an_inverted_auto_gain_range_is_refused_as_an_arm_refuses_it_and_nothing_is_built():
    """A handed dict whose auto-gain minimum gain is above its maximum is
    refused at the call, by the arm's own rule, before anything runs on it.

    Each bound alone is in range, so only the pair rule refuses it. Every
    layer's auto-gain is off, so no arm at bring-up meets the pair: without
    the handed judge's pair check the scope comes up holding it.
    """
    settings = complete_settings()
    auto_gain_paths = [path for path, _ in _leaf_paths(settings) if path.endswith('.auto_gain')]
    assert auto_gain_paths
    assert [path for path in auto_gain_paths if _value_at(settings, path)] == []
    settings['protocol']['autogain'].update(min_gain_db=15.0, max_gain_db=5.0)
    before = set(threading.enumerate())
    with pytest.raises(ArgumentRefusedError) as refused:
        ScopeSession.create(settings, simulate=True)
    assert (refused.value.reason, refused.value.argument, refused.value.value) == (
        'gain_range_inverted',
        'protocol.autogain',
        {'min_gain_db': 15.0, 'max_gain_db': 5.0},
    )
    assert [t.name for t in threading.enumerate() if t not in before] == []


def test_a_container_of_the_wrong_kind_is_refused_as_the_argument():
    settings = complete_settings()
    settings['motion'] = [100]
    with pytest.raises(ArgumentRefusedError) as refused:
        ScopeSession.create(settings, simulate=True)
    assert (refused.value.reason, refused.value.argument, refused.value.takes) == (
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


def test_the_dict_handed_to_create_stays_the_callers(tmp_path):
    """``create`` takes its own copy of the dict it is handed: preparing it
    changes nothing in the caller's dict, and a write into that dict after
    ``create`` changes no setting, so every write reaches the store through
    ``update_settings`` and its range, control-session and save."""
    handed = complete_settings(live_folder=str(tmp_path), microscope='LS850', simulator_tier='fast')
    handed.pop('video')
    as_handed = copy.deepcopy(handed)
    session = ScopeSession.create(handed, simulate=True)
    try:
        assert handed == as_handed
        stored = session.channel_settings('BF').gain_db
        handed['BF']['gain_db'] = stored + 1.0
        handed['live_folder'] = str(tmp_path / 'elsewhere')
        assert session.channel_settings('BF').gain_db == stored
        assert session.setting('live_folder') != handed['live_folder']
    finally:
        session.shutdown()


def test_a_count_written_as_a_whole_float_is_stored_as_an_int_and_captured(tmp_path):
    """A count the store's rule takes as ``2.0`` (JSON, or a MATLAB double) is stored as 2.

    The rule is whole by value; every reader after the store takes an
    integer, so a capture asked for the stored sum sums that many frames
    rather than refusing a float count.
    """
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), BF={'sum': 2.0}), simulate=True
    )
    try:
        stored = session.settings['BF']['sum']
        assert (stored, type(stored)) == (2, int)
        record = session.capture_still(
            frames_summed=stored, force_to_8bit=False, accept_dark=True, timeout_s=2.0
        ).record
        assert record.frames_summed == 2
    finally:
        session.shutdown()
