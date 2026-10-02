# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope reads the labware and objective catalogues once, and everything reads its copy.

The catalogues had no owner: the session built one pair from its data folder,
while the scope's runtime state, autofocus and the run each built their own
from the installation's folder. A session started on any other folder then
disagreed with its own scope about which objectives and plates exist. A
catalogue that failed to load was popped up, left as None, and raised again
later as bad stored settings.

Now the scope reads both, once, from the folder it was started on, before
anything starts. The session, runtime state, autofocus and the run read that
copy. A file that cannot be used stops the bring-up with one error naming it,
and nothing is posted or left running.
"""

import json
import logging
import pathlib
import shutil
import threading
import time

import pytest

from modules.exceptions import InstallationFileError
from modules.lumascope_api import Lumascope
from modules.notification_center import Severity, notifications
from modules.path_utils import get_source_root
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTRA_OBJECTIVE = 'Demo 100x only-in-this-folder'
EXTRA_PLATE = 'Demo plate only-in-this-folder'


def _data_folder(tmp_path: pathlib.Path) -> pathlib.Path:
    shutil.copytree(REPO_ROOT / 'data', tmp_path / 'data')
    return tmp_path


def _add_extra_entries(root: pathlib.Path) -> None:
    objectives_file = root / 'data' / 'objectives.json'
    objectives = json.loads(objectives_file.read_text())
    objectives[EXTRA_OBJECTIVE] = dict(objectives['4x Oly'])
    objectives_file.write_text(json.dumps(objectives))

    labware_file = root / 'data' / 'labware.json'
    labware = json.loads(labware_file.read_text())
    labware['Wellplate'][EXTRA_PLATE] = dict(labware['Wellplate']['96 well microplate'])
    labware_file.write_text(json.dumps(labware))


def _shut(session: ScopeSession) -> None:
    session.shutdown()
    session.scope.disconnect()


def test_every_part_of_the_session_reads_the_scopes_catalogues(tmp_path):
    root = _data_folder(tmp_path)
    _add_extra_entries(root)
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)), source_path=root, simulate=True
    )
    try:
        scope = session.scope
        assert EXTRA_OBJECTIVE in session.objective_helper.get_objectives_list()
        assert EXTRA_OBJECTIVE in scope.runtime_state.get_available_objectives()
        assert EXTRA_PLATE in session.wellplate_loader.get_plate_list()
        # One object each, the scope's, wherever the session reaches one.
        assert session.objective_helper is scope.objective_helper
        assert session.wellplate_loader is scope.wellplate_loader
        assert session.autofocus_runner._scope is scope
        assert session.sequenced_capture_runner._scope is scope
        assert session.select_labware(EXTRA_PLATE) is not None
        assert session.source_path == str(root)
        assert scope.protocols.tiling_configs_path() == root / 'data' / 'tiling.json'
    finally:
        _shut(session)


def test_a_protocol_is_built_and_validated_against_the_scopes_catalogues(tmp_path):
    import modules.config_helpers as config_helpers

    root = _data_folder(tmp_path)
    _add_extra_entries(root)
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)), source_path=root, simulate=True
    )
    try:
        config = config_helpers.get_sequenced_capture_config_from_settings(
            session.settings,
            objective_helper=session.objective_helper,
            wellplate_loader=session.wellplate_loader,
            current_z=session.get_current_plate_position()['z'],
            tiling='1x1',
            use_zstacking=False,
        )
        config['layer_configs'] = {'BF': config['layer_configs']['BF']}
        config['layer_configs']['BF']['acquire'] = 'image'
        config['objective_id'] = EXTRA_OBJECTIVE
        config['labware_id'] = EXTRA_PLATE

        protocol = session.scope.protocols.create_protocol(input_config=config)

        assert set(protocol.steps()['Objective']) == {EXTRA_OBJECTIVE}
        assert protocol.validate_steps(session.objective_helper) == []
    finally:
        _shut(session)


def _live_threads() -> set[threading.Thread]:
    # By object, not name: a same-named lane still winding down from an
    # earlier case would hide a new one leaked here.
    return {t for t in threading.enumerate() if t.is_alive()}


def _settle(before: set[threading.Thread], deadline_s: float = 2.0) -> set[str]:
    end = time.monotonic() + deadline_s
    extra = _live_threads() - before
    while extra and time.monotonic() < end:
        time.sleep(0.02)
        extra = _live_threads() - before
    return {t.name for t in extra}


FAILURES = [
    ('objectives.json', None),
    ('objectives.json', '{not json'),
    ('labware.json', None),
    ('labware.json', '{not json'),
]
FAILURE_IDS = ['objectives-missing', 'objectives-corrupt', 'labware-missing', 'labware-corrupt']


@pytest.mark.parametrize('simulate', [True, False], ids=['simulated', 'no-hardware'])
@pytest.mark.parametrize(('name', 'content'), FAILURES, ids=FAILURE_IDS)
def test_an_unusable_catalogue_stops_the_bring_up_before_anything_starts(
    tmp_path, caplog, simulate, name, content
):
    root = _data_folder(tmp_path)
    if content is None:
        (root / 'data' / name).unlink()
    else:
        (root / 'data' / name).write_text(content)

    posted = []

    def listener(notification):
        posted.append(notification)

    notifications.add_listener(listener, min_severity=Severity.DEBUG)
    before = _live_threads()
    try:
        with caplog.at_level(logging.DEBUG), pytest.raises(InstallationFileError) as failure:
            ScopeSession.create(
                complete_settings(live_folder=str(tmp_path)),
                source_path=root,
                simulate=simulate,
                warn_pre_release=False,
            )
    finally:
        notifications.remove_listener(listener)

    assert failure.value.file_path == root / 'data' / name
    assert posted == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert _settle(before) == set(), 'a thread was left running by a refused bring-up'


class TestTheSessionsFolderIsTheScopes:
    def test_a_folder_beside_a_callers_scope_is_refused(self, tmp_path):
        scope = Lumascope(simulate=True, warn_pre_release=False, source_path=REPO_ROOT)
        try:
            with pytest.raises(ValueError, match='source_path is refused beside a scope'):
                ScopeSession.create(complete_settings(), source_path=tmp_path, scope=scope)
        finally:
            scope.disconnect()

    def test_a_callers_scope_brings_its_folder(self, tmp_path):
        root = _data_folder(tmp_path)
        scope = Lumascope(simulate=True, warn_pre_release=False, source_path=root)
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), scope=scope)
        try:
            assert session.source_path == str(root)
            assert session.objective_helper is scope.objective_helper
        finally:
            session.shutdown()
            scope.disconnect()

    def test_no_folder_is_the_installations_own(self, tmp_path):
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
        )
        try:
            assert session.source_path == str(get_source_root(None))
        finally:
            _shut(session)
