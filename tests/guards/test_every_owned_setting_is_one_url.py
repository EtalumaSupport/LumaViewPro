# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting a member owns on the wire is one URL: its setter carries ``sets``.

``SETTINGS_WITH_A_MEMBER`` names the member that changes each owned setting.
A wire member there that is a setter marks the read it writes
(``@api(sets=...)``), so the server serves the two as one resource and a
client reads and writes the setting at one URL. An action that stores a
setting as one of its effects is not a setter of it, and stays a call. Its
subject is a class's members, the ones the server routes. A static read: no
session.
"""

from modules import settings_paths
from modules.api_surface import API, mark_of, sets_of
from modules.lumascope_api.camera import CameraAPI
from modules.scope_session import ScopeSession
from tests.ast_seams import MemberNames

_OWNERS = {'ScopeSession': ScopeSession, 'CameraAPI': CameraAPI}

# Saving the bookmark stores where the stage is, and confirming the
# objective records the answer to a question: each an action, read by
# ``bookmark`` and ``objective_question``.
_ACTIONS = MemberNames('modules/scope_session.py', {'save_bookmark', 'confirm_objective'})


def test_every_wire_setter_of_an_owned_setting_names_its_read():
    unmarked = []
    actions = set()
    for path, member in sorted(settings_paths.SETTINGS_WITH_A_MEMBER.items()):
        function = getattr(_OWNERS[member.owner], member.name)
        if mark_of(function) != API:
            continue
        if member.name in _ACTIONS:
            actions.add(member.name)
            assert sets_of(function) is None, f'{member.words} is an action, not a setter'
            continue
        if sets_of(function) is None:
            unmarked.append(f'{path}: {member.words}')
    assert unmarked == [], 'mark the read each writes, @api(sets=...):\n' + '\n'.join(unmarked)
    assert actions == _ACTIONS, f'actions no longer in the table: {sorted(_ACTIONS - actions)}'
