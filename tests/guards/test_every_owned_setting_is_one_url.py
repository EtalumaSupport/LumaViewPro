# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting a member owns on the wire has one read, which the table or its setter names.

``SETTINGS_WITH_A_MEMBER`` names the member that changes each owned setting.
A wire member there that is a setter marks the read it writes
(``@api(sets=...)``), so the server serves the two as one resource and a
client reads and writes the setting at one URL. An action that stores a
setting as one of its effects is not a setter of it, and stays a call: its
row names the setting's read (``SettingMember.read``), so ``setting``
refuses the setting for that read. A row names its read one way, never
both. Its subject is a class's members, the ones the server routes. A static
read: no session.
"""

from _lumascope.modules import settings_paths
from _lumascope.modules.api_surface import API, mark_of, sets_of
from _lumascope.modules.lumascope_api.camera import CameraAPI
from _lumascope.modules.scope_session import ScopeSession

_OWNERS = {'ScopeSession': ScopeSession, 'CameraAPI': CameraAPI}


def _unread(table: dict[str, settings_paths.SettingMember]) -> list[str]:
    # Each wire writer's setting names exactly one read, which is published;
    # an in-process writer's setting (one the host moves) names none, as
    # setting reads it.
    offenders = []
    for path, member in sorted(table.items()):
        owner = _OWNERS[member.owner]
        writer = getattr(owner, member.name)
        setter_read = sets_of(writer)
        if mark_of(writer) != API:
            if member.read is not None:
                offenders.append(f'{path}: {member.words} is in-process; setting is its read')
            continue
        if (setter_read is None) == (member.read is None):
            offenders.append(
                f'{path}: {member.words} names its read '
                + ('twice, in sets= and the table' if setter_read else 'nowhere')
            )
            continue
        read = setter_read or member.read
        if mark_of(getattr(owner, read, None)) != API:
            offenders.append(f'{path}: {member.path_of(read)} is not a published read')
    return offenders


def test_every_wire_writer_of_an_owned_setting_names_its_read():
    assert _unread(settings_paths.SETTINGS_WITH_A_MEMBER) == []


def test_a_row_whose_action_names_no_read_is_named():
    # A planted action row with no read, and one naming a read that is not
    # published: each named, so a new action cannot leave setting
    # answering its setting.
    planted = {
        'bookmark': settings_paths.SettingMember('ScopeSession', 'save_bookmark'),
        'objective_confirmed': settings_paths.SettingMember(
            'ScopeSession', 'confirm_objective', 'get_channel_configs'
        ),
    }
    assert _unread(planted) == [
        'bookmark: ScopeSession.save_bookmark names its read nowhere',
        'objective_confirmed: get_channel_configs is not a published read',
    ]
