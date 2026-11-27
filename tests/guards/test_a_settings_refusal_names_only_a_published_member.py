# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A settings refusal names only a member the API publishes.

A setting a member owns is refused at the session's read and write doors,
and the refusal names the member to call instead, or the one that moves
the setting (``member``): a caller acts on that name. A member the API does
not publish is a name no caller can reach, in-process or on the wire, so a
refusal names it to nobody (``protocol.filepath``,
``motion.acceleration_max_pct``) and its words say only the application
hosting the scope changes the setting.

One test over the table of owned settings, ``SETTINGS_WITH_A_MEMBER``:
every path is read and written once, and every member a refusal names is
listed in the failure.
"""

from __future__ import annotations

from _lumascope.modules import api_surface, settings_paths
from _lumascope.modules.exceptions import ArgumentRefusedError, SettingRefusedError
from _lumascope.modules.lumascope_api.camera import CameraAPI
from _lumascope.modules.scope_session import ScopeSession


def _published(path: str) -> bool:
    """Whether the member at ``path`` from the session (``scope.camera.<name>`` or ``<name>``) is published."""
    owner, _, name = path.rpartition('.')
    member = getattr(CameraAPI if owner == 'scope.camera' else ScopeSession, name, None)
    return member is not None and api_surface.mark_of(member) is not None


def test_every_member_a_settings_refusal_names_is_published(sim_session):
    named = {}
    for owned in settings_paths.SETTINGS_WITH_A_MEMBER:
        path = owned.replace('*', 'BF')
        for door, ask, args in (
            ('setting', sim_session.setting, (path,)),
            ('update_settings', sim_session.update_settings, (path, 0)),
        ):
            try:
                ask(*args)
            except (ArgumentRefusedError, SettingRefusedError) as refused:
                if refused.member is not None:
                    named[f'{door}({path!r}): {refused.reason}'] = refused.member

    unpublished = {where: member for where, member in named.items() if not _published(member)}

    # The instrument reports presence: the owned settings' members are named.
    assert named["update_settings('binning'): has_member"] == 'scope.camera.set_binning'
    assert not unpublished, f'a refusal names a member the API does not publish: {unpublished}'
