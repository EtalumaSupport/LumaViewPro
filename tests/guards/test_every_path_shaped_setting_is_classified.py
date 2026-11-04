# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting that names a file, a folder, a port, a host or a key is classified.

``update_settings`` is the one settings write for every caller, a REST
caller included. A setting that names where the scope reads or writes, or
how it is reached, either changes through its own Session member, which
checks it, or is read only from the installation's settings file; stored
as any caller gave it, it would let that caller send the next start's
files, profile or server anywhere. A new such leaf in the shipped template
is refused here until it is classified.
"""

from __future__ import annotations

import re

from modules import settings_paths
from tests.settings_fixtures import shipped_template

_SHAPED = re.compile(r'(_dir|_folder|_path|filepath|_port|host|api_key)$|^mode$')


def _leaves(block: dict, prefix: str = ''):
    for key, value in block.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from _leaves(value, f'{path}.')
        else:
            yield path, key


def _unclassified(template: dict) -> list[str]:
    return [
        path
        for path, leaf in _leaves(template)
        if _SHAPED.search(leaf)
        and settings_paths.member_for(path) is None
        and not settings_paths.is_installation_only(path)
    ]


def test_every_path_shaped_setting_in_the_template_is_classified():
    template = shipped_template()
    unclassified = _unclassified(template)
    assert unclassified == [], (
        'these settings name a file, folder, port, host or key, yet any caller may '
        'write them: give each a Session member or add it to '
        f'settings_paths.INSTALLATION_ONLY: {unclassified}'
    )


def test_a_new_unclassified_folder_setting_is_caught():
    template = shipped_template()
    template['video']['export_dir'] = None
    assert _unclassified(template) == ['video.export_dir']
