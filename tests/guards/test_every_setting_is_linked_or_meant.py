# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting a new install holds says what it means, in one place.

The reference's settings table links a setting a member writes to that
member, whose entry is what it means, and gives every other setting the
line ``settings_paths.SETTING_MEANINGS`` holds for it. A setting with
neither renders a row a caller cannot read; one with both says it twice,
in two places that drift; a meaning whose setting is gone describes a
setting nobody can write. The paths are the ones a new install holds, as
the application reads the template.
"""

from __future__ import annotations

from _lumascope.modules import settings_paths
from tests.settings_fixtures import shipped_template


def test_every_setting_is_linked_to_its_member_or_meant_and_no_meaning_outlives_its_setting():
    paths = [path for path, _ in settings_paths._leaf_paths(shipped_template())]
    meant = settings_paths.SETTING_MEANINGS
    unread = [p for p in paths if settings_paths.member_for(p) is None and p not in meant]
    twice = [p for p in paths if settings_paths.member_for(p) is not None and p in meant]
    orphaned = sorted(set(meant) - set(paths))
    assert not (unread or twice or orphaned), (
        f'A setting is linked to its member or meant in SETTING_MEANINGS, never both: '
        f'neither {unread}; both {twice}; meant with no such setting {orphaned}'
    )
