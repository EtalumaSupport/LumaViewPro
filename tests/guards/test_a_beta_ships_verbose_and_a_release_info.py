# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped settings name VERBOSE while the version is a beta, and INFO on a release.

The shipped template is the one store of the level an install starts with:
a beta's field logs carry the routine hardware traffic, a release's carry
events and anomalies. The flip is made by hand at the release, so this fails
the build until the template agrees with ``version.txt``.
"""

import json

from modules.path_utils import read_version
from tests.ast_seams import REPO_ROOT


def test_the_shipped_level_is_the_versions():
    version, _ = read_version()
    shipped = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text(encoding='utf-8'))
    level = shipped['logging']['default']['level']
    expected = 'VERBOSE' if 'beta' in version else 'INFO'
    assert level == expected, (
        f'data/settings.json ships logging.default.level {level!r} with version.txt '
        f'{version!r}: a beta ships VERBOSE, a release INFO'
    )
