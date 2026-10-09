# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored log level that names a level in any case is applied.

``logging.default.level`` holds a level's name in any case (Eric, 2026-10-09,
#832): the settings writer and the load take ``'debug'`` as they take
``'DEBUG'``. The reader turned the name into a level with
``logging.getLevelName``, which answers ``'Level debug'`` for a lowercase
name, so ``setLevel`` raised, the traceback was logged and the template's
level applied instead of the one stored. The reader upper-cases the name.
"""

import json
import logging
import shutil

import pytest

from modules import app_config
from tests.ast_seams import REPO_ROOT

TEMPLATE = REPO_ROOT / 'data' / 'settings.json'


@pytest.fixture
def restored_level():
    before = app_config.logger.level
    yield
    app_config.logger.setLevel(before)


@pytest.mark.parametrize(
    ('stored', 'applied'), [('debug', logging.DEBUG), ('Error', logging.ERROR)]
)
def test_the_stored_level_is_applied_whatever_its_case(tmp_path, restored_level, stored, applied):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    current = json.loads(TEMPLATE.read_text())
    current['logging']['default']['level'] = stored
    (data_dir / 'current.json').write_text(json.dumps(current))

    app_config.load_log_level(str(tmp_path))

    assert app_config.logger.level == applied
