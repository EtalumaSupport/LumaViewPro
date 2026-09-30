# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A catalogue file the installation ships fails as one error naming the file.

labware.json and objectives.json are part of the installation, not the
user's settings. When one is missing, unreadable or not the shape its loader
needs, the loader raises one ``InstallationFileError`` naming the file and
its folder. It logs nothing itself: whoever catches the error logs it once.
The error is deliberately not a ``ConfigError``, because a ``ConfigError``
reads as bad stored settings; the GUI answered one by replacing the user's
settings with the template and locking saves, and then failed again on the
same missing file.
"""

import json
import logging
import pathlib

import pytest

from modules.exceptions import ConfigError, InstallationFileError
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import DEFAULT_PROPOSED_OBJECTIVE_ID, ObjectiveLoader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _shipped(name: str) -> dict:
    return json.loads((REPO_ROOT / 'data' / name).read_text())


def _install(tmp_path: pathlib.Path, name: str, content: str | None) -> pathlib.Path:
    (tmp_path / 'data').mkdir()
    if content is not None:
        (tmp_path / 'data' / name).write_text(content)
    return tmp_path


def _objectives_without_default() -> str:
    catalogue = _shipped('objectives.json')
    del catalogue[DEFAULT_PROPOSED_OBJECTIVE_ID]
    return json.dumps(catalogue)


def _objectives_with_colliding_short_names() -> str:
    catalogue = _shipped('objectives.json')
    catalogue['20x collar'] = dict(catalogue[DEFAULT_PROPOSED_OBJECTIVE_ID])
    return json.dumps(catalogue)


CASES = [
    (WellPlateLoader, 'labware.json', None),
    (WellPlateLoader, 'labware.json', '{not json'),
    (WellPlateLoader, 'labware.json', '[]'),
    (ObjectiveLoader, 'objectives.json', None),
    (ObjectiveLoader, 'objectives.json', '{not json'),
    (ObjectiveLoader, 'objectives.json', '[]'),
    (ObjectiveLoader, 'objectives.json', json.dumps({'4x Oly': 'not an entry'})),
    (ObjectiveLoader, 'objectives.json', _objectives_without_default()),
    (ObjectiveLoader, 'objectives.json', _objectives_with_colliding_short_names()),
]
IDS = [
    'labware-missing',
    'labware-corrupt',
    'labware-not-a-dict',
    'objectives-missing',
    'objectives-corrupt',
    'objectives-not-a-dict',
    'objectives-entry-not-a-dict',
    'objectives-without-the-default',
    'objectives-colliding-short-names',
]


@pytest.mark.parametrize(('loader', 'name', 'content'), CASES, ids=IDS)
def test_the_loader_raises_one_error_naming_the_file_and_folder(
    tmp_path, caplog, loader, name, content
):
    root = _install(tmp_path, name, content)

    with caplog.at_level(logging.DEBUG), pytest.raises(InstallationFileError) as failure:
        loader(source_path=root)

    assert name in str(failure.value)
    assert str(root / 'data') in str(failure.value)
    assert failure.value.file_path == root / 'data' / name
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        'the loader logged the failure itself; the catcher logs it once'
    )


def test_the_error_is_not_read_as_bad_settings():
    assert not issubclass(InstallationFileError, ConfigError)
