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

from _lumascope.modules.exceptions import InstallationFileError
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.objectives_loader import DEFAULT_PROPOSED_OBJECTIVE_ID, ObjectiveLoader

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


@pytest.mark.parametrize(
    ('loader', 'name'), [(WellPlateLoader, 'labware.json'), (ObjectiveLoader, 'objectives.json')]
)
def test_an_unreadable_catalogue_path_names_the_file(tmp_path, loader, name):
    (tmp_path / 'data' / name).mkdir(parents=True)

    with pytest.raises(InstallationFileError, match='cannot be read') as failure:
        loader(source_path=tmp_path)

    assert failure.value.file_path == tmp_path / 'data' / name


def _objectives_with(change) -> str:
    catalogue = _shipped('objectives.json')
    change(catalogue['4x Oly'])
    return json.dumps(catalogue)


@pytest.mark.parametrize(
    ('change', 'words'),
    [
        (lambda entry: entry.update(AF_rnge=50.0), "the undeclared key 'AF_rnge'"),
        (lambda entry: entry.pop('focal_length'), "no 'focal_length'"),
        (lambda entry: entry.update(DOF='5.8'), "a 'DOF' that is not a number"),
    ],
    ids=['undeclared-key', 'missing-field', 'not-a-number'],
)
def test_an_objective_entry_off_its_declared_shape_names_the_objective_and_key(
    tmp_path, change, words
):
    """Every declared key is required and no other is taken: a misspelt key
    would leave its field at a value nobody wrote."""
    root = _install(tmp_path, 'objectives.json', _objectives_with(change))

    with pytest.raises(InstallationFileError, match=f"objective '4x Oly' {words}"):
        ObjectiveLoader(source_path=root)


def _labware_with(change) -> str:
    catalogue = _shipped('labware.json')
    change(catalogue['Wellplate']['96 well microplate'])
    return json.dumps(catalogue)


@pytest.mark.parametrize(
    ('change', 'words'),
    [
        (lambda entry: entry['spacing'].pop('x'), "no 'spacing.x'"),
        (lambda entry: entry.update(rows='8'), "no whole-number 'rows'"),
        (lambda entry: entry['offset'].update(y=None), "a 'offset.y' that is not a number"),
    ],
    ids=['missing-key', 'rows-not-whole', 'not-a-number'],
)
def test_a_plate_entry_the_record_cannot_be_built_from_names_the_plate_and_key(
    tmp_path, change, words
):
    root = _install(tmp_path, 'labware.json', _labware_with(change))

    with pytest.raises(InstallationFileError, match=f"plate '96 well microplate' {words}"):
        WellPlateLoader(source_path=root)


def test_a_plate_value_out_of_its_range_names_the_plate_key_and_value(tmp_path):
    """A plate is located by dividing by its spacing and drawn by dividing by
    its dimensions, so a zero there loaded and then raised ZeroDivisionError
    on New Protocol, every capture and the stage view. A negative count or a
    diameter that is neither -1 (not stated) nor positive locates nothing.
    Each is refused where the file is read."""
    refused = {
        'a zero spacing': (lambda e: e['spacing'].update(x=0), "a 'spacing.x' of 0, not above 0"),
        'a negative spacing': (
            lambda e: e['spacing'].update(y=-9),
            "a 'spacing.y' of -9, not above 0",
        ),
        'a zero dimension': (
            lambda e: e['dimensions'].update(x=0),
            "a 'dimensions.x' of 0, not above 0",
        ),
        'a negative count': (lambda e: e.update(rows=-1), "a 'rows' of -1, not 0 or more"),
        'a zero diameter': (
            lambda e: e.update(diameter=0),
            "a 'diameter' of 0, neither -1 (not stated) nor above 0",
        ),
    }
    wrong = {}
    for case, (change, words) in refused.items():
        root = tmp_path / case.replace(' ', '_')
        root.mkdir()
        _install(root, 'labware.json', _labware_with(change))
        try:
            WellPlateLoader(source_path=root)
            wrong[case] = 'loaded'
        except InstallationFileError as e:
            if f"plate '96 well microplate' {words}" not in str(e):
                wrong[case] = str(e)

    assert wrong == {}


def test_a_plate_keyed_by_a_retired_name_is_refused(tmp_path):
    """A retired spelling is translated before every lookup: a plate keyed
    'Center Dish' was listed, but its name reached the Center Plate."""
    catalogue = _shipped('labware.json')
    catalogue['Wellplate']['Center Dish'] = dict(catalogue['Wellplate']['96 well microplate'])
    root = _install(tmp_path, 'labware.json', json.dumps(catalogue))

    with pytest.raises(
        InstallationFileError, match="plate 'Center Dish', a retired name read as 'Center Plate'"
    ):
        WellPlateLoader(source_path=root)


def test_every_shipped_plate_loads():
    """The Blank (0 x 0, spacing and diameter 0.001) and the Center Plate
    (1 x 1, spacing 70, diameter -1) are the shipped edges of every range."""
    plates = WellPlateLoader().plates

    assert set(plates) == set(_shipped('labware.json')['Wellplate'])
