# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective the question proposes by default is one the catalogue has.

A default the catalogue lacks would be offered in a dropdown that cannot
hold it; the catalogue refuses to load instead, naming the missing id.
"""

import json
import pathlib

import pytest

from modules.exceptions import ConfigError
from modules.objectives_loader import DEFAULT_PROPOSED_OBJECTIVE_ID, ObjectiveLoader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_shipped_catalogue_has_the_default():
    assert DEFAULT_PROPOSED_OBJECTIVE_ID in ObjectiveLoader().get_objectives_list()


def test_a_catalogue_without_the_default_refuses_to_load(tmp_path):
    catalogue = json.loads((REPO_ROOT / 'data' / 'objectives.json').read_text())
    del catalogue[DEFAULT_PROPOSED_OBJECTIVE_ID]
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'objectives.json').write_text(json.dumps(catalogue))

    with pytest.raises(ConfigError, match=DEFAULT_PROPOSED_OBJECTIVE_ID):
        ObjectiveLoader(source_path=tmp_path)
