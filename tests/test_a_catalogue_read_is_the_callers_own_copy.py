# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every published catalogue read hands out the caller's own copy.

The scope reads its catalogues once, at bring-up, and every consumer reads
the scope's copy. The accessors handed out that copy itself, so a caller
that changed what it was given -- a plate's spacing, a model's axes, an
objective's field of view, a shipped setting -- changed it for every later
reader in the process, the scope included.
"""

import pytest

from tests.scope_fakes import build_scope


@pytest.fixture
def scope():
    return build_scope(simulate=True)


def test_a_changed_plate_leaves_the_next_read_unchanged(scope):
    plate = scope.wellplate_loader.get_plate('96 well microplate')
    shipped = plate.config['spacing']['x']
    plate.config['spacing']['x'] = shipped + 1
    assert scope.wellplate_loader.get_plate('96 well microplate').config['spacing']['x'] == shipped


def test_a_changed_scope_model_leaves_the_next_read_unchanged(scope):
    model = next(iter(scope.scope_models))
    entry = scope.scope_models[model]
    shipped = dict(entry)
    entry['changed by a caller'] = True
    assert dict(scope.scope_models[model]) == shipped


def test_a_changed_objectives_table_leaves_the_next_read_unchanged(scope):
    table = scope.objective_helper.get_objectives_dataframe()
    shipped = table.copy(deep=True)
    table.iloc[0, 0] = 'changed by a caller'
    assert scope.objective_helper.get_objectives_dataframe().equals(shipped)


def test_a_changed_settings_template_leaves_the_next_read_unchanged(scope):
    template = scope.settings_template
    shipped = template['protocol']['labware']
    template['protocol']['labware'] = 'changed by a caller'
    assert scope.settings_template['protocol']['labware'] == shipped
