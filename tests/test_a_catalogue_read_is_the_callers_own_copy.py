# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No caller can change a published catalogue for the readers after it.

The scope reads its catalogues once, at bring-up, and every consumer reads
the scope's copy. The accessors handed out that copy itself, so a caller
that changed what it was given -- a plate's spacing, a model's axes, an
objective's focal length, a shipped setting -- changed it for every later
reader in the process, the scope included. A read hands out the caller's
own copy, or a record no caller can change.
"""

import dataclasses

import pytest

from tests.scope_fakes import build_scope


@pytest.fixture
def scope():
    return build_scope(simulate=True)


def test_a_plate_refuses_a_change(scope):
    plate = scope.wellplate_loader.get_plate('96 well microplate')
    with pytest.raises(dataclasses.FrozenInstanceError):
        plate.spacing_x_mm = 10.0
    assert scope.wellplate_loader.get_plate('96 well microplate').spacing_x_mm == 9


def test_the_scope_models_refuse_a_change(scope):
    models = scope.scope_models
    with pytest.raises(TypeError):
        models['LS850T'] = models['LS620']
    with pytest.raises(dataclasses.FrozenInstanceError):
        models['LS850T'].has_turret = False
    assert scope.scope_models['LS850T'].has_turret is True


def test_the_objectives_refuse_a_change(scope):
    objectives = scope.objectives
    with pytest.raises(TypeError):
        objectives['10x Oly'] = objectives['4x Oly']
    with pytest.raises(dataclasses.FrozenInstanceError):
        objectives['10x Oly'].focal_length_mm = 1.0
    assert scope.objectives['10x Oly'].focal_length_mm == 18.0


def test_a_changed_settings_template_leaves_the_next_read_unchanged(scope):
    template = scope.settings_template
    shipped = template['protocol']['labware']
    template['protocol']['labware'] = 'changed by a caller'
    assert scope.settings_template['protocol']['labware'] == shipped
