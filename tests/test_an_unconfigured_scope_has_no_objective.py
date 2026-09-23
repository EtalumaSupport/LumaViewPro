# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope nobody has configured cannot say which objective is in the light path.

Whether the objective is the turret slot's assignment or a stored selection
depends on whether the scope has a turret, and that answer is recorded at
bring-up (``Lumascope.initialize``). Before it, the answer used to default to
"no turret": a bare turret scope accepted ``set_objective`` and then reported
that stored objective as the one in the light path, with the turret in no
known slot -- the wrong scale in every image named by it. Until bring-up
answers, the objective is unknown and cannot be set.
"""

import pytest

import modules.lumascope_api as lumascope_api
from modules.exceptions import ConfigError, ObjectiveUnknownError


@pytest.fixture
def bare_turret_scope():
    scope = lumascope_api.Lumascope(
        simulate=True,
        sim_model='LS850T',
        register_atexit=False,
        register_metrics=False,
        warn_pre_release=False,
    )
    assert scope.capabilities.has_turret
    yield scope
    scope.disconnect()


def test_the_objective_is_unknown_and_says_why(bare_turret_scope):
    state = bare_turret_scope.runtime_state
    with pytest.raises(ObjectiveUnknownError) as excinfo:
        state.resolve_current_objective()
    assert excinfo.value.reason == 'turret_undecided'
    assert state.get_current_objective_id() is None
    assert state.get_current_objective() is None


def test_an_objective_cannot_be_set(bare_turret_scope):
    state = bare_turret_scope.runtime_state
    with pytest.raises(ConfigError, match='initialize'):
        state.set_objective('10x Oly')
    assert state.get_current_objective_id() is None


def test_whether_it_has_a_turret_is_refused_rather_than_guessed(bare_turret_scope):
    with pytest.raises(ConfigError, match='initialize'):
        bare_turret_scope.runtime_state.is_turreted()
