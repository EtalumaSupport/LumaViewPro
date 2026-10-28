# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A turret scope nobody has configured cannot say which objective is in the light path.

Whether the objective is the turret slot's assignment or a stored selection
depends on whether the scope has a turret, which the scope's capabilities
answer from its construction. A bare turret scope once defaulted to "no
turret" until bring-up, accepted ``set_objective`` and then reported that
stored objective as the one in the light path, with the turret in no known
slot -- the wrong scale in every image named by it. Before bring-up its
turret is in no known slot, so its objective is unknown.
"""

import pytest

from modules.exceptions import ObjectiveUnknownError
from tests.scope_fakes import build_scope


@pytest.fixture
def bare_turret_scope():
    scope = build_scope(
        simulate=True,
        sim_model='LS850T',
        register_atexit=False,
        warn_pre_release=False,
    )
    assert scope.capabilities.has_turret
    yield scope
    scope.disconnect()


def test_the_objective_is_unknown_and_says_why(bare_turret_scope):
    state = bare_turret_scope.runtime_state
    with pytest.raises(ObjectiveUnknownError) as excinfo:
        state.resolve_current_objective()
    assert excinfo.value.reason == 'slot_unknown'
    assert state.get_current_objective_id() is None
    assert state.get_current_objective() is None
