# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The active-objective pair (id + resolved info) must never tear.

The bug class: a setter that assigns one half of a paired state before
the operation that can fail, so a failed set leaves the id describing a
different objective than the info. Every consumer that reads one half
and trusts the other (metadata writers, optics logging) then emits a
claim about the wrong objective.
"""

import pytest

from modules.exceptions import ConfigError
from modules.lumascope_api import Lumascope
from modules.lumascope_api.runtime_state import RuntimeState


def _make_runtime_state() -> RuntimeState:
    scope = Lumascope.__new__(Lumascope)
    return RuntimeState(scope)


def _any_real_objective_id(state: RuntimeState) -> str:
    objectives = state.get_available_objectives()
    assert objectives, 'objective catalogue unexpectedly empty'
    return objectives[0]


def test_failed_set_objective_leaves_state_coherent():
    state = _make_runtime_state()
    good_id = _any_real_objective_id(state)
    state.set_objective(good_id)

    with pytest.raises(ConfigError):
        state.set_objective('zz-no-such-objective')

    assert state.get_current_objective_id() == good_id
    objective = state.get_current_objective()
    assert objective is not None
    assert objective == state.get_objective_info(objective_id=good_id)


def test_failed_set_objective_from_unset_stays_unset():
    state = _make_runtime_state()

    with pytest.raises(ConfigError):
        state.set_objective('zz-no-such-objective')

    assert state.get_current_objective_id() is None
    assert state.get_current_objective() is None
