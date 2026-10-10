# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A new protocol names only the objective in the light path.

``session.new_protocol`` stamps every step with the objective the scope
has in its light path, which the scope can always address, so a new
protocol is admissible by construction and nothing asks the objective
rule of it afterwards. While that objective is unknown -- a turret slot
with no objective assigned -- the build is refused and nothing is built.
"""

import pytest

from modules.exceptions import ObjectiveUnknownError
from tests.test_composite_run_e2e import headless_settings, open_composite_session


def _turret_session(tmp_path):
    settings = headless_settings(tmp_path)
    settings['microscope'] = 'LS850T'
    return open_composite_session(settings)


def test_every_step_names_the_objective_in_the_light_path(tmp_path):
    with _turret_session(tmp_path) as (session, _runner):
        assert session.scope.capabilities.has_turret
        current, _ = session.scope.runtime_state.resolve_current_objective()

        protocol = session.new_protocol()

        assert protocol.num_steps() > 0
        assert set(protocol.steps()['Objective']) == {current}


def test_a_slot_with_no_objective_builds_nothing(tmp_path):
    with _turret_session(tmp_path) as (session, _runner):
        session.clear_current_turret_objective()

        with pytest.raises(ObjectiveUnknownError) as refusal:
            session.new_protocol()

        assert refusal.value.reason == 'slot_unassigned'
