# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A headless composite, autofocus or z-stack builds with the objective in
the light path, never the stored one.

On a turret scope the stored ``objective_id`` is a leftover of whatever was
selected last; the glass in the light path is the slot's assignment. These
three runner members assemble their protocol from settings, so settings read
raw would stamp the leftover's name and scale into every capture. They read
the capture snapshot, which carries the active objective.
"""

import pytest

from tests.scope_fakes import TEST_TURRET_OBJECTIVES
from tests.test_composite_run_e2e import headless_settings, open_composite_session


class _BuiltError(Exception):
    pass


@pytest.mark.parametrize(
    'start',
    [
        lambda runner: runner.start_composite(),
        lambda runner: runner.run_autofocus(layer='BF'),
        lambda runner: runner.run_zstack(layer='BF'),
    ],
    ids=['composite', 'autofocus', 'zstack'],
)
def test_the_protocol_names_the_slots_objective(tmp_path, monkeypatch, start):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        slot = session.scope.motion.get_turret_slot()
        in_light_path = TEST_TURRET_OBJECTIVES[slot]
        stale = next(
            obj for obj in TEST_TURRET_OBJECTIVES.values() if obj not in (None, in_light_path)
        )
        session.settings['objective_id'] = stale
        built = []

        def _record(input_config):
            built.append(input_config)
            raise _BuiltError

        monkeypatch.setattr(session.scope.protocols, 'create_protocol', _record)

        with pytest.raises(_BuiltError):
            start(runner)

    assert built[0]['objective_id'] == in_light_path
