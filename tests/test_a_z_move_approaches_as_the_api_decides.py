# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Z's backlash approach is the API's: a move to a place approaches from below, a move by a distance does not.

A downward Z move to a place first drives to the backlash below its
target, then up to it, so the backlash is always taken the same way. Every
production move of Z to a place asked for that and every move by a distance
did not, each caller choosing for itself. ``move_z`` and ``move_z_by`` decide
it; ``overshoot_enabled`` stays for a bench script that measures the backlash
and says so explicitly.

Observed as the targets the board is sent, on the simulated LS850's board,
which has a backlash.
"""

from __future__ import annotations

import pytest

from tests.scope_fakes import build_scope, home_sim_scope

_HIGH_UM = 3_000.0
_LOW_UM = 1_000.0


@pytest.fixture
def scope():
    scope = home_sim_scope(build_scope(simulate=True, sim_model='LS850', register_atexit=False))
    yield scope
    scope.disconnect()


def _z_targets_sent(scope, monkeypatch, move) -> list[float]:
    """The Z targets the board is sent while ``move`` runs, from Z at ``_HIGH_UM``."""
    scope.motion.move_z(_HIGH_UM)
    driver = scope._motion_driver
    real_write = driver.move_abs_pos
    sent: list[float] = []

    def recorded(axis, pos):
        if axis == 'Z':
            sent.append(pos)
        real_write(axis, pos)

    monkeypatch.setattr(driver, 'move_abs_pos', recorded)
    move(scope.motion)
    return sent


@pytest.mark.parametrize(
    ('move', 'approaches'),
    [
        pytest.param(lambda m: m.move_z(_LOW_UM), True, id='to-a-place'),
        pytest.param(lambda m: m.move_z_by(_LOW_UM - _HIGH_UM), False, id='by-a-distance'),
        pytest.param(
            lambda m: m.move_z(_LOW_UM, overshoot_enabled=False), False, id='to-a-place-bench-off'
        ),
        pytest.param(
            lambda m: m.move_z_by(_LOW_UM - _HIGH_UM, overshoot_enabled=True),
            True,
            id='by-a-distance-bench-on',
        ),
    ],
)
def test_a_downward_z_move_approaches_from_below_as_the_rule_or_the_bench_says(
    scope, monkeypatch, move, approaches
):
    backlash = scope._motion_driver.backlash_um()
    assert backlash > 0, 'the board has no backlash, so no move has a leg'

    sent = _z_targets_sent(scope, monkeypatch, move)

    leg = [_LOW_UM - backlash, _LOW_UM]
    assert sent == (leg if approaches else [_LOW_UM])
    assert scope.motion.get_target_position('Z') == _LOW_UM
