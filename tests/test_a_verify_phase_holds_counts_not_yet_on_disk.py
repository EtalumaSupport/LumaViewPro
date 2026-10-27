# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A verify phase holds counts not yet on disk, and the installation's are back after it.

``ImagingAPI.held_frame_rules(record)`` hands the gate rules built from a
record in memory, checked as a file is, for its ``with`` block: a change
recorded inside it waits the held count, one recorded after it the
installation's, however the block ended.
"""

import pytest

from modules.activity_claim import ActivityClaim
from modules.caller import Caller, calling
from modules.exceptions import ArgumentRefusedError, ControlSessionRefusedError

_MEASURED = {
    'delivered_frames': {'p50': 3, 'p95': 6, 'max': 6},
    'trials': 10,
    'provenance': {
        'run_id': 'run-verify',
        'host': 'bench-pc',
        'os': 'Windows 11',
        'lvp_version': '4.0.0',
        'plugin_version': '1.0.110',
        'measured_date': '2026-10-26',
        'microscope': 'LS850T',
    },
}


def _record(validity):
    return {'schema': 3, 'camera_model': 'SimulatedCamera-3840x2160', 'validity': validity}


def _gain_change(scope):
    scope.camera.set_gain_db(scope.camera.read_gain_db() + 1.0)
    return scope.imaging.recorded_changes()['gain']


def test_a_change_inside_the_block_waits_the_held_count_and_one_after_it_the_installations(
    sim_scope,
):
    installation = sim_scope.capabilities.frame_rules['gain']
    assert installation.count != 6, 'precondition: the held count differs from the installation'

    with sim_scope.imaging.held_frame_rules(_record({'gain': {'up': [_MEASURED]}})):
        inside = _gain_change(sim_scope)
    after = _gain_change(sim_scope)

    assert (inside.skip, inside.rule.provenance['run_id']) == (6, 'run-verify')
    assert (after.skip, after.rule) == (installation.count, installation)


def test_the_installations_counts_are_back_after_a_block_that_raised(sim_scope):
    with (
        pytest.raises(RuntimeError, match='the trial failed'),
        sim_scope.imaging.held_frame_rules(_record({'gain': {'up': [_MEASURED]}})),
    ):
        raise RuntimeError('the trial failed')

    assert _gain_change(sim_scope).rule == sim_scope.capabilities.frame_rules['gain']


def test_a_record_this_release_cannot_take_is_refused_and_nothing_is_held(sim_scope):
    one_value = _record({'gain': {'up': _MEASURED}})

    with (
        pytest.raises(ArgumentRefusedError) as refused,
        sim_scope.imaging.held_frame_rules(one_value),
    ):
        pass

    assert (refused.value.reason, refused.value.argument) == ('wrong_kind', 'record')
    assert _gain_change(sim_scope).rule == sim_scope.capabilities.frame_rules['gain']


def test_a_hold_is_refused_while_someone_else_holds_control_and_nothing_is_held(sim_scope):
    claim = ActivityClaim()
    sim_scope.set_activity_claim(claim)
    with calling(Caller.REST):
        claim.take_control('plate reader', finishing=lambda: False)

    with (
        pytest.raises(ControlSessionRefusedError) as refused,
        sim_scope.imaging.held_frame_rules(_record({'gain': {'up': [_MEASURED]}})),
    ):
        pass

    assert refused.value.reason == 'control_session_held'
    assert sim_scope.imaging.frame_validity.rules() == sim_scope.capabilities.frame_rules
