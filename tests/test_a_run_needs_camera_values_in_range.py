# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run may only ask the camera for a gain and an exposure it can take.

A stored step value outside the camera's range -- the template once shipped
Lumi at 48 dB, a ``current.json`` copied from it still says so, and a new step
copies its layer's gain -- would be refused by the camera setter at every
step, and the run loop would abandon scan after scan. prepare() refuses such
a run before anything moves, naming the step, the value and the camera's
range, so the one fix is the user's: edit the step. Nothing stored is
rewritten, because the same value may be right on the next, larger camera.

A limit the camera does not declare is not checked: a missing floor is not a
floor of zero.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ProtocolRunRefusedError
from tests.test_a_protocol_needs_its_objectives_on_the_turret import _prepare
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)

OUT_OF_RANGE = 'camera_setting_out_of_range'


def _steps(**values):
    return _make_multi_step_protocol([{'name': 'Lumi step', 'color': 'Lumi', **values}])


def test_the_api_caches_the_camera_floors(scope):
    imaging = scope.imaging
    assert imaging.min_gain_db_cached == pytest.approx(scope._camera_driver.min_gain)
    # The simulated camera declares no exposure floor.
    assert imaging.min_exposure_ms_cached is None


def test_a_gain_above_the_camera_maximum_is_refused_naming_the_step(executor, scope, tmp_path):
    maximum = scope.imaging.max_gain_db_cached
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, _steps(gain_db=maximum + 28.0), tmp_path)

    assert refusal.value.reason == OUT_OF_RANGE
    message = str(refusal.value)
    assert 'Lumi step' in message
    assert f'{maximum + 28.0:g}' in message
    assert f'{maximum:g}' in message


def test_a_gain_below_the_camera_minimum_is_refused(executor, scope, tmp_path):
    """Protocol validation already refuses a negative gain, so the floor that
    matters here is a camera's above 0 dB."""
    scope.imaging._commit_camera_writes({'min_gain_db': 2.0})
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, _steps(gain_db=1.0), tmp_path)

    assert refusal.value.reason == OUT_OF_RANGE


def test_an_exposure_above_the_camera_maximum_is_refused(executor, scope, tmp_path):
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, _steps(exposure_ms=scope.imaging.max_exposure_ms_cached + 1.0), tmp_path)

    assert refusal.value.reason == OUT_OF_RANGE


def test_values_in_range_are_admitted(executor, scope, tmp_path):
    _prepare(executor, _steps(gain_db=scope.imaging.max_gain_db_cached, exposure_ms=10.0), tmp_path)


def test_an_undeclared_floor_is_not_checked(executor, scope, tmp_path):
    """The simulated camera declares no exposure floor, so a tiny exposure is
    the camera's to take or refuse, not the run's to refuse."""
    _prepare(executor, _steps(exposure_ms=0.001), tmp_path)
