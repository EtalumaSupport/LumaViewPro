# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Builders that drive a real AutofocusRunner.run() headlessly.

The scope is a MagicMock tuned so the AF loop actually executes:
motion reports idle with numeric Z reads, the camera snapshot is a
real dict, the default frame is a mid-gray ndarray, and
get_target_position sits past z_max so every coarse pass completes
after a single sample. Tests patch
modules.autofocus_functions.focus_function to control the curve shape
(flat -> degenerate abort; positive -> two-pass success).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np

AF_CENTER_Z = 500.0


def park_z(scope, z: float) -> None:
    """Move a simulated scope's Z to ``z`` and wait for it to arrive.

    A built or homed simulated scope stands at the bottom of Z travel,
    where every autofocus window reaches past it and is refused. A test
    that wants a sweep (or a later exit path) starts it where its window
    fits.
    """
    scope.motion.move_absolute('Z', z)
    while scope.motion.is_moving():
        time.sleep(0.01)


def af_runner_and_scope():
    """Real AutofocusRunner wired to a MagicMock scope that run() can
    drive headlessly. The scope's objective catalogue is a stub, so no
    objective config file is read."""
    from modules.autofocus_runner import AutofocusRunner

    scope = MagicMock()
    scope.led_connected = False
    scope.motion.is_moving.return_value = False
    scope.motion.get_current_position.return_value = AF_CENTER_Z
    scope.motion.get_target_position.return_value = 600.0
    scope.motion.get_axis_limits.return_value = {'min': 0.0, 'max': 14000.0}
    scope.imaging.save_camera_state.return_value = {'gain_db': 1.0, 'exposure_ms': 10.0}
    # AF grabs through the public capture_and_wait, under the run's taking.
    scope.imaging.capture_and_wait.return_value = np.full((40, 40), 50, dtype=np.uint8)
    runner = AutofocusRunner(
        scope=scope,
    )
    scope.objective_helper.get_objective_info.return_value = {
        'AF_range': 10.0,
        'AF_max': 30.0,
        'AF_min': 10.0,
    }
    # The Gaussian fit needs a dense real curve; these drives feed 1-2
    # samples per pass, so pin the fit result to the scan center.
    runner._find_best = lambda df: AF_CENTER_Z
    return runner, scope


def af_lease(scope):
    """The child lease AF takes under the run's lease (scope.protocol_lease).

    The run's lease hangs off the scope mock so its acquire_child call lands
    in scope.mock_calls, in order with the AF-lease writes it spawns.
    """
    return scope.protocol_lease.acquire_child.return_value


def drive_af(runner, **overrides):
    """Call runner.run() with minimal interactive-trigger kwargs.

    AF always runs inside a run, so it is handed the run's lease
    (scope.protocol_lease); af_lease() is the child it takes under it.
    """
    kwargs = {
        'objective_id': 'objective-under-test',
        'run_trigger_source': 'manual',
        'abort_event': threading.Event(),
        # No channel unless a test names one; run() requires both.
        'led_color': None,
        'led_illumination': 0.0,
        'led_lease': runner._scope.protocol_lease,
    }
    kwargs.update(overrides)
    return runner.run(**kwargs)
