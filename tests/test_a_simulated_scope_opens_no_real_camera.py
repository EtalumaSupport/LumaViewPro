# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope cannot be told to build a real camera.

The camera registry picks the driver for the scope's mode, a simulator for a
simulated scope. A scope that took a driver's name built the real Pylon
driver by name on a simulated scope, which opened an attached camera, or
recorded a missing one as a failed bring-up. No caller names the driver.
"""

import pytest

from tests.scope_fakes import construct_scope


def test_a_simulated_scope_takes_no_camera_driver_name():
    with pytest.raises(TypeError, match='camera_type'):
        construct_scope(simulate=True, camera_type='pylon', register_atexit=False)
