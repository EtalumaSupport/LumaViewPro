# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon teardown never sends Close() to a camera marked removed.

Close() on a removed device handle is the call pypylon #225 names as a
native-abort hazard, so a teardown that finds the camera removed releases
only the Python-side handle (DetachDevice, DestroyDevice). A removal can
be latched while the teardown runs -- during its StopGrabbing, or its
bounded wait for the acquisition to go idle -- and it read the mark only
once, at entry.
"""

from tests.camera_fakes import disconnectable_pylon_camera


def test_a_removal_latched_during_the_teardown_skips_close():
    cam = disconnectable_pylon_camera()
    handle = cam.active
    cam.is_grabbing = lambda: True

    def removed_while_draining(timeout_s):
        cam._device_removed = True
        return True

    cam._wait_for_acquisition_idle.side_effect = removed_while_draining

    assert cam.disconnect() is True

    cam.stop_grabbing.assert_called_once()
    handle.Close.assert_not_called()
    handle.DetachDevice.assert_called_once()
    handle.DestroyDevice.assert_called_once()
    assert cam.active is None
