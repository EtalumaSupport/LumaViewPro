# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Asking whether a camera is connected answers "found and not removed", and changes nothing.

``Camera.is_connected`` is asked per displayed frame and by every connection
question. A Pylon camera never connected answered True, and asking it about a
camera it had released marked that camera removed; an IDS camera never
connected was marked removed and logged ``Camera disconnected`` at ERROR. A
removal has no reset short of a reconnect, so a question must never make one.

The SDK is a stand-in by design: the states asked about (never connected, the
handle released) are a Pylon's and an IDS's own, which the simulated camera
does not go through.
"""

import logging

from tests.camera_fakes import bare_ids_camera, bare_pylon_camera

# The value ``active`` holds in each state (``Camera.active``'s three states).
_STATES = {'never connected': False, 'disconnected': None}
_DRIVERS = {'Pylon': bare_pylon_camera, 'IDS': bare_ids_camera}


def test_a_camera_not_connected_answers_no_and_is_left_as_it_was(caplog):
    offenders = []
    for driver, build in _DRIVERS.items():
        for state, active in _STATES.items():
            cam = build()
            # The real latch, which the fake replaces with a recorder.
            del cam._mark_disconnected
            cam.active = active
            caplog.clear()
            with caplog.at_level(logging.ERROR):
                answer = cam.is_connected()
            if answer is not False:
                offenders.append(f'{driver}, {state}: is_connected() answered {answer!r}')
            if cam.is_device_removed():
                offenders.append(f'{driver}, {state}: asking marked the camera removed')
            if any('Camera disconnected' in r.getMessage() for r in caplog.records):
                offenders.append(f'{driver}, {state}: asking logged "Camera disconnected"')
            # Left as a disconnect leaves it, so the end-of-test disconnect
            # has no handle to release.
            cam.active = None
    assert not offenders, '\n'.join(offenders)
