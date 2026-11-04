"""A bare camera stand-in runs the real ``disconnect()`` to its end.

The contract is ``tests/camera_fakes.py``'s: its fakes are REAL driver
objects, so production methods run on them. ``Camera.__del__`` runs
``disconnect()`` on any fake still active when it is collected, which can be
inside a later test; a fake missing base state raises there and logs the
error on the camera log inside whatever test is running (a failed slow-stop
test once failed the fast-stop test beside it this way).

A stand-in by design: the subject is the fake itself; the camera log is
read from ``caplog``, the driver's own ``LVP.camera`` lines.
"""

import logging

from tests.camera_fakes import bare_ids_camera, bare_pylon_camera
from tests.log_capture import records_of


def test_a_bare_pylon_camera_disconnects_without_an_error(caplog):
    from drivers import pyloncamera

    cam = bare_pylon_camera()

    assert cam.disconnect() is True

    assert cam.active is None
    told = [
        (r.levelname, r.getMessage())
        for r in records_of(caplog, pyloncamera)
        if r.name == 'LVP.camera' and r.levelno >= logging.WARNING
    ]
    assert told == [], told


def test_a_bare_ids_camera_disconnects_without_an_error(caplog):
    from drivers import idscamera

    cam = bare_ids_camera()

    assert cam.disconnect() is True

    assert cam.active is None
    told = [
        (r.levelname, r.getMessage())
        for r in records_of(caplog, idscamera)
        if r.levelno >= logging.WARNING
    ]
    assert told == [], told
