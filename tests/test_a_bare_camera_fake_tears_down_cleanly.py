"""A bare Pylon stand-in runs the real ``disconnect()`` to its end.

The contract is ``tests/camera_fakes.py``'s: its fakes are REAL driver
objects, so production methods run on them. ``Camera.__del__`` runs
``disconnect()`` on any fake still active when it is collected, which can be
inside a later test; a fake missing base state raises there and logs the
error on whatever ``pyloncamera._cam_log`` that test has patched (a failed
slow-stop test once failed the fast-stop test beside it this way).

A stand-in by design: the subject is the fake itself; the camera log is a
MagicMock patched over ``pyloncamera._cam_log`` (the suite mocks
``lvp_logger``, so ``caplog`` would see nothing).
"""

from unittest import mock

from tests.camera_fakes import bare_pylon_camera


def test_a_bare_pylon_camera_disconnects_without_an_error():
    from drivers import pyloncamera

    cam = bare_pylon_camera()
    log = mock.MagicMock()

    with mock.patch.object(pyloncamera, '_cam_log', log):
        assert cam.disconnect() is True

    assert cam.active is None
    assert not log.exception.called, log.exception.call_args_list
    assert not log.warning.called, log.warning.call_args_list
