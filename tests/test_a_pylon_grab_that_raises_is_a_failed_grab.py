# Copyright Etaluma, Inc.
"""A Pylon grab whose result raises is one failed grab, not the camera's removal.

``GrabSucceeded()`` raising on the SDK's grab thread (Stage A) and
``GetArray()`` raising on the worker (Stage B) each used to mark the camera
removed on the first occurrence, and a removal has no reset short of a
reconnect: frames stop, every later command is refused, a run ends
(#726). Each is now counted toward the consecutive-failure cascade, the one
grab-path signal that the camera is gone; when that count trips, the camera
is marked removed and torn down, as every definite removal is.

a stand-in by design: the parent camera's SDK handle and the worker's failure
counter are the subjects' collaborators, and no simulated camera runs
pyloncamera.py.
"""

from unittest.mock import MagicMock

from tests.camera_fakes import bare_grab_worker, bare_image_handler


def _raising(*_args, **_kwargs):
    raise RuntimeError('grab result unreadable')


class TestStageA:
    def test_a_grab_succeeded_that_raises_is_handed_on_as_a_failure(self):
        handler, parent = bare_image_handler()
        del parent._mark_disconnected  # the real latch
        result = MagicMock()
        result.GrabSucceeded.side_effect = _raising

        handler.OnImageGrabbed(camera=None, grabResult=result)

        assert not parent.is_device_removed()
        kinds = [c.args[0] for c in handler._worker.enqueue.call_args_list]
        assert kinds == ['fail']


class TestStageB:
    def _worker(self, trips):
        worker, base = bare_grab_worker()
        parent = worker._parent
        del parent._mark_disconnected  # the real latch
        parent._schedule_async_teardown = MagicMock()
        base._record_failure.return_value = trips
        result = MagicMock()
        result.GetArray.side_effect = _raising
        return worker, base, parent, result

    def test_a_get_array_that_raises_is_counted_and_keeps_the_camera(self):
        worker, base, parent, result = self._worker(trips=False)

        worker._process_frame(result, 0.0, 0)

        assert not parent.is_device_removed()
        base._record_failure.assert_called_once()

    def test_the_failure_that_trips_the_cascade_removes_and_tears_down(self):
        worker, _, parent, result = self._worker(trips=True)

        worker._process_frame(result, 0.0, 0)

        assert parent.is_device_removed()
        parent._schedule_async_teardown.assert_called_once()
