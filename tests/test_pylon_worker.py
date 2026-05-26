# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""_PylonImageGrabWorker unit tests (Stage B of OnImageGrabbed split).

Pure-mock tests -- no hardware flag, no real Pylon SDK. Exercise the
worker lifecycle (start / stop / restart / daemon flag) plus the
per-item processing paths (_process_frame, _process_failure,
_drain_and_release) by injecting items directly into the worker queue.

The cross-thread smart-pointer release contract (per
class_pylon_1_1_c_grab_result_ptr.html: "grabbing will stop with an
input queue underrun, when the grab results are never released") is
the load-bearing invariant -- _drain_and_release plus the per-item
`finally: del grabResult` in _run are the mechanism. Tests assert
those refs are released after stop().
"""

import queue
import time
import unittest
import weakref
from unittest.mock import MagicMock

from drivers.pyloncamera import (
    _PYLON_ERR_BUFFER_CANCELED,
    _PYLON_ERR_PAYLOAD_DISCARDED,
    _PylonImageGrabWorker,
)


class _FakeGrabResult:
    """Minimal stand-in for a Pylon CGrabResultPtr.

    GetArray returns a tiny numpy-like list-of-list so _process_frame
    can call .copy(); err_code / err_desc return preset values so
    _process_failure can classify them. Real numpy isn't imported in
    the test deliberately -- conftest leaves numpy unmocked but the
    test path doesn't need it.
    """

    def __init__(self, err_code=0, err_desc='ok'):
        self._err_code = err_code
        self._err_desc = err_desc

    def GetArray(self):
        # Object exposing .copy() that returns itself; sufficient for
        # the worker's "img = grabResult.GetArray().copy()" line.
        arr = MagicMock()
        arr.copy.return_value = arr
        return arr

    def GetErrorCode(self):
        return self._err_code

    def GetErrorDescription(self):
        return self._err_desc


def _make_worker(queue_depth=None):
    """Build a worker with mocked parent + base for unit testing.

    queue_depth is passed straight to _PylonImageGrabWorker.__init__
    as the queue_depth kwarg; None leaves the default in place.
    """
    parent = MagicMock()
    parent._device_removed = False
    parent.active = MagicMock()
    parent.is_grabbing.return_value = True
    base = MagicMock()
    base._record_failure.return_value = False
    frame_queue = queue.Queue(maxsize=1)
    worker = _PylonImageGrabWorker(
        parent,
        base,
        frame_queue,
        queue_depth=queue_depth,
    )
    return worker, parent, base, frame_queue


class TestWorkerLifecycle(unittest.TestCase):
    def test_start_spawns_daemon_thread(self):
        worker, _, _, _ = _make_worker()
        try:
            worker.start()
            self.assertIsNotNone(worker._thread)
            self.assertTrue(worker._thread.is_alive())
            self.assertTrue(worker._thread.daemon)
            self.assertEqual(worker._thread.name, 'PylonImageGrabWorker')
        finally:
            worker.stop()

    def test_start_is_idempotent(self):
        worker, _, _, _ = _make_worker()
        try:
            worker.start()
            first_thread = worker._thread
            worker.start()
            self.assertIs(worker._thread, first_thread)
        finally:
            worker.stop()

    def test_stop_joins_within_timeout(self):
        worker, _, _, _ = _make_worker()
        worker.start()
        t0 = time.monotonic()
        worker.stop(timeout=2.0)
        elapsed = time.monotonic() - t0
        self.assertFalse(worker._thread.is_alive())
        # Stop should be quick (well under timeout) because the sentinel
        # wakes the get() immediately. Real-world margin in test env.
        self.assertLess(elapsed, 1.5, f'stop took {elapsed:.2f}s')

    def test_stop_idempotent(self):
        worker, _, _, _ = _make_worker()
        worker.start()
        worker.stop()
        # Second stop on an already-stopped worker should not raise.
        worker.stop()
        self.assertFalse(worker._thread.is_alive())

    def test_stop_without_start_is_noop(self):
        worker, _, _, _ = _make_worker()
        # Should not raise; thread was never spawned.
        worker.stop()
        self.assertIsNone(worker._thread)

    def test_restart_after_stop(self):
        worker, _, _, _ = _make_worker()
        worker.start()
        worker.stop()
        worker.start()
        try:
            self.assertTrue(worker._thread.is_alive())
        finally:
            worker.stop()


class TestWorkerQueueDepth(unittest.TestCase):
    def test_default_queue_depth(self):
        worker, _, _, _ = _make_worker()
        self.assertEqual(worker._worker_queue.maxsize, 8)

    def test_kwarg_override_queue_depth(self):
        worker, _, _, _ = _make_worker(queue_depth=16)
        self.assertEqual(worker._worker_queue.maxsize, 16)

    def test_none_kwarg_falls_back_to_default(self):
        worker, _, _, _ = _make_worker(queue_depth=None)
        self.assertEqual(worker._worker_queue.maxsize, 8)


class TestProcessFrame(unittest.TestCase):
    def test_success_path_stores_and_publishes(self):
        # _FakeGrabResult has no Chunk* attributes, so _read_validity_chunks
        # returns None via the getattr-default path -- no need to patch
        # the static method.
        worker, _, base, frame_queue = _make_worker()
        gr = _FakeGrabResult()
        ts = 12345.0
        worker._process_frame(gr, ts)
        base._store_frame.assert_called_once()
        success, _, got_ts = frame_queue.get_nowait()
        self.assertTrue(success)
        self.assertEqual(got_ts, ts)

    def test_drains_stale_before_putting(self):
        worker, _, base, frame_queue = _make_worker()
        frame_queue.put_nowait((True, 'stale_img', 0.0))
        worker._process_frame(_FakeGrabResult(), 99.0)
        success, _, ts = frame_queue.get_nowait()
        self.assertEqual(ts, 99.0)
        self.assertTrue(frame_queue.empty())


class TestProcessFailure(unittest.TestCase):
    def test_buffer_canceled_does_not_record_failure(self):
        worker, _, base, _ = _make_worker()
        gr = _FakeGrabResult(err_code=_PYLON_ERR_BUFFER_CANCELED, err_desc='cancelled')
        worker._process_failure(gr, 0.0)
        base._record_failure.assert_not_called()

    def test_payload_discarded_does_not_record_failure(self):
        worker, _, base, _ = _make_worker()
        gr = _FakeGrabResult(err_code=_PYLON_ERR_PAYLOAD_DISCARDED, err_desc='discarded')
        worker._process_failure(gr, 0.0)
        base._record_failure.assert_not_called()

    def test_device_removed_skips_failure_counter(self):
        worker, parent, base, _ = _make_worker()
        parent._device_removed = True
        gr = _FakeGrabResult(err_code=9999, err_desc='whatever')
        worker._process_failure(gr, 0.0)
        base._record_failure.assert_not_called()

    def test_generic_failure_records(self):
        worker, _, base, _ = _make_worker()
        gr = _FakeGrabResult(err_code=999, err_desc='transport err')
        worker._process_failure(gr, 0.0)
        base._record_failure.assert_called_once()

    def test_cascade_triggers_auto_stop(self):
        worker, parent, base, _ = _make_worker()
        base._record_failure.return_value = True
        gr = _FakeGrabResult(err_code=999, err_desc='transport err')
        worker._process_failure(gr, 0.0)
        parent.stop_grabbing.assert_called_once()
        parent._mark_disconnected.assert_called_once()

    def test_grabresult_introspection_failure_is_tolerated(self):
        """err_code / desc readers may themselves raise on a dead handle."""
        worker, _, base, _ = _make_worker()
        gr = MagicMock()
        gr.GetErrorCode.side_effect = RuntimeError('handle dead')
        gr.GetErrorDescription.side_effect = RuntimeError('handle dead')
        # Should not raise; treats as a generic failure.
        worker._process_failure(gr, 0.0)
        base._record_failure.assert_called_once()


class TestRunLoop(unittest.TestCase):
    """End-to-end: items enqueued, worker drains them via _run."""

    def test_frame_kind_routes_to_process_frame(self):
        worker, _, base, frame_queue = _make_worker()
        worker.start()
        try:
            worker._worker_queue.put_nowait(('frame', _FakeGrabResult(), 42.0))
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if base._store_frame.called:
                    break
                time.sleep(0.01)
            self.assertTrue(base._store_frame.called)
        finally:
            worker.stop()

    def test_fail_kind_routes_to_process_failure(self):
        worker, _, base, _ = _make_worker()
        worker.start()
        try:
            worker._worker_queue.put_nowait(
                ('fail', _FakeGrabResult(err_code=999, err_desc='x'), 0.0)
            )
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if base._record_failure.called:
                    break
                time.sleep(0.01)
            self.assertTrue(base._record_failure.called)
        finally:
            worker.stop()

    def test_device_removed_short_circuits_in_run(self):
        worker, parent, base, _ = _make_worker()
        parent._device_removed = True
        worker.start()
        try:
            worker._worker_queue.put_nowait(('frame', _FakeGrabResult(), 0.0))
            # Worker should consume but NOT call _process_frame.
            time.sleep(0.2)
            base._store_frame.assert_not_called()
        finally:
            worker.stop()

    def test_exception_in_processing_does_not_kill_thread(self):
        worker, _, base, _ = _make_worker()
        base._store_frame.side_effect = RuntimeError('boom')
        worker.start()
        try:
            worker._worker_queue.put_nowait(('frame', _FakeGrabResult(), 1.0))
            time.sleep(0.2)
            self.assertTrue(worker._thread.is_alive())
            base._store_frame.side_effect = None
            worker._worker_queue.put_nowait(('frame', _FakeGrabResult(), 2.0))
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if base._store_frame.call_count >= 2:
                    break
                time.sleep(0.01)
            self.assertGreaterEqual(base._store_frame.call_count, 2)
        finally:
            worker.stop()


class TestDrainAndRelease(unittest.TestCase):
    def test_drain_releases_items(self):
        worker, _, _, _ = _make_worker()
        worker._worker_queue.put_nowait(('frame', _FakeGrabResult(), 0.0))
        worker._worker_queue.put_nowait(('fail', _FakeGrabResult(err_code=1), 0.0))
        worker._drain_and_release()
        self.assertTrue(worker._worker_queue.empty())

    def test_stop_sentinel_drains_pending_items(self):
        """Stop must drain whatever Stage A enqueued before SDK teardown.

        Otherwise grabResult smart pointers leak past disconnect and the
        SDK input queue underruns on the next connect cycle.
        """
        worker, parent, _, _ = _make_worker()
        # Block processing by flagging device_removed -- items will be
        # consumed-and-released without calling _process_* mocks.
        parent._device_removed = True
        worker.start()
        try:
            for i in range(3):
                worker._worker_queue.put_nowait(('frame', _FakeGrabResult(), float(i)))
        finally:
            worker.stop()
        self.assertFalse(worker._thread.is_alive())
        # Whatever the worker didn't drain pre-stop should have been
        # drained by the stop-sentinel handler.
        self.assertTrue(worker._worker_queue.empty())


class TestGrabResultRelease(unittest.TestCase):
    """Verify the worker drops its reference to each grabResult.

    Uses weakref to assert the smart pointer is releasable after the
    worker has processed an item. The SDK's underlying CGrabResultPtr
    has well-defined cross-thread semantics provided the smart pointer
    is released promptly -- this test pins the release contract on the
    Python side.
    """

    def test_processed_grabresult_is_dereferenced(self):
        worker, _, base, _ = _make_worker()

        class _Releasable:
            """Smart-pointer stand-in supporting weakref."""

            def __init__(self):
                self._err_code = _PYLON_ERR_BUFFER_CANCELED
                self._err_desc = 'cancelled'

            def GetErrorCode(self):
                return self._err_code

            def GetErrorDescription(self):
                return self._err_desc

        gr = _Releasable()
        ref = weakref.ref(gr)
        worker.start()
        try:
            worker._worker_queue.put_nowait(('fail', gr, 0.0))
            # Wait for processing.
            time.sleep(0.2)
        finally:
            worker.stop()
        # Drop our local reference; if the worker properly del'd, the
        # weakref should be None on the next gc cycle.
        del gr
        import gc

        gc.collect()
        self.assertIsNone(ref(), 'worker did not release its grabResult reference after processing')


if __name__ == '__main__':
    unittest.main()
