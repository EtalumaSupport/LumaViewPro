# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pylon hardware tests -- opt-in via --run-pylon-hardware.

These tests require:
  1. Real pypylon SDK installed (not the conftest MagicMock)
  2. A connected Basler camera

Skipped by default. Run with:
    pytest tests/test_pylon_hardware.py --run-pylon-hardware

The `pylon_hardware` marker is gated by conftest.pytest_collection_modifyitems --
test bodies do NOT need their own skip dance.

Mirrors the shape of test_ids_hardware.py so the abstraction is symmetric
across camera vendors. Add coverage as Pylon-specific behaviors come up.
"""

import time
import unittest

import pytest

# When --run-pylon-hardware is set, conftest skips installing the pypylon
# mock so the real SDK loads here. When the flag is NOT set, this import
# succeeds against the conftest mock and the marker below skips the
# tests at collection time.
from drivers.pyloncamera import PylonCamera


@pytest.mark.pylon_hardware
class TestPylon(unittest.TestCase):
    def setUp(self):
        self.camera = PylonCamera()

    def tearDown(self):
        self.camera.disconnect()
        time.sleep(0.5)

    def test_connect_disconnect(self):
        self.assertTrue(self.camera.disconnect())
        self.assertTrue(self.camera.connect())

    def test_grab(self):
        self.assertTrue(self.camera.is_grabbing())
        self.camera.stop_grabbing()
        self.assertFalse(self.camera.is_grabbing())
        self.camera.start_grabbing()
        self.assertTrue(self.camera.is_grabbing())

    def test_pixel_format(self):
        formats = self.camera.get_supported_pixel_formats()
        self.assertTrue(len(formats) > 0)
        self.camera.set_pixel_format(formats[0])
        self.assertEqual(self.camera.get_pixel_format(), formats[0])

    def test_exposure_t(self):
        # Use a short exposure that's well within range for any model
        self.camera.exposure_t(15)
        self.assertAlmostEqual(self.camera.get_exposure_t(), 15.0, delta=0.5)

    def test_binning_size(self):
        # Most Basler models support 1x and 2x; reject 0 and very large.
        self.assertTrue(self.camera.set_binning_size(1))
        self.assertEqual(self.camera.get_binning_size(), 1)
        self.assertFalse(self.camera.set_binning_size(0))

    def test_grab_frame(self):
        time.sleep(0.5)  # Allow grabbing to settle
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertIsNotNone(timestamp)
        self.assertIsNotNone(self.camera.array)

    def test_gain(self):
        self.camera.gain(0)  # 0 dB is always in range
        self.assertAlmostEqual(self.camera.get_gain(), 0.0, delta=0.5)

    def test_chunks_flow_through_grab(self):
        """Path C commit 3 end-to-end: connect enables chunks, OnImageGrabbed
        reads them per frame, ImageHandlerBase.last_chunks exposes them.
        """
        time.sleep(0.5)  # let streaming settle
        result, _ts = self.camera.grab()
        self.assertTrue(result, 'grab failed -- chunks check requires successful frame')

        chunks = self.camera.cam_image_handler._base.get_last_chunks()
        print('\n=== last_chunks after first grab ===')
        print(f'  chunks: {chunks}')
        print('=====================================\n')

        self.assertIsNotNone(chunks, 'Path C: last_chunks should populate after a successful grab')
        # All three target chunks should be present
        for key in ('ExposureTime', 'Gain', 'FrameID'):
            self.assertIn(
                key,
                chunks,
                f"Path C: chunk '{key}' missing from last_chunks; got keys={sorted(chunks.keys())}",
            )
        # Sanity: values are floats/ints, not None or weird types
        self.assertIsInstance(chunks['ExposureTime'], (int, float))
        self.assertIsInstance(chunks['Gain'], (int, float))
        self.assertIsInstance(chunks['FrameID'], int)

    def test_measure_chunk_tolerances(self):
        """Sweep gain and exposure across the supported range, observe the
        max delta between requested value and ChunkGain / ChunkExposureTime
        in steady-state grabs. Output drives the DEFAULT_CHUNK_TOLERANCE
        constants in modules/frame_validity.py.

        Methodology notes:
          - Off-grid values (5.3 dB, 7.123 ms) expose any internal SDK
            quantization that might round set values to grid points.
          - grab_new_capture() forces a fresh frame each observation;
            grab() can return cached frames and yield zero-delta artifacts.
          - FrameID printed per observation to confirm frames advance
            (catches "we got the same cached frame 10 times" failure mode).
        """
        time.sleep(0.5)

        print('\n=== ChunkGain delta sweep ===')
        gain_deltas = []
        for target in [0.0, 1.0, 5.3, 7.123, 10.0, 15.0, 20.0, 23.999]:
            try:
                self.camera.gain(target)
            except Exception as e:
                print(f'  target={target:7.3f} dB | gain set failed: {e}')
                continue
            time.sleep(0.5)
            obs = []
            frame_ids = []
            for _ in range(10):
                ok, _ = self.camera.grab_new_capture(timeout_s=2.0)
                if not ok:
                    continue
                chunks = self.camera.cam_image_handler._base.get_last_chunks()
                if chunks and 'Gain' in chunks:
                    obs.append(chunks['Gain'])
                    frame_ids.append(chunks.get('FrameID'))
            deltas = [abs(o - target) for o in obs]
            max_d = max(deltas) if deltas else float('nan')
            gain_deltas.append(max_d)
            print(
                f'  target={target:7.3f} dB | observed_first5={obs[:5]} | '
                f'max_delta={max_d:.4f} dB | FrameIDs={frame_ids[:5]}'
            )

        print('\n=== ChunkExposureTime delta sweep ===')
        exp_deltas = []
        for target_ms in [1.0, 5.0, 7.123, 10.0, 50.0, 100.0, 199.999]:
            try:
                self.camera.exposure_t(target_ms)
            except Exception as e:
                print(f'  target={target_ms:7.3f} ms | exposure set failed: {e}')
                continue
            time.sleep(max(0.5, target_ms / 100.0))
            obs_us = []
            frame_ids = []
            for _ in range(5):
                ok, _ = self.camera.grab_new_capture(timeout_s=max(2.0, target_ms / 1000.0 * 5))
                if not ok:
                    continue
                chunks = self.camera.cam_image_handler._base.get_last_chunks()
                if chunks and 'ExposureTime' in chunks:
                    obs_us.append(chunks['ExposureTime'])
                    frame_ids.append(chunks.get('FrameID'))
            deltas = [abs(o - target_ms * 1000.0) for o in obs_us]
            max_d = max(deltas) if deltas else float('nan')
            exp_deltas.append(max_d)
            print(
                f'  target={target_ms:7.3f} ms ({target_ms * 1000:.1f} us) | '
                f'observed_first5={obs_us[:5]} | max_delta_us={max_d:.4f} | '
                f'FrameIDs={frame_ids[:5]}'
            )

        overall_gain_max = max(gain_deltas) if gain_deltas else float('nan')
        overall_exp_max = max(exp_deltas) if exp_deltas else float('nan')
        from modules.frame_validity import FrameValidity

        cur_gain_tol = FrameValidity.DEFAULT_CHUNK_TOLERANCE['gain']
        cur_exp_tol = FrameValidity.DEFAULT_CHUNK_TOLERANCE['exposure']
        print('\n=== Tolerance summary ===')
        print(
            f'  gain     max delta = {overall_gain_max:.6e} dB   (current default: {cur_gain_tol})'
        )
        print(f'  exposure max delta = {overall_exp_max:.6e} us   (current default: {cur_exp_tol})')
        print(f'  gain     headroom  = {cur_gain_tol / max(overall_gain_max, 1e-12):.1f}x')
        print(f'  exposure headroom  = {cur_exp_tol / max(overall_exp_max, 1e-12):.1f}x')
        print('=========================\n')

        # Sanity: observed deltas must fit inside the configured tolerance.
        # If this assertion fires, the camera's chunk quantization grew
        # past our tolerance budget -- bench-measure and refine the constants.
        self.assertLessEqual(
            overall_gain_max,
            cur_gain_tol,
            f'Gain chunk delta {overall_gain_max} exceeds tolerance {cur_gain_tol}',
        )
        self.assertLessEqual(
            overall_exp_max,
            cur_exp_tol,
            f'Exposure chunk delta {overall_exp_max} exceeds tolerance {cur_exp_tol}',
        )

        self.assertGreater(
            len(gain_deltas),
            0,
            'gain sweep produced no data -- chunks flow broken or all gain sets failed',
        )
        self.assertGreater(
            len(exp_deltas),
            0,
            'exposure sweep produced no data -- chunks flow broken or all exposure sets failed',
        )

    def test_chunk_clear_short_circuits_skip_frames(self):
        """End-to-end: chunks clear pending sources on the FIRST frame
        after a parameter change, bypassing the skip-frames count entirely.
        This is the deterministic behavior chunk-driven validity provides
        over the empirical skip-frames calibration."""
        from modules.frame_validity import FrameValidity

        fv = FrameValidity()

        target_gain = 5.0
        self.camera.gain(target_gain)
        time.sleep(0.5)  # let the new gain propagate

        # Invalidate + record target the same way Lumascope.set_gain does.
        fv.invalidate('gain')
        fv.set_target('gain', target_gain)
        pending_before = dict(fv.pending_sources)
        self.assertIn('gain', pending_before)

        # Grab ONE fresh frame and feed its chunks to count_frame.
        ok, _ = self.camera.grab_new_capture(timeout=2.0)
        self.assertTrue(ok)
        chunks = self.camera.cam_image_handler._base.get_last_chunks()
        fv.count_frame(chunk_data=chunks)

        pending_after = dict(fv.pending_sources)
        print('\n=== chunk-clear short-circuit ===')
        print(f'  pending before    : {pending_before}')
        print(f'  observed ChunkGain: {chunks.get("Gain") if chunks else None}')
        print(f'  pending after 1 frame: {pending_after}')
        print('==================================\n')

        self.assertNotIn(
            'gain',
            pending_after,
            f"Chunk-match must clear 'gain' from pending after 1 frame; "
            f'chunks={chunks}, target={target_gain}',
        )

    def test_probe_chunk_capabilities(self):
        """T1 (FRAME_VALIDITY_PLAN.md sec.3): static introspection probe
        for chunk-data support. Answers whether ExposureTime / Gain /
        FrameID are supported on the connected camera. Print-heavy
        rather than strict-assert because the answer drives architecture
        decisions (Path A vs Path C) and we want raw data on the bench.
        """
        result = self.camera.probe_chunk_capabilities()
        print('\n=== probe_chunk_capabilities ===')
        print(f'  model:      {result["model"]}')
        print(f'  firmware:   {result["firmware"]}')
        print(f'  serial:     {result["serial"]}')
        print(f'  advertised: {result["advertised"]}')
        print(f'  enabled:    {result["enabled"]}')
        print(f'  errors:     {result["errors"]}')
        print('================================\n')

        # Hard assertions: probe ran without per-step explosion.
        self.assertIsInstance(result, dict)
        self.assertIn('advertised', result)
        self.assertIn('enabled', result)
        # ChunkSelector node should exist on any modern Basler USB3 cam.
        self.assertGreater(
            len(result['advertised']),
            0,
            f'Camera advertised no ChunkSelector entries -- chunks unsupported. '
            f'Errors: {result["errors"]}',
        )

    def test_worker_thread_alive_after_connect(self):
        """After PylonCamera.connect() returns, the Stage B worker thread
        must be alive (daemon, named 'PylonImageGrabWorker') so the SDK
        can hand off the first OnImageGrabbed frame without queue-full
        drops or no-consumer leakage.

        Lifecycle wiring per LIFECYCLE_INVENTORY: worker.start() runs
        AFTER RegisterImageEventHandler and BEFORE camera.Open()'s
        auto-StartGrabbing window.
        """
        import threading as _t

        worker = self.camera.cam_image_handler._worker
        self.assertIsNotNone(worker._thread)
        self.assertTrue(worker._thread.is_alive())
        self.assertTrue(worker._thread.daemon)
        self.assertEqual(worker._thread.name, 'PylonImageGrabWorker')
        # And the live thread enumeration agrees.
        names = [t.name for t in _t.enumerate()]
        self.assertIn('PylonImageGrabWorker', names)

    def test_worker_queue_drains_at_steady_state(self):
        """At 30 FPS steady-state, Stage B must catch up faster than
        Stage A enqueues -- the bounded worker_queue should NOT grow.
        Sample queue depth over a window of frames and assert the
        average stays below the catch-up threshold."""
        # Let grabbing settle.
        time.sleep(1.0)
        worker_queue = self.camera.cam_image_handler._worker._worker_queue
        samples = []
        start = time.monotonic()
        # 3-second sampling window at 50 Hz; sufficient signal at 30 FPS.
        while time.monotonic() - start < 3.0:
            samples.append(worker_queue.qsize())
            time.sleep(0.02)
        avg_depth = sum(samples) / len(samples) if samples else 0
        max_depth = max(samples) if samples else 0
        print(f'\n=== worker_queue depth sample (n={len(samples)}) ===')
        print(f'  avg: {avg_depth:.2f}, max: {max_depth}')
        print('=================================================\n')
        self.assertLess(
            avg_depth,
            4.0,
            f'Stage B not keeping up: avg worker_queue depth = {avg_depth:.2f} '
            f'(expected < 4 at 30 FPS)',
        )

    def test_disconnect_drains_worker_queue(self):
        """worker.stop() during disconnect must drain in-flight grab
        results so the SDK input queue doesn't underrun on the next
        connect cycle per `class_pylon_1_1_c_grab_result_ptr.html`.

        After disconnect the worker thread must be gone and the worker
        queue must be empty (drain_and_release happened).
        """
        worker = self.camera.cam_image_handler._worker
        worker_thread = worker._thread
        # Let some frames flow.
        time.sleep(0.5)
        self.camera.disconnect()
        # disconnect's _stop_image_grab_worker is bounded at timeout=1.0
        # plus our test slack.
        if worker_thread is not None:
            worker_thread.join(timeout=2.0)
            self.assertFalse(
                worker_thread.is_alive(), 'worker thread did not exit within 2s of disconnect'
            )
        self.assertTrue(
            worker._worker_queue.empty(),
            f'worker_queue should be empty after disconnect; '
            f'has {worker._worker_queue.qsize()} item(s)',
        )
        # Reconnect for tearDown's disconnect.
        self.camera.connect()


if __name__ == '__main__':
    unittest.main()
