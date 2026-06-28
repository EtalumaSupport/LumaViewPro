# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stub-testable units for the rebuilt IDS driver (no SDK / no camera).

Covers the Mac-authorable pieces of the native-depth rebuild:
  - significant_bits derived from the wire format name (the depth that pairs
    with each delivered frame, replacing the old pinned-8 behavior)
  - the IPL conversion target that unpacks each wire format to native depth
  - the frame-rate crash-stop cap (soft AcquisitionFrameRateTarget), which
    replaced the USB-saturating maximize path

The live grab loop, recovery, and real conversion are bench-gated.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import ids_peak_ipl
import pytest

from drivers.camera_profiles import CameraProfile
from drivers.idscamera import IDSCamera, _ids_ipl_target, ids_significant_bits
from tests.camera_fakes import bare_ids_camera


class _RecordingNode:
    """Node that records writes and serves configured min/max/value, for
    asserting SetValue calls and exercising range-dependent logic."""

    _UNSET = object()

    def __init__(self, value=_UNSET, minimum=None, maximum=None):
        self.value = value
        self._min = minimum
        self._max = maximum
        self.entry = None

    def SetValue(self, v):
        self.value = v

    def Value(self):
        return self.value

    def Minimum(self):
        return self._min

    def Maximum(self):
        return self._max

    def SetCurrentEntry(self, entry):
        self.entry = entry


class _RecordingNodemap:
    """Minimal nodemap: distinct recording node per name (MagicMock collapses
    them all to one return_value, so a real fake is needed to tell them apart).
    Pre-seed specific nodes via `preset`; unknown names auto-create."""

    def __init__(self, preset=None):
        self.nodes: dict[str, _RecordingNode] = dict(preset or {})

    def FindNode(self, name):
        return self.nodes.setdefault(name, _RecordingNode())


class _RecordingDataStream:
    """Records the timeout passed to WaitForFinishedBuffer, then raises to
    short-circuit the rest of the grab (we only assert the timeout arg)."""

    def __init__(self):
        self.timeout_arg = None

    def WaitForFinishedBuffer(self, timeout):
        self.timeout_arg = timeout
        raise RuntimeError('short-circuit after recording the timeout')


class TestSignificantBitsFromFormat:
    @pytest.mark.parametrize(
        'wire,expected',
        [
            ('Mono12g24IDS', 12),
            ('Mono12p', 12),
            ('Mono12', 12),
            ('Mono10g40IDS', 10),
            ('Mono10p', 10),
            ('Mono10', 10),
            ('Mono8', 8),
            ('Mono8g', 8),
            ('Mono16', 16),  # leading-bit-count fallback for an unprefixed name
            ('BayerRG8', 8),  # no Mono token -> safe 8-bit default
            ('', 8),
        ],
    )
    def test_significant_bits(self, wire, expected):
        assert ids_significant_bits(wire) == expected


class TestIplTarget:
    def test_mono12_unpacks_to_mono12(self):
        assert _ids_ipl_target('Mono12g24IDS') is ids_peak_ipl.PixelFormatName_Mono12

    def test_mono10_unpacks_to_mono10(self):
        assert _ids_ipl_target('Mono10g40IDS') is ids_peak_ipl.PixelFormatName_Mono10

    def test_mono8_stays_mono8(self):
        assert _ids_ipl_target('Mono8') is ids_peak_ipl.PixelFormatName_Mono8


class TestDepthContract:
    """The driver no longer pins depth to 8; the container is the inherited
    16-bit width and the payload depth is derived from the active format."""

    def test_native_bit_depth_is_container_width_16(self):
        assert IDSCamera.native_bit_depth == 16

    @pytest.mark.parametrize(
        'wire,expected',
        [('Mono12g24IDS', 12), ('Mono10g40IDS', 10), ('Mono8', 8)],
    )
    def test_significant_bits_property_tracks_format(self, wire, expected):
        cam = bare_ids_camera()
        cam._pixel_format_cache = wire
        assert cam.significant_bits == expected


class TestFrameRateCap:
    """set_max_acquisition_frame_rate is the manual rate-limiter lever (char
    tool / API video cap). The driver itself free-runs and only ever calls this
    with enabled=False; these cover the lever's own enable/disable behavior."""

    def test_enable_writes_target_and_enable(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap()
        cam.set_max_acquisition_frame_rate(True, 16.0)
        nodes = cam.remote_nodemap.nodes
        assert nodes['AcquisitionFrameRateTargetEnable'].value is True
        assert nodes['AcquisitionFrameRateTarget'].value == 16.0

    def test_disable_skips_target_write(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap()
        cam.set_max_acquisition_frame_rate(False)
        nodes = cam.remote_nodemap.nodes
        assert nodes['AcquisitionFrameRateTargetEnable'].value is False
        # Target is only written when enabled.
        assert 'AcquisitionFrameRateTarget' not in nodes

    def test_inactive_camera_is_a_noop(self):
        cam = bare_ids_camera()
        cam.active = False
        cam.remote_nodemap = _RecordingNodemap()
        cam.set_max_acquisition_frame_rate(True, 16.0)
        assert cam.remote_nodemap.nodes == {}


class TestFreeRunConfig:
    """_configure_free_run lifts every throttle so the camera free-runs:
    DeviceLinkThroughputLimitComponent -> Link (the keystone -- in the default
    Sensor mode the limit is computed against the full raw readout and caps
    fps), the link limit to its max, the rate-target limiter off, and
    AcquisitionFrameRate to its max."""

    def _cam(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap(
            {
                'DeviceLinkThroughputLimit': _RecordingNode(maximum=400_000_000),
                'AcquisitionFrameRate': _RecordingNode(maximum=45.0),
            }
        )
        return cam

    def test_sets_throughput_component_to_link(self):
        cam = self._cam()
        cam._configure_free_run()
        assert cam.remote_nodemap.nodes['DeviceLinkThroughputLimitComponent'].entry == 'Link'

    def test_maximizes_throughput_limit(self):
        cam = self._cam()
        cam._configure_free_run()
        assert cam.remote_nodemap.nodes['DeviceLinkThroughputLimit'].value == 400_000_000

    def test_disables_the_rate_target_limiter(self):
        cam = self._cam()
        cam._configure_free_run()
        assert cam.remote_nodemap.nodes['AcquisitionFrameRateTargetEnable'].value is False

    def test_maximizes_acquisition_frame_rate(self):
        cam = self._cam()
        cam._configure_free_run()
        assert cam.remote_nodemap.nodes['AcquisitionFrameRate'].value == 45.0

    def test_inactive_camera_writes_nothing(self):
        cam = self._cam()
        cam.active = False
        cam._configure_free_run()
        assert cam.remote_nodemap.nodes['AcquisitionFrameRate'].value is _RecordingNode._UNSET


class TestGainDbConversion:
    """The IDS Gain node is a linear multiplier; LVP drives gain in dB. The
    driver converts dB <-> factor so 0 dB maps to the node's 1.0x unity floor
    (the previous unconverted 0.0 write was rejected as out-of-range)."""

    def _cam_with_gain_node(self, value=1.0, minimum=1.0, maximum=31.62):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap(
            {
                'Gain': _RecordingNode(value=value, minimum=minimum, maximum=maximum),
                'GainSelector': _RecordingNode(),
            }
        )
        return cam

    def test_zero_db_maps_to_unity_factor(self):
        cam = self._cam_with_gain_node()
        assert cam.gain(0.0) is True
        assert cam.remote_nodemap.nodes['Gain'].value == pytest.approx(1.0)

    def test_twenty_db_maps_to_ten_x(self):
        cam = self._cam_with_gain_node()
        assert cam.gain(20.0) is True
        assert cam.remote_nodemap.nodes['Gain'].value == pytest.approx(10.0)

    def test_thirty_db_maps_to_full_scale_factor(self):
        cam = self._cam_with_gain_node()
        assert cam.gain(30.0) is True
        assert cam.remote_nodemap.nodes['Gain'].value == pytest.approx(31.62, abs=0.05)

    def test_selects_analog_all(self):
        cam = self._cam_with_gain_node()
        cam.gain(6.0)
        assert cam.remote_nodemap.nodes['GainSelector'].entry == 'AnalogAll'

    def test_get_gain_returns_db(self):
        cam = self._cam_with_gain_node(value=10.0)
        assert cam.get_gain() == pytest.approx(20.0)  # 20*log10(10)

    def test_capability_range_reported_in_db(self):
        cam = bare_ids_camera()
        cam.profile = CameraProfile()
        cam.remote_nodemap = _RecordingNodemap(
            {
                'Gain': _RecordingNode(value=1.0, minimum=1.0, maximum=31.62),
                'ExposureTime': _RecordingNode(value=1e4, minimum=20.0, maximum=2e6),
            }
        )
        cam._query_dynamic_capabilities()
        assert cam.profile.gain.total_min_db == pytest.approx(0.0, abs=0.01)
        assert cam.profile.gain.total_max_db == pytest.approx(30.0, abs=0.1)


class TestGrabNewCaptureTimeout:
    """grab_new_capture takes float seconds but WaitForFinishedBuffer wants an
    integer millisecond timeout -- passing the float made every capture-path
    grab fail with a SWIG type error."""

    def test_passes_integer_millisecond_timeout(self):
        cam = bare_ids_camera()
        cam.cam_image_handler = object()  # only needs to be non-None
        cam.data_stream = _RecordingDataStream()
        ok, _ts = cam.grab_new_capture(3.0)
        assert ok is False  # the fake raises after recording
        assert cam.data_stream.timeout_arg == 3000
        assert isinstance(cam.data_stream.timeout_arg, int)


class _FakeBuffer:
    """A finished/incomplete GenTL buffer stand-in (identity == the buffer)."""

    def __init__(self, tag, complete=True):
        self.tag = tag
        self._complete = complete

    def IsIncomplete(self):
        return not self._complete

    def SizeFilled(self):
        return 1

    def Size(self):
        return 2

    def __repr__(self):
        return f'<FakeBuffer {self.tag}>'


class TestLatestBufferSlot:
    """The newest-wins buffer handoff: a put() that lands on an unconsumed
    buffer displaces the older one and re-queues it (so the converter only ever
    unpacks the freshest buffer and no displaced buffer leaks from the pool),
    and stop() re-queues a still-held buffer before reporting the sentinel."""

    def _slot(self):
        from drivers.idscamera import _LatestBufferSlot

        requeued = []
        return _LatestBufferSlot(requeued.append), requeued

    def test_put_then_get_returns_buffer_without_requeue(self):
        slot, rq = self._slot()
        slot.put('a')
        assert slot.get(timeout=0) == 'a'
        assert rq == []  # nothing displaced -> nothing re-queued

    def test_newest_wins_requeues_the_displaced_buffer(self):
        slot, rq = self._slot()
        slot.put('old')
        slot.put('new')
        assert slot.get(timeout=0) == 'new'
        assert rq == ['old']  # the stale buffer went back to the pool
        assert slot.dropped == 1
        assert slot.get(timeout=0) is None

    def test_get_on_empty_returns_none_on_timeout(self):
        slot, _rq = self._slot()
        assert slot.get(timeout=0) is None

    def test_stop_empty_returns_stop_sentinel(self):
        from drivers.idscamera import _LatestBufferSlot

        slot, rq = self._slot()
        slot.stop()
        assert slot.get(timeout=0) is _LatestBufferSlot._STOP
        assert rq == []

    def test_stop_requeues_a_held_buffer(self):
        from drivers.idscamera import _LatestBufferSlot

        slot, rq = self._slot()
        slot.put('held')
        slot.stop()
        # The held buffer goes back to the pool (not to the worker), then STOP.
        assert rq == ['held']
        assert slot.get(timeout=0) is _LatestBufferSlot._STOP

    def test_get_blocks_until_put(self):
        slot, _rq = self._slot()
        out = []

        def consume():
            out.append(slot.get(timeout=2.0))

        t = threading.Thread(target=consume)
        t.start()
        time.sleep(0.05)  # let the consumer park in get()
        slot.put('arrived')
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert out == ['arrived']


class _FakeDataStream:
    """A scripted buffer source for the two-stage pipeline lifecycle tests.

    WaitForFinishedBuffer yields each scripted buffer in turn, then blocks until
    KillWait/stop releases it (mirroring the real blocking poll). Records
    QueueBuffer / KillWait / FlushPendingKillWaits so teardown + re-queue counts
    are assertable; QueueBuffer is called from both the poll and worker threads,
    so its record is lock-guarded.
    """

    def __init__(self, buffers):
        self._pending = list(buffers)
        self._gate = threading.Event()
        self._lock = threading.Lock()
        self.requeued = []
        self.kill_wait_calls = 0
        self.flush_calls = 0

    def WaitForFinishedBuffer(self, timeout):
        if self._pending:
            return self._pending.pop(0)
        # Out of scripted buffers: park until KillWait releases us, then signal
        # the abort the real SDK raises so the poll loop tears down cleanly.
        self._gate.wait(timeout=5)
        raise RuntimeError('aborted: KillWait')

    def QueueBuffer(self, buffer):
        with self._lock:
            self.requeued.append(buffer)

    def KillWait(self):
        self.kill_wait_calls += 1
        self._gate.set()

    def FlushPendingKillWaits(self):
        self.flush_calls += 1


def _ids_handler(data_stream):
    """An ImageHandler with the SDK unpack seam replaced by a deterministic
    fake, so the tests cover the threading/handoff/buffer-lifecycle (the
    Mac-testable core) without the real ids_peak SDK."""
    from drivers import idscamera

    handler = idscamera.ImageHandler.__new__(idscamera.ImageHandler)
    idscamera.ImageHandlerBase.__init__(handler)
    handler.data_stream = data_stream
    handler.timeout_ms = 50
    handler._parent = MagicMock()
    handler._stop_event = threading.Event()
    handler._requeue_lock = threading.Lock()
    handler._slot = idscamera._LatestBufferSlot(handler._requeue)
    handler._poll_thread = None
    handler._worker_thread = None
    return handler


class TestSpuriousAbortKeepsPolling:
    """_handle_wait_error is reached only AFTER _poll_loop checks the stop event,
    so an AbortedException here is never our own teardown -- it is a KillWait
    leaked by a previous stop() that outlived its FlushPendingKillWaits and
    landed on a freshly (re)started poll thread's first wait (rapid binning
    toggles). The handler must flush it and keep polling, not kill the thread,
    which stranded the live view with no frames after fast toggles."""

    def test_abort_without_stop_request_flushes_and_continues(self):
        ds = MagicMock()
        h = _ids_handler(ds)
        assert not h._stop_event.is_set()  # stop was NOT requested

        should_stop = h._handle_wait_error(RuntimeError('WaitForFinishedBuffer aborted'))

        assert should_stop is False  # keep polling -- do NOT break the loop
        ds.FlushPendingKillWaits.assert_called_once()

    def test_non_abort_error_is_not_flushed_as_spurious(self):
        # A genuine fault must not be mistaken for a spurious abort: no flush,
        # and it still counts toward the disconnect threshold via _record_failure.
        ds = MagicMock()
        h = _ids_handler(ds)

        h._handle_wait_error(RuntimeError('malformed buffer'))

        ds.FlushPendingKillWaits.assert_not_called()


class TestPipelineLifecycle:
    """Stage A (poll -> slot) / Stage B (unpack -> store -> re-queue) wiring, with
    the key invariant: EVERY finished buffer is re-queued exactly once -- the
    worker re-queues what it unpacks, the slot re-queues what newest-wins
    displaces, the poll loop re-queues an incomplete buffer. stop() unblocks the
    parked poll via KillWait and joins both threads. _unpack is faked so these
    cover the threading + buffer lifecycle without the SDK."""

    @staticmethod
    def _wait_until(pred, timeout=2.0):
        deadline = time.time() + timeout
        while not pred() and time.time() < deadline:
            time.sleep(0.01)

    def test_complete_buffer_stored_and_requeued_once(self):
        b0 = _FakeBuffer('b0')
        ds = _FakeDataStream([b0])
        h = _ids_handler(ds)
        stored = []
        h._unpack = lambda buf: (buf.tag, 12)
        h._store_frame = lambda img, ts, *, significant_bits: stored.append((img, significant_bits))
        h.start()
        self._wait_until(lambda: stored)
        h.stop()
        assert stored == [('b0', 12)]
        assert ds.requeued.count(b0) == 1  # re-queued once, by the worker
        assert ds.kill_wait_calls == 1  # stop() unblocked the parked poll
        assert h._poll_thread is None and h._worker_thread is None

    def test_incomplete_buffer_requeued_once_not_stored(self):
        bad = _FakeBuffer('bad', complete=False)
        ds = _FakeDataStream([bad])
        h = _ids_handler(ds)
        stored = []
        h._unpack = lambda buf: (buf.tag, 12)
        h._store_frame = lambda *a, **k: stored.append(a)
        h.start()
        self._wait_until(lambda: bad in ds.requeued)
        h.stop()
        assert stored == []  # nothing stored for an incomplete buffer
        assert ds.requeued.count(bad) == 1  # re-queued once, by the poll loop

    def test_newest_wins_requeues_every_buffer_exactly_once(self):
        bufs = [_FakeBuffer(f'b{i}') for i in range(3)]
        ds = _FakeDataStream(list(bufs))
        h = _ids_handler(ds)
        release = threading.Event()
        stored = []

        def slow_unpack(buf):
            release.wait(timeout=5)  # hold the worker so a backlog forms
            return buf.tag, 12

        h._unpack = slow_unpack
        h._store_frame = lambda img, ts, *, significant_bits: stored.append(img)
        h.start()
        time.sleep(0.3)  # all three drain while the worker is held on b0
        release.set()
        time.sleep(0.3)
        h.stop()
        # b0 was in-flight when the stall hit; b1 is displaced by b2 in the slot,
        # so the worker stores b0 then the freshest (b2), never b1.
        assert 'b2' in stored
        assert 'b1' not in stored
        assert h._slot.dropped >= 1
        # The invariant: every buffer returned to the pool exactly once, whether
        # unpacked (worker) or displaced (slot).
        for b in bufs:
            assert ds.requeued.count(b) == 1

    def test_worker_survives_unpack_exception_and_still_requeues(self):
        boom = _FakeBuffer('boom')
        good = _FakeBuffer('good')
        ds = _FakeDataStream([boom, good])
        h = _ids_handler(ds)
        stored = []

        def flaky_unpack(buf):
            if buf.tag == 'boom':
                raise ValueError('synthetic unpack failure')
            return buf.tag, 12

        h._unpack = flaky_unpack
        h._store_frame = lambda img, ts, *, significant_bits: stored.append(img)
        h.start()
        self._wait_until(lambda: 'good' in stored)
        h.stop()
        assert 'good' in stored  # the exception on 'boom' didn't kill the worker
        assert ds.requeued.count(boom) == 1  # re-queued via finally despite raising

    def test_stop_is_idempotent_and_joins(self):
        ds = _FakeDataStream([])
        h = _ids_handler(ds)
        h._unpack = lambda buf: (buf.tag, 12)
        h.start()
        time.sleep(0.05)
        h.stop()
        h.stop()  # second stop must not raise
        assert h._poll_thread is None and h._worker_thread is None
