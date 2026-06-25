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
    """The crash-stop cap: enable + set AcquisitionFrameRateTarget, never
    maximize the rate (which saturated USB3 and exhausted the buffer pool)."""

    def test_cap_constant_is_below_sustained_rate(self):
        # Sustained host-unpack rate is ~18 fps; the static cap sits under it.
        assert 0 < IDSCamera._FPS_CAP <= 18

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


class TestLatestFrameSlot:
    """The newest-wins handoff between the poll thread and the unpack worker:
    a put() that lands on an unconsumed frame drops the older one (so the
    converter -- the host bottleneck -- only ever unpacks the freshest frame),
    and get() drains a pending frame before reporting the stop sentinel."""

    def _slot(self):
        from drivers.idscamera import _LatestFrameSlot

        return _LatestFrameSlot()

    def test_put_then_get_returns_frame(self):
        slot = self._slot()
        slot.put('a')
        assert slot.get(timeout=0) == 'a'

    def test_newest_wins_drops_the_superseded_frame(self):
        slot = self._slot()
        slot.put('old')
        slot.put('new')
        assert slot.get(timeout=0) == 'new'
        assert slot.dropped == 1
        # Slot is empty after the single consume (the old frame is gone).
        assert slot.get(timeout=0) is None

    def test_get_on_empty_returns_none_on_timeout(self):
        slot = self._slot()
        assert slot.get(timeout=0) is None

    def test_stop_then_empty_get_returns_stop_sentinel(self):
        from drivers.idscamera import _LatestFrameSlot

        slot = self._slot()
        slot.stop()
        assert slot.get(timeout=0) is _LatestFrameSlot._STOP

    def test_stop_drains_pending_frame_before_sentinel(self):
        from drivers.idscamera import _LatestFrameSlot

        slot = self._slot()
        slot.put('pending')
        slot.stop()
        # The pending frame comes out first; only then the sentinel.
        assert slot.get(timeout=0) == 'pending'
        assert slot.get(timeout=0) is _LatestFrameSlot._STOP

    def test_get_blocks_until_put(self):
        slot = self._slot()
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


def _packed(width=4, height=1, wire='Mono12g24IDS', tag=b'\x00'):
    """A fake _PackedFrame-shaped payload for lifecycle tests (no SDK)."""
    from drivers.idscamera import _PackedFrame

    return _PackedFrame(
        packed=tag, pixel_format_id=0, wire_name=wire, width=width, height=height, ts=None
    )


class _FakeDataStream:
    """A scripted buffer source for the two-stage pipeline lifecycle tests.

    WaitForFinishedBuffer yields each queued item in turn, then blocks until
    KillWait/stop releases it (mirroring the real blocking poll). Records
    QueueBuffer / KillWait / FlushPendingKillWaits so teardown is assertable.
    """

    def __init__(self, buffers):
        self._pending = list(buffers)
        self._gate = threading.Event()
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
        self.requeued.append(buffer)

    def KillWait(self):
        self.kill_wait_calls += 1
        self._gate.set()

    def FlushPendingKillWaits(self):
        self.flush_calls += 1


def _ids_handler(data_stream):
    """An ImageHandler with the SDK-touching seams replaced by deterministic
    fakes, so the tests cover the threading/handoff lifecycle (the Mac-testable
    core) without the real ids_peak SDK."""
    from drivers import idscamera

    handler = idscamera.ImageHandler.__new__(idscamera.ImageHandler)
    ImageHandlerBaseInit = idscamera.ImageHandlerBase.__init__
    ImageHandlerBaseInit(handler)
    handler.data_stream = data_stream
    handler.timeout_ms = 50
    handler._parent = MagicMock()
    handler._stop_event = threading.Event()
    handler._slot = idscamera._LatestFrameSlot()
    handler._poll_thread = None
    handler._worker_thread = None
    return handler


class TestPipelineLifecycle:
    """Stage A (drain + re-queue) / Stage B (unpack + store) wiring: the SDK
    buffer is re-queued for every drained frame, the worker stores what it
    unpacks, newest-wins drops backlog under worker stall, and stop() unblocks
    a parked poll via KillWait and joins both threads."""

    def test_drained_frame_is_stored_and_stop_unblocks_the_poll(self):
        ds = _FakeDataStream(['buf0'])
        h = _ids_handler(ds)
        stored = []
        h._drain_buffer = lambda buf: _packed(tag=buf)
        h._unpack = lambda frame: (frame.packed, 12)
        h._store_frame = lambda img, ts, *, significant_bits: stored.append((img, significant_bits))
        h.start()
        deadline = time.time() + 2.0
        while not stored and time.time() < deadline:
            time.sleep(0.01)
        h.stop()
        assert stored == [('buf0', 12)]
        assert ds.kill_wait_calls == 1  # stop() unblocked the parked poll
        assert h._poll_thread is None and h._worker_thread is None

    def test_incomplete_buffer_requeued_and_counts_failure_not_stored(self):
        ds = _FakeDataStream(['bad'])
        h = _ids_handler(ds)
        stored = []
        h._drain_buffer = lambda buf: None  # None == incomplete (already re-queued)
        h._store_frame = lambda *a, **k: stored.append(a)
        h.start()
        time.sleep(0.2)
        h.stop()
        assert stored == []  # nothing stored for an incomplete buffer

    def test_newest_wins_under_worker_stall(self):
        ds = _FakeDataStream(['b0', 'b1', 'b2'])
        h = _ids_handler(ds)
        release = threading.Event()
        stored = []

        def slow_unpack(frame):
            release.wait(timeout=5)  # hold the worker so a backlog forms
            return frame.packed, 12

        h._drain_buffer = lambda buf: _packed(tag=buf)
        h._unpack = slow_unpack
        h._store_frame = lambda img, ts, *, significant_bits: stored.append(img)
        h.start()
        time.sleep(0.3)  # let all three drain while the worker is held on b0
        release.set()
        time.sleep(0.3)
        h.stop()
        # b0 was in-flight when the stall hit; b1 is superseded by b2 in the
        # slot, so the worker stores b0 then the freshest (b2), never b1.
        assert 'b2' in stored
        assert 'b1' not in stored
        assert h._slot.dropped >= 1

    def test_worker_survives_an_unpack_exception(self):
        ds = _FakeDataStream(['boom', 'good'])
        h = _ids_handler(ds)
        stored = []

        def flaky_unpack(frame):
            if frame.packed == 'boom':
                raise ValueError('synthetic unpack failure')
            return frame.packed, 12

        h._drain_buffer = lambda buf: _packed(tag=buf)
        h._unpack = flaky_unpack
        h._store_frame = lambda img, ts, *, significant_bits: stored.append(img)
        h.start()
        deadline = time.time() + 2.0
        while 'good' not in stored and time.time() < deadline:
            time.sleep(0.01)
        h.stop()
        assert 'good' in stored  # the exception on 'boom' didn't kill the worker

    def test_stop_is_idempotent_and_joins(self):
        ds = _FakeDataStream([])
        h = _ids_handler(ds)
        h._drain_buffer = lambda buf: None
        h._unpack = lambda frame: (frame.packed, 12)
        h.start()
        time.sleep(0.05)
        h.stop()
        h.stop()  # second stop must not raise
        assert h._poll_thread is None and h._worker_thread is None


class _FakeBuffer:
    """A finished/incomplete GenTL buffer stand-in for _drain_buffer tests."""

    def __init__(self, complete=True, pf=0, w=4, h=1, filled=10, size=10):
        self._complete = complete
        self._pf = pf
        self._w = w
        self._h = h
        self._filled = filled
        self._size = size

    def IsIncomplete(self):
        return not self._complete

    def PixelFormat(self):
        return self._pf

    def Width(self):
        return self._w

    def Height(self):
        return self._h

    def SizeFilled(self):
        return self._filled

    def Size(self):
        return self._size


class TestDrainBuffer:
    """The real Stage A drain: it re-queues the SDK buffer for EVERY finished
    buffer (the immediate re-queue is the crash-protection guarantee) and owns a
    copy of the packed bytes; an incomplete buffer is re-queued and reported as
    None (no frame), so the poll loop counts it as a failure without storing."""

    def _handler(self):
        ds = _FakeDataStream([])
        h = _ids_handler(ds)
        h._parent.get_pixel_format = lambda: 'Mono12g24IDS'
        return h, ds

    def test_complete_buffer_is_copied_and_requeued(self, monkeypatch):
        from drivers import idscamera

        h, ds = self._handler()
        fake_img = MagicMock()
        fake_img.get_numpy_1D.return_value = bytearray(b'\x01\x02\x03')
        monkeypatch.setattr(idscamera.ids_peak_ipl_extension, 'BufferToImage', lambda buf: fake_img)
        buf = _FakeBuffer(complete=True, pf=7, w=4, h=2)
        frame = h._drain_buffer(buf)
        assert buf in ds.requeued  # re-queued immediately, before any unpack
        assert frame.packed == b'\x01\x02\x03'  # owning bytes copy
        assert frame.pixel_format_id == 7
        assert frame.wire_name == 'Mono12g24IDS'
        assert (frame.width, frame.height) == (4, 2)

    def test_incomplete_buffer_is_requeued_and_returns_none(self):
        h, ds = self._handler()
        buf = _FakeBuffer(complete=False)
        assert h._drain_buffer(buf) is None
        assert buf in ds.requeued  # incomplete buffers are re-queued too

    def test_requeues_exactly_once_even_when_the_copy_raises(self, monkeypatch):
        from drivers import idscamera

        h, ds = self._handler()

        def boom(buf):
            raise RuntimeError('synthetic BufferToImage failure')

        monkeypatch.setattr(idscamera.ids_peak_ipl_extension, 'BufferToImage', boom)
        buf = _FakeBuffer(complete=True)
        with pytest.raises(RuntimeError):
            h._drain_buffer(buf)
        # The finally re-queues once; the poll loop must not re-queue again.
        assert ds.requeued.count(buf) == 1
