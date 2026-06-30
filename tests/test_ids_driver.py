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
from unittest.mock import MagicMock, call, patch

import ids_peak_ipl
import pytest

from drivers.camera_profiles import CameraProfile
from drivers.idscamera import (
    IDSCamera,
    _ids_delivery_significant_bits,
    _ids_delivery_target,
    _ids_ipl_target,
    ids_significant_bits,
)
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


class TestDeliveryTarget:
    """The production unpack delivers Mono10 (the 8-bit-mode wire) straight to
    8-bit in one ConvertTo pass; Mono12 (the 12-bit modes) stays native uint16."""

    def test_mono10_delivers_mono8_directly(self):
        # The win: no uint16 intermediate + no host downconvert in 8-bit mode.
        assert _ids_delivery_target('Mono10g40IDS') is ids_peak_ipl.PixelFormatName_Mono8

    def test_mono12_keeps_native_uint16(self):
        assert _ids_delivery_target('Mono12g24IDS') is ids_peak_ipl.PixelFormatName_Mono12

    def test_mono8_stays_mono8(self):
        assert _ids_delivery_target('Mono8') is ids_peak_ipl.PixelFormatName_Mono8

    def test_delivered_significant_bits(self):
        assert _ids_delivery_significant_bits('Mono10g40IDS') == 8
        assert _ids_delivery_significant_bits('Mono12g24IDS') == 12
        assert _ids_delivery_significant_bits('Mono8') == 8

    def test_mono10_delivery_depth_differs_from_native_wire_depth(self):
        # The Mono10 wire is 10-bit native but delivered as 8-bit in 8-bit mode.
        assert ids_significant_bits('Mono10g40IDS') == 10
        assert _ids_delivery_significant_bits('Mono10g40IDS') == 8


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
        # Connect resolves the analog selector from the live GainSelector enum
        # and caches it here; the simple _RecordingNode has no AvailableEntries,
        # so seed the cached value as connect would (the body exposes AnalogAll).
        cam._gain_selector = 'AnalogAll'
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

    def test_gain_at_cap_clamps_factor_to_node_maximum(self):
        """The reported max gain converts back to a factor a float epsilon ABOVE
        Gain.Maximum(), so the camera rejected its own cap (OUT_OF_RANGE). The
        written factor must be reconciled to the node maximum, never exceed it."""
        cam = self._cam_with_gain_node(maximum=31.622776)
        # 30 dB -> 10**(30/20) = 31.6227766..., just over the 31.622776 cap.
        assert cam.gain(30.0) is True
        written = cam.remote_nodemap.nodes['Gain'].value
        assert written <= 31.622776  # clamped, not the overshoot
        assert written == pytest.approx(31.622776)

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

    def test_exposure_below_live_min_clamps_to_floor(self):
        # The live-view slider's 0.01ms startup default is applied before the
        # saved exposure loads; 10us is below the camera's live minimum and an
        # unclamped SetValue throws OUT_OF_RANGE. exposure_t reconciles it UP to
        # the node Minimum() (mirroring the Pylon driver) and applies the floor.
        cam = bare_ids_camera()
        cam.profile = CameraProfile()
        cam.active = MagicMock()
        cam.cam_image_handler = None
        cam.remote_nodemap = _RecordingNodemap(
            {'ExposureTime': _RecordingNode(value=1e4, minimum=31.245791, maximum=2e6)}
        )
        assert cam.exposure_t(0.01) is True
        node = cam.remote_nodemap.nodes['ExposureTime']
        assert node.value == pytest.approx(31.245791)
        assert cam._last_exposure_ms == pytest.approx(31.245791 / 1000)

    def test_exposure_above_live_min_passes_through(self):
        cam = bare_ids_camera()
        cam.profile = CameraProfile()
        cam.active = MagicMock()
        cam.cam_image_handler = None
        cam.remote_nodemap = _RecordingNodemap(
            {'ExposureTime': _RecordingNode(value=1e4, minimum=31.245791, maximum=2e6)}
        )
        assert cam.exposure_t(10.0) is True  # 10ms = 10000us, well above the floor
        node = cam.remote_nodemap.nodes['ExposureTime']
        assert node.value == pytest.approx(10000.0)


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
    handler._frame_generation = 0
    handler._frame_gen_cond = threading.Condition()
    handler._stall_started = None
    handler._last_presence_probe = 0.0
    handler._absence_confirmations = 0
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


class TestStallNotRemoval:
    """Recovery contract: a poll-loop STALL (timeout / incomplete) is
    never promoted to a disconnect -- removal is owned solely by DeviceLost. The
    old 'N consecutive failures = removed' heuristic mislabeled a host stall as a
    removal; this pins that it no longer does."""

    def test_timeout_stall_never_marks_disconnected(self):
        ds = MagicMock()
        h = _ids_handler(ds)
        # Far more than any old consecutive-failure threshold.
        for _ in range(50):
            should_stop = h._handle_wait_error(RuntimeError('WaitForFinishedBuffer timeout'))
            assert should_stop is False  # keep polling
        h._parent._mark_disconnected.assert_not_called()
        h._parent._handle_device_lost.assert_not_called()

    def test_generic_fault_keeps_polling_without_disconnect(self):
        ds = MagicMock()
        h = _ids_handler(ds)
        should_stop = h._handle_wait_error(RuntimeError('malformed buffer'))
        assert should_stop is False
        h._parent._mark_disconnected.assert_not_called()
        h._parent._handle_device_lost.assert_not_called()

    def test_typed_device_lost_routes_to_single_owner(self):
        # The authoritative typed removal (fallback to the callback) stops the
        # loop and routes to the single removal owner, not a bare _mark_disconnected.
        ds = MagicMock()
        h = _ids_handler(ds)
        should_stop = h._handle_wait_error(RuntimeError('DeviceLostException: removed'))
        assert should_stop is True
        h._parent._handle_device_lost.assert_called_once()


class TestSustainedStallPresenceProbe:
    """DeviceLost never fires on a USB unplug for the U3-34L, so a real removal
    is indistinguishable from a wedged-but-present stall -- both just time out
    forever. The poll loop escalates a SUSTAINED stall to an active presence
    probe; these pin WHEN it probes and the confirm-twice debounce. A single
    timeout must not probe; only sustained no-frame time does, rate-limited; and
    removal needs repeated absent confirmations, not one truncated Update."""

    def _stalled(self, base, *, probe_returns=None):
        ds = MagicMock()
        h = _ids_handler(ds)
        h._parent._probe_device_presence.return_value = probe_returns
        with patch('drivers.idscamera.time.monotonic', return_value=base):
            h._handle_wait_error(RuntimeError('WaitForFinishedBuffer timeout'))
        # First timeout only arms the stall clock; it never probes.
        h._parent._probe_device_presence.assert_not_called()
        assert h._stall_started == base
        return h

    def _timeout_at(self, h, when):
        with patch('drivers.idscamera.time.monotonic', return_value=when):
            return h._handle_wait_error(RuntimeError('timeout'))

    def test_first_timeout_arms_without_probing(self):
        self._stalled(100.0)

    def test_short_stall_does_not_probe(self):
        from drivers import idscamera

        h = self._stalled(100.0)
        self._timeout_at(h, 100.0 + idscamera._PRESENCE_PROBE_STALL_S - 0.5)
        h._parent._probe_device_presence.assert_not_called()

    def test_sustained_stall_probes_once(self):
        from drivers import idscamera

        h = self._stalled(100.0, probe_returns=True)  # present -> no teardown
        self._timeout_at(h, 100.0 + idscamera._PRESENCE_PROBE_STALL_S + 0.1)
        h._parent._probe_device_presence.assert_called_once()
        h._parent._handle_device_lost.assert_not_called()

    def test_probe_is_rate_limited_then_repeats(self):
        from drivers import idscamera

        h = self._stalled(100.0, probe_returns=True)
        first = 100.0 + idscamera._PRESENCE_PROBE_STALL_S + 0.1
        self._timeout_at(h, first)
        # A timeout within the probe interval does NOT re-probe.
        self._timeout_at(h, first + idscamera._PRESENCE_PROBE_INTERVAL_S - 0.5)
        assert h._parent._probe_device_presence.call_count == 1
        # ...but once the interval has elapsed it probes again.
        self._timeout_at(h, first + idscamera._PRESENCE_PROBE_INTERVAL_S + 0.1)
        assert h._parent._probe_device_presence.call_count == 2

    def test_one_absent_probe_does_not_tear_down(self):
        # A single absent reading is not enough -- a truncated Update on a busy
        # but present USB3 stack must not remove a connected camera.
        from drivers import idscamera

        h = self._stalled(100.0, probe_returns=False)
        self._timeout_at(h, 100.0 + idscamera._PRESENCE_PROBE_STALL_S + 0.1)
        assert h._absence_confirmations == 1
        h._parent._handle_device_lost.assert_not_called()

    def test_consecutive_absent_probes_route_to_removal(self):
        from drivers import idscamera

        h = self._stalled(100.0, probe_returns=False)
        t = 100.0 + idscamera._PRESENCE_PROBE_STALL_S + 0.1
        stop = False
        for _ in range(idscamera._PRESENCE_PROBE_CONFIRMATIONS):
            stop = self._timeout_at(h, t)
            t += idscamera._PRESENCE_PROBE_INTERVAL_S + 0.1
        h._parent._handle_device_lost.assert_called_once()
        # The confirmed removal stops the poll loop, like the typed-removal path.
        assert stop is True

    def test_inconclusive_probe_does_not_advance_streak(self):
        from drivers import idscamera

        h = self._stalled(100.0, probe_returns=None)  # inconclusive (e.g. read fault)
        stop = self._timeout_at(h, 100.0 + idscamera._PRESENCE_PROBE_STALL_S + 0.1)
        assert h._absence_confirmations == 0
        assert stop is False
        h._parent._handle_device_lost.assert_not_called()

    def test_a_present_reading_resets_the_absence_streak(self):
        # Absent, then present (transient hiccup recovered), then absent again
        # must NOT reach the 2-confirmation threshold on the trailing absent.
        from drivers import idscamera

        h = self._stalled(100.0, probe_returns=False)
        t = 100.0 + idscamera._PRESENCE_PROBE_STALL_S + 0.1
        self._timeout_at(h, t)  # absent #1 -> streak 1
        assert h._absence_confirmations == 1
        h._parent._probe_device_presence.return_value = True  # present
        t += idscamera._PRESENCE_PROBE_INTERVAL_S + 0.1
        self._timeout_at(h, t)  # present -> streak resets to 0
        assert h._absence_confirmations == 0
        h._parent._probe_device_presence.return_value = False  # absent again
        t += idscamera._PRESENCE_PROBE_INTERVAL_S + 0.1
        self._timeout_at(h, t)  # absent #1 of a fresh streak
        assert h._absence_confirmations == 1
        h._parent._handle_device_lost.assert_not_called()

    def test_frame_arrival_resets_stall_and_streak(self):
        # A finished buffer means the device is delivering: the poll loop clears
        # the stall clock AND any partial absence streak, so a later isolated
        # timeout starts fresh rather than inheriting stale accrual.
        ds = MagicMock()
        h = _ids_handler(ds)
        h._stall_started = 999.0  # as if mid-stall
        h._absence_confirmations = 1

        buf = _FakeBuffer('f1', complete=True)

        def _wait(_timeout):
            h._stop_event.set()  # break the loop right after the post-success reset
            return buf

        ds.WaitForFinishedBuffer.side_effect = _wait
        h._poll_loop()
        assert h._stall_started is None
        assert h._absence_confirmations == 0

    def test_start_resets_a_stale_stall_clock(self):
        # The handler is reused across stop/start; a stale stall clock would make
        # the first post-restart timeout probe immediately. start() clears it.
        # The poll/worker loops are no-oped so start() only spawns instantly-
        # returning threads -- this exercises the field reset, not live polling.
        ds = MagicMock()
        h = _ids_handler(ds)
        h._stall_started = 12345.0
        h._last_presence_probe = 12345.0
        h._absence_confirmations = 1
        h._poll_loop = lambda: None
        h._worker_loop = lambda: None
        h.start()
        try:
            assert h._stall_started is None
            assert h._last_presence_probe == 0.0
            assert h._absence_confirmations == 0
        finally:
            h.stop()


class TestProbeDevicePresence:
    """The presence probe is a PURE present/absent/inconclusive query: it never
    latches removal or calls _handle_device_lost (the caller's confirm-twice
    debounce owns routing). It returns None (inconclusive) while a reset is in
    flight or teardown has begun and on an enumeration fault, and restores the
    update timeout so a later connect() never inherits the tightened value."""

    def _cam(self, *, serial='SN123'):
        cam = bare_ids_camera()
        cam._async_teardown_started = False
        cam._device_serial = serial
        cam.device_manager = MagicMock()
        cam._handle_device_lost = MagicMock()
        return cam

    def _descriptor(self, serial):
        d = MagicMock()
        d.SerialNumber.return_value = serial
        return d

    def test_present_device_returns_true(self):
        cam = self._cam(serial='SN123')
        cam.device_manager.Devices.return_value = [self._descriptor('SN123')]
        assert cam._probe_device_presence() is True
        cam._handle_device_lost.assert_not_called()

    def test_absent_device_returns_false_without_routing(self):
        cam = self._cam(serial='SN123')
        cam.device_manager.Devices.return_value = [self._descriptor('OTHER')]
        assert cam._probe_device_presence() is False
        cam._handle_device_lost.assert_not_called()

    def test_empty_enumeration_returns_false(self):
        cam = self._cam(serial='SN123')
        cam.device_manager.Devices.return_value = []
        assert cam._probe_device_presence() is False

    def test_descriptor_read_fault_is_inconclusive_not_absent(self):
        # Our device may BE present but its descriptor's SerialNumber() raises; a
        # read fault without a clean match must be inconclusive (None), never
        # absent (False) -- otherwise two such faults would false-tear-down a
        # present camera despite the confirm-twice debounce.
        cam = self._cam(serial='SN123')
        bad = MagicMock()
        bad.SerialNumber.side_effect = RuntimeError('GenTL descriptor read failed')
        cam.device_manager.Devices.return_value = [bad]
        assert cam._probe_device_presence() is None
        cam._handle_device_lost.assert_not_called()

    def test_restores_update_timeout_after_probe(self):
        from drivers import idscamera

        cam = self._cam(serial='SN123')
        cam.device_manager.Devices.return_value = [self._descriptor('SN123')]
        cam._probe_device_presence()
        # Last SetDeviceUpdateTimeout call restores the generous recovery value,
        # so a later connect()/recovery Update() does not inherit the 500ms probe
        # timeout and miss a present-but-slow camera.
        cam.device_manager.SetDeviceUpdateTimeout.assert_called_with(
            idscamera._RECOVERY_UPDATE_TIMEOUT_MS
        )

    def test_skipped_during_recovery(self):
        cam = self._cam()
        cam._in_recovery = True
        cam.device_manager.Devices.return_value = []
        assert cam._probe_device_presence() is None
        cam.device_manager.Update.assert_not_called()

    def test_skipped_when_disconnect_requested(self):
        cam = self._cam()
        cam._disconnect_requested = True
        assert cam._probe_device_presence() is None
        cam.device_manager.Update.assert_not_called()

    def test_skipped_when_teardown_started(self):
        cam = self._cam()
        cam._async_teardown_started = True
        assert cam._probe_device_presence() is None
        cam.device_manager.Update.assert_not_called()

    def test_update_fault_is_inconclusive(self):
        # An Update/enumeration fault is not proof of removal; inconclusive.
        cam = self._cam()
        cam.device_manager.Update.side_effect = RuntimeError('GenTL update failed')
        assert cam._probe_device_presence() is None
        cam._handle_device_lost.assert_not_called()


class TestWedgedBufferBreaksLoop:
    """A finished buffer whose handle is invalid (the data stream wedged out from
    under the live consumer) makes buffer.IsIncomplete() raise. That access is
    NOT inside the WaitForFinishedBuffer try, so the raise used to escape
    _poll_loop and kill the IDSPoll thread with a traceback. The handler must
    classify the invalid handle as a wedge: break the loop (no hot-spin on a dead
    stream), route the log through the camera log, and escalate to in-software
    recovery -- never mark disconnected (a wedge is recoverable; removal is owned
    solely by DeviceLost)."""

    def test_invalid_buffer_access_breaks_poll_loop_without_propagating(self):
        bad = MagicMock()
        bad.IsIncomplete.side_effect = RuntimeError('bufferHandle is invalid')
        ds = _FakeDataStream([bad])
        h = _ids_handler(ds)
        # Direct (synchronous) poll: the invalid-handle raise must be caught and
        # break the loop, so the call RETURNS instead of propagating the error.
        h._poll_loop()
        assert ds.requeued == []  # never touch an invalid handle to re-queue it
        h._parent._schedule_async_recovery.assert_called_once()  # escalate to recovery
        h._parent._mark_disconnected.assert_not_called()
        h._parent._handle_device_lost.assert_not_called()

    def test_handle_buffer_error_breaks_logs_and_escalates_to_recovery(self):
        from drivers import idscamera

        ds = MagicMock()
        h = _ids_handler(ds)
        log = MagicMock()
        with patch.object(idscamera, '_cam_log', log):
            should_stop = h._handle_buffer_error(RuntimeError('bufferHandle is invalid'))
        assert should_stop is True  # break the loop -- do NOT catch-and-continue
        log.error.assert_called_once()  # routed through the camera log
        h._parent._schedule_async_recovery.assert_called_once()  # in-software recovery
        h._parent._mark_disconnected.assert_not_called()
        h._parent._handle_device_lost.assert_not_called()

    def test_buffer_device_lost_routes_to_single_owner(self):
        # A removal surfacing as a buffer-access raise (the DeviceLost callback was
        # missed) must route to the single removal owner -- mirrors the typed
        # fallback _handle_wait_error carries -- not be logged as a mere wedge.
        ds = MagicMock()
        h = _ids_handler(ds)
        should_stop = h._handle_buffer_error(RuntimeError('DeviceLostException: removed'))
        assert should_stop is True
        h._parent._handle_device_lost.assert_called_once()

    def test_generic_buffer_error_keeps_polling(self):
        # An unknown, non-removal, non-wedge fault must not tear the poll thread
        # down for good: keep polling (removal is owned solely by DeviceLost).
        ds = MagicMock()
        h = _ids_handler(ds)
        should_stop = h._handle_buffer_error(RuntimeError('some transient fault'))
        assert should_stop is False
        h._parent._mark_disconnected.assert_not_called()
        h._parent._handle_device_lost.assert_not_called()


class TestStartGrabbingRollback:
    """start_grabbing() brackets acquisition with TLParamsLocked 1/0 and announces
    a buffer pool. If a step raises partway, it must roll back to a clean stopped
    state: otherwise TLParamsLocked stays set (disconnect() skips stop_grabbing()
    because is_grabbing() is still False) and the announced buffers leak, growing
    the pool on every retry."""

    def _cam(self):
        cam = bare_ids_camera()
        cam.is_grabbing = MagicMock(return_value=False)
        cam._configure_free_run = MagicMock()
        cam.cam_image_handler = MagicMock()
        cam.data_stream.NumBuffersAnnouncedMinRequired.return_value = 1
        return cam

    def test_acquisition_failure_rolls_back_lock_and_buffers(self):
        cam = self._cam()
        announced = [MagicMock(), MagicMock()]
        cam.data_stream.AnnouncedBuffers.return_value = announced
        cam.data_stream.StartAcquisition.side_effect = RuntimeError('start failed')
        cam.start_grabbing()
        # Transport lock released after it was taken (SetValue(1) then SetValue(0)).
        lock_node = cam.remote_nodemap.FindNode('TLParamsLocked')
        assert call(0) in lock_node.SetValue.call_args_list
        # Announced buffers revoked, not left to leak into the next start.
        assert cam.data_stream.RevokeBuffer.call_count == len(announced)
        # Handler quiesced (before any revoke) and acquisition stopped.
        cam.cam_image_handler.stop.assert_called_once()
        cam.data_stream.StopAcquisition.assert_called()

    def test_alloc_failure_revokes_announced_pool(self):
        cam = self._cam()
        announced = [MagicMock()]
        cam.data_stream.AnnouncedBuffers.return_value = announced
        cam.data_stream.AllocAndAnnounceBuffer.side_effect = RuntimeError('alloc failed')
        cam.start_grabbing()
        assert cam.data_stream.RevokeBuffer.call_count == len(announced)


def _ids_cam_for_recovery():
    cam = bare_ids_camera()
    cam._device_key = 'KEY-OURS'
    cam._device_serial = 'SN-OURS'
    cam._device_lost_callback = None
    cam._device_lost_callback_handle = None
    cam._async_teardown_started = False
    cam._in_recovery = False
    cam._recovery_started = False
    cam._recovery_attempts = 0
    cam._last_recovery_time = 0.0
    cam._recovery_abort = threading.Event()
    cam._disconnect_requested = False
    cam._recovery_thread = None
    cam.cam_image_handler = MagicMock()
    return cam


def _descriptor(serial, key='KEY-NEW'):
    d = MagicMock()
    d.SerialNumber.return_value = serial
    d.Key.return_value = key
    return d


class TestDeviceResetRecovery:
    """In-software recovery from a wedged data stream: the poll-loop wedge break
    escalates to _schedule_async_recovery (one-shot, daemon thread), which issues
    the SFNC DeviceReset, re-discovers the camera by serial, reopens, and restarts
    grabbing. A deliberate reset must NOT be torn down as a removal (the
    _in_recovery latch suppresses _handle_device_lost), and a recovery that fails
    falls back to the permanent teardown so the user sees a clean removal."""

    @staticmethod
    def _wait_until(pred, timeout=2.0):
        deadline = time.time() + timeout
        while not pred() and time.time() < deadline:
            time.sleep(0.01)

    def test_schedule_async_recovery_is_one_shot(self):
        cam = _ids_cam_for_recovery()
        gate = threading.Event()
        calls = []

        def _recover():
            calls.append(1)
            gate.wait(2.0)  # hold the first recovery in-flight across the 2nd schedule

        cam._recover_wedged_stream = _recover
        cam._schedule_async_recovery()
        self._wait_until(lambda: len(calls) >= 1)  # first recovery is now in-flight
        cam._schedule_async_recovery()  # latch still set -> no second recovery
        gate.set()
        time.sleep(0.05)
        assert len(calls) == 1

    def test_recovery_failure_falls_back_to_permanent_teardown(self):
        cam = _ids_cam_for_recovery()
        cam._recover_wedged_stream = MagicMock(side_effect=RuntimeError('reset failed'))
        cam._handle_device_lost = MagicMock()
        cam._schedule_async_recovery()
        self._wait_until(lambda: cam._handle_device_lost.call_count >= 1)
        cam._handle_device_lost.assert_called_once()  # clean removal on failed recovery
        assert cam._in_recovery is False  # latch cleared
        assert cam._recovery_started is False

    def test_in_recovery_suppresses_device_lost_teardown(self):
        cam = _ids_cam_for_recovery()
        cam._schedule_async_teardown = MagicMock()
        cam._in_recovery = True
        cam._handle_device_lost()
        # A DeviceLost firing during a deliberate reset must NOT tear the device
        # down -- recovery owns the reopen.
        cam._mark_disconnected.assert_not_called()
        cam._schedule_async_teardown.assert_not_called()

    def _recoverable_cam(self):
        cam = _ids_cam_for_recovery()
        cam.remote_nodemap.FindNode.return_value = MagicMock()
        cam.device_manager = MagicMock()
        cam._rediscover_by_serial = MagicMock(return_value=_descriptor('SN-OURS'))
        cam._unregister_device_callbacks = MagicMock()
        cam._register_device_callbacks = MagicMock()
        cam.init_camera_config = MagicMock()
        cam.start_grabbing = MagicMock()
        cam.is_grabbing = MagicMock(return_value=True)
        cam._snapshot_settings = MagicMock(return_value={'exposure_ms': 5})
        cam._restore_settings = MagicMock()
        return cam

    def test_recover_wedged_stream_resets_rediscovers_and_restarts(self):
        cam = self._recoverable_cam()
        reset_node = MagicMock()
        cam.remote_nodemap.FindNode.return_value = reset_node
        cam._recover_wedged_stream()
        # DeviceReset executed (Execute + WaitUntilDone) on the live control channel.
        reset_node.Execute.assert_called_once()
        reset_node.WaitUntilDone.assert_called_once()
        cam._rediscover_by_serial.assert_called_once_with('SN-OURS')
        cam._unregister_device_callbacks.assert_called_once()  # no callback leak
        cam._restore_settings.assert_called_once_with({'exposure_ms': 5})  # settings re-applied
        # Reopened against the re-discovered descriptor + restarted grabbing.
        assert cam.active is not None
        assert cam._device_key == 'KEY-NEW'  # key refreshed from the new descriptor
        cam.init_camera_config.assert_called_once()
        cam.start_grabbing.assert_called_once()

    def test_recover_wedged_stream_raises_when_rediscovery_fails(self):
        cam = self._recoverable_cam()
        cam._rediscover_by_serial = MagicMock(return_value=None)
        from drivers.exceptions import HardwareError

        with pytest.raises(HardwareError):
            cam._recover_wedged_stream()

    def test_recover_raises_without_captured_serial(self):
        cam = self._recoverable_cam()
        cam._device_serial = None  # connect() never read the serial
        from drivers.exceptions import HardwareError

        with pytest.raises(HardwareError):
            cam._recover_wedged_stream()

    def test_recover_raises_when_not_grabbing_after_reopen(self):
        cam = self._recoverable_cam()
        cam.is_grabbing = MagicMock(return_value=False)  # reconfiguration failed
        from drivers.exceptions import HardwareError

        with pytest.raises(HardwareError):
            cam._recover_wedged_stream()

    def test_recover_wedged_stream_bails_when_aborted(self):
        # A disconnect requested before recovery runs must abort it BEFORE the
        # irreversible DeviceReset -- never reboot a camera the user is closing.
        cam = self._recoverable_cam()
        reset_node = MagicMock()
        cam.remote_nodemap.FindNode.return_value = reset_node
        cam._recovery_abort.set()
        from drivers.exceptions import HardwareError

        with pytest.raises(HardwareError):
            cam._recover_wedged_stream()
        reset_node.Execute.assert_not_called()  # bailed before the reset
        cam._rediscover_by_serial.assert_not_called()

    def test_disconnect_signals_abort_and_joins_recovery(self):
        # disconnect() during an in-flight recovery signals the abort, joins the
        # recovery thread, and the camera ends torn down -- the user's disconnect
        # wins regardless of timing.
        cam = self._recoverable_cam()
        started = threading.Event()

        def _recover():
            started.set()
            for _ in range(300):
                if cam._recovery_abort.is_set():
                    raise RuntimeError('aborted by disconnect')
                time.sleep(0.01)

        cam._recover_wedged_stream = _recover
        cam._handle_device_lost = MagicMock()
        cam._schedule_async_recovery()
        self._wait_until(started.is_set)
        rec = cam._recovery_thread
        cam.disconnect()
        assert cam._disconnect_requested is True
        assert cam._recovery_abort.is_set()
        assert not rec.is_alive()  # disconnect joined the recovery thread
        assert cam.active is None

    def test_recovery_reopen_with_pending_disconnect_tears_back_down(self):
        # If a recovery reopens the camera despite a disconnect requested mid-flight
        # (it slipped past the abort checks), the finally latch tears the freshly
        # reopened camera back down so the disconnect is honored.
        cam = self._recoverable_cam()
        cam._recover_wedged_stream = MagicMock()  # succeeds -> recovered=True
        cam.disconnect = MagicMock()
        cam._disconnect_requested = True  # user asked to disconnect mid-recovery
        cam._schedule_async_recovery()
        self._wait_until(lambda: cam.disconnect.call_count >= 1)
        cam.disconnect.assert_called_once()

    def test_snapshot_and_restore_round_trip(self):
        cam = _ids_cam_for_recovery()
        cam.get_pixel_format = MagicMock(return_value='Mono12g24IDS')
        cam.get_binning_size = MagicMock(return_value=2)
        cam.get_frame_size = MagicMock(return_value={'width': 800, 'height': 600})
        cam.get_exposure_t = MagicMock(return_value=42.0)
        cam.get_gain = MagicMock(return_value=3.5)
        snap = cam._snapshot_settings()
        assert snap == {
            'pixel_format': 'Mono12g24IDS',
            'binning': 2,
            'frame_size': {'width': 800, 'height': 600},
            'exposure_ms': 42.0,
            'gain': 3.5,
        }
        cam.set_pixel_format = MagicMock()
        cam.set_binning_size = MagicMock()
        cam.set_frame_size = MagicMock()
        cam.exposure_t = MagicMock()
        cam.gain = MagicMock()
        cam._restore_settings(snap)
        cam.set_pixel_format.assert_called_once_with('Mono12g24IDS')
        cam.set_binning_size.assert_called_once_with(2)
        cam.set_frame_size.assert_called_once_with(800, 600)
        cam.exposure_t.assert_called_once_with(42.0)
        cam.gain.assert_called_once_with(3.5)

    def test_snapshot_drops_sentinel_reads(self):
        # The driver getters return sentinels (None / -1) on a failed read rather
        # than raising; those must be DROPPED, not captured and later re-applied
        # as a bad write that de-bins or zeroes the operator's settings.
        cam = _ids_cam_for_recovery()
        cam.get_pixel_format = MagicMock(return_value=None)  # sentinel
        cam.get_binning_size = MagicMock(return_value=2)  # valid
        cam.get_frame_size = MagicMock(return_value={'width': 800, 'height': 600})
        cam.get_exposure_t = MagicMock(return_value=-1)  # sentinel
        cam.get_gain = MagicMock(return_value=-1)  # sentinel
        snap = cam._snapshot_settings()
        assert snap == {'binning': 2, 'frame_size': {'width': 800, 'height': 600}}

    def test_get_binning_size_returns_minus_one_on_read_failure(self):
        # 1 is a LEGAL binning factor, so it cannot double as the failure
        # sentinel: an active read that throws must return the out-of-band -1
        # (rejected by the snapshot validator), not 1 (which would survive
        # validation and silently de-bin a 2x camera on restore).
        cam = bare_ids_camera()
        cam.remote_nodemap.FindNode.side_effect = Exception('node read failed')
        assert cam.get_binning_size() == -1

    def test_snapshot_drops_binning_read_failure(self):
        # A failed binning read (-1) during the recovery snapshot must be
        # DROPPED, not captured as 1 and re-applied by _restore_settings.
        cam = _ids_cam_for_recovery()
        cam.get_pixel_format = MagicMock(return_value='Mono8')
        cam.get_binning_size = MagicMock(return_value=-1)  # read failure
        cam.get_frame_size = MagicMock(return_value={'width': 800, 'height': 600})
        cam.get_exposure_t = MagicMock(return_value=10.0)
        cam.get_gain = MagicMock(return_value=0.0)
        snap = cam._snapshot_settings()
        assert 'binning' not in snap

    def test_is_connected_true_during_recovery_despite_null_active(self):
        # The reset transiently nulls active; is_connected must not report a
        # removal (or latch _device_removed) mid-recovery.
        cam = _ids_cam_for_recovery()
        cam.active = None
        cam._device_removed = False
        cam._in_recovery = True
        assert cam.is_connected() is True
        assert cam._device_removed is False

    def test_recovery_gives_up_after_max_attempts(self):
        from drivers import idscamera

        cam = _ids_cam_for_recovery()
        cam._recover_wedged_stream = MagicMock(side_effect=RuntimeError('still wedged'))
        cam._handle_device_lost = MagicMock()
        for _ in range(idscamera._RECOVERY_MAX_ATTEMPTS + 3):
            cam._schedule_async_recovery()
            self._wait_until(lambda: not cam._recovery_started)
        # Capped: only _RECOVERY_MAX_ATTEMPTS real resets; the rest go straight to
        # permanent teardown rather than loop forever.
        assert cam._recover_wedged_stream.call_count == idscamera._RECOVERY_MAX_ATTEMPTS

    def test_rediscover_by_serial_matches_the_right_camera(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam.device_manager.Devices.return_value = [
            _descriptor('SN-OTHER', key='K1'),
            _descriptor('SN-OURS', key='K2'),
        ]
        found = cam._rediscover_by_serial('SN-OURS', timeout_s=0.0)
        assert found is not None
        assert found.SerialNumber() == 'SN-OURS'

    def test_rediscover_by_serial_returns_none_when_absent(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam.device_manager.Devices.return_value = [_descriptor('SN-OTHER')]
        assert cam._rediscover_by_serial('SN-OURS', timeout_s=0.0) is None


class TestDeviceLostCallback:
    """The DeviceLost callback is the single owner of camera removal: registered
    with both wrapper + handle kept alive, filtered to our device key, marking
    disconnected inline and deferring close/destroy to a daemon thread."""

    def test_register_stores_wrapper_and_handle(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam._register_device_callbacks()
        cam.device_manager.DeviceLostCallback.assert_called_once_with(cam._on_device_lost)
        cam.device_manager.RegisterDeviceLostCallback.assert_called_once_with(
            cam._device_lost_callback
        )
        assert cam._device_lost_callback is not None  # wrapper kept alive
        assert cam._device_lost_callback_handle is not None  # handle kept alive

    def test_unregister_uses_handle_and_clears_refs(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        handle = object()
        cam._device_lost_callback = object()
        cam._device_lost_callback_handle = handle
        cam._unregister_device_callbacks()
        cam.device_manager.UnregisterDeviceLostCallback.assert_called_once_with(handle)
        assert cam._device_lost_callback is None
        assert cam._device_lost_callback_handle is None

    def test_unregister_is_idempotent_with_nothing_registered(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam._unregister_device_callbacks()  # no handle -> no SDK call, no raise
        cam.device_manager.UnregisterDeviceLostCallback.assert_not_called()

    def test_device_lost_for_our_key_triggers_removal(self):
        cam = _ids_cam_for_recovery()
        cam._handle_device_lost = MagicMock()
        cam._on_device_lost('KEY-OURS')
        cam._handle_device_lost.assert_called_once()

    def test_device_lost_for_other_key_is_ignored(self):
        cam = _ids_cam_for_recovery()
        cam._handle_device_lost = MagicMock()
        cam._on_device_lost('SOME-OTHER-CAMERA')
        cam._handle_device_lost.assert_not_called()

    def test_handle_device_lost_marks_and_schedules_teardown(self):
        cam = _ids_cam_for_recovery()
        cam._schedule_async_teardown = MagicMock()
        cam._handle_device_lost()
        cam._mark_disconnected.assert_called_once()
        cam._schedule_async_teardown.assert_called_once()

    def test_async_teardown_is_one_shot(self):
        cam = _ids_cam_for_recovery()
        cam.disconnect = MagicMock()
        cam._schedule_async_teardown()
        cam._schedule_async_teardown()  # second is a no-op via the latch
        deadline = time.time() + 2.0
        while cam.disconnect.call_count < 1 and time.time() < deadline:
            time.sleep(0.01)
        assert cam.disconnect.call_count == 1  # exactly once despite two schedules

    def test_disconnect_on_removed_device_skips_nodemap_stop(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam.active = MagicMock()
        cam._device_removed = True
        cam.stop_grabbing = MagicMock()
        cam.is_grabbing = MagicMock(return_value=True)
        handler = cam.cam_image_handler  # captured before disconnect nulls it
        cam.disconnect()
        cam.stop_grabbing.assert_not_called()  # nodemap stops skipped on dead handle
        handler.stop.assert_called_once()  # safe teardown still runs before release

    def test_disconnect_on_live_device_uses_stop_grabbing(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam.active = MagicMock()
        cam._device_removed = False
        cam.stop_grabbing = MagicMock()
        cam.is_grabbing = MagicMock(return_value=True)
        cam.disconnect()
        cam.stop_grabbing.assert_called_once()

    def test_disconnect_nulls_handler_to_release_stream(self):
        # disconnect() must drop cam_image_handler: it was constructed with the
        # data stream and ids_peak has no explicit Close() -- release happens by
        # dropping the last Python ref. Leaving the handler pinned keeps the USB3
        # endpoint bound, so a reconnect rebinds the same (possibly wedged) stream.
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam.active = MagicMock()
        cam._device_removed = False
        cam.stop_grabbing = MagicMock()
        cam.is_grabbing = MagicMock(return_value=True)
        cam.disconnect()
        assert cam.cam_image_handler is None

    def test_disconnect_on_removed_device_also_nulls_handler(self):
        cam = _ids_cam_for_recovery()
        cam.device_manager = MagicMock()
        cam.active = MagicMock()
        cam._device_removed = True
        cam.disconnect()
        assert cam.cam_image_handler is None


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
        # stop() unblocked the parked poll. The count is >= 1, not == 1: stop() is
        # designed for multiple KillWait+join rounds (_STOP_JOIN_CEILING_S), so the
        # exact number is scheduling-dependent, not a correctness property.
        assert ds.kill_wait_calls >= 1
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


class TestFrameGenerationGate:
    """grab_new_capture returns a NEW frame via the handler's frame-generation
    gate -- not by driving the data stream itself (the QueueBuffer-race fix)."""

    def test_worker_store_advances_generation(self):
        b0 = _FakeBuffer('b0')
        ds = _FakeDataStream([b0])
        h = _ids_handler(ds)
        h._unpack = lambda buf: (buf.tag, 12)
        h._store_frame = lambda *a, **k: None
        gen0 = h.frame_generation()
        h.start()
        TestPipelineLifecycle._wait_until(lambda: h.frame_generation() > gen0)
        h.stop()
        assert h.frame_generation() > gen0  # storing a frame advanced the counter

    def test_wait_for_new_frame_times_out_when_no_advance(self):
        ds = _FakeDataStream([])
        h = _ids_handler(ds)
        # No worker -> generation never advances -> the bounded wait returns False.
        assert h.wait_for_new_frame(h.frame_generation(), 0.05) is False

    def test_wait_for_new_frame_returns_true_after_bump(self):
        ds = _FakeDataStream([])
        h = _ids_handler(ds)
        since = h.frame_generation()

        def _bump():
            with h._frame_gen_cond:
                h._frame_generation += 1
                h._frame_gen_cond.notify_all()

        threading.Timer(0.02, _bump).start()
        assert h.wait_for_new_frame(since, 1.0) is True

    def test_grab_new_capture_does_not_drive_data_stream(self):
        # Structural guard: the fix is only safe if grab_new_capture never calls
        # data_stream.WaitForFinishedBuffer / QueueBuffer directly -- those belong
        # to the poll thread / _requeue, the single owners of the SDK stream.
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(IDSCamera.grab_new_capture))
        tree = ast.parse(src)
        bad = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in ('WaitForFinishedBuffer', 'QueueBuffer')
        ]
        assert bad == [], (
            'grab_new_capture must not drive the data stream directly; route '
            f'through the handler frame-gate instead. Found: {bad}'
        )


class TestLiveExposureMinimum:
    """get_min_exposure must report the camera's LIVE ExposureTime node
    minimum, not the value cached in the profile at connect.

    The node minimum drifts above the connect-time value once other settings
    change (pixel clock, frame rate, AOI), so a sweep that sources the floor
    from the stale cache requests an exposure below what the camera will now
    accept and the SDK rejects it. The accessor reads the live node so callers
    get the floor the camera will honor right now.
    """

    def test_reports_live_node_minimum_not_cached_profile(self):
        from types import SimpleNamespace

        cam = bare_ids_camera()
        # Live node floor (us) has risen above the connect-time cache.
        cam.remote_nodemap = _RecordingNodemap(
            preset={'ExposureTime': _RecordingNode(minimum=37.171717)}
        )
        cam.profile = SimpleNamespace(exposure_min_us=31.0)

        # ms: live 0.037171717, NOT the cached 0.031.
        assert cam.get_min_exposure() == pytest.approx(0.037171717)

    def test_falls_back_to_cached_floor_when_inactive(self):
        from types import SimpleNamespace

        cam = bare_ids_camera()
        cam.active = False
        cam.profile = SimpleNamespace(exposure_min_us=31.0)

        assert cam.get_min_exposure() == pytest.approx(0.031)

    def test_falls_back_to_cached_floor_when_node_read_fails(self):
        from types import SimpleNamespace

        cam = bare_ids_camera()
        # Minimum() returns a non-numeric, so the live read raises and the
        # accessor falls back to the cached profile floor rather than crashing.
        cam.remote_nodemap = _RecordingNodemap(
            preset={'ExposureTime': _RecordingNode(minimum=None)}
        )
        cam.profile = SimpleNamespace(exposure_min_us=31.0)

        assert cam.get_min_exposure() == pytest.approx(0.031)
