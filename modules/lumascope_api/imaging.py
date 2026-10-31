# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ImagingAPI -- sub-API for camera capture / image acquisition.

ImagingAPI owns the capture: _last_capture_info, _focusing_event, the frame
listeners, the stream check and _camera_temp_event. The camera's settings and state are CameraAPI's
(``modules/lumascope_api/camera.py``), which it holds and calls.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import logging as _logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from modules.finite_number import refuse_unless_wait_s
import modules.image_utils as image_utils
from drivers import host_clock
from drivers.camera import Frame
from drivers.exceptions import CameraReadFailedError, HardwareError
from lib import profile_trace
from lvp_logger import logger
from modules.activity_claim import a_guarded_write
from modules.exceptions import (
    ArgumentRefusedError,
    CameraStreamStalledError,
    CaptureFailedError,
    FrameHandlerRemovedError,
    HardwareCommandRefusedError,
)
from modules.frame_validity import Change, Counter, FrameValidity, FrameValiditySource, Judgment
from modules.lumascope_api.camera import (
    AutoGainConvergence,
    AutoGainLock,
    LayerCameraRequest,
    LayerHold,
    _camera_not_connected,
    _refuse_unless_auto_gain_settings,
)
from modules.lumascope_api.frame_record import (
    BLOWN_FRACTION,
    CapturedStill,
    ClockMapping,
    FrameRecord,
    grab_record,
)
from modules.lumascope_api.illumination import lit_other_than, live_lit_pairs
from modules.notification_center import notifications
from modules.video_cadence import StallWatch, prologue_stall_threshold_s
from modules.api_surface import Admission, api, api_fields


@dataclasses.dataclass(frozen=True)
class _NoFrame:
    """A grab that ended without a frame, handed back for its caller to raise.

    The grab raises nothing itself: ``_capture_still_impl`` first asks whether
    its window was dirtied (a re-grab, not a failure) and runs its own
    cleanup, then raises. ``reason`` is a ``CaptureFailedError`` reason, or
    ``'removed'`` for a camera found gone, which is refused as every camera
    command is, naming the caller's member. ``leg`` is where the grab
    stopped -- ``'entry'``, ``'grab'`` or ``'sum'`` -- for the log.
    """

    reason: str
    leg: str
    source: FrameValiditySource | None = None

    @property
    def removed(self) -> bool:
        return self.reason == 'removed'

    def error(self, member: str) -> CaptureFailedError | HardwareCommandRefusedError:
        if self.removed:
            return _camera_not_connected(member)
        return CaptureFailedError(self.reason, source=self.source)


@dataclasses.dataclass(frozen=True)
class _Grabbed:
    """A grab that got its frame: the image it returns, and the frame it came from.

    ``image`` is what the caller is handed -- the frame's own pixels, a sum,
    an 8-bit rendering or a scale-bar copy -- read-only in every case.
    ``frame`` is the camera's record of the frame the image was made from
    (the last one, for a sum), so the capture's record and its log describe
    that frame and no later one. ``earliest`` is the first frame the image
    was made from (``frame`` itself, but for a sum): the one a change could
    have reached last, so the one the capture is judged by.
    ``saturated_fraction`` and ``dark`` are what the grab found of the frames
    it kept, for the capture's record (``FrameRecord``): None when it was not
    asked to look.
    """

    image: np.ndarray
    frame: Frame
    earliest: Frame
    saturated_fraction: float | None
    dark: bool | None


@dataclasses.dataclass(frozen=True)
class _Judged:
    """What a capture's grab judges each frame by: a frame is kept only when
    every change it does not exclude counts it (``FrameValidity.judge``).

    ``mapping`` is the camera's clock as the capture's cycle took it, the one
    its record carries. ``clock_bound_ns`` is how long the grab takes frames
    when a change runs the clock rule: the latest instant a frame must start
    after, plus one grab window (the scope's steady clock,
    ``host_clock.steady_ns``); None when every change runs the credit rule,
    whose grab keeps the caller's ``timeout_s``.
    """

    mapping: ClockMapping | None
    exclude_sources: tuple[FrameValiditySource, ...]
    clock_bound_ns: int | None

    def unmet(self, frame_validity: FrameValidity, frame: Frame) -> Judgment | None:
        """The first change ``frame`` does not count for, by source; None when it counts for all."""
        for judgment in frame_validity.judgments(frame, self.mapping):
            if judgment.change.source not in self.exclude_sources and not judgment.counts:
                return judgment
        return None


@dataclasses.dataclass(frozen=True)
class _FeedMark:
    """The camera's feed at one instant on the scope's steady clock: where a wait starts.

    A wait that gives up states against its mark how long it waited and what
    the feed did meanwhile (``account``). Both counts are read because they
    stop apart: frames handed over stop when the feed stops, frames stored
    stop when the store stage wedges while the feed runs, and both rising
    with no frame taken is a wait that missed frames the store had. The last
    stored frame is read at the give-up; a driver whose fresh-frame grab
    empties the store as it starts (the FX2's) holds none after a wait in
    which nothing arrived.
    """

    at_ns: int
    handed_over: int
    stored: int

    @classmethod
    def read(cls, driver: Camera) -> _FeedMark:
        return cls(host_clock.steady_ns(), driver.frames_delivered, driver.delivered_counts[0])

    def account(self, driver: Camera) -> str:
        """The wait from this mark to now and what the feed did in it, for a give-up line."""
        now = _FeedMark.read(driver)
        last = driver.grab()
        last_stored = (
            f'last stored seq={last.seq}, {(now.at_ns - last.arrived_ns) / 1e9:.3f}s before'
            if last is not None
            else 'no frame in the store'
        )
        return (
            f'waited {(now.at_ns - self.at_ns) / 1e9:.3f}s from steady {self.at_ns / 1e9:.3f}s; '
            f'handed over {self.handed_over}->{now.handed_over}, '
            f'stored {self.stored}->{now.stored}; {last_stored}'
        )


def _capture_line(grabbed: _Grabbed, record: FrameRecord) -> str:
    """The one line a successful capture leaves: the frame it returned, how
    well the camera's clock was mapped beside it, and how long after each
    change it came, so a support bundle shows whether the rule held."""
    frame = grabbed.earliest
    frame_id = (frame.chunks or {}).get('FrameID')
    parts = [
        f'capture_still: frame seq={frame.seq} id={frame_id if frame_id is not None else "na"}'
    ]
    if grabbed.frame is not frame:
        parts[0] += f' (sum to seq={grabbed.frame.seq})'
    mapping = record.clock_mapping
    parts[0] += (
        ' clock=none' if mapping is None else f' clock=+-{mapping.uncertainty_ns / 1000:.0f}us'
    )
    judgments = record.judgments
    for judgment in judgments:
        f = judgment.fields()
        in_force = f' in_force+{f["after_in_force_ms"]}ms' if 'after_in_force_ms' in f else ''
        parts.append(
            f'{f["source"]} {"counts" if f["counts"] else "DOES NOT COUNT"} '
            f'+{f["frames_after"]}fr +{f["after_write_ms"]}ms{in_force} '
            f'skip={f["skip"]} by={",".join(f["counted_by"]) or "none"}'
        )
    return ' | '.join(parts)


def _refuse_unusable_depth(significant_bits: int) -> None:
    """Refuse a payload depth no frame has: outside 1 to 16.

    A negative depth raised from the shift that builds full scale, and 0
    made full scale 0, so every pixel read as saturated. A depth that is
    not an int is the API door's to refuse.
    """
    if not 1 <= significant_bits <= 16:
        raise ArgumentRefusedError(
            'out_of_range', argument='significant_bits', value=significant_bits, limits=(1, 16)
        )


@api_fields('bytes_per_s', 'frame_rate_fps')
@dataclasses.dataclass(frozen=True)
class DeliveredRate:
    """What the camera delivered to the host over the last second.

    ``frame_rate_fps`` counts the frames stored, whatever any display does with
    them. ``bytes_per_s`` is the bytes those same frames took on the link --
    the payload the camera sent, not the transport's protocol overhead -- so
    ``bytes_per_s / frame_rate_fps`` is the bytes one frame takes on the link.
    Both are 0 when nothing was delivered in the last window: no camera, a
    camera not streaming, or no recent window measured.
    """

    frame_rate_fps: float
    bytes_per_s: float

    @property
    def megabytes_per_s(self) -> float:
        """``bytes_per_s`` in decimal megabytes, the unit links are rated in."""
        return self.bytes_per_s / 1_000_000


NOT_DELIVERING = DeliveredRate(frame_rate_fps=0.0, bytes_per_s=0.0)


if TYPE_CHECKING:
    from drivers.camera import Camera
    from modules.lumascope_api._lumascope import Lumascope
    from modules.lumascope_api.camera import CameraAPI
    from modules.scheduler import Scheduler

_api_log = _logging.getLogger('LVP.api')


# Per Firmware/docs/PERFORMANCE_BUDGETS.md plugin_live_processing_handler_ms
# row. Budget anchors to the 30 fps realistic-cap target (measured on the
# rigs: daA3840 max sustained median 17.6 fps; a2A3536 49.9 fps).
# Plugin gets ~70% of the inter-frame window; driver keeps ~10 ms headroom.
HANDLER_BUDGET_MS = 24

# K consecutive over-budget invocations before auto-removal. ~1 second at
# 30 fps. Forgiving enough to absorb a single bursty frame; strict enough
# to protect the imaging pipeline from a hung plugin.
HANDLER_DROP_K = 30


class _BudgetedHandler:
    """Wraps a user-supplied frame listener with budget + drop-policy
    enforcement. Created by ImagingAPI.add_frame_listener; transparent
    to plugin authors -- the wrapper itself is what gets registered with
    the driver, not the user's handler.

    Re-entrancy: not a concern. Each driver's fire-site is single-
    threaded (Pylon SDK contract / IDS grab loop / sim acquisition thread). Auto-
    removal calls ImagingAPI._remove_wrapper which takes the driver
    lock, but the driver's _store_frame snapshots callbacks under
    lock and invokes outside -- no deadlock risk on the same-thread
    auto-remove path.
    """

    __slots__ = (
        '_budget_trace',
        '_consecutive_over',
        '_consecutive_raised',
        '_handler',
        '_imaging',
        '_name',
        '_removed',
    )

    def __init__(self, imaging: ImagingAPI, handler, name: str) -> None:
        self._imaging = imaging
        self._handler = handler
        self._name = name
        self._consecutive_over = 0
        self._consecutive_raised = 0
        self._removed = False
        # Budget-consumption census. The over-budget branch below already
        # logs, but only once it is ALREADY over -- so a handler sitting just
        # under the cap is indistinguishable from one costing nothing, right
        # up until a faster frame rate pushes it over and the listener is
        # auto-removed mid-recording. One writer per handler, and a handler
        # outlives the recordings it serves, so the row's identity is the
        # handler rather than a recording; correlate by timestamp against a
        # recording-scoped trace.
        self._budget_trace = profile_trace.BatchTrace(
            'handler_budget_trace.csv',
            'ts_ms,handler,elapsed_ms,budget_ms,consecutive_over,drop_k',
            profile_trace.NO_RECORDING,
        )

    def __call__(self, frame: Frame) -> None:
        if self._removed:
            return
        t0 = time.perf_counter()
        try:
            self._handler(frame)
        except Exception as e:
            # A handler that raises on every frame would otherwise log a
            # traceback per frame for as long as the camera streams. The
            # first of a run of failures is logged with its traceback; the
            # rest are counted, and a handler still failing after the same
            # number of frames that drops a slow one is dropped the same way.
            # Not counted toward the time budget: failing and being slow are
            # different faults, told apart in the removal's words.
            self._consecutive_raised += 1
            if self._consecutive_raised == 1:
                logger.exception(f"[SCOPE API ] live_processing handler '{self._name}' raised: {e}")
            if self._consecutive_raised >= HANDLER_DROP_K:
                self._auto_remove('raised')
            return
        self._consecutive_raised = 0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if profile_trace.ENABLE_PROFILE_TRACE:
            self._budget_trace.add(
                [
                    f'{time.time() * 1000.0:.3f}',
                    self._name,
                    f'{elapsed_ms:.3f}',
                    HANDLER_BUDGET_MS,
                    self._consecutive_over,
                    HANDLER_DROP_K,
                ]
            )
        if elapsed_ms > HANDLER_BUDGET_MS:
            self._consecutive_over += 1
            logger.warning(
                f"[SCOPE API ] live_processing handler '{self._name}' "
                f'over budget: {elapsed_ms:.1f}ms (budget {HANDLER_BUDGET_MS}ms) '
                f'-- consecutive {self._consecutive_over}/{HANDLER_DROP_K}'
            )
            if self._consecutive_over >= HANDLER_DROP_K:
                self._auto_remove('over_budget', elapsed_ms)
        else:
            self._consecutive_over = 0

    def _auto_remove(self, reason: str, last_elapsed_ms: float = 0.0) -> None:
        """Drop trigger: K consecutive over-budget or raising calls.

        Removes self from ImagingAPI's listener registry and reports the
        removal once. The frame thread has no caller to raise to, so the
        outcome is built and handed to the reporter; the author of the
        handler learns it from the warning.
        """
        self._removed = True
        self._imaging._remove_wrapper(self)
        notifications.report_outcome(
            FrameHandlerRemovedError(
                self._name,
                reason,
                budget_ms=HANDLER_BUDGET_MS,
                drop_k=HANDLER_DROP_K,
                last_ms=last_elapsed_ms,
            ),
            solicited=False,
            category='Live Processing',
        )


@api_fields('frame')
@dataclasses.dataclass(frozen=True)
class FrameDelivered:
    """A frame the camera delivered: its ``Frame`` record -- the read-only
    image, its arrival number and times, its depth, and its chunk data when
    the camera sends any."""

    frame: Frame


class ImagingAPI:
    """Imaging sub-API: the capture. Owns the grab, the capture's record,
    the frame listeners, the stream check, the scale bar and the
    capture/focus events. The camera's settings and state are
    ``CameraAPI``'s, which this holds and calls.
    """

    def __init__(self, scope: Lumascope, camera: CameraAPI, frame_validity: FrameValidity) -> None:
        self._scope = scope
        # The camera this captures with: its lock serializes the grab against
        # its writes, its dispatch puts a capture on the camera lane, and its
        # hold and auto-gain lock run inside a capture's own lane task.
        self._camera = camera

        # _state_lock guards _last_capture_info and _last_frame_summing.
        self._state_lock = threading.Lock()

        # Once-per-episode latch for the enabled-but-no-objective scale-bar
        # skip log; the gates run per displayed frame, so an unlatched
        # warning would flood the log at display rate.
        self._scale_bar_objective_skip_logged = False

        # Per-frame listener registry. Each user handler is wrapped in
        # a _BudgetedHandler; the wrapper is what's registered with the
        # driver's _frame_callbacks list. The dict keys are the user
        # handlers so remove_frame_listener(handler) can look up the
        # wrapper. Lock protects register + remove + auto-remove
        # mutations -- the read path (driver fan-out) doesn't touch
        # this dict.
        self._frame_listener_lock = threading.Lock()
        self._frame_listener_wrappers: dict = {}

        # Boolean operation flags -- threading.Event for wait/signal.
        self._focusing_event = threading.Event()  # set => autofocus in progress

        # Capture / autofocus return slots. Reads/writes under
        # self._state_lock. Per the Sentinel-return contract: None
        # means "no result yet."

        # Evidence about the most recent capture past its call (hold
        # duration, drained frame count, rechecks, deadline): what a capture
        # leaves frame or not. A frame's own facts are its record's, handed
        # back with it. Read via last_capture_info.
        self._last_capture_info = None
        # (frames summed, each frame's delivered depth) of the frame the last
        # capture produced, set by the reducer that summed it, or None before
        # the first. The one account of a capture's sum, read only by the
        # record the capture builds, so no caller restates the count.
        self._last_frame_summing: tuple[int, int] | None = None

        # Frame validity -- ground-truth for "is this frame ready to score /
        # save?" The scope builds the one instance and hands it to the camera,
        # which invalidates it on every write, and to the capture, which
        # drains it.
        self.frame_validity = frame_validity
        # The stream check's state (start_stream_check). The watch and its
        # bound are written only on the scheduler's thread, by _check_stream.
        self._stream_check_scheduler: Scheduler | None = None
        self._stream_check_handle: object | None = None
        self._stream_watch: StallWatch | None = None
        self._stream_watch_bound_s: float | None = None
        self._stream_stall_reported = False
        # The delivered rate (_sample_delivery), written only on the
        # scheduler's thread: the last (frames, wire bytes, time) reading, and
        # the published (rate, time) pair, replaced whole so a reader on any
        # thread sees one window's rate.
        self._delivery_reading: tuple[int, int, float] | None = None
        self._delivered_rate: tuple[DeliveredRate, float] | None = None

        # Camera temp logging scheduler handle.
        self._camera_temp_event = None
        self._camera_temp_unschedule_fn = None

    @property
    def _driver(self) -> Camera | None:
        """Resolve the camera driver via the composition root each access.

        Lumascope's `_camera_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps ImagingAPI in sync without rebinding.
        """
        return self._scope._camera_driver

    def _chunk_target_mismatch(self, chunks: Mapping[str, int | float] | None) -> str | None:
        """Name the first chunk-validatable source whose value in ``chunks``
        does not match its recorded target, or None if all match.

        ``chunks`` are the frame's own, the frame being judged. None also
        covers every can't-verify case (no chunk support, chunk key absent,
        no target recorded) -- absence of evidence is not a mismatch, and
        cameras without chunks rely on skip-count settling.
        """
        if not chunks:
            return None
        fv = self.frame_validity
        for source in fv.CHUNK_VALIDATABLE_SOURCES:
            key = fv.CHUNK_KEY_FOR_SOURCE.get(source)
            value = chunks.get(key) if key else None
            if value is None:
                continue
            if fv.target(source) is None:
                continue
            if not fv.chunk_match(source, value):
                return source
        return None

    # The capture bound is wider, and it is a BASE: a dispatched capture
    # legitimately spends time draining stale frames before it returns, and
    # the dispatcher adds the caller's declared work on top -- the
    # content-gate retry budget and the summed-frame time -- so a healthy
    # long capture is never timed out by its own liveness bound.
    _CAPTURE_WAIT_TIMEOUT_S = 30.0

    # Budget for the capture drain-and-recheck deadline. The deadline bounds
    # how long a capture keeps draining and re-grabbing while invalidations
    # arrive; it is frozen at capture entry so a sustained invalidation
    # stream extends the WORK but never the BUDGET -- recomputing it from
    # the live validity counter is exactly the unbounded-drain failure the
    # deadline exists to end. The floor absorbs one full grab timeout
    # (>= 1.0 s) plus one re-check cycle at the deepest camera-source skip
    # count (3 frames); the frame-period floor is conservative against the
    # slowest observed sim frame period (~0.126 s) -- overestimating only
    # widens the budget, never re-admits a live-lock, because the frame
    # count is frozen; the margin covers frame-period jitter on top of that
    # conservative floor.
    _CAPTURE_DEADLINE_FLOOR_S = 3.0
    _CAPTURE_DEADLINE_MIN_FRAME_PERIOD_S = 0.15
    _CAPTURE_DEADLINE_MARGIN = 1.5

    # --- SDK-perf knobs (write-only by design) ---
    #
    # Considered get_X companions for the cluster below
    # (set_acquisition_stop_mode, set_bandwidth_reserve_mode,
    #  set_device_link_throughput_limit, set_max_transfer_size,
    #  set_num_max_queued_urbs, set_gev_packet_size,
    #  set_gev_inter_packet_delay, set_max_acquisition_frame_rate).
    # Rejected because: these are Pylon / GEV / USB SDK perf knobs
    # the customer typically configures once at startup; the SDK
    # exposes them as nodemap writes and readback either returns a
    # stale value or raises depending on the camera firmware. The
    # write-only shape matches the Pylon Configuration class pattern.
    # Revisit if a future caller has a concrete need to read the
    # current setting (e.g. a self-tuning bandwidth limiter); the
    # capability tokens for "supports readback" would belong on
    # scope.capabilities.

    def _set_acquisition_stop_mode(self, mode: str) -> bool:
        """Set BslAcquisitionStopMode (Pylon-only; no-op on IDS).

        Controls camera behavior when StopGrabbing fires during an
        in-flight exposure:

          - ``'Complete'`` (Pylon default): waits for the current
            exposure to finish before stopping.
          - ``'CancelExposure'``: stops cleanly; partial frame
            discarded.
          - ``'AbortExposure'``: aborts immediately; partial frame
            discarded.

        Default ``'Complete'`` waits up to the full exposure on long
        fluorescence captures (5-10 s) -- presents identically to a
        multi-second app-side stall when the user toggles modes.
        ``'AbortExposure'`` resolves the symptom but is bench-
        unvalidated on Etaluma's cameras. Setter is provided for
        bench characterization; default is unchanged.

        Args:
            mode: One of ``'Complete'``, ``'CancelExposure'``,
                ``'AbortExposure'``.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, mode is invalid, the driver doesn't
                implement the setter (IDS), or
                BslAcquisitionStopMode is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_acquisition_stop_mode'):
            logger.warning(
                f'[SCOPE API ] set_acquisition_stop_mode: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return False
        return bool(self._driver.set_acquisition_stop_mode(mode=mode))

    def _set_bandwidth_reserve_mode(self, mode: str) -> bool:
        """Set BandwidthReserveMode (GigE-only Pylon node).

        ``'Default'`` reserves a portion of GigE bandwidth for
        retransmits; ``'Performance'`` dedicates all bandwidth to
        image transmit. Per dmA3536-9gm spec, ``'Performance'``
        unlocks 9.5 fps vs the default 9.3 fps.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            mode: ``'Default'`` or ``'Performance'``.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter,
                or the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_bandwidth_reserve_mode'):
            return False
        return bool(self._driver.set_bandwidth_reserve_mode(mode=mode))

    def _set_device_link_throughput_limit(
        self,
        mode: str,
        value_bps: int | None = None,
    ) -> bool:
        """Set the camera's DeviceLinkThroughputLimit mode and value.

        Both nodes are live-writable per the SDK lock-state table -- no
        StopGrabbing/StartGrabbing wrap. Per-camera defaults bench-
        witnessed (USB3): ace 2 a2A3536-31umBAS at 360 MB/s -> 28.8 fps;
        dart daA3840-45um at 160 MB/s -> 18.7 fps. Setting ``mode='Off'``
        lets the camera run at sensor-readout maximum (~31.2 fps ace 2;
        ~44.9 fps dart on USB3).

        Used by the diagnostic-probe sweep in ``tools/`` to characterize
        failure rate vs throughput across camera + firmware + host
        cells. Per Basler docs: "Corrupt or dropped frames may occur if
        the DeviceLinkThroughputLimit parameter is too high" -- bench-
        test failure rate alongside fps before settling on a per-camera
        production default.

        **Transport caveat (GigE):** on GigE cameras (e.g. dmA3536-9gm)
        DLTL is bounded above by the GigE wire limit (~110 MB/s usable
        on 1 Gbps Ethernet). Setting above wire limit is a no-op; below
        caps fps proportionally. For GigE bandwidth control use
        ``set_gev_inter_packet_delay`` / ``set_bandwidth_reserve_mode``
        instead -- those are the GigE-side tools.

        Args:
            mode: ``'On'`` or ``'Off'`` (case-sensitive; matches Pylon
                enum entry symbolic names).
            value_bps: Throughput cap in bytes per second when
                ``mode='On'``. Ignored when ``mode='Off'``. If None
                while ``mode='On'``, only the mode is changed and the
                existing limit value is preserved.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the mode argument is invalid, or the driver
                returned False (unsupported by this driver).

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_device_link_throughput_limit'):
            logger.warning(
                f'[SCOPE API ] set_device_link_throughput_limit: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return False
        return bool(
            self._driver.set_device_link_throughput_limit(
                mode=mode,
                value_bps=value_bps,
            )
        )

    def _set_max_transfer_size(self, value_bytes: int) -> bool:
        """Set Pylon StreamGrabber MaxTransferSize (USB3 only).

        Bytes-per-USB-transfer the SDK requests from the kernel. Per
        Basler `stream-grabber-parameters.html` this is the named lever
        for the symptom "fails to receive image stream" -- decreasing
        the value works around kernel / driver USB-transfer-size
        constraints on some Windows hosts.

        USB3-only. The node is absent on GigE cameras and on the IDS
        SDK; returns False so the bench-probe sweep can call this
        method unconditionally per cell.

        Args:
            value_bytes: New MaxTransferSize in bytes.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter, or
                the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_max_transfer_size'):
            return False
        return bool(self._driver.set_max_transfer_size(value_bytes=value_bytes))

    def _set_num_max_queued_urbs(self, value: int) -> bool:
        """Set Pylon StreamGrabber NumMaxQueuedUrbs (USB3 only).

        Number of USB Request Blocks the SDK keeps in flight to the
        kernel. Per Basler `stream-grabber-parameters.html` this is
        the named lever for "insufficient system memory"
        (0xe2010130 / 0xe2100001) -- decreasing the value reduces
        kernel URB allocation pressure on memory-constrained hosts.

        USB3-only. The node is absent on GigE cameras and on the IDS
        SDK; returns False so the bench-probe sweep can call this
        method unconditionally per cell.

        Args:
            value: New NumMaxQueuedUrbs (count).

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter, or
                the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_num_max_queued_urbs'):
            return False
        return bool(self._driver.set_num_max_queued_urbs(value=value))

    def _set_max_num_buffer(self, value: int) -> bool:
        """Set Pylon InstantCamera MaxNumBuffer.

        Per Basler `instant-camera-parameters.html`: count of buffers the
        SDK allocates for grabbing. Default 10; Etaluma production caps
        to 3 (Windows non-paged-pool bound). Bench characterization
        uses 10 to match Pylon Viewer.

        Pylon-only. The driver stores the value on `_max_num_buffer` so
        the connect() lifecycle applies it post-Open() before the node
        locks. An immediate SetValue is also attempted; pypylon 26.4.x
        makes the node read-only once AcquireContinuousConfiguration
        auto-starts grabbing inside Open(), so a live override needs
        StopGrabbing first.

        Args:
            value: New MaxNumBuffer (positive int).

        Returns:
            bool: True on immediate SetValue success. False if camera is
                absent / inactive, the driver doesn't implement the
                setter, or the node is currently locked (the value is
                still stored for the next connect).

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_max_num_buffer'):
            return False
        return bool(self._driver.set_max_num_buffer(value=int(value)))

    def _set_grab_strategy(self, name: str) -> bool:
        """Set the Pylon GrabStrategy used by the next start_grabbing().

        Production default is LatestImageOnly (the contract for
        frame_validity, capture_still, and the auto-discard
        skip_frames floor). OneByOne delivers every frame for
        apples-to-apples bench comparisons against Pylon Viewer.

        Pure attribute write on the driver: takes effect on the NEXT
        start_grabbing() call, not on the live grab loop. To switch
        strategies on an active camera: call this lever, then
        stop_grabbing(), then start_grabbing().

        Args:
            name: 'LatestImageOnly' or 'OneByOne'.

        Returns:
            bool: True on accepted name. False if camera is absent or
                the driver doesn't implement the setter.

        Raises:
            ValueError: name is not a recognized strategy.
        """
        if not self._driver:
            return False
        if not hasattr(self._driver, 'set_grab_strategy'):
            return False
        return bool(self._driver.set_grab_strategy(name=name))

    def _set_gev_packet_size(self, size_bytes: int) -> bool:
        """Set GevSCPSPacketSize (GigE-only Pylon node).

        Packet size in bytes. 1500 = standard Ethernet MTU; 9000 =
        typical jumbo-frame size. Larger packets reduce per-camera
        CPU + packet rate but require OS-level jumbo-frame config.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            size_bytes: Packet size in bytes (positive int).

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, size_bytes is non-positive, the driver
                doesn't implement the setter, or the node is not
                exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_gev_packet_size'):
            return False
        return bool(self._driver.set_gev_packet_size(size_bytes=size_bytes))

    def _set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """Set GevSCPD (GigE inter-packet delay, in clock ticks).

        Inserts a wait between successive packets to throttle the
        camera. Used when multiple cameras share a single GigE link
        or when the host CPU can't keep up. 0 = no delay.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            delay_ticks: Non-negative int; camera-specific tick rate.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, delay_ticks is negative, the driver doesn't
                implement the setter, or the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_gev_inter_packet_delay'):
            return False
        return bool(self._driver.set_gev_inter_packet_delay(delay_ticks=delay_ticks))

    # --- Capture ---
    def _capture_still_impl(
        self,
        force_to_8bit: bool = True,
        *,
        member: str,
        accept_dark: bool = False,
        exclude_sources: tuple[FrameValiditySource, ...] = (),
        all_ones_check: bool = False,
        timeout_s: float = 0.0,
        sum_count: int = 1,
        sum_delay_s: float = 0.0,
        sum_iteration_callback=None,
    ) -> CapturedStill:
        """Capture a frame guaranteed to reflect the current hardware state.

        Drains stale frames from the camera pipeline until every change (LED,
        gain, exposure, motion, ...) can be met by its source's rule
        (``capabilities.frame_rules``): a credit-rule change by its frame
        count, a clock-rule change by time on the camera's clock, a move by
        its axis settling too. Then grabs, keeping only a frame that the one
        judgment (``FrameValidity.judge``) counts for every change it does not
        exclude, and taking the next frame otherwise.

        Counting frames adapts to the camera's frame rate -- fast exposures
        drain quickly, slow exposures drain slowly.

        Args:
            force_to_8bit: Convert to 8-bit output.
            member: The public member the capture is refused as.
            exclude_sources: Sources to ignore for validity, e.g. ``(FrameValiditySource.Z_MOVE,)``.
                The frame returned can predate an excluded change, so a
                frame that is measured or recorded never excludes one.
            all_ones_check: Measure the frame's saturation. A blown frame
                is grabbed again once and, blown again, kept; the record
                carries the fraction either way (``FrameRecord``
                ``saturated_fraction``, ``blown``). Without it both are None.
            accept_dark: Caller-intent override for the derived dark-floor
                expectation. The capture derives whether illumination is
                commanded ON from the illumination API itself (a channel
                counts as lit only at strictly positive current, so an
                enabled 0 mA channel is dark by design); when a channel is
                lit, a frame with essentially no lit pixel is retried until
                ``timeout_s`` and then kept, its record's ``dark`` True.
                accept_dark=True skips the judgment, the record's ``dark``
                None -- for callers whose dark frames are legitimate while
                lit: autofocus sweeps (an out-of-focus fluorescence plane can
                carry no signal) and benchmark probes.
            timeout_s: Timeout (seconds) for the final get_image call.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.

        Returns:
            CapturedStill: The captured image with its record.

        Raises:
            CaptureFailedError: no frame: ``'no_frame'``, the camera
                delivered none within its grab window, draining, grabbing or
                waiting for a sum's next frame; ``'deadline_expired'``, state
                changes outran the budget below; ``'no_fresh_frame'``, frames
                came but none within the grab's bound counted for the
                ``setting`` changed (under the credit rule that bound is
                ``timeout_s``; under the clock rule one grab window past the
                instant a frame must start after); ``'chunk_rejected'``,
                frames counted but none at the ``setting`` asked;
                ``'record_read_failed'``, a frame came but the camera did not
                answer a read its record needs, chained from the
                ``CameraReadFailedError``.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, absent at the call or removed during the capture;
                ``'camera_not_streaming'``, connected but not grabbing.
                Nothing reports either here: the raise is the report, so a
                live-view auto-gain arm is not re-armed on a removal.

        The capture honors invalidation across its whole window: hardware
        state changing between the drain's settle and the grab's return is
        detected via the invalidation counters, and the capture re-drains,
        re-derives its expectations (dark floor included), and re-grabs.
        The drain-and-recheck deadline bounds that loop -- a sustained
        invalidation stream raises ``'deadline_expired'`` in bounded time
        instead of an open-ended hold. The deadline clock
        suspends while commanded motion is still physically settling
        (motion has its own authoritative completion signal, and a frame
        budget must not out-vote it), and it governs only the drain and
        re-check loops: the interior of one grab is bounded by
        ``timeout_s`` (under the clock rule, by its own bound, one grab
        window past the instant a frame must start after) and, for summed
        captures, the sum window itself.
        The one state change no polling design can see is a driver write
        whose invalidate has not yet landed when the grab returns; that
        residual is microseconds wide and accepted.
        """
        # A capture refused at its call records nothing; every capture past
        # it records its evidence in last_capture_info, frame or not.
        if not self._driver or not self._driver.active:
            raise _camera_not_connected(member)

        # active (connected) is necessary but not sufficient to deliver a
        # frame: connect() configures the camera without starting the feed
        # (the lifecycle split), and disable_streaming() deliberately halts it.
        # A not-grabbing camera cannot produce a capture, so it is refused
        # as that, distinctly, so a stopped or not-yet-started feed is not
        # misread as a grab timeout.
        if not self._driver.is_grabbing():
            logger.warning(
                '[SCOPE API ] capture_still: no active grab (streaming not '
                'started or stopped) -- refused until the feed is running.'
            )
            raise HardwareCommandRefusedError('camera_not_streaming', member)

        hold_start = time.monotonic()
        hold_mark = _FeedMark.read(self._driver)
        exposure_s = self._camera._grab_sizing_exposure_s()
        grab_timeout_s = max(exposure_s * 3, 1.0)

        # The deadline is frozen at entry: the pending frame count and the
        # exposure are read once and never re-raised, so invalidations
        # arriving after entry extend the work but not the budget. The sum
        # term mirrors the public dispatcher's liveness wait -- a long
        # summed capture legitimately grinds for its whole sum window, and
        # a re-grab repeats that window. The timeout term is the grab's own
        # retry window for a lit frame, which the dispatcher's wait carries
        # too: without it, a change landing while the grab waits out dark
        # frames charges that whole wait to a budget that never held it, and
        # a capture that would have saved its dark frame is refused as
        # invalidation churn instead. The settle term is what the changes
        # are owed, the public dispatcher's own (``_settle_budget``).
        frame_cost_s = max(exposure_s, self._CAPTURE_DEADLINE_MIN_FRAME_PERIOD_S)
        n_entry, settle_s = self._settle_budget(exclude_sources, frame_cost_s)
        deadline_s = (
            self._CAPTURE_DEADLINE_FLOOR_S
            + settle_s
            + sum_count * (frame_cost_s + sum_delay_s)
            + timeout_s
        )
        # Pause-and-resume accounting: an interval is charged to the budget
        # only when no commanded motion was still settling at the check.
        # Charging suspended intervals would expire the budget across one
        # legitimate long stage move and fail a capture that motion's own
        # completion signal is about to release.
        _clock = {'last': hold_start, 'active': 0.0}

        def _deadline_expired() -> bool:
            now = time.monotonic()
            if not self.frame_validity.unsettled_motion_sources(exclude_sources=exclude_sources):
                _clock['active'] += now - _clock['last']
            _clock['last'] = now
            return _clock['active'] > deadline_s

        def _record_capture_info() -> None:
            with self._state_lock:
                self._last_capture_info = {
                    'hold_ms': (time.monotonic() - hold_start) * 1000.0,
                    'drained': drain_iterations,
                    'rechecks': recheck_cycles,
                    'deadline_s': deadline_s,
                    'n_entry': n_entry,
                    'active_s': _clock['active'],
                }

        def _deadline_expired_error(where: str) -> CaptureFailedError:
            if lock is not None:
                self._camera._resume_auto_gain_impl(lock)
            logger.warning(
                f'[SCOPE API ] capture_still: capture DEADLINE EXPIRED '
                f'({where}) -- active={_clock["active"]:.3f}s > '
                f'deadline_s={deadline_s:.3f}s (n_entry={n_entry}, '
                f'rechecks={recheck_cycles}, '
                f'wall={time.monotonic() - hold_start:.3f}s); invalidation '
                f'outran the capture budget. Distinct from grab failure. '
                f'{hold_mark.account(self._driver)}'
            )
            _record_capture_info()
            return CaptureFailedError('deadline_expired')

        # `drained` accumulates across every drain re-entry; `rechecks`
        # counts how many times the window was dirtied and re-run.
        drain_iterations = 0
        recheck_cycles = 0
        # A standing auto-gain arm is locked once, after its settle drains
        # and before the grab; the loop then re-enters so the lock's own
        # setter writes drain and the gate sees the locked targets.
        lock: AutoGainLock | None = None
        while True:
            if _deadline_expired():
                raise _deadline_expired_error('recheck-top')
            # Grab plumbing is re-derived each cycle so a mid-window
            # exposure change sizes the NEXT drain and grab for the state
            # that now holds -- a grab timeout sized to a stale short
            # exposure fails healthy long-exposure frames and reports the
            # staleness as a drain failure. Plumbing only: the budget
            # above stays frozen.
            exposure_s = self._camera._grab_sizing_exposure_s()
            grab_timeout_s = max(exposure_s * 3, 1.0)

            # The camera's clock, mapped once per cycle and before its drain:
            # the clock rule's drain, the grab's judge and the frame's record
            # all read this one mapping. A failed latch fails the capture, as
            # a failed read for the record does.
            try:
                mapping = self._clock_mapping_for_record()
            except HardwareError:
                if lock is not None:
                    self._camera._resume_auto_gain_impl(lock)
                raise
            uncertainty_ns = mapping.uncertainty_ns if mapping is not None else 0

            # Snapshot the invalidation counters BEFORE the drain: any
            # invalidation landing after this line -- during the drain,
            # the derivation, or the grab itself -- differs at the compare
            # below, so no gap exists in which a change can hide. The
            # counters are immune to frames settling the pending state
            # (that erasure is exactly why pending-state snapshots cannot
            # do this job).
            counts_before = {
                s: c
                for s, c in self.frame_validity.invalidation_counts.items()
                if s not in exclude_sources
            }

            # Drain stale frames until every change can be met, each by its
            # source's rule (``FrameValidity.owed``): a credit-rule change
            # until its frame count is met, a clock-rule change until the
            # host's clock passes the instant a frame must start after, and
            # a move until its axis settles. A chunk certifies the register
            # in force when the frame was tagged, not the light integrated
            # into it, so it never stands in for this wait. Each drained grab
            # passes its frame's ordinal so a frame concurrently counted by
            # the preview poller is not counted twice.
            while True:
                owed = self.frame_validity.owed(
                    exclude_sources=exclude_sources, uncertainty_ns=uncertainty_ns
                )
                if owed.frames == 0 and (
                    owed.until_ns is None or host_clock.steady_ns() >= owed.until_ns
                ):
                    break
                if _deadline_expired():
                    raise _deadline_expired_error('drain-loop')
                drain_mark = _FeedMark.read(self._driver)
                drained = self._driver.grab_new_capture(timeout_s=grab_timeout_s)
                if drained is not None:
                    self.frame_validity.count_frame(drained.seq, by=Counter.DRAIN)
                    drain_iterations += 1
                else:
                    remaining = owed.frames
                    # The removal latch is set the moment a removal is seen;
                    # ``active`` is released later, off-thread.
                    removed = self._camera.camera_removed or not self._driver.active
                    logger.warning(
                        f'[SCOPE API ] capture_still: frame drain failed -- '
                        f'grab_new_capture returned status=False after '
                        f'{grab_timeout_s:.1f}s timeout '
                        f'(drained={drain_iterations}, frames_until_valid={remaining}, '
                        f'device_removed={removed}); {drain_mark.account(self._driver)}'
                    )
                    _record_capture_info()
                    if removed:
                        # The raise is the report: nothing to re-arm.
                        raise _camera_not_connected(member)
                    if lock is not None:
                        self._camera._resume_auto_gain_impl(lock)
                    raise CaptureFailedError('no_frame')

            if lock is None:
                # The lock consumes the arm under _state_lock and reports
                # state None when nothing stood; no unguarded peek first.
                lock = self._camera._lock_auto_gain_impl()
                if lock.state is not None:
                    continue

            # The dark-floor expectation is the API's own fact, derived
            # after the drain settles -- never posted by callers -- so the
            # value cannot drift from commanded state. A re-run of this
            # loop re-derives it, because the state change that dirtied
            # the window is exactly what makes the old derivation stale.
            # The same read is the frame's LED record: an LED change during
            # the grab dirties the window and re-runs this loop, so a frame
            # that survives the compare below was lit exactly as read here.
            # A read after the grab would not be -- an emergency LED-off can
            # run on another thread at any moment.
            lit = live_lit_pairs(self._scope.illumination)
            expected_lit = bool(lit)

            # The grab keeps only a frame every change counts, and under the
            # clock rule takes frames until one grab window past the latest
            # instant a frame must start after.
            judged = _Judged(
                mapping=mapping,
                exclude_sources=exclude_sources,
                clock_bound_ns=(
                    owed.until_ns + int(grab_timeout_s * 1e9) if owed.until_ns is not None else None
                ),
            )
            grabbed = self._get_image_impl(
                force_to_8bit=force_to_8bit,
                all_ones_check=all_ones_check,
                dark_floor_check=expected_lit and not accept_dark,
                timeout_s=timeout_s,
                sum_count=sum_count,
                sum_delay_s=sum_delay_s,
                sum_iteration_callback=sum_iteration_callback,
                force_new_capture=True,
                new_capture_timeout_s=grab_timeout_s,
                verify_chunk_targets=True,
                judged=judged,
            )
            grabbed_at = datetime.datetime.now()

            # Post-grab compare: a changed counter means the window was
            # dirtied and the frame (or failure) predates the state the
            # caller commanded. The compare runs BEFORE any failure is
            # raised: a dirtied-window failure is recoverable -- it was
            # caused by the very change this loop exists to honor -- so
            # only a clean-window failure is raised as a genuine one. Compare
            # spans both key sets: a source's first-ever invalidation adds a
            # key the snapshot lacks.
            counts_after = {
                s: c
                for s, c in self.frame_validity.invalidation_counts.items()
                if s not in exclude_sources
            }
            if counts_after != counts_before:
                recheck_cycles += 1
                continue
            break

        # The chunks are the returned frame's own; a capture that got no
        # frame has none.
        chunks = dict(grabbed.frame.chunks or {}) if isinstance(grabbed, _Grabbed) else {}
        record: FrameRecord | None = None
        failed = grabbed if isinstance(grabbed, _NoFrame) else None
        if failed is not None:
            logger.warning(
                f'[SCOPE API ] capture_still: nothing captured -- {failed.reason} at the '
                f'{failed.leg} (drained={drain_iterations}, rechecks={recheck_cycles})'
            )
        # A camera found gone is refused, and its refusal is the report: a
        # locked auto-gain arm is re-armed on every other exit, and on that
        # one there is nothing to re-arm.
        removed = failed is not None and failed.removed
        try:
            if failed is None:
                # A frame is kept only with its record: a gain, exposure or
                # black level read the camera does not answer fails the
                # capture, in the capture failure's words, so every caller
                # that handles a failed capture handles this one; one that
                # failed because the camera left between the grab and the
                # read is that removal.
                # The frame is judged against the mapping its record carries,
                # the one its cycle took before the drain.
                judgments = self.frame_validity.judgments(grabbed.earliest, mapping)
                try:
                    record = self._build_frame_record(
                        chunks=chunks,
                        lit=lit,
                        captured_at=grabbed_at,
                        clock_mapping=mapping,
                        judgments=judgments,
                        saturated_fraction=grabbed.saturated_fraction,
                        dark=grabbed.dark,
                        auto_gain=lock.state if lock is not None else None,
                    )
                except CameraReadFailedError as unread:
                    _record_capture_info()
                    if not self._camera.camera_removed:
                        logger.warning(
                            f'[SCOPE API ] capture_still: record read failed -- {unread}'
                        )
                        raise CaptureFailedError('record_read_failed') from unread
                    removed = True
                    raise _camera_not_connected(member) from unread
            if failed is None:
                _api_log.info(_capture_line(grabbed, record))
            _record_capture_info()
        finally:
            if lock is not None and not removed:
                self._camera._resume_auto_gain_impl(lock)
        if failed is not None:
            raise failed.error(member)
        return CapturedStill(image=grabbed.image, record=record)

    def _capture_holding_layer_impl(
        self, layer: str, request: LayerCameraRequest, member: str, **capture
    ) -> CapturedStill:
        """``_capture_still_impl``, with the camera holding ``layer``'s stored settings.

        For a capture that is already its own camera-lane task (the still):
        the camera's hold for a capture (``CameraAPI._hold_layer_for_capture``)
        runs inline, as the capture body does, so neither is admitted again
        against whatever holds the scope now. ``capture`` is the capture
        body's keyword arguments. A frame named and set for ``layer`` is lit
        by no other LED, so another one lit refuses it before the hold.

        Raises:
            HardwareCommandRefusedError: ``'other_led_lit'``, an LED other
                than ``layer``'s is lit; nothing was written or captured. As
                ``_hold_layer_for_capture`` says, and whatever the capture
                raises.
            CameraSettingRejected: as ``_hold_layer_for_capture`` says.
        """
        if lit_other_than(self._scope.illumination, layer):
            raise HardwareCommandRefusedError('other_led_lit', member)
        self._camera._hold_layer_for_capture(layer, request, member)
        return self._capture_still_impl(member=member, **capture)

    def _build_frame_record(
        self,
        *,
        chunks: dict[str, int | float],
        lit: frozenset[tuple[int, float]],
        captured_at: datetime.datetime,
        clock_mapping: ClockMapping | None,
        judgments: tuple[Judgment, ...],
        saturated_fraction: float | None,
        dark: bool | None,
        auto_gain: AutoGainConvergence | None,
    ) -> FrameRecord:
        """The instrument's account of the frame just grabbed, on the grab's lane.

        Exposure and gain come from the frame's own chunk values where the
        camera stamps them (Pylon ace 2 / dart, the simulated camera), which
        are what frame validity checked the frame against. A frame without
        them (IDS, FX2, a Basler frame whose chunk was not delivered) takes
        a fresh read here, beside the grab -- the nearest the frame has to
        its own account, and nothing queued behind the grab on this lane can
        reach it. A read the camera does not answer raises
        ``CameraReadFailedError`` and fails the capture, as a failed black
        level read does, rather than saving a frame whose exposure or gain
        is unknown. The record says which each is (``SettingSource``).
        """
        camera = self._scope.capabilities
        with self._state_lock:
            frames_summed, frame_significant_bits = self._last_frame_summing
        return grab_record(
            captured_at=captured_at,
            chunks=chunks,
            read_exposure_ms=self._camera.read_exposure_ms,
            read_gain_db=self._camera.read_gain_db,
            black_level=self._black_level_for_record(),
            lit=dict(lit),
            frames_summed=frames_summed,
            frame_significant_bits=frame_significant_bits,
            binning_size=self._camera._held_or_read('binning'),
            camera_model=camera.camera_model,
            camera_tick_hz=camera.camera_timestamp_tick_hz,
            clock_mapping=clock_mapping,
            judgments=judgments,
            saturated_fraction=saturated_fraction,
            dark=dark,
            held_channel=self._camera.get_held_layer(),
            auto_gain=auto_gain,
        )

    def _clock_mapping_for_record(self) -> ClockMapping | None:
        """The camera's clock mapped onto the host's, taken at the top of each
        capture cycle, before its drain: the latch alone is bracketed by the
        host's steady clock, and the read of what it latched follows outside
        the bracket. None for a camera
        without a latch. A failed latch or read raises ``HardwareError`` and
        fails the capture, as a failed black level read does.
        """
        camera = self._scope.capabilities
        if not camera.camera_has_clock_latch:
            return None
        before = host_clock.steady_ns()
        self._driver.latch_clock()
        after = host_clock.steady_ns()
        return ClockMapping(
            host_ns=(before + after) // 2,
            camera_ticks=self._driver.latched_clock_ticks(),
            tick_hz=camera.camera_timestamp_tick_hz,
            uncertainty_ns=(after - before + 1) // 2,
        )

    def _settle_budget(
        self, exclude_sources: tuple[FrameValiditySource, ...], frame_cost_s: float
    ) -> tuple[int, float]:
        """The frames the changes are owed, and the seconds a capture budgets for them.

        The one settle term the capture's deadline and the public
        dispatcher's wait both read: the frames owed under the credit rule,
        each at ``frame_cost_s`` and the deadline's margin, plus the time
        until the host's clock passes the instant a frame must start after
        under the clock rule (``FrameValidity.owed``, without the clock
        mapping's uncertainty, which no budget is taken beside).
        """
        owed = self.frame_validity.owed(exclude_sources=exclude_sources)
        clock_s = (
            max(0, owed.until_ns - host_clock.steady_ns()) / 1e9
            if owed.until_ns is not None
            else 0.0
        )
        return owed.frames, owed.frames * frame_cost_s * self._CAPTURE_DEADLINE_MARGIN + clock_s

    def _black_level_for_record(self) -> float | None:
        """The black level a frame's record carries: a fresh read beside the
        grab, None when the camera reports none. A setter makes it a capture
        variable, so a frame states it as it states its gain; a failed read
        raises ``CameraReadFailedError`` and fails the capture rather than
        saving a frame whose black level is unknown.
        """
        return self._camera.read_black_level()

    def capture_still(
        self,
        hold: LayerHold | None = None,
        *,
        force_to_8bit: bool = True,
        accept_dark: bool = False,
        exclude_sources: tuple[FrameValiditySource, ...] = (),
        all_ones_check: bool = False,
        timeout_s: float = 0.0,
        sum_count: int = 1,
        sum_delay_s: float = 0.0,
        sum_iteration_callback: Callable[[], None] | None = None,
    ) -> CapturedStill:
        """Capture a frame-valid image on the camera worker, and hand it back with its record.

        A frame guaranteed to reflect the hardware as it is now: the capture
        passes over stale frames until every change since the last one (an
        LED, the gain, the exposure, a move) is met by its source's rule
        (``ScopeCapabilities.frame_rules``) -- a frame count, time on the
        camera's clock, an axis settling -- and keeps only a frame that
        counts for every change it does not exclude. A change landing while
        the capture runs is seen, and the capture starts its wait again,
        within a deadline; the deadline stands still while a commanded move
        is settling. The capture is one command on the camera's lane, and
        this returns once it has run, with the frame's record beside the
        image (``CapturedStill``), so the record is the frame's and no later
        capture's; a caller that needs only the pixels reads ``.image``.

        Given ``hold``, the camera holds that layer's stored settings for the
        grab (``CameraAPI.hold_layer``), in the same camera-lane task; given
        None, it captures with the camera as set. In-process only, as
        ``hold_layer`` is: the Session reads the layer's store and hands it
        in (``ScopeSession.capture_still``), so a still is held at no
        settings but its layer's.

        Args:
            hold: The layer whose stored settings the camera holds for the
                grab, or None for the camera as set.
            force_to_8bit: True for an 8-bit image; False for the camera's
                own depth, in a uint16 array for a 12- or 16-bit camera.
            accept_dark: While a channel is lit (at a current above 0 mA), a
                frame with essentially no lit pixel is taken again until
                ``timeout_s`` passes, then kept and recorded dark. True skips
                that judgment, for a caller whose dark frames are legitimate
                while lit, such as an autofocus sweep.
            exclude_sources: Changes not to wait for, e.g.
                ``(FrameValiditySource.Z_MOVE,)``. The frame can then predate
                such a change, so a frame that is measured or saved never
                excludes one.
            all_ones_check: Measure the frame's saturation: a blown frame is
                grabbed again once and, blown again, kept.
            timeout_s: Seconds the grab, and each summed frame, keeps taking
                frames for one that passes its checks, once the stale ones are
                passed over; 0, the default, gives each one try.
            sum_count: Frames summed into the image, for noise reduction; a
                sum's array is uint16.
            sum_delay_s: Seconds between summed frames.
            sum_iteration_callback: Called after each summed frame, on the
                camera's lane.

        Returns:
            The frame's ``CapturedStill``: its image, (H, W) for a mono
            camera and (H, W, 3) for a color one, and its record.

        Raises:
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``, a
                time is not a wait from 0 to 30 days, or the wait for the
                capture and every frame of its sum together is longer.
                Nothing reached the camera.
            CaptureFailedError: ``'no_frame'``, the camera delivered none
                within its grab window; ``'deadline_expired'``, changes kept
                landing until the deadline passed; ``'no_fresh_frame'``,
                frames came but none within the grab's bound counted for the
                ``setting`` changed; ``'chunk_rejected'``, frames counted but
                none at the ``setting`` asked; ``'record_read_failed'``, a
                frame came but the camera did not answer a read of the gain,
                exposure or black level its record needs, so it was not kept,
                chained from the ``CameraReadFailedError``.
            HardwareCommandRefusedError: ``'not_connected'``, naming this
                member, with no camera at the call or removed during the
                capture; ``'camera_not_streaming'``, connected but not
                grabbing; or, given ``hold``, a run, a diagnostic or a home
                holds the scope, or ``'other_led_lit'``, an LED other than its
                layer's is lit.
            CameraSettingRejected: given ``hold``, the camera refused one of
                its settings; nothing was captured.
        """
        wait_s = self._capture_wait_s(exclude_sources, timeout_s, sum_count, sum_delay_s)
        capture = {
            'force_to_8bit': force_to_8bit,
            'accept_dark': accept_dark,
            'exclude_sources': exclude_sources,
            'all_ones_check': all_ones_check,
            'timeout_s': timeout_s,
            'sum_count': sum_count,
            'sum_delay_s': sum_delay_s,
            'sum_iteration_callback': sum_iteration_callback,
        }
        if hold is None:
            return self._camera._dispatch_camera(
                self._capture_still_impl,
                'capture_still',
                kwargs={'member': 'capture_still', **capture},
                timeout_s=wait_s,
            )
        if hold.request.auto_gain:
            _refuse_unless_auto_gain_settings(hold.request.auto_gain_settings, 'hold')
        # The hold's writes run in the same task, ahead of the capture.
        return self._camera._dispatch_camera(
            self._capture_holding_layer_impl,
            'capture_still',
            args=(hold.layer, hold.request, 'capture_still'),
            kwargs=capture,
            timeout_s=wait_s + self._camera._CAMERA_WRITE_TIMEOUT_S,
        )

    def _capture_wait_s(
        self,
        exclude_sources: tuple[FrameValiditySource, ...],
        timeout_s: float,
        sum_count: int,
        sum_delay_s: float,
    ) -> float:
        """How long a public capture waits for its camera-lane task.

        Raises:
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``, as
                ``capture_still`` says.
        """
        for name, value in (('timeout_s', timeout_s), ('sum_delay_s', sum_delay_s)):
            refuse_unless_wait_s(value, name)
        # The executor wait is a liveness bound, not a budget, so it scales
        # with the work the caller declared: the content-gate retry budget
        # runs inside the body, and each summed frame costs an exposure plus
        # the configured inter-frame delay. A flat bound times out a healthy
        # long capture (large sum_count at long exposure -- luminescence)
        # while the worker is still legitimately grinding. It also scales
        # with the settle work already pending at submit, and must dominate
        # the body's own drain-and-recheck deadline -- otherwise a deep
        # legitimate drain (a 20-frame auto-gain settle at the luminescence
        # exposure ceiling) surfaces to the caller as an executor
        # TimeoutError instead of the body's own 'deadline_expired'.
        frame_cost_s = max(
            self._camera._grab_sizing_exposure_s(),
            self._CAPTURE_DEADLINE_MIN_FRAME_PERIOD_S,
        )
        _, settle_s = self._settle_budget(exclude_sources, frame_cost_s)
        wait_s = (
            self._CAPTURE_WAIT_TIMEOUT_S
            + timeout_s
            + sum_count * (frame_cost_s + sum_delay_s)
            + settle_s
        )
        # Each time in range, a long sum can still make a wait no host holds.
        refuse_unless_wait_s(wait_s, 'the wait for the capture and every frame of its sum')
        return wait_s

    # A pixel at or above this fraction of full scale is saturated; how many
    # of them make a frame blown is the record's (``BLOWN_FRACTION``).
    _SATURATION_NEAR_MAX_FRACTION = 0.99

    @api(in_process=True, engineering=True)
    @staticmethod
    def saturated_fraction(arr: np.ndarray | None, significant_bits: int) -> float:
        """Fraction of pixels at or above the near-full-scale threshold.

        Full scale comes from the frame's payload depth, not the container
        dtype: a 12-bit frame in a uint16 container tops out at 4095, so
        measuring against 65535 would report a fully blown frame as 0%
        saturated and let it slip past the evidence check.

        Public because clipping is a property of the frame that callers
        outside this class have to be able to ask about: a whole-frame mean
        cannot distinguish an evenly lit field from one half blown out and
        half black, so any caller judging whether an exposure is usable needs
        the pixel count, not a summary statistic. `significant_bits` is
        required rather than defaulted -- a caller that has to guess the
        payload depth is the case this measures wrong.

        Args:
            arr: The frame's pixels; None or an empty array answers 0.
            significant_bits: The frame's payload depth in bits, 1 to 16, from which
                full scale is taken.

        Raises:
            ArgumentRefusedError: ``'out_of_range'``, *significant_bits* is
                outside 1 to 16, the depths a frame's 16-bit container holds.
        """
        _refuse_unusable_depth(significant_bits)
        if arr is None or arr.size == 0:
            return 0.0
        full_scale = (1 << significant_bits) - 1
        near_max = full_scale * ImagingAPI._SATURATION_NEAR_MAX_FRACTION
        return float(np.count_nonzero(arr >= near_max)) / arr.size

    # Symmetric counterpart of the saturation gate: a frame with essentially
    # no pixel above the dark floor carries no signal. Field causes: a frame
    # that began integrating before the LED lit, and an external camera
    # consumer starving the feed so black frames are delivered. The metric is
    # lit-pixel COUNT, never mean/median -- a sparse fluorescence field (a few
    # bright cells on a 99%-black background) must pass, and a handful of hot
    # pixels must not fake signal (a 3.5 MP frame needs ~350 lit pixels).
    _DARK_FLOOR_FRACTION = 0.03  # pixel <= 3% of full scale carries no signal
    _DARK_MIN_LIT_FRACTION = 1e-4  # < 0.01% of pixels lit = dark frame

    @staticmethod
    def _lit_fraction(arr: np.ndarray | None, significant_bits: int) -> float:
        """Fraction of pixels above the dark-floor threshold.

        Measured against the frame's payload depth, not the container
        dtype -- the same depth rule as ``saturated_fraction``, and the same
        refusal of a depth no frame has.
        """
        _refuse_unusable_depth(significant_bits)
        if arr is None or arr.size == 0:
            return 0.0
        full_scale = (1 << significant_bits) - 1
        floor = full_scale * ImagingAPI._DARK_FLOOR_FRACTION
        return float(np.count_nonzero(arr > floor)) / arr.size

    def _get_image_impl(
        self,
        force_to_8bit: bool = True,
        timeout_s: float = 5.0,
        all_ones_check: bool = False,
        dark_floor_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0.0,
        sum_iteration_callback: Callable[[], None] | None = None,
        force_new_capture: bool = False,
        new_capture_timeout_s: float = 5.0,
        verify_chunk_targets: bool = False,
        judged: _Judged | None = None,
    ) -> np.ndarray | _NoFrame:
        """Grab and return an image from the camera.

        By default returns the last buffered frame. Set force_new_capture=True
        to trigger a fresh capture. Multiple frames can be summed for noise
        reduction via sum_count.

        This is the ungated primitive: it does NOT wait for frame validity,
        so the returned frame may not reflect a just-changed gain, exposure,
        LED, or stage position. For any frame you will SAVE or MEASURE as
        truth, call capture_still, which drains the camera pipeline until
        the validity marker is GREEN before grabbing.

        Args:
            force_to_8bit: Convert 12-bit images to 8-bit output.
            timeout_s: Max seconds to wait for a valid frame.
            all_ones_check: Measure each kept frame's saturation; a blown one
                is grabbed again once, then kept (``_Grabbed.saturated_fraction``).
            dark_floor_check: Pass over frames with essentially no pixel
                above the dark floor until a lit frame arrives, and when
                timeout_s expires keep the dark one (``_Grabbed.dark``).
                Internal transport of the value
                capture_still derives from commanded LED state; the
                public get_image never sets it (probes and diagnostics
                must see what the camera sees, dark or not).
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay in seconds between summed frames.
            sum_iteration_callback: Called after each summed frame.
            force_new_capture: If True, wait for a new camera capture.
            new_capture_timeout_s: Max seconds to wait for the new-capture
                grab (passed positionally to driver.grab_new_capture). The
                historical bare name `new_capture_timeout` claimed "ms" in
                docs while the value flowed unchanged into a seconds API;
                the unit-suffix rename closes the contract ambiguity.
            verify_chunk_targets: Reject frames whose per-frame chunk
                metadata (ChunkExposureTime / ChunkGain) does not match the
                most recently requested exposure / gain targets, retrying
                until a matching frame arrives or timeout_s expires. This is
                the deterministic backstop against saving a frame that was
                still exposing when the settings changed. No-op for cameras
                without chunk support or when no target is recorded (e.g.
                hardware auto-gain owns the value).
            judged: The capture's judgment of each frame (``_Judged``): a
                frame is passed over, before any other check, until one
                counts for every change it does not exclude, and when its
                bound passes the grab ends ``'no_fresh_frame'`` naming the
                change. None for the ungated grab, which judges nothing.

        Returns:
            numpy.ndarray | _NoFrame: The captured image, or the record of
                a grab that got none, which the caller raises once its own
                cleanup has run (``_NoFrame.error``). Nothing is raised or
                reported here.

                Shape is (H, W) 2D mono for mono-native cameras and
                (H, W, 3) RGB for color-native cameras. Probe
                `scope.capabilities.is_color_native` to disambiguate.
                This method does NOT apply layer false-color -- apply
                at the display / encode boundary via
                `image_utils.mono_to_rgb_falsecolor(img, color)`.

                Dtype is uint8 when force_to_8bit=True, and for a single
                frame from an 8-bit camera; uint16 when force_to_8bit=False
                for 12/16-bit cameras (uint16 container holds the native bit
                width), and for a sum on every camera (its counts pass one
                frame's range). ``capture_still`` hands back the depth the
                frame carries (``CapturedStill.significant_bits``).
        """

        if not self._driver or not self._driver.active:
            return _NoFrame('removed', 'entry')

        frames: list[Frame] = []
        # What the kept frames were found to be, for the capture's record:
        # the most saturated of them, and whether any was kept dark. None
        # where the check was not asked.
        kept_saturation = 0.0 if all_ones_check else None
        kept_dark = False if dark_floor_check else None
        # Arrival ordinal of the frame most recently taken into the sum; None
        # until the first one lands. Local by construction: a bound that
        # outlived one capture would reject the next capture's frames.
        earliest_seq = None
        timeout_ns = int(timeout_s * 1e9)
        for frame_index in range(sum_count):
            # Waiting for a sum's next frame is its own leg, for the log.
            leg = 'sum' if frame_index else 'grab'
            # Timed on the clock the frames' arrivals carry, so a give-up
            # line's wait and the last frame's age are on one clock.
            mark = _FeedMark.read(self._driver)
            stop_ns = mark.at_ns + timeout_ns

            while True:
                # Acquire cam_lock for camera grab -- prevents concurrent
                # set_gain_db/set_exposure from another thread mid-frame.
                with self._camera._cam_lock:
                    frame = (
                        self._driver.grab_new_capture(new_capture_timeout_s)
                        if force_new_capture
                        else self._driver.grab()
                    )
                    if frame is not None:
                        self.frame_validity.count_frame(frame.seq, by=Counter.GRAB)

                if frame is None:
                    # A camera that is gone will not deliver; don't retry for
                    # the whole timeout. The removal latch is set the moment a
                    # removal is seen; ``active`` is released later, off-thread,
                    # so it alone would miss a failure in between.
                    if self._camera.camera_removed or not self._driver.active:
                        logger.error('[SCOPE API ] get_image: camera disconnected')
                        return _NoFrame('removed', leg)
                    if host_clock.steady_ns() > stop_ns:
                        logger.error(
                            f'[SCOPE API ] get_image timeout ({timeout_s:.1f}s) exceeded; '
                            f'{mark.account(self._driver)}'
                        )
                        return _NoFrame('no_frame', leg)
                    logger.debug('[SCOPE API ] get_image grab failed, retrying')
                    time.sleep(0.05)
                    continue

                # A frame the one judgment does not count for some change was
                # exposed before that change reached it: the next frame is
                # taken, and every later check sees only a frame that counts.
                unmet = judged.unmet(self.frame_validity, frame) if judged is not None else None
                if unmet is not None:
                    expired = (
                        host_clock.steady_ns() > judged.clock_bound_ns
                        if judged.clock_bound_ns is not None
                        else host_clock.steady_ns() > stop_ns
                    )
                    source = unmet.change.source
                    if expired:
                        logger.warning(
                            f'[SCOPE API ] get_image: no frame counted for the {source} '
                            f'change within the grab bound (last frame seq={frame.seq}, '
                            f'{unmet.change.rule.rule.value} rule) -- capture rejected. '
                            f'{mark.account(self._driver)}'
                        )
                        return _NoFrame('no_fresh_frame', leg, source=source)
                    logger.debug(
                        f'[SCOPE API ] get_image: frame seq={frame.seq} does not count '
                        f'for the {source} change; waiting for the next'
                    )
                    if not force_new_capture:
                        time.sleep(0.05)
                    continue

                # Saturation measures against the frame's own payload depth,
                # never the container dtype -- a 12-bit frame in a uint16
                # container is blown at 4095, not 65535.
                frame_saturation = (
                    self.saturated_fraction(frame.image, frame.significant_bits)
                    if all_ones_check
                    else None
                )
                frame_dark = False
                if frame_saturation is not None and frame_saturation >= BLOWN_FRACTION:
                    # Near-fully-saturated frame -- retry once in case it was a
                    # transient blip, then surface it. A blown frame is usually
                    # an over-exposure or stale-camera-gain symptom; accepting it
                    # silently (the prior behavior) hid real data corruption.
                    with self._camera._cam_lock:
                        retry = (
                            self._driver.grab_new_capture(new_capture_timeout_s)
                            if force_new_capture
                            else self._driver.grab()
                        )
                        if retry is not None:
                            self.frame_validity.count_frame(retry.seq, by=Counter.GRAB)
                    # Saturation walk is outside cam_lock -- no camera state needed,
                    # and the walk would otherwise block concurrent set_gain_db/set_exposure.
                    retry_saturation = (
                        self.saturated_fraction(retry.image, retry.significant_bits)
                        if retry is not None
                        else None
                    )
                    if retry_saturation is not None and retry_saturation < BLOWN_FRACTION:
                        frame, frame_saturation = retry, retry_saturation
                    else:
                        # Kept, and recorded as blown on the capture's record;
                        # whoever took the capture tells it (a run once per
                        # step, a still once), so it is not told here.
                        logger.debug(
                            f'[SCOPE API ] get_image: kept a frame '
                            f'{frame_saturation * 100.0:.0f}% saturated'
                        )

                if dark_floor_check:
                    lit_fraction = self._lit_fraction(frame.image, frame.significant_bits)
                    if lit_fraction < self._DARK_MIN_LIT_FRACTION:
                        # The caller declared illumination ON, yet no pixel
                        # clears the floor: the frame integrated before the
                        # LED lit, or the camera is delivering black frames.
                        # Retry first -- the next frame usually integrates
                        # under the lit LED. When the budget runs out the
                        # frame is SAVED anyway: a dark frame is an
                        # observation the operator can see on screen, not a
                        # failure, and destroying it loses real data (a dim
                        # transmitted setting, a genuinely dark sample).
                        # Darkness says nothing about the LED -- only tracked
                        # illumination state does -- so it decides nothing
                        # here beyond what gets logged and recorded.
                        if host_clock.steady_ns() > stop_ns:
                            # Recorded as dark on the capture's record, and
                            # told by whoever took the capture, as a blown
                            # frame is.
                            logger.debug(
                                f'[SCOPE API ] get_image: frame is dark -- '
                                f'{lit_fraction:.6f} of pixels above '
                                f'{self._DARK_FLOOR_FRACTION:.0%} of full scale '
                                f'(minimum {self._DARK_MIN_LIT_FRACTION}) with '
                                f'illumination expected ON; no lit frame within '
                                f'{timeout_s:.1f}s. Saving the dark frame.'
                            )
                            # Carried out as a FACT about the frame, never a
                            # failure cause: it rides the frame's record, so
                            # the writer, the saved file and an in-process
                            # caller can tell a dark frame from a lit one
                            # without anyone re-measuring pixels downstream.
                            frame_dark = True
                            # No break/continue: the dark frame is this
                            # capture, so it falls through the remaining
                            # gates (chunk targets, sum accumulation) on
                            # exactly the path a lit frame takes.
                        else:
                            logger.debug(
                                '[SCOPE API ] get_image: dark frame; waiting for a lit frame'
                            )
                            if not force_new_capture:
                                # Buffered grabs return the same frame until a
                                # new one arrives; pace the retry instead of
                                # spinning.
                                time.sleep(0.05)
                            continue

                if verify_chunk_targets:
                    # The frame must prove its own settings: its chunk
                    # exposure / gain must match the requested targets.
                    # Skip-count settling is a heuristic; a frame that
                    # started exposing before a long->short exposure change
                    # can arrive after the counter says valid, and saving it
                    # produces a saturated or mis-exposed image. Targets are
                    # set by the same thread that captures, so they are
                    # stable for the duration of this call -- a newer frame
                    # racing in here was exposed under the same settings and
                    # is equally acceptable.
                    stale_source = self._chunk_target_mismatch(frame.chunks)
                    if stale_source is not None:
                        if host_clock.steady_ns() > stop_ns:
                            chunks = frame.chunks or {}
                            chunk_key = self.frame_validity.CHUNK_KEY_FOR_SOURCE.get(stale_source)
                            logger.warning(
                                f'[SCOPE API ] get_image: frame chunk for '
                                f'{stale_source} never matched the requested '
                                f'target within {timeout_s:.1f}s '
                                f'(chunk={chunks.get(chunk_key)}, '
                                f'target={self.frame_validity.target(stale_source)}) -- '
                                f'either the camera is still delivering frames '
                                f'exposed under the previous settings, or it '
                                f'clamped the requested value. Capture rejected. '
                                f'{mark.account(self._driver)}'
                            )
                            return _NoFrame('chunk_rejected', leg, source=stale_source)
                        logger.debug(
                            f'[SCOPE API ] get_image: rejecting frame exposed '
                            f'under stale {stale_source}; waiting for a frame '
                            f'matching the requested value'
                        )
                        if not force_new_capture:
                            # Buffered grabs return the same frame until a new
                            # one arrives; pace the retry instead of spinning.
                            time.sleep(0.05)
                        continue

                # Accept the frame. Each frame of a sum must be NEWER than the
                # one before it, judged by arrival ordinal: frame numbers only
                # go up, while wall time runs backwards across a DST fall-back,
                # an NTP correction or a host resume -- and a sum ordered on a
                # clock that jumped forward rejects every genuinely new frame
                # until it times out, losing the capture. The first frame of a
                # sum has nothing to be newer than.
                if earliest_seq is None or frame.seq > earliest_seq:
                    frames.append(frame)
                    if frame_saturation is not None:
                        kept_saturation = max(kept_saturation, frame_saturation)
                    if frame_dark:
                        kept_dark = True
                    break

                # The expected path for a buffered camera, not a fault: the
                # driver returned the frame we already summed, so poll again.
                logger.debug(
                    f'[SCOPE API ] get_image: frame {frame.seq} already summed '
                    f'(need > {earliest_seq}); waiting for the next arrival'
                )

                # Timestamp not met -- check timeout then retry
                if host_clock.steady_ns() > stop_ns:
                    logger.error(
                        f'[SCOPE API ] get_image timeout ({timeout_s:.1f}s) exceeded; '
                        f'{mark.account(self._driver)}'
                    )
                    return _NoFrame('no_frame', leg)
                time.sleep(0.05)

            if sum_count > 1:
                earliest_seq = frame.seq
                if sum_iteration_callback is not None:
                    sum_iteration_callback()

                time.sleep(sum_delay_s)

        # Chain via a local variable instead of self.image_buffer. The old
        # field was a permanent shadow copy of the latest get_image result,
        # only ever read by get_image itself -- shadow state that pinned a
        # frame indefinitely between calls. The _state_lock around per-write
        # didn't actually serialize concurrent get_image calls anyway (chained
        # writes from different threads could still interleave).
        # The depth the frames were delivered at, carried by the frames
        # themselves: with the count, it is the sum's whole account.
        frame_bits = frames[0].significant_bits
        with self._state_lock:
            self._last_frame_summing = (sum_count, frame_bits)

        if sum_count == 1:
            image = frames[0].image
        else:
            # A sum is stored in a 16-bit container on every camera, an 8-bit
            # one included: its counts pass any one frame's range, and the
            # container is where they saturate.
            container_max = np.iinfo(np.uint16).max

            combined = np.zeros_like(frames[0].image, dtype=np.uint32)
            for summed in frames:
                combined += summed.image

            image = np.minimum(combined, container_max).astype(np.uint16)

        objective = self._scope.runtime_state.get_current_objective()
        use_scale_bar = self._resolve_use_scale_bar(
            self._scope.read_setting('scale_bar.enabled'), objective
        )

        need_8bit = force_to_8bit and image.dtype != np.uint8

        # A sum carries the bits it can reach; a single frame carries the
        # camera's native payload depth. The scale bar's white value follows
        # that depth so the bar is full white in the file, and the 8-bit
        # rendering scales a single frame against it and a sum against one
        # frame's white -- a sum is rendered brighter, which is what summing
        # is for.
        if use_scale_bar:
            image = image_utils.add_scale_bar(
                image=image,
                objective=objective,
                binning_size=self._camera.binning,
                frames_summed=sum_count,
                frame_bits=frame_bits,
                capabilities=self._scope.capabilities,
            )

        if need_8bit:
            image = image_utils.convert_sum_to_8bit(image, sum_count, frame_bits)

        # Read-only whichever path made it, as the camera's own frame is: the
        # capture's image is the instrument's account, not a scratch buffer.
        image.flags.writeable = False
        return _Grabbed(
            image=image,
            frame=frames[-1],
            earliest=frames[0],
            saturated_fraction=kept_saturation,
            dark=kept_dark,
        )

    @api(in_process=True)
    def judge_frame(self, frame: Frame, mapping: ClockMapping | None) -> tuple[Judgment, ...]:
        """Whether ``frame`` counts for each source's last change, by the rule a capture waits on.

        The one decision (``FrameValidity.judge``), asked of every change
        recorded so far (gain, exposure, black level, LED, a move, ...), by
        the rule each change's source runs (``capabilities.frame_rules``): a
        capture's record carries it for the frame it returns, and a frame
        listener can ask it of any frame it receives. Each ``Judgment``
        carries the change it was judged against: the last frame the driver
        had been handed when it was recorded, when it was recorded, its rule,
        the frames it needed and who counted them, and for a move when its
        axis settled.

        Args:
            frame: A frame the camera delivered (a frame listener's ``Frame``).
            mapping: The camera's clock mapped onto the host's, which the
                clock rule judges a frame's start by: a capture's
                ``frame_record.clock_mapping``. None for a camera without a
                latch; a clock-rule change then counts no frame.

        Raises:
            Nothing.
        """
        return self.frame_validity.judgments(frame, mapping)

    @api(in_process=True)
    def recorded_changes(self) -> dict[FrameValiditySource, Change]:
        """Each source's last recorded change, settled or not, by source.

        What a frame is judged against (``judge_frame``): when the change was
        recorded, the last frame the driver had been handed then, its rule,
        and for a move when its axis settled. A change stays its source's until the
        next one replaces it.

        Raises:
            Nothing.
        """
        return self.frame_validity.changes()

    @api(in_process=True)
    @contextlib.contextmanager
    def held_frame_rules(self, record: dict) -> Iterator[None]:
        """Judge the changes recorded inside the ``with`` block by counts not yet on disk.

        Use as ``with scope.imaging.held_frame_rules(record): ...``. The
        rules are built from ``record`` as bring-up builds them from the
        camera's timing record file (``capabilities.frame_rules``), and are
        handed to the gate for the block: each change recorded inside it
        waits its held count. When the block ends, however it ends, the
        installation's rules are back, and each change recorded after it
        waits the installation's count. A change keeps the rule it was
        recorded under, on either side. For a characterization's verify
        phase, which captures under the counts it just measured before they
        are saved; ``capabilities.frame_rules`` still publishes the
        installation's.

        Args:
            record: A camera timing record, schema 3, checked as a file is
                (``FrameValidity.check_timing_record``).

        Raises:
            ArgumentRefusedError: ``'wrong_kind'``, a record this release
                cannot take as measured; nothing was held.
            ControlSessionRefusedError: ``'control_session_held'``, a control
                session is held by someone else; nothing was held.
        """
        try:
            if not isinstance(record, dict):
                raise ValueError('not a dictionary')
            FrameValidity.check_timing_record(record)
        except ValueError as refused:
            raise ArgumentRefusedError(
                'wrong_kind',
                argument='record',
                value=record,
                takes=f'a camera timing record this release reads ({refused})',
            ) from refused
        rules = FrameValidity.rules_from_record(
            record, self._scope.capabilities.camera_has_clock_latch
        )
        with a_guarded_write(self._scope._activity_claim):
            self.frame_validity.use_rules(rules)
        try:
            yield
        finally:
            self.frame_validity.use_rules(self._scope.capabilities.frame_rules)

    @api(in_process=True)
    def get_image(
        self,
        force_to_8bit: bool = True,
        timeout_s: float = 5.0,
        all_ones_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0.0,
        sum_iteration_callback: Callable[[], None] | None = None,
        force_new_capture: bool = False,
        new_capture_timeout_s: float = 5.0,
        verify_chunk_targets: bool = False,
    ) -> np.ndarray:
        """Grab and return an image from the camera -- the UNGATED primitive.

        Never dark-rejects: liveness probes and diagnostics must see what
        the camera sees, dark or not. For any frame you will SAVE or
        MEASURE as truth, call capture_still, which drains the camera
        pipeline until the validity marker is GREEN and derives the
        dark-floor expectation from commanded LED state.

        Explicit parameters, never a **kwargs forward: the signature is a
        pinned contract (introspected by tests), and a silent kwarg
        pass-through would re-open the door this split closed.

        By default the last buffered frame is returned; ``force_new_capture``
        waits for a fresh one. The grab is one command on the camera's lane,
        waited for. It does not wait for frame validity, so the frame may not
        reflect a just-changed gain, exposure, LED or stage position.

        Args:
            force_to_8bit: True for an 8-bit image; False for the camera's
                own depth, in a uint16 array for a 12- or 16-bit camera.
            timeout_s: Seconds to wait for a frame.
            all_ones_check: Measure each frame's saturation: a blown frame is
                grabbed again once, then kept.
            sum_count: Frames summed into the image, for noise reduction; a
                sum's array is uint16.
            sum_delay_s: Seconds between summed frames.
            sum_iteration_callback: Called after each summed frame, on the
                camera's lane.
            force_new_capture: True to wait for a frame the camera captures
                after the call, rather than take the last buffered one.
            new_capture_timeout_s: Seconds to wait for that new frame.
            verify_chunk_targets: Pass over frames whose own record of
                exposure and gain (the camera's chunk data) differs from the
                last values asked for, until one matches or ``timeout_s``
                passes; no effect on a camera without chunk data, or while
                hardware auto-gain owns the values.

        Returns:
            The image: (H, W) for a mono camera, (H, W, 3) for a color one,
            with no channel's false colour applied.

        Raises:
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``, a
                time is not a wait from 0 to 30 days, or the wait for every
                frame of the sum together is longer. Nothing reached the
                camera.
            CaptureFailedError: ``'no_frame'``, no frame within the timeouts;
                ``'chunk_rejected'``, frames came but none at the ``setting``
                asked (``verify_chunk_targets``).
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, absent at the call or removed during the grab.
        """
        for name, value in (
            ('timeout_s', timeout_s),
            ('sum_delay_s', sum_delay_s),
            ('new_capture_timeout_s', new_capture_timeout_s),
        ):
            refuse_unless_wait_s(value, name)
        # A liveness bound, not a budget: each frame of a sum may wait out its
        # retry window and then one last new-capture grab, plus its delay.
        wait_s = self._CAPTURE_WAIT_TIMEOUT_S + sum_count * (
            timeout_s + new_capture_timeout_s + sum_delay_s
        )
        # Each time in range, a long sum can still make a wait no host holds.
        refuse_unless_wait_s(wait_s, 'the wait for every frame of the sum')
        grabbed = self._camera._dispatch_camera(
            self._get_image_impl,
            'get_image',
            kwargs={
                'force_to_8bit': force_to_8bit,
                'timeout_s': timeout_s,
                'all_ones_check': all_ones_check,
                'sum_count': sum_count,
                'sum_delay_s': sum_delay_s,
                'sum_iteration_callback': sum_iteration_callback,
                'force_new_capture': force_new_capture,
                'new_capture_timeout_s': new_capture_timeout_s,
                'verify_chunk_targets': verify_chunk_targets,
            },
            timeout_s=wait_s,
        )
        if isinstance(grabbed, _NoFrame):
            raise grabbed.error('get_image')
        return grabbed.image

    @api(in_process=True)
    def get_image_from_buffer(
        self, force_to_8bit: bool = True, out_8bit: np.ndarray | None = None
    ) -> tuple:
        """Grab the latest buffered frame from the camera without forcing a new capture.

        Copy budget (per frame):
          - grab(): 0 copies (returns the read-only stored frame)
          - add_scale_bar(): 1 copy when the bar is drawn (the stored frame
            is shared with every other consumer and is never written)
          - convert_to_8bit(): 1 copy (LUT indexing creates new array),
            or 0 fresh allocations when the caller supplies out_8bit.
          - Total without the bar: 0 copies (8-bit) or 1 copy (12-bit with
            force_to_8bit); the bar adds 1
          The caller adds 1 more copy via tobytes() for GPU blit.

        Args:
            force_to_8bit: Convert 12-bit images to 8-bit output.
            out_8bit: Optional caller-owned (H, W) uint8 buffer reused as the
                12->8 LUT destination, avoiding a fresh per-frame allocation
                on the 30 fps preview path. Each caller must supply its OWN
                buffer (the preview thread and the histogram run on different
                threads); a None / mismatched buffer falls back to a fresh
                allocation. The returned array is the buffer itself when used,
                so the caller must copy (e.g. tobytes()) before the next call
                overwrites it.

        Returns:
            tuple: (image, timestamp) where image is numpy.ndarray and timestamp
                   is from the camera SDK, or (None, None) if unavailable.
                   Per the Sentinel-return contract: `if image is None: ...`.

        Raises:
            ConfigurationDefectError: ``'scope_not_composed'``, no session has bound
                this scope, so it has no scale-bar setting to read.
        """
        if not self._driver or not self._driver.active:
            return None, None

        frame = self._driver.grab()
        if frame is None:
            return None, None
        # grab() returns the same buffered frame on every poll, and
        # this preview path can poll faster than the camera delivers. The
        # frame's arrival ordinal dedupes the count so validity skip counts
        # retire against real frames, not poll rate -- counting polls let a
        # capture accept a frame exposed under the previous gain/exposure/LED
        # state. The ordinal also keeps a frame this path grabbed before a
        # write from retiring any of that write's wait.
        self.frame_validity.count_frame(frame.seq, by=Counter.PREVIEW)
        tmp = frame.image

        objective = self._scope.runtime_state.get_current_objective()
        use_scale_bar = self._resolve_use_scale_bar(
            self._scope.read_setting('scale_bar.enabled'), objective
        )

        if use_scale_bar:
            tmp = image_utils.add_scale_bar(
                image=tmp,
                objective=objective,
                binning_size=self._camera.binning,
                frames_summed=1,
                frame_bits=frame.significant_bits,
                capabilities=self._scope.capabilities,
            )

        if force_to_8bit and tmp.dtype != np.uint8:
            tmp = image_utils.convert_to_8bit(tmp, frame.significant_bits, out=out_8bit)

        return tmp, frame.timestamp

    @api
    @property
    def significant_bits(self) -> int:
        """Meaningful payload bits of frames the current camera delivers.

        The depth a single captured frame should be scaled / tagged by (12 for a
        Mono12 sensor, 8 for an 8-bit one); it follows ``set_image_mode``. A
        summed frame carries the bits the sum can reach and is not described
        by this -- ``CapturedStill.significant_bits`` answers for it. The deepest the
        camera offers is ``capabilities.camera_max_significant_bits``.

        Derived from the HELD pixel format (the one the camera last reported
        or took)
        via the driver's own depth rule (``significant_bits_for_format``, so
        a constant-depth driver like FX2 stays authoritative), not a live
        driver read: a transient format-read failure once made a 12-bit
        camera report depth 16 here, so a fully blown frame passed the
        saturation evidence check at 0.0% and files were stamped at the
        wrong depth. A frame you are HOLDING carries its own depth
        (``Frame.significant_bits``; ``CapturedStill.significant_bits`` for a capture).

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, the scope has no
                camera, so no frames have a depth.
        """
        driver = self._driver
        if driver is None:
            raise _camera_not_connected('significant_bits')
        return driver.significant_bits_for_format(self._camera.pixel_format)

    # --- Streaming control ---
    @api(admitted_by=(Admission.LANE_CALL,))
    def enable_streaming(self) -> None:
        """Begin camera streaming -- the public way to start the live feed, and wait.

        After ``connect()`` the camera is configured but NOT grabbing (the
        camera-lifecycle split); ``Lumascope.initialize`` makes the one-time
        bring-up start. This restarts a feed that was deliberately stopped,
        and is idempotent on a feed already running. Refused, naming the camera,
        when none is connected, or when the start finds it removed. A start
        the FX2 does not take with the device still present raises its fault
        (``CameraStreamStartFailedError`` for a START the device refused), the
        camera left connected and not streaming; a later call tries again.
        The start is one command on the camera's lane, waited for up to the
        lane's geometry bound, since a start of the grab has been measured
        near 11 s on a Pylon body.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one the start finds removed.
            CameraStreamStartFailedError: the FX2 did not take the START with
                the device still present; the camera is left connected and not
                streaming.
        """
        return self._camera._dispatch_camera(
            self._enable_streaming_impl,
            'enable_streaming',
            timeout_s=self._camera._CAMERA_GEOMETRY_TIMEOUT_S,
        )

    def _enable_streaming_impl(self) -> None:
        driver = self._driver
        if driver is None:
            return
        # open_and_start reports whether it just fired the one-time start;
        # the restart poll runs only when the gate was ALREADY open (a feed
        # deliberately stopped earlier). This keeps the two ensure-running
        # mechanisms from stacking: a just-attempted start that failed is
        # not immediately retried against the same failed device.
        if not driver.open_and_start() and not driver.is_grabbing():
            driver.start_grabbing()

    @api(admitted_by=(Admission.LANE_CALL,))
    def disable_streaming(self) -> None:
        """Stop camera streaming, and wait.

        After this, ``capture_still()`` fails until streaming resumes;
        ``get_image()`` still returns the last frame the camera delivered.
        Satisfied with no camera connected: nothing is streaming. Dispatched
        as ``enable_streaming`` is.

        Raises:
            Nothing.
        """
        return self._camera._dispatch_camera(
            self._disable_streaming_impl,
            'disable_streaming',
            timeout_s=self._camera._CAMERA_GEOMETRY_TIMEOUT_S,
            satisfied_when_absent=None,
        )

    def _disable_streaming_impl(self) -> None:
        driver = self._driver
        if driver is None:
            return
        driver.stop_grabbing()

    @api(reads=True)
    def is_streaming(self) -> bool:
        """Whether the camera is currently acquiring frames.

        Queries the driver directly (unlike ``scope.camera.active``, which
        reads the held connected-state). False when no camera is attached.

        Raises:
            Nothing.
        """
        driver = self._driver
        if driver is None:
            return False
        return driver.is_grabbing()

    def is_streaming_once_settled(self) -> bool:
        """Whether the camera is acquiring once the camera commands queued ahead have run.

        A consult seam for a gate that must not refuse a camera mid-way
        through a change: a frame-size or binning change stops the grab and
        starts it again on the camera lane, so the direct read says "not
        grabbing" for its length. The direct read answers first; only when
        it says "not grabbing" is the question put on the lane, behind what
        is queued, so a streaming camera costs nothing. The lane read
        changes nothing a holder depends on, so it is admitted whatever
        holds the scope. False when no camera is attached.

        Raises:
            HardwareCommandRefusedError: ``'scope_disconnected'``, the lane
                is closed.
        """
        if self.is_streaming():
            return True
        return self._camera._dispatch_camera(
            self.is_streaming,
            'is_streaming_once_settled',
            timeout_s=self._camera._CAMERA_GEOMETRY_TIMEOUT_S,
            override=True,
            satisfied_when_absent=False,
        )

    # --- Operation flags ---
    @property
    def is_focusing(self) -> bool:
        """True while the microscope is running autofocus.

        Internal coordination flag -- set and read by the autofocus
        machinery and not part of the L2 API surface (clients read the
        session's run-state derivations).

        Returns:
            bool: True if an autofocus run is in progress.
        """
        return self._focusing_event.is_set()

    @is_focusing.setter
    def is_focusing(self, value: bool) -> None:
        """Set the autofocus-in-progress flag.

        Internal coordination flag -- set and read by the autofocus
        machinery and not part of the L2 API surface (clients read the
        session's run-state derivations).
        """
        if value:
            self._focusing_event.set()
        else:
            self._focusing_event.clear()

    # --- Frame validity ---
    @api(reads=True)
    def frames_until_valid(self, exclude_sources: tuple[FrameValiditySource, ...] = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Delegates to frame_validity.

        Args:
            exclude_sources: Sources to exclude from the validity check.

        Returns:
            int: Number of additional frames to drain before validity. 0 if
                already valid.

        Raises:
            Nothing.
        """
        return self.frame_validity.frames_until_valid(
            exclude_sources=exclude_sources,
        )

    @api(in_process=True, engineering=True)
    @property
    def last_capture_info(self) -> dict | None:
        """How the most recent capture on this scope waited, frame or not.

        What a capture leaves whether or not it returned a frame: the
        protocol file writer's log line, the manual still's retake bound
        and a measure of the hold and drain read it.
        What a frame is -- its exposure and gain and where each came from,
        its auto-gain outcome, its judgments -- is its ``FrameRecord``,
        handed back with it (``capture_still``), never this slot.

        Returns:
            dict | None: ``{'hold_ms', 'drained', 'rechecks', 'deadline_s',
                'n_entry', 'active_s'}`` for the latest capture past its
                call, whether it returned a frame or raised why it got none;
                None before the first. A capture refused at its call (no
                camera, or one not streaming) records nothing, so read this
                only after a capture that returned or raised
                ``CaptureFailedError``. Why a capture got no frame is what it
                raised, never this record. The next capture replaces it.

        Raises:
            Nothing.
        """
        with self._state_lock:
            return dict(self._last_capture_info) if self._last_capture_info else None

    # --- Scale bar ---
    def _resolve_use_scale_bar(self, enabled: bool, objective) -> bool:
        """Whether to draw the scale bar, loud once when it cannot be drawn.

        The no-objective skip is by design -- without an objective there is
        no pixel size, and a bar of invented length is a false measurement
        claim -- but a user who ENABLED the bar deserves one log line saying
        why it is not appearing. Latched once per no-objective episode
        because the callers run per displayed frame.
        """
        if objective is not None:
            self._scale_bar_objective_skip_logged = False
            return enabled
        if enabled and not self._scale_bar_objective_skip_logged:
            self._scale_bar_objective_skip_logged = True
            logger.warning(
                '[SCOPE API ] Scale bar is enabled but the objective in the '
                'light path is unknown; skipping the bar until it is known.'
            )
        return False

    # --- Camera diagnostics (live in-flight only; data source = DiagnosticsAPI) ---
    def _log_camera_temps(self) -> None:
        """Emit one INFO line per camera temperature sensor.

        No-op when no camera is connected. Called once on startup and
        periodically by ``start_camera_temp_logging``. Reads temperatures
        through `scope.diagnostics.get_camera_temperatures_degc` -- the canonical
        camera-temp probe (cold probes live on DiagnosticsAPI).

        Raises:
            HardwareError: The read failed. On a tick the session's scheduler
                reports it and keeps the schedule.
        """
        temps = self._scope.diagnostics.get_camera_temperatures_degc()
        if temps is None:
            return
        for source, temp in temps.items():
            logger.info(f'[CAM Class ] Camera {source} Temperature : {temp:.2f} degC')

    def start_camera_temp_logging(
        self,
        schedule_interval_fn: Callable[[Callable[..., None], float], object],
        unschedule_fn: Callable[[object], None],
        *,
        interval_s: float = 14400.0,
    ) -> None:
        """Own the periodic camera-temp logging schedule.

        Internal metrics scheduling -- not part of the L2 API surface
        (clients read ``get_camera_temperatures_degc`` and schedule their
        own logging).

        Was previously a Clock.schedule_interval registered by the App
        and stored as a fresh attribute on the MainDisplay widget -- if
        MainDisplay was ever recreated (LS850/LS620 scope swap), the
        Clock event became orphaned and continued logging temps from a
        now-disconnected camera.

        Args:
            schedule_interval_fn: Callable matching ``Clock.schedule_interval(func, interval)``.
                Passed in so this module stays GUI-agnostic.
            unschedule_fn: Callable matching ``Clock.unschedule(event)``,
                used by ``stop_camera_temp_logging``.
            interval_s: Seconds between log emissions; default 4 hours.
        """
        # Defensive: if a previous logger is already running, stop it
        # before starting a new one (idempotent -- safe to call repeatedly).
        # No unschedule_fn arg: the OLD event must be cancelled with the
        # scheduler it was registered on (the stored fn), not the one
        # being handed in for the new schedule.
        if getattr(self, '_camera_temp_event', None) is not None:
            self.stop_camera_temp_logging()

        self._camera_temp_unschedule_fn = unschedule_fn
        # One immediate sample. Its caller is the host bringing metrics up,
        # which a failed read must not stop, so the failure is reported here,
        # where its flight ends, as a tick's is by the scheduler.
        try:
            self._log_camera_temps()
        except HardwareError as ex:
            notifications.report_outcome(ex, solicited=False, category='Camera')

        def _tick(_dt=0):
            # camera_connected is an instantaneous poll and a False can be
            # transient (a single flaky connectivity query), so the tick
            # skips the sample (_log_camera_temps guards internally) but
            # STAYS SCHEDULED -- self-unscheduling here permanently ended
            # temperature logging for the rest of a multi-day soak on one
            # transient False. Teardown belongs to the explicit owners:
            # stop_camera_temp_logging via the metrics logger stop and the
            # scope-swap path.
            self._log_camera_temps()

        self._camera_temp_event = schedule_interval_fn(_tick, interval_s)
        logger.info(f'[SCOPE API ] start_camera_temp_logging: interval={interval_s}s')

    def stop_camera_temp_logging(self, unschedule_fn=None) -> None:
        """Cancel the periodic camera-temp logger if active.

        Internal metrics scheduling -- pair of
        ``start_camera_temp_logging``; not part of the L2 API surface.

        Idempotent -- safe to call when no logger is running. The
        unschedule_fn arg is optional; falls back to the function passed
        at start_camera_temp_logging time.
        """
        ev = getattr(self, '_camera_temp_event', None)
        if ev is None:
            return
        try:
            (unschedule_fn or self._camera_temp_unschedule_fn)(ev)
        except Exception as e:
            logger.warning(f'[SCOPE API ] stop_camera_temp_logging unschedule failed: {e}')
        self._camera_temp_event = None

    # --- Stream check ---
    # How often the stream check looks at the frame count. Well inside the
    # shortest stall bound (video_cadence.STALL_FLOOR_S), so a stall is seen
    # within a second of the bound passing.
    _STREAM_CHECK_INTERVAL_S = 1.0

    def start_stream_check(self, scheduler: Scheduler) -> None:
        """Watch the camera stream, and report a stall when frames stop arriving.

        A camera can stall without being removed: it stays connected and
        grabbing and no frame arrives. Nobody is waiting on the stream then,
        so the imaging API watches its own frame count on the session's
        scheduler, in every host, and reports ``CameraStreamStalledError``
        unsolicited once per stall. A stall is the count standing still
        while the camera is connected and streaming, or its grab is wanted
        and has stopped on its own (the driver's ``grab_wanted``), for ten
        times the longest exposure the camera may be using
        (``longest_exposure_ms``, the sensor's maximum while the exposure is
        unknown) and never less than 5 s (``prologue_stall_threshold_s``).
        A stop someone asked for -- a
        frame-size, format or binning change restarts it -- starts the watch
        again from the next frame.

        Internal scheduling -- the session arms it at bring-up and the
        scope's disconnect stops it; not part of the L2 API surface.

        Args:
            scheduler: The session's scheduler (``schedule_interval`` /
                ``unschedule``).
        """
        self.stop_stream_check()
        self._stream_check_scheduler = scheduler
        self._stream_watch = None
        self._stream_stall_reported = False
        self._stream_check_handle = scheduler.schedule_interval(
            self._check_stream, self._STREAM_CHECK_INTERVAL_S
        )

    def stop_stream_check(self) -> None:
        """Stop the stream check. Idempotent.

        Internal scheduling -- pair of ``start_stream_check``, called by the
        scope's disconnect; not part of the L2 API surface.
        """
        handle = self._stream_check_handle
        self._stream_check_handle = None
        if handle is not None:
            self._stream_check_scheduler.unschedule(handle)

    @api(reads=True)
    def get_delivered_rate(self) -> DeliveredRate:
        """Read the rate the camera delivered frames, and their bytes on the link.

        Measured over the last second from the frames every driver stores,
        in every host, so it is the camera's rate whether a display is
        drawing, paused or absent. A window older than two checks -- the
        camera stopped streaming, was disconnected, or the check stopped --
        reads 0, never the last rate seen.

        Returns:
            DeliveredRate: Frames and link bytes per second; ``NOT_DELIVERING``
                (both 0) when nothing was delivered in the last window.

        Raises:
            Nothing.
        """
        published = self._delivered_rate
        if published is None:
            return NOT_DELIVERING
        rate, measured_at = published
        if time.monotonic() - measured_at > 2 * self._STREAM_CHECK_INTERVAL_S:
            return NOT_DELIVERING
        return rate

    def _sample_delivery(self, streaming: bool) -> None:
        """Take the delivered rate over the window since the last reading."""
        driver = self._driver
        if not streaming or driver is None:
            # Nothing is delivered by design; the next window starts from the
            # stream's next reading.
            self._delivery_reading = None
            self._delivered_rate = None
            return
        now = time.monotonic()
        frames, wire_bytes = driver.delivered_counts
        previous = self._delivery_reading
        if (
            previous is not None
            and frames >= previous[0]
            and now - previous[2] < self._STREAM_CHECK_INTERVAL_S / 2
        ):
            # The scheduler ran this check again straight after the last one;
            # the window is measured from the earlier reading, over a full tick.
            return
        self._delivery_reading = (frames, wire_bytes, now)
        if previous is None or frames < previous[0]:
            # The first reading, or a rebuilt handler (a reconnect, a driver
            # swap) whose count started again from 0: no window yet.
            return
        interval_s = now - previous[2]
        self._delivered_rate = (
            DeliveredRate(
                frame_rate_fps=(frames - previous[0]) / interval_s,
                bytes_per_s=(wire_bytes - previous[1]) / interval_s,
            ),
            now,
        )

    def _check_stream(self) -> None:
        # A grab that is wanted and not running stopped when nobody asked,
        # and is watched as a stall like a grab that delivers nothing.
        driver = self._driver
        streaming = self._scope.camera_connected and (
            self.is_streaming() or (driver is not None and driver.grab_wanted)
        )
        self._sample_delivery(streaming)
        if not streaming:
            # A camera that is not streaming delivers nothing by design; the
            # watch starts again from the stream's next frame.
            self._stream_watch = None
            self._stream_watch_bound_s = None
            self._stream_stall_reported = False
            return
        exposure_s = max(0.0, self._camera.longest_exposure_ms or 0.0) / 1000.0
        bound_s = prologue_stall_threshold_s(exposure_s)
        # Frames stored, not handed over: a store stage that has wedged while
        # the camera still hands frames over (Pylon's worker, the IDS unpack
        # worker, the FX2 grab loop) delivers nothing a reader can have. A
        # stream that is wanted has a driver (``streaming`` above).
        stored, _ = driver.delivered_counts
        held_s = self._stream_watch_bound_s
        if held_s is not None and bound_s < held_s:
            # A shorter exposure: the frames already in the camera's pipeline
            # were exposed at the longer one, so the longer bound holds until
            # as many frames as the camera's measured exposure change takes
            # to reach its output have been handed over since the change.
            # Counted from the change, never from when this check first sees
            # the shorter bound: ``longest_exposure_ms`` answers the longer
            # exposure for a while after the write, and frames counted only
            # from its end hold the longer bound over a stall that begins
            # after the old frames have all come.
            change = self.frame_validity.changes().get(FrameValiditySource.EXPOSURE)
            if change is not None and driver.frames_delivered - change.at_seq < change.skip:
                bound_s = held_s
        watch = self._stream_watch
        if watch is None or self._stream_watch_bound_s != bound_s:
            # A new exposure changes how long a healthy frame can take; the
            # bound is set from it and the window starts again.
            watch = self._stream_watch = StallWatch(bound_s)
            self._stream_watch_bound_s = bound_s
        now = time.monotonic()
        if not watch.stalled(stored, now):
            if self._stream_stall_reported:
                logger.info(f'[CAM Class ] Camera stream resumed: frames_stored={stored}')
            self._stream_stall_reported = False
            return
        if self._stream_stall_reported:
            return
        self._stream_stall_reported = True
        seconds = watch.quiet_for_s(now)
        logger.info(
            f'[CAM Class ] Camera stream stalled: frames_stored={stored} unchanged '
            f'for {seconds:.1f} s (bound {bound_s:.1f} s at exposure {exposure_s * 1000.0:.1f} ms)'
        )
        notifications.report_outcome(
            CameraStreamStalledError(seconds), solicited=False, category='Camera'
        )

    @api(in_process=True, event=FrameDelivered)
    def add_frame_listener(
        self,
        cb: Callable[[Frame], None],
        name: str | None = None,
    ) -> None:
        """Register a per-frame listener fired on every successful grab.

        The canonical entry point for live_processing plugins (see
        ``ctx.plugins.live_processing``) and the manual-record path.
        The supplied handler is wrapped in a budget enforcer
        (``HANDLER_BUDGET_MS`` per call; ``HANDLER_DROP_K`` consecutive
        over-budget or raising invocations remove it, reported once as a
        ``FrameHandlerRemovedError`` warning). Callback
        signature is ``cb(frame)``, the camera's ``Frame`` record
        (``modules.lumascope_api.Frame``: ``image``, ``seq``,
        ``timestamp``, ``arrived_ns``, ``significant_bits``, ``chunks``);
        runs on the SDK
        callback thread (Pylon ``PylonImageGrab`` / IDS grab loop /
        simulated acquisition thread). Listeners MUST NOT block -- heavy work belongs
        on an executor.

        Args:
            cb: Per-frame handler. Signature ``cb(frame)``.
            name: Display name for log + notification messages on
                  over-budget / auto-removal. Defaults to the handler's
                  qualname.

        The record is shared across all listeners, the preview and a
        capture, and is read-only: ``frame.image`` refuses a write
        (``ValueError``) and ``frame.chunks`` is a read-only mapping. Never
        set the image writeable again: every other holder would see the
        change. Draw on your own copy. Keeping the record is safe: no later
        frame changes it.

        Registration is idempotent for the same callable -- a second
        call with the same ``cb`` is a no-op (the original wrapper +
        name are kept).

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera. Nothing is registered.
        """
        self._camera.refuse_camera_not_connected('add_frame_listener')
        if name is None:
            name = getattr(cb, '__qualname__', None) or repr(cb)
        with self._frame_listener_lock:
            if cb in self._frame_listener_wrappers:
                return  # idempotent
            wrapper = _BudgetedHandler(self, cb, name)
            self._frame_listener_wrappers[cb] = wrapper
        try:
            self._driver.register_frame_callback(wrapper)
        except BaseException:
            # Rolled back so a later registration can retry.
            with self._frame_listener_lock:
                self._frame_listener_wrappers.pop(cb, None)
            raise

    @api(in_process=True)
    def remove_frame_listener(self, cb: Callable[[Frame], None]) -> None:
        """Remove a listener registered via ``add_frame_listener``.

        No new call reaches the handler once this returns (a call already
        running completes), whatever the camera driver does: the wrapper is
        marked removed before the driver is asked, so a driver that fails to
        unregister it goes on calling a wrapper that no longer calls through.
        No-op when the listener was never registered.

        Args:
            cb: The callback given to ``add_frame_listener``.

        Raises:
            Nothing.
        """
        with self._frame_listener_lock:
            wrapper = self._frame_listener_wrappers.pop(cb, None)
        if wrapper is None:
            return
        wrapper._removed = True
        if not self._driver:
            return
        try:
            self._driver.unregister_frame_callback(wrapper)
        except Exception as ex:
            logger.warning(
                f'[SCOPE API ] remove_frame_listener: the driver did not unregister '
                f"'{wrapper._name}' ({type(ex).__name__}: {ex}); it is no longer called"
            )

    def _remove_wrapper(self, wrapper: _BudgetedHandler) -> None:
        """Internal: auto-removal path. Called by _BudgetedHandler when
        K consecutive over-budget or raising calls trigger the drop. Idempotent --
        callable safely from the SDK callback thread."""
        with self._frame_listener_lock:
            cb_to_remove = None
            for cb, w in self._frame_listener_wrappers.items():
                if w is wrapper:
                    cb_to_remove = cb
                    break
            if cb_to_remove is None:
                return
            self._frame_listener_wrappers.pop(cb_to_remove, None)
        if self._driver:
            try:
                self._driver.unregister_frame_callback(wrapper)
            except Exception as ex:
                # The wrapper is already marked removed, so it no longer
                # calls through whatever the driver goes on doing.
                logger.warning(
                    f'[SCOPE API ] _remove_wrapper: the driver did not unregister '
                    f"'{wrapper._name}' ({type(ex).__name__}: {ex}); it is no longer called"
                )
