# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
from abc import ABC, abstractmethod
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import re
import threading
import time
from typing import Any

import numpy as np

from lib import profile_trace
from lvp_logger import logger

try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    # Fall back to the general logger, never None: the _cam_log call sites below
    # are unguarded, so a None fallback turns every one into an AttributeError the
    # moment the dedicated camera logger is unavailable.
    _cam_log = logger
from drivers.aoi_geometry import AoiPlan, center_crop, plan_aoi
from drivers.camera_profiles import CameraProfile, lookup_profile

default_max_exposure = 1_000  # in ms


@dataclass(frozen=True)
class FrameGrid:
    """The hardware windows a camera can acquire at its current binning, in displayed pixels.

    ``plan_aoi`` reads it: the legal sizes are ``size_min + k * step`` up to
    ``max_size``, and the legal offsets ``offset_min + k * offset_step``. A
    camera that centres its own window (``offset_step`` 1) is handed offsets
    it does not write; the crop centres on the window either way.
    """

    step: tuple[int, int]
    max_size: tuple[int, int]
    size_min: tuple[int, int] = (0, 0)
    offset_step: tuple[int, int] = (1, 1)
    offset_min: tuple[int, int] = (0, 0)
    bias: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class FrameWindow:
    """The window a camera acquires and the part of it the camera delivers.

    The acquired size travels with the crop, so a frame of any other size, made
    under a window no longer set, is recognised and not stored rather than
    cropped wrong. ``crop`` is ``(x0, y0, width, height)``, or None when the
    acquired window is the delivered one.
    """

    acquired_width: int
    acquired_height: int
    crop: tuple[int, int, int, int] | None

    @property
    def size(self) -> dict:
        """The delivered size, ``{'width', 'height'}``."""
        if self.crop is None:
            return {'width': self.acquired_width, 'height': self.acquired_height}
        return {'width': self.crop[2], 'height': self.crop[3]}

    @classmethod
    def from_plan(cls, plan: AoiPlan) -> 'FrameWindow':
        crop = (
            (plan.crop_x0, plan.crop_y0, plan.crop_width, plan.crop_height)
            if plan.needs_crop
            else None
        )
        return cls(plan.acq_width, plan.acq_height, crop)


class ImageHandlerBase:
    """Base class for camera image handlers (IDS and Pylon).

    Provides thread-safe frame buffer storage, copy-on-read, failure counting
    with auto-stop, and a consistent API for the Camera.grab() method.
    """

    MAX_CONSECUTIVE_FAILURES = 128

    # The no-frame answer of get_last_image(), in the delivered-frame shape.
    # One owner: every reader unpacks this tuple positionally, and a second
    # copy of it went short each time this one grew.
    NO_FRAME = (False, None, None, None, None)

    def __init__(self):
        self._frame_lock = threading.Lock()
        self.last_result = False
        self.last_img = None
        self.last_img_ts = None
        # Payload depth of the buffered frame, captured WITH it at store time so a
        # consumer reads the depth this frame was acquired under -- not whatever
        # the camera's pixel format reports later. A format switch leaves a prior
        # frame in this buffer; pairing it with a freshly-queried depth is what
        # mis-scaled (and crashed) the downconvert.
        self.last_img_significant_bits = None
        self.last_chunks = None  # per-frame chunk metadata dict (None when unsupported)
        # Arrival ordinal: how many frames this handler has delivered, and the
        # ordinal of the buffered one. Frame validity decides whether a frame
        # predates a hardware write by comparing these, so the value must be
        # something that cannot run backwards -- a wall clock can (DST, an NTP
        # step, a VM resume), and a settle count measured against a timestamp
        # that jumped forward never completes.
        self._frames_delivered = 0
        self.last_img_seq = None
        self._failed_grabs = 0
        # The camera's window, pushed here by the camera (set_frame_size, a
        # binning change, a rebuilt handler). None stores every frame as it
        # arrives.
        self.frame_window: FrameWindow | None = None
        self._frames_of_another_window = 0
        self._reported_window: FrameWindow | None = None
        # Per-frame consumers (manual record today; per-frame plugins later).
        # Snapshotted-then-released under _frame_lock at _store_frame time so
        # a slow callback never holds the SDK thread.
        self._frame_callbacks: list = []
        # Grab-arrival census. This is the ONE site every driver funnels
        # through, and it fires whether or not a listener is attached -- the
        # only way to learn whether the camera keeps delivering during a
        # protocol run, when the record listener is unregistered and nothing
        # downstream would notice a stalled stream. Batched because a per-row
        # write here would land inside the interval it times. Arrivals are a
        # property of the camera rather than of a recording: this instance
        # outlives every recording and keeps firing between them, so rows
        # correlate to a recording by timestamp against a recording-scoped
        # trace rather than by an id carried here.
        self._arrival_trace = profile_trace.BatchTrace(
            'camera_arrival_trace.csv',
            'ts_ms,interarrival_ms,n_listeners,fanout_ms,significant_bits,frame_bytes',
            profile_trace.NO_RECORDING,
        )
        self._last_arrival_t = None

    @property
    def frames_delivered(self) -> int:
        """How many frames this handler has stored since construction.

        Read at the moment a hardware write is issued, this is the ordinal of
        the last frame that had already arrived -- so a later frame can be
        told apart from one that was already in flight.
        """
        with self._frame_lock:
            return self._frames_delivered

    def _detached(self) -> bool:
        """True when the buffered frame's device is no longer attached.

        Every reader of the buffer consults this before it answers, so a frame
        stored from a device that has since been removed is reported as absent
        rather than as current. A driver whose handler can outlive its device
        overrides this; the base handler is never detached.
        """
        return False

    def get_last_image(self) -> tuple:
        """Return (success, image, timestamp, significant_bits, seq). Thread-safe.

        No copy needed here -- the stored frame is already a copy from the SDK
        callback (GetArray().copy() in Pylon, copy() in IDS). _store_frame()
        replaces the reference (not in-place), so the returned array remains
        valid even after the next frame arrives.

        The depth is returned WITH the frame (the value stamped at store time) so
        a caller scales the buffered frame by the depth it was captured under, not
        a depth queried separately afterward.

        On a stalled stream this keeps returning the last stored frame, so a
        live preview can freeze without an error surfacing here. Capture and
        autofocus paths must not rely on this method for freshness: they go
        through grab_new_capture(), which resets the handler and waits for a
        genuinely new frame, and a summed capture additionally requires each
        frame's arrival ordinal to exceed the one before it.
        """
        if self._detached():
            return self.NO_FRAME
        with self._frame_lock:
            if not self.last_result:
                return self.NO_FRAME
            return (
                True,
                self.last_img,
                self.last_img_ts,
                self.last_img_significant_bits,
                self.last_img_seq,
            )

    def get_last_chunks(self) -> dict | None:
        """Return per-frame chunk metadata for the most recent successful grab.

        Cameras that support GenICam chunk data populate this dict in their
        ImageHandler grab callback. Cameras without chunk support return
        None (default).

        Returned dict keys are GenICam attribute symbolic names
        ('ExposureTime', 'Gain', 'FrameID' on Basler USB3). Values are
        floats / ints as reported by the camera. Returns None when no
        successful grab has occurred yet. Only the Pylon driver currently
        populates chunks; the IDS driver stores frames without them, so
        IDS consumers always see None and fall back to live read-back.
        """
        if self._detached():
            return None
        with self._frame_lock:
            if not self.last_result:
                return None
            return self.last_chunks

    def reset(self) -> None:
        """Clear frame buffer and failure counter.

        Deliberately leaves ``_frames_delivered`` alone. It is an ordinal, not
        buffer state: restarting it would let a frame arriving after a reset
        carry a lower number than one that arrived before, which is exactly
        the backwards step the ordinal exists to rule out. The buffered
        frame's own ordinal clears with the frame it describes.
        """
        with self._frame_lock:
            self.last_result = False
            self.last_img = None
            self.last_img_ts = None
            self.last_img_significant_bits = None
            self.last_chunks = None
            self.last_img_seq = None
        self._failed_grabs = 0

    def register_frame_callback(self, cb: Callable[[Any, Any, Any], None]) -> None:
        """Register a per-frame callback fired after every successful grab.

        Callback signature: ``cb(image, timestamp, chunks)``. Runs on the
        worker thread that processes SDK callbacks (Pylon
        ``PylonImageGrabWorker`` for Stage B of the OnImageGrabbed split
        / IDS grab loop / simulated acquisition thread). The Pylon SDK's native grab
        thread (``PylonImageGrab``) only enqueues to Stage B and does
        not fire callbacks directly. Callbacks MUST NOT block -- they
        share the worker thread with the next frame. Heavy work (file IO,
        image conversion) belongs on an executor; the callback's job is
        fast decision + enqueue.

        Registration is idempotent for the same callable.
        """
        with self._frame_lock:
            if cb not in self._frame_callbacks:
                self._frame_callbacks.append(cb)

    def unregister_frame_callback(self, cb) -> None:
        """Remove a callback registered via ``register_frame_callback``.

        No-op when ``cb`` is not currently registered.
        """
        with self._frame_lock, contextlib.suppress(ValueError):
            self._frame_callbacks.remove(cb)

    def _store_frame(self, image, timestamp, chunks: dict | None = None, *, significant_bits: int):
        """Called by subclass when a new frame is successfully grabbed.

        Args:
            image: numpy array (already copied from SDK buffer).
            timestamp: datetime when the frame arrived host-side.
            chunks: optional per-frame chunk metadata dict. None for cameras
                without chunk support; backward-compatible with callers that
                don't pass it.
            significant_bits: payload depth of THIS frame -- REQUIRED, so a frame
                can never be buffered without the depth needed to interpret it.
                The subclass derives it from the frame itself (the grab result's
                pixel type, or the delivered array's container width for cameras
                that deliver true container-depth frames), so the depth and the
                pixels stay together and a later format switch cannot make the
                buffered frame's depth read wrong.

        The frame is cropped to the camera's window here, the one place every
        driver's frames are stored; a frame of another size than the window
        acquires was made under a window no longer set and is not stored.
        """
        window = self.frame_window
        if window is not None:
            if image.shape[:2] != (window.acquired_height, window.acquired_width):
                self._skip_frame_of_another_window(image, window)
                return
            if window.crop is not None:
                image = center_crop(image, *window.crop)
        _tracing = profile_trace.ENABLE_PROFILE_TRACE
        if _tracing:
            _arrive_t = time.perf_counter()
            _gap_ms = (
                -1.0
                if self._last_arrival_t is None
                else (_arrive_t - self._last_arrival_t) * 1000.0
            )
            self._last_arrival_t = _arrive_t
        with self._frame_lock:
            self.last_result = True
            self.last_img = image
            self.last_img_ts = timestamp
            self.last_img_significant_bits = significant_bits
            self.last_chunks = chunks
            self._frames_delivered += 1
            self.last_img_seq = self._frames_delivered
            cbs = list(self._frame_callbacks)
        self._failed_grabs = 0
        # Snapshot under lock + invoke outside: a callback that takes >0
        # microseconds never extends the SDK thread's lock hold past the
        # storage write. One failing callback can't block its peers.
        _fanout_t = time.perf_counter() if _tracing else None
        for cb in cbs:
            try:
                cb(image, timestamp, chunks)
            except Exception as e:
                _cam_log.exception(f'[CAM Class ] frame callback raised: {e}')
        if _tracing:
            self._arrival_trace.add(
                [
                    f'{time.time() * 1000.0:.3f}',
                    f'{_gap_ms:.3f}',
                    len(cbs),
                    f'{(time.perf_counter() - _fanout_t) * 1000.0:.3f}',
                    significant_bits,
                    getattr(image, 'nbytes', 0),
                ]
            )

    def _skip_frame_of_another_window(self, image, window: FrameWindow) -> None:
        """Count a frame made under another window, said once per window."""
        self._frames_of_another_window += 1
        if self._reported_window is not window:
            self._reported_window = window
            _cam_log.info(
                f'[CAM Class ] a {image.shape[1]}x{image.shape[0]} frame arrived for the '
                f'{window.acquired_width}x{window.acquired_height} window: made before the '
                f'window changed, not stored ({self._frames_of_another_window} so far)'
            )

    def _record_failure(self):
        """Called by subclass when a grab fails.

        Returns True if the failure count has reached MAX_CONSECUTIVE_FAILURES,
        indicating the caller should stop grabbing.
        """
        with self._frame_lock:
            self.last_result = False
        self._failed_grabs += 1
        if self._failed_grabs % 5 == 1:
            _cam_log.warning(f'[CAM Class ] Grab failed ({self._failed_grabs} consecutive)')
        return self._failed_grabs >= self.MAX_CONSECUTIVE_FAILURES


def no_hardware_auto_mode(driver: str, member: str, mode: str) -> NotImplementedError:
    """The error an auto-mode member raises on a camera with no such mode.

    Its profile declares the mode absent and the API reads the profile
    before any auto-mode call, so reaching one of these members is a caller
    that skipped that read. An answer here would be a lie: nothing was
    written, and there is no loop to report on.
    """
    return NotImplementedError(
        f'{driver} {member}: this camera has no hardware {mode}; '
        'the API reads the camera profile and never asks'
    )


class Camera(ABC):
    # Color + native bit depth contract surfaced through scope.capabilities.
    # Drivers override as needed: True for true color cameras (Bayer / 3-channel
    # sensors); 8-bit for sensors that report only 8-bit Mono natively (IDS
    # IMX676 / U3-34L0XCP-M); 16-bit container for sensors that pack Mono10 /
    # Mono12 / Mono16 into uint16 buffers (Pylon family). The container width
    # rather than the wire-level payload bits: downstream allocators size
    # buffers to the container, not the payload.
    is_color_native: bool = False
    native_bit_depth: int = 16

    def __init__(self):
        self._state_lock = threading.Lock()
        self._array_lock = threading.Lock()
        # CAM-3: serializes the entire stop/yield/start critical section
        # of update_camera_config() so two threads can't both be inside it
        # at once. update_camera_config() can yield arbitrarily long
        # configuration work (set_pixel_format, set_frame_size,
        # init_camera_config), so this is a separate lock from
        # _state_lock -- _state_lock holds for ms, _lifecycle_lock can
        # hold for seconds.
        self._lifecycle_lock = threading.RLock()
        self._active = False
        self.array = np.array([])
        self.cam_image_handler: ImageHandlerBase | None = None
        self.model_name = None
        self._device_removed = False
        self._async_teardown_started = False
        self._device_serial = None
        # Camera-side timestamp tick rate (Hz). Set by the driver at init
        # if the camera supports a Timestamp chunk; None for cameras
        # without chunk timestamps (downstream code skips per-frame
        # camera-tick metadata when None).
        self.timestamp_tick_frequency_hz: int | None = None
        self.profile: CameraProfile = CameraProfile()
        # Re-entrancy depth for ``update_camera_config()`` (CAM-4).
        # Protected by ``_state_lock``; only the outermost level
        # toggles the grab loop.
        self._update_config_depth = 0

        # Durable per-frame callback registry, owned by the Camera rather than
        # the ephemeral image handler. The handler's own callback list is a
        # working copy the SDK thread dispatches from; it starts empty on every
        # freshly-built handler (connect / recovery), so a driver that rebuilds
        # its handler would silently drop every listener registered before the
        # rebuild. Recording it here and re-pushing it via
        # _reapply_frame_callbacks() after each handler build keeps manual-record
        # and per-frame plugin listeners alive across a reconnect. Initialized
        # before connect() below, which re-applies it on the first handler.
        self._frame_callback_lock = threading.Lock()
        self._registered_frame_callbacks: list = []
        # The window set_frame_size planned, owned here for the same reason:
        # a driver sets its window before it builds a handler, and rebuilds
        # the handler on a reconnect.
        self._frame_window: FrameWindow | None = None

        # Start gate: the camera-lifecycle split. connect() returns the
        # camera CONFIGURED but NOT grabbing; streaming begins exactly once
        # via open_and_start() (the configure-complete -> start transition).
        # The latch is per-INSTANCE (set here, never a class attribute, so a
        # reconnect's fresh camera always starts CLOSED) and is read/written
        # under _lifecycle_lock so gate checks stay coherent with the grab
        # loop's stop/start. CLOSED at construction; OPEN after release.
        self._grab_gate_open = False

        self.connect()
        # `found` is a derived property (below) that reads `active`, so it
        # reflects connect()'s outcome here AND stays correct across a later
        # disconnect / same-instance reconnect -- no stale one-time snapshot to
        # refresh (it used to be assigned once here and never recomputed).

    @property
    def active(self) -> Any:
        """Thread-safe access to camera active state.

        Three-state semantics:
          False  -- not connected (initial state)
          <obj>  -- connected camera instance (truthy; e.g. pylon.InstantCamera)
          None   -- disconnected or a connect failed (after a removal this
                    lags ``is_device_removed()``, which _mark_disconnected sets;
                    the handle is released later, by disconnect())

        Returns:
            False, the connected camera instance, or None.
        """
        with self._state_lock:
            return self._active

    @active.setter
    def active(self, value) -> None:
        """Set the active-state value under the state lock."""
        with self._state_lock:
            self._active = value

    @property
    def found(self) -> bool:
        """Whether the driver found its hardware -- derived from `active`.

        Registry contract: drivers signal "I couldn't find my hardware" via
        `found=False`, and `drivers/registry.py::create('auto')` skips such
        instances and tries the next candidate (PylonCamera / IDSCamera catch
        their connect-failure internally and set `active = None` without raising,
        so without this the registry would return the broken instance and FX2
        never gets a turn -- the LS620 first-bring-up failure). Deriving it from
        `active`'s three-state semantics (False=initial, <obj>=connected,
        None=disconnected) keeps it current after a disconnect / reconnect
        instead of the old once-in-__init__ snapshot that went stale.
        """
        return self._active not in (False, None)

    def _reset_lifecycle_state(self) -> None:
        """Return per-instance lifecycle state to its just-constructed baseline.

        Called by each driver's disconnect() so a reconnect that REUSES the same
        instance starts clean. Resets only the genuine mutable state that would
        otherwise persist: the start gate (else open_and_start() sees it already
        OPEN and never restarts grabbing) and the last-frame buffer (else
        get_array() returns the pre-disconnect image until the first new grab).
        `found` needs no reset -- it is a property derived from `active`, which
        the driver has already nulled by disconnect time. Each field is written
        under its own documented lock (the gate under _lifecycle_lock, coherent
        with open_and_start's stop/start; the buffer under _array_lock). Callers
        must NOT hold a lock that either of these is ever acquired-after, to keep
        the acquisition order consistent.
        """
        with self._lifecycle_lock:
            self._grab_gate_open = False
        with self._array_lock:
            self.array = np.array([])

    def __del__(self):
        # Subclass __init__ may raise before super().__init__() runs, leaving
        # a partially constructed instance whose _state_lock + _active never
        # got set. Python
        # still runs __del__ on the partial object; the hasattr gate
        # short-circuits to a clean no-op instead of firing a misleading
        # "__del__ disconnect failed: no attribute _state_lock" warning.
        if not hasattr(self, '_state_lock'):
            return
        try:
            with self._state_lock:
                is_active = bool(self._active)
            if is_active:
                self.disconnect()
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] __del__ disconnect failed: {e}')

    def is_device_removed(self) -> bool:
        """Return whether the camera was marked as physically removed.

        Returns:
            bool: True after ``_mark_disconnected`` has been invoked.
        """
        with self._state_lock:
            return self._device_removed

    def _mark_disconnected(self):
        """Atomically mark camera as disconnected.

        Safe to call from any thread (including SDK callbacks). Sets
        the device-removed flag synchronously so subsequent state
        queries early-return; the actual Python-side release of the
        SDK handle (``self._active = None``) is deliberately deferred
        to ``disconnect()`` on the async-teardown daemon thread.

        Dropping ``self._active`` here would fire the C++ device
        wrapper's destructor synchronously on whichever thread called
        us. When that thread is the SDK callback thread (the inline
        disconnect fast-path from OnImageGrabbed), the destructor's
        SDK teardown calls race the SDK's concurrent in-flight grab
        work and trigger a native abort. The disconnect() path on the
        async-teardown daemon thread does the same SDK teardown in a
        safe Python-owned context after StopGrabbing has drained the
        in-flight work.
        """
        was_connected = False
        with self._state_lock:
            was_connected = self._active is not None and not self._device_removed
            self._device_removed = True
        if was_connected:
            _cam_log.error('[CAM Class ] Camera disconnected')

    def _schedule_async_teardown(self) -> None:
        """Run disconnect() on a daemon thread of its own, once per removal.

        Called with ``_mark_disconnected`` by whatever noticed the removal: an
        SDK callback, a presence probe, a grab loop. None of those threads may
        tear the camera down itself -- an SDK callback that closes its own
        device deadlocks or aborts natively, and a grab loop cannot join
        itself -- so the teardown runs here, after a short delay that lets
        the caller return first.

        One-shot while a teardown is in flight: a second trigger (a callback
        racing a probe) is a no-op. The latch re-arms when the teardown ends,
        so a removal after a later reconnect is torn down too; a trigger
        arriving after the teardown still finds the camera marked removed.
        """
        with self._state_lock:
            if self._async_teardown_started:
                return
            self._async_teardown_started = True

        def _run_teardown():
            try:
                time.sleep(0.05)
                _cam_log.info('[CAM Class ] removal teardown: calling disconnect()')
                self.disconnect()
            except BaseException as e:
                # Nothing above this daemon thread can act on the failure, so
                # it is reported here, where an operator reading the log sees
                # that the camera was not released cleanly.
                _cam_log.warning(f'[CAM Class ] removal teardown failed: {type(e).__name__}: {e}')
            finally:
                with self._state_lock:
                    self._async_teardown_started = False

        threading.Thread(
            target=_run_teardown, name=f'{type(self).__name__}RemovalTeardown', daemon=True
        ).start()

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the camera hardware.

        Subclasses must catch their own connect-failure exceptions and
        leave ``self._active`` as False or None on failure (the registry
        uses ``self.found`` to skip non-functional drivers).

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the camera hardware and release resources.

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the camera is currently connected.

        Returns:
            bool: True if the SDK reports a live, attached camera.
        """
        pass

    @contextlib.contextmanager
    def update_camera_config(self):
        """Cross-thread-safe, re-entrant guard around the camera grab loop.

        Combines the CAM-3 (cross-thread serialization) and CAM-4
        (nested-from-same-thread) fixes into a single mechanism:

          - ``_lifecycle_lock`` (RLock) holds for the full
            stop/yield/start critical section so two threads can't both
            be mutating the grab loop simultaneously, and the same
            thread can re-enter without deadlock.
          - ``_update_config_depth`` counts the nesting level so only
            the OUTERMOST call toggles the grab loop. Inner re-entries
            are no-ops on the SDK side. Counter mutates only while we
            hold the RLock, so no separate lock is required.
          - ``camera.log`` :enter / :exit lines emit on every level so
            nested patterns (init_camera_config wrapping
            set_pixel_format) stay visible in diagnostic logs.
        """
        try:
            from lvp_logger import camera_logger as _cam_log
        except Exception:
            _cam_log = None
        with self._lifecycle_lock:
            self._update_config_depth += 1
            depth = self._update_config_depth
            was_grabbing = False
            if _cam_log is not None:
                _cam_log.info(f'update_camera_config:enter depth={depth}')
            try:
                if depth == 1:
                    was_grabbing = self.is_grabbing()
                    if was_grabbing:
                        self.stop_grabbing()
                yield
            finally:
                self._update_config_depth -= 1
                end_depth = self._update_config_depth
                if end_depth == 0 and was_grabbing:
                    self.start_grabbing()
                if _cam_log is not None:
                    _cam_log.info(
                        f'update_camera_config:exit depth={depth} '
                        f'restarted={was_grabbing and end_depth == 0}'
                    )

    def open_and_start(self) -> bool:
        """Release the start gate: begin streaming exactly once.

        The single configure-complete -> start transition. ``connect()``
        leaves the camera configured but NOT grabbing (gate CLOSED); this
        opens the gate and fires the one ``start_grabbing()``.

        Flag-idempotent: a no-op when the gate is already OPEN, so the two
        bring-up release sites (startup + reconnect, which fire ~0.3s
        apart) cannot double-start -- this does NOT rely on
        ``start_grabbing`` idempotency. Restarting an already-released
        camera after a deliberate stop is the primitive ``start_grabbing``
        path, not this one.

        The gate is opened BEFORE the start, so even if the start fails the
        camera is RELEASED -- a later restart can recover it instead of the
        gate stranding CLOSED (a permanently blank live view). The start
        itself is not wrapped here: every ``start_grabbing()`` is already
        exception-tolerant by contract (SDK failures are logged, not
        raised), so callers in a ``finally`` need no guard.

        Returns:
            bool: True when this call released the gate and fired the
                start; False when the gate was already open (no-op). The
                return lets a caller distinguish "start just attempted"
                from "already released" without a second SDK poll, so an
                ensure-running wrapper does not immediately re-start a
                device whose start just failed.
        """
        with self._lifecycle_lock:
            if self._grab_gate_open:
                return False
            self._grab_gate_open = True
            self.start_grabbing()
            return True

    @abstractmethod
    def init_camera_config(self) -> None:
        """Apply the camera's startup configuration after connect().

        Subclasses set pixel format, default frame size, exposure, gain,
        binning, etc. Called inside ``update_camera_config()`` so the
        grab loop is paused while configuration is in progress.
        """
        pass

    @abstractmethod
    def start_grabbing(self) -> None:
        """Begin acquiring frames into the image handler."""
        pass

    @abstractmethod
    def stop_grabbing(self) -> None:
        """Stop acquiring frames and release any pending buffers."""
        pass

    @abstractmethod
    def is_grabbing(self) -> bool:
        """Return whether the camera is currently acquiring.

        Returns:
            bool: True when the SDK reports an active grab loop.
        """
        pass

    def set_frame_size(self, w: int, h: int) -> dict | bool:
        """Deliver frames of exactly ``w`` x ``h``: acquire the next window up and crop back.

        A camera sets its window on a grid, so a size off it cannot be
        acquired. The window planned is the next legal one up, centred, and
        every stored frame is cropped back to the request (``_store_frame``).
        The window is recorded before the hardware is written, so a frame
        made under the old window after the write is not stored at the old
        size. Only a request within one grid step of the camera's maximum
        comes back smaller: no legal window holds it.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            The delivered size ``{'width': int, 'height': int}`` on success, so
            a caller records it without a read-back; ``False`` when the camera
            is inactive or the hardware refuses the window, in which case
            frames are stored as they arrive until a window is set.
        """
        grid = self._frame_grid()
        if grid is None:
            _cam_log.warning(f'[CAM Class ] Cannot set frame size {w}x{h}: camera inactive')
            return False
        target = (int(w), int(h))
        plan = plan_aoi(
            target=target,
            step=grid.step,
            max_size=grid.max_size,
            offset_step=grid.offset_step,
            size_min=grid.size_min,
            offset_min=grid.offset_min,
            bias=grid.bias,
        )
        window = FrameWindow.from_plan(plan)
        self._set_frame_window(window)
        if not self._set_hardware_window(plan):
            self._set_frame_window(None)
            return False
        if (plan.crop_width, plan.crop_height) != target:
            _cam_log.warning(
                f'[CAM Class ] set_frame_size delivers {plan.crop_width}x{plan.crop_height}, '
                f'smaller than the {target[0]}x{target[1]} asked for: no window on the '
                f'camera grid holds it'
            )
        _cam_log.info(
            f'[CAM Class ] set_frame_size {target[0]}x{target[1]}: acquires '
            f'{plan.acq_width}x{plan.acq_height}, crop {window.crop}'
        )
        return window.size

    def _set_frame_window(self, window: FrameWindow | None) -> None:
        """Record the window and hand it to the handler that stores frames."""
        self._frame_window = window
        handler = self.cam_image_handler
        if handler is not None:
            handler.frame_window = window

    @abstractmethod
    def _frame_grid(self) -> FrameGrid | None:
        """The windows this camera can acquire now; None when it is inactive."""

    @abstractmethod
    def _set_hardware_window(self, plan: AoiPlan) -> bool:
        """Acquire ``plan.acq_width`` x ``plan.acq_height``, centred.

        Returns:
            True once the camera acquires the window; False when it refused
            or the write failed.
        """

    @abstractmethod
    def get_min_frame_size(self) -> dict:
        """Return the minimum supported frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        pass

    @abstractmethod
    def get_max_frame_size(self) -> dict:
        """Return the maximum supported frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        pass

    def get_frame_size(self) -> dict | None:
        """Return the size of the frames the camera delivers.

        Returns:
            dict: ``{'width': int, 'height': int}``: the window's delivered
                size, or the camera's own window while none is set. What the
                driver's own read returns when the camera cannot answer.
        """
        acquired = self._hardware_frame_size()
        window = self._frame_window
        if not acquired or window is None:
            return acquired
        return window.size

    @abstractmethod
    def _hardware_frame_size(self) -> dict | None:
        """The window the camera acquires, ``{'width', 'height'}``."""

    @abstractmethod
    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Format identifier (e.g. ``'Mono8'``,
                ``'Mono12'``).

        Returns:
            bool: True on success.
        """
        pass

    @abstractmethod
    def get_pixel_format(self) -> str | None:
        """Return the current camera pixel format.

        Returns:
            The format identifier (e.g. ``'Mono8'``), or ``None`` when the
            camera is inactive or the read failed. ``None`` is the shared
            failed-read sentinel -- distinct from every real format name,
            so a consumer that forgets to handle it fails loudly instead
            of silently treating the failure as a legal format.
        """
        pass

    @abstractmethod
    def get_supported_pixel_formats(self) -> tuple:
        """Return all pixel formats this camera can be set to.

        Returns:
            tuple: Format identifier strings supported by the SDK.
        """
        pass

    @staticmethod
    def format_significant_bits(pixel_format: str | None, fallback: int) -> int:
        """Payload bit count named by a Mono-style GenICam format string.

        ``Mono12`` -> 12, ``Mono10`` -> 10, ``Mono8`` -> 8 (the FIRST digit
        run in the format name). Returns ``fallback`` when the format is
        None (failed read / inactive) or carries no digits. Scope: the
        base-class depth rule for the shipping mono fleet -- drivers whose
        format vocabulary breaks the first-digits-are-depth assumption
        (IDS packed wire names like Mono12g24IDS, any future color
        vocabulary like YCbCr422_8) keep their own parsers and override
        ``significant_bits_for_format`` instead of feeding names here.
        """
        match = re.search(r'(\d+)', pixel_format or '')
        return int(match.group(1)) if match else fallback

    def significant_bits_for_format(self, pixel_format: str | None) -> int:
        """Payload depth this driver DELIVERS for a given pixel-format name.

        The override hook for the depth rule: the base implementation
        derives from the format name; drivers whose delivered depth does
        not follow the format string (FX2 delivers Mono8 regardless; a
        converting driver could deliver a fixed depth) override this so
        every depth consumer -- including the API layer, which calls this
        with its validated last-known-good format so a transient format
        read cannot change the answer -- honors the driver's word.
        """
        return self.format_significant_bits(pixel_format, self.native_bit_depth)

    def last_stamped_significant_bits(self) -> int | None:
        """Per-frame delivery stamp of the most recently buffered frame.

        Returns None when no frame has been stored (or the handler recorded
        no stamp) -- deliberately NO fallback, so callers choose their own
        no-frame depth source: the driver property below falls back to its
        live format read; the API layer falls back to its validated
        last-known-good format instead, keeping a transient format-read
        failure from turning into a wrong depth.

        Read through the handler's get_last_image() method, not the raw
        last_img_significant_bits attribute: the tuple is read atomically
        under the handler's frame lock, so the stamp cannot describe a
        different frame than the one returned beside it.
        """
        handler = self.cam_image_handler
        if handler is not None:
            success, _image, _ts, significant_bits, _seq = handler.get_last_image()
            if success and significant_bits is not None:
                return significant_bits
        return None

    @property
    def significant_bits(self) -> int:
        """Meaningful low bits of a delivered frame (payload, not container).

        Derived from the active pixel format: ``Mono12`` -> 12, ``Mono10`` ->
        10, ``Mono8`` -> 8 (the leading bit-count in the GenICam format name).
        Right-aligned, so a value of ``(1 << significant_bits) - 1`` is full
        scale. Distinct from ``native_bit_depth`` (the container width): a
        Mono12 frame is significant_bits 12 in a 16-wide container. A summed
        capture is promoted to a 16-bit container by the imaging layer and is
        not described by this field. Drivers that deliver a fixed converted
        depth regardless of the sensor's format -- IDS converts to Mono8 at the
        SDK boundary, FX2 is Mono8-only -- override with a constant. Falls back
        to the container width when the format name carries no bit count.
        """
        return self.significant_bits_for_format(self.get_pixel_format())

    @property
    def last_significant_bits(self):
        """Payload depth of the most recently buffered frame (stamped at store).

        The grab() + get_array() capture path (unlike grab_latest) hands back a
        bare array, so this exposes the depth recorded WITH that frame. A caller
        downconverting the just-grabbed frame uses this rather than the live
        ``significant_bits``, which can already reflect a newer pixel format than
        the buffered frame was captured under. Falls back to the live depth when
        no frame has been stored yet.

        Reads the stamp via ``last_stamped_significant_bits`` (see its
        docstring for the handler-method contract).
        """
        stamped = self.last_stamped_significant_bits()
        return stamped if stamped is not None else self.significant_bits

    @abstractmethod
    def exposure_t(self, exposure_ms: float) -> float | bool | None:
        """Set exposure time and report the value actually in effect.

        A driver may not be able to honor the request exactly: the node
        has a minimum it clamps up to, an increment it snaps to, or a
        row-time grid it quantizes onto. The caller records a chunk-match
        target from the return, so a driver that reports the request
        instead of what it applied makes every subsequent frame fail the
        match and be rejected forever.

        Args:
            exposure_ms: Requested exposure time in milliseconds.

        Returns:
            float: Microseconds now in effect -- what the hardware will
                stamp into frame chunk data. Returned on the no-write
                path too (a short-circuited write still leaves that
                value in effect).
            False: The write was refused and the hardware did NOT move.
                The caller must not record a target for a value the
                camera never took.
            None: Applied, but the effective value is unknown -- drivers
                that cannot report one. The caller falls back to the
                request.
        """
        pass

    @abstractmethod
    def get_exposure_t(self) -> float:
        """Return the current exposure time.

        Returns:
            float: Exposure time in milliseconds.
        """
        pass

    @abstractmethod
    def auto_exposure_t(self, state: bool = True) -> bool | None:
        """Enable or disable hardware auto-exposure.

        Args:
            state: True to enable, False to disable.

        Returns:
            bool | None: Applied, refused or not attempted, as
                ``Camera.gain`` defines them.
        """
        pass

    def get_model_name(self) -> str | None:
        """Return the cached camera model name.

        Returns:
            str | None: Cached model identifier, or None if not yet
                discovered.
        """
        return self.model_name

    @abstractmethod
    def get_all_temperatures(self) -> dict:
        """Read all available camera temperature sensors.

        Returns:
            dict: Sensor-name-keyed temperatures in degrees Celsius.
                Empty dict when the camera does not expose temperature
                telemetry.
        """
        pass

    def get_sdk_info(self) -> dict:
        """Return the camera SDK provenance label for diagnostic snapshots.

        Driver-neutral so the diagnostic collector can stamp whichever SDK
        actually produced a snapshot instead of assuming Pylon. The base
        returns an unknown SDK; SDK-backed drivers override with the real
        name + version.

        Returns:
            dict: ``{'name': <sdk name or None>, 'version': <str or None>}``.
        """
        return {'name': None, 'version': None}

    def _load_profile(self):
        """Load the camera profile based on model_name.

        Called by subclass connect() after model_name is known. Subclasses
        should then call _query_dynamic_capabilities() to populate
        SDK-queried fields (gain min/max, exposure min/max).
        `profile.exposure_max_us` is the single source of truth for the
        max-exposure cap -- `Camera.max_exposure` is a derived property
        that reads from it.
        """
        self.profile = lookup_profile(self.model_name)
        logger.info(
            f'[CAM Class ] Loaded profile: {self.profile.model_name} '
            f'(sensor={self.profile.sensor}, driver={self.profile.driver})'
        )

    def _query_dynamic_capabilities(self):  # noqa: B027 -- optional no-op hook; subclasses override only if needed, abstractmethod would force needless overrides
        """Query SDK for dynamic values and merge into profile.

        Subclasses should override to query gain min/max, exposure min/max,
        etc. from the camera SDK. The base implementation is a no-op.
        """
        pass

    @property
    def max_exposure(self) -> float:
        """Maximum exposure cap in milliseconds.

        Derived from `profile.exposure_max_us` -- the single source of
        truth. The profile's value is the sensor-datasheet ceiling by
        default and may be overwritten by `_query_dynamic_capabilities()`
        at connect time with an SDK-queried or driver-narrowed cap
        (e.g. FX2's 1000 ms ceiling).
        """
        if self.profile and self.profile.exposure_max_us:
            return self.profile.exposure_max_us / 1000.0
        return float(default_max_exposure)

    def get_max_exposure(self) -> float:
        """Return the maximum exposure cap in milliseconds.

        Returns:
            float: Same value as the ``max_exposure`` property.
        """
        return self.max_exposure

    @property
    def min_exposure(self) -> float | None:
        """Minimum exposure floor in milliseconds, or None if undeclared.

        Mirror of `max_exposure`, derived from `profile.exposure_min_us` --
        an optional profile field, so returns None when the profile carries
        no floor and the caller should fall back. Drivers whose SDK exposes a
        LIVE node minimum override `get_min_exposure` to report it: the node
        floor can drift above the connect-time value once other settings
        change, so a cached value goes stale.
        """
        if self.profile and self.profile.exposure_min_us:
            return self.profile.exposure_min_us / 1000.0
        return None

    def get_min_exposure(self) -> float | None:
        """Return the minimum exposure floor in milliseconds.

        Returns:
            float | None: Same value as the ``min_exposure`` property
            (None when the profile declares no floor).
        """
        return self.min_exposure

    @property
    def max_gain(self) -> float:
        """Maximum gain cap in dB.

        Derived from `profile.gain.total_max_db` -- the single source of
        truth. The profile's value is the sensor-datasheet ceiling by
        default and may be overwritten by `_query_dynamic_capabilities()`
        at connect time (Pylon / IDS live-query their SDK; FX2 hardcodes
        the MT9P031 value).
        """
        if self.profile and self.profile.gain and self.profile.gain.total_max_db is not None:
            return float(self.profile.gain.total_max_db)
        return 48.0  # legacy kv default -- kept for cameras without a profile

    @property
    def min_gain(self) -> float | None:
        """Minimum gain in dB, or None if the profile declares none.

        Derived from `profile.gain.total_min_db`, which the drivers fill at
        connect from the SDK's own range (pylon, IDS) or the sensor datasheet
        (FX2). Unlike `max_gain` there is no fallback: a missing floor is
        not a floor of zero, and a caller checking a request against one
        must be able to tell that it has no floor to check.
        """
        if self.profile and self.profile.gain and self.profile.gain.total_min_db is not None:
            return float(self.profile.gain.total_min_db)
        return None

    def get_max_gain(self) -> float:
        """Return the maximum gain cap in dB.

        Returns:
            float: Same value as the ``max_gain`` property.
        """
        return self.max_gain

    def get_resulting_frame_rate(self) -> float | None:
        """Read the frame rate the camera reports its current settings
        allow, live, in frames per second.

        The camera's own figure, not a measurement: the delivered rate is
        measured from frame arrivals. The default describes a camera that
        reports none.

        Returns:
            float | None: Frames per second; None when the camera reports
                none.

        Raises:
            HardwareError: The camera reports one and the read failed.
        """
        return None

    @abstractmethod
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0) -> None:
        """Enable or disable the SDK's frame-rate cap.

        Args:
            enabled: True to enforce ``fps`` as the upper bound.
            fps: Cap value in frames per second.
        """
        pass

    def set_binning_size(self, size: int) -> bool:
        """Set hardware binning factor.

        A change of binning resizes the frames the camera makes, so the
        window set before it no longer fits them: frames are stored as they
        arrive until a frame size is set at the new binning.

        Args:
            size: Binning factor (1, 2, 4, ...).

        Returns:
            bool: True on success.
        """
        before = self.get_binning_size()
        applied = self._set_hardware_binning(size)
        if applied and size != before:
            self._set_frame_window(None)
        return applied

    @abstractmethod
    def _set_hardware_binning(self, size: int) -> bool:
        """Write the binning factor to the camera; True on success."""

    @abstractmethod
    def get_binning_size(self) -> int:
        """Return the current hardware binning factor.

        Returns:
            int: Binning factor (1 = no binning); 1 when the camera is
                inactive (no camera means no binning); -1 on a read
                failure. -1 (not 1) is the failure sentinel because 1 is
                a legal factor -- an in-band failure value would let a
                value-validating caller silently de-bin a 2x camera.
        """
        pass

    @property
    def frames_delivered(self) -> int:
        """Frames this camera has delivered since the handler was built.

        Read when a hardware register is written, this names the last frame
        that had already arrived, so a frame handed to frame validity later
        can be told apart from one that was already in flight when the write
        went out. Returns 0 before a handler exists -- no frame has arrived,
        which is what the number means.
        """
        handler = self.cam_image_handler
        return handler.frames_delivered if handler is not None else 0

    def grab(self) -> tuple:
        """Grab the most recent frame from the image handler.

        On success, the image is also stored in ``self.array``.

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None,
                seq: int | None)``. The arrival ordinal travels WITH the
                frame: read separately afterwards it could describe a newer
                frame than the one returned here, and frame validity uses it
                to decide whether this frame predates a hardware write.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None, None

        if not self.cam_image_handler:
            return False, None, None

        try:
            result, image, image_ts, _significant_bits, image_seq = (
                self.cam_image_handler.get_last_image()
            )
            if not result:
                return False, None, None

            with self._array_lock:
                self.array = image
            return True, image_ts, image_seq
        except Exception as ex:
            _cam_log.exception(f'[CAM Class ] grab() - get_last_image() failed: {ex}')
            return False, None, None

    def get_array(self) -> np.ndarray:
        """Return a copy of the last grabbed image. Thread-safe.

        Returns:
            np.ndarray: Copy of the most recent frame, or an empty
                array when no frame has been grabbed yet.
        """
        with self._array_lock:
            return self.array.copy() if self.array.size > 0 else self.array

    def grab_latest(self) -> tuple:
        """Grab the latest frame and return it in one operation (single copy).

        Combines grab() + get_array() but avoids the second copy.
        The returned image is already a copy from the image handler,
        safe to use without further copying.

        Returns:
            tuple: ``(success: bool, image: np.ndarray | None,
                timestamp: datetime | None, significant_bits: int | None,
                seq: int | None)``.
                The depth is carried with the frame so the caller scales it by
                the depth it was captured under, not a separately-queried one.
                The arrival ordinal rides along for the same reason: the live
                preview counts these frames toward settle counts, and a frame
                read back separately could be a newer one.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None, None, None, None

        if not self.cam_image_handler:
            return False, None, None, None, None

        try:
            result, image, image_ts, image_significant_bits, image_seq = (
                self.cam_image_handler.get_last_image()
            )
            if not result or image is None:
                return False, None, None, None, None

            # self.array feeds get_array(); nothing else reads it. Stored
            # and returned arrays are the SAME object (copied once at the
            # SDK callback; reference replaced per frame) -- safe against
            # overwrite by the next frame, but not a private copy: caller
            # in-place mutation would leak into get_array() of this frame.
            with self._array_lock:
                self.array = image
            return True, image, image_ts, image_significant_bits, image_seq
        except Exception as ex:
            _cam_log.exception(f'[CAM Class ] grab_latest() failed: {ex}')
            return False, None, None, None, None

    def register_frame_callback(self, cb: Callable[[Any, Any, Any], None]) -> None:
        """Register a per-frame callback.

        Records the callback in the Camera's durable registry (so it survives a
        handler rebuild) AND applies it to the current handler for immediate
        dispatch. Idempotent for the same callable.
        """
        with self._frame_callback_lock:
            if cb not in self._registered_frame_callbacks:
                self._registered_frame_callbacks.append(cb)
        # Apply to the live handler OUTSIDE _frame_callback_lock: the handler
        # takes its own _frame_lock, so nesting the two would couple the locks.
        if self.cam_image_handler:
            self.cam_image_handler.register_frame_callback(cb)

    def unregister_frame_callback(self, cb) -> None:
        """Unregister a callback from the durable registry and the current handler."""
        with self._frame_callback_lock, contextlib.suppress(ValueError):
            self._registered_frame_callbacks.remove(cb)
        if self.cam_image_handler:
            self.cam_image_handler.unregister_frame_callback(cb)

    def _reapply_frame_callbacks(self) -> None:
        """Re-register the durable callback set and the window onto the current handler.

        A driver calls this immediately after building a new cam_image_handler
        (connect / recovery). The handler owns the dispatch list and starts
        empty, so without this every listener registered before the rebuild
        stops receiving frames, and frames go uncropped. No-op when the driver
        has no handler yet.
        """
        handler = self.cam_image_handler
        if handler is None:
            return
        handler.frame_window = self._frame_window
        # Hold the registry lock ACROSS the re-push, not just the snapshot: an
        # unregister interleaving here (e.g. a per-frame plugin auto-dropped on
        # the SDK callback thread mid-reconnect) must not lose to a stale
        # snapshot and get resurrected onto the fresh handler. Deadlock-safe --
        # the lock order is always _frame_callback_lock -> handler._frame_lock
        # (here and in register/unregister); frame dispatch runs OUTSIDE
        # _frame_lock, so nothing acquires the two in the reverse order.
        with self._frame_callback_lock:
            for cb in self._registered_frame_callbacks:
                handler.register_frame_callback(cb)

    @abstractmethod
    def grab_new_capture(self, timeout_s: float) -> tuple:
        """Grab a fresh capture-quality frame, waiting if necessary.

        Used by the still-capture path to guarantee the returned frame
        was acquired after the call (not a stale live-preview frame).

        Args:
            timeout_s: Maximum wait in seconds.

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None,
                seq: int | None)``. The frame itself is read back with
                ``get_array()``; the arrival ordinal travels with the
                grab for the same reason as in ``grab()``.
        """
        pass

    @abstractmethod
    def update_auto_gain_target_brightness(self, auto_target_brightness: float) -> bool | None:
        """Update the target brightness for the auto-gain loop.

        Args:
            auto_target_brightness: Normalized target brightness (0.0
                to 1.0).

        Returns:
            bool | None: Applied, refused or not attempted, as
                ``Camera.gain`` defines them.
        """
        pass

    @abstractmethod
    def update_auto_gain_min_max(
        self, min_gain_db: float | None, max_gain_db: float | None
    ) -> None:
        """Update the auto-gain bounds.

        Args:
            min_gain_db: Minimum gain in dB, or None to leave unchanged.
            max_gain_db: Maximum gain in dB, or None to leave unchanged.
        """
        pass

    @abstractmethod
    def get_gain(self) -> float:
        """Return the current camera gain.

        Returns:
            float: Gain in dB.
        """
        pass

    @abstractmethod
    def gain(self, value: float) -> float | bool | None:
        """Set the camera gain, and report the gain now in effect.

        The caller records the answer in its camera cache, as the
        frame-validity chunk target and as the value its listeners hear. A
        driver may not be able to honour the request exactly -- a node that
        clamps to its range, a register that quantizes -- so a driver that
        answered only "applied" would leave all three naming a gain the
        sensor is not at, and on a camera that reports gain in chunk data
        every subsequent frame would then fail the match. So a driver that
        can tell a refusal from a success says which, and when it applied
        the value it says what it applied.

        This is also the canonical statement of the three-case return the
        bool-reporting setters share (the auto-mode setters point here):
        ``True`` applied, ``False`` refused, ``None`` not attempted. ``gain``
        and ``exposure_t`` answer with the value in effect in place of
        ``True``.

        Args:
            value: Gain in dB.

        Returns:
            float: Applied -- the gain in dB the hardware now holds,
                including when the write was skipped because it already
                held it.
            True: Applied, but the value in effect could not be read back
                (the write succeeded and the confirming read failed). The
                caller keeps the request as its best knowledge.
            False: Refused -- the hardware did NOT move. The caller must
                not record the request as truth.
            None: Not attempted, because no camera is active. Not a
                refusal: nothing was asked of any hardware.
        """
        pass

    # Black level is the camera's own offset parameter, in the camera's own
    # units (Basler's step in DN differs by model and sensor bit depth; IDS
    # states DN of the current format), never converted to output DN here:
    # no probe supplies the factor. The defaults describe a camera that
    # neither reports nor offers one; each driver overrides what it has.

    def supports_black_level(self) -> bool:
        """Whether the black level can be set on this camera.

        Raises:
            HardwareError: The probe failed.
        """
        return False

    def get_black_level(self) -> float | None:
        """Read the black level in effect, live.

        Returns:
            float | None: The camera's black level parameter; None when the
                camera does not report one.

        Raises:
            HardwareError: The camera reports one and the read failed.
        """
        return None

    def get_black_level_range(self) -> tuple[float, float] | None:
        """Read the settable black level range, live: it can change with the
        pixel format.

        Returns:
            tuple[float, float] | None: ``(minimum, maximum)``; None when the
                black level cannot be set.

        Raises:
            HardwareError: The read failed.
        """
        return None

    def set_black_level(self, value: float) -> float | bool | None:
        """Set the black level, and report the value now in effect.

        Called only when ``supports_black_level`` is True and ``value`` is
        inside ``get_black_level_range``. Returns as ``gain`` does: ``False``
        is the camera's refusal, or the camera lost.

        Raises:
            HardwareError: The write or its read-back failed.
        """
        raise NotImplementedError(f'{type(self).__name__} offers no black level setting')

    @abstractmethod
    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> bool | None:
        """Enable or disable continuous auto-gain.

        Args:
            state: True to enable, False to disable.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
            ae_max_exposure_ms: Optional per-channel-class upper bound (ms)
                on the exposure auto-exposure may drive to. Honored where
                the driver supports auto-exposure bounds; ignored otherwise.

        Returns:
            bool | None: Applied, refused or not attempted, as
                ``Camera.gain`` defines them.
        """
        pass

    @abstractmethod
    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> bool | None:
        """Run a single auto-gain iteration.

        Args:
            state: True to run, False to no-op.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
            ae_max_exposure_ms: Optional per-channel-class exposure upper
                bound (ms); honored where the driver supports it.

        Returns:
            bool | None: Applied, refused or not attempted, as
                ``Camera.gain`` defines them.
        """
        pass

    @abstractmethod
    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black') -> None:
        """Enable or disable the SDK's test pattern generator.

        Args:
            enabled: True to enable the pattern, False to disable.
            pattern: Pattern name (SDK-specific; e.g. ``'Black'``).
        """
        pass
