# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
from abc import ABC, abstractmethod
import contextlib
import re
import threading

import numpy as np

from lvp_logger import logger

try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    # Fall back to the general logger, never None: the _cam_log call sites below
    # are unguarded, so a None fallback turns every one into an AttributeError the
    # moment the dedicated camera logger is unavailable.
    _cam_log = logger
from drivers.camera_profiles import CameraProfile, lookup_profile

default_max_exposure = 1_000  # in ms


class ImageHandlerBase:
    """Base class for camera image handlers (IDS and Pylon).

    Provides thread-safe frame buffer storage, copy-on-read, failure counting
    with auto-stop, and a consistent API for the Camera.grab() method.
    """

    MAX_CONSECUTIVE_FAILURES = 128

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
        self._failed_grabs = 0
        # Per-frame consumers (manual record today; per-frame plugins later).
        # Snapshotted-then-released under _frame_lock at _store_frame time so
        # a slow callback never holds the SDK thread.
        self._frame_callbacks: list = []

    def get_last_image(self):
        """Return (success, image, timestamp, significant_bits). Thread-safe.

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
        genuinely new frame, backed by the imaging layer's timestamp gate.
        """
        with self._frame_lock:
            if not self.last_result:
                return False, None, None, None
            return True, self.last_img, self.last_img_ts, self.last_img_significant_bits

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
        with self._frame_lock:
            if not self.last_result:
                return None
            return self.last_chunks

    def reset(self):
        """Clear frame buffer and failure counter."""
        with self._frame_lock:
            self.last_result = False
            self.last_img = None
            self.last_img_ts = None
            self.last_img_significant_bits = None
            self.last_chunks = None
        self._failed_grabs = 0

    def register_frame_callback(self, cb) -> None:
        """Register a per-frame callback fired after every successful grab.

        Callback signature: ``cb(image, timestamp, chunks)``. Runs on the
        worker thread that processes SDK callbacks (Pylon
        ``PylonImageGrabWorker`` for Stage B of the OnImageGrabbed split
        / IDS grab loop / simulated pump). The Pylon SDK's native grab
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
        """
        with self._frame_lock:
            self.last_result = True
            self.last_img = image
            self.last_img_ts = timestamp
            self.last_img_significant_bits = significant_bits
            self.last_chunks = chunks
            cbs = list(self._frame_callbacks)
        self._failed_grabs = 0
        # Snapshot under lock + invoke outside: a callback that takes >0
        # microseconds never extends the SDK thread's lock hold past the
        # storage write. One failing callback can't block its peers.
        for cb in cbs:
            try:
                cb(image, timestamp, chunks)
            except Exception as e:
                _cam_log.exception(f'[CAM Class ] frame callback raised: {e}')

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
    def active(self):
        """Thread-safe access to camera active state.

        Three-state semantics:
          False  -- not connected (initial state)
          <obj>  -- connected camera instance (truthy; e.g. pylon.InstantCamera)
          None   -- disconnected / device removed (set by _mark_disconnected)

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
        # Subclass __init__ may raise before super().__init__() runs (e.g.
        # FX2Camera grabs _FX2Connection.get() first so it has self._fx2
        # ready for the base class's self.connect() call -- if that get()
        # raises on the Pylon-fallback path, this instance is partially
        # constructed and _state_lock + _active never got set). Python
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

    @abstractmethod
    def set_frame_size(self, w: int, h: int) -> dict | bool:
        """Set the output frame size.

        Drivers clamp or snap the request to their legal geometry grid, so
        the delivered size can differ from the request. Returning it from the
        write itself lets callers cache the real geometry without a follow-up
        getter round-trip.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            The delivered size ``{'width': int, 'height': int}`` on success;
            ``False`` when the camera is inactive or the apply fails.
        """
        pass

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

    @abstractmethod
    def get_frame_size(self) -> dict:
        """Return the current frame size.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        pass

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
        last_img_significant_bits attribute: the Pylon handler composes
        ImageHandlerBase (to avoid a metaclass conflict with the SDK event
        handler) and exposes only the method surface, so reaching the
        attribute directly raises AttributeError on a Pylon camera. The
        4-tuple read is atomic under the handler's frame lock, so the
        stamp cannot describe a different frame than the one returned
        beside it.
        """
        handler = self.cam_image_handler
        if handler is not None:
            success, _image, _ts, significant_bits = handler.get_last_image()
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
    def exposure_t(self, exposure_ms: float) -> None:
        """Set exposure time.

        Args:
            exposure_ms: Exposure time in milliseconds.
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
    def auto_exposure_t(self, state: bool = True) -> None:
        """Enable or disable hardware auto-exposure.

        Args:
            state: True to enable, False to disable.
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
        (e.g. FX2's 178 ms safe-frame ceiling).
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

    def get_max_gain(self) -> float:
        """Return the maximum gain cap in dB.

        Returns:
            float: Same value as the ``max_gain`` property.
        """
        return self.max_gain

    @abstractmethod
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0) -> None:
        """Enable or disable the SDK's frame-rate cap.

        Args:
            enabled: True to enforce ``fps`` as the upper bound.
            fps: Cap value in frames per second.
        """
        pass

    @abstractmethod
    def set_binning_size(self, size: int) -> bool:
        """Set hardware binning factor.

        Args:
            size: Binning factor (1, 2, 4, ...).

        Returns:
            bool: True on success.
        """
        pass

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

    def grab(self) -> tuple:
        """Grab the most recent frame from the image handler.

        On success, the image is also stored in ``self.array``.

        Returns:
            tuple: ``(success: bool, timestamp: datetime | None)``.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None

        if not self.cam_image_handler:
            return False, None

        try:
            result, image, image_ts, _significant_bits = self.cam_image_handler.get_last_image()
            if not result:
                return False, None

            with self._array_lock:
                self.array = image
            return True, image_ts
        except Exception as ex:
            _cam_log.exception(f'[CAM Class ] grab() - get_last_image() failed: {ex}')
            return False, None

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
                timestamp: datetime | None, significant_bits: int | None)``.
                The depth is carried with the frame so the caller scales it by
                the depth it was captured under, not a separately-queried one.
        """
        with self._state_lock:
            if self._active is None or self._device_removed:
                return False, None, None, None

        if not self.cam_image_handler:
            return False, None, None, None

        try:
            result, image, image_ts, image_significant_bits = (
                self.cam_image_handler.get_last_image()
            )
            if not result or image is None:
                return False, None, None, None

            # Store for other consumers (e.g. recording), but the returned
            # image IS the copy -- callers don't need get_array().
            with self._array_lock:
                self.array = image
            return True, image, image_ts, image_significant_bits
        except Exception as ex:
            _cam_log.exception(f'[CAM Class ] grab_latest() failed: {ex}')
            return False, None, None, None

    def register_frame_callback(self, cb) -> None:
        """Register a per-frame callback.

        Records the callback in the Camera's durable registry (so it survives a
        handler rebuild) AND applies it to the current handler for immediate
        dispatch. Idempotent for the same callable. SimulatedCamera extends this
        to also drive its host-side pump.
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
        """Re-register the durable callback set onto the current handler.

        A driver calls this immediately after building a new cam_image_handler
        (connect / recovery). The handler owns the dispatch list and starts
        empty, so without this every listener registered before the rebuild
        stops receiving frames. No-op when the driver has no handler
        (SimulatedCamera, which delivers via its own pump reading the registry).
        """
        handler = self.cam_image_handler
        if handler is None:
            return
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
            tuple: ``(success: bool, image: np.ndarray | None,
                timestamp: datetime | None)``.
        """
        pass

    @abstractmethod
    def update_auto_gain_target_brightness(self, auto_target_brightness: float) -> None:
        """Update the target brightness for the auto-gain loop.

        Args:
            auto_target_brightness: Normalized target brightness (0.0
                to 1.0).
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
    def gain(self, value: float) -> None:
        """Set the camera gain.

        Args:
            value: Gain in dB.
        """
        pass

    @abstractmethod
    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> None:
        """Enable or disable continuous auto-gain.

        Args:
            state: True to enable, False to disable.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
            ae_max_exposure_ms: Optional per-channel-class upper bound (ms)
                on the exposure auto-exposure may drive to. Honored where
                the driver supports auto-exposure bounds; ignored otherwise.
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
    ) -> None:
        """Run a single auto-gain iteration.

        Args:
            state: True to run, False to no-op.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
            ae_max_exposure_ms: Optional per-channel-class exposure upper
                bound (ms); honored where the driver supports it.
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
