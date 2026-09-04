# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ImagingAPI -- sub-API for camera capture / image acquisition.

ImagingAPI owns _camera_cache, _frame_buffer, _scale_bar,
_focusing_event, _camera_listeners, _camera_temp_event,
_suppress_value_warnings, and the frame_validity instance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import enum
import logging as _logging
import threading
import time
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterator

import numpy as np

from lib import profile_trace
from lvp_logger import logger
import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.exceptions import CameraSettingRejected, HardwareCommandRefusedError
from modules.frame_validity import FrameValidity
from modules.lumascope_api.illumination import live_lit_pairs
from modules.notification_center import notifications
from modules.sequential_io_executor import IOTask


class AutoGainConvergence(enum.Enum):
    """Where a continuous auto-gain arm landed when the capture locked it.

    CONVERGED: exposure inside the layer class's usable range.
    MAXED: exposure pinned at the class ceiling -- the scene is too dark
        for the range, so target brightness was not reached.
    AT_MINIMUM: exposure at or below the class usable floor -- the scene
        is too bright for the range.
    FAILED: the camera reported no usable achieved value, so the capture
        ran without an exposure/gain evidence gate.
    The limit states are outcomes, not failures: the frame is saved with
    the achieved values and a run continues.
    """

    CONVERGED = 'CONVERGED'
    MAXED = 'MAXED'
    AT_MINIMUM = 'AT_MINIMUM'
    FAILED = 'FAILED'


@dataclasses.dataclass(frozen=True)
class _AutoGainArm:
    """A commanded continuous auto-gain arm, held until a capture locks it.

    resume_after_capture: a live-view arm is re-armed after the capture
    so the view keeps adjusting; a protocol step's arm stays locked Off
    until the next step arms again.
    """

    settings: dict
    resume_after_capture: bool


@dataclasses.dataclass(frozen=True)
class AutoGainLock:
    """The result of locking an auto-gain arm; ``state`` is None when no
    arm was recorded (nothing was locked, no driver traffic happened)."""

    state: AutoGainConvergence | None
    exposure_ms: float | None = None
    gain_db: float | None = None
    floor_ms: float | None = None
    ceiling_ms: float | None = None
    resume_after_capture: bool = False
    settings: dict | None = None


if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.camera import Camera

_api_log = _logging.getLogger('LVP.api')


# Per Firmware/docs/PERFORMANCE_BUDGETS.md plugin_live_processing_handler_ms
# row + WAVE7_PHASE_4D5_PLAN sec 9 alignment 2026-05-19. Budget anchors to
# the 30 fps realistic-cap target (per FRAME_VALIDITY_RIG_COMPARISON_2026-
# 05-19.md: daA3840 max sustained median 17.6 fps; a2A3536 49.9 fps).
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
    threaded (Pylon SDK contract / IDS grab loop / Sim pump). Auto-
    removal calls ImagingAPI._remove_wrapper which takes the driver
    lock, but the driver's _store_frame snapshots callbacks under
    lock and invokes outside -- no deadlock risk on the same-thread
    auto-remove path.
    """

    __slots__ = ('_budget_trace', '_consecutive_over', '_handler', '_imaging', '_name', '_removed')

    def __init__(self, imaging: ImagingAPI, handler, name: str) -> None:
        self._imaging = imaging
        self._handler = handler
        self._name = name
        self._consecutive_over = 0
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

    def __call__(self, image, timestamp, chunks) -> None:
        if self._removed:
            return
        t0 = time.perf_counter()
        try:
            self._handler(image, timestamp, chunks)
        except Exception as e:
            # Log every error with context. Exception does not count
            # toward budget -- a handler that crashes is a different
            # failure class from a handler that's too slow.
            logger.exception(f"[SCOPE API ] live_processing handler '{self._name}' raised: {e}")
            return
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
                self._auto_remove(elapsed_ms)
        else:
            self._consecutive_over = 0

    def _auto_remove(self, last_elapsed_ms: float) -> None:
        """Drop trigger: K consecutive over-budget hits. Removes self
        from ImagingAPI's listener registry and fires a warning
        notification so L1 sees the degradation."""
        self._removed = True
        try:
            self._imaging._remove_wrapper(self)
        except Exception as e:
            logger.warning(
                f"[SCOPE API ] live_processing handler '{self._name}' "
                f'auto-remove cleanup failed: {e}'
            )
        notifications.warning(
            'Live Processing',
            f"Plugin '{self._name}' removed",
            f"The plugin's frame handler exceeded the {HANDLER_BUDGET_MS}ms "
            f'budget for {HANDLER_DROP_K} consecutive frames '
            f'(last: {last_elapsed_ms:.0f}ms). It has been disabled to '
            f"protect the imaging pipeline. Reduce the handler's per-frame "
            f'cost and re-register, or restart the application.',
        )


class ImagingAPI:
    """Imaging sub-API. Owns camera setters/getters/orchestration plus
    camera state slots (cache, frame_buffer, scale_bar, listeners,
    capture/focus events) and the frame_validity instance.
    """

    def __init__(self, scope: Lumascope, driver: Camera | None) -> None:
        self._scope = scope
        # driver argument kept for API compatibility but unused; `_driver`
        # is a @property that re-resolves `self._scope._camera_driver` so
        # disconnect / reconnect / test hot-swap propagate without
        # rebinding ImagingAPI. Same pattern as MotionAPI._driver /
        # IlluminationAPI._driver.
        del driver  # intentionally unused, kept for backward call sites

        # State / camera locks. _state_lock guards _scale_bar,
        # _last_capture_info and _auto_gain_arm; _cam_lock serializes
        # access to the camera driver itself (any path that touches
        # the SDK reads/writes goes through this lock).
        self._state_lock = threading.Lock()
        self._cam_lock = profile_trace.TimedLock(threading.RLock(), name='imaging._cam_lock')

        # Once-per-episode latch for the enabled-but-no-objective scale-bar
        # skip log; the gates run per displayed frame, so an unlatched
        # warning would flood the log at display rate.
        self._scale_bar_objective_skip_logged = False

        # Camera change listeners -- push-based UI update mechanism.
        # Each listener is called with (param: str, value: float) whenever
        # camera gain or exposure changes. param is 'gain' or 'exposure'.
        # Fires from the thread that caused the change, so listeners MUST
        # schedule UI work via Clock.schedule_once.
        self._camera_listeners_lock = threading.Lock()
        self._camera_listeners: list = []

        # Per-frame listener registry. Each user handler is wrapped in
        # a _BudgetedHandler; the wrapper is what's registered with the
        # driver's _frame_callbacks list. The dict keys are the user
        # handlers so remove_frame_listener(handler) can look up the
        # wrapper. Lock protects register + remove + auto-remove
        # mutations -- the read path (driver fan-out) doesn't touch
        # this dict.
        self._frame_listener_lock = threading.Lock()
        self._frame_listener_wrappers: dict = {}

        # Latest captured frame; populated by get_image_from_buffer.
        self._frame_buffer = None

        # Boolean operation flags -- threading.Event for wait/signal.
        self._focusing_event = threading.Event()  # set => autofocus in progress

        # Capture / autofocus return slots. Reads/writes under
        # self._state_lock. Per the Sentinel-return contract: None
        # means "no result yet."

        # Evidence about the most recent capture_and_wait (hold duration,
        # drained frame count, chunk-verified exposure / gain). Read via
        # last_capture_info by callers that log per-capture provenance.
        self._last_capture_info = None

        # The commanded continuous auto-gain arm, or None. Only the API
        # commands the auto mode and no driver reads it back, so this is
        # the single record that an arm is standing. Consumed atomically
        # by the capture that locks it (under _state_lock).
        self._auto_gain_arm: _AutoGainArm | None = None

        # When True, programmatic value-range warnings (sub-0.1ms exposure,
        # future similar setters) are silenced. Internal callers that sweep
        # full ranges (camera characterization, dynamic-range tests) enter
        # this via suppress_value_warnings(); the warnings exist for L1
        # researchers who type microsecond values thinking ms.
        self._suppress_value_warnings = False

        # Frame validity instance -- ground-truth for "is this frame
        # ready to score / save?" Phase 4d closes the cross-API loop
        # from Phase 3 where motion.py / illumination.py referenced
        # self._scope.frame_validity. Lumascope.__init__ wires up the
        # motion-settle check after this slot is set.
        self.frame_validity = FrameValidity()

        # Camera temp logging scheduler handle.
        self._camera_temp_event = None
        self._camera_temp_unschedule_fn = None

        # Scale-bar overlay config -- defaults disabled; users opt in via
        # set_scale_bar(...). Written from the GUI thread and read from the
        # capture and live-view threads, so every access outside this
        # constructor goes through self._state_lock, and readers take a
        # snapshot rather than reading the fields one at a time.
        self._scale_bar = {
            'enabled': False,
            'color': None,
        }

        # Camera state cache -- the single store every public camera getter
        # answers from. Updated when the camera connects, after every
        # set_gain_db/set_exposure/etc. write-through, and by each getter's
        # validated live read. Every entry is either a validated hardware
        # reading or its seed below, and the seeds are the documented
        # camera-absent defaults -- so a driver failure sentinel can never
        # be cached or returned, and for every key except 'binning' the
        # matching is_valid_* predicate rejects the seed, making "never
        # successfully read" detectable from the value itself. gain_db
        # seeds to -1.0 (not 0.0) because 0 dB is a legal gain and would
        # read as hardware truth. 'binning' seeds to 1, which is also its
        # absent default AND a legal factor: never-read and genuinely-1x1
        # are indistinguishable by design -- both honestly answer 1.
        self._camera_cache_lock = threading.Lock()
        self._camera_cache = {
            'active': False,
            'gain_db': -1.0,
            'exposure_ms': 0.0,
            'frame_size': {'width': 0, 'height': 0},
            'max_frame_size': {'width': 0, 'height': 0},
            'min_frame_size': {'width': 0, 'height': 0},
            'max_exposure_ms': 0.0,
            'max_gain_db': 0.0,
            'pixel_format': None,
            'binning': 1,
        }
        # Per-key write generation, bumped by every authoritative cache
        # write (_commit_camera_writes). A validated live READ snapshots
        # the generation before touching the driver and commits only if
        # it is unchanged, so a read that was in flight while a setter
        # landed can never overwrite the setter's newer truth with the
        # pre-write hardware value.
        self._camera_cache_write_gen: dict[str, int] = {}
        # Per-key monotonic timestamp of the last WARNING about a failed
        # camera read; failures inside the window log at debug instead,
        # so a dead camera polled at frame rate warns once per window,
        # not per frame.
        self._camera_read_warn_ts: dict[str, float] = {}
        self._populate_camera_cache()

    @property
    def _driver(self) -> Camera | None:
        """Resolve the camera driver via the composition root each access.

        Lumascope's `_camera_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps ImagingAPI in sync without rebinding.
        """
        return self._scope._camera_driver

    @property
    def _binning_size(self) -> int:
        """Last-known binning factor (cache-backed; no SDK read).

        Per-frame consumers (scale-bar sizing, FOV math) read this instead
        of get_binning_size() so the hot path never touches the SDK.
        """
        with self._camera_cache_lock:
            return self._camera_cache['binning']

    # --- Private helpers (relocated from Lumascope) ---

    def _load_camera_timing(self) -> None:
        """Load per-camera timing config if available.

        Looks for data/camera_timing/<model>.json and overrides
        FrameValidity.SKIP_FRAMES with measured values.
        """
        if not self._driver or not self._driver.active:
            return
        try:
            import json
            import os
            import pathlib

            model = getattr(self._driver, 'model_name', None)
            if not model:
                return
            safe_name = model.replace(' ', '_')
            timing_dir = (
                pathlib.Path(os.path.dirname(__file__)).parent.parent / 'data' / 'camera_timing'
            )
            timing_path = timing_dir / f'{safe_name}.json'
            if not timing_path.exists():
                return
            with open(timing_path) as f:
                config = json.load(f)
            self.frame_validity.load_camera_timing(config)
            logger.info(f'[SCOPE API ] Loaded camera timing config from {timing_path}')
        except Exception as e:
            logger.warning(f'[SCOPE API ] Failed to load camera timing config: {e}')

    def _populate_camera_cache(self) -> None:
        """Populate camera cache from hardware. Called at init and on reconnect.

        One canonical read path: every key refreshes through the same
        validated-read commit the public getters answer by, so a failed or
        sentinel read never overwrites a cached last-known-good value,
        one raising key cannot abort the rest of the round, and populate
        cannot drift from the getters' validation rules. The direct
        _live_validated_read calls cover the keys with no public getter
        (sensor minimum, exposure/gain ceilings).
        """
        if not self._driver or not self._driver.active:
            with self._camera_cache_lock:
                self._camera_cache['active'] = False
            return

        try:
            self.get_binning_size()
            self.get_gain_db()
            self.get_exposure_ms()
            self._get_frame_size()
            self._get_pixel_format()
            self._get_max_frame_size()
            self._live_validated_read(
                'min_frame_size',
                lambda driver: driver.get_min_frame_size(),
                common_utils.is_valid_frame_size,
                lambda v: {'width': int(v['width']), 'height': int(v['height'])},
            )
            self._live_validated_read(
                'max_exposure_ms',
                lambda driver: driver.get_max_exposure(),
                common_utils.is_valid_exposure_ms,
                float,
            )
            self._live_validated_read(
                'max_gain_db',
                lambda driver: driver.get_max_gain(),
                lambda v: isinstance(v, (int, float)) and v > 0,
                float,
            )
            with self._camera_cache_lock:
                self._camera_cache['active'] = True
            logger.info('[SCOPE API ] Camera cache populated')
        except Exception as e:
            logger.warning(f'[SCOPE API ] Failed to populate camera cache: {e}')
            with self._camera_cache_lock:
                self._camera_cache['active'] = bool(self._driver and self._driver.active)

    def _refresh_cache_from_hardware_after_auto(self) -> None:
        """Resync gain + exposure cache to hardware after an auto cycle.

        The auto-gain / auto-exposure SDK paths drive hardware values
        directly without going through this layer's cache, so when the
        auto cycle toggles off the cache may hold a stale pre-auto
        value. Without this refresh, the cache-equality skip at
        ``set_gain_db`` / ``set_exposure_ms`` short-circuits subsequent
        setter calls and hardware silently stays at the converged auto
        value -- the user-visible failure shape was a protocol's first
        run capturing at an unintended exposure inherited from a
        pre-scan live-mode AG cycle.

        Soft-fails on hardware-read exceptions by clearing the cached
        values to a sentinel so the next setter falls through (better
        than a stale cached value silently passing the equality check).
        """
        if not self._driver or not self._driver.active:
            return
        try:
            gain = self._driver.get_gain()
            exp = self._driver.get_exposure_t()
        except Exception as e:
            self._commit_camera_writes({'gain_db': -1.0, 'exposure_ms': -1.0})
            logger.warning(
                f'[SCOPE API ] cache refresh after auto-off failed: {e}; '
                f'cache invalidated to force next setter through.'
            )
            return
        # A non-physical reading (the drivers' negative failed-read
        # sentinel) routes to the SAME -1.0 invalidation the exception
        # path above writes -- NOT keep-prior: the pre-auto cached value
        # is known-stale here (hardware moved during the auto cycle), and
        # keeping it would let the setter equality check short-circuit.
        # Committed as an authoritative write (generation bump): a stale
        # getter read racing this resync must not resurrect the pre-auto
        # value.
        resync = {}
        if gain is not None:
            resync['gain_db'] = float(gain) if common_utils.is_valid_gain_db(gain) else -1.0
        if exp is not None:
            resync['exposure_ms'] = float(exp) if common_utils.is_valid_exposure_ms(exp) else -1.0
        if resync:
            self._commit_camera_writes(resync)
        # Diagnostic: record where hardware auto-gain/exposure actually
        # converged when the cycle ended. A converged gain that stayed near
        # the floor on a dim scene means the settle window ended before AG
        # ramped (under-converged -> dark capture); a gain at the ceiling
        # means the scene needs more light, not more settle frames.
        logger.debug(
            f'[AG CONVERGE] auto cycle ended; camera converged to gain={gain} dB exposure={exp} ms'
        )

    def _invalidate_camera_cache(self) -> None:
        """Mark camera cache as inactive (e.g. on disconnect)."""
        with self._camera_cache_lock:
            self._camera_cache['active'] = False

    def _fire_camera_listeners(self, param: str, value: float) -> None:
        """Notify all camera listeners of a setting change."""
        with self._camera_listeners_lock:
            listeners = list(self._camera_listeners)
        for fn in listeners:
            try:
                fn(param, value)
            except Exception as ex:
                _api_log.debug(f'camera listener error: {ex}')

    def _get_latest_chunks(self) -> dict | None:
        """Return per-frame chunk metadata for the most recent successful
        grab, or None if chunks aren't available.

        Camera handlers expose chunks differently:
          - PylonCamera.ImageHandler: composition -- chunks at handler._base
          - IDSCamera.ImageHandler: inheritance -- chunks at handler directly
          - FX2 / simulators: no chunks at all -> None

        Always returns None on any access path failure -- frame_validity
        falls back to skip-frames calibration when chunks aren't available.
        """
        if self._driver is None:
            return None
        handler = getattr(self._driver, 'cam_image_handler', None)
        if handler is None:
            return None
        # Composition (Pylon) first, then inheritance (IDS / direct base).
        base = getattr(handler, '_base', handler)
        if not hasattr(base, 'get_last_chunks'):
            return None
        try:
            return base.get_last_chunks()
        except Exception:
            return None

    def _chunk_target_mismatch(self) -> str | None:
        """Name the first chunk-validatable source whose latest frame chunk
        does not match its recorded target, or None if all match.

        None also covers every can't-verify case (no chunk support, chunk
        key absent, no target recorded) -- absence of evidence is not a
        mismatch, and cameras without chunks rely on skip-count settling.
        """
        chunks = self._get_latest_chunks()
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

    def _camera_write(
        self,
        write_fn: Callable[[], object],
        *,
        invalidates: tuple[str, ...] = (),
        force_invalidate: tuple[str, ...] = (),
        targets: tuple[tuple[str, float | None], ...] = (),
        force_clear: tuple[str, ...] = (),
        cache_update: dict[str, object] | None = None,
    ) -> object:
        """Single sanctioned path for a camera-state write and its validity
        consequence. Every camera setter routes its hardware write through here
        so the write and the frame-validity invalidation it requires are
        declared together -- a new setter cannot write a camera node and forget
        to invalidate.

        Order is load-bearing: the write happens first, then ``force_invalidate``
        fires unconditionally (the always-mark-RED contract for the manual value
        setters, so a hardware-rejected write still expires validity rather than
        leaving a stale frame acceptable), then the applied-only block runs. A
        write is "applied" when the driver did not return ``False`` -- an explicit
        ``False`` is the only rejection signal; a ``None`` return (drivers without
        a confirmation signal) and any truthy result both count as applied.

        Args:
            write_fn: Zero-arg callable performing the driver write (including
                any lock it needs) and returning the driver's result.
            invalidates: Sources to invalidate only when the write was applied.
            force_invalidate: Sources to invalidate unconditionally, regardless
                of the result.
            targets: ``(source, value)`` pairs passed to ``set_target`` when the
                write was applied (``value`` None clears the target).
            force_clear: Sources whose chunk target is cleared (set to None)
                unconditionally -- the mode/one-shot setters drop their manual
                target whether or not the driver reported the write applied, so
                chunk-match falls back to skip-frames settling (target
                maintenance stays outside the applied gate). Clear-only by
                construction: an unconditional write can only ever drop a
                target, never record one for a possibly-rejected value.
            cache_update: Keys to write into the ``_camera_cache`` snapshot when
                the write was applied.

        Returns:
            The driver write's result, so the caller can do its own rejection
            handling / listener fire.
        """
        result = write_fn()
        for source in force_invalidate:
            self.frame_validity.invalidate(source)
        for source in force_clear:
            self.frame_validity.set_target(source, None)
        applied = result is not False
        if applied:
            for source in invalidates:
                self.frame_validity.invalidate(source)
            for source, value in targets:
                self.frame_validity.set_target(source, value)
            if cache_update:
                self._commit_camera_writes(cache_update)
        return result

    # Failed camera reads inside this window after a WARNING log at debug;
    # mirrors the notification framework's short non-fatal dedup window so
    # a dead camera polled at frame rate warns once per window, not per
    # frame, while the first failure of a streak is always visible in the
    # main log.
    _READ_FAILURE_WARN_INTERVAL_S = 5.0

    def _commit_camera_writes(self, updates: dict) -> None:
        """Cache write-through for authoritative values.

        Authoritative means the value did not come from a read: a driver
        write that was applied, or a deliberate invalidation. Bumps each
        key's write generation so a validated live read that was already
        in flight when this landed cannot commit its now-stale value over
        the newer truth.
        """
        with self._camera_cache_lock:
            self._camera_cache.update(updates)
            for key in updates:
                self._camera_cache_write_gen[key] = self._camera_cache_write_gen.get(key, 0) + 1

    def _log_camera_read_failure(self, key: str, cause: object) -> None:
        """Surface a failed camera read without per-frame log spam.

        The read failures this layer absorbs (getters answer
        last-known-good) must still be visible in the main log, or a
        camera whose every read fails looks healthy in a tech-support
        bundle. First failure per key per window logs WARNING; repeats
        within the window log debug.
        """
        now = time.monotonic()
        with self._camera_cache_lock:
            last_warn = self._camera_read_warn_ts.get(key, 0.0)
            warn = (now - last_warn) >= self._READ_FAILURE_WARN_INTERVAL_S
            if warn:
                self._camera_read_warn_ts[key] = now
        if warn:
            logger.warning(
                f'[SCOPE API ] camera {key} read failed ({cause}); '
                f'answering last-known-good until a read succeeds'
            )
        else:
            _api_log.debug(f'camera {key} read failed: {cause}')

    # --- Setters ---
    def _set_gain_db_impl(self, gain_db: float) -> None:
        """Set the camera gain.

        Args:
            gain_db: Gain value in dB.
        """
        if not self._driver or not self._driver.active:
            return
        # The validity invalidate must never be gated by the software cache:
        # a cache desynced from hardware once short-circuited it, so a frame at
        # a stale gain was captured as valid. force_invalidate marks 'gain' RED
        # on every write (even a rejected one); the requested value is recorded
        # as the chunk target and cached only when the write was not rejected.
        # The driver compares against live hardware and skips a truly redundant
        # SDK write; the cache-equality check here gates only the UI listener +
        # info log, where a missed redundant update is harmless.
        changed = abs(float(gain_db) - self.gain_db_cached) >= 0.001

        def _write_gain():
            with self._cam_lock:
                return self._driver.gain(gain_db)

        ok = self._camera_write(
            _write_gain,
            force_invalidate=('gain',),
            targets=(('gain', float(gain_db)),),
            cache_update={'gain_db': float(gain_db)},
        )
        if ok is False:
            # Confirmed hardware rejection (drivers without a confirmation
            # signal return None). Frames keep streaming at the OLD gain,
            # and IDS has no chunk backstop to catch the mismatch
            # downstream -- surface it instead of recording the requested
            # value as truth in the cache.
            notifications.error(
                'Camera',
                'Camera Setting Not Applied',
                f'The camera rejected the gain change to {float(gain_db):.1f} dB. '
                'Captures will continue at the previous gain. Check that '
                'the value is within the camera limits.',
            )
        elif changed:
            _api_log.info(f'set_gain_db {gain_db}dB')
            self._fire_camera_listeners('gain', float(gain_db))

    def _set_exposure_ms_impl(self, exposure_ms: float) -> None:
        """Set the camera exposure time.

        Args:
            exposure_ms: Exposure time in milliseconds.
        """
        if not self._driver or not self._driver.active:
            return
        # The validity invalidate must never be gated by the software cache:
        # a cache desynced from hardware once short-circuited it, capturing a
        # frame at a stale exposure as valid. Always invalidate + drive the
        # setter; the cache-equality check gates only the UI listener + log.
        changed = abs(float(exposure_ms) - self.exposure_ms_cached) >= 0.001
        # Sanity-check threshold: 5 microseconds. Pylon physical
        # ExposureTime minimum across Basler USB3 sensors is 10-35 us;
        # below 5 us is impossible on any sensor we ship with and
        # indicates a unit-confusion bug (e.g. seconds-treated-as-ms).
        # Bright-field captures legitimately use 0.03 ms (30 us) on
        # bright samples, so the threshold sits below that range.
        if exposure_ms < 0.005 and not self._suppress_value_warnings:
            import traceback

            _caller = ''.join(traceback.format_stack(limit=6)[-4:-1]).strip()
            logger.warning(
                f'[SCOPE API ] set_exposure_ms({exposure_ms}ms) is below '
                f'any Basler sensor physical minimum -- camera '
                f'will clamp the request. Confirm the value is '
                f'in milliseconds, not seconds or microseconds.\n'
                f'Call stack:\n{_caller}'
            )

        # Record requested exposure for chunk-match. ChunkExposureTime is
        # microseconds; the API takes milliseconds. Convert at the seam so the
        # chunk value and frame_validity's tolerance share units. force_invalidate
        # marks 'exposure' RED on every write; target + cache only when the write
        # was not rejected.
        def _write_exposure():
            with self._cam_lock:
                return self._driver.exposure_t(exposure_ms)

        ok = self._camera_write(
            _write_exposure,
            force_invalidate=('exposure',),
            targets=(('exposure', float(exposure_ms) * 1000.0),),
            cache_update={'exposure_ms': float(exposure_ms)},
        )
        if ok is False:
            # Confirmed hardware rejection (drivers without a confirmation
            # signal return None). Frames keep streaming at the OLD
            # exposure, and IDS has no chunk backstop to catch the
            # mismatch downstream -- surface it instead of recording the
            # requested value as truth in the cache.
            notifications.error(
                'Camera',
                'Camera Setting Not Applied',
                f'The camera rejected the exposure change to {float(exposure_ms):g} ms. '
                'Captures will continue at the previous exposure. Check '
                'that the value is within the camera limits.',
            )
        elif changed:
            _api_log.info(f'set_exposure {exposure_ms}ms')
            self._fire_camera_listeners('exposure', float(exposure_ms))

    # --- Public dispatch ---
    # These three are what an external caller reaches: an SDK script, a REST
    # handler, the GUI. Every internal caller binds the matching `_impl`
    # instead, so nothing already running on an executor worker or on the
    # protocol or autofocus thread ever arrives here.

    # How long a dispatched camera write waits on the camera worker before
    # giving up. A gain or exposure write is a short SDK call behind at most
    # a few queued camera commands, so this is a liveness bound rather than
    # a budget -- if it ever expires, the worker is wedged and the caller
    # should hear about it instead of blocking forever. Deliberately not a
    # public parameter: an external caller has no way to know what value
    # would be right.
    _CAMERA_WRITE_TIMEOUT_S = 5.0

    # The geometry class (frame size, pixel format, binning) is slower
    # than a value write: a large-frame resize on a Pylon body has been
    # measured near 11 s, and the dispatcher's wait bounds QUEUE TIME
    # plus execution -- a geometry write queued behind another one must
    # not time out while both are healthy. The bound stays a liveness
    # verdict, not a budget: `fut.result` ABANDONS on timeout without
    # cancelling, so a timed-out write still lands later and the caller's
    # view of the camera diverges -- which is why this must be sized so a
    # healthy write can never hit it.
    _CAMERA_GEOMETRY_TIMEOUT_S = 30.0

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

    def _dispatch_camera(self, impl, name, args=(), kwargs=None, *, timeout_s):
        """Run one camera command for an external caller, on the right thread.

        Three outcomes. With no executor registered the body runs on the
        calling thread -- a bare `Lumascope()` in a script or an example has
        no executors and still has to drive hardware. With a live executor
        the body runs on the camera worker, serialized against every other
        camera-bus operation, and this blocks until it has. With an executor
        that will not accept work the caller is told so, because the
        alternative is `put` returning None and the command disappearing
        with nothing raised and nothing logged.

        The refusal asks only WHETHER work is accepted. A run disables the
        camera executor outright (io and file are fenced instead), and `put`
        reports both states the same way, so a branch that asked WHY would
        need a list of executor states kept in sync with the executor.

        Unlike the LED dispatcher there is no connected pre-check here: the
        camera slot holds None when no camera is present -- there is no Null
        camera object -- so each `_impl` opens with a live driver guard and
        answers correctly on whichever thread it runs.
        """
        kwargs = kwargs or {}
        ex = self._scope._camera_executor
        if ex is None:
            return impl(*args, **kwargs)
        if not ex.accepts_work():
            raise HardwareCommandRefusedError('exclusive_activity_running', name)
        fut = ex.put(IOTask(action=impl, args=args, kwargs=kwargs), return_future=True)
        if fut is None:
            # A protocol fence can land between the check above and the
            # submit; without this the race surfaces as an AttributeError on
            # the missing future instead of the typed refusal.
            raise HardwareCommandRefusedError('exclusive_activity_running', name)
        return fut.result(timeout=timeout_s)

    def set_gain_db(self, gain_db: float) -> None:
        """Set the camera gain, and wait for it.

        See ``_set_gain_db_impl`` for the value contract and the rejection
        notification; this adds only the dispatch described on
        ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._set_gain_db_impl,
            'set_gain_db',
            args=(gain_db,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def set_exposure_ms(self, exposure_ms: float) -> None:
        """Set the camera exposure time, and wait for it.

        See ``_set_exposure_ms_impl`` for the value contract and the
        unit-confusion warning it carries.
        """
        return self._dispatch_camera(
            self._set_exposure_ms_impl,
            'set_exposure_ms',
            args=(exposure_ms,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def set_auto_gain(
        self, state: bool, settings: dict, *, resume_after_capture: bool = True
    ) -> None:
        """Enable or disable automatic gain adjustment, and wait for it.

        See ``_set_auto_gain_impl`` for the value contract; this adds
        only the dispatch described on ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._set_auto_gain_impl,
            'set_auto_gain',
            args=(state, settings),
            kwargs={'resume_after_capture': resume_after_capture},
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _set_auto_gain_impl(
        self, state: bool, settings: dict, *, resume_after_capture: bool = True
    ) -> None:
        """Enable or disable automatic gain adjustment.

        Args:
            state: True to enable auto gain, False to disable.
            settings: Dict with 'target_brightness', 'min_gain_db', 'max_gain_db',
                and optionally 'max_exposure_ms' / 'min_exposure_ms' (the
                per-channel-class bounds of the exposure AG/AE may settle
                in; the caller supplies them since it knows the layer).
            resume_after_capture: with ``state=True``, whether a capture
                that locks this arm re-arms it afterwards (live view) or
                leaves the camera at the locked values (a protocol step).
        """

        if not self._driver or not self._driver.active:
            return

        def _write_auto_gain():
            self._driver.auto_gain(
                state,
                target_brightness=settings['target_brightness'],
                min_gain_db=settings['min_gain_db'],
                max_gain_db=settings['max_gain_db'],
                ae_max_exposure_ms=settings.get('max_exposure_ms'),
            )

        # Auto-gain dynamically adjusts the value, so clear the manual gain
        # target (chunk-match falls back to skip-frames calibration). Arming
        # hardware continuous AG needs the camera several frames to settle
        # against the lit scene, so invalidate 'auto_gain' to hold capture for
        # the settle count -- gated on the camera actually having hardware AG
        # (cameras without it reach correct exposure through a future software-AG
        # loop that reuses the gain/exposure settle sources, not this one). The
        # mode flip leaves the gain value node unchanged, so these are forced,
        # not gated on a value delta.
        arm_settle = state and getattr(self._driver.profile, 'has_auto_gain', False)
        self._camera_write(
            _write_auto_gain,
            force_invalidate=('gain', 'auto_gain') if arm_settle else ('gain',),
            force_clear=('gain',),
        )
        with self._state_lock:
            self._auto_gain_arm = (
                _AutoGainArm(dict(settings), resume_after_capture) if arm_settle else None
            )
        # Hardware-truth wins over cache after the auto cycle ends.
        if not state:
            self._refresh_cache_from_hardware_after_auto()

    def _lock_auto_gain_impl(self) -> AutoGainLock:
        """Turn a standing continuous auto-gain arm into locked manual values.

        The camera's auto loop chose an exposure and gain the caller never
        requested, so the capture gate's recorded targets are stale. The
        lock writes the achieved values back as manual values through the
        ordinary setters, which re-targets the gate; the next frame must
        then prove them like any manual setting. The achieved values come
        from the last stored frame's chunks when the camera has them (the
        same node and unit the gate compares), else from the cache the
        disarm just resynced from hardware.

        Returns an AutoGainLock; ``state`` is None when no arm was
        recorded, and nothing was written.
        """
        with self._state_lock:
            arm, self._auto_gain_arm = self._auto_gain_arm, None
        if arm is None:
            return AutoGainLock(state=None)
        settings = arm.settings
        floor = settings.get('min_exposure_ms') or None
        ceiling = settings.get('max_exposure_ms') or None
        self._set_auto_gain_impl(False, settings)
        chunks = self._get_latest_chunks() or {}
        exp_us = chunks.get('ExposureTime')
        gain = chunks.get('Gain')
        if isinstance(exp_us, (int, float)) and isinstance(gain, (int, float)):
            exp_ms: object = exp_us / 1000.0
        else:
            exp_ms = self.exposure_ms_cached
            gain = self.gain_db_cached
        if not (common_utils.is_valid_exposure_ms(exp_ms) and common_utils.is_valid_gain_db(gain)):
            # No usable achieved value: drop the stale exposure target too
            # (the disarm dropped the gain target) so the capture proceeds
            # without an exposure/gain gate rather than rejecting forever.
            self._camera_write(lambda: None, force_clear=('exposure',))
            logger.warning(
                '[AG CONVERGE] locked: state=FAILED -- the camera reported no '
                f'usable achieved exposure/gain (exposure={exp_ms} gain={gain}); '
                'the capture proceeds ungated on exposure and gain'
            )
            return AutoGainLock(
                AutoGainConvergence.FAILED,
                None,
                None,
                floor,
                ceiling,
                arm.resume_after_capture,
                settings,
            )
        exp_ms = float(exp_ms)
        gain = float(gain)
        self._set_exposure_ms_impl(exp_ms)
        self._set_gain_db_impl(gain)
        if ceiling is not None and exp_ms >= ceiling * 0.99:
            state = AutoGainConvergence.MAXED
        elif floor is not None and exp_ms <= floor:
            state = AutoGainConvergence.AT_MINIMUM
        else:
            state = AutoGainConvergence.CONVERGED
        line = (
            f'[AG CONVERGE] locked: state={state.value} exposure={exp_ms:.3f} ms '
            f'gain={gain:.2f} dB (class floor={floor} ms ceiling={ceiling} ms)'
        )
        if state is AutoGainConvergence.CONVERGED:
            logger.debug(line)
        else:
            logger.info(line)
        return AutoGainLock(state, exp_ms, gain, floor, ceiling, arm.resume_after_capture, settings)

    def _resume_auto_gain_impl(self, lock: AutoGainLock) -> None:
        """Re-arm continuous auto-gain after a capture locked a live-view arm."""
        if lock.state is not None and lock.resume_after_capture and lock.settings is not None:
            self._set_auto_gain_impl(True, lock.settings, resume_after_capture=True)

    def set_auto_exposure_time(self, state: bool = True) -> None:
        """Enable or disable automatic exposure adjustment, and wait for it.

        See ``_set_auto_exposure_time_impl`` for the value contract; this
        adds only the dispatch described on ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._set_auto_exposure_time_impl,
            'set_auto_exposure_time',
            args=(state,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _set_auto_exposure_time_impl(self, state: bool = True) -> None:
        """Enable or disable automatic exposure adjustment.

        Args:
            state: True to enable auto exposure, False to disable.
        """

        if not self._driver or not self._driver.active:
            return
        # Auto-exposure dynamically adjusts the value, so clear the manual
        # exposure target (chunk-match falls back to skip-frames calibration).
        # The mode flip leaves the exposure value node unchanged, so the
        # invalidate + target-clear are forced, not gated on a value delta.
        self._camera_write(
            lambda: self._driver.auto_exposure_t(state),
            force_invalidate=('exposure',),
            force_clear=('exposure',),
        )
        # Hardware-truth wins over cache after the auto cycle ends.
        if not state:
            self._refresh_cache_from_hardware_after_auto()

    def _camera_setting_rejection(
        self, setting: str, requested, title: str, body: str
    ) -> CameraSettingRejected:
        """Log + notify + build the typed rejection for a camera-setting apply.

        Callers ``raise self._camera_setting_rejection(...)`` so the raise
        is explicit at every rejection site while the load-bearing ordering
        (log, then notify, then the exception -- the exception class
        documents that the rejection is already surfaced when it arrives)
        lives in one place for all setters.
        """
        logger.error(f'[SCOPE API ] {setting}: driver rejected {requested!r}')
        notifications.error('Camera', title, body)
        return CameraSettingRejected(setting, requested)

    def set_frame_size(self, w: int, h: int) -> dict | None:
        """Set the camera frame size in pixels, and wait for it.

        See ``_set_frame_size_impl`` for the delivered-geometry contract
        and the rejection semantics; this adds only the dispatch
        described on ``_dispatch_camera``, on the geometry timeout (a
        large-frame resize is a slow write).
        """
        return self._dispatch_camera(
            self._set_frame_size_impl,
            'set_frame_size',
            args=(w, h),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
        )

    def _set_frame_size_impl(self, w: int, h: int) -> dict | None:
        """Set the camera frame size in pixels.

        Success is observed by receiving the DELIVERED geometry; rejection
        is a raise. A caller therefore cannot record a rejected or clamped
        apply as current geometry -- the only value available to record is
        the one the camera actually took (a rejected resize once left the
        UI, FOV math, and saved settings claiming a size the camera never
        held, and the retry was absorbed by a dedupe record built from the
        request).

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            dict | None: The DELIVERED ``{'width', 'height'}`` -- the
                clamped/snapped geometry actually in effect, which may
                differ from the request. None when no camera is active
                (quiet no-op per the missing-hardware contract; a
                notification fires).

        Raises:
            CameraSettingRejected: A live driver rejected the apply. The
                rejection is logged and notified before the raise.
        """

        if not self._driver or not self._driver.active:
            self._notify_camera_absent('frame size')
            return None
        # A frame-size change reallocates buffers; the pipeline must flush, so
        # invalidate unconditionally. Cache the DELIVERED size, not the request:
        # a driver may clamp or snap the request to its legal grid
        # (oversize-then-crop, alignment multiples), and FOV / pixel-size math
        # reads this cache, so it must reflect what the camera actually
        # delivers. The delivered geometry comes from the write's own return
        # value -- a separate get_frame_size() read-back is deliberately
        # avoided because a transient read error there can spuriously drop the
        # camera. A falsy result means the WRITE failed, so the prior cache
        # entry (still describing the hardware) is left in place.
        delivered = self._camera_write(
            lambda: self._driver.set_frame_size(w, h),
            force_invalidate=('frame_size',),
        )
        if not delivered:
            raise self._camera_setting_rejection(
                'frame_size',
                {'width': w, 'height': h},
                'Frame size change failed',
                f'The camera did not accept the frame size {w}x{h}. '
                f'It remains at the previous size -- try again, or check '
                f'the USB connection if this repeats.',
            )
        delivered_size = {'width': int(delivered['width']), 'height': int(delivered['height'])}
        self._commit_camera_writes({'frame_size': dict(delivered_size)})
        return delivered_size

    def _notify_camera_absent(self, op_label: str) -> None:
        """Fire a deduped notification when a camera-required operation
        is invoked without an active camera. notification_center collapses
        repeats by (category, title) over 5s so a chain of failed setter
        calls during a disconnected window yields one popup, not dozens.
        Internal-poll callers (scope_display auto-gain readback) and
        cleanup paths intentionally do NOT route through this.

        Suppressed in no_hardware mode (cold-start with nothing
        connected) -- the consolidated "No hardware detected" popup
        fires from lumaviewpro.on_start and the per-setter popup
        would stack on top of it. Runtime disconnects (camera unplugged
        mid-session) still notify because no_hardware is False then.
        """
        if getattr(self._scope, 'no_hardware', False):
            return
        notifications.warning(
            'Camera',
            'Camera not connected',
            f'Cannot change {op_label} -- camera is not connected. '
            f'Check USB and reconnect, then try again.',
        )

    def set_binning_size(self, size: int) -> bool:
        """Set camera pixel binning size, and wait for it.

        See ``_set_binning_size_impl`` for the apply/rejection contract;
        this adds only the dispatch described on ``_dispatch_camera``,
        on the geometry timeout (binning reallocates buffers).
        """
        return self._dispatch_camera(
            self._set_binning_size_impl,
            'set_binning_size',
            args=(size,),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
        )

    def _set_binning_size_impl(self, size: int) -> bool:
        """Set camera pixel binning size.

        Args:
            size: Binning factor (1 = no binning, 2 = 2x2, etc.).

        Returns:
            bool: True when the driver applied the binning. False only
                when the camera is absent (quiet no-op per the
                missing-hardware contract; a notification fires).

        Raises:
            CameraSettingRejected: A live driver rejected the apply or
                raised from it. The rejection is logged and notified
                before the raise, so a rejected binning cannot be
                recorded as current by a caller that drops the return --
                a rejected binning silently poisons every native-ROI /
                FOV / stitch derivation built on the recorded factor.
        """
        if not self._driver or not self._driver.active:
            self._notify_camera_absent('binning')
            return False
        try:
            # Binning realloc only takes effect when the driver applied it,
            # so the invalidate is gated on the driver result.
            # The factor is committed to the cache only when the driver
            # applied it. A rejected write must not leave the requested
            # value where scale-bar / FOV math reads it -- the hardware
            # is still at the previous binning.
            ok = self._camera_write(
                lambda: self._driver.set_binning_size(size=size),
                invalidates=('binning',),
                cache_update={'binning': int(size)},
            )
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting binning size: {ex}')
            raise self._camera_setting_rejection(
                'binning',
                size,
                'Binning change failed',
                f'Could not set binning to {size}x{size}: {type(ex).__name__}: {ex}. '
                f'Camera may still be at previous binning -- verify actual frame size.',
            ) from ex
        # `is False` (not falsy): an explicit False is the driver's only
        # rejection signal; a None return (drivers without a confirmation
        # signal) counts as applied, matching _camera_write's own applied
        # semantics -- treating None as rejection here would raise while
        # the cache had already committed the factor.
        if ok is False:
            raise self._camera_setting_rejection(
                'binning',
                size,
                'Binning change failed',
                f'The camera did not accept {size}x{size} binning. It remains '
                f'at the previous binning -- try again, or check the USB '
                f'connection if this repeats.',
            )
        # Both cached geometries are binning-dependent: the sensor
        # minimum halves at 2x, and the delivered frame size halves
        # with it. Left stale, the UI's frame-size clamp reads the 1x
        # minimum and FOV math reads the 1x frame size until the next
        # full hardware refresh. set_binning_size returns no geometry,
        # so this refresh reads the getters -- a deliberate exception
        # to the no-read-back preference, accepted because binning
        # changes are rare user actions, not per-frame traffic. The
        # validated-read path keeps a failed refresh from caching a
        # sentinel (the prior geometry stays in place).
        self._live_validated_read(
            'min_frame_size',
            lambda driver: driver.get_min_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
        )
        self._get_frame_size()
        _api_log.info(f'set_binning {size}x{size} -> True')
        return True

    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format, and wait for it.

        See ``_set_pixel_format_impl`` for the apply/rejection contract;
        this adds only the dispatch described on ``_dispatch_camera``,
        on the geometry timeout (a format change reallocates geometry).
        """
        return self._dispatch_camera(
            self._set_pixel_format_impl,
            'set_pixel_format',
            args=(pixel_format,),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
        )

    def _set_pixel_format_impl(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Format string (e.g. 'Mono8', 'Mono12').

        Returns:
            bool: True when the driver applied the format. False only
                when the camera is absent / inactive (quiet no-op per the
                missing-hardware contract; a notification fires).

        Raises:
            CameraSettingRejected: A live driver rejected the format
                (unsupported) or raised from the apply. Logged and
                notified before the raise, so a caller that drops the
                return cannot record a rejected format as current --
                capture depth, saved-file tagging, and data-rate math all
                key off the recorded format.
        """
        if not self._driver or not self._driver.active:
            self._notify_camera_absent('pixel format')
            return False
        try:
            # Format change reallocates geometry only when the driver applied
            # it, so invalidate + snapshot are gated on the driver result.
            result = self._camera_write(
                lambda: self._driver.set_pixel_format(pixel_format),
                invalidates=('pixel_format',),
                cache_update={'pixel_format': pixel_format},
            )
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting pixel format: {ex}')
            raise self._camera_setting_rejection(
                'pixel_format',
                pixel_format,
                'Pixel format change failed',
                f'Could not set pixel format to {pixel_format}: '
                f'{type(ex).__name__}: {ex}. Camera may still be at the '
                f'previous format.',
            ) from ex
        if result is False:
            raise self._camera_setting_rejection(
                'pixel_format',
                pixel_format,
                'Pixel format change failed',
                f'The camera did not accept the pixel format {pixel_format}. '
                f'It remains at the previous format -- try again, or check '
                f'the USB connection if this repeats.',
            )
        return True

    def set_conversion_gain_mode(self, mode: str) -> bool:
        """Set the camera sensor conversion gain mode, and wait for it.

        See ``_set_conversion_gain_mode_impl`` for the mode contract;
        this adds only the dispatch described on ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._set_conversion_gain_mode_impl,
            'set_conversion_gain_mode',
            args=(mode,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _set_conversion_gain_mode_impl(self, mode: str) -> bool:
        """Set the camera sensor conversion gain mode.

        High conversion gain lowers the sensor read-noise floor (better
        low-light signal-to-noise) at the cost of dynamic range; Low is
        the standard wide-range mode. Pylon-only -- returns False on
        cameras/drivers that don't implement the setter.

        Args:
            mode: 'High' (low noise) or 'Low' (wide dynamic range).

        Returns:
            bool: True on success. False if the camera is absent, the
                driver doesn't implement the setter, or the driver
                returned False / raised. Never raises.
        """
        if not self._driver or not self._driver.active:
            self._notify_camera_absent('conversion gain mode')
            return False
        if not hasattr(self._driver, 'set_conversion_gain_mode'):
            logger.debug(
                f'[SCOPE API ] set_conversion_gain_mode: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return False
        try:
            result = self._camera_write(
                lambda: self._driver.set_conversion_gain_mode(mode),
                invalidates=('conversion_gain_mode',),
            )
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting conversion gain mode: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Conversion gain mode change failed',
                f'Could not set conversion gain mode to {mode!r}. '
                f'Camera may still be at the previous mode. See the log for details.',
            )
            return False
        return result

    def set_line_noise_reduction(self, enabled: bool) -> bool:
        """Enable or disable the line-noise filter, and wait for it.

        See ``_set_line_noise_reduction_impl`` for the contract; this
        adds only the dispatch described on ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._set_line_noise_reduction_impl,
            'set_line_noise_reduction',
            args=(enabled,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _set_line_noise_reduction_impl(self, enabled: bool) -> bool:
        """Enable or disable the camera line-noise reduction filter.

        A camera-side filter that smooths horizontal stripe artifacts in
        the sensor readout. Pylon-only -- returns False on cameras/drivers
        that don't implement the setter.

        Args:
            enabled: True turns the filter on; False off.

        Returns:
            bool: True on success. False if the camera is absent, the
                driver doesn't implement the setter, or the driver
                returned False / raised. Never raises.
        """
        if not self._driver or not self._driver.active:
            self._notify_camera_absent('line noise reduction')
            return False
        if not hasattr(self._driver, 'set_line_noise_reduction'):
            logger.debug(
                f'[SCOPE API ] set_line_noise_reduction: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return False
        try:
            result = self._camera_write(
                lambda: self._driver.set_line_noise_reduction(enabled=enabled),
                invalidates=('line_noise_reduction',),
            )
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting line noise reduction: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Line noise reduction change failed',
                f'Could not {"enable" if enabled else "disable"} line noise reduction. '
                f'See the log for details.',
            )
            return False
        return result

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
        try:
            return bool(self._driver.set_acquisition_stop_mode(mode=mode))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting acquisition_stop_mode: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'BslAcquisitionStopMode change failed',
                f'Could not set acquisition_stop_mode to {mode!r}. '
                f'Camera may still be at the previous stop-mode setting. '
                f'See the log for details.',
            )
            raise

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
        try:
            return bool(self._driver.set_bandwidth_reserve_mode(mode=mode))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting BandwidthReserveMode: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'BandwidthReserveMode change failed',
                f'Could not set BandwidthReserveMode to {mode!r}. See the log for details.',
            )
            raise

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
        try:
            return bool(
                self._driver.set_device_link_throughput_limit(
                    mode=mode,
                    value_bps=value_bps,
                )
            )
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting DLTL: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'DeviceLinkThroughputLimit change failed',
                f'Could not set DLTL to mode={mode}, value_bps={value_bps}. '
                f'Camera may still be at the previous DLTL setting. '
                f'See the log for details.',
            )
            raise

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
        try:
            return bool(self._driver.set_max_transfer_size(value_bytes=value_bytes))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting MaxTransferSize: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'MaxTransferSize change failed',
                f'Could not set MaxTransferSize to {value_bytes}. See the log for details.',
            )
            raise

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
        try:
            return bool(self._driver.set_num_max_queued_urbs(value=value))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting NumMaxQueuedUrbs: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'NumMaxQueuedUrbs change failed',
                f'Could not set NumMaxQueuedUrbs to {value}. See the log for details.',
            )
            raise

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
        try:
            return bool(self._driver.set_max_num_buffer(value=int(value)))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting MaxNumBuffer: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'MaxNumBuffer change failed',
                f'Could not set MaxNumBuffer to {value}. See the log for details.',
            )
            raise

    def _set_grab_strategy(self, name: str) -> bool:
        """Set the Pylon GrabStrategy used by the next start_grabbing().

        Production default is LatestImageOnly (the contract for
        frame_validity, capture_and_wait, and the auto-discard
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
        try:
            return bool(self._driver.set_gev_packet_size(size_bytes=size_bytes))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting GevSCPSPacketSize: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'GevSCPSPacketSize change failed',
                f'Could not set GevSCPSPacketSize to {size_bytes}. See the log for details.',
            )
            raise

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
        try:
            return bool(self._driver.set_gev_inter_packet_delay(delay_ticks=delay_ticks))
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting GevSCPD: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'GevSCPD change failed',
                f'Could not set GevSCPD to {delay_ticks}. See the log for details.',
            )
            raise

    def _live_validated_read(
        self,
        key: str,
        reader: Callable[[Camera], object],
        is_valid: Callable[[object], bool],
        coerce: Callable[[Any], Any],
    ) -> object | None:
        """Attempt one live driver read: validate, commit to cache, return.

        Returns None when no camera is active or the read raised or came
        back as the driver's failure sentinel (-1 / None / {}). The commit
        is generation-guarded: if an authoritative write
        (_commit_camera_writes) landed while the driver read was in
        flight, the stale read is discarded and the newer cache value is
        returned instead, so a slow read can never overwrite a setter's
        write-through.

        Callers decide what None means: the value getters fall back to
        the cached last-known-good; the live-confirmed consumers
        (metadata writers, state snapshots) omit the field, because for
        them a stale value recorded as truth is worse than no value.
        """
        driver = self._driver
        if not driver or not driver.active:
            return None
        with self._camera_cache_lock:
            gen_before = self._camera_cache_write_gen.get(key, 0)
        try:
            live = reader(driver)
        except Exception as ex:
            self._log_camera_read_failure(key, ex)
            return None
        if not is_valid(live):
            self._log_camera_read_failure(key, f'sentinel value {live!r}')
            return None
        value = coerce(live)
        with self._camera_cache_lock:
            if self._camera_cache_write_gen.get(key, 0) == gen_before:
                self._camera_cache[key] = value
            else:
                value = self._camera_cache[key]
        return value

    def _validated_camera_read(
        self,
        key: str,
        reader: Callable[[Camera], object],
        is_valid: Callable[[object], bool],
        coerce: Callable[[Any], Any],
        absent: object,
    ) -> object:
        """Refresh-then-read: the single path a public camera getter answers by.

        Attempts the live driver read on every call (so a getter tracks
        hardware while auto-gain moves it), commits the result to the camera
        cache only when it validates, and answers from the cache either way.
        A driver failure sentinel (-1 / None / {}) or a raising read can
        therefore never cross this boundary: consumers receive the
        last-known-good value, or ``absent`` (the documented camera-absent
        default) when no camera is active or no valid value has ever been
        read.

        Args:
            key: Camera-cache key backing this getter.
            reader: Live read, called with the resolved driver.
            is_valid: Shared validity predicate for this value.
            coerce: Normalizes a validated live value for caching.
            absent: Documented return when no camera is active or no valid
                value is known.

        Returns:
            The validated live value, the cached last-known-good, or
            ``absent``.
        """
        driver = self._driver
        if not driver or not driver.active:
            return absent
        live = self._live_validated_read(key, reader, is_valid, coerce)
        if live is not None:
            return live
        with self._camera_cache_lock:
            cached = self._camera_cache[key]
        return cached if is_valid(cached) else absent

    def get_live_camera_settings(self) -> dict:
        """Live-confirmed camera settings, omitting any field whose read
        did not just succeed.

        For consumers that record what the hardware was at a specific
        moment (saved-image metadata, state snapshots): the value getters
        (get_gain_db / get_exposure_ms) deliberately hide read failures
        behind the last-known-good cache, which is right for control flow
        but would record a value the frame was not captured at. Here,
        unknown stays unknown.

        Returns:
            dict: Any of 'gain_db', 'exposure_ms', 'frame_size'
                (``{'width': int, 'height': int}``), and 'pixel_format' --
                only fields whose live driver read succeeded and
                validated. Empty when no camera is active.
        """
        settings = {}
        gain = self._live_validated_read(
            'gain_db',
            lambda driver: driver.get_gain(),
            common_utils.is_valid_gain_db,
            float,
        )
        if gain is not None:
            settings['gain_db'] = gain
        exposure = self._live_validated_read(
            'exposure_ms',
            lambda driver: driver.get_exposure_t(),
            common_utils.is_valid_exposure_ms,
            float,
        )
        if exposure is not None:
            settings['exposure_ms'] = exposure
        frame_size = self._live_validated_read(
            'frame_size',
            lambda driver: driver.get_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
        )
        if frame_size is not None:
            settings['frame_size'] = dict(frame_size)
        pixel_format = self._live_validated_read(
            'pixel_format',
            lambda driver: driver.get_pixel_format(),
            common_utils.is_valid_pixel_format,
            str,
        )
        if pixel_format is not None:
            settings['pixel_format'] = pixel_format
        return settings

    def get_gain_db(self) -> float:
        """Get the current camera gain.

        Returns:
            float: Gain in dB -- the live reading when it succeeds, else the
                last-known-good value. -1 when no camera is active or gain
                has never been read.
        """
        return self._validated_camera_read(
            'gain_db',
            lambda driver: driver.get_gain(),
            common_utils.is_valid_gain_db,
            float,
            -1.0,
        )

    def get_exposure_ms(self) -> float:
        """Get the current camera exposure time.

        Returns:
            float: Exposure time in milliseconds -- the live reading when it
                succeeds, else the last-known-good value. 0 when no camera
                is active or exposure has never been read.
        """
        return self._validated_camera_read(
            'exposure_ms',
            lambda driver: driver.get_exposure_t(),
            common_utils.is_valid_exposure_ms,
            float,
            0.0,
        )

    def _get_frame_size(self) -> dict | None:
        """Get the current camera frame size.

        Returns:
            dict | None: Contains 'width' and 'height' in pixels -- the live
                reading when it succeeds, else the last-known-good value.
                None when no camera is active or the size has never been
                read.
        """
        frame_size = self._validated_camera_read(
            'frame_size',
            lambda driver: driver.get_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
            None,
        )
        return dict(frame_size) if frame_size is not None else None

    def _get_pixel_format(self) -> str | None:
        """Get the current camera pixel format.

        Returns:
            str | None: Pixel format string (e.g. 'Mono8') -- the live
                reading when it succeeds, else the last-known-good value
                (seeded by connect-time configuration and the
                set_pixel_format write-through). None when no camera is
                active or no format is known.
        """
        return self._validated_camera_read(
            'pixel_format',
            lambda driver: driver.get_pixel_format(),
            common_utils.is_valid_pixel_format,
            str,
            None,
        )

    def _get_max_frame_size(self) -> dict | None:
        """Validated sensor-max frame size, or None when never read."""
        return self._validated_camera_read(
            'max_frame_size',
            lambda driver: driver.get_max_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
            None,
        )

    def get_width(self) -> int:
        """Get the current frame width setting.

        Returns:
            int: Current width in pixels -- last-known-good when the live
                read fails. 0 when no camera is active or the size has
                never been read.
        """
        frame_size = self._get_frame_size()
        return int(frame_size['width']) if frame_size else 0

    def get_height(self) -> int:
        """Get the current frame height setting.

        Returns:
            int: Current height in pixels -- last-known-good when the live
                read fails. 0 when no camera is active or the size has
                never been read.
        """
        frame_size = self._get_frame_size()
        return int(frame_size['height']) if frame_size else 0

    def get_binning_size(self) -> int:
        """Get the current camera binning size.

        Returns:
            int: Current binning factor, always >= 1 -- last-known-good when
                the live read fails, 1 when no camera is active. The driver's
                -1 failed-read sentinel never surfaces here: a -1 would flow
                into frame-geometry arithmetic (native size = displayed *
                binning) as a sign flip.
        """
        return self._validated_camera_read(
            'binning',
            lambda driver: driver.get_binning_size(),
            common_utils.is_valid_binning_size,
            int,
            1,
        )

    def get_supported_pixel_formats(self) -> tuple:
        """Get the list of supported camera pixel formats.

        Returns:
            tuple: Supported format strings, or empty tuple if inactive.
        """
        if not self._driver or not self._driver.active:
            return ()
        return self._driver.get_supported_pixel_formats()

    def get_available_binning_sizes(self) -> list:
        """Return list of binning sizes supported by connected camera.

        Returns:
            list: Supported binning factors (e.g. ``[1, 2, 4]``). Defaults
                to ``[1]`` if no camera is active.
        """
        if not self._driver or not self._driver.active:
            return [1]
        try:
            return self._driver.profile.binning_sizes
        except (AttributeError, TypeError):
            return [1]

    def get_native_resolution(self) -> dict:
        """Return the sensor's physical unbinned resolution.

        This is the static per-model ceiling for the native (unbinned) ROI,
        independent of the current binning factor. For the sensor max as the
        driver reports it at boot, see
        ``scope.capabilities.camera_max_frame_size``. Empty dict if no
        camera or the profile does not declare it.

        Returns:
            dict: ``{'width': int, 'height': int}`` or ``{}`` if unknown.
        """
        if not self._driver or not self._driver.active:
            return {}
        try:
            return dict(self._driver.profile.native_resolution)
        except (AttributeError, TypeError):
            return {}

    def get_pixel_alignment(self) -> dict:
        """Return the camera's deliverable frame-size granularity.

        The frame width/height a caller can request, floored to these values, is
        what the camera will actually deliver. For a floor-only driver (Pylon,
        FX2, simulator) this is the hardware AOI grid -- a request off the grid
        is floored down (e.g. multiple-of-4 on most Pylon models). The IDS
        driver instead delivers any even size exactly via oversize-then-crop, so
        it reports ``{2, 2}`` -- the only constraint is even dimensions (H.264).
        Defaults to 4x4 when unknown.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        default = {'width': 4, 'height': 4}
        if not self._driver or not self._driver.active:
            return default
        try:
            return dict(self._driver.profile.alignment)
        except (AttributeError, TypeError):
            return default

    # --- Capture ---
    def _capture_and_wait_impl(
        self,
        force_to_8bit: bool = True,
        *,
        accept_dark: bool = False,
        exclude_sources: tuple = (),
        all_ones_check: bool = False,
        earliest_image_ts: datetime.datetime | None = None,
        timeout_s: float = 0.0,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback=None,
    ) -> np.ndarray | None:
        """Capture a frame guaranteed to reflect the current hardware state.

        Uses frame-based settling: drains stale frames from the camera pipeline
        until frame_validity confirms all pending state changes (LED, gain,
        exposure, motion) have settled. Then grabs a fresh valid frame.

        Frame-based settling automatically adapts to the camera's frame rate --
        fast exposures drain quickly, slow exposures drain slowly, matching
        the actual camera pipeline depth.

        Args:
            force_to_8bit: Convert to 8-bit output.
            exclude_sources: Sources to ignore for validity (e.g. ('z_move',)
                for autofocus where Z motion doesn't need to fully settle).
            all_ones_check: Reject all-max-value frames (camera hardware issue).
            accept_dark: Caller-intent override for the derived dark-floor
                expectation. The capture derives whether illumination is
                commanded ON from the illumination API itself (a channel
                counts as lit only at strictly positive current, so an
                enabled 0 mA channel is dark by design); when a channel is
                lit, a frame with essentially no lit pixel is retried then
                rejected loudly. accept_dark=True admits the dark frame
                anyway -- for callers whose dark frames are legitimate
                while lit: autofocus sweeps (an out-of-focus fluorescence
                plane can carry no signal) and benchmark probes.
            earliest_image_ts: Reject frames captured before this timestamp.
                Forwarded to the final get_image call; complements the
                frame-validity drain for callers that also want a wall-clock
                lower bound on the returned frame.
            timeout_s: Timeout (seconds) for the final get_image call.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.

        Returns:
            numpy.ndarray | None: Captured image array on success, None
                on camera-inactive, frame-drain failure, or deadline
                expiry. Per the Sentinel-return contract:
                `if image is None: ...`.

        The capture honors invalidation across its whole window: hardware
        state changing between the drain's settle and the grab's return is
        detected via the invalidation counters, and the capture re-drains,
        re-derives its expectations (dark floor included), and re-grabs.
        The drain-and-recheck deadline bounds that loop -- a sustained
        invalidation stream produces a loud, distinctly-logged None in
        bounded time instead of an open-ended hold. The deadline clock
        suspends while commanded motion is still physically settling
        (motion has its own authoritative completion signal, and a frame
        budget must not out-vote it), and it governs only the drain and
        re-check loops: the interior of one grab is bounded by
        ``timeout_s`` and, for summed captures, the sum window itself.
        The one state change no polling design can see is a driver write
        whose invalidate has not yet landed when the grab returns; that
        residual is microseconds wide and accepted.
        """
        # Every exit -- including these not-ready sentinels -- records
        # last_capture_info, so a consumer relaying the failure cause is
        # always reading THIS attempt, never a stale dict from the
        # previous capture.
        if not self._driver or not self._driver.active:
            with self._state_lock:
                self._last_capture_info = {'not_ready': True}
            return None

        # active (connected) is necessary but not sufficient to deliver a
        # frame: connect() configures the camera without starting the feed
        # (the lifecycle split), and stop_streaming() deliberately halts it.
        # A not-grabbing camera cannot produce a capture, so return the same
        # not-ready sentinel as the active check above -- named distinctly so
        # a stopped or not-yet-started feed is not misread as a grab timeout.
        if not self._driver.is_grabbing():
            logger.warning(
                '[SCOPE API ] capture_and_wait: no active grab (streaming not '
                'started or stopped) -- returning None until the feed is running.'
            )
            with self._state_lock:
                self._last_capture_info = {'not_ready': True}
            return None

        hold_start = time.monotonic()
        exposure_s = self.get_exposure_ms() / 1000
        grab_timeout_s = max(exposure_s * 3, 1.0)

        # The deadline is frozen at entry: the pending frame count and the
        # exposure are read once and never re-raised, so invalidations
        # arriving after entry extend the work but not the budget. The sum
        # term mirrors the public dispatcher's liveness wait -- a long
        # summed capture legitimately grinds for its whole sum window, and
        # a re-grab repeats that window. Each frame is costed at the larger
        # of the exposure and the conservative frame-period floor: a camera
        # cannot deliver frames faster than its readout, whatever the
        # exposure says.
        n_entry = self.frame_validity.frames_until_valid(exclude_sources=exclude_sources)
        frame_cost_s = max(exposure_s, self._CAPTURE_DEADLINE_MIN_FRAME_PERIOD_S)
        deadline_s = (
            self._CAPTURE_DEADLINE_FLOOR_S
            + n_entry * frame_cost_s * self._CAPTURE_DEADLINE_MARGIN
            + sum_count * (frame_cost_s + sum_delay_s)
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

        def _record_capture_info(**extra) -> None:
            with self._state_lock:
                self._last_capture_info = {
                    'hold_ms': (time.monotonic() - hold_start) * 1000.0,
                    'drained': drain_iterations,
                    'rechecks': recheck_cycles,
                    'deadline_s': deadline_s,
                    'n_entry': n_entry,
                    'active_s': _clock['active'],
                    **extra,
                }

        def _deadline_none(where: str):
            if lock is not None:
                self._resume_auto_gain_impl(lock)
            logger.warning(
                f'[SCOPE API ] capture_and_wait: capture DEADLINE EXPIRED '
                f'({where}) -- active={_clock["active"]:.3f}s > '
                f'deadline_s={deadline_s:.3f}s (n_entry={n_entry}, '
                f'rechecks={recheck_cycles}, '
                f'wall={time.monotonic() - hold_start:.3f}s); invalidation '
                f'outran the capture budget. Distinct from grab failure.'
            )
            _record_capture_info(deadline_expired=True)
            return None

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
                return _deadline_none('recheck-top')
            # Grab plumbing is re-derived each cycle so a mid-window
            # exposure change sizes the NEXT drain and grab for the state
            # that now holds -- a grab timeout sized to a stale short
            # exposure fails healthy long-exposure frames and reports the
            # staleness as a drain failure. Plumbing only: the budget
            # above stays frozen.
            exposure_s = self.get_exposure_ms() / 1000
            grab_timeout_s = max(exposure_s * 3, 1.0)

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

            # Drain stale frames until all pending state changes have
            # settled. Per-frame chunk metadata flows into count_frame so
            # chunks short-circuit skip-frames for chunk-validatable
            # sources (gain, exposure). Cameras without chunks return None
            # and fall back to the existing skip-frames + settle-check
            # path. Each drained grab passes its frame timestamp so a
            # frame concurrently counted by the preview poller is not
            # counted twice.
            while self.frame_validity.frames_until_valid(exclude_sources=exclude_sources) > 0:
                if _deadline_expired():
                    return _deadline_none('drain-loop')
                status, drain_frame_ts = self._driver.grab_new_capture(timeout_s=grab_timeout_s)
                if status:
                    self.frame_validity.count_frame(
                        chunk_data=self._get_latest_chunks(), frame_ts=drain_frame_ts
                    )
                    drain_iterations += 1
                else:
                    remaining = self.frame_validity.frames_until_valid(
                        exclude_sources=exclude_sources
                    )
                    device_removed = (
                        self._driver.is_device_removed()
                        if self._driver and hasattr(self._driver, 'is_device_removed')
                        else None
                    )
                    logger.warning(
                        f'[SCOPE API ] capture_and_wait: frame drain failed -- '
                        f'grab_new_capture returned status=False after '
                        f'{grab_timeout_s:.1f}s timeout '
                        f'(drained={drain_iterations}, frames_until_valid={remaining}, '
                        f'device_removed={device_removed})'
                    )
                    # None is the contract's failure sentinel. A bool here
                    # slips every `is None` caller check: the stills leg
                    # skipped its capture strike -- and reset the accumulated
                    # counter -- on exactly this stalled-feed failure mode.
                    _record_capture_info(drain_failed=True)
                    if lock is not None:
                        self._resume_auto_gain_impl(lock)
                    return None

            if lock is None and self._auto_gain_arm is not None:
                lock = self._lock_auto_gain_impl()
                if lock.state is not None:
                    continue

            # The dark-floor expectation is the API's own fact, derived
            # after the drain settles -- never posted by callers -- so the
            # value cannot drift from commanded state. A re-run of this
            # loop re-derives it, because the state change that dirtied
            # the window is exactly what makes the old derivation stale.
            expected_lit = bool(live_lit_pairs(self._scope.illumination))

            image = self._get_image_impl(
                force_to_8bit=force_to_8bit,
                earliest_image_ts=earliest_image_ts,
                all_ones_check=all_ones_check,
                dark_floor_check=expected_lit and not accept_dark,
                timeout_s=timeout_s,
                sum_count=sum_count,
                sum_delay_s=sum_delay_s,
                sum_iteration_callback=sum_iteration_callback,
                force_new_capture=True,
                new_capture_timeout_s=grab_timeout_s,
                verify_chunk_targets=True,
            )

            # Post-grab compare: a changed counter means the window was
            # dirtied and the frame (or failure) predates the state the
            # caller commanded. The compare runs BEFORE any None
            # propagates: a dirtied-window failure is recoverable -- it
            # was caused by the very change this loop exists to honor --
            # so only a clean-window None falls through as a genuine grab
            # failure. Compare spans both key sets: a source's first-ever
            # invalidation adds a key the snapshot lacks.
            counts_after = {
                s: c
                for s, c in self.frame_validity.invalidation_counts.items()
                if s not in exclude_sources
            }
            if counts_after != counts_before:
                recheck_cycles += 1
                continue
            break

        # Record per-capture evidence for the caller's log line (protocol
        # captures log brightness + the chunk-verified settings per frame so
        # a support bundle shows what each saved frame was exposed with).
        chunks = self._get_latest_chunks() or {}
        extra: dict[str, object] = {}
        if lock is not None and lock.state is not None:
            extra['auto_gain'] = lock.state.value
            extra['auto_gain_exposure_ms'] = lock.exposure_ms
            extra['auto_gain_gain_db'] = lock.gain_db
        if image is None:
            # A clean-window None with a target still mismatching is the
            # gate's rejection; name it so the writer's cause is truthful.
            stale = self._chunk_target_mismatch()
            if stale is not None:
                extra['chunk_rejected'] = stale
        _record_capture_info(
            chunk_exposure_us=chunks.get('ExposureTime'),
            chunk_gain_db=chunks.get('Gain'),
            **extra,
        )
        if lock is not None:
            self._resume_auto_gain_impl(lock)
        return image

    def capture_and_wait(
        self,
        force_to_8bit: bool = True,
        *,
        accept_dark: bool = False,
        exclude_sources: tuple = (),
        all_ones_check: bool = False,
        earliest_image_ts: datetime.datetime | None = None,
        timeout_s: float = 0.0,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback=None,
    ) -> np.ndarray | None:
        """Capture a frame-valid image on the camera worker, and wait for it.

        See ``_capture_and_wait_impl`` for the argument contract, the
        frame-drain behaviour, and the None-on-failure sentinel; this adds
        only the dispatch described on ``_dispatch_camera``. ``timeout_s``
        stays the content-gate retry budget the body reads; the executor
        wait is bounded separately and internally.
        """
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
        # TimeoutError instead of the body's loud, distinctly-logged None.
        frame_cost_s = max(
            self.exposure_ms_cached / 1000.0,
            self._CAPTURE_DEADLINE_MIN_FRAME_PERIOD_S,
        )
        wait_s = (
            self._CAPTURE_WAIT_TIMEOUT_S
            + timeout_s
            + sum_count * (frame_cost_s + sum_delay_s)
            + self.frame_validity.frames_until_valid(exclude_sources=exclude_sources)
            * frame_cost_s
            * self._CAPTURE_DEADLINE_MARGIN
        )
        return self._dispatch_camera(
            self._capture_and_wait_impl,
            'capture_and_wait',
            kwargs={
                'force_to_8bit': force_to_8bit,
                'accept_dark': accept_dark,
                'exclude_sources': exclude_sources,
                'all_ones_check': all_ones_check,
                'earliest_image_ts': earliest_image_ts,
                'timeout_s': timeout_s,
                'sum_count': sum_count,
                'sum_delay_s': sum_delay_s,
                'sum_iteration_callback': sum_iteration_callback,
            },
            timeout_s=wait_s,
        )

    # A frame at least this saturated is treated as blown -- a stale-gain
    # or over-exposure symptom. Set high so a legitimately bright field
    # does not trip it; a true blown-white frame saturates essentially
    # every pixel. Surfaced (warn + notify) rather than saved silently.
    _SATURATION_NEAR_MAX_FRACTION = 0.99  # pixel >= 99% of full scale = saturated
    _SATURATION_BLOWN_FRACTION = 0.98  # >= 98% of pixels saturated = blown frame

    @staticmethod
    def _saturated_fraction(arr: np.ndarray | None, significant_bits: int) -> float:
        """Fraction of pixels at or above the near-full-scale threshold.

        Full scale comes from the frame's payload depth, not the container
        dtype: a 12-bit frame in a uint16 container tops out at 4095, so
        measuring against 65535 would report a fully blown frame as 0%
        saturated and let it slip past the evidence check.
        """
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
        dtype -- the same depth rule as ``_saturated_fraction``.
        """
        if arr is None or arr.size == 0:
            return 0.0
        full_scale = (1 << significant_bits) - 1
        floor = full_scale * ImagingAPI._DARK_FLOOR_FRACTION
        return float(np.count_nonzero(arr > floor)) / arr.size

    def _get_image_impl(
        self,
        force_to_8bit: bool = True,
        earliest_image_ts: datetime.datetime | None = None,
        timeout_s: float = 5.0,
        all_ones_check: bool = False,
        dark_floor_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback: Callable | None = None,
        force_new_capture: bool = False,
        new_capture_timeout_s: float = 5.0,
        verify_chunk_targets: bool = False,
    ) -> np.ndarray | None:
        """Grab and return an image from the camera.

        By default returns the last buffered frame. Set force_new_capture=True
        to trigger a fresh capture. Multiple frames can be summed for noise
        reduction via sum_count.

        This is the ungated primitive: it does NOT wait for frame validity,
        so the returned frame may not reflect a just-changed gain, exposure,
        LED, or stage position. For any frame you will SAVE or MEASURE as
        truth, call capture_and_wait, which drains the camera pipeline until
        the validity marker is GREEN before grabbing.

        Args:
            force_to_8bit: Convert 12-bit images to 8-bit output.
            earliest_image_ts: Reject frames captured before this timestamp.
            timeout_s: Max seconds to wait for a valid frame.
            all_ones_check: Reject saturated (all-max-value) frames.
            dark_floor_check: Reject frames with essentially no pixel above
                the dark floor, retrying until a lit frame arrives or
                timeout_s expires. Internal transport of the value
                capture_and_wait derives from commanded LED state; the
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

        Returns:
            numpy.ndarray | None: Captured image array, or None on failure
                (camera inactive, frame drain failed, timeout exceeded).
                Per the Sentinel-return contract preface in LumascopeSkills:
                `if image is None: ...` to detect failure.

                Shape is (H, W) 2D mono for mono-native cameras and
                (H, W, 3) RGB for color-native cameras. Probe
                `scope.capabilities.is_color_native` to disambiguate.
                This method does NOT apply layer false-color -- apply
                at the display / encode boundary via
                `image_utils.mono_to_rgb_falsecolor(img, layer)`.

                Dtype is uint8 when force_to_8bit=True or for 8-bit
                cameras; uint16 when force_to_8bit=False for 12/16-bit
                cameras (uint16 container holds the native bit width).
                Probe `scope.capabilities.native_bit_depth` for the
                source depth.
        """

        if not self._driver or not self._driver.active:
            return None

        tmp_buffer = []
        timeout_td = datetime.timedelta(seconds=timeout_s)
        for _ in range(sum_count):
            start_time = datetime.datetime.now()
            stop_time = start_time + timeout_td

            while True:
                # Acquire cam_lock for camera grab -- prevents concurrent
                # set_gain_db/set_exposure from another thread mid-frame.
                with self._cam_lock:
                    if force_new_capture:
                        grab_status, grab_image_ts = self._driver.grab_new_capture(
                            new_capture_timeout_s
                        )
                    else:
                        grab_status, grab_image_ts = self._driver.grab()

                    if grab_status:
                        self.frame_validity.count_frame(
                            chunk_data=self._get_latest_chunks(), frame_ts=grab_image_ts
                        )
                        tmp = self._driver.get_array()  # thread-safe copy

                if not grab_status:
                    # Check if camera disconnected -- don't retry for 5 seconds
                    # if the camera is gone (H20).
                    if not self._driver.active:
                        logger.error('[SCOPE API ] get_image: camera disconnected')
                        from modules.notification_center import notifications

                        notifications.error(
                            'Camera',
                            'Camera Disconnected',
                            'Camera is no longer available. Check USB connection.',
                        )
                        return None
                    if datetime.datetime.now() > stop_time:
                        logger.error(f'[SCOPE API ] get_image timeout ({stop_time}) exceeded')
                        return None
                    logger.debug('[SCOPE API ] get_image grab failed, retrying')
                    time.sleep(0.05)
                    continue

                # Saturation measures against the just-grabbed frame's own
                # payload depth (the delivery stamp), never the container
                # dtype -- a 12-bit frame in a uint16 container is blown at
                # 4095, not 65535. Read only on the checking path: the stamp
                # read takes the handler's frame lock, which the streaming
                # thread contends on every delivery.
                if all_ones_check:
                    frame_depth = self.last_significant_bits
                if (
                    all_ones_check
                    and self._saturated_fraction(tmp, frame_depth)
                    >= self._SATURATION_BLOWN_FRACTION
                ):
                    # Near-fully-saturated frame -- retry once in case it was a
                    # transient blip, then surface it. A blown frame is usually
                    # an over-exposure or stale-camera-gain symptom; accepting it
                    # silently (the prior behavior) hid real data corruption.
                    retry_frame = None
                    with self._cam_lock:
                        retry_status, retry_image_ts = (
                            self._driver.grab_new_capture(new_capture_timeout_s)
                            if force_new_capture
                            else self._driver.grab()
                        )
                        if retry_status:
                            self.frame_validity.count_frame(
                                chunk_data=self._get_latest_chunks(), frame_ts=retry_image_ts
                            )
                            retry_frame = self._driver.get_array()
                    # Saturation walk is outside cam_lock -- no camera state needed,
                    # and the walk would otherwise block concurrent set_gain_db/set_exposure.
                    if retry_frame is not None and (
                        self._saturated_fraction(retry_frame, self.last_significant_bits)
                        < self._SATURATION_BLOWN_FRACTION
                    ):
                        tmp = retry_frame  # retry was clean, use it
                    else:
                        # Log (not notify): a blown frame is self-evident on
                        # screen and in the saved file, so a popup adds nothing.
                        # The log line is for the post-mortem / log-analysis pass.
                        sat_pct = self._saturated_fraction(tmp, frame_depth) * 100.0
                        logger.warning(
                            f'[SCOPE API ] get_image: captured frame is {sat_pct:.0f}% '
                            f'saturated -- likely over-exposure or a stale camera gain; '
                            f'the frame may be unusable.'
                        )

                if dark_floor_check:
                    lit_fraction = self._lit_fraction(tmp, self.last_significant_bits)
                    if lit_fraction < self._DARK_MIN_LIT_FRACTION:
                        # The caller declared illumination ON, yet no pixel
                        # clears the floor: the frame integrated before the
                        # LED lit, or the camera is delivering black frames.
                        # Retry (the next frame usually integrates under the
                        # lit LED), then reject loudly -- a black file must
                        # become either a good file or a named failure, never
                        # a silent save.
                        if datetime.datetime.now() > stop_time:
                            logger.warning(
                                f'[SCOPE API ] get_image: frame is dark -- '
                                f'{lit_fraction:.6f} of pixels above '
                                f'{self._DARK_FLOOR_FRACTION:.0%} of full scale '
                                f'(minimum {self._DARK_MIN_LIT_FRACTION}) with '
                                f'illumination expected ON; no lit frame within '
                                f'{timeout_s:.1f}s. Capture rejected.'
                            )
                            return None
                        logger.debug(
                            '[SCOPE API ] get_image: rejecting dark frame; waiting for a lit frame'
                        )
                        if not force_new_capture:
                            # Buffered grabs return the same frame until a new
                            # one arrives; pace the retry instead of spinning.
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
                    stale_source = self._chunk_target_mismatch()
                    if stale_source is not None:
                        if datetime.datetime.now() > stop_time:
                            chunks = self._get_latest_chunks() or {}
                            chunk_key = self.frame_validity.CHUNK_KEY_FOR_SOURCE.get(stale_source)
                            logger.warning(
                                f'[SCOPE API ] get_image: frame chunk for '
                                f'{stale_source} never matched the requested '
                                f'target within {timeout_s:.1f}s '
                                f'(chunk={chunks.get(chunk_key)}, '
                                f'target={self.frame_validity.target(stale_source)}) -- '
                                f'either the camera is still delivering frames '
                                f'exposed under the previous settings, or it '
                                f'clamped the requested value. Capture rejected.'
                            )
                            return None
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

                # Accept the frame
                if earliest_image_ts is None:
                    tmp_buffer.append(tmp)
                    break

                if grab_image_ts > earliest_image_ts:
                    tmp_buffer.append(tmp)
                    break

                logger.warning(
                    f'[SCOPE API ] get_image earliest_image_time {earliest_image_ts} not met -> Image TS: {grab_image_ts}'
                )

                # Timestamp not met -- check timeout then retry
                if datetime.datetime.now() > stop_time:
                    logger.error(f'[SCOPE API ] get_image timeout ({stop_time}) exceeded')
                    return None
                time.sleep(0.05)

            if sum_count > 1:
                earliest_image_ts = grab_image_ts + datetime.timedelta(milliseconds=1)
                if sum_iteration_callback is not None:
                    sum_iteration_callback()

                time.sleep(sum_delay_s)

        # Chain via a local variable instead of self.image_buffer. The old
        # field was a permanent shadow copy of the latest get_image result,
        # only ever read by get_image itself -- shadow state that pinned a
        # frame indefinitely between calls. The _state_lock around per-write
        # didn't actually serialize concurrent get_image calls anyway (chained
        # writes from different threads could still interleave).
        if sum_count == 1:
            image = tmp if len(tmp_buffer) < 1 else tmp_buffer[0]
        else:
            orig_dtype = tmp_buffer[0].dtype
            max_value = np.iinfo(orig_dtype).max

            combined = np.zeros_like(tmp_buffer[0], dtype=np.uint32)
            for img in tmp_buffer:
                combined += img

            image = np.clip(combined, None, max_value).astype(orig_dtype)

        # One snapshot for the whole overlay decision: enabled and color must
        # come from the same configuration even if the GUI toggles mid-frame.
        scale_bar = self.scale_bar_config
        objective = self._scope.runtime_state.get_current_objective()
        use_scale_bar = self._resolve_use_scale_bar(scale_bar['enabled'], objective)

        need_8bit = force_to_8bit and image.dtype != np.uint8

        # A summed capture lives in a 16-bit container; a single frame carries
        # the camera's native payload depth. The scale bar's white value and the
        # 8-bit downconvert divisor both follow this depth so a summed 12-bit
        # value never indexes the 12-bit display table, a 10-bit frame is not
        # crushed as if 12-bit, and the bar maps to full white not a dim gray.
        # Query the driver only when a consumer needs it -- a raw passthrough
        # frame returns without touching the driver's depth.
        if use_scale_bar or need_8bit:
            significant_bits = self.capture_frame_depth(image, sum_count)

        if use_scale_bar:
            image = image_utils.add_scale_bar(
                image=image,
                objective=objective,
                binning_size=self._binning_size,
                color=scale_bar.get('color'),
                significant_bits=significant_bits,
                capabilities=self._scope.capabilities,
            )

        if need_8bit:
            image = image_utils.convert_to_8bit(image, significant_bits)

        return image

    def get_image(
        self,
        force_to_8bit: bool = True,
        earliest_image_ts: datetime.datetime | None = None,
        timeout_s: float = 5.0,
        all_ones_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback: Callable | None = None,
        force_new_capture: bool = False,
        new_capture_timeout_s: float = 5.0,
        verify_chunk_targets: bool = False,
    ) -> np.ndarray | None:
        """Grab and return an image from the camera -- the UNGATED primitive.

        Never dark-rejects: liveness probes and diagnostics must see what
        the camera sees, dark or not. For any frame you will SAVE or
        MEASURE as truth, call capture_and_wait, which drains the camera
        pipeline until the validity marker is GREEN and derives the
        dark-floor expectation from commanded LED state.

        Explicit parameters, never a **kwargs forward: the signature is a
        pinned contract (introspected by tests), and a silent kwarg
        pass-through would re-open the door this split closed.

        See ``_get_image_impl`` for the full argument contract.
        """
        return self._get_image_impl(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            timeout_s=timeout_s,
            all_ones_check=all_ones_check,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
            force_new_capture=force_new_capture,
            new_capture_timeout_s=new_capture_timeout_s,
            verify_chunk_targets=verify_chunk_targets,
        )

    def get_image_from_buffer(
        self, force_to_8bit: bool = True, out_8bit: np.ndarray | None = None
    ) -> tuple:
        """Grab the latest buffered frame from the camera without forcing a new capture.

        Copy budget (per frame):
          - grab_latest(): 0 copies (returns reference from ImageHandler)
          - add_scale_bar(): 0 copies (modifies array in-place)
          - convert_to_8bit(): 1 copy (LUT indexing creates new array),
            or 0 fresh allocations when the caller supplies out_8bit.
          - Total: 0 copies (8-bit) or 1 copy (12-bit with force_to_8bit)
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
        """
        if not self._driver or not self._driver.active:
            return None, None

        # Single-copy grab: grab_latest() returns the image directly,
        # avoiding the extra copy that grab() + get_array() would make.
        # This saves ~2.3MB copy + 1 lock acquisition per frame.
        grab_status, tmp, grab_image_ts, frame_significant_bits = self._driver.grab_latest()
        if not grab_status or tmp is None:
            return None, None
        # grab_latest() returns the same buffered frame on every poll, and
        # this preview path can poll faster than the camera delivers. The
        # frame timestamp dedupes the count so validity skip counts expire
        # against real frames, not poll rate -- counting polls let a capture
        # accept a frame exposed under the previous gain/exposure/LED state.
        self.frame_validity.count_frame(
            chunk_data=self._get_latest_chunks(), frame_ts=grab_image_ts
        )

        with self._state_lock:
            self._frame_buffer = tmp

        # Snapshot both fields together; see get_image for why.
        scale_bar = self.scale_bar_config
        objective = self._scope.runtime_state.get_current_objective()
        use_scale_bar = self._resolve_use_scale_bar(scale_bar['enabled'], objective)

        if use_scale_bar:
            tmp = image_utils.add_scale_bar(
                image=tmp,
                objective=objective,
                binning_size=self._binning_size,
                color=scale_bar.get('color'),
                significant_bits=frame_significant_bits,
                capabilities=self._scope.capabilities,
            )

        if force_to_8bit and tmp.dtype != np.uint8:
            tmp = image_utils.convert_to_8bit(tmp, frame_significant_bits, out=out_8bit)

        return tmp, grab_image_ts

    @property
    def significant_bits(self) -> int:
        """Meaningful payload bits of frames the current camera delivers.

        The depth a single captured frame should be scaled / tagged by (12 for a
        Mono12 sensor, 8 for an 8-bit one). A summed frame is promoted to a
        16-bit container by get_image and is not described by this -- summed
        callers declare 16 themselves. Falls back to the container width when no
        camera is attached.

        Derived from the CACHED pixel format (the validated last-known-good)
        via the driver's own depth rule (``significant_bits_for_format``, so
        a constant-depth driver like FX2 stays authoritative), not a live
        driver read: a transient format-read failure once made a 12-bit
        camera report depth 16 here, so a fully blown frame passed the
        saturation evidence check at 0.0% and files were stamped at the
        wrong depth. For the depth of a frame you are HOLDING, prefer
        ``last_significant_bits`` -- the per-frame stamp.
        """
        driver = self._driver
        if driver is None:
            return 16
        return driver.significant_bits_for_format(self._get_pixel_format())

    @property
    def last_significant_bits(self) -> int:
        """Payload depth of the most recently delivered frame (per-frame stamp).

        The depth to save, evidence-check, or downconvert a just-captured
        frame at: every driver stamps the depth WITH the frame at delivery,
        so it cannot fail a live read and always describes a frame the
        camera actually produced. Read it as closely as possible to the
        grab that produced the frame you hold -- a later read can describe
        a newer frame. The stamp is an in-memory read (no SDK node touch),
        so it has no transient-failure mode. Falls back to
        ``significant_bits`` (the validated cached-format depth -- NOT the
        driver's live-read fallback, which a transient format failure
        could turn into a wrong container-width answer) when no frame has
        been stored yet; 16 when no camera is attached.
        """
        driver = self._driver
        if driver is None:
            return 16
        stamped = driver.last_stamped_significant_bits()
        return int(stamped) if stamped is not None else self.significant_bits

    def capture_frame_depth(self, array: np.ndarray | None, sum_count: int = 1) -> int:
        """Payload depth of a frame just produced by a capture call.

        The one depth-classification rule every save / evidence / display
        consumer shares: an 8-bit container carries 8 significant bits, a
        summed capture fills its promoted 16-bit container, and a single
        wider frame carries the per-frame delivery stamp. Read it at
        capture time, next to the grab that produced ``array``, and hand
        it DOWN with the frame -- re-deriving depth later reads the
        camera's state at that later moment, not the frame's.
        """
        if array is not None and getattr(array, 'dtype', None) == np.uint8:
            return 8
        if sum_count > 1:
            return 16
        return self.last_significant_bits

    # --- Streaming control ---
    def start_streaming(self) -> None:
        """Begin camera streaming -- the public way to start the live feed.

        After ``connect()`` the camera is configured but NOT grabbing (the
        camera-lifecycle split); this is the sanctioned release. Opens the
        start gate (idempotent) and ensures the grab is running, so it both
        performs the one-time bring-up start and restarts a feed that was
        deliberately stopped. No-op when no camera is attached.

        The UI bring-up calls this in load_settings / reconnect; headless
        callers (scripts, tests) call it after constructing the scope
        instead of reaching into the private camera driver.
        """
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

    def stop_streaming(self) -> None:
        """Stop camera streaming.

        After this, ``get_image()`` / ``capture_and_wait()`` time out until
        streaming resumes. No-op when no camera is attached.
        """
        driver = self._driver
        if driver is None:
            return
        driver.stop_grabbing()

    def is_streaming(self) -> bool:
        """Whether the camera is currently acquiring frames.

        Queries the driver directly (unlike ``active_cached``, which reads
        the cached connected-state). False when no camera is attached.
        """
        driver = self._driver
        if driver is None:
            return False
        return driver.is_grabbing()

    # --- State / lifecycle properties ---
    @property
    def active_cached(self) -> bool:
        """Whether the camera is connected and active (reads cache).

        Returns:
            bool: True if the camera is currently active.
        """
        with self._camera_cache_lock:
            return self._camera_cache['active']

    @property
    def gain_db_cached(self) -> float:
        """Current camera gain in dB (reads cache).

        Returns:
            float: Cached gain value in dB.
        """
        with self._camera_cache_lock:
            return self._camera_cache['gain_db']

    @property
    def exposure_ms_cached(self) -> float:
        """Current camera exposure time in ms (reads cache).

        Returns:
            float: Cached exposure time in milliseconds.
        """
        with self._camera_cache_lock:
            return self._camera_cache['exposure_ms']

    @property
    def frame_size_cached(self) -> dict:
        """Current camera frame size as {'width': int, 'height': int} (reads cache).

        Returns:
            dict: Copy of the cached frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['frame_size'])

    @property
    def camera_identity(self) -> dict:
        """Connected camera's identity for provenance records.

        Returns:
            dict: ``{'model': str | None, 'serial': str | None,
            'timestamp_tick_frequency_hz': float | None}``. All None when
            no camera is connected -- callers record the absence rather
            than probe drivers directly.
        """
        driver = self._driver
        if not driver or not driver.active:
            return {'model': None, 'serial': None, 'timestamp_tick_frequency_hz': None}
        return {
            'model': getattr(driver, 'model_name', None),
            'serial': getattr(driver, '_device_serial', None),
            'timestamp_tick_frequency_hz': getattr(driver, 'timestamp_tick_frequency_hz', None),
        }

    @property
    def min_frame_size_cached(self) -> dict | None:
        """Minimum camera frame size, or None if no camera is connected.

        Returns None (not a zero-sized dict) so callers can distinguish
        "camera missing" from a real driver value -- the same contract as
        its siblings max_exposure_ms_cached and max_gain_db_cached.

        Returns:
            dict | None: Copy of the cached min frame size dict, or None
                if unavailable.
        """
        with self._camera_cache_lock:
            value = dict(self._camera_cache['min_frame_size'])
        if value.get('width', 0) <= 0 or value.get('height', 0) <= 0:
            return None
        return value

    @property
    def max_exposure_ms_cached(self) -> float | None:
        """Maximum camera exposure time in ms, or None if no camera is connected.

        Returns None (not a sentinel 0.0) so callers can distinguish
        "camera missing" from a real driver value. See #616.

        Returns:
            float | None: Max exposure time in ms, or None if unavailable.
        """
        with self._camera_cache_lock:
            value = self._camera_cache.get('max_exposure_ms')
        if not value or value <= 0:
            return None
        return float(value)

    @property
    def max_gain_db_cached(self) -> float | None:
        """Maximum camera gain in dB, or None if no camera is connected.

        Parallel to max_exposure_ms_cached -- lets the UI size the gain
        slider to the connected camera's profile-declared cap instead
        of a universal hardcoded 48 dB that can drive the image past
        the sensor's usable range (observed on LS620 2026-04-16).

        Returns:
            float | None: Max gain in dB, or None if unavailable.
        """
        with self._camera_cache_lock:
            value = self._camera_cache.get('max_gain_db')
        if value is None or value <= 0:
            return None
        return float(value)

    @property
    def pixel_format_cached(self) -> str | None:
        """Current camera pixel format (e.g. 'Mono8', 'Mono12') (reads cache).

        Returns:
            str | None: Cached pixel format string, or None when no format
                has ever been successfully read or configured. A failed
                driver read never overwrites this entry
                (validate-before-store at cache population), so a known
                format survives transient read failures.
        """
        with self._camera_cache_lock:
            return self._camera_cache['pixel_format']

    # --- Save / restore ---
    def save_camera_state(self, tag: str) -> dict:
        """Snapshot the current camera gain and exposure for later restoration.

        Omit-if-unknown: a field enters the snapshot only when a usable
        value exists (the getters answer last-known-good, so a missing
        field means the value was NEVER successfully read). Restore can
        therefore trust every field it finds, and name the ones it
        cannot restore.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            dict: Snapshot suitable for passing to ``restore_camera_state``.
        """
        snapshot = {'tag': tag}
        gain_db = self.get_gain_db()
        if common_utils.is_valid_gain_db(gain_db):
            snapshot['gain_db'] = gain_db
        else:
            # The warning belongs HERE, not at restore: an absent field at
            # restore time can be a deliberate caller trim (autofocus keeps
            # the fields its run explicitly targeted), but at save time a
            # missing value has exactly one meaning -- the camera never
            # reported it -- and a later partial restore would otherwise be
            # untraceable.
            logger.warning(
                f'[SCOPE API ] save_camera_state tag={tag}: gain has never '
                f'been successfully read; snapshot omits it and the restore '
                f'will leave gain unchanged'
            )
        exposure_ms = self.get_exposure_ms()
        if common_utils.is_valid_exposure_ms(exposure_ms):
            snapshot['exposure_ms'] = exposure_ms
        else:
            logger.warning(
                f'[SCOPE API ] save_camera_state tag={tag}: exposure has '
                f'never been successfully read; snapshot omits it and the '
                f'restore will leave exposure unchanged'
            )
        _api_log.info(
            f'save_camera_state tag={tag}: '
            f'gain={snapshot.get("gain_db", "never-read")} '
            f'exp={snapshot.get("exposure_ms", "never-read")}'
        )
        return snapshot

    def restore_camera_state(self, snapshot: dict) -> None:
        """Restore camera gain and exposure from a previously saved state.

        Fields absent from the snapshot are skipped and named in the log:
        either the caller deliberately trimmed them (autofocus keeps the
        values its run explicitly targeted) or they were never readable at
        save time -- save_camera_state already WARNed about the latter.

        Args:
            snapshot: Return value from ``save_camera_state``.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        # An ABSENT field is a legitimate trim (skip quietly); a PRESENT
        # field that fails validation is a caller bug -- the sanctioned
        # producer only ever emits valid fields -- so that case warns.
        gain_db = snapshot.get('gain_db')
        gain_known = common_utils.is_valid_gain_db(gain_db)
        if not gain_known and 'gain_db' in snapshot:
            logger.warning(
                f'[SCOPE API ] restore_camera_state tag={tag}: snapshot '
                f'carries a non-physical gain ({gain_db!r}); gain left as-is'
            )
        exposure_ms = snapshot.get('exposure_ms')
        exposure_known = common_utils.is_valid_exposure_ms(exposure_ms)
        if not exposure_known and 'exposure_ms' in snapshot:
            logger.warning(
                f'[SCOPE API ] restore_camera_state tag={tag}: snapshot '
                f'carries a non-physical exposure ({exposure_ms!r}); '
                f'exposure left as-is'
            )
        # The log of record goes BEFORE the hardware writes: a setter may
        # raise a typed exception mid-restore, and the post-mortem then
        # needs the line stating what this restore was about to do (a
        # partial restore with no record once misattributed wrong-
        # brightness images to protocol settings).
        _api_log.info(
            f'restore_camera_state tag={tag}: '
            f'gain={gain_db if gain_known else "skipped"} '
            f'exp={exposure_ms if exposure_known else "skipped"}'
        )
        if gain_known:
            self._set_gain_db_impl(gain_db)
        if exposure_known:
            self._set_exposure_ms_impl(exposure_ms)

    # --- Camera config orchestration ---
    def apply_layer_camera_settings(
        self,
        gain_db: float,
        exposure_ms: float,
        auto_gain: bool = False,
        auto_gain_settings: dict | None = None,
        resume_after_capture: bool = True,
    ) -> None:
        """Apply per-layer camera settings in one batched call, and wait.

        See ``_apply_layer_camera_settings_impl`` for the contract; this
        adds only the dispatch described on ``_dispatch_camera``. The
        batch is one dispatched task, so the three writes stay atomic on
        the camera lane.
        """
        return self._dispatch_camera(
            self._apply_layer_camera_settings_impl,
            'apply_layer_camera_settings',
            args=(gain_db, exposure_ms),
            kwargs={
                'auto_gain': auto_gain,
                'auto_gain_settings': auto_gain_settings,
                'resume_after_capture': resume_after_capture,
            },
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _apply_layer_camera_settings_impl(
        self,
        gain_db: float,
        exposure_ms: float,
        auto_gain: bool = False,
        auto_gain_settings: dict | None = None,
        resume_after_capture: bool = True,
    ) -> None:
        """Apply per-layer camera settings in a single batched call.

        Sets gain, exposure, and auto-gain state. Replaces 3 separate
        IOTask queues with a single call for atomicity.

        Args:
            gain_db: Camera gain in dB.
            exposure_ms: Exposure time in milliseconds.
            auto_gain: Whether auto-gain is enabled for this layer.
            auto_gain_settings: Dict with target_brightness, min_gain_db, max_gain_db
                               (required if auto_gain is True).
        """
        if not self._driver or not self._driver.active:
            self._notify_camera_absent('gain / exposure')
            return
        self._set_gain_db_impl(gain_db)
        self._set_exposure_ms_impl(exposure_ms)
        if auto_gain_settings is not None:
            self._set_auto_gain_impl(
                auto_gain, settings=auto_gain_settings, resume_after_capture=resume_after_capture
            )
        _api_log.info(
            f'apply_layer_camera_settings gain={gain_db}dB exp={exposure_ms}ms auto_gain={auto_gain}'
        )

    def update_auto_gain_target_brightness(self, target_brightness: float) -> None:
        """Set the auto-gain target brightness, and wait for it.

        See ``_update_auto_gain_target_brightness_impl`` for the settle
        contract; this adds only the dispatch described on
        ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._update_auto_gain_target_brightness_impl,
            'update_auto_gain_target_brightness',
            args=(target_brightness,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _update_auto_gain_target_brightness_impl(self, target_brightness: float) -> None:
        """Set the auto-gain target brightness on the camera.

        Args:
            target_brightness: Target brightness value (0.0 to 1.0).
        """
        if not self._driver or not self._driver.active:
            return
        # Changing the target re-drives the auto-gain loop: gain (and, under
        # auto-exposure, exposure) converge to a new operating point, so a frame
        # grabbed before they resettle is captured at the old brightness. Route
        # through the sanctioned write path and mark the settle sources RED so
        # capture waits for the convergence -- the same sources set_auto_gain
        # arms, since this is the same convergence. The auto_gain settle source
        # applies only when the camera has hardware auto-gain (others settle via
        # the gain source alone).
        arm_settle = getattr(self._driver.profile, 'has_auto_gain', False)
        self._camera_write(
            lambda: self._driver.update_auto_gain_target_brightness(target_brightness),
            force_invalidate=('gain', 'auto_gain') if arm_settle else ('gain',),
        )

    def auto_gain_once(
        self,
        state: bool,
        target_brightness: float,
        min_gain_db: float,
        max_gain_db: float,
        ae_max_exposure_ms: float | None = None,
    ) -> None:
        """Run one-shot auto-gain, and wait for it.

        See ``_auto_gain_once_impl`` for the settle contract; this adds
        only the dispatch described on ``_dispatch_camera``.
        """
        return self._dispatch_camera(
            self._auto_gain_once_impl,
            'auto_gain_once',
            args=(state, target_brightness, min_gain_db, max_gain_db),
            kwargs={'ae_max_exposure_ms': ae_max_exposure_ms},
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _auto_gain_once_impl(
        self,
        state: bool,
        target_brightness: float,
        min_gain_db: float,
        max_gain_db: float,
        ae_max_exposure_ms: float | None = None,
    ) -> None:
        """Run auto-gain for a single frame on the camera.

        Args:
            state: True to enable one-shot auto-gain.
            target_brightness: Target brightness (0.0 to 1.0).
            min_gain_db: Minimum gain in dB.
            max_gain_db: Maximum gain in dB.
            ae_max_exposure_ms: Optional per-channel-class upper bound (ms)
                on the exposure auto-exposure may drive to.
        """
        if not self._driver or not self._driver.active:
            return
        # One-shot AG changes both gain and exposure on the camera from a single
        # driver call; the pipeline still needs frames to flush the converged
        # values, so both validity markers must go RED until they settle. The
        # converged values are chosen by the SDK, so clear any manual chunk-match
        # target and fall back to skip-frames settling. Both invalidations and
        # target-clears are forced -- one call, two sources, no value write the
        # authority could observe.
        self._camera_write(
            lambda: self._driver.auto_gain_once(
                state=state,
                target_brightness=target_brightness,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                ae_max_exposure_ms=ae_max_exposure_ms,
            ),
            force_invalidate=('gain', 'exposure'),
            force_clear=('gain', 'exposure'),
        )
        with self._state_lock:
            self._auto_gain_arm = None
        # One-shot AG always ends with the auto cycle complete and the
        # SDK toggled back to Off internally; hardware holds the
        # converged value while LVP's cache is still pre-auto.
        self._refresh_cache_from_hardware_after_auto()

    @contextlib.contextmanager
    def suppress_value_warnings(self) -> Iterator[None]:
        """Suppress programmatic value-range warnings (sub-0.1ms exposure
        and similar) for the duration of the `with` block.

        Used by sweep-style internal callers (camera characterization
        dynamic_range / linearity stages) that walk the full setting
        range deliberately. The warnings exist for L1 researchers who
        type microsecond values thinking ms; they're noise when the
        char tool is exercising the API as designed.

        Restores the prior flag value (not unconditionally False) so
        nested `with` blocks behave correctly. Restoration runs on
        exception too -- exiting an exception-aborted char run leaves
        the API in a clean state for the next user action.
        """
        prior = self._suppress_value_warnings
        self._suppress_value_warnings = True
        try:
            yield
        finally:
            self._suppress_value_warnings = prior

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
    @property
    def frame_is_valid(self) -> bool:
        """True if all pending hardware state changes have settled.

        ``frame_validity`` is the SSOT (see modules/frame_validity.py).

        Returns:
            bool: True when no pending state changes are outstanding.
        """
        return self.frame_validity.is_valid

    def frames_until_valid(self, exclude_sources: tuple = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Delegates to frame_validity.

        Args:
            exclude_sources: Sources to exclude from the validity check.

        Returns:
            int: Number of additional frames to drain before validity. 0 if
                already valid.
        """
        return self.frame_validity.frames_until_valid(
            exclude_sources=exclude_sources,
        )

    @property
    def last_capture_info(self) -> dict | None:
        """Evidence about the most recent capture_and_wait on this scope.

        Consumed by the protocol file writer and by L2 callers that need
        the auto-gain outcome of a capture.

        Returns:
            dict | None: ``{'hold_ms', 'drained', 'chunk_exposure_us',
                'chunk_gain_db'}`` for the latest capture, or None before
                the first capture. Chunk values are None on cameras
                without chunk support. When the capture locked a standing
                auto-gain arm it also carries ``'auto_gain'`` (one of
                ``AutoGainConvergence``'s values: CONVERGED / MAXED /
                AT_MINIMUM / FAILED), ``'auto_gain_exposure_ms'`` and
                ``'auto_gain_gain_db'`` (the locked values, None on
                FAILED). A capture the chunk gate rejected carries
                ``'chunk_rejected'`` naming the source.
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
                '[SCOPE API ] Scale bar is enabled but no objective is '
                'selected; skipping the bar until an objective is set.'
            )
        return False

    @property
    def scale_bar_config(self) -> dict:
        """Return a snapshot of scale bar settings.

        The one read for this state: a defensive copy of the whole
        ``{'enabled', 'color', ...}`` configuration, so a caller reading more
        than one field sees a single consistent setting rather than fields
        from either side of a concurrent ``set_scale_bar``.

        Returns:
            dict: Copy of the scale bar config (e.g. enabled, color).
        """
        with self._state_lock:
            return dict(self._scale_bar)

    def set_scale_bar(self, enabled: bool, color: str | None = None) -> None:
        """Configure the scale bar overlay on captured images.

        Args:
            enabled: Whether to draw the scale bar.
            color: Scale bar color (e.g. "white"). Uses default if None.
        """
        # One critical section for both fields: the capture path reads
        # enabled and color as a pair, and a toggle landing between two
        # separate writes would draw the bar in the previous colour.
        with self._state_lock:
            self._scale_bar['enabled'] = enabled
            if color is not None:
                self._scale_bar['color'] = color

    # --- Camera diagnostics (live in-flight only; data source = DiagnosticsAPI) ---
    def _log_camera_temps(self) -> None:
        """Emit one INFO line per camera temperature sensor.

        No-op when no camera is connected. Called once on startup and
        periodically by ``start_camera_temp_logging``. Reads temperatures
        through `scope.diagnostics.get_camera_temperatures_degc` -- the canonical
        camera-temp probe (cold probes live on DiagnosticsAPI).
        """
        if not self._scope.camera_connected:
            return
        for source, temp in self._scope.diagnostics.get_camera_temperatures_degc().items():
            logger.info(f'[CAM Class ] Camera {source} Temperature : {temp:.2f} degC')

    def start_camera_temp_logging(
        self, schedule_interval_fn, unschedule_fn, *, interval_s: float = 14400.0
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
        self._log_camera_temps()  # one immediate sample

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

    # --- Frame-flow listeners ---
    def add_camera_listener(self, listener) -> None:
        """Register a callback for camera setting changes.

        The listener is called with ``(param, value)`` whenever camera
        gain or exposure changes.  *param* is ``'gain'`` or ``'exposure'``.
        It fires from the thread that caused the change, so listeners
        **must** schedule UI work via ``Clock.schedule_once``.

        Note: this fires on set_gain_db/set_exposure_ms (user actions),
        NOT on every camera frame grab -- zero overhead on display framerate.

        Args:
            listener: ``callable(param: str, value: float)``
        """
        with self._camera_listeners_lock:
            self._camera_listeners.append(listener)

    def remove_camera_listener(self, listener) -> None:
        """Unregister a camera listener.

        Args:
            listener: A callable previously passed to
                ``add_camera_listener``. Silently ignores listeners that
                are not currently registered.
        """
        with self._camera_listeners_lock:
            try:
                self._camera_listeners.remove(listener)
            except ValueError:
                pass

    def add_frame_listener(self, cb, name: str | None = None) -> None:
        """Register a per-frame listener fired on every successful grab.

        The canonical entry point for live_processing plugins (see
        ``ctx.plugins.live_processing``) and the manual-record path.
        The supplied handler is wrapped in a budget enforcer
        (``HANDLER_BUDGET_MS`` per call; ``HANDLER_DROP_K`` consecutive
        over-budget invocations triggers auto-removal). Callback
        signature is ``cb(image, timestamp, chunks)``; runs on the SDK
        callback thread (Pylon ``PylonImageGrab`` / IDS grab loop /
        simulated pump). Listeners MUST NOT block -- heavy work belongs
        on an executor. No-op when no camera is connected.

        Args:
            cb: Per-frame handler. Signature ``cb(image, timestamp, chunks)``.
            name: Display name for log + notification messages on
                  over-budget / auto-removal. Defaults to the handler's
                  qualname.

        The ``image`` array is shared across all listeners (don't-mutate
        contract); write to your own output buffer if you need to keep
        results. Mutating the supplied array affects later listeners
        plus downstream display / capture consumers.

        Registration is idempotent for the same callable -- a second
        call with the same ``cb`` is a no-op (the original wrapper +
        name are kept).
        """
        if not self._driver or not self._driver.active:
            return
        if name is None:
            name = getattr(cb, '__qualname__', None) or repr(cb)
        with self._frame_listener_lock:
            if cb in self._frame_listener_wrappers:
                return  # idempotent
            wrapper = _BudgetedHandler(self, cb, name)
            self._frame_listener_wrappers[cb] = wrapper
        try:
            self._driver.register_frame_callback(wrapper)
        except Exception as ex:
            # Rollback the dict entry if the driver registration
            # failed so a future register attempt can retry.
            with self._frame_listener_lock:
                self._frame_listener_wrappers.pop(cb, None)
            logger.exception(f"[SCOPE API ] add_frame_listener failed for '{name}': {ex}")
            # Driver-side registration failed -- the listener will
            # never fire. Surface to the user so a plugin author
            # whose frame handler quietly stopped receiving frames
            # has a signal to investigate, instead of seeing no
            # data and no error.
            notifications.warning(
                'Frame Listener',
                f"Listener '{name}' failed to register",
                'The camera driver rejected the frame-listener '
                'registration. The handler will not receive frames. '
                'Restart the application; if the failure repeats, '
                'check the log for the underlying driver error.',
            )

    def remove_frame_listener(self, cb) -> None:
        """Remove a listener registered via ``add_frame_listener``.

        No-op when no camera is connected or the listener was never
        registered. The user supplies the original handler; this
        method looks up the wrapper and unregisters that.
        """
        if not self._driver:
            return
        with self._frame_listener_lock:
            wrapper = self._frame_listener_wrappers.pop(cb, None)
        if wrapper is None:
            return
        try:
            self._driver.unregister_frame_callback(wrapper)
        except Exception as ex:
            logger.exception(f'[SCOPE API ] remove_frame_listener failed: {ex}')

    def _remove_wrapper(self, wrapper: _BudgetedHandler) -> None:
        """Internal: auto-removal path. Called by _BudgetedHandler when
        K consecutive over-budget hits trigger drop. Idempotent --
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
                logger.exception(f'[SCOPE API ] _remove_wrapper driver-unregister failed: {ex}')
