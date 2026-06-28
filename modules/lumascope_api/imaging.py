# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ImagingAPI -- sub-API for camera capture / image acquisition.

ImagingAPI owns _camera_cache, _frame_buffer, _scale_bar,
_capturing_event, _focusing_event, _camera_listeners,
_camera_temp_event, _binning_size, _suppress_value_warnings,
_capture_return, _autofocus_return, and the frame_validity instance.
"""

from __future__ import annotations

import contextlib
import datetime
import logging as _logging
import threading
import time
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterator

import numpy as np

from lib import profile_trace
from lvp_logger import logger
import modules.image_utils as image_utils
from modules.frame_validity import FrameValidity
from modules.notification_center import notifications
from modules.sequential_io_executor import IOTask

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

    __slots__ = ('_consecutive_over', '_handler', '_imaging', '_name', '_removed')

    def __init__(self, imaging: ImagingAPI, handler, name: str) -> None:
        self._imaging = imaging
        self._handler = handler
        self._name = name
        self._consecutive_over = 0
        self._removed = False

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

        # State / camera locks. _state_lock guards _capture_return /
        # _autofocus_return / _scale_bar; _cam_lock serializes
        # access to the camera driver itself (any path that touches
        # the SDK reads/writes goes through this lock).
        self._state_lock = threading.Lock()
        self._cam_lock = profile_trace.TimedLock(threading.RLock(), name='imaging._cam_lock')

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
        self._capturing_event = threading.Event()  # set => capture in progress
        self._focusing_event = threading.Event()  # set => autofocus in progress

        # Capture / autofocus return slots. Reads/writes under
        # self._state_lock. Per the Sentinel-return contract: None
        # means "no result yet."
        self._capture_return = None
        self._autofocus_return = None

        # Evidence about the most recent capture_and_wait (hold duration,
        # drained frame count, chunk-verified exposure / gain). Read via
        # last_capture_info by callers that log per-capture provenance.
        self._last_capture_info = None

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

        # Binning size cache -- read live from driver if camera is
        # connected, otherwise default to 1. Defensive against fake/test
        # cameras that may not implement get_binning_size.
        if self._driver and hasattr(self._driver, 'get_binning_size'):
            try:
                self._binning_size = self._driver.get_binning_size()
            except Exception:
                self._binning_size = 1
        else:
            self._binning_size = 1

        # Scale-bar overlay config -- defaults disabled; users opt in via
        # set_scale_bar(...). Reads/writes under self._state_lock.
        self._scale_bar = {
            'enabled': False,
            'color': None,
        }

        # Camera state cache -- push-based, not polled.
        # Updated when camera connects and after every
        # set_gain/set_exposure/etc. UI reads from cache with zero SDK calls.
        self._camera_cache_lock = threading.Lock()
        self._camera_cache = {
            'active': False,
            'gain_db': 0.0,
            'exposure_ms': 0.0,
            'frame_size': {'width': 0, 'height': 0},
            'max_frame_size': {'width': 0, 'height': 0},
            'min_frame_size': {'width': 0, 'height': 0},
            'max_exposure_ms': 0.0,
            'max_gain_db': 0.0,
            'pixel_format': None,
            'binning': 1,
        }
        self._populate_camera_cache()

    @property
    def _driver(self) -> Camera | None:
        """Resolve the camera driver via the composition root each access.

        Lumascope's `_camera_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps ImagingAPI in sync without rebinding.
        """
        return self._scope._camera_driver

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
        """Populate camera cache from hardware. Called at init and on reconnect."""
        if not self._driver or not self._driver.active:
            with self._camera_cache_lock:
                self._camera_cache['active'] = False
            return

        try:
            cache = {
                'active': True,
                'gain_db': self._driver.get_gain() or 0.0,
                'exposure_ms': self._driver.get_exposure_t() or 0.0,
                'frame_size': self._driver.get_frame_size() or {'width': 0, 'height': 0},
                'max_frame_size': self._driver.get_max_frame_size() or {'width': 0, 'height': 0},
                'min_frame_size': self._driver.get_min_frame_size() or {'width': 0, 'height': 0},
                'max_exposure_ms': self._driver.get_max_exposure() or None,
                'max_gain_db': self._driver.get_max_gain()
                if hasattr(self._driver, 'get_max_gain')
                else None,
                'pixel_format': self._driver.get_pixel_format()
                if hasattr(self._driver, 'get_pixel_format')
                else None,
                'binning': self._driver.get_binning_size()
                if hasattr(self._driver, 'get_binning_size')
                else 1,
            }
            with self._camera_cache_lock:
                self._camera_cache.update(cache)
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
        ``set_gain`` / ``set_exposure_time`` short-circuits subsequent
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
            with self._camera_cache_lock:
                self._camera_cache['gain_db'] = -1.0
                self._camera_cache['exposure_ms'] = -1.0
            logger.warning(
                f'[SCOPE API ] cache refresh after auto-off failed: {e}; '
                f'cache invalidated to force next setter through.'
            )
            return
        with self._camera_cache_lock:
            if gain is not None:
                self._camera_cache['gain_db'] = float(gain)
            if exp is not None:
                self._camera_cache['exposure_ms'] = float(exp)
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
        force_targets: tuple[tuple[str, float | None], ...] = (),
        cache_update: dict[str, object] | None = None,
        gate_on_result: bool = False,
    ) -> object:
        """Single sanctioned path for a camera-state write and its validity
        consequence. Every camera setter routes its hardware write through here
        so the write and the frame-validity invalidation it requires are
        declared together -- a new setter cannot write a camera node and forget
        to invalidate.

        Order is load-bearing: the write happens first, then ``force_invalidate``
        fires unconditionally (the always-mark-RED contract for the manual value
        setters, so a hardware-rejected write still expires validity rather than
        leaving a stale frame acceptable), then the applied-only block runs.

        Args:
            write_fn: Zero-arg callable performing the driver write (including
                any lock it needs) and returning the driver's result.
            invalidates: Sources to invalidate only when the write was applied.
            force_invalidate: Sources to invalidate unconditionally, regardless
                of the result.
            targets: ``(source, value)`` pairs passed to ``set_target`` when the
                write was applied (``value`` None clears the target).
            force_targets: ``(source, value)`` pairs passed to ``set_target``
                unconditionally -- the mode/one-shot setters clear their chunk
                target whether or not the driver reported the write applied
                (target maintenance stays outside the applied gate).
            cache_update: Keys to write into the ``_camera_cache`` snapshot when
                the write was applied.
            gate_on_result: True means "applied" is ``bool(result)`` (the
                result-gated setters); False means ``result is not False`` (the
                value/auto setters, where a None return still counts as applied).

        Returns:
            The driver write's result, so the caller can do its own rejection
            handling / listener fire.
        """
        result = write_fn()
        for source in force_invalidate:
            self.frame_validity.invalidate(source)
        for source, value in force_targets:
            self.frame_validity.set_target(source, value)
        applied = bool(result) if gate_on_result else (result is not False)
        if applied:
            for source in invalidates:
                self.frame_validity.invalidate(source)
            for source, value in targets:
                self.frame_validity.set_target(source, value)
            if cache_update:
                with self._camera_cache_lock:
                    self._camera_cache.update(cache_update)
        return result

    # --- Setters ---
    def set_gain(self, gain_db: float) -> None:
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
        changed = abs(float(gain_db) - self.camera_gain) >= 0.001

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
            _api_log.info(f'set_gain {gain_db}dB')
            self._fire_camera_listeners('gain', float(gain_db))

    def set_exposure_time(self, exposure_ms: float) -> None:
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
        changed = abs(float(exposure_ms) - self.camera_exposure_ms) >= 0.001
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
                f'[SCOPE API ] set_exposure_time({exposure_ms}ms) is below '
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

    def set_auto_gain(self, state: bool, settings: dict) -> None:
        """Enable or disable automatic gain adjustment.

        Args:
            state: True to enable auto gain, False to disable.
            settings: Dict with 'target_brightness', 'min_gain_db', 'max_gain_db',
                and optionally 'max_exposure_ms' (the per-channel-class upper
                bound on the exposure AG/AE may drive to; the caller supplies it
                since it knows the layer).
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
            force_targets=(('gain', None),),
        )
        # Hardware-truth wins over cache after the auto cycle ends.
        if not state:
            self._refresh_cache_from_hardware_after_auto()

    def set_auto_exposure_time(self, state: bool = True) -> None:
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
            force_targets=(('exposure', None),),
        )
        # Hardware-truth wins over cache after the auto cycle ends.
        if not state:
            self._refresh_cache_from_hardware_after_auto()

    def set_frame_size(self, w: int, h: int) -> None:
        """Set the camera frame size in pixels.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.
        """

        if not self._driver or not self._driver.active:
            self._notify_camera_absent('frame size')
            return
        self._driver.set_frame_size(w, h)
        self.frame_validity.invalidate('frame_size')
        with self._camera_cache_lock:
            self._camera_cache['frame_size'] = {'width': int(w), 'height': int(h)}

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
        """Set camera pixel binning size.

        Args:
            size: Binning factor (1 = no binning, 2 = 2x2, etc.).

        Returns:
            bool: True if the driver applied the binning. False if the
                camera is absent, the driver returned False (size out of
                range, camera inactive), or the driver raised an
                exception. Caller can use the result to decide whether to
                proceed with operations that depend on the new binning.
        """
        try:
            self._binning_size = size

            if self._driver:
                ok = self._driver.set_binning_size(size=size)
            else:
                ok = False
                self._notify_camera_absent('binning')
            if ok:
                self.frame_validity.invalidate('binning')
            _api_log.info(f'set_binning {size}x{size} -> {ok}')
            return ok
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting binning size: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Binning change failed',
                f'Could not set binning to {size}x{size}: {type(ex).__name__}: {ex}. '
                f'Camera may still be at previous binning -- verify actual frame size.',
            )
            return False

    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Format string (e.g. 'Mono8', 'Mono12').

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver returned False (unsupported format),
                or the driver raised. Never raises -- caller may safely
                check `if not scope.imaging.set_pixel_format(...)` for fallback.
        """
        if not self._driver or not self._driver.active:
            self._notify_camera_absent('pixel format')
            return False
        try:
            result = self._driver.set_pixel_format(pixel_format)
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting pixel format: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Pixel format change failed',
                f'Could not set pixel format to {pixel_format}: '
                f'{type(ex).__name__}: {ex}. Camera may still be at the '
                f'previous format.',
            )
            return False
        if result:
            self.frame_validity.invalidate('pixel_format')
            with self._camera_cache_lock:
                self._camera_cache['pixel_format'] = pixel_format
        return result

    def set_conversion_gain_mode(self, mode: str) -> bool:
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
            result = self._driver.set_conversion_gain_mode(mode)
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting conversion gain mode: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Conversion gain mode change failed',
                f'Could not set conversion gain mode to {mode!r}: '
                f'{type(ex).__name__}: {ex}. Camera may still be at the previous mode.',
            )
            return False
        if result:
            self.frame_validity.invalidate('conversion_gain_mode')
        return result

    def set_line_noise_reduction(self, enabled: bool) -> bool:
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
            result = self._driver.set_line_noise_reduction(enabled=enabled)
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting line noise reduction: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Line noise reduction change failed',
                f'Could not {"enable" if enabled else "disable"} line noise reduction: '
                f'{type(ex).__name__}: {ex}.',
            )
            return False
        if result:
            self.frame_validity.invalidate('line_noise_reduction')
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
                f'Could not set acquisition_stop_mode to {mode!r}: '
                f'{type(ex).__name__}: {ex}. Camera may still be at '
                f'the previous stop-mode setting.',
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
                f'Could not set BandwidthReserveMode to {mode!r}: {type(ex).__name__}: {ex}.',
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
                f'Could not set DLTL to mode={mode}, value_bps={value_bps}: '
                f'{type(ex).__name__}: {ex}. Camera may still be at the '
                f'previous DLTL setting.',
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
                f'Could not set MaxTransferSize to {value_bytes}: {type(ex).__name__}: {ex}.',
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
                f'Could not set NumMaxQueuedUrbs to {value}: {type(ex).__name__}: {ex}.',
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
                f'Could not set MaxNumBuffer to {value}: {type(ex).__name__}: {ex}.',
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
                f'Could not set GevSCPSPacketSize to {size_bytes}: {type(ex).__name__}: {ex}.',
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
                f'Could not set GevSCPD to {delay_ticks}: {type(ex).__name__}: {ex}.',
            )
            raise

    def set_gain_async(self, gain_db, *, callback=None, cb_kwargs=None) -> None:
        """Submit ``set_gain`` to the camera_executor; return immediately.

        Args:
            gain_db: Gain value in dB.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'set_gain_async')
        ex.put(
            IOTask(
                action=self.set_gain,
                args=(gain_db,),
                callback=callback,
                cb_kwargs=cb_kwargs,
            )
        )

    def set_gain_sync(self, gain_db, *, timeout_s: float = 5.0) -> None:
        """Run ``set_gain`` through the camera_executor and block until done.

        Args:
            gain_db: Gain value in dB.
            timeout_s: Max seconds to wait for completion.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'set_gain_sync')
        task = IOTask(action=self.set_gain, args=(gain_db,))
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout_s)

    def set_exposure_time_async(self, exposure_ms, *, callback=None, cb_kwargs=None) -> None:
        """Submit ``set_exposure_time`` to the camera_executor; return immediately.

        Args:
            exposure_ms: Exposure time in milliseconds.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'set_exposure_time_async')
        ex.put(
            IOTask(
                action=self.set_exposure_time,
                args=(exposure_ms,),
                callback=callback,
                cb_kwargs=cb_kwargs,
            )
        )

    def set_exposure_sync(self, exposure_ms, *, timeout_s: float = 5.0) -> None:
        """Run ``set_exposure_time`` through the camera_executor and block.

        Args:
            exposure_ms: Exposure time in milliseconds.
            timeout_s: Max seconds to wait for completion.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'set_exposure_sync')
        task = IOTask(action=self.set_exposure_time, args=(exposure_ms,))
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout_s)

    def set_max_acquisition_frame_rate(
        self,
        enabled: bool,
        fps: float = 1.0,
    ) -> None:
        """Enable or disable the camera's acquisition frame-rate cap.

        Passthrough to the camera driver's ``set_max_acquisition_frame_rate``.
        When enabled, the camera will not produce frames faster than
        ``fps`` regardless of sensor-readout capability. Used by the
        manual-record path (#633 Stage 2C) to clamp video to the user's
        requested FPS, and by char-tool crash protection.

        Args:
            enabled: True to cap frame rate, False to remove the cap.
            fps: Target frame rate in fps when ``enabled=True``.
                Ignored when ``enabled=False``.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return
        if not hasattr(self._driver, 'set_max_acquisition_frame_rate'):
            logger.warning(
                f'[SCOPE API ] set_max_acquisition_frame_rate: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return
        try:
            self._driver.set_max_acquisition_frame_rate(enabled=enabled, fps=fps)
        except Exception as ex:
            logger.exception(f'[SCOPE API ] Error setting max_acquisition_frame_rate: {ex}')
            from modules.notification_center import notifications

            notifications.error(
                'Camera',
                'Frame-rate cap change failed',
                f'Could not set frame-rate cap to enabled={enabled}, '
                f'fps={fps}: {type(ex).__name__}: {ex}. Camera may still be '
                f'at the previous setting.',
            )
            raise

    # --- Getters ---
    def get_gain(self) -> float:
        """Get the current camera gain.

        Returns:
            float: Gain in dB, or -1 if camera inactive.
        """

        if not self._driver or not self._driver.active:
            return -1
        return self._driver.get_gain()

    def get_exposure_time(self) -> float:
        """Get the current camera exposure time.

        Returns:
            float: Exposure time in milliseconds, or 0 if camera inactive.
        """

        if not self._driver or not self._driver.active:
            return 0
        exposure = self._driver.get_exposure_t()
        return exposure

    def get_frame_size(self) -> dict | None:
        """Get the current camera frame size.

        Returns:
            dict | None: Contains 'width' and 'height' in pixels, or
                None if inactive.
        """

        if not self._driver or not self._driver.active:
            return
        return self._driver.get_frame_size()

    def get_pixel_format(self) -> str | None:
        """Get the current camera pixel format.

        Returns:
            str | None: Pixel format string (e.g. 'Mono8'), or None if inactive.
        """
        if not self._driver or not self._driver.active:
            return None
        return self._driver.get_pixel_format()

    def get_max_width(self) -> int:
        """Get the maximum pixel width of the camera sensor.

        Returns:
            int: Max width in pixels, or 0 if camera inactive.
        """
        if (not self._driver) or (not self._driver.active):
            return 0
        return self._driver.get_max_frame_size()['width']

    def get_max_height(self) -> int:
        """Get the maximum pixel height of the camera sensor.

        Returns:
            int: Max height in pixels, or 0 if camera inactive.
        """
        if (not self._driver) or (not self._driver.active):
            return 0
        return self._driver.get_max_frame_size()['height']

    def get_width(self) -> int:
        """Get the current frame width setting.

        Returns:
            int: Current width in pixels, or 0 if camera unavailable.
        """
        if not self._driver:
            return 0
        return self._driver.get_frame_size()['width']

    def get_height(self) -> int:
        """Get the current frame height setting.

        Returns:
            int: Current height in pixels, or 0 if camera unavailable.
        """
        if not self._driver:
            return 0
        return self._driver.get_frame_size()['height']

    def get_binning_size(self) -> int:
        """Get the current camera binning size.

        Returns:
            int: Current binning factor (1 if camera inactive).
        """
        if not self._driver or not self._driver.active:
            return 1

        return self._driver.get_binning_size()

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
        independent of the current binning factor (unlike get_max_width/height,
        which reflect the max settable at the current binning). Empty dict if
        no camera or the profile does not declare it.

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
        """Return the camera's frame-size pixel alignment.

        Frame width/height must be a multiple of these values (a Pylon
        constraint on most current models). Defaults to 4x4 when unknown.

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
    def capture_and_wait(
        self,
        force_to_8bit: bool = True,
        *,
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
                on camera-inactive or frame-drain failure. Per the
                Sentinel-return contract: `if image is None: ...`.
        """
        if not self._driver or not self._driver.active:
            return None

        hold_start = time.monotonic()
        exposure_s = self.get_exposure_time() / 1000
        grab_timeout_s = max(exposure_s * 3, 1.0)

        # Drain stale frames until all pending state changes have settled.
        # Per-frame chunk metadata flows into count_frame so chunks short-
        # circuit skip-frames for chunk-validatable sources (gain, exposure).
        # Cameras without chunks return None and fall back to the existing
        # skip-frames + settle-check path. Each drained grab passes its frame
        # timestamp so a frame concurrently counted by the preview poller is
        # not counted twice.
        drain_iterations = 0
        while self.frame_validity.frames_until_valid(exclude_sources=exclude_sources) > 0:
            status, drain_frame_ts = self._driver.grab_new_capture(timeout_s=grab_timeout_s)
            if status:
                self.frame_validity.count_frame(
                    chunk_data=self._get_latest_chunks(), frame_ts=drain_frame_ts
                )
                drain_iterations += 1
            else:
                remaining = self.frame_validity.frames_until_valid(exclude_sources=exclude_sources)
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
                return False

        image = self.get_image(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            all_ones_check=all_ones_check,
            timeout_s=timeout_s,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
            force_new_capture=True,
            new_capture_timeout_s=grab_timeout_s,
            verify_chunk_targets=True,
        )

        # Record per-capture evidence for the caller's log line (protocol
        # captures log brightness + the chunk-verified settings per frame so
        # a support bundle shows what each saved frame was exposed with).
        chunks = self._get_latest_chunks() or {}
        with self._state_lock:
            self._last_capture_info = {
                'hold_ms': (time.monotonic() - hold_start) * 1000.0,
                'drained': drain_iterations,
                'chunk_exposure_us': chunks.get('ExposureTime'),
                'chunk_gain_db': chunks.get('Gain'),
            }
        return image

    def capture_and_wait_async(
        self,
        *,
        callback=None,
        cb_kwargs=None,
        force_to_8bit: bool = True,
        exclude_sources: tuple = (),
        all_ones_check: bool = False,
        earliest_image_ts: datetime.datetime | None = None,
        timeout_s: float = 0.0,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback=None,
    ) -> None:
        """Submit ``capture_and_wait`` to the camera_executor; return
        immediately. The captured image is delivered via ``callback``.

        Args:
            callback: Completion callback; receives the captured array
                (or ``None`` on capture failure) as the first arg.
            cb_kwargs: Optional kwargs passed to the callback.
            force_to_8bit: Convert to 8-bit output.
            exclude_sources: Sources to ignore for validity (e.g. ('z_move',)).
            all_ones_check: Reject all-max-value frames.
            earliest_image_ts: Reject frames captured before this timestamp.
            timeout_s: Timeout (seconds) for the final get_image call.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'capture_and_wait_async')
        ex.put(
            IOTask(
                action=self.capture_and_wait,
                kwargs={
                    'force_to_8bit': force_to_8bit,
                    'exclude_sources': exclude_sources,
                    'all_ones_check': all_ones_check,
                    'earliest_image_ts': earliest_image_ts,
                    'timeout_s': timeout_s,
                    'sum_count': sum_count,
                    'sum_delay_s': sum_delay_s,
                    'sum_iteration_callback': sum_iteration_callback,
                },
                callback=callback,
                cb_kwargs=cb_kwargs,
            )
        )

    def capture_and_wait_sync(
        self,
        *,
        timeout_s: float = 30.0,
        force_to_8bit: bool = True,
        exclude_sources: tuple = (),
        all_ones_check: bool = False,
        earliest_image_ts: datetime.datetime | None = None,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback=None,
    ) -> np.ndarray | None:
        """Run ``capture_and_wait`` through the camera_executor and block.

        Args:
            timeout_s: Max seconds to wait for completion (wraps the executor
                Future.result wait, not the inner capture_and_wait grab).
            force_to_8bit: Convert to 8-bit output.
            exclude_sources: Sources to ignore for validity (e.g. ('z_move',)).
            all_ones_check: Reject all-max-value frames.
            earliest_image_ts: Reject frames captured before this timestamp.
            sum_count: Number of frames to sum for noise reduction.
            sum_delay_s: Delay between summed frames.
            sum_iteration_callback: Called after each summed frame.

        Returns:
            The captured image array, or None on failure (camera-inactive,
            frame-drain failed, executor absent, or future not delivered).
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'capture_and_wait_sync')
        task = IOTask(
            action=self.capture_and_wait,
            kwargs={
                'force_to_8bit': force_to_8bit,
                'exclude_sources': exclude_sources,
                'all_ones_check': all_ones_check,
                'earliest_image_ts': earliest_image_ts,
                'sum_count': sum_count,
                'sum_delay_s': sum_delay_s,
                'sum_iteration_callback': sum_iteration_callback,
            },
        )
        fut = ex.put(task, return_future=True)
        if fut:
            return fut.result(timeout=timeout_s)
        return None

    # A frame at least this saturated is treated as blown -- a stale-gain
    # or over-exposure symptom. Set high so a legitimately bright field
    # does not trip it; a true blown-white frame saturates essentially
    # every pixel. Surfaced (warn + notify) rather than saved silently.
    _SATURATION_NEAR_MAX_FRACTION = 0.99  # pixel >= 99% of full scale = saturated
    _SATURATION_BLOWN_FRACTION = 0.98  # >= 98% of pixels saturated = blown frame

    @staticmethod
    def _saturated_fraction(arr: np.ndarray | None) -> float:
        """Fraction of pixels at or above the near-full-scale threshold."""
        if arr is None or arr.size == 0:
            return 0.0
        near_max = np.iinfo(arr.dtype).max * ImagingAPI._SATURATION_NEAR_MAX_FRACTION
        return float(np.count_nonzero(arr >= near_max)) / arr.size

    def get_image(
        self,
        force_to_8bit: bool = True,
        earliest_image_ts: datetime.datetime | None = None,
        timeout_s: float = 5.0,
        all_ones_check: bool = False,
        sum_count: int = 1,
        sum_delay_s: float = 0,
        sum_iteration_callback=None,
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
                # set_gain/set_exposure from another thread mid-frame.
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

                if (
                    all_ones_check
                    and self._saturated_fraction(tmp) >= self._SATURATION_BLOWN_FRACTION
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
                    # and the walk would otherwise block concurrent set_gain/set_exposure.
                    if (
                        retry_frame is not None
                        and self._saturated_fraction(retry_frame) < self._SATURATION_BLOWN_FRACTION
                    ):
                        tmp = retry_frame  # retry was clean, use it
                    else:
                        # Log (not notify): a blown frame is self-evident on
                        # screen and in the saved file, so a popup adds nothing.
                        # The log line is for the post-mortem / log-analysis pass.
                        sat_pct = self._saturated_fraction(tmp) * 100.0
                        logger.warning(
                            f'[SCOPE API ] get_image: captured frame is {sat_pct:.0f}% '
                            f'saturated -- likely over-exposure or a stale camera gain; '
                            f'the frame may be unusable.'
                        )

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

        use_scale_bar = self._scale_bar['enabled']
        if self._scope.runtime_state._objective is None:
            use_scale_bar = False

        need_8bit = force_to_8bit and image.dtype != np.uint8

        # A summed capture lives in a 16-bit container; a single frame carries
        # the camera's native payload depth. The scale bar's white value and the
        # 8-bit downconvert divisor both follow this depth so a summed 12-bit
        # value never indexes the 12-bit display table, a 10-bit frame is not
        # crushed as if 12-bit, and the bar maps to full white not a dim gray.
        # Query the driver only when a consumer needs it -- a raw passthrough
        # frame returns without touching the driver's depth.
        if use_scale_bar or need_8bit:
            # A summed capture is promoted to a 16-bit container by the loop
            # above (the sum transform declares the new depth). A single frame
            # carries the depth it was captured under, read from the frame just
            # grabbed -- not a fresh format query that a mid-capture switch could
            # have moved ahead of this frame.
            significant_bits = 16 if sum_count > 1 else self._driver.last_significant_bits

        if use_scale_bar:
            image = image_utils.add_scale_bar(
                image=image,
                objective=self._scope.runtime_state._objective,
                binning_size=self._binning_size,
                color=self._scale_bar.get('color'),
                significant_bits=significant_bits,
            )

        if need_8bit:
            image = image_utils.convert_to_8bit(image, significant_bits)

        return image

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

        use_scale_bar = self._scale_bar['enabled']
        if self._scope.runtime_state._objective is None:
            use_scale_bar = False

        if use_scale_bar:
            tmp = image_utils.add_scale_bar(
                image=tmp,
                objective=self._scope.runtime_state._objective,
                binning_size=self._binning_size,
                color=self._scale_bar.get('color'),
                significant_bits=frame_significant_bits,
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
        """
        return self._driver.significant_bits if self._driver is not None else 16

    # --- State / lifecycle properties ---
    @property
    def camera_active(self) -> bool:
        """Whether the camera is connected and active (reads cache).

        Returns:
            bool: True if the camera is currently active.
        """
        with self._camera_cache_lock:
            return self._camera_cache['active']

    @property
    def camera_gain(self) -> float:
        """Current camera gain in dB (reads cache).

        Returns:
            float: Cached gain value in dB.
        """
        with self._camera_cache_lock:
            return self._camera_cache['gain_db']

    @property
    def camera_exposure_ms(self) -> float:
        """Current camera exposure time in ms (reads cache).

        Returns:
            float: Cached exposure time in milliseconds.
        """
        with self._camera_cache_lock:
            return self._camera_cache['exposure_ms']

    @property
    def camera_frame_size(self) -> dict:
        """Current camera frame size as {'width': int, 'height': int} (reads cache).

        Returns:
            dict: Copy of the cached frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['frame_size'])

    @property
    def camera_min_frame_size(self) -> dict:
        """Minimum camera frame size (reads cache).

        Returns:
            dict: Copy of the cached min frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['min_frame_size'])

    @property
    def camera_max_exposure(self) -> float | None:
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
    def camera_max_gain(self) -> float | None:
        """Maximum camera gain in dB, or None if no camera is connected.

        Parallel to camera_max_exposure -- lets the UI size the gain
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
    def camera_pixel_format(self) -> str:
        """Current camera pixel format (e.g. 'Mono8', 'Mono12') (reads cache).

        Returns:
            str: Cached pixel format string.
        """
        with self._camera_cache_lock:
            return self._camera_cache.get('pixel_format', 'Mono8')

    # --- Save / restore ---
    def save_camera_state(self, tag: str) -> dict:
        """Snapshot the current camera gain and exposure for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            dict: Snapshot suitable for passing to ``restore_camera_state``.
        """
        gain_db = self.get_gain()
        exposure_ms = self.get_exposure_time()
        snapshot = {'tag': tag, 'gain_db': gain_db, 'exposure_ms': exposure_ms}
        _api_log.info(f'save_camera_state tag={tag}: gain={gain_db} exp={exposure_ms}')
        return snapshot

    def restore_camera_state(self, snapshot: dict) -> None:
        """Restore camera gain and exposure from a previously saved state.

        Args:
            snapshot: Return value from ``save_camera_state``.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        _api_log.info(f'restore_camera_state tag={tag}')
        gain_db = snapshot.get('gain_db', -1)
        exposure_ms = snapshot.get('exposure_ms', 0)
        if gain_db >= 0:
            self.set_gain(gain_db)
        if exposure_ms > 0:
            self.set_exposure_time(exposure_ms)

    # --- Camera config orchestration ---
    def apply_layer_camera_settings(
        self,
        gain_db: float,
        exposure_ms: float,
        auto_gain: bool = False,
        auto_gain_settings: dict | None = None,
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
        self.set_gain(gain_db)
        self.set_exposure_time(exposure_ms)
        if auto_gain_settings is not None:
            self.set_auto_gain(auto_gain, settings=auto_gain_settings)
        _api_log.info(
            f'apply_layer_camera_settings gain={gain_db}dB exp={exposure_ms}ms auto_gain={auto_gain}'
        )

    def update_auto_gain_target_brightness(self, target_brightness: float) -> None:
        """Set the auto-gain target brightness on the camera.

        Args:
            target_brightness: Target brightness value (0.0 to 1.0).
        """
        if not self._driver or not self._driver.active:
            return
        self._driver.update_auto_gain_target_brightness(target_brightness)

    def auto_gain_once(
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
        self._driver.auto_gain_once(
            state=state,
            target_brightness=target_brightness,
            min_gain_db=min_gain_db,
            max_gain_db=max_gain_db,
            ae_max_exposure_ms=ae_max_exposure_ms,
        )
        # One-shot AG changes both gain and exposure on the camera; the
        # pipeline still needs frames to flush the converged values, so
        # the validity marker must go RED until they settle. The converged
        # values are chosen by the SDK, so clear any manual chunk-match
        # target and fall back to skip-frames settling.
        self.frame_validity.invalidate('gain')
        self.frame_validity.invalidate('exposure')
        self.frame_validity.set_target('gain', None)
        self.frame_validity.set_target('exposure', None)
        # One-shot AG always ends with the auto cycle complete and the
        # SDK toggled back to Off internally; hardware holds the
        # converged value while LVP's cache is still pre-auto.
        self._refresh_cache_from_hardware_after_auto()

    def update_camera_config(self) -> contextlib.AbstractContextManager[Any]:
        """Context manager for batched camera config updates.

        Usage::

            with scope.imaging.update_camera_config():
                scope.imaging.set_gain(5.0)
                scope.imaging.set_exposure_time(100)

        Returns:
            A context manager. Falls back to ``contextlib.nullcontext()``
            when no camera is active.
        """
        if not self._driver or not self._driver.active:
            return contextlib.nullcontext()
        return self._driver.update_camera_config()

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
    def is_capturing(self) -> bool:
        """True while the microscope is capturing an image.

        Returns:
            bool: True if a capture is in progress.
        """
        return self._capturing_event.is_set()

    @is_capturing.setter
    def is_capturing(self, value: bool) -> None:
        """Set the capture-in-progress flag."""
        if value:
            self._capturing_event.set()
        else:
            self._capturing_event.clear()

    @property
    def is_focusing(self) -> bool:
        """True while the microscope is running autofocus.

        Returns:
            bool: True if an autofocus run is in progress.
        """
        return self._focusing_event.is_set()

    @is_focusing.setter
    def is_focusing(self, value: bool) -> None:
        """Set the autofocus-in-progress flag."""
        if value:
            self._focusing_event.set()
        else:
            self._focusing_event.clear()

    @property
    def capture_return(self) -> np.ndarray | None:
        """Latest capture result (image array or None).

        Returns:
            Image array on success, or None when no capture has
            completed yet. Per the Sentinel-return contract:
            `if scope.imaging.capture_return is None: ...`.
        """
        with self._state_lock:
            return self._capture_return

    @capture_return.setter
    def capture_return(self, value) -> None:
        """Store the latest capture result."""
        with self._state_lock:
            self._capture_return = value

    @property
    def autofocus_return(self) -> Any | None:
        """Latest autofocus result.

        Returns:
            The most recent autofocus return value (driver-defined), or
            None if autofocus has not run.
        """
        with self._state_lock:
            return self._autofocus_return

    @autofocus_return.setter
    def autofocus_return(self, value) -> None:
        """Store the latest autofocus result."""
        with self._state_lock:
            self._autofocus_return = value

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

    def count_frame(self) -> None:
        """Record that a frame was grabbed from the camera.

        Delegates to frame_validity (no driver call).
        """
        self.frame_validity.count_frame()

    @property
    def last_capture_info(self) -> dict | None:
        """Evidence about the most recent capture_and_wait on this scope.

        Returns:
            dict | None: ``{'hold_ms', 'drained', 'chunk_exposure_us',
                'chunk_gain_db'}`` for the latest capture, or None before
                the first capture. Chunk values are None on cameras
                without chunk support.
        """
        with self._state_lock:
            return dict(self._last_capture_info) if self._last_capture_info else None

    # --- Scale bar ---
    @property
    def scale_bar_config(self) -> dict:
        """Return a snapshot of scale bar settings.

        Returns:
            dict: Copy of the scale bar config (e.g. enabled, color).
        """
        with self._state_lock:
            return dict(self._scale_bar)

    @property
    def scale_bar_enabled(self) -> bool:
        """Whether the scale bar overlay is enabled.

        Returns:
            bool: True if the scale bar is enabled.
        """
        with self._state_lock:
            return bool(self._scale_bar.get('enabled', False))

    def set_scale_bar(self, enabled: bool, color: str | None = None) -> None:
        """Configure the scale bar overlay on captured images.

        Args:
            enabled: Whether to draw the scale bar.
            color: Scale bar color (e.g. "white"). Uses default if None.
        """
        self._scale_bar['enabled'] = enabled
        if color is not None:
            self._scale_bar['color'] = color

    def get_scale_bar(self) -> dict:
        """Get the full scale-bar configuration.

        Companion getter to ``set_scale_bar``; ``scale_bar_enabled`` covers
        just the on/off flag, but this returns the full
        ``{'enabled', 'color', ...}`` snapshot so a caller can read what
        was previously set.

        Returns:
            Snapshot dict (defensive copy) of the scale-bar state.
        """
        with self._state_lock:
            return dict(self._scale_bar)

    # --- Camera diagnostics (live in-flight only; data source = DiagnosticsAPI) ---
    def log_camera_temps(self) -> None:
        """Emit one INFO line per camera temperature sensor.

        No-op when no camera is connected. Called once on startup and
        periodically by ``start_camera_temp_logging``. Reads temperatures
        through `scope.diagnostics.get_camera_temperatures` -- the canonical
        camera-temp probe (cold probes live on DiagnosticsAPI).
        """
        if not self._scope.camera_connected:
            return
        for source, temp in self._scope.diagnostics.get_camera_temperatures().items():
            logger.info(f'[CAM Class ] Camera {source} Temperature : {temp:.2f} degC')

    def start_camera_temp_logging(
        self, schedule_interval_fn, unschedule_fn, *, interval_s: float = 14400.0
    ) -> None:
        """Own the periodic camera-temp logging schedule.

        Was previously a Clock.schedule_interval registered by the App
        and stored as a fresh attribute on the MainDisplay widget -- if
        MainDisplay was ever recreated (LS850/LS620 scope swap), the
        Clock event became orphaned and continued logging temps from a
        now-disconnected camera.

        Args:
            schedule_interval_fn: Callable matching ``Clock.schedule_interval(func, interval)``.
                Passed in so this module stays GUI-agnostic.
            unschedule_fn: Callable matching ``Clock.unschedule(event)``,
                used by ``stop_camera_temp_logging`` and on
                disconnect-while-logging.
            interval_s: Seconds between log emissions; default 4 hours.
        """
        # Defensive: if a previous logger is already running, stop it
        # before starting a new one (idempotent -- safe to call repeatedly).
        if getattr(self, '_camera_temp_event', None) is not None:
            self.stop_camera_temp_logging(unschedule_fn)

        self._camera_temp_unschedule_fn = unschedule_fn
        self.log_camera_temps()  # one immediate sample

        def _tick(_dt=0):
            # Self-unschedule when the camera disconnects so a stale
            # event doesn't survive scope switches.
            if not self._scope.camera_connected:
                self.stop_camera_temp_logging(unschedule_fn)
                return
            self.log_camera_temps()

        self._camera_temp_event = schedule_interval_fn(_tick, interval_s)
        logger.info(f'[SCOPE API ] start_camera_temp_logging: interval={interval_s}s')

    def stop_camera_temp_logging(self, unschedule_fn=None) -> None:
        """Cancel the periodic camera-temp logger if active.

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

        Note: this fires on set_gain/set_exposure_time (user actions),
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
