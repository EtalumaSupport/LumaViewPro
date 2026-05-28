# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Frame validity tracking for hardware state changes.

When hardware state changes (LED on/off, gain/exposure change, motor movement),
frames from the camera may not yet reflect the new state due to:
  1. Camera pipeline latency (2-3 frames to flush)
  2. Physical hardware settle time (motor moves take seconds)

Frame validity is the SINGLE source of truth for capture readiness. No capture
should proceed until frame_validity confirms all pending state changes have
settled. This includes both camera pipeline flush AND physical completion.

For camera-only sources (LED, gain, exposure): settled = frame count met.
For motion sources (xy_move, z_move, turret): settled = frame count met AND
axis has physically stopped moving (via settle callback).

Usage:
    fv = FrameValidity()
    fv.set_settle_check(my_axis_check_fn)  # Register motion completion callback
    fv.invalidate('z_move')                # Z axis started moving
    fv.frames_until_valid()                # Returns >0 (motion not complete)
    # ... frames are grabbed by live view, counter increments ...
    fv.frames_until_valid()                # Still >0 if Z still moving
    # ... Z arrives at target, settle check returns True ...
    fv.frames_until_valid()                # Returns 0 -- next frame is valid

Autofocus can exclude Z motion from validity checks since a slightly
defocused frame still produces a valid focus score:
    fv.is_valid_for(exclude_sources=('z_move',))
"""

import threading
import time

try:
    from lib import profile_trace
except ImportError:
    profile_trace = None


class FrameValidity:
    """Tracks frame validity after hardware state changes.

    Each hardware state change source has a configurable number of frames
    that must be skipped before the camera output reflects the new state.
    Motion sources additionally require physical completion (axis stopped).
    """

    DEFAULT_SKIP_FRAMES = 2

    # Per-source skip frame counts (camera pipeline flush).
    # Default skip counts -- overridden by per-camera measured values
    # from data/camera_timing/<model>.json via load_camera_timing().
    SKIP_FRAMES = {
        'led': 2,  # LED on/off or current change (measured: 2 on a2A3536)
        'gain': 2,  # Camera gain change (measured: 2 on a2A3536)
        'exposure': 3,  # Camera exposure time change (measured: 3 on a2A3536)
        'xy_move': 2,  # X or Y axis movement
        'z_move': 2,  # Z axis movement (autofocus may exclude this)
        'turret': 2,  # Turret rotation
    }

    # Sources that require physical hardware completion in addition to frame count.
    MOTION_SOURCES = frozenset({'xy_move', 'z_move', 'turret'})

    # Sources whose validity can be confirmed deterministically via per-frame
    # chunk metadata. When chunk_data is passed to count_frame() and the chunk
    # value matches the requested target within tolerance, the source is
    # cleared regardless of skip-frames count. LED has no chunk equivalent;
    # motion is firmware-gated via _settle_check_fn.
    CHUNK_VALIDATABLE_SOURCES = frozenset({'gain', 'exposure'})

    # Maps our source names to the chunk_data dict keys used by camera
    # drivers (chunk_data uses the genicam attribute symbolic names).
    CHUNK_KEY_FOR_SOURCE = {
        'gain': 'Gain',
        'exposure': 'ExposureTime',
    }

    # Float-tolerance for chunk-match equality. ChunkExposureTime is in
    # microseconds (the API converts ms -> us when calling set_target);
    # ChunkGain is in dB. Values measured by sweeping set values across
    # the supported range and reading back ChunkGain / ChunkExposureTime
    # on multiple Basler USB3 cameras (a2A3536-31umBAS ace 2; daA3840-45um
    # dart with firmware 1.1.0 and 2.6.0): observed deltas were bit-
    # identical across hardware and firmware -- quantization happens at
    # the Pylon SDK / genicam nodemap layer, not in camera firmware.
    # Gain round-trip error peaked at ~5e-5 dB (float epsilon), exposure was
    # bit-exact in microseconds. Tolerances set ~20x above observed max
    # for safety across future firmware revisions.
    DEFAULT_CHUNK_TOLERANCE = {
        'gain': 0.001,  # dB
        'exposure': 2.0,  # microseconds
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._pending = {}  # source -> frame_counter threshold for validity
        self._settle_check_fn = None  # Optional: (source) -> bool
        self._target_values = {}  # source -> requested value (for chunk-match)

    def set_settle_check(self, fn):
        """Register a callback that checks if a source has physically settled.

        Args:
            fn: callable(source: str) -> bool. Returns True if the hardware
                for this source has physically completed its state change.
                For motion sources, this typically checks axis state == IDLE.
                For non-motion sources, should return True.

        Called during validity checks for MOTION_SOURCES. Without this
        callback, motion sources settle based on frame count only (legacy
        behavior, incorrect for long moves).
        """
        self._settle_check_fn = fn

    def invalidate(self, source: str):
        """Record that hardware state changed and frames need to settle.

        Args:
            source: What changed ('led', 'gain', 'exposure', 'xy_move',
                    'z_move', 'turret'). Unknown sources use DEFAULT_SKIP_FRAMES.
        """
        skip = self.SKIP_FRAMES.get(source, self.DEFAULT_SKIP_FRAMES)
        with self._lock:
            self._pending[source] = self._frame_counter + skip
            counter = self._frame_counter
        if profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE:
            profile_trace.trace(
                'frame_validity_trace.csv',
                'ts_ms,event,source,frame_counter,target_frame,pending_count',
                [
                    int(time.time() * 1000),
                    'invalidate',
                    source,
                    counter,
                    counter + skip,
                    len(self._pending),
                ],
            )

    def count_frame(self, chunk_data: dict | None = None):
        """Record that a frame was grabbed from the camera.

        Call this after every successful camera grab (grab() or grab_new_capture()).
        Automatically clears non-motion sources that have settled by frame count.
        Motion sources are cleared only when both frame count AND settle check pass.

        Args:
            chunk_data: Optional per-frame chunk metadata from the camera
                (e.g. {'ExposureTime': 14530.0, 'Gain': 1.0, 'FrameID': 12345}).
                If provided, chunk-validatable sources (gain, exposure) whose
                target value matches the chunk value are cleared from pending,
                short-circuiting the skip-frames count for those sources.
                LED + motion + turret sources are unaffected (no chunk
                equivalent or firmware-gated). Backward compat: if None, the
                existing skip-frames + settle-check path is used unchanged.
        """
        with self._lock:
            self._frame_counter += 1
            settled = [
                s
                for s, target in self._pending.items()
                if self._is_source_settled_unlocked(s, target)
            ]
            # Chunks short-circuit skip-frames for chunk-validatable sources:
            # a source is cleared if either the settle-check path OR a chunk
            # value matches the requested target.
            if chunk_data is not None:
                for source in list(self._pending):
                    if source in settled:
                        continue
                    if self._chunk_match_unlocked(source, chunk_data):
                        settled.append(source)
            for s in settled:
                del self._pending[s]
            counter = self._frame_counter
            pending = len(self._pending)
        if profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE and settled:
            profile_trace.trace(
                'frame_validity_trace.csv',
                'ts_ms,event,source,frame_counter,target_frame,pending_count',
                [int(time.time() * 1000), 'settled', '+'.join(settled), counter, counter, pending],
            )

    def _is_source_settled_unlocked(self, source: str, target: int) -> bool:
        """Check if a source has settled. Must be called with _lock held."""
        if self._frame_counter < target:
            return False
        # Motion sources also require physical completion
        if source in self.MOTION_SOURCES and self._settle_check_fn is not None:
            return self._settle_check_fn(source)
        return True

    def _chunk_match_unlocked(self, source: str, chunk_data: dict) -> bool:
        """Return True if chunk_data's value for source matches the recorded
        target within tolerance. Must be called with _lock held.

        Returns False if any of: source has no chunk mapping, chunk_data
        lacks the relevant key, target was never recorded, or value is
        outside tolerance.
        """
        chunk_key = self.CHUNK_KEY_FOR_SOURCE.get(source)
        if chunk_key is None:
            return False
        chunk_value = chunk_data.get(chunk_key)
        if chunk_value is None:
            return False
        target = self._target_values.get(source)
        if target is None:
            return False
        tolerance = self.DEFAULT_CHUNK_TOLERANCE.get(source, 0.0)
        return abs(float(chunk_value) - target) <= tolerance

    def set_target(self, source: str, value):
        """Record the requested value for a chunk-validatable source.

        The API layer (Lumascope.set_gain / set_exposure_time) calls this
        after invalidate() so that when chunk metadata arrives via
        count_frame(chunk_data=...), the validity module can match the
        chunk against the target and clear the source deterministically.

        Args:
            source: Source name (e.g. 'gain', 'exposure'). Sources outside
                CHUNK_VALIDATABLE_SOURCES are accepted but never consulted.
            value: Target value to compare chunks against. None clears any
                prior target.
        """
        with self._lock:
            if value is None:
                self._target_values.pop(source, None)
            else:
                self._target_values[source] = float(value)

    def chunk_match(self, source: str, chunk_value, tolerance: float | None = None) -> bool:
        """Public float-tolerant equality between a chunk value and the recorded target.

        Used by tests and diagnostics. The internal count_frame() uses
        _chunk_match_unlocked() against the full chunk_data dict.

        Args:
            source: Source name.
            chunk_value: Observed value from chunk metadata. May be None.
            tolerance: Optional tolerance override; defaults to DEFAULT_CHUNK_TOLERANCE.
        """
        if chunk_value is None:
            return False
        with self._lock:
            target = self._target_values.get(source)
        if target is None:
            return False
        if tolerance is None:
            tolerance = self.DEFAULT_CHUNK_TOLERANCE.get(source, 0.0)
        return abs(float(chunk_value) - target) <= tolerance

    @property
    def is_valid(self) -> bool:
        """True if all pending state changes have settled."""
        with self._lock:
            return all(self._is_source_settled_unlocked(s, t) for s, t in self._pending.items())

    def is_valid_for(self, exclude_sources: tuple = ()) -> bool:
        """True if valid, ignoring specified sources.

        Useful for autofocus which can accept frames during Z motion:
            fv.is_valid_for(exclude_sources=('z_move',))
        """
        with self._lock:
            return all(
                self._is_source_settled_unlocked(s, t)
                for s, t in self._pending.items()
                if s not in exclude_sources
            )

    def frames_until_valid(self, exclude_sources: tuple = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Returns 0 if already valid. For motion sources that have met the frame
        count but are still physically moving, returns 1 (keep draining).
        """
        with self._lock:
            max_remaining = 0
            for source, target in self._pending.items():
                if source in exclude_sources:
                    continue
                frame_remaining = target - self._frame_counter
                if frame_remaining > 0:
                    max_remaining = max(max_remaining, frame_remaining)
                elif source in self.MOTION_SOURCES and self._settle_check_fn is not None:
                    # Frame count met but axis still moving -- keep draining
                    if not self._settle_check_fn(source):
                        max_remaining = max(max_remaining, 1)
            return max(0, max_remaining)

    @property
    def pending_sources(self) -> dict:
        """Current pending sources and their target frame counts (for debugging)."""
        with self._lock:
            return dict(self._pending)

    @property
    def frame_counter(self) -> int:
        """Current frame counter value (for debugging)."""
        with self._lock:
            return self._frame_counter

    def load_camera_timing(self, config: dict):
        """Override SKIP_FRAMES from measured per-camera timing config.

        Args:
            config: dict with 'skip_frames' key mapping source names to
                    measured frame counts. Only sources present in the config
                    are overridden; others keep their defaults.

        Typically called after camera connects with data loaded from
        data/camera_timing/<model>.json.
        """
        measured = config.get('skip_frames', {})
        for source, count in measured.items():
            if isinstance(count, int) and count >= 0:
                self.SKIP_FRAMES[source] = count

    def reset(self):
        """Clear all pending invalidations and reset frame counter."""
        with self._lock:
            self._pending.clear()
            self._frame_counter = 0
            self._target_values.clear()
