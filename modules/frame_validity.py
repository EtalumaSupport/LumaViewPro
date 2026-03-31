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
    fv.frames_until_valid()                # Returns 0 — next frame is valid

Autofocus can exclude Z motion from validity checks since a slightly
defocused frame still produces a valid focus score:
    fv.is_valid_for(exclude_sources=('z_move',))
"""

import threading


class FrameValidity:
    """Tracks frame validity after hardware state changes.

    Each hardware state change source has a configurable number of frames
    that must be skipped before the camera output reflects the new state.
    Motion sources additionally require physical completion (axis stopped).
    """

    DEFAULT_SKIP_FRAMES = 2

    # Per-source skip frame counts (camera pipeline flush).
    SKIP_FRAMES = {
        'led':      2,   # LED on/off or current change
        'gain':     2,   # Camera gain change
        'exposure': 2,   # Camera exposure time change
        'xy_move':  2,   # X or Y axis movement
        'z_move':   2,   # Z axis movement (autofocus may exclude this)
        'turret':   2,   # Turret rotation
    }

    # Sources that require physical hardware completion in addition to frame count.
    MOTION_SOURCES = frozenset({'xy_move', 'z_move', 'turret'})

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._pending = {}  # source -> frame_counter threshold for validity
        self._settle_check_fn = None  # Optional: (source) -> bool

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

    def count_frame(self):
        """Record that a frame was grabbed from the camera.

        Call this after every successful camera grab (grab() or grab_new_capture()).
        Automatically clears non-motion sources that have settled by frame count.
        Motion sources are cleared only when both frame count AND settle check pass.
        """
        with self._lock:
            self._frame_counter += 1
            settled = [s for s, target in self._pending.items()
                       if self._is_source_settled_unlocked(s, target)]
            for s in settled:
                del self._pending[s]

    def _is_source_settled_unlocked(self, source: str, target: int) -> bool:
        """Check if a source has settled. Must be called with _lock held."""
        if self._frame_counter < target:
            return False
        # Motion sources also require physical completion
        if source in self.MOTION_SOURCES and self._settle_check_fn is not None:
            return self._settle_check_fn(source)
        return True

    @property
    def is_valid(self) -> bool:
        """True if all pending state changes have settled."""
        with self._lock:
            return all(self._is_source_settled_unlocked(s, t)
                       for s, t in self._pending.items())

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
                    # Frame count met but axis still moving — keep draining
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
