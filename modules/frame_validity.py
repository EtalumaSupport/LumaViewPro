# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Frame validity tracking for hardware state changes.

When hardware state changes (LED on/off, gain/exposure change, motor movement),
frames from the camera may not yet reflect the new state due to pipeline latency.
This module tracks how many frames must be grabbed and discarded before the camera
output is guaranteed to reflect the current hardware configuration.

Frame counts are used instead of time delays because they automatically adapt to
the camera's frame rate and exposure time — fast exposures drain quickly, slow
exposures drain slowly, matching the actual camera pipeline depth.

Usage:
    fv = FrameValidity()
    fv.invalidate('led')           # LED state changed
    fv.frames_until_valid()        # Returns 2 (default skip count)
    fv.count_frame()               # A frame was grabbed
    fv.count_frame()               # Another frame was grabbed
    fv.frames_until_valid()        # Returns 0 — next frame is valid

Autofocus can exclude Z motion from validity checks since a slightly
defocused frame still produces a valid focus score:
    fv.is_valid_for(exclude_sources=('z_move',))
"""

import threading


class FrameValidity:
    """Tracks frame validity after hardware state changes.

    Each hardware state change source has a configurable number of frames
    that must be skipped before the camera output reflects the new state.

    Skip counts are conservative placeholders (2 frames). They will be tuned
    with per-camera timing measurements (rolling shutter vs global shutter,
    camera pipeline depth, LED settling time, etc.).
    """

    DEFAULT_SKIP_FRAMES = 2

    # Per-source skip frame counts.
    # These represent how many frames must be grabbed and discarded after
    # a state change before the next frame reflects the new state.
    SKIP_FRAMES = {
        'led':      2,   # LED on/off or current change
        'gain':     2,   # Camera gain change
        'exposure': 2,   # Camera exposure time change
        'xy_move':  2,   # X or Y axis movement
        'z_move':   2,   # Z axis movement (autofocus may exclude this)
        'turret':   2,   # Turret rotation
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._pending = {}  # source -> frame_counter threshold for validity

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
        Automatically clears sources that have settled.
        """
        with self._lock:
            self._frame_counter += 1
            settled = [s for s, target in self._pending.items()
                       if self._frame_counter >= target]
            for s in settled:
                del self._pending[s]

    @property
    def is_valid(self) -> bool:
        """True if all pending state changes have settled."""
        with self._lock:
            return all(self._frame_counter >= target
                       for target in self._pending.values())

    def is_valid_for(self, exclude_sources: tuple = ()) -> bool:
        """True if valid, ignoring specified sources.

        Useful for autofocus which can accept frames during Z motion:
            fv.is_valid_for(exclude_sources=('z_move',))
        """
        with self._lock:
            return all(
                self._frame_counter >= target
                for source, target in self._pending.items()
                if source not in exclude_sources
            )

    def frames_until_valid(self, exclude_sources: tuple = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Returns 0 if already valid.
        """
        with self._lock:
            remaining = [
                target - self._frame_counter
                for source, target in self._pending.items()
                if source not in exclude_sources
            ]
            if not remaining:
                return 0
            return max(0, max(remaining))

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

    def reset(self):
        """Clear all pending invalidations and reset frame counter."""
        with self._lock:
            self._pending.clear()
            self._frame_counter = 0
