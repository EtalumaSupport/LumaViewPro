# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope capability dataclass — the canonical "what does this scope have" query.

Pre-B7, callers asked capability questions piecemeal:
    scope.axes_present()            # list[str]
    scope.has_turret()              # bool
    scope.has_axis('Z')             # bool
    scope.motor_connected           # bool property
    scope.led.available_channels()  # tuple[int, ...]
    scope.camera.profile.pixel_formats  # list[str]

Each query touched the driver layer. Queries from different subsystems had
subtly different code paths, different error-handling, and different names
for the same underlying facts ("has_turret" vs "'T' in axes_present" vs
"motion.has_turret()"). Rule 9 ("Query capabilities, don't assume") called
for a single place where this information lives.

ScopeCapabilities is that place. It's a frozen dataclass built once at
init from the three drivers (motion / LED / camera). Callers read fields
directly. The existing capability methods on Lumascope (`axes_present`,
`has_turret`, etc.) stay as thin wrappers so no caller code has to
change — but new code should prefer `scope.capabilities.*`.

**Scope:** ScopeCapabilities contains static hardware *structure* (what
axes exist, what LED channels exist, what camera profile is loaded) —
things that don't change at runtime. It deliberately does NOT include
live connection state (`motor_connected`, `led_connected`, etc.) — those
must reflect disconnects at runtime and stay as live Lumascope
properties, not frozen snapshot fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeCapabilities:
    """Immutable snapshot of what a scope has.

    Built once at `Lumascope.__init__` from the three drivers. Fields
    are tuples (not lists) to reinforce immutability — a caller that
    wants to mutate would have to shallow-copy into their own list.
    """

    # ---- Motion ----
    axes: tuple[str, ...]
    """Axes physically present on this scope — from
    `motion.detect_present_axes()`. e.g. ('Z',) for LS820/LVC LS620,
    ('X','Y','Z') for LS850, ('X','Y','Z','T') for LS850T, () for no
    motor hardware."""

    has_focus: bool           # 'Z' in axes
    has_xy_stage: bool        # 'X' and 'Y' in axes
    has_turret: bool          # 'T' in axes

    motor_model: str
    """Scope model string reported by `motion.get_microscope_model()`, or
    empty string if unknown / not connected."""

    # ---- LED ----
    led_channels: tuple[int, ...]
    """LED channel indices available — from `led.available_channels()`.
    RP2040 = (0,1,2,3,4,5), FX2/LVC = (0,1,2,3). NullLEDBoard also returns
    the 6-channel set for Rule 8 silent-noop compatibility."""

    led_colors: tuple[str, ...]
    """Color names available — from `led.available_colors()`."""

    led_max_ma: int
    """Maximum LED current per channel, in mA. Currently a constant
    (1000 mA) matching firmware CH_MAX; may become per-driver later."""

    # ---- Camera ----
    camera_model: str
    """Model name from `camera.profile.model_name`, or empty string."""

    camera_supports_auto_gain: bool
    camera_supports_auto_exposure: bool

    camera_pixel_formats: tuple[str, ...]
    camera_binning_sizes: tuple[int, ...]
    camera_max_exposure_ms: int
    camera_pixel_size_um: float

    @classmethod
    def from_drivers(cls, motion, led, camera, led_max_ma: int) -> 'ScopeCapabilities':
        """Build a ScopeCapabilities snapshot from the three drivers.

        Tolerant of None / Null implementations. Never raises — if a
        driver method blows up or returns something unexpected, the
        corresponding field gets a safe default (empty tuple, empty
        string, False).

        Args:
            motion: A `MotorBoardProtocol` implementation (may be
                NullMotionBoard).
            led: An `LEDBoardProtocol` implementation (may be NullLEDBoard).
            camera: A camera object or None.
            led_max_ma: The API's LED current cap (today a class constant).
        """
        # Motion
        try:
            axes = tuple(motion.detect_present_axes())
        except Exception:
            axes = ()
        try:
            model = motion.get_microscope_model() or ''
        except Exception:
            model = ''

        # LED
        try:
            led_channels = tuple(led.available_channels())
        except Exception:
            led_channels = ()
        try:
            led_colors = tuple(led.available_colors())
        except Exception:
            led_colors = ()

        # Camera
        camera_model = ''
        camera_supports_auto_gain = False
        camera_supports_auto_exposure = False
        camera_pixel_formats: tuple[str, ...] = ()
        camera_binning_sizes: tuple[int, ...] = ()
        camera_max_exposure_ms = 0
        camera_pixel_size_um = 0.0
        if camera is not None:
            profile = getattr(camera, 'profile', None)
            if profile is not None:
                camera_model = getattr(profile, 'model_name', '') or ''
                camera_supports_auto_gain = bool(getattr(profile, 'has_auto_gain', False))
                camera_supports_auto_exposure = bool(getattr(profile, 'has_auto_exposure', False))
                camera_pixel_formats = tuple(getattr(profile, 'pixel_formats', ()) or ())
                camera_binning_sizes = tuple(getattr(profile, 'binning_sizes', ()) or ())
                camera_max_exposure_ms = int(getattr(profile, 'max_exposure_ms', 0) or 0)
                camera_pixel_size_um = float(getattr(profile, 'pixel_size_um', 0.0) or 0.0)

        return cls(
            axes=axes,
            has_focus='Z' in axes,
            has_xy_stage=('X' in axes and 'Y' in axes),
            has_turret='T' in axes,
            motor_model=model,
            led_channels=led_channels,
            led_colors=led_colors,
            led_max_ma=led_max_ma,
            camera_model=camera_model,
            camera_supports_auto_gain=camera_supports_auto_gain,
            camera_supports_auto_exposure=camera_supports_auto_exposure,
            camera_pixel_formats=camera_pixel_formats,
            camera_binning_sizes=camera_binning_sizes,
            camera_max_exposure_ms=camera_max_exposure_ms,
            camera_pixel_size_um=camera_pixel_size_um,
        )
