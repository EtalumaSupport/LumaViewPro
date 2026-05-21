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
from types import MappingProxyType
from typing import Any, Callable, Mapping

from drivers.exceptions import HardwareError
from lvp_logger import logger


def _probe(label: str, fn: Callable[[], Any], fallback: Any) -> Any:
    """Run a capability probe; return fallback on expected absence; log
    on hardware fault. Other exceptions (TypeErrors, KeyErrors from buggy
    code) propagate so they surface for debugging.

    Catches:
        - AttributeError / NotImplementedError: feature absent per Rule 8.
        - HardwareError: real driver fault. Logged at warning so it's
          visible in main log; fallback used so capability dataclass
          still constructs.
    """
    try:
        return fn()
    except (AttributeError, NotImplementedError):
        return fallback
    except HardwareError as e:
        logger.warning(f'[CAPABILITIES] {label} probe failed: {e}; using fallback')
        return fallback


# Canonical home for the LED current cap (matches firmware CH_MAX).
# Lumascope previously carried this as a `LED_MAX_MA` class constant
# (freeze-audit Finding #38) which surfaced the same value on two
# layers with inconsistent SoT; capabilities is the right home.
LED_MAX_MA: int = 1000


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

    axis_travel_limits_um: Mapping[str, float]
    """Per-axis travel limit in um, populated only for present axes.

    Read-only mapping (MappingProxyType wrapper) so the frozen-dataclass
    immutability contract holds for the contents as well as the field
    binding. A caller passing an absent axis gets KeyError -- which is
    the correct contract per the Rule 8 capability-probe corollary
    (test `axis in caps.axes` first; the travel-limit query is only
    meaningful for present axes).

    Values come from `motion.motorconfig.travel_limit_um(axis)` (mm in
    motorconfig.json, multiplied by 1000 for um). Empty mapping if
    motion driver has no motorconfig (NullMotionBoard) or all axes
    failed to read."""

    pixel_size_um: float
    """Per-scope camera pixel size in um/pixel, sourced from
    motorconfig.json's Optics.PixelSize. Per-installation override of
    the camera SDK's reported value -- some sites adjust this for
    calibration. Used by FOV / scale-bar / coordinate-transform
    helpers. Default 2.0 if motorconfig is unavailable."""

    lens_focal_length_mm: float
    """Tube lens focal length in mm, sourced from motorconfig.json's
    Optics.LensFocalLength. Per-installation override (default Etaluma
    47.8 mm). Used together with pixel_size_um and the objective focal
    length to compute per-objective effective um/pixel. Default 47.8
    if motorconfig is unavailable."""

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

    camera_max_frame_size: 'tuple[int, int]'
    """Maximum camera frame size as ``(width, height)`` in pixels.
    Per-camera-immutable: sourced from the camera driver's
    get_max_frame_size() at boot. (0, 0) when no camera driver is
    connected. Use ``scope.imaging.set_frame_size`` to request a
    smaller-than-max region; this field gives the upper bound."""

    # ---- Cross-cutting feature flags ----
    hardware_features: frozenset[str] = frozenset()
    """Set of hardware-feature tokens this scope advertises. Per Rule 8
    empty-default semantic: empty means 'feature set unknown / no
    features advertised,' not 'feature X is absent.' Use
    ``caps.supports(feature)`` to test for a token; that helper also
    searches has_X / camera_supports_X fields so callers don't need to
    know which surface owns a particular capability.

    Reserved tokens (populated as drivers mature):
      'trigger_in', 'trigger_out'    -- external trigger hardware
      'temperature_sensor'           -- camera temp probe
      'cooled_sensor'                -- TEC / Peltier camera
      'global_shutter'               -- non-rolling shutter
    Tokens are deliberately documented per L2 contract; new tokens
    require a LumascopeSkills entry."""

    def supports(self, feature: str) -> bool:
        """Return True if the scope advertises the named feature.

        Cross-surface helper that the Rule 8 capability-probe corollary
        cites: callers test for a feature by token rather than by
        knowing which surface owns it. Searches the boolean
        `has_<feature>` fields (motion-shape: focus / xy_stage /
        turret) and the boolean `camera_supports_<feature>` fields
        (camera-shape: auto_gain / auto_exposure) for a match. Unknown
        feature names return False, never raise.

        Example:
            caps.supports('turret')      # True if has_turret
            caps.supports('xy_stage')    # True if has_xy_stage
            caps.supports('auto_gain')   # True if camera_supports_auto_gain
            caps.supports('warp_drive')  # False (unknown)

        Also searches the ``hardware_features`` frozenset by token:
        ``caps.supports('trigger_in')`` returns True iff 'trigger_in' is
        in ``caps.hardware_features``. The empty-default contract means
        an empty set yields False for any token -- never raises.
        """
        if getattr(self, f'has_{feature}', False):
            return True
        if getattr(self, f'camera_supports_{feature}', False):
            return True
        if feature in self.hardware_features:
            return True
        return False

    @classmethod
    def from_drivers(cls, motion, led, camera, led_max_ma: int = LED_MAX_MA) -> 'ScopeCapabilities':
        """Build a ScopeCapabilities snapshot from the three drivers.

        Tolerant of None / Null implementations. Never raises -- if a
        driver method blows up or returns something unexpected, the
        corresponding field gets a safe default (empty tuple, empty
        string, False).

        Args:
            motion: A `MotorBoardProtocol` implementation (may be
                NullMotionBoard).
            led: An `LEDBoardProtocol` implementation (may be NullLEDBoard).
            camera: A camera object or None.
            led_max_ma: The API's LED current cap. Defaults to the
                module-level ``LED_MAX_MA`` (1000 mA, matches firmware
                CH_MAX); callers may override per-board if a future
                driver advertises a different cap.
        """
        # Motion
        axes = _probe('detect_present_axes',
                      lambda: tuple(motion.detect_present_axes()),
                      ())
        model = _probe('get_microscope_model',
                       lambda: motion.get_microscope_model() or '',
                       '')

        # Travel limits + optics per present axis (read once at boot;
        # motorconfig is loaded once at driver init and is immutable
        # for the run).
        travel_limits: dict[str, float] = {}
        motorconfig = getattr(motion, 'motorconfig', None)
        if motorconfig is not None:
            for ax in axes:
                limit = _probe(f'travel_limit_um[{ax}]',
                               lambda ax=ax: float(motorconfig.travel_limit_um(ax)),
                               None)
                if limit is not None:
                    travel_limits[ax] = limit
        pixel_size_um = _probe('motorconfig.pixel_size',
                               lambda: float(motorconfig.pixel_size()) if motorconfig is not None else 2.0,
                               2.0)
        lens_focal_length_mm = _probe('motorconfig.lens_focal_length',
                                      lambda: float(motorconfig.lens_focal_length()) if motorconfig is not None else 47.8,
                                      47.8)

        # LED
        led_channels = _probe('led.available_channels',
                              lambda: tuple(led.available_channels()),
                              ())
        led_colors = _probe('led.available_colors',
                            lambda: tuple(led.available_colors()),
                            ())

        # Camera
        camera_model = ''
        camera_supports_auto_gain = False
        camera_supports_auto_exposure = False
        camera_pixel_formats: tuple[str, ...] = ()
        camera_binning_sizes: tuple[int, ...] = ()
        camera_max_exposure_ms = 0
        camera_max_frame_size: tuple[int, int] = (0, 0)
        if camera is not None:
            profile = getattr(camera, 'profile', None)
            if profile is not None:
                camera_model = getattr(profile, 'model_name', '') or ''
                camera_supports_auto_gain = bool(getattr(profile, 'has_auto_gain', False))
                camera_supports_auto_exposure = bool(getattr(profile, 'has_auto_exposure', False))
                camera_pixel_formats = tuple(getattr(profile, 'pixel_formats', ()) or ())
                camera_binning_sizes = tuple(getattr(profile, 'binning_sizes', ()) or ())
                exposure_max_us = getattr(profile, 'exposure_max_us', 0) or 0
                camera_max_exposure_ms = int(exposure_max_us / 1000)
            size = _probe('camera.get_max_frame_size',
                          lambda: camera.get_max_frame_size(),
                          None)
            if size:
                camera_max_frame_size = (int(size.get('width', 0)), int(size.get('height', 0)))

        return cls(
            axes=axes,
            has_focus='Z' in axes,
            has_xy_stage=('X' in axes and 'Y' in axes),
            has_turret='T' in axes,
            motor_model=model,
            axis_travel_limits_um=MappingProxyType(travel_limits),
            pixel_size_um=pixel_size_um,
            lens_focal_length_mm=lens_focal_length_mm,
            led_channels=led_channels,
            led_colors=led_colors,
            led_max_ma=led_max_ma,
            camera_model=camera_model,
            camera_supports_auto_gain=camera_supports_auto_gain,
            camera_supports_auto_exposure=camera_supports_auto_exposure,
            camera_pixel_formats=camera_pixel_formats,
            camera_binning_sizes=camera_binning_sizes,
            camera_max_exposure_ms=camera_max_exposure_ms,
            camera_max_frame_size=camera_max_frame_size,
        )
