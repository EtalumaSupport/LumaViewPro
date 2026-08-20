# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope capability dataclass -- the canonical "what does this scope have" query.

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
"motion.has_turret()"). Callers need a single place where this information
lives -- query capabilities, don't assume.

ScopeCapabilities is that place. It's a frozen dataclass built once at
init from the three drivers (motion / LED / camera). Callers read fields
directly; the per-API alias wrappers are retired, so
`scope.capabilities.*` is the single spelling.

**Scope:** ScopeCapabilities contains static hardware *structure* (what
axes exist, what LED channels exist, what camera profile is loaded) --
things that don't change at runtime. It deliberately does NOT include
live connection state (`motor_connected`, `led_connected`, etc.) -- those
must reflect disconnects at runtime and stay as live Lumascope
properties, not frozen snapshot fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from collections.abc import Callable, Mapping

from drivers.exceptions import HardwareError
from lvp_logger import logger
from modules.path_utils import resolve_data_file


def _probe(label: str, fn: Callable[[], Any], fallback: Any) -> Any:
    """Run a capability probe; return fallback on expected absence; log
    on hardware fault. Other exceptions (TypeErrors, KeyErrors from buggy
    code) propagate so they surface for debugging.

    Catches:
        - AttributeError / NotImplementedError: feature absent.
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
# which surfaced the same value on two layers with inconsistent SoT;
# capabilities is the right home.
LED_MAX_MA: int = 1000


def _scopes_json_optics(model: str) -> dict[str, float]:
    """Return the numeric Optics block declared for `model` in scopes.json.

    The Lumascope Classic line has no motorconfig, so scopes.json is its
    declared optics source (keyed by scope model). Returns an empty mapping
    when the file, the model entry, or the Optics block is absent -- the
    caller then falls through to the next source in the resolution order. A
    non-numeric or unreadable entry is logged and treated as absent so a
    corrupt data file degrades the scale rather than aborting scope bring-up.
    """
    if not model:
        return {}
    try:
        with open(resolve_data_file('scopes.json'), encoding='utf-8') as f:
            scopes = json.load(f)
        # A model with no scopes.json entry, or an entry with no Optics block,
        # is a legitimate resolution-order branch (the LS850T sources optics
        # from motorconfig; an unknown scope has none) -- an empty mapping tells
        # the caller to fall through to the next source, not a missing value.
        raw = scopes.get(model, {}).get('Optics', {})
        return {key: float(raw[key]) for key in ('PixelSize', 'LensFocalLength') if key in raw}
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f'[CAPABILITIES] scopes.json Optics unreadable for {model!r}: {e}')
        return {}


def _resolve_pixel_size_um(motorconfig, model: str, camera) -> float | None:
    """Resolve image pixel pitch (um) from the first real source.

    Order: motorconfig Optics (LS820/850/850T) -> scopes.json Optics
    (Classic) -> the camera profile / SDK-reported pitch -> None. No
    hardcoded fallback: a scope that reports none of these cannot measure,
    and None is the honest signal for that.
    """
    if motorconfig is not None:
        mc = _probe('motorconfig.pixel_size', motorconfig.pixel_size, None)
        if mc is not None:
            return float(mc)
    optics_px = _scopes_json_optics(model).get('PixelSize')
    if optics_px is not None:
        return optics_px
    if camera is not None:
        profile = getattr(camera, 'profile', None)
        px = getattr(profile, 'pixel_size_um', None) if profile is not None else None
        # A generic profile carries 0.0 until the driver fills it live from
        # the SDK's SensorPixelWidth; only a real, positive pitch counts.
        if px:
            return float(px)
    return None


def _resolve_lens_focal_length_mm(motorconfig, model: str) -> float | None:
    """Resolve tube-lens focal length (mm) from the first real source.

    Order: motorconfig Optics (LS820/850/850T) -> scopes.json Optics
    (Classic) -> None. No camera source (a lens is not a sensor property)
    and no hardcoded fallback.
    """
    if motorconfig is not None:
        mc = _probe('motorconfig.lens_focal_length', motorconfig.lens_focal_length, None)
        if mc is not None:
            return float(mc)
    optics_fl = _scopes_json_optics(model).get('LensFocalLength')
    if optics_fl is not None:
        return optics_fl
    return None


@dataclass(frozen=True)
class ScopeCapabilities:
    """Immutable snapshot of what a scope has.

    Built once at `Lumascope.__init__` from the three drivers. Fields
    are tuples (not lists) to reinforce immutability -- a caller that
    wants to mutate would have to shallow-copy into their own list.
    """

    # ---- Motion ----
    axes: tuple[str, ...]
    """Axes physically present on this scope -- from
    `motion.detect_present_axes()`. e.g. ('Z',) for LS820/LVC LS620,
    ('X','Y','Z') for LS850, ('X','Y','Z','T') for LS850T, () for no
    motor hardware."""

    has_focus: bool  # 'Z' in axes
    has_xy_stage: bool  # 'X' and 'Y' in axes
    has_turret: bool  # 'T' in axes

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

    pixel_size_um: float | None
    """Per-scope camera pixel size in um/pixel, resolved from the first
    real source: motorconfig Optics.PixelSize (LS820/850/850T) ->
    scopes.json Optics.PixelSize (Classic) -> the camera profile /
    SDK-reported pitch -> None. None when no source can supply it: the
    scope cannot measure, and consumers (FOV / scale bar / coordinate
    transform) degrade honestly rather than using an invented scale.
    Never a hardcoded default -- a guessed pixel size is written into
    every image and cannot be told from a measured one."""

    lens_focal_length_mm: float | None
    """Tube lens focal length in mm, resolved from motorconfig
    Optics.LensFocalLength -> scopes.json Optics.LensFocalLength
    (Classic) -> None. Used with pixel_size_um and the objective focal
    length to compute per-objective effective um/pixel. None when no
    source supplies it (there is no camera source -- a lens is not a
    sensor property); never a hardcoded default."""

    # ---- LED ----
    led_channels: tuple[int, ...]
    """LED channel indices available -- from `led.available_channels()`.
    RP2040 = (0,1,2,3,4,5), FX2/LVC = (0,1,2,3). NullLEDBoard also returns
    the 6-channel set for Rule 8 silent-noop compatibility."""

    led_colors: tuple[str, ...]
    """Color names available -- from `led.available_colors()`."""

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

    camera_max_frame_size: tuple[int, int]
    """Maximum camera frame size as ``(width, height)`` in pixels.
    Per-camera-immutable: sourced from the camera driver's
    get_max_frame_size() at boot. (0, 0) when no camera driver is
    connected. Use ``scope.imaging.set_frame_size`` to request a
    smaller-than-max region; this field gives the upper bound."""

    is_color_native: bool = False
    """True if the camera natively produces 3-channel color frames
    (Bayer-decoded RGB out of the SDK). False for mono cameras (the
    LVP shipping fleet -- all Pylon and IDS sensors used to date).
    Defaults to False so unknown/missing camera path treats output
    as mono."""

    native_bit_depth: int = 16
    """Container bit depth that the driver delivers to downstream code.
    Mono10 / Mono12 / Mono16 packed into uint16 buffers all report 16
    (the container width, not the payload bits). Sensors that report
    Mono8 directly (IDS IMX676 -- U3-34L0XCP-M) report 8. Drives
    buffer sizing decisions in pipeline stages."""

    camera_supports_conversion_gain_mode: bool = False
    """True if the camera exposes a switchable sensor conversion-gain
    mode (High = low read noise / narrow range, Low = wide range). Gates
    the UI toggle. Pylon Bsl feature; absent on cameras without it."""

    camera_supports_line_noise_reduction: bool = False
    """True if the camera exposes the line-noise-reduction filter (smooths
    horizontal stripe artifacts). Gates the UI toggle. Pylon Bsl feature;
    absent on cameras without it."""

    # ---- Cross-cutting feature flags ----
    has_firmware_stim: bool = False
    """True when the LED firmware advertises the STIM pulse-train command
    (LED firmware v3.0.8+). Probed at boot via `led.supports_firmware_stim()`.
    Host-side pulse scheduling is unreliable below ~20 ms pulse width
    because the USB-UART bridge batches back-to-back fast-path writes;
    firmware STIM eliminates the bridge-batching problem by running the
    pulse train inside the LED firmware with sub-microsecond pulse-edge
    accuracy. Caller gates with `caps.supports('firmware_stim')`."""

    has_motor_stop: bool = False
    """True when the motor firmware implements the STOP emergency-stop
    command (sets target=actual on every axis). Probed at boot via
    `motion.supports_motor_stop()`. Field firmware from 2024 replies
    ERROR to STOP; the driver returns False from motor_stop there and
    motors latch on host disconnect instead."""

    has_fan: bool = False
    """True when the motor firmware implements the fan commands
    (FAN:<duty> PWM control + FANSPEED tachometer query). Probed at
    boot via `motion.supports_fan()`."""

    has_diagnostics: bool = False
    """True when the motor firmware implements the diagnostic queries
    (VOLTAGE power-rail check, DRVSTAT_<axis> driver status). Probed
    at boot via `motion.supports_diagnostics()`."""

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

        Cross-surface helper for the capability-probe pattern: callers
        test for a feature by token rather than by knowing which surface
        owns it. Searches the boolean
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
        return feature in self.hardware_features

    @classmethod
    def from_drivers(cls, motion, led, camera, led_max_ma: int = LED_MAX_MA) -> ScopeCapabilities:
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
        axes = _probe('detect_present_axes', lambda: tuple(motion.detect_present_axes()), ())
        model = _probe('get_microscope_model', lambda: motion.get_microscope_model() or '', '')

        # Travel limits + optics per present axis (read once at boot;
        # motorconfig is loaded once at driver init and is immutable
        # for the run).
        travel_limits: dict[str, float] = {}
        motorconfig = getattr(motion, 'motorconfig', None)
        if motorconfig is not None:
            for ax in axes:
                limit = _probe(
                    f'travel_limit_um[{ax}]',
                    lambda ax=ax: float(motorconfig.travel_limit_um(ax)),
                    None,
                )
                if limit is not None:
                    travel_limits[ax] = limit
        pixel_size_um = _resolve_pixel_size_um(motorconfig, model, camera)
        lens_focal_length_mm = _resolve_lens_focal_length_mm(motorconfig, model)

        # Motor firmware command families. Probe-and-cache on the
        # driver: one wire exchange each at boot (motors idle), then
        # cached for the life of the connection.
        has_motor_stop = _probe(
            'motion.supports_motor_stop',
            lambda: bool(motion.supports_motor_stop()),
            False,
        )
        has_fan = _probe(
            'motion.supports_fan',
            lambda: bool(motion.supports_fan()),
            False,
        )
        has_diagnostics = _probe(
            'motion.supports_diagnostics',
            lambda: bool(motion.supports_diagnostics()),
            False,
        )

        # LED
        led_channels = _probe('led.available_channels', lambda: tuple(led.available_channels()), ())
        led_colors = _probe('led.available_colors', lambda: tuple(led.available_colors()), ())
        has_firmware_stim = _probe(
            'led.supports_firmware_stim',
            lambda: bool(led.supports_firmware_stim()),
            False,
        )

        # Camera
        camera_model = ''
        camera_supports_auto_gain = False
        camera_supports_auto_exposure = False
        camera_pixel_formats: tuple[str, ...] = ()
        camera_binning_sizes: tuple[int, ...] = ()
        camera_max_exposure_ms = 0
        camera_max_frame_size: tuple[int, int] = (0, 0)
        is_color_native = False
        native_bit_depth = 16
        camera_supports_conversion_gain_mode = False
        camera_supports_line_noise_reduction = False
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
            size = _probe('camera.get_max_frame_size', lambda: camera.get_max_frame_size(), None)
            if size:
                camera_max_frame_size = (int(size.get('width', 0)), int(size.get('height', 0)))
            is_color_native = bool(getattr(camera, 'is_color_native', False))
            native_bit_depth = int(getattr(camera, 'native_bit_depth', 16))
            camera_supports_conversion_gain_mode = _probe(
                'camera.supports_conversion_gain_mode',
                lambda: bool(camera.supports_conversion_gain_mode()),
                False,
            )
            camera_supports_line_noise_reduction = _probe(
                'camera.supports_line_noise_reduction',
                lambda: bool(camera.supports_line_noise_reduction()),
                False,
            )
            # Record the detected low-noise toggles so a support bundle shows
            # whether they were available on this camera without debug mode.
            logger.info(
                f'[CAPABILITIES] camera={camera_model!r} '
                f'conversion_gain_mode={camera_supports_conversion_gain_mode} '
                f'line_noise_reduction={camera_supports_line_noise_reduction}'
            )

        return cls(
            axes=axes,
            has_focus='Z' in axes,
            has_xy_stage=('X' in axes and 'Y' in axes),
            has_turret='T' in axes,
            motor_model=model,
            has_motor_stop=has_motor_stop,
            has_fan=has_fan,
            has_diagnostics=has_diagnostics,
            axis_travel_limits_um=MappingProxyType(travel_limits),
            pixel_size_um=pixel_size_um,
            lens_focal_length_mm=lens_focal_length_mm,
            led_channels=led_channels,
            led_colors=led_colors,
            led_max_ma=led_max_ma,
            has_firmware_stim=has_firmware_stim,
            camera_model=camera_model,
            camera_supports_auto_gain=camera_supports_auto_gain,
            camera_supports_auto_exposure=camera_supports_auto_exposure,
            camera_pixel_formats=camera_pixel_formats,
            camera_binning_sizes=camera_binning_sizes,
            camera_max_exposure_ms=camera_max_exposure_ms,
            camera_max_frame_size=camera_max_frame_size,
            is_color_native=is_color_native,
            native_bit_depth=native_bit_depth,
            camera_supports_conversion_gain_mode=camera_supports_conversion_gain_mode,
            camera_supports_line_noise_reduction=camera_supports_line_noise_reduction,
        )
