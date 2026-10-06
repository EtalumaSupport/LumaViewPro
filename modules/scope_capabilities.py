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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from drivers.exceptions import HardwareError
from lvp_logger import logger

if TYPE_CHECKING:
    from drivers.protocols import LEDBoardProtocol, MotorBoardProtocol
    from modules.layer_record import LayerIdentity


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


def _declared_optics(scope_models: Mapping, model: str) -> dict[str, float]:
    """Return the numeric Optics block the model catalogue declares for `model`.

    The Lumascope Classic line has no motorconfig, so the catalogue is its
    declared optics source (keyed by scope model). A model with no entry,
    or an entry with no Optics block, is a legitimate resolution-order
    branch (the LS850T sources optics from motorconfig; an unknown scope
    has none): an empty mapping tells the caller to fall through to the
    next source. A non-numeric entry is logged and treated as absent so a
    bad value degrades the scale rather than aborting scope bring-up.
    """
    entry = scope_models.get(model) if model else None
    raw = entry.get('Optics', {}) if entry is not None else {}
    try:
        return {key: float(raw[key]) for key in ('PixelSize', 'LensFocalLength') if key in raw}
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f'[CAPABILITIES] catalogue Optics unusable for {model!r}: {e}')
        return {}


def _declared_max_frame(scope_models: Mapping, model: str) -> tuple[int, int] | None:
    """Return the frame maximum the model catalogue declares for `model`, or None.

    A model whose lens images less than its sensor declares the largest frame
    it delivers (``MaxFrame``, unbinned pixels); the LS560's lens is why its
    frame ends at 1700. Most models declare none, and the camera's own maximum
    is the scope's. A malformed entry is logged and treated as absent.
    """
    entry = scope_models.get(model) if model else None
    raw = entry.get('MaxFrame') if entry is not None else None
    if raw is None:
        return None
    try:
        return (int(raw['width']), int(raw['height']))
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f'[CAPABILITIES] catalogue MaxFrame unusable for {model!r}: {e}')
        return None


def _smallest_frame(*sizes: tuple[int, int] | None) -> tuple[int, int] | None:
    """The smallest of the known frame maxima, per side; None when none is known."""
    known = [size for size in sizes if size is not None]
    if not known:
        return None
    return (min(w for w, _ in known), min(h for _, h in known))


def _resolve_pixel_size_um(motorconfig, optics: dict, camera) -> float | None:
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
    optics_px = optics.get('PixelSize')
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


def _resolve_lens_focal_length_mm(motorconfig, optics: dict) -> float | None:
    """Resolve tube-lens focal length (mm) from the first real source.

    Order: motorconfig Optics (LS820/850/850T) -> scopes.json Optics
    (Classic) -> None. No camera source (a lens is not a sensor property)
    and no hardcoded fallback.
    """
    if motorconfig is not None:
        mc = _probe('motorconfig.lens_focal_length', motorconfig.lens_focal_length, None)
        if mc is not None:
            return float(mc)
    optics_fl = optics.get('LensFocalLength')
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

    model: str
    """The scope model this scope runs as: the one its layer identity
    settled on (the board's report, else the operator's selection), or
    empty string when there is none. Fixed for the life of the scope; a
    new selection applies at the next start."""

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
    """Maximum LED current per channel, in mA, as published by the connected
    LED driver (`led.max_ma()`); 0 when no driver answers."""

    # ---- Camera ----
    camera_model: str | None
    """The model the camera reports at connect; None when no camera is
    connected or it did not say. Read from the driver, not the profile: an
    unrecognised camera is given a default profile whose name is a
    placeholder, not the camera's."""

    camera_serial_number: str | None
    """The serial number the camera reports at connect, or None."""

    camera_timestamp_tick_hz: int | None
    """The rate of the camera's frame timestamp clock, in Hz, or None for a
    camera whose frames carry no timestamp."""

    camera_supports_auto_gain: bool
    camera_supports_auto_exposure: bool

    camera_pixel_formats: tuple[str, ...]
    camera_binning_sizes: tuple[int, ...]

    camera_max_frame_size: tuple[int, int] | None
    """The largest frame the scope delivers, unbinned, as ``(width, height)``
    in pixels: the smallest of the sensor size the camera's profile
    documents, the maximum the camera reports at connect, and the model's
    catalogue ``MaxFrame`` (a lens that images less than the sensor: the
    LS560's 1700), each where it is known. The profile stays in the
    comparison because a camera can report a few rows and columns more than
    its documented sensor (the LS850T's daA3840 reports 3860 x 2178 against
    3840 x 2160). None when none of them is known: no camera, or its boot
    read failed (logged at warning). ``scope.imaging.set_frame_size``
    refuses a frame above it, divided by the binning in force."""

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

    camera_supports_black_level: bool = False
    """True if the camera's black level can be set
    (``scope.imaging.set_black_level``). The black level is read with
    ``scope.imaging.get_black_level`` on any camera that reports one,
    settable or not (the FX2 reports its fixed Row Black Target)."""

    # ---- Cross-cutting feature flags ----
    has_firmware_stim: bool = False
    """True when the LED firmware advertises the STIM pulse-train command
    (LED firmware v3.0.8+). Probed at boot via `led.supports_firmware_stim()`.
    Host-side pulse scheduling is unreliable below ~20 ms pulse width
    because the USB-UART bridge batches back-to-back fast-path writes;
    firmware STIM eliminates the bridge-batching problem by running the
    pulse train inside the LED firmware with sub-microsecond pulse-edge
    accuracy. Caller gates with `caps.has_firmware_stim`."""

    @classmethod
    def from_drivers(
        cls,
        motion: MotorBoardProtocol,
        led: LEDBoardProtocol,
        camera: object | None,
        layer_identity: LayerIdentity,
        scope_models: Mapping,
    ) -> ScopeCapabilities:
        """Build a ScopeCapabilities snapshot from the three drivers.

        Internal constructor -- called by Lumascope at init and not part
        of the L2 API surface (L2 callers read the built snapshot via
        `scope.capabilities`).

        Tolerant of None / Null implementations. Never raises -- if a
        driver method blows up or returns something unexpected, the
        corresponding field gets its absent value (empty tuple, None,
        False).

        Args:
            motion: A `MotorBoardProtocol` implementation (may be
                NullMotionBoard).
            led: An `LEDBoardProtocol` implementation (may be NullLEDBoard).
            camera: A camera object or None.
            layer_identity: The scope's resolved identity; its model is
                the capabilities' model, so the two cannot disagree.
            scope_models: The scope's model catalogue, read once at its
                construction; the model's declared optics come from it.
        """
        # Motion
        axes = _probe('detect_present_axes', lambda: tuple(motion.detect_present_axes()), ())
        model = layer_identity.model or ''

        # Optics (read once at boot; motorconfig is loaded once at driver
        # init and is immutable for the run).
        motorconfig = getattr(motion, 'motorconfig', None)
        optics = _declared_optics(scope_models, model)
        pixel_size_um = _resolve_pixel_size_um(motorconfig, optics, camera)
        lens_focal_length_mm = _resolve_lens_focal_length_mm(motorconfig, optics)

        # LED
        led_channels = _probe('led.available_channels', lambda: tuple(led.available_channels()), ())
        # Colour NAMES come from the unit's resolved layer identity, not
        # the driver: the driver knows which board channels it can drive,
        # while which layer names exist (and what they drive) is unit
        # data. A scope with no resolved identity honestly reports no
        # colour names -- the channels stay visible via led_channels.
        led_colors = tuple(r.key_name for r in layer_identity.layers if r.led_channel)
        has_firmware_stim = _probe(
            'led.supports_firmware_stim',
            lambda: bool(led.supports_firmware_stim()),
            False,
        )
        # The cap is the driver's to publish; there is no value to assume
        # in its place. A driver that does not answer leaves no legal
        # current above zero.
        led_max_ma = _probe('led.max_ma', lambda: int(led.max_ma()), 0)

        # Camera
        camera_model: str | None = None
        camera_serial_number: str | None = None
        camera_timestamp_tick_hz: int | None = None
        camera_supports_auto_gain = False
        camera_supports_auto_exposure = False
        camera_pixel_formats: tuple[str, ...] = ()
        camera_binning_sizes: tuple[int, ...] = ()
        camera_max_frame_size: tuple[int, int] | None = None
        is_color_native = False
        native_bit_depth = 16
        camera_supports_conversion_gain_mode = False
        camera_supports_line_noise_reduction = False
        camera_supports_black_level = False
        if camera is not None:
            camera_model = camera.model_name or None
            camera_serial_number = camera.device_serial or None
            tick_hz = camera.timestamp_tick_frequency_hz
            camera_timestamp_tick_hz = int(tick_hz) if tick_hz is not None else None
            documented: tuple[int, int] | None = None
            profile = getattr(camera, 'profile', None)
            if profile is not None:
                camera_supports_auto_gain = bool(getattr(profile, 'has_auto_gain', False))
                camera_supports_auto_exposure = bool(getattr(profile, 'has_auto_exposure', False))
                camera_pixel_formats = tuple(getattr(profile, 'pixel_formats', ()) or ())
                camera_binning_sizes = tuple(getattr(profile, 'binning_sizes', ()) or ())
                native = getattr(profile, 'native_resolution', None)
                if native:
                    documented = (int(native['width']), int(native['height']))
            size = _probe('camera.get_max_frame_size', lambda: camera.get_max_frame_size(), None)
            reported = (int(size['width']), int(size['height'])) if size else None
            camera_max_frame_size = _smallest_frame(
                documented, reported, _declared_max_frame(scope_models, model)
            )
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
            camera_supports_black_level = _probe(
                'camera.supports_black_level',
                lambda: bool(camera.supports_black_level()),
                False,
            )
            # Record the detected low-noise toggles so a support bundle shows
            # whether they were available on this camera without debug mode.
            logger.info(
                f'[CAPABILITIES] camera={camera_model!r} '
                f'conversion_gain_mode={camera_supports_conversion_gain_mode} '
                f'line_noise_reduction={camera_supports_line_noise_reduction} '
                f'black_level={camera_supports_black_level}'
            )

        return cls(
            axes=axes,
            has_focus='Z' in axes,
            has_xy_stage=('X' in axes and 'Y' in axes),
            has_turret='T' in axes,
            model=model,
            pixel_size_um=pixel_size_um,
            lens_focal_length_mm=lens_focal_length_mm,
            led_channels=led_channels,
            led_colors=led_colors,
            led_max_ma=led_max_ma,
            has_firmware_stim=has_firmware_stim,
            camera_model=camera_model,
            camera_serial_number=camera_serial_number,
            camera_timestamp_tick_hz=camera_timestamp_tick_hz,
            camera_supports_auto_gain=camera_supports_auto_gain,
            camera_supports_auto_exposure=camera_supports_auto_exposure,
            camera_pixel_formats=camera_pixel_formats,
            camera_binning_sizes=camera_binning_sizes,
            camera_max_frame_size=camera_max_frame_size,
            is_color_native=is_color_native,
            native_bit_depth=native_bit_depth,
            camera_supports_conversion_gain_mode=camera_supports_conversion_gain_mode,
            camera_supports_line_noise_reduction=camera_supports_line_noise_reduction,
            camera_supports_black_level=camera_supports_black_level,
        )
