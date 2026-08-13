#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib
import warnings
from typing import TYPE_CHECKING

from lvp_logger import logger

# Import Lumascope Hardware files
from drivers.motorboard import MotorBoard
from drivers.ledboard import LEDBoard
from modules.lumascope_api import _constants as _api_constants
import modules.image_mode as image_mode

try:
    from drivers.idscamera import IDSCamera
except ImportError as _ids_exc:
    IDSCamera = None
    # The reason MUST reach the log: the driver silently never registers,
    # so without it an IDS scope just "has no camera" -- a swallowed
    # bundling gap in a frozen build cost a full client misdiagnosis.
    logger.warning(f'[SCOPE API ] IDS camera driver unavailable: {_ids_exc}')
# FX2 (Lumaview Classic LS560/LS620/LS720) -- the import side-effect is
# the entire point: it fires the @camera_registry.register('fx2') and
# @led_registry.register('fx2') decorators inside the module. Nothing
# in this file references fx2driver names directly; the registry
# instantiates FX2Camera + FX2LEDController via 'auto' fallthrough when
# Pylon/IDS aren't found. Wrapped in try/except so dev machines without
# pyusb / libusb1 don't crash LVP at startup (matches IDS pattern above).
try:
    import drivers.fx2driver  # noqa: F401
except ImportError as _fx2_exc:
    # Same silent-degradation shape as the IDS guard above: without this
    # line a Classic scope's missing camera has no named cause anywhere.
    logger.warning(f'[SCOPE API ] FX2 (Classic) drivers unavailable: {_fx2_exc}')
from drivers.camera import Camera

# Registration-only imports: loading each driver module fires its
# @*_registry.register(...) decorator so the registry can instantiate it
# by kind ('pylon', 'sim') via create(). No name below is referenced
# directly here; dropping these empties the registry -- simulate mode then
# finds no 'sim' drivers and startup aborts.
from drivers.pyloncamera import PylonCamera  # noqa: F401
from drivers.simulated_camera import SimulatedCamera  # noqa: F401
from drivers.simulated_motorboard import SimulatedMotorBoard  # noqa: F401
from drivers.simulated_ledboard import SimulatedLEDBoard  # noqa: F401
from drivers.null_motorboard import NullMotionBoard
from drivers.null_ledboard import NullLEDBoard
from drivers.protocols import MotorBoardProtocol, LEDBoardProtocol
from drivers.registry import motor_registry, led_registry, camera_registry
import modules.binning as binning
from modules.exceptions import CameraSettingRejected
from modules.scope_capabilities import ScopeCapabilities

# Import additional libraries
import logging as _logging

from modules.notification_center import notifications

_api_log = _logging.getLogger('LVP.api')

# PRE-RELEASE 4-mechanism warning bundle: this is the runtime
# FutureWarning piece. The other three are the README banner, the
# LumascopeSkills.md preface, and the CHANGELOG note. All four
# retire together in one commit at the freeze trigger; do not
# retire this one without the bundle.
_PRE_RELEASE_WARNING_FIRED = False
_PRE_RELEASE_WARNING_TEXT = (
    'The Lumascope SDK API is PRE-RELEASE and subject to breaking '
    'changes through LVP 4.2 (Wave 7 sub-API decomposition, capability '
    '+ wire-contract changes, REST endpoint conventions). See '
    'LumaViewPro/docs/LumascopeSkills.md preface for the migration '
    'plan. Contact Etaluma support if you depend on this API.'
)


def _fire_pre_release_warning(stacklevel: int = 3) -> None:
    """Fire the PRE-RELEASE runtime FutureWarning once per process.

    Called from `Lumascope.__init__` and from `ScopeSession.create` /
    `create_headless` so any L2 entry point trips the warning, even
    callers that bypass `Lumascope` directly (e.g. tests that mock
    the scope).

    stacklevel default is 3: the caller of __init__ / create is two
    frames above this helper. Callers that wrap deeper can override.
    """
    global _PRE_RELEASE_WARNING_FIRED
    if _PRE_RELEASE_WARNING_FIRED:
        return
    _PRE_RELEASE_WARNING_FIRED = True
    warnings.warn(_PRE_RELEASE_WARNING_TEXT, FutureWarning, stacklevel=stacklevel)


# Protocol is imported function-locally in the load/create methods (keeps
# the data class off this module's import surface); declare it for the
# return annotations without adding a runtime import.
if TYPE_CHECKING:
    from modules.protocol import Protocol


# AxisState lives in the package's leaf _constants.py (so sub-API modules can
# import it without depending on this composition root). Re-exported here as a
# module-level name so `from modules.lumascope_api import AxisState` and
# `from modules.lumascope_api._lumascope import AxisState` keep working.
AxisState = _api_constants.AxisState


# ---------------------------------------------------------------------------
# Notify-on-failure helpers
#
# #632/#539 introduced `_try_connect_board` to replace the silent
# `try/except: NullBoard()` pattern that hid LED-side failures. The
# helpers are hoisted to module scope so they can be reused by
# `__init__`, `create_diagnostic`, and any future connect path without
# duplicating the error-class routing. The module-scope helpers are
# the single source of truth; call sites should be one-liners.
# ---------------------------------------------------------------------------


def _notify_board_failure(label, short, message):
    """Surface a board-connect failure to the user via notification_center.

    Safe to call from any thread. Falls back to a debug log if the
    notification_center import fails (e.g. during very-early startup).
    """
    try:
        from modules.notification_center import notifications

        notifications.warning(label, f'{label} {short}', message)
    except Exception as nx:
        logger.debug(f'{label}: notification center unavailable: {nx}')


def _try_connect_board(label, ctor, null_ctor):
    """Construct a board, classify any failure, notify the user, fall back
    to `null_ctor()` so callers don't crash on missing hardware.

    The board constructor (LEDBoard / MotorBoard / ...) calls
    SerialBoard.connect() internally, which catches its OWN exceptions and
    logs without re-raising. That means a PermissionError on open leaves
    `board.found=True` (port was discovered) but `board.driver=None` (open
    failed) -- we detect that here and surface it as a clear failure instead
    of silently substituting Null*.

    Every case logs visibly and notifies the user with an actionable,
    error-class-specific message.
    """
    try:
        board = ctor()
        if not getattr(board, 'found', False):
            logger.error(f'{label}: not detected on USB')
            _notify_board_failure(
                label, 'not detected', f'{label} not found on USB. Check USB cable and 24V power.'
            )
            return null_ctor()
        if getattr(board, 'driver', None) is None:
            logger.error(
                f'{label}: detected on {board.port} but driver failed to open '
                f'(port may be held by another program -- Thonny, etc.)'
            )
            _notify_board_failure(
                label,
                'port in use or unreachable',
                f'{label} detected on {board.port} but the port could not be opened. '
                f'Close other programs holding the port (Thonny, serial monitors), '
                f'then restart LVP.',
            )
            return null_ctor()
        # Surface board-specific post-connect safety failures. LEDBoard
        # uses last_safety_off_error to report a connect-time LEDS_OFF
        # send failure (sample safety -- pre-v3.0.4 firmware can leave
        # channels stuck on, photobleaching the sample). Caller sees a
        # clear notification rather than the warning-level log getting
        # buried.
        safety_err = getattr(board, 'last_safety_off_error', None)
        if safety_err:
            _notify_board_failure(
                label,
                'safety LEDS_OFF failed',
                f'{label} connected but the safety LEDS_OFF command did '
                f'not complete ({safety_err}). If the LEDs are stuck on, '
                f'turn off illumination manually before placing a sample.',
            )
        return board
    except PermissionError as e:
        logger.error(f'{label}: PermissionError opening port: {e}')
        _notify_board_failure(
            label,
            'port in use',
            f'{label} port is in use by another program (e.g. Thonny). '
            f'Close the other program and restart LVP to reconnect.',
        )
        return null_ctor()
    except FileNotFoundError as e:
        logger.error(f'{label}: FileNotFoundError on port: {e}')
        _notify_board_failure(
            label, 'port not found', f'{label} port disappeared during connect. Check USB cable.'
        )
        return null_ctor()
    except Exception as e:
        logger.error(f'{label}: connect failed: {type(e).__name__}: {e}')
        _notify_board_failure(
            label,
            'connect failed',
            f'Could not connect to {label}. Check the USB cable and 24V power, then restart LVP.',
        )
        return null_ctor()


def _is_total_cold_start(led_driver, motion_driver) -> bool:
    """True when LED + motor have already both fallen back to Null* drivers,
    which means the about-to-fail camera will trigger the
    no_hardware path. In that case the per-component notifications
    are redundant -- the consolidated 'No hardware detected' popup
    in lumaviewpro.py says it all -- so the individual notifications
    are skipped to avoid 4 popups stacking on top of each other.
    """
    return isinstance(led_driver, NullLEDBoard) and isinstance(motion_driver, NullMotionBoard)


def _notify_camera_failure(exc, *, suppress_if_cold_start: bool = False):
    """Surface camera-init failure to the user.

    The camera registry raises a variety of exception types depending on
    which backend (pypylon, ids_peak, FX2, simulated). pypylon's
    RuntimeException for "camera already open in another application"
    is the high-frequency case that Pylon Viewer / a second LVP instance
    produces and deserves a dedicated message.
    """
    exc_type = type(exc).__name__
    # Don't import pypylon at module load (adds cold-start time on
    # non-Pylon rigs). Match by type name string instead.
    if exc_type in ('RuntimeException', 'GenericException', 'LogicalErrorException'):
        title = 'Camera in use'
        body = (
            'Camera appears to be open in another application '
            '(Pylon Viewer, another LVP instance, etc.). '
            'Close it and restart LVP.'
        )
    elif isinstance(exc, PermissionError):
        title = 'Camera port in use'
        body = 'Camera port is in use by another program. Close the other program and restart LVP.'
    elif isinstance(exc, FileNotFoundError):
        title = 'Camera not detected'
        body = 'Camera not found. Check USB cable and power.'
    else:
        title = 'Camera not initialized'
        body = (
            'Could not connect to the camera. '
            'Check USB cable, power, and close other programs that '
            'may hold the camera.'
        )
    if suppress_if_cold_start:
        # Cold-start with no hardware -- caller has already detected
        # this is the third strike and a consolidated "No hardware
        # detected" popup will fire from lumaviewpro.on_start. Per-
        # component popups stacking with the consolidated one is the
        # 4-popup spam Eric reported.
        logger.warning(
            f'[SCOPE API ] Camera not initialized (suppressed user '
            f'notification, no_hardware path will fire consolidated): '
            f'{title}: {body}'
        )
        return
    _notify_board_failure('Camera', title, body)


class Lumascope:
    # --- Input validation constants ---
    # `LED_MAX_MA` has been retired here. Canonical home is
    # `modules.scope_capabilities.LED_MAX_MA` (also surfaced at
    # `scope.capabilities.led_max_ma`). Callers that need the cap
    # read from capabilities; the class constant created a parallel
    # SoT with the same value.
    # LED channel set comes from self._led_driver.available_channels() -- varies by
    # Canonical home for these is `_constants.py`; alias on the class so
    # existing callers (`scope._VALID_AXIS_NAMES`, `Lumascope.MOTOR_POSITION_LIMIT`)
    # keep working. Sub-API modules import from `_constants.py` directly
    # to avoid a circular dep with this file.
    _VALID_AXIS_NAMES = _api_constants._VALID_AXIS_NAMES
    MOTOR_POSITION_LIMIT = _api_constants.MOTOR_POSITION_LIMIT

    def _init_minimal(self, simulated: bool) -> None:
        """Shared init for state slots both __init__ and create_diagnostic need.

        Sets the non-driver state that every Lumascope instance must
        carry: transformers, locks, camera cache, objective state slots,
        executor slot defaults, source path. Both __init__ and
        create_diagnostic call this first; each then does its
        driver-connection-specific work.

        Pre-#35, create_diagnostic open-coded a subset of these
        assignments and left ~12 attributes unset, which made diagnostic
        instances second-class. Centralizing here makes the slot list
        the single point of truth.
        """
        self._simulated = simulated

        # Driver slot defaults -- __init__ overrides _camera_driver with
        # the real driver; create_diagnostic leaves it None.
        self._camera_driver = None

        # Settings-host state (_labware / _objective / _objective_id /
        # _turret_config / _stage_offset) plus its helpers
        # (_objectives_loader / _coordinate_transformer) live on
        # self.runtime_state (constructed below in __init__ /
        # create_diagnostic). _state_lock + _cam_lock + ImagingAPI's
        # own caches live on self.imaging. _last_turret_position lives
        # on self.motion. engineering_mode lives on the app context
        # (ctx.engineering_mode).

        # Executor slot defaults (registered post-construction via
        # register_executors / register_executor_bundle)
        self._camera_executor = None
        self._io_executor = None
        self._file_io_executor = None
        self._executor_bundle = None
        self._source_path = None

        # Metrics logger pre-constructed in __init__; diagnostic mode
        # leaves it None.
        self.metrics_logger = None

    def __init__(
        self,
        simulate: bool = False,
        camera_type: str = 'auto',
        register_atexit: bool = True,
        register_metrics: bool = True,
        sim_model: str | None = None,
    ):
        """Initialize Microscope.

        Args:
            simulate: If True, use simulated hardware (no USB devices needed).
            camera_type: Camera registry kind. 'auto' (default) tries the
                registered real cameras in descending priority order
                (Pylon -> IDS today). Accepted explicit values: 'pylon',
                'ids', 'sim', or any other key registered in
                `drivers/registry.py::camera_registry`. Post-B2 this is
                the only parameter the caller needs to steer driver
                selection -- motion and LED drivers always use 'auto'.
            register_atexit: If True (default), register a Python atexit
                hook that turns off all LEDs and disconnects on
                interpreter shutdown. Tests that construct Lumascope
                outside the Kivy app should leave this enabled -- the LED
                stays on if a test crashes mid-LED-on otherwise. Set to
                False only when the caller has its own equivalent
                shutdown path that supersedes the atexit hook.
            register_metrics: If True (default), construct a
                MetricsLogger on this Lumascope. Doesn't START it --
                callers must call ``self.metrics_logger.start(scheduler)``
                with an environment-appropriate Scheduler (Kivy app
                uses KivyClockScheduler, REST/headless use
                ThreadingTimerScheduler). Tests that don't need
                periodic logging set False.
            sim_model: When simulating, the scope model the simulated
                motor board reports (e.g. 'LS850', 'LS850T'). Selects
                which axes the simulated scope presents -- an LS850 has
                no turret, an LS850T does -- so capabilities.axes reflect
                the chosen model end to end. Ignored when simulate is
                False; defaults to the 'microscope' setting then 'LS850T'.
        """
        _fire_pre_release_warning()

        # Shared state-slot init (audit #35) -- transformers, locks,
        # camera cache, objective/turret state, executor slot defaults.
        # Driver construction + sub-API wiring happen below.
        self._init_minimal(simulated=simulate)

        # LED state slots (_led_listeners, _led_state, _led_owners,
        # _led_owner_lock, _led_listeners_lock, _led_lock) live on
        # IlluminationAPI.

        # Camera state slots (_camera_listeners + lock, _frame_buffer,
        # _capturing_event, _focusing_event, _capture_return,
        # _autofocus_return, _suppress_value_warnings, _scale_bar,
        # _camera_cache + lock, _camera_temp_event,
        # _camera_temp_unschedule_fn, frame_validity) live on ImagingAPI.

        # ----- Motion Control Board -----
        # Constructed BEFORE MotionAPI so MotionAPI._driver resolves on
        # the first call. Driver selection goes through the motor registry
        # -- 'auto' tries real drivers in descending priority order and
        # falls back to NullMotionBoard if all fail, so no manual
        # try/except needed.
        motor_kwargs: dict = {}
        if simulate:
            from modules.settings_init import settings

            default_model = settings.get('microscope', 'LS850T') if settings else 'LS850T'
            motor_kwargs['model'] = sim_model or default_model
        self._motion_driver: MotorBoardProtocol = motor_registry.create(
            'auto', simulate=simulate, **motor_kwargs
        )
        if simulate:
            logger.info(
                f'[SCOPE API ] Using SIMULATED Motor Board (model={motor_kwargs.get("model")})'
            )

        # ----- MotionAPI -----
        # Constructed AFTER the motion driver so _driver resolves correctly.
        # init_axes() sizes per-axis dicts to detect_present_axes(); then
        # start_monitor() spawns the background poll thread. NullMotionBoard
        # returns [] from detect_present_axes(), so a system with no motor
        # hardware ends up with empty dicts throughout.
        from modules.lumascope_api.motion import MotionAPI  # local-import: avoid cycle

        self.motion = MotionAPI(self, self._motion_driver)
        present_axes = self._motion_driver.detect_present_axes()
        self.motion.init_axes(present_axes)
        self.motion.start_monitor()

        # ----- LED Control Board -----
        # Same registry-based selection as motion.
        self._led_driver: LEDBoardProtocol = led_registry.create('auto', simulate=simulate)
        if simulate:
            logger.info('[SCOPE API ] Using SIMULATED LED Board')

        # ----- Camera -----
        # Driver selection via camera_registry. `camera_type` accepts:
        # 'auto' (tries pylon -> ids by priority), 'pylon', 'ids',
        # 'sim', or any other registered camera kind. Default 'auto' is
        # the right choice for most callers; legacy callers that need
        # the prior "pylon" default pass camera_type='pylon' explicitly.
        # _frame_buffer slot lives on ImagingAPI. _camera_driver slot
        # defaulted to None in _init_minimal; the registry call below
        # overrides it on a successful connect.
        camera_kwargs: dict = {}
        if simulate:
            camera_kwargs['z_position_func'] = lambda: self._motion_driver.current_pos('Z')
        try:
            self._camera_driver: Camera = camera_registry.create(
                camera_type, simulate=simulate, **camera_kwargs
            )
            if simulate:
                self._camera_driver.load_cycle_images()
                logger.info('[SCOPE API ] Using SIMULATED Camera')
        except Exception as _cam_exc:
            logger.error(
                f'[SCOPE API ] Camera Board Not Initialized: {type(_cam_exc).__name__}: {_cam_exc}'
            )
            # Prior behavior logged only; the user saw no popup and
            # every camera-dependent UI action silently returned None/False.
            # Same pattern #632/#539 fixed for the LED + motor boards.
            # Suppress the per-component popup when LED + motor have
            # already fallen back to Null*: the consolidated "No
            # hardware detected" popup will fire later and the
            # individual one is redundant.
            _notify_camera_failure(
                _cam_exc,
                suppress_if_cold_start=_is_total_cold_start(
                    self._led_driver,
                    self._motion_driver,
                ),
            )

        # ----- ScopeCapabilities -----
        # Single source of truth for "what does this scope have" -- built
        # once from the three drivers, frozen thereafter. Callers should
        # prefer `scope.capabilities.*` over the wrapper methods below.
        # Runtime connection state (`motor_connected`, `led_connected`)
        # stays as live properties on Lumascope -- those must reflect
        # disconnects and can't be snapshotted.
        self.capabilities = ScopeCapabilities.from_drivers(
            motion=self._motion_driver,
            led=self._led_driver,
            camera=self._camera_driver,
        )

        # ----- Sub-API wiring -----
        # Six sub-APIs: motion, illumination, imaging, diagnostics,
        # capabilities, io. motion was already constructed above (it needs
        # earlier construction so init_axes / start_monitor can run
        # before the LED/camera drivers are set up). Remaining sub-APIs:
        from modules.lumascope_api.illumination import IlluminationAPI
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from modules.lumascope_api.io import IOAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        self.illumination = IlluminationAPI(self, self._led_driver)
        self.imaging = ImagingAPI(self, self._camera_driver)
        self.diagnostics = DiagnosticsAPI(self)
        self.io = IOAPI(self)
        self.runtime_state = RuntimeState(self)

        # Partial-hardware notification deferred to initialize(config) --
        # we need scope-config knowledge to distinguish "LS620 correctly
        # has no motor" from "LS820 motor failed to connect."

        # Track whether any real hardware was found.
        # Camera check reads the (private) driver handle directly because
        # there is no public camera attribute to read: the camera surface
        # is `self.imaging`, and `self.camera` does not exist. Do not add
        # one without checking for probes that assume it -- code has been
        # written against that name before, and `getattr(scope, 'camera',
        # None)` silently yields None rather than failing, so the branch
        # behind it simply never runs.
        self._no_hardware = (
            not simulate
            and isinstance(self._led_driver, NullLEDBoard)
            and isinstance(self._motion_driver, NullMotionBoard)
            and self._camera_driver is None
        )
        if self._no_hardware:
            logger.warning(
                '[SCOPE API ] No hardware detected (LED, motor, and camera all failed to initialize)'
            )

        # Most per-instance state lives on the sub-APIs: imaging owns
        # camera-stream state + locks, motion owns per-axis state +
        # _last_turret_position, illumination owns LED state,
        # runtime_state owns settings-host state (labware / objective /
        # turret_config / stage_offset). Lumascope holds driver slots,
        # executor handles, source_path, and metrics_logger.

        # Frame validity, camera_cache, scale_bar, +
        # _camera_listeners/_frame_buffer/_capturing_event/_focusing_event/
        # _capture_return/_autofocus_return/_suppress_value_warnings/
        # _camera_temp_event init live on ImagingAPI.__init__.
        # _load_camera_timing + _populate_camera_cache are ImagingAPI
        # methods and run automatically during ImagingAPI.__init__.
        # Lumascope wires up the motion-settle check against the
        # frame_validity instance below.
        def _motion_settle_check(source: str) -> bool:
            # For absent axes (e.g., LS820 has no X/Y), treat UNKNOWN as settled.
            # Axes that were never homed or moved stay UNKNOWN -- they shouldn't
            # block frame validity for sources that don't apply.
            idle_or_absent = (AxisState.IDLE, AxisState.UNKNOWN)
            if source == 'z_move':
                return self.motion.get_axis_state('Z') in idle_or_absent
            elif source == 'xy_move':
                return (
                    self.motion.get_axis_state('X') in idle_or_absent
                    and self.motion.get_axis_state('Y') in idle_or_absent
                )
            elif source == 'turret':
                return self.motion.get_axis_state('T') in idle_or_absent
            return True

        self.imaging.frame_validity.set_settle_check(_motion_settle_check)
        # _load_camera_timing relocated to ImagingAPI; called during
        # ImagingAPI.__init__ via the settle-check setup completion.
        self.imaging._load_camera_timing()

        # Populate position cache from firmware so get_current_position()
        # returns correct values immediately (not 0.0 from empty cache).
        # Critical for standalone scripts that read position right after
        # creating Lumascope (e.g., backlash characterization).
        if self.motor_connected:
            try:
                self.motion.refresh_position_cache()
            except Exception:
                pass  # OK -- cache stays at 0.0 if firmware unresponsive

        # LVP-A-13: pre-construct MetricsLogger so every Lumascope user
        # (Kivy app, REST API, headless tests, CLI tools) shares the
        # same metrics surface -- engineering plugin / status endpoints
        # can call self.metrics_logger.snapshot_executors() etc. without
        # waiting for the host to register one. Lifecycle is two-phase:
        # __init__ constructs (this block); the host calls
        # self.metrics_logger.start(scheduler) once it knows which
        # scheduler is appropriate for its environment. Doesn't start
        # any timers / Clock events here, so test fixtures don't pay
        # for periodic work they don't want.
        #
        # metrics_logger + _executor_bundle slots defaulted to None in
        # _init_minimal. The host calls register_executor_bundle() after
        # the bundle exists, before calling metrics_logger.start.
        if register_metrics:
            try:
                from modules.metrics_logger import MetricsLogger

                self.metrics_logger = MetricsLogger(
                    scope=self,
                    executor_bundle=None,  # set later via register_executor_bundle
                    settings={},  # ditto
                )
            except Exception as _e:
                logger.warning(f'[SCOPE API ] MetricsLogger construction failed: {_e}')

        # LVP-A-7: register the emergency-shutdown atexit hook so EVERY
        # Lumascope user (Kivy app, REST server, headless tests, CLI
        # tools) gets the LED-off-and-disconnect safety net automatically.
        # Was previously inline in lumaviewpro.py:541-549, leaving every
        # non-GUI entry point silently unprotected -- exactly the failure
        # mode the comment cited (LED stays on, sample overheats).
        if register_atexit:
            try:
                import atexit

                atexit.register(self._emergency_shutdown)
            except Exception as _e:
                logger.warning(f'[SCOPE API ] atexit registration failed: {_e}')

    def initialize(self, config) -> None:
        """Configure scope from connected to ready-to-use.

        Call once after construction.  Sets all scope-level hardware
        configuration.  Does NOT set per-layer camera settings (gain,
        exposure, auto-gain) -- those are the caller's responsibility
        for the active layer.

        Args:
            config: ScopeInitConfig instance with all scope-level settings.
        """
        self._notify_partial_hardware(config)
        self.illumination.leds_off()
        self.runtime_state.set_labware(config.labware)
        if config.turret_config:
            self.runtime_state.set_turret_config(config.turret_config)
        self.runtime_state.set_objective(config.objective_id)
        # Startup applies push PERSISTED settings at the connect boundary, so
        # each value is reconciled to the capabilities the connected hardware
        # actually reports BEFORE the apply -- a settings file written against
        # a different camera (the swap case) must not send an unsupportable
        # value. Reconciliation only makes sense against a real camera: with
        # none connected the applies are quiet no-ops and the absent-fallback
        # capability values must not masquerade as a camera's answer.
        frame_width, frame_height = config.frame_width, config.frame_height
        binning_size = config.binning_size
        if self.camera_connected:
            available_binning = self.imaging.get_available_binning_sizes()
            if binning_size not in available_binning:
                camera_binning = self.imaging.get_binning_size()
                # The persisted frame is a DISPLAYED size at the persisted
                # factor; at a different factor it would come up as a
                # fraction-area ROI (driver clamping only protects against
                # too-large requests). Re-derive it from the native intent
                # at the factor actually being applied.
                native = binning.displayed_to_native(
                    {'width': frame_width, 'height': frame_height},
                    binning_size,
                    self.imaging.get_native_resolution()
                    or {
                        'width': frame_width * binning_size,
                        'height': frame_height * binning_size,
                    },
                )
                refit = binning.native_to_displayed(
                    native, camera_binning, self.imaging.get_pixel_alignment()
                )
                logger.error(
                    f'[SCOPE API ] initialize: persisted binning {binning_size} '
                    f'is not supported by the connected camera '
                    f'(available: {available_binning}); keeping the '
                    f'camera-reported {camera_binning} and refitting the '
                    f'frame {frame_width}x{frame_height} -> '
                    f'{refit["width"]}x{refit["height"]}'
                )
                notifications.warning(
                    'Camera',
                    'Saved binning not supported',
                    f'The saved {binning_size}x{binning_size} binning is not '
                    f'supported by this camera; it starts at '
                    f'{camera_binning}x{camera_binning} instead. Pick a '
                    f'binning in Microscope Settings to update the saved '
                    f'value.',
                )
                binning_size = camera_binning
                frame_width, frame_height = refit['width'], refit['height']
        # A rejection surviving reconciliation is a live hardware fault
        # mid-apply. Each apply is contained individually so one faulted
        # setting cannot skip the rest of bring-up: the callers of
        # initialize are the app build and the reconnect button, where a
        # propagated raise aborts startup entirely (no live view, no
        # motion config, no session) over a single transient -- the
        # rejection is already logged AND notified at the API layer, and
        # every downstream consumer reads delivered geometry, never these
        # requests, so nothing is left believing a rejected value.
        for label, apply_fn in (
            ('binning', lambda: self.imaging.set_binning_size(binning_size)),
            ('frame size', lambda: self.imaging.set_frame_size(frame_width, frame_height)),
        ):
            try:
                apply_fn()
            except CameraSettingRejected as ex:
                logger.error(
                    f'[SCOPE API ] initialize: {label} apply rejected by a '
                    f'connected camera ({ex}); bring-up continues at the '
                    f'camera-held value'
                )
        # Apply the capture pixel format HERE, synchronously, while the start
        # gate is still closed (this runs before the bring-up start_streaming).
        # Resolving + setting it now -- instead of via the async camera-executor
        # push that the image-mode spinner enqueues -- removes the race where
        # the format lands after streaming begins and forces a redundant
        # grab-loop restart. The spinner handler skips its push during init.
        pixel_format = image_mode.select_capture_pixel_format(
            config.capture_depth, self.imaging.get_supported_pixel_formats()
        )
        if pixel_format is not None:
            try:
                self.imaging.set_pixel_format(pixel_format)
            except CameraSettingRejected as ex:
                logger.error(
                    f'[SCOPE API ] initialize: pixel format apply rejected by '
                    f'a connected camera ({ex}); bring-up continues at the '
                    f'camera-held format'
                )
        if self.capabilities.camera_supports_conversion_gain_mode:
            self.imaging.set_conversion_gain_mode('High' if config.high_conversion_gain else 'Low')
        if self.capabilities.camera_supports_line_noise_reduction:
            self.imaging.set_line_noise_reduction(config.line_noise_reduction)
        self.runtime_state.set_stage_offset(config.stage_offset)
        self.imaging.set_scale_bar(enabled=config.scale_bar_enabled)
        self.motion.set_acceleration_limit(val_pct=config.acceleration_pct)
        logger.info('[SCOPE API ] Scope initialized')

    def _notify_partial_hardware(self, config) -> None:
        """Warn user about missing hardware, filtered by scope expectations.

        An LS620 with no motor is not a failure -- its scopes.json says
        Focus/XYStage/Turret are all false. Only warn for hardware the
        scope was supposed to have. Simulators never warn. The
        no_hardware total-cold-start case skips this notification --
        lumaviewpro.on_start fires a single consolidated "No hardware
        detected" popup that covers the same ground.
        """
        if self._simulated:
            return
        if self._no_hardware:
            return
        missing = []
        if config.expects_led and isinstance(self._led_driver, NullLEDBoard):
            missing.append('LED Board')
        if config.expects_motion and isinstance(self._motion_driver, NullMotionBoard):
            missing.append('Motor Controller')
        if not getattr(self._camera_driver, 'active', None):
            missing.append('Camera')
        if missing:
            notifications.warning(
                'Hardware',
                'Partial Hardware Detected',
                f'Not connected: {", ".join(missing)}. Some features will be unavailable.',
            )

    # --- Executor-backed command API ---
    #
    # Single canonical path for hardware operations that need executor
    # dispatch: caller invokes scope.X_async(...) or scope.X_sync(...);
    # Lumascope picks the right executor internally. Replaces the older
    # modules/scope_commands.py helper functions where the caller had
    # to pass an executor on every call (parallel-paths anti-pattern).

    def register_executors(
        self, *, camera_executor=None, io_executor=None, file_io_executor=None
    ) -> None:
        """Register the executor handles used by the X_async / X_sync command methods.

        Call once at startup after the executors are constructed. Tests
        that don't drive the executor-backed API can skip this -- those
        methods raise RuntimeError if invoked without executors registered.

        Args:
            camera_executor: Executor for camera-bound IOTasks.
            io_executor: Executor for general IO/motion IOTasks.
            file_io_executor: Executor for file-IO IOTasks.
        """
        self._camera_executor = camera_executor
        self._io_executor = io_executor
        self._file_io_executor = file_io_executor

    def register_executor_bundle(self, executor_bundle, settings=None) -> None:
        """LVP-A-13: register the ExecutorBundle + settings dict for MetricsLogger.

        Lumascope construction (__init__) creates a MetricsLogger but
        cannot fill in the bundle yet -- the bundle is created later by
        ExecutorRegistry.create_default in the host's startup path.
        Call this once after the bundle exists, BEFORE calling
        ``self.metrics_logger.start(scheduler)``. Settings dict is
        optional; defaults to ``{}`` if MetricsLogger was created with
        a placeholder.

        Args:
            executor_bundle: ExecutorBundle instance to attach.
            settings: Optional settings dict for MetricsLogger.
        """
        self._executor_bundle = executor_bundle
        if self.metrics_logger is not None:
            self.metrics_logger._bundle = executor_bundle
            if settings is not None:
                self.metrics_logger._settings = settings

    def register_source_path(self, source_path) -> None:
        """Register the LVP source/data path used by protocol API methods.

        Used by ``load_protocol()`` and ``create_protocol()`` to find
        ``data/tiling.json``. Called once at startup. Tests that don't
        drive the protocol API can skip this.

        Args:
            source_path: Path-like to the LVP source/data root.
        """
        self._source_path = source_path

    def _tiling_configs_path(self):
        """Resolve data/tiling.json from the registered source path."""

        if self._source_path is None:
            raise RuntimeError(
                'Lumascope.load_protocol/create_protocol require '
                'register_source_path() to have been called.'
            )
        return pathlib.Path(self._source_path) / 'data' / 'tiling.json'

    # --- Protocol API (LAYER-I / LV-16) ---

    def load_protocol(self, file_path) -> 'Protocol':
        """Load a Protocol from disk.

        Wraps ``Protocol.from_file(...)`` and resolves
        ``data/tiling.json`` from the registered source_path.

        Args:
            file_path: Path to the protocol file.

        Returns:
            Protocol: The loaded Protocol instance.

        Raises:
            ProtocolFormatError: On format issues (same surface as
                Protocol.from_file).
        """
        from modules.protocol import Protocol

        return Protocol.from_file(
            file_path=file_path,
            tiling_configs_file_loc=self._tiling_configs_path(),
        )

    def create_protocol(self, *, config=None, input_config=None, empty_config=None) -> 'Protocol':
        """Construct a Protocol in-memory.

        Three modes (pass exactly one):
          - config={...}: full config dict passed to Protocol() directly.
          - input_config={...}: partial config (positions, layer_configs,
            etc.); routed through Protocol.from_config which fills defaults.
          - empty_config={...}: labware/objective config for an empty-steps
            protocol; routed through Protocol.create_empty.
        tiling_configs_file_loc is resolved internally from the registered
        source_path.

        Args:
            config: Full config dict, or None.
            input_config: Partial config dict, or None.
            empty_config: Empty-steps config dict, or None.

        Returns:
            Protocol: Newly constructed Protocol instance.

        Raises:
            ValueError: If exactly one of config/input_config/empty_config
                was not provided.
        """
        from modules.protocol import Protocol

        provided = sum(1 for x in (config, input_config, empty_config) if x is not None)
        if provided != 1:
            raise ValueError(
                'create_protocol(): pass exactly one of config=, input_config=, or empty_config='
            )
        tcfg = self._tiling_configs_path()
        if input_config is not None:
            return Protocol.from_config(
                input_config=input_config,
                tiling_configs_file_loc=tcfg,
            )
        if empty_config is not None:
            return Protocol.create_empty(
                config=empty_config,
                tiling_configs_file_loc=tcfg,
            )
        return Protocol(
            tiling_configs_file_loc=tcfg,
            config=config,
        )

    @staticmethod
    def sanitize_step_name(input: str) -> str:
        """Sanitize a step name string.

        Thin pass-through to ``Protocol.sanitize_step_name`` so UI /
        module callers don't need to import the Protocol data class for
        this utility.

        Args:
            input: Raw step name to sanitize.

        Returns:
            str: Sanitized step name.
        """
        from modules.protocol import Protocol

        return Protocol.sanitize_step_name(input=input)

    def _require_executor(self, executor, name):
        if executor is None:
            raise RuntimeError(
                f'Lumascope.{name} requires register_executors() to have '
                f'been called with the relevant executor handle.'
            )
        return executor

    # --- LED command API ---
    # All LED methods + change-listener registry live on IlluminationAPI;
    # forwarders have been retired. Callers use scope.illumination.

    # --- Camera command API ---
    # All camera/imaging methods + state slots + change-listener registry
    # live on ImagingAPI; forwarders have been retired. Callers use
    # scope.imaging.

    @property
    def motor_connected(self) -> bool:
        """Whether the motor controller is connected.

        Returns:
            bool: True if a real (non-Null) motor board is connected.
        """
        return (
            not isinstance(self._motion_driver, NullMotionBoard)
            and self._motion_driver.is_connected()
        )

    @property
    def led_connected(self) -> bool:
        """Whether the LED controller is connected.

        Returns:
            bool: True if a real (non-Null) LED board is connected.
        """
        return not isinstance(self._led_driver, NullLEDBoard) and self._led_driver.is_connected()

    @property
    def camera_connected(self) -> bool:
        """Whether the camera is connected and active.

        Returns:
            bool: True if a real camera driver is connected and active.
        """
        driver = getattr(self, '_camera_driver', None)
        if driver is None or not getattr(driver, 'active', False):
            return False
        try:
            return driver.is_connected()
        except Exception:
            return False

    def disconnect(self) -> bool:
        """Disconnect from all hardware (LED, motion, camera).

        Best-effort teardown: every sub-system is attempted even if a
        prior one raises. State is always reset to the Null variants
        and `_invalidate_camera_cache` always runs, so a partial failure
        cannot leave the API holding a stale connected driver.

        Returns:
            bool: True if all three sub-disconnects succeeded. False if
                any sub-system raised or the camera driver returned
                False. Each failure is logged and surfaced via
                notification_center; programmatic callers can branch
                on the bool for diagnostic / shutdown-sequencing
                decisions.
        """
        logger.info('[SCOPE API ] Disconnecting from microscope...')

        # LVP-A-1: stop motors before tearing down the serial port so we
        # don't leave a stage/turret moving against an end-stop after
        # the host stops responding to status polls. Defense in depth --
        # every disconnect path benefits without relying on the caller
        # to remember.
        self.motion.stop_motion()

        # Stop the motion monitor and reset axis states -- MotionAPI.disconnect()
        # handles both: signals the monitor thread, waits for it, then resets
        # all axes to UNKNOWN and sets arrival events so waiters unblock.
        self.motion.disconnect()

        # Each sub-system: only attempt disconnect on a driver that
        # has one. Skips both the canonical no-op states (NullLEDBoard,
        # NullMotionBoard, self._camera_driver is None) and edge-case test
        # fixtures that bend the type system (e.g. `scope.led = object()`
        # for partial-hardware-warning tests). A skipped sub-system
        # counts as ok=True -- "nothing to tear down" is success, not
        # failure. Real drivers that raise inside disconnect() still
        # flip *_ok to False and fire a Rule-14 notification.
        led_ok = True
        if not isinstance(self._led_driver, NullLEDBoard) and hasattr(
            self._led_driver, 'disconnect'
        ):
            try:
                self._led_driver.disconnect()
            except Exception as ex:
                led_ok = False
                logger.exception(f'[SCOPE API ] LED disconnect failed: {ex}')
                notifications.error(
                    'Hardware',
                    'LED disconnect failed',
                    'The LED board did not shut down cleanly. '
                    'The serial port may be left open; reconnecting '
                    'may require a process restart.',
                )
        self._led_driver = NullLEDBoard()

        motion_ok = True
        if not isinstance(self._motion_driver, NullMotionBoard) and hasattr(
            self._motion_driver, 'disconnect'
        ):
            try:
                self._motion_driver.disconnect()
            except Exception as ex:
                motion_ok = False
                logger.exception(f'[SCOPE API ] Motion disconnect failed: {ex}')
                notifications.error(
                    'Hardware',
                    'Motor disconnect failed',
                    'The motor board did not shut down cleanly. '
                    'The serial port may be left open; reconnecting '
                    'may require a process restart.',
                )
        self._motion_driver = NullMotionBoard()

        camera_ok = True
        if self._camera_driver is not None and hasattr(self._camera_driver, 'disconnect'):
            try:
                camera_ok = bool(self._camera_driver.disconnect())
            except Exception as ex:
                camera_ok = False
                logger.exception(f'[SCOPE API ] Camera disconnect failed: {ex}')
                notifications.error(
                    'Hardware',
                    'Camera disconnect failed',
                    'The camera did not shut down cleanly. '
                    'USB resources may not be fully released until the '
                    'app restarts.',
                )
            self._camera_driver = None
        elif self._camera_driver is not None:
            # Camera lacked a `disconnect` method (test-fixture artifact);
            # clear the slot but don't claim success on a real teardown.
            self._camera_driver = None
        self.imaging._invalidate_camera_cache()
        # This scope's periodic camera-temp schedule dies WITH the scope:
        # the tick deliberately never self-cancels (a transient
        # connectivity False must not end logging), so the lifecycle edge
        # here is the owner that keeps a scope swap (reconnect) from
        # leaving an orphaned schedule sampling a discarded scope -- and
        # pinning its whole object graph -- for the rest of the session.
        self.imaging.stop_camera_temp_logging()

        all_ok = led_ok and motion_ok and camera_ok
        if all_ok:
            logger.info('[SCOPE API ] Microscope disconnected')
        else:
            logger.warning(
                f'[SCOPE API ] Microscope disconnected with errors '
                f'(led_ok={led_ok}, motion_ok={motion_ok}, '
                f'camera_ok={camera_ok})'
            )

        # Symmetric to atexit.register in __init__: each instance removes its
        # own hook on disconnect so test fixtures that construct + disconnect
        # many Lumascope instances do not leak atexit registrations.
        # atexit.unregister silently no-ops if the hook was never registered.
        try:
            import atexit

            atexit.unregister(self._emergency_shutdown)
        except Exception as _e:
            logger.warning(f'[SCOPE API ] atexit unregister failed: {_e}')

        return all_ok

    def _emergency_shutdown(self):
        """LVP-A-7: best-effort safety shutdown for atexit / abnormal exit.

        Guards LEDs and motor against the interpreter terminating mid-
        operation: turns off all LEDs, then disconnects (which now also
        stops motion via the LVP-A-1 chain). Swallows every exception so
        atexit completes cleanly even when the logging stack or hardware
        access is already torn down.

        Uses `leds_off_emergency` (bounded `_led_lock` acquire) rather
        than `leds_off` to avoid atexit deadlock when an in-flight LED
        command holds the lock.
        """
        try:
            self.illumination.leds_off_emergency()
        except Exception:
            pass
        try:
            self.disconnect()
        except Exception:
            pass
        try:
            logger.info('[SCOPE API ] _emergency_shutdown complete (LEDs off, disconnected)')
        except Exception:
            pass

    @property
    def no_hardware(self) -> bool:
        """True if no real hardware was detected (LED, motor, and camera all missing).

        Returns:
            bool: True when all three subsystems are absent or stubbed.
        """
        return self._no_hardware

    def are_all_connected(self) -> bool:
        """Check if LED, motion, and camera boards are all connected.

        Returns:
            bool: True if all three components are connected.
        """
        logger.debug('[SCOPE API ] Performing connection check...')
        led = not isinstance(self._led_driver, NullLEDBoard) and self._led_driver.is_connected()
        motion = self.motor_connected
        camera = self._camera_driver is not None and self._camera_driver.is_connected()

        if not led:
            logger.info('[SCOPE API ] Connection Check: LED Board not connected')
        if not motion:
            logger.info('[SCOPE API ] Connection Check: Motion Board not connected')
        if not camera:
            logger.info('[SCOPE API ] Connection Check: Camera not connected')

        if led and motion and camera:
            logger.debug('[SCOPE API ] Connection Check: All components connected')

        return led and motion and camera

    @classmethod
    def create_diagnostic(cls) -> 'Lumascope':
        """Create a minimal Lumascope for diagnostics (no camera init).

        Connects to LED and motor boards only. For use by tools like
        the tech support report that need board access without the full
        application stack.

        Returns:
            Lumascope: Instance with led/motion connected, camera=None.
        """
        instance = cls.__new__(cls)
        # Shared state-slot init (audit #35) -- same call __init__ makes.
        instance._init_minimal(simulated=False)

        # Connect boards -- motion driver first so MotionAPI._driver resolves
        # correctly at construction time. The helpers are at module scope so
        # __init__, create_diagnostic, and future callers share one code path.
        from drivers.null_ledboard import NullLEDBoard
        from drivers.null_motorboard import NullMotionBoard

        instance._led_driver = _try_connect_board('LED board', LEDBoard, NullLEDBoard)
        instance._motion_driver = _try_connect_board('Motor board', MotorBoard, NullMotionBoard)

        # Construct MotionAPI and populate per-axis state (mirrors __init__ sequence).
        from modules.lumascope_api.motion import MotionAPI  # local-import: avoid cycle

        instance.motion = MotionAPI(instance, instance._motion_driver)
        present_axes = instance._motion_driver.detect_present_axes()
        instance.motion.init_axes(present_axes)
        instance.motion.start_monitor()

        instance.camera = None
        instance._frame_buffer = None

        # Build capabilities -- diagnostic instances still need this so
        # any code that reads scope.capabilities.* works.
        instance.capabilities = ScopeCapabilities.from_drivers(
            motion=instance._motion_driver,
            led=instance._led_driver,
            camera=None,
        )

        # Sub-API wiring -- diagnostic instances are first-class enough
        # that disconnect / scope.imaging / scope.illumination do not
        # raise AttributeError. ImagingAPI tolerates camera=None (per
        # its docstring); IlluminationAPI gets the connected LED driver
        # (real or NullLEDBoard).
        from modules.lumascope_api.illumination import IlluminationAPI
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from modules.lumascope_api.io import IOAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        instance.illumination = IlluminationAPI(instance, instance._led_driver)
        instance.imaging = ImagingAPI(instance, None)
        instance.diagnostics = DiagnosticsAPI(instance)
        instance.io = IOAPI(instance)
        instance.runtime_state = RuntimeState(instance)

        # No-hardware probe mirrors __init__ -- diagnostic mode is never
        # simulate=True, so a NullLED + NullMotor + no camera means we
        # really do have no hardware.
        instance._no_hardware = isinstance(instance._led_driver, NullLEDBoard) and isinstance(
            instance._motion_driver, NullMotionBoard
        )

        logger.info(
            '[SCOPE API ] Diagnostic scope created '
            f'(LED={instance.led_connected}, '
            f'Motor={instance.motor_connected})'
        )
        return instance

    ########################################################################
    # INTEGRATED SCOPE FUNCTIONS
    ########################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ILLUMINATE AND CAPTURE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # AUTOFOCUS Functionality
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Legacy autofocus methods (autofocus, autofocus_iterate, focus_best) removed
    # 2026-03-31 -- superseded by AutofocusRunner. No callers remained.
