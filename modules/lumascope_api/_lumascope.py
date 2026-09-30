#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import warnings

from lvp_logger import logger

# Import Lumascope Hardware files
from drivers.motorboard import MotorBoard
from drivers.ledboard import LEDBoard
from modules.lumascope_api import _constants as _api_constants
from modules.lumascope_api._constants import SIMULATOR_TIERS
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
from modules.exceptions import CameraSettingRejected, ScopeDisconnectError
from modules.scope_capabilities import ScopeCapabilities
from modules.sequential_io_executor import SequentialIOExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.layer_record import LayerIdentity
    from modules.scope_init_config import ScopeInitConfig

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

    Called from `Lumascope.__init__` and from `ScopeSession.create`
    so any L2 entry point trips the warning, even
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
    # There is no LED current cap on this class: the connected LED driver
    # publishes its own through `scope.capabilities.led_max_ma`, and a
    # constant here would be a second copy that drifts.
    # LED channel set comes from self._led_driver.available_channels() -- varies by
    # Canonical home for these is `_constants.py`; alias on the class so
    # existing callers (`scope._VALID_AXIS_NAMES`, `Lumascope._MOTOR_POSITION_LIMIT`)
    # keep working. Sub-API modules import from `_constants.py` directly
    # to avoid a circular dep with this file.
    _VALID_AXIS_NAMES = _api_constants._VALID_AXIS_NAMES
    _MOTOR_POSITION_LIMIT = _api_constants.MOTOR_POSITION_LIMIT

    def _init_minimal(self, simulated: bool, ui_dispatcher=None) -> None:
        """Shared init for state slots both __init__ and create_diagnostic need.

        Sets the non-driver state that every Lumascope instance must
        carry: transformers, locks, camera cache, objective state slots,
        the scope's two lanes, source path. Both __init__ and
        create_diagnostic call this first; each then does its
        driver-connection-specific work.

        Pre-#35, create_diagnostic open-coded a subset of these
        assignments and left ~12 attributes unset, which made diagnostic
        instances second-class. Centralizing here makes the slot list
        the single point of truth.
        """
        self._simulated = simulated
        # Whether this scope's model has a motor board. Until initialize()
        # reads the model's catalogue entry, a missing board counts as
        # disconnected: holding an unconfigured scope to every board is the
        # answer that cannot admit a run on a board that fell off.
        self._motion_expected = True

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

        # The scope's two lanes, built and started here and shut by
        # disconnect(). Every LED, motion and camera command goes through
        # one, so commands from any caller -- a session's GUI or REST, or a
        # script's bare scope -- run one at a time per bus, in order, and a
        # run holding the scope can refuse what is not its own. A session
        # composed over this scope asks the lanes its activity claim.
        self._io_executor = SequentialIOExecutor(name='IO', ui_dispatcher=ui_dispatcher)
        self._camera_executor = SequentialIOExecutor(name='CAMERA', ui_dispatcher=ui_dispatcher)
        self._io_executor.start()
        self._camera_executor.start()
        # The key the camera lane's claim returned to the session, which
        # hands it here: the camera temperature read carries it, so the
        # temperature log keeps running while a run or a diagnostic holds
        # the scope. None until a session asks the claim.
        self._camera_override_key = None

    @staticmethod
    def _build_simulated_motor_board(model: str, sim_tier: str) -> MotorBoardProtocol:
        """The simulated scope's motor board, on the tier asked for.

        Both tiers ask the catalogue which axes the model has: a model
        with no axes has no motor board, so it gets the null driver on
        either tier. The fast tier goes through the registry's simulator
        selection with those axes. The firmware tier builds the production
        driver by name against the emulator: a model with axes whose
        emulator does not come up raises, because the registry's auto path
        would fall back to the null driver and a dead emulator would then
        look exactly like a manual scope.
        """
        if sim_tier not in SIMULATOR_TIERS:
            raise ValueError(f'sim_tier {sim_tier!r} is not one of {SIMULATOR_TIERS}')
        from modules.layer_record import load_scope_models, model_axes

        axes = model_axes(load_scope_models(), model)
        if not axes:
            logger.info(f'[SCOPE API ] Model {model} has no motor axes: no motor board')
            return NullMotionBoard()
        if sim_tier == 'fast':
            board = motor_registry.create('auto', simulate=True, model=model, axes=axes)
            logger.info(f'[SCOPE API ] Using SIMULATED Motor Board (model={model})')
            return board
        from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend

        backend = SimWireBackend(MotorBoardSpec(model, axes))
        board = motor_registry.create('rp2040', backend=backend)
        logger.info(
            f'[SCOPE API ] Using the motor FIRMWARE in simulation '
            f'(model={model}, axes={"".join(sorted(axes))})'
        )
        return board

    @staticmethod
    def _build_simulated_led_board(model: str, sim_tier: str) -> LEDBoardProtocol:
        """The simulated scope's LED board, on the tier asked for.

        The catalogue names no LED board, so the model's axes stand in for
        one: a model with motor axes is an EL-0940 scope, whose LEDs are on
        their own board; a model with none is an FX2 scope, whose LEDs the
        FX2 drives, and the simulator does not run the FX2, so that scope
        keeps the Python stand-in on both tiers. The firmware tier builds
        the production driver by name against the emulator, so an emulator
        that does not come up raises instead of becoming a stand-in. The
        tier is the one the motor board was just built on, which refused
        any tier that is not one of the two.
        """
        from modules.layer_record import load_scope_models, model_axes

        if sim_tier == 'fast' or not model_axes(load_scope_models(), model):
            board = led_registry.create('auto', simulate=True)
            logger.info(f'[SCOPE API ] Using SIMULATED LED Board (model={model})')
            return board
        from drivers.sim_wire.backend import LedBoardSpec, SimWireBackend

        backend = SimWireBackend(None, led=LedBoardSpec(model))
        board = led_registry.create('rp2040', backend=backend)
        logger.info(f'[SCOPE API ] Using the LED FIRMWARE in simulation (model={model})')
        return board

    def __init__(
        self,
        simulate: bool = False,
        camera_type: str = 'auto',
        register_atexit: bool = True,
        sim_model: str | None = None,
        warn_pre_release: bool = True,
        configured_model: str | None = None,
        sim_tier: str = 'fast',
        ui_dispatcher=None,
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
            sim_model: When simulating, the scope model the simulated
                motor board reports (e.g. 'LS850', 'LS850T'). Selects
                which axes the simulated scope presents -- an LS850 has
                no turret, an LS850T does -- so capabilities.axes reflect
                the chosen model end to end. Ignored when simulate is
                False; defaults to ``configured_model``, then the
                'microscope' setting, then 'LS850T'.
            configured_model: The scope model selected in settings, for
                units whose hardware cannot report one (the Classic/FX2
                line has no motor board to ask). Optional: a
                motor-reported model always wins over it (hardware truth
                outranks a user selection), so callers on self-reporting
                hardware construct unchanged. A SIMULATED scope reports
                this as its model (a declared 'LS850' has no turret
                axis), so the driver and the selection agree from
                construction. Left None on
                a unit that also reports no model, layer identity
                resolves empty and LED use fails loudly by name rather
                than silently guessing.
            sim_tier: Which simulated motor board a simulated scope gets.
                ``'fast'`` (default) is ``SimulatedMotorBoard``, a Python
                stand-in with no timing, for routine tests. ``'firmware'``
                is the production ``MotorBoard`` driver connected to the
                real motor firmware running in a MicroPython process
                behind an emulated serial port, so every line of the
                driver runs; it costs the driver's real connect (about a
                second) and needs a runtime built for this platform.
                Ignored when simulate is False.
            warn_pre_release: Whether this construction should fire the
                PRE-RELEASE FutureWarning. The warning tells a caller its
                code may break under a future release, which is only
                meaningful for callers that ship SEPARATELY from this API
                -- scripts, examples, integrations. LumaViewPro's own GUI
                moves with the API in the same commit, so the warning
                tells it nothing and reaches the user as noise on every
                launch. Defaults True: a new caller that has not thought
                about it is warned.
            ui_dispatcher: ``Clock.schedule_once(func, dt)``'s shape. The
                scope's IO and CAMERA lanes hand a finished command's
                callback to it, so a GUI host gets its callbacks on its UI
                thread. None (default) runs them on the lane's worker.
        """
        if warn_pre_release:
            _fire_pre_release_warning()

        # Shared state-slot init (audit #35) -- transformers, locks,
        # camera cache, objective/turret state, the scope's lanes.
        # Driver construction + sub-API wiring happen below.
        self._init_minimal(simulated=simulate, ui_dispatcher=ui_dispatcher)

        # LED state slots (_led_listeners, _led_state, _lit_by,
        # _led_state_lock, _led_listeners_lock, _led_lock) live on
        # IlluminationAPI.

        # Camera state slots (_camera_listeners + lock, _frame_buffer,
        # _focusing_event, _suppress_value_warnings, _scale_bar,
        # _camera_cache + lock, _camera_temp_event,
        # _camera_temp_unschedule_fn, frame_validity) live on ImagingAPI.

        # ----- Motion Control Board -----
        # Constructed BEFORE MotionAPI so MotionAPI._driver resolves on
        # the first call. Driver selection goes through the motor registry
        # -- 'auto' tries real drivers in descending priority order and
        # falls back to NullMotionBoard if all fail, so no manual
        # try/except needed.
        if simulate:
            from modules.settings_init import settings

            default_model = settings.get('microscope', 'LS850T') if settings else 'LS850T'
            model = sim_model or configured_model or default_model
            self._motion_driver: MotorBoardProtocol = self._build_simulated_motor_board(
                model, sim_tier
            )
        else:
            self._motion_driver = motor_registry.create('auto')

        # ----- MotionAPI -----
        # Constructed AFTER the motion driver so _driver resolves correctly.
        # _init_axes() sizes per-axis dicts to detect_present_axes() and seeds
        # each axis known/unknown from detect_homed_axes(); then
        # _start_monitor() spawns the background poll thread. NullMotionBoard
        # returns [] from detect_present_axes(), so a system with no motor
        # hardware ends up with empty dicts throughout.
        from modules.lumascope_api.motion import MotionAPI  # local-import: avoid cycle

        self.motion = MotionAPI(self, self._motion_driver)
        present_axes = self._motion_driver.detect_present_axes()
        self.motion._init_axes(present_axes, self._motion_driver.detect_homed_axes())
        self.motion._start_monitor()

        # ----- LED Control Board -----
        # Same selection as motion: the simulated board on the session's tier.
        if simulate:
            self._led_driver: LEDBoardProtocol = self._build_simulated_led_board(model, sim_tier)
        else:
            self._led_driver = led_registry.create('auto')

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
            # Light reaches the simulated sensor the same way Z does: the
            # composition root hands it over, because it is the only object
            # holding both halves and no driver may reach into a peer. The
            # illumination API is asked rather than the board -- it is where
            # which channel is lit is decided, and the board holds no state.
            # Imported here like the other illumination references in this
            # file: at module scope it closes an import cycle.
            #
            # Ordering: this reads self.illumination, which is built further
            # down, and is safe because the callable only runs while a frame
            # is being generated and nothing starts the camera grabbing
            # during construction. Anything that begins streaming before the
            # sub-APIs exist breaks that, so start it after them.
            from modules.lumascope_api.illumination import live_lit_pairs

            camera_kwargs['illumination_func'] = lambda: sum(
                ma for _, ma in live_lit_pairs(self.illumination)
            )
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

        # ----- Layer identity -----
        # What the layers on this unit ARE (names, LED addresses,
        # excitations, filterset) -- resolved from the unit's motorconfig
        # LED block or the model's scopes.json rows, here at construction
        # so every caller (GUI, headless session, script) holds identity
        # without any separate initialize step. Distinct from the live
        # LED state the illumination API keeps: identity says what a
        # layer IS, state says what is lit now.
        self._configured_model = configured_model
        self.layer_identity = self._resolve_layer_identity()

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
            layer_identity=self.layer_identity,
        )

        # ----- Sub-API wiring -----
        # motion was already constructed above (it needs earlier
        # construction so _init_axes / _start_monitor can run before the
        # LED/camera drivers are set up). Remaining sub-APIs:
        from modules.lumascope_api.illumination import IlluminationAPI
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from modules.lumascope_api.io import IOAPI
        from modules.lumascope_api.protocols import ProtocolsAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        self.illumination = IlluminationAPI(self, self._led_driver)
        self.imaging = ImagingAPI(self, self._camera_driver)
        self.diagnostics = DiagnosticsAPI(self)
        self.io = IOAPI(self)
        self.protocols = ProtocolsAPI(self)
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
        elif not simulate and isinstance(self._led_driver, NullLEDBoard):
            # Illumination is gone but the rest of the scope came up, so the
            # consolidated no-hardware popup above stays silent and nothing
            # else would tell the operator. Without this the first symptom is
            # a sample under a dark objective and controls that appear to do
            # nothing. Say it once here rather than once per failed command.
            logger.warning(
                '[SCOPE API ] LED board unavailable; illumination controls will not work'
            )
            notifications.warning(
                'Illumination',
                'LED Board Unavailable',
                'The LED control board did not respond, so illumination is '
                'not available this session. The rest of the microscope is '
                'working. Power-cycle the microscope and restart LumaViewPro '
                'to restore illumination.',
            )

        # Most per-instance state lives on the sub-APIs: imaging owns
        # camera-stream state + locks, motion owns per-axis state +
        # _last_turret_position, illumination owns LED state,
        # runtime_state owns settings-host state (labware / objective /
        # turret_config / stage_offset). Lumascope holds driver slots,
        # its two lanes and source_path.

        # Frame validity, camera_cache, scale_bar, +
        # _camera_listeners/_frame_buffer/_focusing_event/
        # _suppress_value_warnings/
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
                self.motion._refresh_position_cache()
            except Exception:
                pass  # OK -- cache stays at 0.0 if firmware unresponsive

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

    def _resolve_layer_identity(self, override_model: str | None = None):
        """Run the identity resolver against the current drivers and config."""
        from modules.layer_record import resolve_layer_identity

        motorconfig = getattr(self._motion_driver, 'motorconfig', None)
        board_block = motorconfig.led_block() if motorconfig is not None else None
        read_ok = motorconfig.board_config_read_ok if motorconfig is not None else True
        try:
            motor_model = self._motion_driver.get_microscope_model()
        except Exception as e:
            logger.warning(f'[SCOPE API ] motor model unavailable for layer identity: {e}')
            motor_model = None
        return resolve_layer_identity(
            board_block=board_block,
            board_config_read_ok=read_ok,
            motor_model=motor_model,
            configured_model=self._configured_model,
            override_model=override_model,
        )

    def refresh_layer_identity(self, override_model: str | None = None) -> 'LayerIdentity':
        """Re-resolve layer identity and atomically replace the snapshot.

        Not how a model selection takes effect: the capabilities are fixed
        at construction, so a running scope that re-resolved its layers for
        a newly selected model would carry one model in its layers and
        another in its capabilities and its files. A selection is saved
        (`ScopeSession.select_model`) and applies at the next start.

        Args:
            override_model: Resolve AS this model for this call only --
                the lab/engineering escape hatch for exercising another
                model's identity on whatever is attached. Session-scoped
                by construction: it is never stored, so the next refresh
                without it resolves the real unit again, and it is never
                persisted anywhere.

        Returns:
            The new LayerIdentity snapshot (also on `self.layer_identity`).
        """
        self.layer_identity = self._resolve_layer_identity(override_model=override_model)
        return self.layer_identity

    def initialize(self, config: 'ScopeInitConfig') -> None:
        """Configure scope from connected to ready-to-use.

        Call once after construction.  Sets all scope-level hardware
        configuration.  Does NOT set per-layer camera settings (gain,
        exposure, auto-gain) -- those are the caller's responsibility
        for the active layer.

        Ends by releasing the camera start gate, so the live feed starts
        once the capture pixel format is on the camera.

        Args:
            config: ScopeInitConfig instance with all scope-level settings.
        """
        self._motion_expected = config.expects_motion
        self._notify_partial_hardware(config)
        # The safety-off is bound to the impl like every other write here,
        # never to the public dispatcher: bring-up is the scope configuring
        # itself, not a command from a caller, so it takes no lane and asks
        # no claim. The board check the dispatcher performs is copied here for
        # the same reason it lives there: with no board the composition root
        # installs a Null driver, which is truthy, so the impl's own `if not
        # self._driver` never fires and the state cache would record LEDs it
        # never drove. The write is bounded by the serial layer's own read
        # and write timeouts; nothing else holds the LED lock at bring-up.
        if self.led_connected:
            self.illumination._leds_off_impl()
        self.runtime_state.set_labware(config.labware)
        if config.turret_config:
            self.runtime_state.set_turret_config(config.turret_config)
        self.motion.seed_preferred_turret_slot(config.preferred_turret_slot)
        self.runtime_state.set_turreted(config.turreted)
        if config.turreted:
            # Nothing to set: the objective is the one assigned to the slot in
            # the light path, derived on every read. An assignment the
            # catalogue does not hold reads as unknown whenever its slot is in
            # the light path; said once here, not raised per read.
            catalogue = set(self.runtime_state.get_available_objectives())
            for slot, objective_id in (config.turret_config or {}).items():
                if objective_id is not None and objective_id not in catalogue:
                    logger.warning(
                        f'[SCOPE API ] turret slot {slot} is assigned {objective_id!r}, '
                        'which is not in the objective catalogue; its objective reads '
                        'as unknown until it is reassigned'
                    )
        else:
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
        # setting cannot skip the rest of bring-up: the caller of
        # initialize is the session's bring-up, where a propagated raise
        # aborts startup entirely (no live view, no
        # motion config, no session) over a single transient -- the
        # rejection is reported here, where its flight ends, and
        # every downstream consumer reads delivered geometry, never these
        # requests, so nothing is left believing a rejected value.
        # Bring-up binds the impls: these writes are the scope's own
        # composition, not external commands, so they stay direct on the
        # calling thread by design and nothing in this method dispatches.
        for apply_fn in (
            lambda: self.imaging._set_binning_size_impl(binning_size),
            lambda: self.imaging._set_frame_size_impl(frame_width, frame_height),
        ):
            try:
                apply_fn()
            except CameraSettingRejected as ex:
                # Bring-up continues at the value the camera holds.
                notifications.report_outcome(ex, solicited=False, category='Camera')
        # Apply the capture pixel format HERE, synchronously, while the start
        # gate is still closed (this runs before the start gate is released, below).
        # Resolving + setting it now -- instead of via the async camera-executor
        # push that the image-mode spinner enqueues -- removes the race where
        # the format lands after streaming begins and forces a redundant
        # grab-loop restart. The spinner handler returns early during init.
        pixel_format = image_mode.select_capture_pixel_format(
            config.capture_depth, self.imaging.get_supported_pixel_formats()
        )
        if pixel_format is not None:
            try:
                self.imaging._set_pixel_format_impl(pixel_format)
            except CameraSettingRejected as ex:
                # As for the geometry above: continue at the camera-held
                # format and report the rejection where it stops.
                notifications.report_outcome(ex, solicited=False, category='Camera')
        if self.capabilities.camera_supports_conversion_gain_mode:
            self.imaging._set_conversion_gain_mode_impl(
                'High' if config.high_conversion_gain else 'Low'
            )
        if self.capabilities.camera_supports_line_noise_reduction:
            self.imaging._set_line_noise_reduction_impl(config.line_noise_reduction)
        self.runtime_state.set_stage_offset(config.stage_offset)
        self.imaging.set_scale_bar(enabled=config.scale_bar_enabled)
        self.motion._set_acceleration_limit_impl(val_pct=config.acceleration_pct)
        # Last: the one-time release of the camera start gate, once the
        # capture pixel format above has been applied with the gate closed.
        self.imaging._start_streaming_impl()
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

    # --- The scope's lanes, for the session that composes around it ---

    def io_lane(self) -> SequentialIOExecutor:
        """The lane this scope runs its LED and motion commands on.

        Composition wiring for the session that holds this scope -- it asks
        the lane its activity claim and builds its run engine around it --
        and not part of the L2 API surface.
        """
        return self._io_executor

    def camera_lane(self) -> SequentialIOExecutor:
        """The lane this scope runs its camera commands on.

        Composition wiring for the session that holds this scope -- it asks
        the lane its activity claim and builds its run engine around it --
        and not part of the L2 API surface.
        """
        return self._camera_executor

    def set_camera_override_key(self, key: object) -> None:
        """Take the key the camera lane's claim returned to the session.

        The camera temperature read carries it, so the temperature log keeps
        running while a run or a diagnostic holds the scope. Composition
        wiring for the session, not part of the L2 API surface.
        """
        self._camera_override_key = key

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
    def motion_expected(self) -> bool:
        """Whether this scope's model has a motor board at all.

        False for a manual scope (an LS620 or LS560): it is complete
        without one, so its absence is not a disconnection. Set from the
        model's catalogue entry by ``initialize()``; True before that.
        """
        return self._motion_expected

    @property
    def led_connected(self) -> bool:
        """Whether the LED controller is connected.

        Returns:
            bool: True if a real (non-Null) LED board is connected.
        """
        return not isinstance(self._led_driver, NullLEDBoard) and self._led_driver.is_connected()

    def _camera_is_connected(self) -> bool:
        """Answer the camera half of every connection question, raising.

        One predicate with two callers that need opposite things from a
        driver that throws: the display paths below want a bool and get
        it from the property, while the run gate wants the throw, because
        "the USB tree went away mid-question" is a different refusal from
        "the camera is not plugged in" and the user has to be told which.

        It exists because the two were written out separately and drifted:
        the run gate's copy tested is_connected() alone while the property
        tested active as well. No camera in the tree tells them apart --
        all three make is_connected() False whenever active is unset -- so
        the drift was invisible rather than harmless, which is the worse
        of the two states to leave a predicate in.

        Returns a real bool rather than the falsy operand that ended the
        chain: a Pylon camera holds None in ``active`` once it is gone,
        and the display paths log this value.
        """
        driver = getattr(self, '_camera_driver', None)
        if driver is None or not getattr(driver, 'active', False):
            return False
        return driver.is_connected()

    @property
    def camera_connected(self) -> bool:
        """Whether the camera is connected and active.

        Returns:
            bool: True if a real camera driver is connected and active.
                A driver that raises reads as not connected: the callers
                here are display and metrics paths, where the question is
                asked per frame and has no answer but False.
        """
        try:
            return self._camera_is_connected()
        except Exception:
            return False

    def disconnect(self) -> None:
        """Disconnect from all hardware (LED, motion, camera) and stop the scope's lanes.

        Every step runs even if an earlier one fails. State is always reset
        to the Null variants and `_invalidate_camera_cache` always runs, so
        a partial failure cannot leave the API holding a stale connected
        driver. Repeatable: a second call finds nothing left to tear down.

        Raises:
            ScopeDisconnectError: after every step has run, naming each part
                that did not shut down cleanly (the motor STOP, the LED
                board, the motor board, the camera), chained from the first
                part's error. Nothing is logged or shown here; the caller
                reports it where it stops.
        """
        logger.info('[SCOPE API ] Disconnecting from microscope...')

        # Shut the lanes before anything is turned off, without waiting for
        # the work in flight: shutting a lane drops what is queued on it, so
        # an LED on or a move still waiting behind the current task cannot
        # run after the off and the stop below and leave the scope lit or
        # moving once it reads as disconnected. The task in flight is not
        # waited for; an LED write in flight holds the LED lock, which the
        # bounded off below waits on. A command sent afterwards is refused
        # at once rather than queued where no worker will run it.
        self._io_executor.shutdown(wait=False)
        self._camera_executor.shutdown(wait=False)

        # Darken the LEDs before any board is torn down. Closing the
        # serial port does not turn a board off -- the channels hold their
        # commanded current until something sends an off or power drops --
        # so a teardown without this leaves the sample illuminated until the
        # next connect's safety off, or indefinitely if there is no next
        # connect. Same defense-in-depth argument as the motor stop below:
        # every teardown path benefits without the caller having to
        # remember, and it means no LED-specific member has to be public for
        # a client to shut a scope down safely.
        #
        # Bounded acquire (the emergency variant) because an in-flight LED
        # write holding the lock must not be able to wedge teardown; it runs
        # first so the ordering the atexit hook relies on is unchanged.
        # Best-effort like every other step here: a failure is logged and
        # the port teardown still proceeds.
        try:
            self.illumination._leds_off_emergency()
        except Exception as ex:
            logger.exception(f'[SCOPE API ] LED shutoff during disconnect failed: {ex}')

        # LVP-A-1: stop motors before tearing down the serial port so we
        # don't leave a stage/turret moving against an end-stop after
        # the host stops responding to status polls. Defense in depth --
        # every disconnect path benefits without relying on the caller
        # to remember. A STOP that failed is recorded and the teardown
        # carries on whatever it raised: the ports still have to close.
        failures: dict[str, BaseException] = {}
        try:
            self.motion.stop_motion()
        except Exception as e:
            failures['motor stop'] = e

        # Stop the motion monitor and reset axis states -- MotionAPI._disconnect()
        # handles both: signals the monitor thread, waits for it, then resets
        # all axes to UNKNOWN and sets arrival events so waiters unblock.
        self.motion._disconnect()

        # Each sub-system: only attempt disconnect on a driver that
        # has one. Skips both the canonical no-op states (NullLEDBoard,
        # NullMotionBoard, self._camera_driver is None) and edge-case test
        # fixtures that bend the type system (e.g. `scope.led = object()`
        # for partial-hardware-warning tests). A skipped sub-system is
        # not a failure: "nothing to tear down" is success. The catches
        # are broad because a driver's teardown runs SDK code (pypylon,
        # ids_peak, pyusb) whose failure types are not ours, and the
        # teardown goes on whatever it raises.
        if not isinstance(self._led_driver, NullLEDBoard) and hasattr(
            self._led_driver, 'disconnect'
        ):
            try:
                self._led_driver.disconnect()
            except Exception as ex:
                failures['LED board'] = ex
        self._led_driver = NullLEDBoard()

        if not isinstance(self._motion_driver, NullMotionBoard) and hasattr(
            self._motion_driver, 'disconnect'
        ):
            try:
                self._motion_driver.disconnect()
            except Exception as ex:
                failures['motor board'] = ex
        self._motion_driver = NullMotionBoard()

        if self._camera_driver is not None and hasattr(self._camera_driver, 'disconnect'):
            # A camera driver's False is not a failure here: it answers
            # False both for a camera already gone (nothing to tear down)
            # and for a teardown error it caught and logged itself, and the
            # two cannot be told apart from here. Only a raise is a part
            # that did not shut down.
            try:
                self._camera_driver.disconnect()
            except Exception as ex:
                failures['camera'] = ex
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

        if not failures:
            logger.info('[SCOPE API ] Microscope disconnected')

        # Symmetric to atexit.register in __init__: each instance removes its
        # own hook on disconnect so test fixtures that construct + disconnect
        # many Lumascope instances do not leak atexit registrations.
        # atexit.unregister silently no-ops if the hook was never registered.
        try:
            import atexit

            atexit.unregister(self._emergency_shutdown)
        except Exception as _e:
            logger.warning(f'[SCOPE API ] atexit unregister failed: {_e}')

        if failures:
            raise ScopeDisconnectError(failures) from next(iter(failures.values()))

    def _emergency_shutdown(self):
        """LVP-A-7: best-effort safety shutdown for atexit / abnormal exit.

        Guards LEDs and motor against the interpreter terminating mid-
        operation: turns off all LEDs, then disconnects (which now also
        stops motion via the LVP-A-1 chain). Swallows every exception so
        atexit completes cleanly even when the logging stack or hardware
        access is already torn down.

        The LED shutoff is not repeated here: `disconnect()` performs it as
        its first step, using the same bounded acquire, so there is one
        place that darkens the channels on the way down rather than two
        that have to be kept in agreement.
        """
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

    def move_to_simulated_sample_plane(self) -> None:
        """Place the stage where the simulated sample is (not part of the L2 API surface).

        Bring-up's own step, called by ``ScopeSession.start_application_session``.
        An L2 caller has no reason to reach it: on real hardware it does
        nothing, and on a simulator bring-up has already run it.

        Homing leaves Z at the bottom of travel on a real instrument and on
        the simulator alike, and that is correct -- it is what homing means.
        On a real scope the operator then focuses; in the simulator nobody
        does, so every session began at the floor, where a z-stack asks for
        slices below zero and an autofocus sweep starts 5 mm from anything
        worth focusing on.

        The height is not chosen here. The simulated camera already declares
        the plane its specimen is sharp at, and reading it is what keeps the
        two halves of one simulated scene agreeing about where the sample
        sits rather than each holding a number.

        Placed only on a scope that has a Z axis to place, and only once Z
        has a reference position. A convenience may not be able to break
        bring-up: an absolute move on an axis that was never homed refuses
        rather than guessing, so this asks first instead of provoking that
        refusal and catching it -- catching would hide a real one.
        """
        if not self._simulated:
            return
        if not self.capabilities.has_focus:
            return
        if not self.motion.position_is_known('Z'):
            logger.info(
                '[SCOPE API ] Simulated sample plane not applied: Z has no reference position'
            )
            return
        # A camera that failed to construct leaves a driver that is not the
        # simulator, so the focal plane is asked for rather than assumed.
        focal_plane = getattr(self._camera_driver, 'get_focal_z', None)
        if focal_plane is None:
            return
        # Waited, like the home before it: bring-up reports the stage ready,
        # and a caller that starts a sweep against a Z still in flight reads
        # a position that is not where the sample is.
        self.motion.move_absolute('Z', focal_plane(), wait_until_complete=True)

    def are_all_connected(self) -> bool:
        """Check if LED, motion, and camera boards are all connected.

        Each term is the one the matching single-board question asks, so
        a run gate and a per-board gate cannot disagree about the same
        hardware. A driver that RAISES propagates: the run gate that asks
        this converts it into a refusal that says the state could not be
        read, which is not the same answer as "not connected".

        A scope whose model has no motor board is not asked for one: a
        manual scope is complete without it.

        Returns:
            bool: True if all three components are connected.
        """
        logger.debug('[SCOPE API ] Performing connection check...')
        led = self.led_connected
        motion = self.motor_connected or not self._motion_expected
        camera = self._camera_is_connected()

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

        Internal degraded-mode constructor for support reports -- not
        part of the L2 API surface.

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
        instance.motion._init_axes(present_axes, instance._motion_driver.detect_homed_axes())
        instance.motion._start_monitor()

        instance._frame_buffer = None

        # Diagnostic instances have no settings to name a configured
        # model, so identity resolves from the motor-reported model alone
        # -- empty (and loud on LED use) when the hardware reports none.
        # That is the honest answer for a support tool: it must never
        # invent an identity for a unit it is diagnosing.
        instance._configured_model = None
        instance.layer_identity = instance._resolve_layer_identity()

        # Build capabilities -- diagnostic instances still need this so
        # any code that reads scope.capabilities.* works.
        instance.capabilities = ScopeCapabilities.from_drivers(
            motion=instance._motion_driver,
            led=instance._led_driver,
            camera=None,
            layer_identity=instance.layer_identity,
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
        from modules.lumascope_api.protocols import ProtocolsAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        instance.illumination = IlluminationAPI(instance, instance._led_driver)
        instance.imaging = ImagingAPI(instance, None)
        instance.diagnostics = DiagnosticsAPI(instance)
        instance.io = IOAPI(instance)
        instance.protocols = ProtocolsAPI(instance)
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
