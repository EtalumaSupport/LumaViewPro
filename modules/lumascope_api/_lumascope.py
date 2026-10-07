#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import copy
import dataclasses
import sys
import warnings

from lvp_logger import logger

# Import Lumascope Hardware files
from modules.lumascope_api import _constants as _api_constants
from modules.lumascope_api._constants import SIMULATOR_TIERS
import modules.image_mode as image_mode

# FX2 (Lumaview Classic LS560/LS620/LS720) -- the import side-effect is
# the entire point: it fires the @camera_registry.register('fx2') and
# @led_registry.register('fx2') decorators inside the module. Nothing
# in this file references fx2driver names directly; the registry
# instantiates FX2Camera + FX2LEDController via 'auto' fallthrough when
# Pylon/IDS aren't found. Wrapped in try/except so dev machines without
# pyusb / libusb1 don't crash LVP at startup (matches the IDS pattern below).
try:
    import drivers.fx2driver  # noqa: F401
except ImportError as _fx2_exc:
    # Same silent-degradation shape as the IDS guard below: without this
    # line a Classic scope's missing camera has no named cause anywhere.
    logger.warning(f'[SCOPE API ] FX2 (Classic) drivers unavailable: {_fx2_exc}')
from drivers.camera import Camera

# Registration-only imports: loading each driver module fires its
# @*_registry.register(...) decorator so the registry can instantiate it
# by kind ('pylon', 'sim') via create(). No name below is referenced
# directly here; dropping these empties the registry -- simulate mode then
# finds no 'sim' drivers and startup aborts.
from drivers.ledboard import LEDBoard  # noqa: F401
from drivers.motorboard import MotorBoard  # noqa: F401
from drivers.pyloncamera import PylonCamera  # noqa: F401
from drivers.simulated_camera import SimulatedCamera  # noqa: F401
from drivers.simulated_motorboard import SimulatedMotorBoard  # noqa: F401
from drivers.simulated_ledboard import SimulatedLEDBoard  # noqa: F401
from drivers.null_motorboard import NullMotionBoard
from drivers.null_ledboard import NullLEDBoard
from drivers.protocols import MotorBoardProtocol, LEDBoardProtocol
from drivers.registry import motor_registry, led_registry, camera_registry
import modules.binning as binning
from modules.exceptions import (
    BinningSubstitutedNotice,
    InstallationFileError,
    FrameRefittedNotice,
    CameraNotAvailableError,
    CameraSettingRejected,
    ConfigError,
    LedBoardUnavailableError,
    LedSafetyOffNotTakenError,
    MissingPart,
    NoHardwareDetectedNotice,
    PartialHardwareError,
    ScopeDisconnectError,
)
from modules.lumascope_api.bring_up import (
    CAMERA,
    LED,
    MOTOR,
    BringUpRecord,
    PartStatus,
    Substitution,
)
from modules.path_utils import get_source_root, read_installation_file, resolve_data_file
from modules.scope_capabilities import ScopeCapabilities
from modules.sequential_io_executor import SequentialIOExecutor
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os

    from drivers.simulated_camera import SimulatedStall
    from modules.layer_record import LayerIdentity
    from modules.scope_init_config import ScopeInitConfig

# Import additional libraries
import logging as _logging

from modules.notification_center import notifications

_api_log = _logging.getLogger('LVP.api')


def _register_ids_camera(platform: str):
    """Import the IDS driver, which registers it; None when it cannot load.

    IDS peak ships Windows and Linux builds only, so on macOS no IDS camera
    can ever run and the driver is not attempted: its absence there is a
    fact of the host, logged once at INFO. Elsewhere a failed import is a
    WARNING with its reason, because the driver then silently never
    registers and an IDS scope just "has no camera" -- a bundling gap in a
    frozen build cost a full client misdiagnosis.
    """
    if platform == 'darwin':
        logger.info(
            '[SCOPE API ] IDS cameras are not supported on macOS (IDS peak has no macOS build)'
        )
        return None
    try:
        from drivers.idscamera import IDSCamera
    except ImportError as exc:
        logger.warning(f'[SCOPE API ] IDS camera driver unavailable: {exc}')
        return None
    return IDSCamera


IDSCamera = _register_ids_camera(sys.platform)

# The boards a catalogue row may name that the simulator can stand in for.
# An FX2 drives its scope's camera and LEDs; an EL-0940 is a board of its own.
_SIMULATED_LED_BOARDS = ('EL-0940', 'FX2')
_SIMULATED_MOTOR_BOARDS = ('EL-0940',)

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
# What a part's failure to come up means, for the bring-up record
# ---------------------------------------------------------------------------


def _camera_failure_cause(exc: BaseException) -> str:
    """The record's cause for a camera that raised while connecting.

    The camera registry raises whatever the backend raised (pypylon,
    ids_peak, FX2, simulated). pypylon's RuntimeException for "camera already
    open in another application" is the frequent case -- Pylon Viewer or a
    second LVP -- and gets its own cause. Matched by type name so pypylon is
    not imported on a host without it.
    """
    if type(exc).__name__ in ('RuntimeException', 'GenericException', 'LogicalErrorException'):
        return 'camera_in_use'
    if isinstance(exc, PermissionError):
        return 'camera_port_in_use'
    if isinstance(exc, FileNotFoundError):
        return 'camera_not_detected'
    return 'camera_not_initialized'


def _board_status(part: str, board, fallback) -> 'PartStatus':
    """The record of a board the registry built, real or null.

    A real LED board reports a connect-time LEDS_OFF that did not complete
    through ``last_safety_off_error``; it is the one problem a board that
    came up can have, and a sample-safety one (older firmware can leave
    channels on), so it is on the record.
    """
    if fallback is not None:
        return PartStatus(part, up=False, cause=fallback.cause, detail=fallback.detail)
    safety_error = getattr(board, 'last_safety_off_error', None)
    if safety_error:
        return PartStatus(part, up=True, cause='safety_off_failed', detail=str(safety_error))
    return PartStatus(part, up=True)


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

    def _read_catalogues(self, source_path: 'str | os.PathLike') -> dict:
        """Read the installation's files once, from ``source_path``; return the motor defaults.

        The scope is the one owner of the labware, objective and model
        catalogues. Its runtime state, protocol construction, the session,
        the run and the GUI all read these objects, so no two parts of one
        session can disagree about which plates, objectives or models
        exist. All are read-only after construction, so every thread may
        share them. The motor defaults are returned rather than kept: the
        motor driver takes them, and nothing else reads them.

        Raises:
            InstallationFileError: a file is missing, unreadable, or not the
                shape its reader needs, naming the file; or the release's
                layer vocabulary is unusable.
        """
        from modules import labware_loader, layer_record, objectives_loader

        # A string, the type the session's source_path has always had.
        self.source_path = str(source_path)
        self.wellplate_loader = labware_loader.WellPlateLoader(source_path=source_path)
        self.objective_helper = objectives_loader.ObjectiveLoader(source_path=source_path)
        # Kept so a refusal of one of its rows names the file it came from.
        self._scope_models_path = resolve_data_file('scopes.json', source_path=source_path)
        self._scope_models = layer_record.load_scope_models(self._scope_models_path)
        # The release's layer vocabulary is process-wide, not this folder's,
        # but the identity resolved after the lanes start needs it: asked
        # here, a broken one refuses before anything is started.
        layer_record.release_catalogue()
        motorconfig_defaults = read_installation_file(
            resolve_data_file('motorconfig_defaults.json', source_path=source_path)
        )
        # What every setting is: the shipped template, which the Session's
        # settings writer checks a path and a value's kind against.
        self._settings_template = read_installation_file(
            resolve_data_file('settings.json', source_path=source_path)
        )
        return motorconfig_defaults

    def _init_minimal(self, simulated: bool) -> None:
        """Shared init for state slots both __init__ and create_diagnostic need.

        Sets the non-driver state that every Lumascope instance must
        carry: transformers, locks, camera cache, objective state slots,
        the scope's two lanes. Both __init__ and
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
        self._settings_reader: Callable[[str], Any] | None = None

        # The labware, stage offset, turret map, selected objective and
        # whether the scale bar is drawn are the session's settings, held
        # nowhere here: runtime_state and imaging read them through the
        # reader a session binds (bind_settings), so there is no second copy
        # to fall out of step. None until bound.
        #
        # _state_lock + _cam_lock + ImagingAPI's
        # own caches live on self.imaging. _last_turret_position lives
        # on self.motion. engineering_mode lives on the app context
        # (ctx.engineering_mode).

        # The scope's two lanes, built and started here and shut by
        # disconnect(). Every LED, motion and camera command goes through
        # one, so commands from any caller -- a session's GUI or REST, or a
        # script's bare scope -- run one at a time per bus, in order, and a
        # run holding the scope can refuse what is not its own. A session
        # composed over this scope asks the lanes its activity claim.
        self._io_executor = SequentialIOExecutor(name='IO')
        self._camera_executor = SequentialIOExecutor(name='CAMERA')
        self._io_executor.start()
        self._camera_executor.start()
        # The key the camera lane's claim returned to the session, which
        # hands it here: the camera temperature read carries it, so the
        # temperature log keeps running while a run or a diagnostic holds
        # the scope. None until a session asks the claim.
        self._camera_override_key = None

    @staticmethod
    def _build_simulated_motor_board(
        model: str,
        axes: frozenset[str],
        motor_board: str | None,
        sim_tier: str,
        motorconfig_defaults: dict,
    ) -> MotorBoardProtocol:
        """The simulated scope's motor board, on the tier asked for.

        ``axes`` and ``motor_board`` are the ones the catalogue gives the
        model: a model that names no motor board has none, so it gets the
        null driver on either tier. The fast tier goes through the
        registry's simulator selection with those axes. The firmware tier
        builds the production driver by name against the emulator: a model
        with axes whose emulator does not come up raises, because the
        registry's auto path would fall back to the null driver and a dead
        emulator would then look exactly like a manual scope.
        """
        if sim_tier not in SIMULATOR_TIERS:
            raise ValueError(f'sim_tier {sim_tier!r} is not one of {SIMULATOR_TIERS}')
        if motor_board is None:
            logger.info(f'[SCOPE API ] Model {model} has no motor axes: no motor board')
            return NullMotionBoard()
        if sim_tier == 'fast':
            board = motor_registry.create(
                'auto',
                simulate=True,
                model=model,
                axes=axes,
                motorconfig_defaults=motorconfig_defaults,
            )
            logger.info(f'[SCOPE API ] Using SIMULATED Motor Board (model={model})')
            return board
        from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend

        backend = SimWireBackend(MotorBoardSpec(model, axes))
        board = motor_registry.create(
            'rp2040', backend=backend, motorconfig_defaults=motorconfig_defaults
        )
        logger.info(
            f'[SCOPE API ] Using the motor FIRMWARE in simulation '
            f'(model={model}, axes={"".join(sorted(axes))})'
        )
        return board

    def _simulated_boards(self, model: str, axes: frozenset[str]) -> tuple[str, str | None]:
        """The LED and motor boards a simulated ``model`` has, as its catalogue row names them.

        The row names an ``LEDBoard`` always and a ``MotorBoard`` exactly
        when it has motor axes. Production never reads either -- the
        bring-up finds the boards it has -- so this is the one place the
        rule is checked: a row that breaks it, or names a board the
        simulator cannot stand in for, would build a scope unlike the one
        the row describes. ``axes`` are the row's, from ``model_axes``.

        Raises:
            InstallationFileError: the row breaks the rule or names a board
                the simulator has no stand-in for, naming the catalogue.
        """
        entry = self._scope_models[model]
        led_board = entry.get('LEDBoard')
        motor_board = entry.get('MotorBoard')
        if led_board not in _SIMULATED_LED_BOARDS:
            problem = f'names LEDBoard {led_board!r}; the simulator has {_SIMULATED_LED_BOARDS}'
        elif axes and motor_board not in _SIMULATED_MOTOR_BOARDS:
            problem = (
                f'gives motor axes and names MotorBoard {motor_board!r}; '
                f'the simulator has {_SIMULATED_MOTOR_BOARDS}'
            )
        elif not axes and 'MotorBoard' in entry:
            problem = f'gives no motor axes but names MotorBoard {motor_board!r}'
        else:
            return led_board, motor_board
        raise InstallationFileError(
            self._scope_models_path, f'has a model {model!r} that {problem}'
        )

    @staticmethod
    def _build_simulated_led_board(model: str, sim_tier: str) -> LEDBoardProtocol:
        """The simulated EL-0940 scope's LED board, on the tier asked for.

        The firmware tier builds the production driver by name against the
        emulator, so an emulator that does not come up raises instead of
        becoming a stand-in. The tier is the one the motor board was just
        built on, which refused any tier that is not one of the two.
        """
        if sim_tier == 'fast':
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
        fx2_debug_wire: bool = False,
        *,
        source_path: 'str | os.PathLike | None' = None,
        sim_camera_stall: 'SimulatedStall | None' = None,
    ):
        """Initialize Microscope.

        Args:
            source_path: The data folder this scope is started on -- the
                folder holding ``data/``. The scope reads the labware and
                objective catalogues from it once, here, and everything
                that asks about plates or objectives reads those copies.
                None (default) is the installation's own folder.
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
            fx2_debug_wire: Log every byte of each LED command an FX2
                (Classic) LED board sends, and the illumination cache check
                in front of it -- a bench diagnostic, off by default. The
                session passes the ``fx2_debug_wire_enabled`` setting.
            sim_camera_stall: A stall for the simulated camera's stream:
                frames stop for a while, the camera staying connected and
                grabbing, so a simulated scope shows a stalled stream without
                hardware. Only the simulated camera has one, so it is refused
                on real hardware and on a model simulated with an FX2.

        Raises:
            ValueError: ``sim_camera_stall`` given for a scope whose camera is
                not the simulated camera.
        """
        if sim_camera_stall is not None and not simulate:
            raise ValueError('a simulated camera stall needs a simulated scope (simulate=True)')
        if warn_pre_release:
            _fire_pre_release_warning()
        self._fx2_debug_wire = fx2_debug_wire

        # Read before anything is started, so a missing or unusable
        # installation file stops the bring-up with nothing to tear down.
        # The motor defaults are read on every model: the motor probe below
        # runs on every model, and a board it finds takes them.
        motorconfig_defaults = self._read_catalogues(get_source_root(source_path))
        from modules.layer_record import entry_expects_motion, model_axes

        # Decided here, before anything is started, for the same reason: a
        # model the catalogue does not list refuses before a lane exists.
        if simulate:
            from modules.settings_init import settings

            default_model = settings.get('microscope', 'LS850T') if settings else 'LS850T'
            model = sim_model or configured_model or default_model
            sim_axes = model_axes(self._scope_models, model)
            sim_led_board, sim_motor_board = self._simulated_boards(model, sim_axes)
            if sim_camera_stall is not None and sim_led_board == 'FX2':
                raise ValueError(
                    f'a simulated camera stall needs the simulated camera, and {model} is '
                    'simulated with an FX2'
                )
        else:
            # Whether the selected model is a manual scope, so a probe that
            # finds no motor board says so as expected rather than warning
            # on every start. The probe still runs: a board it finds corrects
            # a wrongly selected model.
            motor_absence_expected = not entry_expects_motion(
                self._scope_models.get(configured_model)
            )

        # Shared state-slot init (audit #35) -- transformers, locks,
        # camera cache, objective/turret state, the scope's lanes.
        # Driver construction + sub-API wiring happen below.
        self._init_minimal(simulated=simulate)
        # What each part did while connecting, written as the drivers are
        # built below and read back as the bring-up record. A simulated part
        # always comes up: the simulator is what it stands in for.
        parts: dict[str, PartStatus] = {}

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
            self._motion_driver: MotorBoardProtocol = self._build_simulated_motor_board(
                model, sim_axes, sim_motor_board, sim_tier, motorconfig_defaults
            )
            # A simulated manual scope gets the null board, as the bench
            # finds none: not up, and not missing once the model says so.
            parts[MOTOR] = PartStatus(MOTOR, up=sim_motor_board is not None)
        else:
            self._motion_driver, fallback = motor_registry.create_with_fallback(
                'auto',
                absence_expected=motor_absence_expected,
                motorconfig_defaults=motorconfig_defaults,
            )
            parts[MOTOR] = _board_status(MOTOR, self._motion_driver, fallback)

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
        # A simulated FX2 scope runs the production FX2 drivers on both tiers
        # over one simulated device: the FX2 has no firmware to emulate, so
        # its device model is the one simulation. Built by name, since the
        # registry lists no FX2 on a host without libusb.
        sim_fx2 = None
        if simulate and sim_led_board == 'FX2':
            from drivers.fx2driver import FX2LEDController
            from drivers.simulated_fx2 import SimulatedFX2

            sim_fx2 = SimulatedFX2()
            self._led_driver: LEDBoardProtocol = FX2LEDController(
                connection=sim_fx2.connection, debug_wire=fx2_debug_wire
            )
            logger.info(f'[SCOPE API ] Using the FX2 LED driver on a SIMULATED FX2 (model={model})')
            parts[LED] = PartStatus(LED, up=True)
        elif simulate:
            self._led_driver = self._build_simulated_led_board(model, sim_tier)
            parts[LED] = PartStatus(LED, up=True)
        else:
            self._led_driver, fallback = led_registry.create_with_fallback(
                'auto', debug_wire=fx2_debug_wire
            )
            parts[LED] = _board_status(LED, self._led_driver, fallback)

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
        if simulate and sim_fx2 is None:
            camera_kwargs['z_position_func'] = lambda: self.motion.get_current_position('Z')
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
        # The exception a camera raised while connecting, kept until bring-up
        # reports it so the report carries the backend's own traceback.
        self._camera_failure: BaseException | None = None
        try:
            if sim_fx2 is not None:
                from drivers.fx2driver import FX2Camera

                self._camera_driver: Camera = FX2Camera(connection=sim_fx2.connection)
                logger.info(
                    f'[SCOPE API ] Using the FX2 camera driver on a SIMULATED FX2 (model={model})'
                )
            else:
                self._camera_driver = camera_registry.create(
                    camera_type, simulate=simulate, **camera_kwargs
                )
                if simulate:
                    self._camera_driver.load_cycle_images()
                    if sim_camera_stall is not None:
                        self._camera_driver.hold_frames(sim_camera_stall)
                    logger.info('[SCOPE API ] Using SIMULATED Camera')
            parts[CAMERA] = PartStatus(CAMERA, up=True)
        except Exception as _cam_exc:
            self._camera_failure = _cam_exc
            parts[CAMERA] = PartStatus(
                CAMERA,
                up=False,
                cause=_camera_failure_cause(_cam_exc),
                detail=f'{type(_cam_exc).__name__}: {_cam_exc}',
            )
        self._bring_up_parts = parts
        self._bring_up_substitutions: list[Substitution] = []

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
            scope_models=self._scope_models,
        )

        # ----- Sub-API wiring -----
        # motion was already constructed above (it needs earlier
        # construction so _init_axes / _start_monitor can run before the
        # LED/camera drivers are set up). Remaining sub-APIs:
        from modules.lumascope_api.illumination import IlluminationAPI
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from modules.lumascope_api.protocols import ProtocolsAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        self.illumination = IlluminationAPI(self, self._led_driver)
        self.imaging = ImagingAPI(self, self._camera_driver)
        self.diagnostics = DiagnosticsAPI(self)
        self.protocols = ProtocolsAPI(self)
        self.runtime_state = RuntimeState(self)

        # What came up is reported by initialize(config): it takes the
        # model's expectations, which say whether a missing motor board on an
        # LS620 is the manual scope it is or an LS820 whose board failed.

        # Whether any real hardware was found: read from the record, the one
        # account of what came up, so this and the report cannot disagree.
        self._no_hardware = not simulate and not any(status.up for status in parts.values())

        # Most per-instance state lives on the sub-APIs: imaging owns
        # camera-stream state + locks, motion owns per-axis state +
        # _last_turret_position, illumination owns LED state,
        # runtime_state owns settings-host state (labware / objective /
        # turret_config / stage_offset). Lumascope holds driver slots,
        # its two lanes, its data folder and the catalogues read from it.

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
        from modules.layer_record import release_catalogue, resolve_layer_identity

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
            models=self._scope_models,
            catalogue=release_catalogue(),
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
        self._report_bring_up(config)
        # The safety-off is the scope's own write, never the public
        # dispatcher: bring-up is the scope configuring itself, not a command
        # from a caller, so it takes no lane and asks no claim. It asks
        # presence first, so with no board connected it writes nothing. The
        # write is bounded by the serial layer's own read and write timeouts;
        # nothing else holds the LED lock at bring-up.
        self.illumination._leds_off_if_present()
        # A saved slot carried over from a turret scope means nothing here.
        if self.capabilities.has_turret:
            self.motion.seed_preferred_turret_slot(config.preferred_turret_slot)
        self.runtime_state.set_turreted(config.turreted)
        if config.turreted:
            # The objective is the one assigned to the slot in the light
            # path, derived on every read. An assignment the catalogue does
            # not hold reads as unknown whenever its slot is in the light
            # path; said once here, not raised per read.
            catalogue = set(self.runtime_state.get_available_objectives())
            for slot, objective_id in self.runtime_state.get_turret_config().items():
                if objective_id is not None and objective_id not in catalogue:
                    logger.warning(
                        f'[SCOPE API ] turret slot {slot} is assigned {objective_id!r}, '
                        'which is not in the objective catalogue; its objective reads '
                        'as unknown until it is reassigned'
                    )
        else:
            # Refused here, before anything is commanded: a stored id that
            # names no catalogue objective would otherwise be stamped as the
            # scale of every capture.
            self.objective_helper.get_objective_info(objective_id=self.read_setting('objective_id'))
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
            available_binning = self.capabilities.camera_binning_sizes
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
                    self.imaging._max_frame_unbinned()
                    or {
                        'width': frame_width * binning_size,
                        'height': frame_height * binning_size,
                    },
                )
                refit = binning.native_to_displayed(
                    native, camera_binning, self.imaging.get_pixel_alignment()
                )
                logger.info(
                    f'[SCOPE API ] initialize: the frame {frame_width}x{frame_height} saved '
                    f'at binning {binning_size} is refit to {refit["width"]}x'
                    f'{refit["height"]} at the camera-reported {camera_binning} '
                    f'(available: {available_binning})'
                )
                self._bring_up_substitutions.append(
                    Substitution('binning', saved=binning_size, used=camera_binning)
                )
                notifications.report_outcome(
                    BinningSubstitutedNotice(binning_size, camera_binning),
                    solicited=False,
                    category='Camera',
                )
                binning_size = camera_binning
                frame_width, frame_height = refit['width'], refit['height']
            # A saved frame can be larger than this scope delivers at the
            # binning applied (a frame saved on another model; the LS560's lens
            # images 1700 of the sensor's 1900): it is refitted to the maximum,
            # as the API would refuse it, and the replacement is on the record
            # and reported once. The session stores the frame that ran.
            maximum = self.imaging._max_frame_unbinned()
            if maximum:
                fitted = (
                    min(frame_width, maximum['width'] // binning_size),
                    min(frame_height, maximum['height'] // binning_size),
                )
                if fitted != (frame_width, frame_height):
                    self._bring_up_substitutions.append(
                        Substitution('frame', saved=(frame_width, frame_height), used=fitted)
                    )
                    notifications.report_outcome(
                        FrameRefittedNotice((frame_width, frame_height), fitted, binning_size),
                        solicited=False,
                        category='Camera',
                    )
                    frame_width, frame_height = fitted
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
        # With no camera connected there is nothing to apply them to, and
        # the camera's absence was reported when it did not come up.
        if self.camera_connected:
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
        # The saved mode runs as saved on every camera: a mode is a save
        # policy (reduce to 8 bits, or keep the depth the frame has), and the
        # format is chosen from the ones this camera reports -- its 12-bit
        # format where it has one, else its own 8-bit format, which a
        # full-depth mode keeps at the depth delivered. Nothing is
        # substituted. None means the camera reported no formats: there is
        # nothing to apply. With no camera connected there is nothing to
        # send the format to.
        pixel_format = image_mode.select_capture_pixel_format(
            image_mode.resolve_image_mode(config.image_mode)['capture_depth'],
            self.capabilities.camera_pixel_formats,
        )
        if pixel_format is not None and self.camera_connected:
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
        # Asked first: a scope with no motor controller has no limit to set.
        if self.motor_connected:
            self.motion._set_acceleration_limit_impl(val_pct=config.acceleration_pct)
        # Last: the one-time release of the camera start gate, once the
        # capture pixel format above has been applied with the gate closed.
        self.imaging._start_streaming_impl()
        logger.info('[SCOPE API ] Scope initialized')

    def bind_settings(self, reader: 'Callable[[str], Any]') -> None:
        """Read the configuration this scope acts on from a session's settings.

        Composition wiring, not part of the L2 API surface: a session binds
        every scope it composes, once, before bring-up, after its lanes have
        refused a second session over the same scope. ``reader`` answers a
        copy of the setting at a dotted path (``ScopeSession.get_setting``).
        """
        self._settings_reader = reader

    def read_setting(self, path: str) -> Any:
        """A copy of the setting at ``path``, from the session this scope is bound to.

        A consult seam for the sub-APIs, not part of the L2 API surface: an
        L2 caller reads settings through its session.

        Raises:
            ConfigError: No session has bound this scope, so it has no
                settings to act on: a bare scope that would capture, convert
                a plate position or draw a scale bar is composed into a
                ``ScopeSession`` first.
        """
        if self._settings_reader is None:
            raise ConfigError(
                f'this scope has no settings to read {path!r} from: compose it into a '
                'ScopeSession (ScopeSession.create) before using it'
            )
        return self._settings_reader(path)

    def bring_up_record(self) -> BringUpRecord:
        """What this scope's bring-up found and substituted.

        Composition wiring for the session, not part of the L2 API surface:
        the session adds what it knows (the settings file set aside) and
        offers the whole as ``ScopeSession.bring_up_record``. Before
        ``initialize`` every part is expected, as an unconfigured scope is
        held to every board.
        """
        return BringUpRecord(
            parts=tuple(self._bring_up_parts.values()),
            substitutions=tuple(self._bring_up_substitutions),
        )

    def _report_bring_up(self, config) -> None:
        """Report, once, what did not come up, from the record's facts.

        The model's expectations arrive with the config: an LS620 with no
        motor board is the manual scope it is, not a failure, so only a part
        the model has is missing. A camera that failed is reported whatever
        the model, with the backend's own traceback behind it. When nothing
        came up the person is told once, not once per part. A simulated part
        is never reported: the simulator is what it stands in for.
        """
        parts = self._bring_up_parts
        parts[MOTOR] = dataclasses.replace(parts[MOTOR], expected=config.expects_motion)
        parts[LED] = dataclasses.replace(parts[LED], expected=config.expects_led)
        if self._no_hardware:
            notifications.report_outcome(
                NoHardwareDetectedNotice(), solicited=False, category='Hardware'
            )
            return
        camera = parts[CAMERA]
        if not camera.up:
            failed = CameraNotAvailableError(camera.cause)
            failed.__cause__ = self._camera_failure
            self._camera_failure = None
            notifications.report_outcome(failed, solicited=False, category='Camera')
        if self._simulated:
            return
        led = parts[LED]
        if not led.up:
            notifications.report_outcome(
                LedBoardUnavailableError(led.cause), solicited=False, category='Illumination'
            )
        elif led.cause == 'safety_off_failed':
            notifications.report_outcome(
                LedSafetyOffNotTakenError(led.detail), solicited=False, category='Illumination'
            )
        missing = self.bring_up_record().missing
        if missing:
            notifications.report_outcome(
                PartialHardwareError(status.describe() for status in missing),
                solicited=False,
                category='Hardware',
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
    def scope_models(self) -> dict:
        """The model catalogue (scopes.json's ``Models``), as the caller's own copy.

        A copy because the scope reads its own: a caller that changed what
        it was given would otherwise change every later reader's catalogue.
        """
        return copy.deepcopy(self._scope_models)

    @property
    def settings_template(self) -> dict:
        """The shipped settings template: which settings exist, and their shipped values.

        The caller's own copy, for the reason ``scope_models`` gives.
        """
        return copy.deepcopy(self._settings_template)

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

        # Stop watching the stream first: the teardown below stops it on
        # purpose, and a stall check still running would report that as a
        # camera fault. Best-effort, like the LED shutoff below: a failure is
        # logged and the teardown carries on.
        try:
            self.imaging.stop_stream_check()
        except Exception as ex:
            logger.exception(f'[SCOPE API ] stopping the stream check failed: {ex}')

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
        # Asked first, as the scope's own write: with no motor controller
        # connected -- a manual scope, a pulled cable, a second disconnect --
        # there is nothing to stop.
        failures: dict[str, BaseException] = {}
        if self.motor_connected:
            try:
                self.motion._stop()
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
        # The emergency off above leaves the state store as it was; with the
        # board gone nothing is lit, and the listeners hear it here (a
        # listener's fault is reported by the listener bus, not raised).
        self.illumination._forget_led_state()

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
        # Waited, like the home before it: move_absolute returns only once Z
        # has arrived, and raises if it did not. Bring-up reports the stage
        # ready, and a caller that starts a sweep against a Z still in flight
        # reads a position that is not where the sample is.
        self.motion.move_absolute('Z', focal_plane())

    def are_all_connected(self) -> bool:
        """Check if LED, motion, and camera boards are all connected.

        See ``unconnected_parts``, which this asks.

        Returns:
            bool: True if all three components are connected.
        """
        return not self.unconnected_parts()

    def unconnected_parts(self) -> tuple[str, ...]:
        """The parts this scope needs that are not connected, by name.

        Each term is the one the matching single-board question asks, so
        a run gate and a per-board gate cannot disagree about the same
        hardware. A driver that RAISES propagates: the run gate that asks
        this converts it into a refusal that says the state could not be
        read, which is not the same answer as "not connected".

        A scope whose model has no motor board is not asked for one: a
        manual scope is complete without it.

        Returns:
            Each of ``'LED controller'``, ``'motor controller'`` and
            ``'camera'`` that is not connected, in that order; empty when
            all are.
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

        return tuple(
            name
            for name, connected in (
                (MissingPart.LED_CONTROLLER.name, led),
                (MissingPart.MOTOR_CONTROLLER.name, motion),
                (MissingPart.CAMERA.name, camera),
            )
            if not connected
        )

    @classmethod
    def create_diagnostic(cls, source_path: 'str | os.PathLike | None' = None) -> 'Lumascope':
        """Create a minimal Lumascope for diagnostics (no camera init).

        Internal degraded-mode constructor for support reports -- not
        part of the L2 API surface.

        Connects to LED and motor boards only. For use by tools like
        the tech support report that need board access without the full
        application stack.

        Args:
            source_path: The data folder to read the catalogues from; None
                (default) is the installation's own folder.

        Returns:
            Lumascope: Instance with led/motion connected, camera=None.
        """
        instance = cls.__new__(cls)
        # The same read __init__ makes, first, before anything is started:
        # one construction path for the installation's files on every
        # scope, and a missing one stops the diagnostic with nothing to
        # tear down.
        motorconfig_defaults = instance._read_catalogues(get_source_root(source_path))
        # Shared state-slot init (audit #35) -- same call __init__ makes.
        instance._init_minimal(simulated=False)

        # Connect boards through the registries, as __init__ does -- motion
        # driver first so MotionAPI._driver resolves correctly at
        # construction time. The diagnostic has no settings to say what the
        # model expects, so it is held to every board, and it reports
        # nothing: its record is on the scope for whoever asks.
        instance._led_driver, led_fallback = led_registry.create_with_fallback('auto')
        instance._motion_driver, motor_fallback = motor_registry.create_with_fallback(
            'auto', motorconfig_defaults=motorconfig_defaults
        )
        instance._bring_up_parts = {
            MOTOR: _board_status(MOTOR, instance._motion_driver, motor_fallback),
            LED: _board_status(LED, instance._led_driver, led_fallback),
        }
        instance._bring_up_substitutions = []
        instance._camera_failure = None

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
        instance._fx2_debug_wire = False
        instance.layer_identity = instance._resolve_layer_identity()

        # Build capabilities -- diagnostic instances still need this so
        # any code that reads scope.capabilities.* works.
        instance.capabilities = ScopeCapabilities.from_drivers(
            motion=instance._motion_driver,
            led=instance._led_driver,
            camera=None,
            layer_identity=instance.layer_identity,
            scope_models=instance._scope_models,
        )

        # Sub-API wiring -- diagnostic instances are first-class enough
        # that disconnect / scope.imaging / scope.illumination do not
        # raise AttributeError. ImagingAPI tolerates camera=None (per
        # its docstring); IlluminationAPI gets the connected LED driver
        # (real or NullLEDBoard).
        from modules.lumascope_api.illumination import IlluminationAPI
        from modules.lumascope_api.imaging import ImagingAPI
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from modules.lumascope_api.protocols import ProtocolsAPI
        from modules.lumascope_api.runtime_state import RuntimeState

        instance.illumination = IlluminationAPI(instance, instance._led_driver)
        instance.imaging = ImagingAPI(instance, None)
        instance.diagnostics = DiagnosticsAPI(instance)
        instance.protocols = ProtocolsAPI(instance)
        instance.runtime_state = RuntimeState(instance)

        # No hardware when neither board came up, read from the record as
        # __init__ reads it; the diagnostic probes no camera.
        instance._no_hardware = not any(status.up for status in instance._bring_up_parts.values())

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
