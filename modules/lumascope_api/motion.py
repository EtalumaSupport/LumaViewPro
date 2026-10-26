# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""MotionAPI -- sub-API for stage / focus / turret motion.

MotionAPI owns the motion state slots (_pos_cache, _axis_state,
_arrival_events, _position_listeners, _motion_wake,
_motion_monitor_stop, _motion_monitor_thread, _commands_open) and
the bodies of all stage / focus / turret methods. Lumascope keeps a
small set of one-line method-name forwarders (home, etc.)
for production callers; those retire as production migrates.

Constructor signature:
    MotionAPI(scope, driver) -- scope is the Lumascope back-ref;
    driver is the MotorBoardProtocol instance (also accessible as
    scope._motion_driver).

Within a relocated body:
    * driver calls use ``self._driver.X`` (the bound MotorBoardProtocol
      handle re-resolved through scope on every access).
    * cross-method calls to a sibling on this surface call directly.
    * cross-method calls to non-motion helpers on Lumascope route via
      ``self._scope.X``.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.1 for the canonical
method list.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import logging as _logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar, NoReturn
from collections.abc import Callable, Iterable, Iterator, Mapping

from drivers import host_clock
from drivers.exceptions import HardwareError, MotionInterlockError, MotionNotCompletedError
from drivers.null_motorboard import NullMotionBoard
from lib import profile_trace
from lvp_logger import logger
from modules.finite_number import refuse_unless_finite_number, refuse_unless_wait_s
from modules.exceptions import (
    ArgumentRefusedError,
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    HomingFailedError,
    HomingFaultError,
    HomingStoppedError,
    MissingPart,
    MotorStopFailedError,
    MoveFaultError,
    MoveNotCompletedError,
    MoveStoppedError,
    MoveWaitExpiredError,
    PositionOutOfRangeError,
)
from modules.notification_center import notifications
from modules.activity_claim import (
    ControlSessionHeldError,
    Taking,
    acting,
    activity_taking,
    command_refused,
    current_taking,
)
from modules.caller import current_caller
from modules.operation import Operation, refuse_waiting_on_ui_thread
from modules.sequential_io_executor import (
    IOTask,
    ask_now,
    asked_as_made,
    asked_before,
    slow_task_budget,
)
from modules.video_cadence import StallWatch

# Declared costs, module-level because the @slow_task_budget decorators run at
# class-body time and cannot reach a class attribute defined further down.
#
# Homing legitimately takes 10-60+ seconds depending on travel distance and
# starting position: how long the homing command may run before the
# elapsed-time warning means anything. It bounds nothing.
_HOMING_SLOW_TASK_S = 120.0

# A turret move is three physically-waited motions (Z park, turret, Z restore),
# so it runs past a threshold written for one and warned on every successful
# move. Measured 5.2s on an LS850T between adjacent positions; this is ~3x
# that, the "unusual" bound rather than the hung bound. Budget row:
# Firmware docs/PERFORMANCE_BUDGETS.md, turret move.
_TURRET_MOVE_SLOW_TASK_S = 15.0

# What a move's progress watch is started on when the monitor first sees the
# move armed: equal to no reading, so the first good read counts as progress
# and a move never read is given up at the window.
_NOT_YET_READ = object()

# The stop generation of a move asked before the last press of stop_motion:
# below every generation the counter takes, so it reads as stopped against
# any of them.
_ASKED_BEFORE_A_STOP = -1


def _home_runs_its_own_moves(body):
    """A home body's own moves are asked as the home makes them.

    A STOP the board cannot take while it homes is refused and the home
    runs to its end, its Z park and restore included, so nothing a press
    withholds is the home's own; what was asked after the home began, and
    waits behind it, is.
    """

    @functools.wraps(body)
    def run(*args, **kwargs):
        with asked_as_made():
            return body(*args, **kwargs)

    return run


# How long the Z overshoot leg may take to reach its point before the move
# fails. The leg's wait is the one loop in a move with no reply timeout of
# its own: the board answers every STATUS_R, it is the stage that may never
# arrive, so without a bound a leg that never arrived held the IO lane for
# good and the caller got the lane's bare TimeoutError at 30 s while the
# loop ran on. Sized to fit inside the motion API's 30 s dispatch bound
# beside one 5 s poll overrun, the move's other exchanges and its queue
# residence. The worst legitimate leg is so far derived, not measured:
# 6.25 s across every config in data/ on the simulator's ramp model, and
# 5.07 s for an 11.2 mm downward Z move with overshoot run there, nearly
# all of it the leg. To be confirmed on the LS850T with a full-travel
# downward Z move (the arrival plan's bench row).
OVERSHOOT_LEG_TIMEOUT_S = 15.0

# The axes a plate coordinate is defined for: the plate lies in X and Y.
_PLATE_AXES = ('X', 'Y')

# Match _lumascope.py's module-level _api_log channel so relocated
# bodies log to the same handler chain.
_api_log = _logging.getLogger('LVP.api')

from lib.stage_interlock import INTERLOCK_TITLES
from modules.axes import Axis, StageAxis
from modules.lumascope_api._constants import (
    AxisPosition,
    AxisState,
    MOTOR_POSITION_LIMIT,
    TurretStatus,
    refuse_acceleration_pct,
    refuse_unless_turret_slot,
)
from modules.api_surface import api, api_fields
from modules.closed_sets import refuse_unless_offered
from modules.frame_validity import FrameValiditySource
from modules.coord_transformations import AxisLimits, PlateLocation
from modules.live_work import WorkKind

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import MotorBoardProtocol


class HomeTarget(enum.StrEnum):
    """What a home takes: Z, the turret, or every axis the board has.

    X and Y home only with the rest.
    """

    Z = Axis.Z.value
    T = Axis.T.value
    ALL = 'ALL'


# Built from the interlocks' one list, which a driver reads too and so
# lives below modules/: the API names the driver's words by this set.
Interlock = enum.StrEnum(
    'Interlock', [(reason.upper(), reason) for reason in INTERLOCK_TITLES], module=__name__
)
Interlock.__doc__ = """A stage hardware interlock that, open, stops the stage.

    Its value is also the reason a motion command is refused or stopped for
    it, which stays a string, as every reason does.
    """


class _Move:
    """One move's outcome, fixed when the move ends.

    Created in the hold that sets its axis MOVING, and settled in the hold
    that ends it -- the monitor's IDLE, an UNKNOWN, or another move or a
    home taking the axis -- so what later happens to the axis is never
    read as this move's outcome. Settled once: the first ending wins.

    Attributes:
        axis: The axis it drives.
        stop_generation: ``MotionAPI._stop_generation`` as the move's body
            read it before driving; a different one at the move's IDLE
            means a STOP the board took ended it.
        armed: Whether the move's final target is known written, so the
            board's reached bit is this move's: set by ``_publish_drive``.
            Until then a reached bit is the previous target's, or the
            backlash leg's, and the monitor writes no verdict for the move.
        outcome: ``'arrived'``, ``'stopped'``, ``'superseded'`` or
            ``'faulted'``; None until settled.
        fault: The error an UNKNOWN ending carried, or None.
        target: The final target the move wrote to the board, set by
            ``_publish_drive``; None until written, and for good when a stop
            withheld it.
        operation: The move as its caller holds it: done by ``publish``,
            once the hold that settled the move has been released and the
            axis's listeners told.
    """

    __slots__ = ('armed', 'axis', 'fault', 'operation', 'outcome', 'stop_generation', 'target')

    def __init__(self, axis: str, stop_generation: int, stop: Callable[[], None]) -> None:
        self.axis = axis
        self.stop_generation = stop_generation
        self.armed = False
        self.outcome: str | None = None
        self.fault: MoveFaultError | None = None
        self.target: float | None = None
        self.operation = Operation(stop=stop)

    def settle(self, outcome: str, fault: MoveFaultError | None = None) -> bool:
        """Fix the outcome, unless the move is already settled; whether this call fixed it."""
        if self.outcome is not None:
            return False
        self.outcome = outcome
        self.fault = fault
        return True

    def publish(self) -> None:
        """Settle the operation with the outcome: None on arrival, else what ended the move.

        Called outside ``_axis_state_lock``: a done-callback, or a waiter
        woken here, reads the motion state through that lock.
        """
        error = self.error()
        if error is None:
            concurrent.futures.Future.set_result(self.operation, None)
        else:
            concurrent.futures.Future.set_exception(self.operation, error)

    def error(self) -> MoveNotCompletedError | None:
        """What ended the move: None on arrival or while unsettled, else its error."""
        if self.outcome in (None, 'arrived'):
            return None
        # Only a fault carries its object; the other endings are stops.
        return self.fault or MoveStoppedError(self.axis, self.outcome)


@api_fields('axis', 'at')
@dataclasses.dataclass(frozen=True)
class PositionChanged:
    """A position event: a stage axis, and its state and position as ``axis_positions`` answers."""

    axis: StageAxis
    at: AxisPosition


@api_fields('at')
@dataclasses.dataclass(frozen=True)
class TurretChanged:
    """A turret event: its state and slot as ``turret_status`` answers."""

    at: TurretStatus


class MotionAPI:
    """Motion sub-API. Hosts stateless (Phase 2b) and stateful (Phase 2c) bodies."""

    _MOTION_POLL_INTERVAL = 0.02  # the sleep between polls, not their period
    # How long an axis may sit MOVING with the motor board disconnected
    # before the monitor faults it to a terminal state. Well above a
    # transient USB blip, well below the 120s motion timeout.
    _DISCONNECT_FAULT_S = 3.0

    # Maps a motion axis to its frame-validity source. X and Y share
    # 'xy_move'; Z and the turret each have their own source so the
    # settle-check gates on the correct axis reaching IDLE. A turret
    # move that recorded 'xy_move' would clear the moment X/Y read idle,
    # before the turret physically finished.
    _AXIS_VALIDITY_SOURCE: ClassVar[dict[str, FrameValiditySource]] = {
        'Z': FrameValiditySource.Z_MOVE,
        'T': FrameValiditySource.TURRET,
    }

    def __init__(self, scope: Lumascope, driver: MotorBoardProtocol) -> None:
        # `driver` is in the signature for backcompat (Phase 1 Lumascope
        # passes it explicitly). It is intentionally unused here: `_driver`
        # is a dynamic property that re-resolves through `_scope` on every
        # access. Lumascope reassigns `_motion_driver` during connect() /
        # disconnect() (e.g. swaps to NullMotionBoard on disconnect);
        # capturing the init-time handle would leave this surface talking
        # to a stale driver after every reconnect.
        self._scope = scope

        # ------------------------------------------------------------------
        # Motion state slots.
        #
        # Locks and events are initialized here; per-axis dicts are
        # populated by _init_axes() called from Lumascope.__init__ after the
        # motion driver is constructed and present_axes is known.
        # ------------------------------------------------------------------
        self._pos_cache_lock = threading.Lock()

        # TimedLock on the hot axis-state lock records contention to
        # lock_trace.csv when profile_trace_enabled is set in settings.json.
        # Never hold it across a serial call (30-200 ms on the motor bus), a
        # log line, a listener or a move's publish: every motion read waits
        # behind it. Its bodies are read for that at build time
        # (tests/guards/test_the_axis_state_lock_holds_only_state_work.py);
        # a wall-clock hold time measures the scheduler, not the body.
        self._axis_state_lock = profile_trace.TimedLock(
            threading.Lock(), name='motion._axis_state_lock'
        )

        # Motion monitor wakeup -- set when any axis starts MOVING, cleared
        # when all axes are back to IDLE. The monitor thread sleeps on this.
        self._motion_wake = threading.Event()

        # Position change listeners -- push-based UI update mechanism.
        # A position listener is called with (axis, AxisPosition) whenever a
        # stage axis's position cache or state changes; a turret listener with
        # the TurretStatus whenever the turret's state or slot does. Both
        # fire from the IO executor thread, so they MUST schedule UI work
        # via Clock.schedule_once.
        self._position_listeners_lock = threading.Lock()
        self._position_listeners: list = []
        self._turret_listeners: list = []
        self._stop_press_listeners_lock = threading.Lock()
        self._stop_press_listeners: list = []

        # The several-move commands open now (``one_command``): while any is,
        # no motion change settles, so a capture cannot take a frame from
        # between a command's moves. A count, since a turret change opens
        # its own inside going to a step's.
        self._commands_lock = threading.Lock()
        self._commands_open = 0
        # The latest still-moving read an arrival handed over while a command
        # was open: the earliest its last move can have stopped, which its
        # close gives the changes it stamps. None when no arrival carried one.
        self._command_moving_seen_ns: int | None = None

        # Motion monitor thread handle -- populated by _start_monitor().
        # Not started at __init__ because the motion driver and per-axis
        # dicts aren't ready yet; Lumascope.__init__ calls _start_monitor()
        # after _init_axes().
        self._motion_monitor_stop = threading.Event()
        self._motion_monitor_thread: threading.Thread | None = None
        # Per-axis monotonic timestamp first seen disconnected-while-moving;
        # used by the monitor to bound how long an axis stays MOVING after
        # the board vanishes. Only the monitor thread touches it.
        self._disconnect_since: dict[str, float] = {}
        # Per-axis (move, its progress watch): the watch is fed the axis's
        # microstep reading on every good read and fires once the reading
        # has held one value for the stall window, the clock starting when
        # the monitor first sees the move armed, so an axis never read fires
        # too. Bounds a connected-but-stalled axis the same way
        # _disconnect_since bounds a vanished board. Kept with its move, so a
        # retarget never inherits the earlier move's clock.
        # Without it, a move whose position_reached never fires wedges
        # every state-reader -- capture settle-checks, is_moving pollers --
        # forever, while only explicit waiters carry their own timeout.
        # Only the monitor thread touches it.
        self._moving_since: dict[str, tuple[_Move, StallWatch]] = {}
        # When the last arrival read that found each moving axis not yet
        # arrived began: the earliest the axis can have stopped, handed to
        # frame validity with the arrival. Only the monitor thread touches it.
        self._moving_seen_ns: dict[str, int] = {}
        # Set when the monitor's loop has ended other than at the scope's
        # close: nothing would judge a move, so every later drive is refused
        # until the monitor is started again.
        self._monitor_ended = False
        # The move whose failed position read the monitor has already
        # warned of: once per move, not once per poll, which is about fifty
        # a second. Only the monitor thread touches it.
        self._unread_warned: dict[str, _Move] = {}
        # The same, for a failed arrival read.
        self._status_unread_warned: dict[str, _Move] = {}
        # The axis's current move: the record its MOVING write created,
        # kept after the move ends so a wait on the axis can read the fault
        # it ended with; None after a home starts, so a later wait never
        # raises an earlier move's fault. A verdict for one move -- the
        # monitor's IDLE or stall, the waiter's timeout -- is written only
        # while that move is still the axis's MOVING one: an IDLE or a fault
        # meant for a move a later one replaced must not end the later one.
        # Under _axis_state_lock.
        self._current_move: dict[str, _Move | None] = {}

        # Per-axis state dicts -- empty until _init_axes() fills them.
        self._pos_cache: dict = {}
        self._axis_state: dict = {}
        self._arrival_events: dict = {}

        # The turret slot in the light path: the slot last commanded by a
        # turret command (the turret move, either home) that returned
        # without error and with no stop issued while it ran. The turret
        # has no encoder, so this is the only truth there is about it; the
        # controller's step counter reports steps issued, not glass in the
        # path, and is never read as a slot. None -- unknown -- from the
        # moment a turret command starts until it succeeds, after any
        # failure, and whenever T goes UNKNOWN (``_set_axis_state``).
        # Written through ``_record_turret_slot``, which tells the turret
        # listeners, but for the UNKNOWN write, whose state change tells them.
        self._last_turret_position: int | None = None

        # Bumped by every stop_motion. A STOP sets target = actual on every
        # axis, so a move in flight then reports "reached" at a place
        # nobody commanded; a waited move compares this against the value
        # it saw before driving to tell a stop from an arrival.
        self._stop_generation = 0
        # Bumped by the Stops someone asked for (stop_motion), not by the
        # scope's own halt (_stop: the teardown, a stall) or a stop the stage
        # made in refusing for an interlock:
        # a home that has not driven yet is ended by an asked Stop, while
        # the lid's stop halts what was moving and the home's own interlock
        # read refuses it with the lid as the cause. Written and read under
        # _stop_lock.
        self._stops_asked = 0
        # Where the last press of stop_motion stands in the executors' one
        # ask order, refused or not: work asked before it drives nothing,
        # because the person pressed STOP after asking for it. Written and
        # read under _stop_lock.
        self._last_press_ask = 0
        self._stop_lock = threading.Lock()
        # How many homes drive now on a board that cannot take a STOP until
        # the home ends: stop_motion is refused while any does, not queued
        # behind it. A count, as a scope built without a Session can run two
        # homes at once. Written and read under _stop_lock.
        self._unstoppable_homes_driving = 0

    def _init_axes(self, present_axes: list[str], homed_axes: list[str]) -> None:
        """Populate per-axis state dicts from the detected axes.

        Called from Lumascope.__init__ (and create_diagnostic) after the
        motion driver's detect_present_axes() has run. NullMotionBoard
        returns [] so a system with no motor hardware ends up with empty
        dicts -- all state-touching methods handle that via no-ops.

        An axis the hardware reports homed starts IDLE, not UNKNOWN.
        Seeding every axis UNKNOWN meant a fresh process was blind to a
        scope that was already homed and still powered, so the pre-drive
        gate refused it -- correct for a scope that has never been homed,
        wrong for one whose reference frame is live. The GUI never saw
        this because it homes at startup; every headless caller saw it
        always, and the only fixes available to them were to re-home
        (destroying the position they attached to measure) or to bypass
        the gate. Neither is acceptable, so the state store is told the
        truth instead.

        Homed is not enough on its own: IDLE says the cache holds where the
        axis is, so each axis's position is read here, and only a homed
        axis whose read succeeded starts IDLE. One whose read failed starts
        UNKNOWN, holding the 0.0 seed, which nothing read. The read is here
        rather than a later call so that no constructor can skip it and
        leave a homed axis IDLE on the seed.

        homed_axes is required rather than defaulted: a caller that
        forgets it would silently reintroduce the blind seeding, and
        that failure is invisible until a headless run is refused.

        Args:
            present_axes: List of axis names the hardware actually has.
            homed_axes: Those axes the hardware reports as already homed.
        """
        # Keyed by the members, so every answer read from these is too.
        present_axes = [Axis(ax) for ax in present_axes]
        self._pos_cache = dict.fromkeys(present_axes, 0.0)
        self._axis_state = dict.fromkeys(present_axes, AxisState.UNKNOWN)
        self._arrival_events = {ax: threading.Event() for ax in present_axes}
        for ev in self._arrival_events.values():
            ev.set()  # Start as "arrived" (not moving)
        self._current_move = dict.fromkeys(present_axes)
        read = self._refresh_position_cache()
        with self._axis_state_lock:
            for ax in present_axes:
                if ax in homed_axes and ax in read:
                    self._axis_state[ax] = AxisState.IDLE

    def _start_monitor(self) -> None:
        """Spawn the motion monitor thread.

        Called from Lumascope.__init__ after _init_axes() so the thread
        always sees fully populated state dicts. Separate from __init__
        so create_diagnostic can control the spawn sequence.
        """
        self._motion_monitor_stop.clear()
        self._monitor_ended = False
        self._motion_monitor_thread = threading.Thread(
            target=self._motion_monitor_loop,
            name='motion-monitor',
            daemon=True,
        )
        self._motion_monitor_thread.start()

    def _disconnect(self) -> None:
        """Stop the motion monitor and reset axis states.

        Called from Lumascope.disconnect() after its STOP and before the
        motor driver is swapped to NullMotionBoard. Until the monitor stops
        here it reads the board, so a move that arrived or that the STOP
        halted ends as such; one still moving now ends ``'stopped'`` at the
        UNKNOWN, as the STOP ended it. Sets all arrival events so any
        blocked waiters unblock cleanly.
        """
        self._motion_monitor_stop.set()
        self._motion_wake.set()  # unblock if sleeping
        if self._motion_monitor_thread is not None and self._motion_monitor_thread.is_alive():
            self._motion_monitor_thread.join(timeout=1.0)

        # Through the one writer of the state: it sets each arrival event,
        # so any blocked waiter unblocks, and clears the turret slot with
        # an UNKNOWN T.
        for ax in list(self._axis_state):
            self._set_axis_state(ax, AxisState.UNKNOWN)

    @property
    def _driver(self) -> MotorBoardProtocol:
        return self._scope._motion_driver

    # ------------------------------------------------------------------
    # Position-knowledge authority.
    #
    # _axis_state is the ONE store for "is this axis position known."
    # Three producers write UNKNOWN into it -- a failed home, the
    # disconnect fault, the stall fault -- and everything that asks the
    # question reads it here, so a new producer is automatically honored
    # by every consumer.
    # ------------------------------------------------------------------

    @staticmethod
    def _position_known(state: str) -> bool:
        """Whether an axis in ``state`` has a valid reference position.

        IDLE and MOVING both have one: a move in flight knows what frame
        it is moving in, it just does not know the instantaneous
        position. HOMING does not have one yet -- the reference is being
        established -- and UNKNOWN has lost it.
        """
        return state in (AxisState.IDLE, AxisState.MOVING)

    @api(reads=True)
    def position_is_known(self, axis: Axis) -> bool:
        """Whether *axis* has a reference position an absolute move can use.

        Offered to callers as a question rather than only as an exception.
        A caller whose move is optional -- one that should be skipped
        rather than attempted on an axis whose position was never
        established -- could otherwise only discover the answer by
        provoking the refusal and catching it, which is indistinguishable
        from swallowing a real one.

        Stricter than ``_pre_drive`` by one state: a HOMING axis answers
        False here, because its reference is still being established,
        while the gate lets it drive so the home can finish.

        Args:
            axis: The axis to ask about.

        Returns:
            bool: True when *axis* is IDLE or MOVING; False when it is
            UNKNOWN or HOMING, or an axis this scope does not have.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, *axis* is no axis name.
        """
        with self._axis_state_lock:
            state = self._axis_state.get(axis)
        return self._position_known(state)

    def _fail_drive(self, axis: str, move: _Move, cause: Exception) -> NoReturn:
        """A commanded move failed at the driver: the axis is UNKNOWN, and raise.

        The axis is MOVING and disarmed from the drive's start, so no
        verdict can end it, and left there it would read as moving forever;
        a raise from the driver lands here, which makes it terminal.

        Raises:
            MoveFaultError: the fault the move ended with --
                ``'driver_failed'``, chained from the driver's error, unless
                the monitor gave the axis up first (a lost board), whose
                object it then is. The one object its caller reports.
        """
        failed = MoveFaultError(axis, 'driver_failed')
        failed.__cause__ = cause
        self._set_axis_state(axis, AxisState.UNKNOWN, fault=failed)
        raise move.fault or failed

    def _refuse_for_interlock(
        self,
        axis: str,
        move: _Move,
        state_before: str,
        cause: MotionInterlockError,
        member: str,
    ) -> NoReturn:
        """The stage's interlock refused a move's drive: raise the API's refusal.

        Refused before anything moved, no target of the move reached the
        board, so the axis gets back the state it had (``_give_back_states``),
        its position known. A stop the driver made in refusing is a Stop
        (``_note_interlock_stop``), moved first, so the refused move settles
        ``'stopped'`` and a move waiting on another axis the stage halted
        raises ``'stopped'`` instead of reading the halt as arrival. A
        refusal after something moved is not a refusal, and fails the
        drive as any driver error does.

        Raises:
            HardwareCommandRefusedError: the interlock's reason, chained
                from the driver's error.
            MoveFaultError: ``'driver_failed'``, when the driver says
                the refused command had moved.
        """
        self._note_interlock_stop(cause)
        if cause.moved:
            self._fail_drive(axis, move, cause)
        self._give_back_states({axis: state_before})
        _api_log.info(f'{member} {axis} REFUSED: {cause.reason}')
        raise HardwareCommandRefusedError(cause.reason, member) from cause

    def _give_back_states(self, states: Mapping[str, str]) -> None:
        """Put back the states a command the interlock refused found its axes in.

        Nothing of the refused command moved, so each axis is where it was.
        An axis that was moving had its move settled by the command's own
        HOMING or MOVING write, and a MOVING axis carries a move of its own,
        so it gets a new one, armed with no target, as a drive a stop
        withheld is: the monitor sets it IDLE where it comes to rest. Any
        other state is written back as it was, and a move it ends is
        settled against the current stop generation, so the refused move
        reads ``'stopped'`` when the stage stopped in refusing it.
        """
        for axis, state in states.items():
            if state == AxisState.MOVING:
                self._begin_move(axis, self._stop_generation)
                self._publish_drive(axis, False, None)
            else:
                self._set_axis_state(axis, state, stop_generation=self._stop_generation)

    def _note_interlock_stop(self, cause: MotionInterlockError) -> None:
        """A stop the stage made in refusing is a Stop: the generation moves
        as ``_send_stop`` moves it, so every move waiting on an axis the
        stage halted raises ``'stopped'`` instead of reading the halt as
        arrival.

        The bump lands once the driver has released its lock, not under
        ``_stop_lock`` across the exchange as ``_send_stop``'s does, so a
        waiter the monitor wakes in that gap reads the old generation, and
        a move that arrived just before the refusal can read the new one:
        the race ``stop_motion`` already has, not widened in kind.
        """
        if cause.stopped:
            with self._stop_lock:
                self._stop_generation += 1

    def _axis_states(self, axes: Iterable[str]) -> dict[str, str]:
        """The states ``axes`` hold now, read together."""
        with self._axis_state_lock:
            return {axis: self._axis_state.get(axis) for axis in axes}

    @contextlib.contextmanager
    def _driving_home(
        self,
        home: str,
        stops_asked: int,
        states_before: Mapping[str, str],
        turret_before: int | None,
    ) -> Iterator[None]:
        """The span in which a home drives, entered immediately before it does.

        A home whose Stop landed before it drives ends here: ``stops_asked``
        is ``_stops_asked`` as the home read it when it was asked. The driver
        forgets a stop that came before its home began, so a Stop pressed
        just after Home would otherwise be lost and the whole home run. A
        stop the stage made for an interlock is not one: it halted what was
        moving, and the home's own interlock read refuses it with its cause.
        Nothing of the home has moved the axes it homes, so each gets back
        the state it had and the turret its slot, as a home the interlock
        refuses before it moves.

        On a board that cannot take a STOP mid-home, ``stop_motion`` is
        refused for the span: a STOP sent would wait out the home and then
        stop nothing. The stop check and the refusal's start share
        ``_stop_lock``, so no Stop falls between them.

        Raises:
            HomingStoppedError: ``'stopped'``, naming no axis left unknown.
        """
        unstoppable = int(not self._driver.supports_stopping_a_home())
        with self._stop_lock:
            stopped = self._stops_asked != stops_asked
            if not stopped:
                self._unstoppable_homes_driving += unstoppable
        if stopped:
            self._give_back_states(states_before)
            self._record_turret_slot(turret_before)
            raise HomingStoppedError(home, 'stopped', ())
        try:
            yield
        finally:
            with self._stop_lock:
                self._unstoppable_homes_driving -= unstoppable

    def _home_ending(self, home: str, stops_asked: int, axes: Iterable[str]) -> HomingFailedError:
        """The outcome of a home that raised: a stop when a person's Stop landed during it, else the fault.

        ``stops_asked`` is ``_stops_asked`` as the home read it when it was
        asked. A halt the scope sent itself -- a stall's, an interlock's --
        moves the stop generation but is no one's Stop, so a home it ended
        ends in its fault. The caller chains it from what was raised.
        """
        with self._stop_lock:
            stopped = self._stops_asked != stops_asked
        if stopped:
            return HomingStoppedError(home, 'stopped', axes)
        return HomingFaultError(home, 'error', axes)

    @api(reads=True)
    def interlocks(self) -> frozenset[Interlock]:
        """The stage's hardware interlocks open now (``'lid_open'``, ``'stage_unpowered'``).

        Empty on a board with none. Asked of the driver directly, not
        queued on the IO lane: the driver reads it under its own lock,
        one exchange serialized with any in flight, so the answer never
        waits for a move or a home to finish, as the motion monitor's own
        reads do not. It does wait for what holds that lock now: one
        exchange or one poll of a home, and at most a Stop's settle (2 s)
        or a reopen of the port.

        Returns:
            frozenset[Interlock]: The open interlocks.

        Raises:
            HardwareError: the board did not answer the read; the stage's
                state is unknown, which is not the same answer as none open.
        """
        # The driver keeps its own words; the API names them by the set.
        return frozenset(Interlock(name) for name in self._driver.interlocks())

    def refuse_absent(self, member: str, axis: str | None = None) -> None:
        """Refuse a command for motion hardware this scope does not have.

        The one presence question every motion command asks, after its axis
        name is checked and before its value is, so a scope without the part
        says so before judging the value. No motor controller is
        ``motor_connected`` False -- none installed, or its cable pulled:
        the controller of a model that has one is not connected, and a
        manual model has no motors. With a controller, an axis outside
        ``capabilities.axes`` is not on this scope. An axis-less command (a
        stop, the acceleration limit, a full home) asks the controller half
        only.

        A consult seam, not part of the L2 API surface: a member above this
        API that acts on an axis (a fan duty, a bookmark save) asks it, so
        it refuses a missing part in the words a motion command uses.

        Raises:
            HardwareCommandRefusedError: ``'scope_disconnected'`` after
                ``disconnect()``; ``'not_connected'`` or ``'axis_absent'``,
                naming the missing part. Nothing was sent.
        """
        self.refuse_controller_not_connected(member)
        if not self._scope.motor_connected:
            part = MissingPart.MOTORS
        elif axis is not None and axis not in self._scope.capabilities.axes:
            part = MissingPart.axis(axis)
        else:
            return
        raise HardwareCommandRefusedError(part.reason, member, missing=part)

    def refuse_controller_not_connected(self, member: str) -> None:
        """Refuse when this scope's model has a motor controller and none is connected.

        The controller half of the presence question every motion command
        asks, offered alone to a caller that works on a scope with no motors
        and must still be refused when a scope's motors are out of reach: a
        manual scope passes, a motorized one whose controller did not come
        up, or whose cable was pulled, does not.

        A consult seam, not part of the L2 API surface: an L2 caller's motion
        command asks it itself.

        After ``disconnect()`` the scope's lanes are shut and it answers as
        every command does then, ``scope_disconnected``, on every model: a
        caller asking off the lane (Go To Step, the plate position read)
        hears what a command on the lane would.

        Raises:
            HardwareCommandRefusedError: ``'scope_disconnected'`` after
                ``disconnect()``; ``'not_connected'``, naming the motor
                controller. Nothing was sent.
        """
        if self._scope._io_executor.pending_shutdown:
            raise HardwareCommandRefusedError('scope_disconnected', member)
        if self._scope.motion_expected and not self._scope.motor_connected:
            part = MissingPart.MOTOR_CONTROLLER
            raise HardwareCommandRefusedError(part.reason, member, missing=part)

    def _has_position(self, axis: str) -> bool:
        """Whether a position read of ``axis`` has hardware behind it.

        Not with the null board installed -- a manual model, a board that
        never came up, or after ``disconnect()`` -- and not for an axis
        the scope does not have. A board whose cable was pulled is still
        installed: its axes keep the last number they reported.
        """
        return (
            not isinstance(self._driver, NullMotionBoard) and axis in self._scope.capabilities.axes
        )

    def _pre_drive(self, axis: str, force: bool = False) -> None:
        """Refuse to drive an axis whose position is not known.

        Every commanded move passes through here before it reaches the
        driver, so the refusal cannot be reintroduced by a new move
        caller that forgets to check -- there is no per-call-site check
        to forget.

        Only UNKNOWN refuses. HOMING is not a refusal: the home itself
        has to be able to drive the axis it is establishing.

        This raises without notifying. The failure that made the axis
        UNKNOWN already notified when it happened (a failed home pops
        its own error), and the caller that provoked THIS refusal owns
        the response -- a REST client needs a status code, the GUI
        surfaces the executor's task failure. Notifying here would
        double the popup on the startup path.

        Args:
            axis: Axis the caller is about to drive.
            force: Skip the check. The recovery paths pass True: the
                turret-safety Z-retract and a deliberate re-home jog
                must move an axis that is legitimately still unknown,
                and a gate that blocked them would deadlock the only
                operations that can clear the state it guards.

        Raises:
            AxisStateUnknownError: The axis position is unknown and the
                caller did not pass ``force=True``.
        """
        if force:
            return
        with self._axis_state_lock:
            state = self._axis_state.get(axis)
        if self._drive_refused(state):
            raise AxisStateUnknownError({axis: state})

    @staticmethod
    def _drive_refused(state: str | None) -> bool:
        """Whether the pre-drive gate refuses to drive an axis in ``state``.

        Only UNKNOWN: a HOMING axis must drive for its home to finish, and a
        move asked for while it homes waits behind the home on the motion
        lane. One rule, read by the gate and by ``refuse_unknown_positions``,
        so an answer given before a move is submitted cannot disagree with
        the gate that later drives it.
        """
        return state == AxisState.UNKNOWN

    def refuse_unknown_positions(self, axes: Iterable[str], *, recording: bool, then: str) -> None:
        """Refuse, once, a gesture that needs axes whose position is not known.

        A person's gesture often touches several axes -- going to a step
        moves X, Y, Z and the turret; saving a bookmark records a position.
        Refused axis by axis on the motion lane, one condition becomes one
        refusal per axis, each arriving after the gesture has moved on; and
        a recorded position is simply the last number the axis reported,
        real-looking and no longer true. This asks once, before anything is
        submitted or written, names every axis in one sentence, and tells
        the user once.

        Two questions, because moving and recording differ over an axis
        that is still homing: its move queues behind the home and lands,
        so moving refuses only what the pre-drive gate refuses; its
        position is not yet one to save, so recording refuses it too.

        A consult seam for the GUI's gestures, not part of the L2 API
        surface: an L2 caller's move meets the pre-drive gate, and the
        positions a script saves go through API members that ask for
        themselves.

        Args:
            axes: The axes the gesture needs. An axis this scope does not
                have is not asked about.
            recording: True when the gesture saves the position, False
                when it moves.
            then: What the user does once the scope knows its position,
                ending the refusal (e.g. ``'move it'``, ``'save the
                bookmark'``).

        Raises:
            ArgumentRefusedError: ``'not_offered'``, a name in *axes* is
                no axis name.
            AxisStateUnknownError: Naming every refused axis. Raised, not
                reported: the gesture's boundary reports it.
        """
        wanted = {refuse_unless_offered(Axis, axis, 'axes') for axis in axes}
        with self._axis_state_lock:
            states = {axis: s for axis, s in self._axis_state.items() if axis in wanted}
        refused = {
            axis: s
            for axis, s in states.items()
            if (not self._position_known(s) if recording else self._drive_refused(s))
        }
        if not refused:
            return
        raise AxisStateUnknownError(refused, then=then)

    # ------------------------------------------------------------------
    # Stateless method bodies.
    #
    # Order mirrors _lumascope.py source order.
    # ------------------------------------------------------------------

    @api
    def stop_motion(self) -> None:
        """Stop all in-flight motor moves.

        Idempotent. Uses the firmware-side ``STOP`` command, which the motor
        controller implements as ``motorstop`` (target=actual on all axes).
        It does not wait behind the lane, so a stop reaches the board while
        a move holds the lane.

        A home is stopped only on a board that can take a STOP while it
        homes (the LS720's); on any other, a Stop while the home drives is
        refused and the home runs to its end.

        A press -- taken, failed or refused -- withholds every move asked
        before it that has not yet driven, from any client and any queue,
        and ends the live run (its press listeners are told first).

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'`` with no motor controller (see
                ``refuse_absent``); ``'home_not_stoppable'``
                while a home drives on a board that cannot stop one.
                Nothing was sent.
            MotorStopFailedError: the board did not take the STOP, so the
                stage may still be moving. Chained from the driver's
                error. The stop generation has moved regardless.
        """
        logger.info(f'[SCOPE API ] stop_motion pressed by {current_caller().words}')
        self._stop_motion_while(lambda: True, pressed=True)

    def _stop_motion_while(self, live: Callable[[], bool], *, pressed: bool = False) -> None:
        """``stop_motion``, for one operation's work: sent only while ``live()`` answers True.

        ``live()`` is asked and the STOP sent in one hold of ``_stop_lock``,
        and the work's end is written where that hold excludes it -- a
        home's body says it has ended under the lock, and the monitor reads
        the generation under it before it judges an arrival -- so a stop
        asked once the work has ended sends nothing. One asked while it is
        live is a Stop like any other: a move that supersedes the work
        while the STOP is being sent has its drive withheld, as every move
        started while a Stop lands does.
        """
        self.refuse_absent('stop_motion')
        if pressed:
            # A press of STOP is told before the STOP is sent, so no work a
            # listener ends starts during the exchange; outside _stop_lock,
            # which a listener's own stop of a run must not wait behind.
            with self._stop_press_listeners_lock:
                listeners = list(self._stop_press_listeners)
            notifications.tell_listeners(listeners, category='Motion')
        with self._stop_lock:
            if not live():
                return
            # Before the send and before a refusal: a press the board cannot
            # take while it homes still withholds what was asked before it,
            # so the home runs to its end and nothing after it moves.
            self._last_press_ask = ask_now()
            if self._unstoppable_homes_driving:
                raise HardwareCommandRefusedError('home_not_stoppable', 'stop_motion')
            # A person's Stop, which a home it ends ends as: counted before
            # the send, so a send that fails is counted too.
            self._stops_asked += 1
            self._send_stop()

    def _generation_asked(self) -> int:
        """The stop generation a move body drives under: ``_ASKED_BEFORE_A_STOP``
        when the work it runs in was asked before the last press of
        ``stop_motion``, else ``_stop_generation``.

        Every reader of a move's generation compares it with the current
        one, so a move asked before the press has its target write withheld
        and ends ``'stopped'`` wherever it is read -- a jog queued behind the
        lane when STOP was pressed drove after it, reporting that it arrived.
        """
        with self._stop_lock:
            if asked_before(self._last_press_ask):
                return _ASKED_BEFORE_A_STOP
            return self._stop_generation

    def _stop(self) -> None:
        """Send the STOP, refusing nothing: the scope's own halt, for its teardown (which asks presence first) and a stall."""
        # The generation moves inside this lock, after the board answered,
        # and a waited move reads it under the same lock: a move the STOP
        # ended cannot read the generation before the bump.
        with self._stop_lock:
            self._send_stop()

    def _send_stop(self) -> None:
        try:
            # The driver halts every axis on every firmware: by STOP where
            # the firmware has it, else axis by axis (field firmware,
            # 2024-09-10 EL-0940-02, has no STOP command).
            self._driver.motor_stop()
            self._stop_generation += 1
            logger.info('[SCOPE API ] stop_motion: motors stopped')
        except Exception as e:
            # The exchange may have failed after the board took the STOP,
            # so a move in flight cannot be vouched for as arrived.
            self._stop_generation += 1
            raise MotorStopFailedError() from e

    @api(reads=True)
    def is_current_slot_objective_set(self) -> bool:
        """Whether the turret slot in the light path has an objective assigned.

        Returns:
            bool: True if the turret's current slot is known and has a
                configured objective ID; False if the slot is unconfigured
                or not known -- an unknown slot has no objective anyone can
                name.

        Raises:
            ConfigError: ``'scope_not_composed'``, no session has bound this
                scope, so it has no slot assignments to read.
        """
        slot = self.get_turret_slot()
        if slot is None:
            return False
        return self._scope.runtime_state.get_turret_config()[slot] is not None

    @contextlib.contextmanager
    def _reference_position_logger(self) -> Iterator[None]:
        """Context manager that logs limit-switch status before and after homing.

        Use as ``with scope.motion._reference_position_logger(): ... home ...``.
        Emits forced-INFO log lines so the limit-switch state pre/post
        homing is preserved for diagnostics. A switch the board did not
        answer is written into the line as unread, with the board's error,
        so a log line never ends the home it describes; the home's own
        reads report a board that has stopped answering.
        """

        def read_axis(axis: str) -> tuple[int, int] | str:
            try:
                return self._driver.limit_switch_status(axis=axis)
            except HardwareError as e:
                return f'unread ({e})'

        def read() -> dict[str, tuple[int, int] | str]:
            # Keyed by each axis's plain name: the dict is written into a log
            # line, where a member's repr would stand in for the name.
            return {axis.value: read_axis(axis) for axis in self._scope.capabilities.axes}

        before = read()
        logger.info(f'Limit switch status before homing: {before}', extra={'force_error': True})
        yield
        after = read()
        logger.info(f'Limit switch status after homing: {after}', extra={'force_error': True})

    @slow_task_budget(_HOMING_SLOW_TASK_S)
    @_home_runs_its_own_moves
    def _home_impl(self) -> None:
        """Home every axis the motor board has.

        This is the unified "home everything" entry point used by
        startup and the GUI Home button. The firmware's home routine
        homes Z, then T, then X/Y -- on a Z-only board (LS820) it homes
        Z and reports the missing X/Y; on a full XYZ scope it homes
        all three. The driver returns True for both cases (full and
        partial), raises HardwareError on real failure.

        Returns only when every homed axis has a known position.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller (see
                ``refuse_absent``); or an interlock reason, the stage's
                interlock refused the home before anything moved. Nothing
                was driven, and every axis keeps the state it had.
            HomingFaultError: the driver raised, or a homed axis's position
                could not be read; ``'lid_open'``, the lid was opened while
                the home moved.
            HomingStoppedError: ``'stopped'``, a Stop ended it.
        """
        # Asked before the driver: without it, a home with no controller
        # dispatches into the driver where exchange_command tries to
        # auto-reconnect and burns its full timeout (~10 s), and the
        # person sees a hang, then a "Homing Failed" that implies a
        # homing-mechanics problem instead of the cable.
        self.refuse_absent('home')
        present_axes = self._scope.capabilities.axes
        _api_log.info('home START')
        # What a home the interlock refuses before anything moves gives back.
        states_before = self._axis_states(present_axes)
        turret_before = self._last_turret_position
        for ax in present_axes:
            self._set_axis_state(ax, AxisState.HOMING)
        # A homing turret is in no known slot until the home succeeds.
        self._record_turret_slot(None)
        stop_generation = self._stop_generation
        stops_asked = self._stops_asked
        if 'Z' in present_axes:
            self._scope.frame_validity.invalidate(FrameValiditySource.Z_MOVE, began_ns=None)
        if 'X' in present_axes or 'Y' in present_axes:
            self._scope.frame_validity.invalidate(FrameValiditySource.XY_MOVE, began_ns=None)
        if 'T' in present_axes:
            self._scope.frame_validity.invalidate(FrameValiditySource.TURRET, began_ns=None)
        # One command, so the axes' changes come into force when it closes,
        # after the turret's slot is recorded: an axis saying IDLE first
        # would put the turret's change in force with no slot in the light
        # path, and a frame made then has no objective.
        with self._command():
            try:
                with (
                    self._reference_position_logger(),
                    self._driving_home('ALL', stops_asked, states_before, turret_before),
                ):
                    self._driver.home()
                # The position is read BEFORE an axis says IDLE: a reader that
                # samples at frame rate would otherwise pair "known" with the
                # pre-home number for the length of the serial round-trips.
                read = self._refresh_position_cache()
                for ax in present_axes:
                    if ax in read:
                        self._set_axis_state(ax, AxisState.IDLE)
                self._raise_unread_axes('ALL', present_axes, read)
                # The firmware homes the turret to slot 1. Recording it also lets
                # a following move_turret(1) -- e.g. the startup select-slot-1 --
                # recognise the turret is already there instead of running a
                # redundant Z-retract / rotate / restore. Not after a stop: a
                # home the stop cut short did not reach slot 1.
                if 'T' in present_axes and not self._stopped_since(stop_generation):
                    self._record_turret_slot(1)
            except HomingFailedError:
                raise
            except MotionInterlockError as e:
                self._note_interlock_stop(e)
                if not e.moved:
                    self._give_back_states(states_before)
                    self._record_turret_slot(turret_before)
                    raise HardwareCommandRefusedError(e.reason, 'home') from e
                for ax in present_axes:
                    self._set_axis_state(ax, AxisState.UNKNOWN)
                raise HomingFaultError('ALL', e.reason, present_axes) from e
            except Exception as e:
                for ax in present_axes:
                    self._set_axis_state(ax, AxisState.UNKNOWN)
                raise self._home_ending('ALL', stops_asked, present_axes) from e
            finally:
                _api_log.info('home DONE')

    @contextlib.contextmanager
    def _safe_turret_move(self, restore_z: bool = True) -> Iterator[int]:
        """Context manager that lowers Z to 0 before turret motion and restores after.

        Use as ``with scope.motion._safe_turret_move() as stop_generation:
        ... move turret ...``; it yields the stop generation read before Z
        was parked, the one a caller judges the whole change by. Holds the
        change open as one command from the park to the restore's arrival
        (``_command``), so no capture returns a frame with Z parked or at
        the old slot, and restores the original Z position even if the body
        raises -- unless a stop landed since Z was parked: the person
        stopped the scope, so Z stays parked.

        Args:
            restore_z: When True (default), restore the original Z
                position on exit. Set to False when the immediate next
                operation will overwrite Z anyway (e.g. protocol
                step-navigation moves T then immediately moves Z to the
                step's target -- the restore is wasted motion). When
                False, Z is left at 0 and the caller is responsible for
                the next Z move. Standalone callers (UI turret button,
                the turret-home body) leave the default True.
        """
        # Save off the Z target before moving Z to 0: the restore returns Z
        # to the number the person commanded, not to the poll of it.
        logger.info('[SCOPE API ] Moving Z to 0', extra={'force_error': True})
        initial_z = self.get_target_position(axis='Z')
        stop_generation = self._generation_asked()
        # The park inside the command too: a park that raises or is stopped
        # still closes it, and a capture cannot land between the park and
        # the turret's move.
        with self._command():
            # force: this retract is the turret-safety move, and it is also
            # the first motion of the turret-home recovery -- the case where
            # Z is legitimately still unknown because the home that would
            # have established it is the operation being recovered. Refusing
            # here would mean an UNKNOWN Z could never be re-homed without
            # restarting the application.
            self._move_absolute_impl('Z', position=0, force=True).result()
            try:
                yield stop_generation
            finally:
                # The restore is a new move, which reads the generation after
                # the stop and so is not withheld by it; this is what keeps a
                # stopped change from driving Z back up.
                if self._stopped_since(stop_generation):
                    logger.info(
                        '[SCOPE API ] Leaving Z parked -- a stop landed during the turret change',
                        extra={'force_error': True},
                    )
                elif restore_z:
                    logger.info(
                        f'[SCOPE API ] Restoring Z to {initial_z}', extra={'force_error': True}
                    )
                    # force for the same reason as the retract: this is the
                    # other half of one recovery, and refusing it would park
                    # the stage at Z=0 with no way back.
                    self._move_absolute_impl('Z', position=initial_z, force=True).result()
                else:
                    logger.info(
                        '[SCOPE API ] Skipping Z restore -- caller will overwrite Z next',
                        extra={'force_error': True},
                    )

    @slow_task_budget(_HOMING_SLOW_TASK_S)
    @_home_runs_its_own_moves
    def _home_turret_impl(self) -> None:
        """Home the turret axis. Moves Z to 0 during turret motion for safety.

        Returns when the turret is homed.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no turret (see
                ``refuse_absent``); refused before Z is parked, and nothing
                was driven or recorded.
            HomingFaultError: the home or its Z park raised, or the
                turret's position could not be read.
            HomingStoppedError: ``'stopped'``, a Stop landed before the
                board began the home.
        """
        # Asked before the Z park: a scope with no turret would otherwise
        # park and restore Z and record slot 1 for a turret it does not
        # have, and one with no controller burn the driver's auto-reconnect
        # timeout first.
        self.refuse_absent('home', 'T')

        # T goes HOMING once Z is parked, not before: until then nothing
        # is turning it.
        _api_log.info('T home START')
        state_before = self._axis_states(('T',))['T']
        turret_before = self._last_turret_position
        # A homing turret is in no known slot until the home succeeds.
        self._record_turret_slot(None)
        stop_generation = self._stop_generation
        stops_asked = self._stops_asked
        # Outside the park's own command, so the turret's change comes into
        # force once its slot is recorded, not when the park's command closes
        # with the turret in no slot (see home()).
        with self._command():
            try:
                with (
                    self._reference_position_logger(),
                    # Before Z's park, and around it and the restore: a Stop
                    # already pressed parks nothing, and the park is the home's.
                    self._driving_home('T', stops_asked, {'T': state_before}, turret_before),
                    self._safe_turret_move(),
                ):
                    self._set_axis_state('T', AxisState.HOMING)
                    self._scope.frame_validity.invalidate(FrameValiditySource.TURRET, began_ns=None)
                    homed = False
                    try:
                        self._driver.thome()
                        homed = True
                    finally:
                        # Transition T out of HOMING on EVERY exit, including a
                        # raised driver call. The motion monitor polls MOVING, not
                        # HOMING, so nothing else ever takes T out of it: a
                        # still-HOMING T reads as the scope moving to every
                        # reader, and holds any wait_until_finished_moving begun
                        # during the home until its timeout. Failure -> UNKNOWN,
                        # success -> IDLE; both set the arrival event.
                        self._set_axis_state('T', AxisState.IDLE if homed else AxisState.UNKNOWN)
                read = self._refresh_position_cache()
                self._raise_unread_axes('T', ('T',), read)
                # Turret homes to slot 1 (see home() for why it is recorded). A
                # home a stop cut short did not reach it.
                if not self._stopped_since(stop_generation):
                    self._record_turret_slot(1)
            except HomingFailedError:
                raise
            except Exception as e:
                self._set_axis_state('T', AxisState.UNKNOWN)
                raise self._home_ending('T', stops_asked, ('T',)) from e
            finally:
                _api_log.info('T home DONE')

    @slow_task_budget(_TURRET_MOVE_SLOW_TASK_S)
    def _move_turret_impl(self, slot: int, restore_z: bool = True) -> None:
        """Move the turret to a slot. Skips if already there.

        Args:
            slot: Target turret slot (1-4).
            restore_z: When True (default), restore the pre-move Z
                position after the turret move completes. Set to False
                when the caller will immediately set Z to a different
                value (e.g. protocol step navigation moves T then Z to
                the new step's target -- restoring Z first is wasted
                motion).

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no turret (see
                ``refuse_absent``); refused before Z is parked, and nothing
                was driven or recorded.
            AxisStateUnknownError: The turret position is unknown.
            PositionOutOfRangeError: The slot is not a whole number 1-4.
            MoveNotCompletedError: The Z park, the turret move or the Z
                restore did not arrive, or was stopped. The slot is unknown
                afterwards, whichever of the three it was.
        """
        self.refuse_absent('move_turret', 'T')
        # Refused here as well as in the move's own refusals below, and both are
        # load-bearing: this one precedes the safety Z-retract and the
        # same-position short-circuit, so a nonsense slot cannot drop Z or
        # poison the position cache on its way to being refused.
        refuse_unless_turret_slot(slot)

        # Refuse BEFORE the safety Z-retract below, not inside it. The
        # retract is real motion; gating only the inner turret move would
        # drop Z to 0 against an unknown reference and refuse afterwards.
        #
        # This also precedes the same-position short-circuit: once the
        # turret reference is gone, the cache saying "already at 3" is a
        # claim about a physical position nothing can still vouch for,
        # and honoring it would report success for a turret that may be
        # anywhere.
        self._pre_drive('T')

        # Commanding a move of the T axis is slow, even if the move is to the current position.
        # A request for the slot the last successful turret command left the
        # turret in is answered without moving.
        if self._last_turret_position == slot:
            return

        # Unknown from the start, and written only once the whole command --
        # park, move, restore -- returned: a raise anywhere in it leaves the
        # turret in no slot anyone can vouch for.
        self._record_turret_slot(None)
        # Outside the park's own command, so the turret's change comes into
        # force once its slot is recorded (see home()).
        with self._command():
            with self._safe_turret_move(restore_z=restore_z) as stop_generation:
                logger.info(f'[SCOPE API ] Moving T to slot {slot}')
                self._move_absolute_impl('T', slot).result()
            # T's own wait judges only T; a stop after T arrived, before the
            # restore, ended the change all the same.
            if self._stopped_since(stop_generation):
                raise MoveStoppedError('T', 'stopped')
            self._record_turret_slot(int(slot))

    @api(reads=True)
    def get_turret_slot(self) -> int | None:
        """The turret slot in the light path, or None when it is not known.

        The slot the last turret command (``move_turret``, a home) left the
        turret in, recorded only when that command returned without error
        and no stop was issued while it ran. None before the first such
        command, while one is in flight, after one failed, and whenever the
        turret's position is lost. The turret has no encoder, so nothing
        else can say which slot is in the light path; the controller's step
        count is not a slot.

        Returns:
            int | None: The slot, 1-4, or None.

        Raises:
            Nothing.
        """
        return self._last_turret_position

    @api(reads=True)
    def turret_status(self) -> TurretStatus | None:
        """The turret's state and the slot in the light path, read together; None on a scope with no turret.

        The slot is ``get_turret_slot``'s; the state is the turret's
        ``AxisState``, which says what the slot alone cannot: whether it is
        homing, moving, or lost. No serial I/O.

        Raises:
            Nothing.
        """
        # The state store holds the present axes from bring-up on, before
        # the scope's capabilities exist; a home at bring-up tells this.
        with self._axis_state_lock:
            state = self._axis_state.get('T')
            slot = self._last_turret_position
        return None if state is None else TurretStatus(state, slot)

    def _record_turret_slot(self, slot: int | None) -> None:
        """Record the slot in the light path, and tell the turret listeners when it changed."""
        if slot == self._last_turret_position:
            return
        self._last_turret_position = slot
        self._fire_turret_listeners()

    @api(reads=True)
    def jog_step_um(self, axis: StageAxis, coarse: bool) -> float:
        """The jog step for ``axis`` under the active objective.

        A jog's size scales with the objective, so the active objective's
        record answers: ``jog_z_coarse_um`` / ``jog_z_fine_um`` for Z,
        ``jog_xy_coarse_um`` / ``jog_xy_fine_um`` for X and Y, the units
        ``move_xy_by`` and ``move_z_by`` take.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is not 'X',
                'Y' or 'Z': the turret has no jog.
            ObjectiveUnknownError: The objective in the light path is
                unknown; no step is guessed, so nothing should move.
        """
        _, objective = self._scope.runtime_state.resolve_current_objective()
        if axis == StageAxis.Z:
            return objective.jog_z_coarse_um if coarse else objective.jog_z_fine_um
        return objective.jog_xy_coarse_um if coarse else objective.jog_xy_fine_um

    @api(reads=True)
    def get_actual_position(self, axis: StageAxis) -> float | None:
        """Query a stage axis's actual hardware position via serial (not cached), in um.

        Unlike get_current_position(), which serves the in-memory cache,
        this asks the motor controller directly. Use during continuous
        motion sweeps where the stage is moving and the cache doesn't
        reflect the true position. The commanded target is a third thing
        again -- get_target_position() answers that.

        Costs one serial round-trip (~5ms).

        Args:
            axis: Axis name ("X", "Y", "Z"). The turret has a slot, not a
                position: ``get_turret_slot``.

        Returns:
            float | None: Current position in um; None with the null board
            installed or for an axis this scope does not have.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is not 'X',
                'Y' or 'Z'.
            HardwareCommandRefusedError: ``'not_connected'``, the installed
                motor controller is not connected (its cable pulled).
            HardwareError: the controller did not report the position.
        """
        if not self._has_position(axis):
            return None
        self.refuse_absent('get_actual_position')
        return self._driver.current_pos(axis)

    @api
    def set_precision_mode(self, axis: Axis, enabled: bool) -> None:
        """Set motor precision mode for an axis.

        Precision mode uses accurate but slightly slower motor stopping.
        Use before autofocus fine passes or any measurement requiring
        precise Z positioning. Disable for coarse moves where speed
        matters more than final position accuracy.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning, False for speed.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name; nothing was sent.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``refuse_absent``); nothing was sent.
        """
        return self._dispatch_motion(
            self._set_precision_mode_impl,
            'set_precision_mode',
            args=(axis, enabled),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    def _set_precision_mode_impl(self, axis: str, enabled: bool) -> None:
        self.refuse_absent('set_precision_mode', axis)
        self._driver.set_precision_mode(axis, enabled)

    def get_target_status(self, axis: str) -> bool:
        """Check if an axis has reached its target position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True when the motor board reports the axis at its target,
            False when it reports the axis short of it. An interlock found
            open while the axis moved (the LS720's lid) stops the stage:
            answered False, and that stop is a Stop, so every move in
            flight ends ``'stopped'``.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``refuse_absent``); nothing was sent.
            HardwareError: the board did not answer the read. A read that
                failed is not "short of the target", so it is never
                answered as False.
        """
        axis = refuse_unless_offered(Axis, axis, 'axis')
        self.refuse_absent('get_target_status', axis)
        return self._read_target_status(axis)

    def _read_target_status(self, axis: str) -> bool:
        """The board's answer to ``get_target_status``, with no refusal.

        The motion monitor's arrival read. Its reads are the scope watching
        its own board, through the close too: the lane-shut gate refuses a
        client's command, and refusing the monitor's read during the close
        blinded it, so a move that arrived or a STOP halted was given up as
        unread.

        Raises:
            HardwareError: the board did not answer the read.
        """
        try:
            return self._driver.target_status(axis)
        except MotionInterlockError as e:
            self._note_interlock_stop(e)
            _api_log.warning(f'{axis} moving with {e.reason}: the stage stopped every axis')
            return False

    @api(reads=True)
    def get_limit_switch_status(self, axis: Axis) -> tuple[int, int]:
        """Get the limit switch status for an axis.

        Tells a caller WHY a move stopped short: an axis that reached a limit
        reports it here rather than raising, so the difference between "the
        move finished" and "the stage ran out of travel" is only visible by
        asking.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            tuple[int, int]: ``(left, right)``, each 1 when that switch is
            engaged and 0 when clear.

        Raises:
            HardwareError: the board did not answer the read.
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``refuse_absent``); a switch the scope does not have is
                never answered as clear.
        """
        self.refuse_absent('get_limit_switch_status', axis)
        return self._driver.limit_switch_status(axis=axis)

    def get_limit_switch_status_all_axes(self) -> dict:
        """Get limit switch status for all axes.

        Returns:
            dict: Axis name -> the ``(left, right)`` pair described on
            ``get_limit_switch_status``. Covers only the axes the board
            reports, so a scope without a turret has no "T" key.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, no motor
                controller (see ``refuse_absent``).
            HardwareError: the board did not answer an axis's read.
        """
        self.refuse_absent('get_limit_switch_status_all_axes')
        resp = {}
        for axis in self._scope.capabilities.axes:
            resp[axis] = self.get_limit_switch_status(axis=axis)
        return resp

    @api(reads=True)
    def is_moving(self) -> bool:
        """Check if any axis is currently moving.

        Reads from in-memory axis state -- zero serial I/O. The motion
        monitor thread handles firmware queries and state transitions.

        Returns:
            bool: True if any axis is MOVING or HOMING. A Z move is MOVING
            from its first target write, its backlash leg included.

        Raises:
            Nothing.
        """
        return self.is_any_axis_moving()

    def set_acceleration_limit(self, limit_pct: int) -> None:
        """Set the motor controller acceleration limit (percent of max).

        Firmware that does not implement the AMAX / DMAX query needs no
        handling here: the driver's own read already answers with a default
        for it, so a legacy board completes this call rather than failing it.
        A refusal that does reach this point is therefore a real one, and it
        belongs to the caller -- swallowing it left a scope configured from a
        value its firmware had rejected, with nothing said to anyone.

        Args:
            limit_pct: Acceleration limit as a percent of the firmware max.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller (see
                ``refuse_absent``); asked before the value, and nothing was
                sent.
            AccelerationLimitRefusedError: ``limit_pct`` is not a whole number
                or is outside ``ACCELERATION_PCT_MIN`` to ``ACCELERATION_PCT_MAX``
                (a ValueError). Refused on every board, real or simulated,
                before it is commanded.
        """
        return self._dispatch_motion(
            self._set_acceleration_limit_impl,
            'set_acceleration_limit',
            kwargs={'limit_pct': limit_pct},
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    def _set_acceleration_limit_impl(self, limit_pct: int) -> None:
        self.refuse_absent('set_acceleration_limit')
        refuse_acceleration_pct(limit_pct)
        self._driver.set_acceleration_limits(val_pct=limit_pct)

    # ------------------------------------------------------------------
    # Stateful method bodies.
    #
    # State slots (_pos_cache, _axis_state, _arrival_events,
    # _position_listeners, _motion_wake, _motion_monitor_*, _commands_open)
    # live on this surface.
    # ------------------------------------------------------------------

    def _command_open(self) -> bool:
        """Whether a several-move command is open (``one_command``).

        Takes no lock: frame validity's settle check asks it under frame
        validity's own lock, and a lock here would be taken in the opposite
        order by a command closing.
        """
        return self._commands_open > 0

    @contextlib.contextmanager
    def one_command(self, axes: Iterable[str], *, then: str) -> Iterator[None]:
        """Run a person's several-move command as one change.

        Use as ``with motion.one_command(axes, then=...): ...start the
        moves...``. Asks ``refuse_unknown_positions(axes, recording=False,
        then=then)`` once, then holds the command open until the body
        returns or raises. While it is open no motion change settles, so a
        capture asked during it waits until its last move is under way and
        then for that move: it never returns a frame from between two of
        its moves, such as Z parked during a turret change or the new X at
        the old Y. A capture asked before the command's first move may
        return the frame from before it, which is true.

        A seam for the session's and the GUI's gestures, not part of the L2
        API surface, as ``refuse_unknown_positions`` is not.

        Args:
            axes: The axes the command needs; see ``refuse_unknown_positions``.
            then: What the user does once the scope knows its position.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, a name in *axes* is no
                axis name; nothing was opened.
            AxisStateUnknownError: Naming every refused axis; nothing was
                opened.
        """
        self.refuse_unknown_positions(axes, recording=False, then=then)
        with self._command():
            yield

    @contextlib.contextmanager
    def _command(self) -> Iterator[None]:
        """Hold a command open around its moves; ``one_command`` without the question.

        For a command whose moves ask for themselves (the turret change's
        pre-drive gate). When the last open command closes, each motion
        change whose axis arrived while it was open, and so was given no
        in-force time, is given the close's: the moment its frames began to
        count. A change stamped before the command keeps its own time, so a
        command never moves the in-force time of a change it did not make.
        Each is given, with it, the latest read that saw one of the
        command's moves still moving, since its arrival stamped nothing: a
        turret change shows only once its last move is done.
        """
        with self._commands_lock:
            if self._commands_open == 0:
                self._command_moving_seen_ns = None
            self._commands_open += 1
        try:
            yield
        finally:
            with self._commands_lock:
                self._commands_open -= 1
                closed = self._commands_open == 0
                moving_seen_ns = self._command_moving_seen_ns
            if closed:
                closed_ns = host_clock.steady_ns()
                frame_validity = self._scope.frame_validity
                for source, change in frame_validity.changes().items():
                    # motion_settled stamps only a change that waits on an
                    # axis now settled; an axis still moving is stamped by
                    # its own arrival, which comes later.
                    if change.in_force_ns is None:
                        frame_validity.motion_settled(
                            source, closed_ns, moving_seen_ns=moving_seen_ns
                        )

    def _home_moves_turret(self, action) -> bool:
        """Whether the home body ``action`` moves the turret.

        The whole-scope home homes every axis the board has, so it moves the
        turret exactly when the scope has one.
        """
        if action == self._home_turret_impl:
            return True
        return action == self._home_impl and self._scope.capabilities.has_turret

    @api(reads=True)
    def get_axis_state(self, axis: Axis) -> AxisState:
        """Get the current state of an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            AxisState: UNKNOWN, IDLE, MOVING or HOMING; UNKNOWN for an axis
            this scope does not have.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name.
        """
        with self._axis_state_lock:
            return self._axis_state.get(axis, AxisState.UNKNOWN)

    @api(in_process=True, event=PositionChanged)
    def add_position_listener(self, listener: Callable[[StageAxis, AxisPosition], None]) -> None:
        """Register a callback for position/state changes on a stage axis: X, Y or Z.

        The listener is called with ``(axis, AxisPosition)`` whenever the
        position cache or axis state changes: the axis's state and its
        polled position as ``axis_positions`` answers them then, from the
        same snapshot -- not the target, and no position (None) while the
        axis is HOMING or UNKNOWN. The turret's changes go to
        ``add_turret_listener``. It fires from the thread that caused the
        change (IO executor, motion monitor, etc.), so listeners **must**
        schedule any UI work via ``Clock.schedule_once``.

        Args:
            listener: ``callable(axis: StageAxis, at: AxisPosition)``

        Raises:
            Nothing.
        """
        with self._position_listeners_lock:
            self._position_listeners.append(listener)

    @api(in_process=True)
    def remove_position_listener(self, listener: Callable[[StageAxis, AxisPosition], None]) -> None:
        """Unregister a position listener.

        Args:
            listener: A callable previously passed to
                ``add_position_listener``. Silently ignores listeners that
                are not currently registered.

        Raises:
            Nothing.
        """
        with self._position_listeners_lock:
            try:
                self._position_listeners.remove(listener)
            except ValueError:
                pass

    @api(in_process=True, event=TurretChanged)
    def add_turret_listener(self, listener: Callable[[TurretStatus], None]) -> None:
        """Register a callback for each change of the turret's state or slot.

        The listener is called with the ``TurretStatus`` that
        ``turret_status`` answers then: on each change of the turret's state
        (homing, moving, idle, unknown), on each read the motion monitor
        makes of it while it turns, and once the slot a turret command
        reached is recorded, which is after the turret says idle. It fires
        from the thread that caused the change, so listeners **must**
        schedule any UI work via ``Clock.schedule_once``. Never called on a
        scope with no turret.

        Args:
            listener: ``callable(at: TurretStatus)``

        Raises:
            Nothing.
        """
        with self._position_listeners_lock:
            self._turret_listeners.append(listener)

    @api(in_process=True)
    def remove_turret_listener(self, listener: Callable[[TurretStatus], None]) -> None:
        """Unregister a turret listener.

        Args:
            listener: A callable previously passed to
                ``add_turret_listener``. Silently ignores listeners that
                are not currently registered.

        Raises:
            Nothing.
        """
        with self._position_listeners_lock:
            try:
                self._turret_listeners.remove(listener)
            except ValueError:
                pass

    def add_stop_press_listener(self, listener: Callable[[], None]) -> None:
        """Register a callback for each press of ``stop_motion`` (not part of the L2 API surface).

        The run engine's hook, registered once for the engine's life: a
        press ends the live run, for every client that presses STOP.

        Called with no arguments as STOP is pressed, before the STOP is sent
        and whether or not the board takes it, on the presser's thread, so a
        listener must not wait.

        Args:
            listener: ``callable()``
        """
        with self._stop_press_listeners_lock:
            self._stop_press_listeners.append(listener)

    def _fire_position_listeners(self, axis: str):
        """Notify the listeners of a change on *axis*: a stage axis's as ``axis_positions`` answers it, the turret's as ``turret_status`` does."""
        if axis == Axis.T:
            self._fire_turret_listeners()
            return
        stage_axis = StageAxis(axis)
        at = self.axis_positions()[stage_axis]
        with self._position_listeners_lock:
            listeners = list(self._position_listeners)
        notifications.tell_listeners(listeners, stage_axis, at, category='Motion')

    def _fire_turret_listeners(self):
        """Notify all turret listeners of the turret's state and slot, as ``turret_status`` answers them."""
        at = self.turret_status()
        if at is None:
            return
        with self._position_listeners_lock:
            listeners = list(self._turret_listeners)
        notifications.tell_listeners(listeners, at, category='Motion')

    def is_any_axis_moving(self) -> bool:
        """Check if any axis is currently MOVING or HOMING.

        Reads from the in-memory state dict -- zero serial I/O.

        Returns:
            bool: True if any axis is in MOVING or HOMING state.
        """
        with self._axis_state_lock:
            return any(s in (AxisState.MOVING, AxisState.HOMING) for s in self._axis_state.values())

    @api(reads=True)
    def get_axis_limits(self, axis: Axis) -> AxisLimits | None:
        """Get the travel limits for an axis, in um.

        Args:
            axis: Axis name ("X", "Y", "Z", or "T").

        Returns:
            The travel moves are refused against, or ``None`` if the axis
            has no configured limits (typical for the turret T axis).
            Callers must handle the None case.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``refuse_absent``); an axis the scope does not have has no
                travel to answer with.
        """
        self.refuse_absent('get_axis_limits', axis)
        limits = self._driver.get_axis_limits(axis=axis)
        if limits is None:
            return None
        return AxisLimits(min_um=float(limits['min']), max_um=float(limits['max']))

    @slow_task_budget(_HOMING_SLOW_TASK_S)
    @_home_runs_its_own_moves
    def _zhome_impl(self) -> None:
        """Home the Z axis (focus).

        Returns when Z is homed and its position read.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no Z (see
                ``refuse_absent``); or an interlock reason, the stage's
                interlock refused the home before anything moved. Nothing
                was driven, and Z keeps the state it had.
            HomingFaultError: the driver raised (e.g. HardwareError on
                no-response / firmware-error), or Z's position could not be
                read; ``'lid_open'``, the lid was opened while the home
                moved.
            HomingStoppedError: ``'stopped'``, a Stop ended it.
        """
        # Asked before the driver, as the full home does: its
        # exchange_command would burn its auto-reconnect timeout and the
        # user see a hang instead of the actual cause.
        self.refuse_absent('home', 'Z')
        _api_log.info('Z home START')
        # What a home the interlock refuses before anything moves gives back.
        state_before = self._axis_states(('Z',))['Z']
        self._set_axis_state('Z', AxisState.HOMING)
        stops_asked = self._stops_asked
        self._scope.frame_validity.invalidate(FrameValiditySource.Z_MOVE, began_ns=None)
        try:
            with (
                self._reference_position_logger(),
                self._driving_home(
                    'Z', stops_asked, {'Z': state_before}, self._last_turret_position
                ),
            ):
                self._driver.zhome()
            read = self._refresh_position_cache()
            if 'Z' in read:
                self._set_axis_state('Z', AxisState.IDLE)
            self._raise_unread_axes('Z', ('Z',), read)
        except HomingFailedError:
            raise
        except MotionInterlockError as e:
            self._note_interlock_stop(e)
            if not e.moved:
                self._give_back_states({'Z': state_before})
                raise HardwareCommandRefusedError(e.reason, 'home') from e
            self._set_axis_state('Z', AxisState.UNKNOWN)
            raise HomingFaultError('Z', e.reason, ('Z',)) from e
        except Exception as e:
            self._set_axis_state('Z', AxisState.UNKNOWN)
            raise self._home_ending('Z', stops_asked, ('Z',)) from e
        finally:
            _api_log.info('Z home DONE')

    @api(reads=True)
    def has_homed(self) -> bool:
        """Whether the stage / focus axes have a known reference position.

        Answers from the axis state rather than the driver's homing
        latch: the latch survives
        every fault short of a physical disconnect, so it keeps
        reporting "homed" after a stall or a dropout has already
        invalidated the reference frame.

        The turret is excluded -- it has its own reference and its own
        question. A scope whose turret faulted has not lost its stage
        coordinates.

        Returns:
            bool: True if every non-turret axis the board has knows its
                position. False on a board with no such axes, which is
                what the driver latch reported for the no-motor case.

        Raises:
            Nothing.
        """
        with self._axis_state_lock:
            stage_states = [state for axis, state in self._axis_state.items() if axis != 'T']
        if not stage_states:
            return False
        return all(self._position_known(state) for state in stage_states)

    @api(reads=True)
    def axes_without_position(self) -> dict[Axis, AxisState]:
        """Which of this scope's axes do not know their position, and why.

        The question a run asks before it starts and before each capture:
        every run moves every axis the scope has, and an image taken where
        an axis is not known is saved with a position that is not true.
        Each axis is answered with its own state because the two causes
        need different words -- a HOMING axis is about to know, and its
        user should wait; an UNKNOWN one has lost its reference, and its
        user must home.

        Unlike ``has_homed``, a scope with no axes answers nothing: it has
        no position to lose, so nothing about it is unknown.

        Returns:
            dict[Axis, AxisState]: Axis to its state (``AxisState.UNKNOWN``
                or ``AxisState.HOMING``) for every axis whose position is
                not known, in the scope's axis order. Empty when every
                axis knows its position.

        Raises:
            Nothing.
        """
        with self._axis_state_lock:
            return {
                axis: state
                for axis, state in self._axis_state.items()
                if not self._position_known(state)
            }

    @api(reads=True)
    def axis_positions(self) -> dict[StageAxis, AxisPosition]:
        """Every stage axis's state and its position, or None where the position is not known.

        One snapshot: the state and the cache are read together, so a
        caller writing a position into a file cannot pair a number with a
        state that changed between two reads. An axis is answered with a
        position only while it is IDLE or MOVING; UNKNOWN and HOMING
        answer None whatever the cache holds, because the cache keeps the
        last number an axis reported after its reference is lost. X, Y and
        Z, in um; the turret has a slot, not a position (``turret_status``).
        No serial I/O.

        The state lock is taken first and the cache lock inside it. No
        other path nests the two, so this order cannot meet its reverse.

        Returns:
            dict[StageAxis, AxisPosition]: Stage axis to (state, position), in
                the scope's axis order.

        Raises:
            Nothing.
        """
        with self._axis_state_lock:
            states = dict(self._axis_state)
            with self._pos_cache_lock:
                cache = dict(self._pos_cache)
        return {
            StageAxis(ax): AxisPosition(
                state, cache.get(ax) if self._position_known(state) else None
            )
            for ax, state in states.items()
            if ax != Axis.T
        }

    def _raise_unread_axes(self, home: str, homed: tuple | list, read: set[str]) -> None:
        """After a home: raise when a homed axis could not be read.

        A home whose mechanics succeeded but whose position could not be
        read has not established a reference: the axis is already UNKNOWN
        (the refresh set it) and the caller must hear a failure, not a
        return that every consumer reads as "the scope knows where it is".
        Only an axis the board has is read, so an absent one (a turret
        home on a scope with no turret) is never counted unread.

        Raises:
            HomingFaultError: ``'unread'``, naming the unread axes.
        """
        present = self._scope.capabilities.axes
        unread = [ax for ax in homed if ax in present and ax not in read]
        if unread:
            raise HomingFaultError(home, 'unread', unread)

    def _refresh_position_cache(self) -> set[str]:
        """Read every axis's position from the hardware into the cache.

        Called after a home, and by ``_init_axes``, to sync the cache
        with the hardware; during normal operation the cache is updated
        by move commands and the motion monitor. An axis whose read fails
        is set UNKNOWN, a move it ends ending ``'position_unread'``, and its
        cache entry is left alone:
        a number nobody read is not a position, and a caller that writes
        positions into a file would otherwise record it as one.

        The axes are the state store's own, the present axes ``_init_axes``
        was given: it runs before the scope's capabilities exist.

        Returns:
            set[str]: The axes whose position was read.
        """
        positions = {}
        for ax in list(self._axis_state):
            try:
                positions[ax] = self._driver.target_pos(axis=ax)
            except HardwareError as e:
                _api_log.warning(f'position read on {ax} failed ({e}); axis UNKNOWN')
                self._set_axis_state(
                    ax, AxisState.UNKNOWN, fault=MoveFaultError(ax, 'position_unread')
                )

        with self._pos_cache_lock:
            self._pos_cache.update(positions)
        for ax in positions:
            self._fire_position_listeners(ax)
        return set(positions)

    def _read_position_cache(self, axis: str) -> float:
        """Shared cache-read primitive for the position-query methods.

        Pure cache access -- no T-axis policy here; callers decide their
        own sentinel for the "axis requested but not present" case (see
        get_target_position's None for no-turret-T). 0.0 if the axis is
        missing from the cache.
        """
        with self._pos_cache_lock:
            return self._pos_cache.get(axis, 0.0)

    @api(reads=True)
    def get_target_position(self, axis: StageAxis) -> float | None:
        """Get the target position for a stage axis (where it is commanded to go), in um.

        The last target this API wrote to the board for the axis, while the
        move runs and after it arrives: the commanded number, not the polled
        position, which sits up to a microstep off it. A relative move's
        target is the board's own target plus the offset, so it carries that
        target's microstep rounding. Where no target was reached it is the
        polled position: before the axis's first move, after a home, after a
        STOP ended the move, while the axis is UNKNOWN, and while a move has
        not yet written its final target (a Z backlash leg). An axis whose
        position has not been read since bring-up -- its bring-up read
        failed -- is UNKNOWN and answers 0.0, which nothing read. A refused
        move writes nothing, so the previous target stands. No serial I/O.

        Args:
            axis: Axis name ("X", "Y", "Z"). The turret has a slot, not a
                position: ``get_turret_slot``.

        Returns:
            float | None: The target in um, or None for an axis with no
                hardware behind it (see ``_has_position``).

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is not 'X',
                'Y' or 'Z'.
        """
        if not self._has_position(axis):
            return None
        with self._axis_state_lock:
            state = self._axis_state.get(axis, AxisState.UNKNOWN)
            move = self._current_move.get(axis)
            # An UNKNOWN after an arrival leaves the move settled 'arrived',
            # so the state is tested as well as the outcome.
            if (
                move is not None
                and move.target is not None
                and state in (AxisState.MOVING, AxisState.IDLE)
                and move.outcome in (None, 'arrived')
            ):
                return float(move.target)
        return self._read_position_cache(axis)

    @api(reads=True)
    def get_target_positions(self) -> dict[StageAxis, float]:
        """The target of every stage axis the scope has a position for, by axis name, as ``get_target_position`` answers each.

        An axis with no hardware behind it is left out, and so is the
        turret, which has a slot (``get_turret_slot``). No serial I/O.

        Raises:
            Nothing.
        """
        return {
            StageAxis(ax): self.get_target_position(ax)
            for ax in self._scope.capabilities.axes
            if ax != Axis.T and self._has_position(ax)
        }

    def get_current_position(self, axis: Axis) -> float | None:
        """Get the current position for an axis; um for X/Y/Z, turret slot (1-4) for T.

        Reads from the in-memory position cache. During MOVING the cache
        is refreshed by _motion_monitor_loop polling the motor's actual
        position from hardware on every cycle; during IDLE the cache
        holds the last confirmed position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float | None: The position in um, or None for an axis with no
                hardware behind it (see ``_has_position``); a present axis
                keeps its number, its reference lost or not, and one never
                read since bring-up is UNKNOWN at 0.0, which nothing read.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name.
        """
        axis = refuse_unless_offered(Axis, axis, 'axis')
        if not self._has_position(axis):
            return None
        return self._read_position_cache(axis)

    def get_current_positions(self) -> dict[Axis, float]:
        """The position of every axis the scope has a position for, by axis name, as ``get_current_position`` answers each.

        An axis with no hardware behind it is left out. No serial I/O.
        """
        return {
            ax: self.get_current_position(ax)
            for ax in self._scope.capabilities.axes
            if self._has_position(ax)
        }

    def _plate_target_to_stage(self, axis: str, plate_mm: float) -> float:
        """Check a plate target for X or Y against what this stage can reach, then convert.

        A plate coordinate is the number a user types into the position
        boxes, in mm from the plate's top-left. Converting it before
        checking it is what produced refusals quoting a negative stage
        micron value for a positive typed number: the transform inverts
        the axis, so a coordinate past the plate becomes a target below
        zero, and the travel check then reported THAT number.

        The bound checked here is the reachable band -- the set of plate
        coordinates whose converted target lies within travel -- rather
        than the labware extent, which is wider. An extent check would
        pass a coordinate the stage still cannot serve and hand the user
        a second refusal in the other frame for the same mistake.

        The band is the inverse image of the travel interval under a
        transform that is affine and strictly decreasing in the plate
        coordinate, so refusing here rejects exactly what the travel
        check downstream would reject, in the unit the caller gave.
        """
        key = axis.lower()
        # Each stage axis depends only on its own plate axis, so the one
        # coordinate stands for both and the other half is not read.
        converted = self._scope.runtime_state.plate_to_stage(plate_mm, plate_mm)
        stage_position = converted.x_um if axis == 'X' else converted.y_um

        limits = self.get_axis_limits(axis)
        if limits is not None:
            dimension = self._scope.runtime_state.get_labware().get_dimensions()[key]
            offset = self._scope.runtime_state.get_stage_offset()
            offset_mm = (offset.x_um if axis == 'X' else offset.y_um) / 1000
            # Inverting sx = (dimension - offset - px) * 1000: the map
            # decreases in px, so the travel MAXIMUM yields the plate
            # minimum and vice versa.
            band_low = round(dimension - offset_mm - limits.max_um / 1000, 2)
            band_high = round(dimension - offset_mm - limits.min_um / 1000, 2)
            if not (band_low <= plate_mm <= band_high):
                raise PositionOutOfRangeError(
                    axis,
                    plate_mm,
                    band_low,
                    band_high,
                    bound='reachable range',
                    quantity='plate position',
                )

        return stage_position

    def _move_absolute_impl(
        self,
        axis: str,
        position: float,
        overshoot_enabled: bool = True,
        ignore_limits: bool = False,
        force: bool = False,
    ) -> Operation:
        """Start an axis moving to an absolute position; return the started move.

        Returns once the board has taken the command and the axis is
        MOVING. The returned operation's ``result()`` is the move's outcome;
        a caller that needs the axis where it was sent waits on it.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            position (float): Target position -- um for X/Y/Z; turret slot (1-4) for T.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            ignore_limits: If True, skip software limit checks.
            force: Drive even when the axis position is unknown. For the
                recovery paths only -- see ``_pre_drive``.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the position is not a
                finite number, refused before any limit, the safety limit on
                the ``ignore_limits`` path included. The callers, the turret
                change's moves, pass axis literals.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``refuse_absent``); nothing was driven.
            PositionOutOfRangeError: The target is outside the axis's
                configured travel and ``ignore_limits`` is False. A
                ValueError subclass.
            AxisStateUnknownError: The axis position is unknown and
                ``force`` is False.
            HardwareCommandRefusedError: ``'position_unread'``, a Z move
                with overshoot whose position the board did not report;
                nothing was driven.
            MoveFaultError: ``'driver_failed'``, the board did not
                take the command.
        """
        self._refuse_absolute('move_turret', axis, position, 'position', ignore_limits, force)
        return self._drive(
            'move_turret', axis, position, overshoot_enabled, f'move_abs {axis}={position:.1f}um'
        )

    def _refuse_absolute(
        self,
        member: str,
        axis: str,
        position: float,
        argument: str,
        ignore_limits: bool,
        force: bool = False,
    ) -> None:
        """Every refusal of a move of ``axis`` to ``position``, in order; nothing is driven.

        The checks of an absolute move without its drive, so a request for
        several axes asks each of them for every axis before any drives.
        ``argument`` names ``position`` in a ``'not_a_number'`` refusal.
        The raises are ``_move_absolute_impl``'s, but for the Z leg's read,
        which is the drive's.
        """
        self.refuse_absent(member, axis)
        refuse_unless_finite_number(position, argument)

        # Refuse a target beyond the axis's travel; this is the only travel
        # check, the drivers have none. A move stopped short at a limit
        # reports success at a position nobody asked for, so a protocol step saved beyond this scope's
        # travel images the wrong place and the log cannot tell that from a
        # step that went where it was told. Axes with no configured travel
        # return None here -- the turret, whose position is a slot rather
        # than a distance -- so THIS check cannot refuse anything for them.
        # That does not make them unbounded: the turret's own bound is
        # checked below, against its slots.
        if not ignore_limits:
            limits = self.get_axis_limits(axis)
            if limits is not None and not (limits.min_um <= position <= limits.max_um):
                raise PositionOutOfRangeError(axis, position, limits.min_um, limits.max_um)

        # The turret's bound, which the travel check above cannot express: a
        # slot is not a distance, so get_axis_limits returns None for T and
        # nothing there refuses anything. Without this, a T target of 99 is
        # 24.5 revolutions. No public move takes T; the caller that reaches
        # this body with T is move_turret, and the bound keeps that one
        # honest too.
        #
        # Before the ceiling below for the same reason travel is: the bound
        # that knows what the number MEANS answers first, so the turret gives
        # one vocabulary for every bad slot rather than naming slots for 5 and
        # a metre for 2000000.
        if axis == 'T':
            refuse_unless_turret_slot(position)

        # The coarse sanity ceiling, checked AFTER the bounds above so the
        # bound that knows the axis answers first. For any axis that publishes
        # travel, travel lies inside this bound, and the turret is refused by
        # slot above, so reaching here means no bound that understands this
        # axis could speak for it. Ordering it the other way gave a user two
        # different answers for one mistake: a typed value a little past
        # travel named the travel range, and a larger one named a 1 m ceiling
        # that means nothing to them. Not gated on ignore_limits: that hatch
        # is for driving outside TRAVEL deliberately, not for handing the
        # motor an arbitrary number.
        if abs(position) > MOTOR_POSITION_LIMIT:
            raise PositionOutOfRangeError(
                axis,
                position,
                -MOTOR_POSITION_LIMIT,
                MOTOR_POSITION_LIMIT,
                bound='safety limit',
            )

        self._pre_drive(axis, force=force)

    def _drive(
        self, member: str, axis: str, position: float, overshoot_enabled: bool, said: str
    ) -> Operation:
        """Drive ``axis`` to ``position``, refused already; return the started move.

        The one drive of a move to a place or by a distance, once its
        refusals have passed. ``said`` is the move as the API log names it.

        Raises:
            HardwareCommandRefusedError: ``'position_unread'``, a Z move
                with overshoot whose position the board did not report;
                the interlock's reason (``'lid_open'``); ``'monitor_ended'``.
                Nothing of this axis was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take the
                command.
        """
        leg = self._backlash_leg(axis, position, overshoot_enabled, member)
        stop_generation = self._generation_asked()
        try:
            move, written = self._send_drive(
                axis,
                stop_generation,
                lambda: self._drive_to(axis, position, leg, stop_generation),
                member,
            )
        except HardwareCommandRefusedError:
            raise
        except Exception:
            _api_log.error(f'{said} FAILED')
            raise
        self._publish_drive(axis, written, float(position))
        # No move-init cache write: the cache holds the CURRENT position
        # until _motion_monitor_loop reads it from hardware on its first
        # cycle. The target is the move's own (_Move.target), where
        # get_target_position reads it.
        self._fire_position_listeners(axis)
        _api_log.info(said)
        return move.operation

    def _wait_for_axis_to_stop(self, axis: str, timeout_s: float) -> bool:
        """Wait for ``axis``'s arrival event; True when it was set within ``timeout_s``.

        The event is set when the axis goes IDLE or UNKNOWN, so True says
        only that the axis stopped moving; the state says how.
        """
        return self._arrival_events[axis].wait(timeout=timeout_s)

    def _stopped_since(self, stop_generation: int) -> bool:
        """Whether a stop landed after ``_stop_generation`` read ``stop_generation``.

        Read under the lock stop_motion holds across its exchange, so a
        caller whose motion that stop ended waits for the bump instead of
        reading the value from before it.
        """
        with self._stop_lock:
            return self._stop_generation != stop_generation

    def _refuse_relative(self, member: str, axis: str, distance: float, argument: str) -> float:
        """Every refusal of a move of ``axis`` by ``distance``, in order; the target it drives to.

        The checks of a relative move without its drive, as
        ``_refuse_absolute`` is for an absolute one. ``argument`` names
        ``distance`` in a ``'not_a_number'`` refusal. The raises are
        ``start_move_z_by``'s, but for the Z leg's read, which is the
        drive's.
        """
        self.refuse_absent(member, axis)
        refuse_unless_finite_number(distance, argument)
        if abs(distance) > MOTOR_POSITION_LIMIT:
            # Same refusal as the absolute path, reachable the same way.
            raise PositionOutOfRangeError(
                axis,
                distance,
                -MOTOR_POSITION_LIMIT,
                MOTOR_POSITION_LIMIT,
                bound='safety limit',
                quantity='distance',
            )

        # This path does NOT route through the absolute one, so it needs
        # the gate of its own.
        self._pre_drive(axis)

        # The offset is added to the board's own target: a move still in
        # flight on this axis is added to, so chained jogs accumulate, and
        # after a stop the target is where the stage stopped. This one
        # number is range-checked, published and driven. The API's cache was
        # the base before, beside the driver driving TARGET_R plus the
        # offset, and the cache can hold a position the monitor never read.
        try:
            start_pos = self._driver.target_pos(axis)
        except HardwareError as e:
            raise HardwareCommandRefusedError('position_unread', member) from e
        target_pos = start_pos + float(distance)

        # The same travel refusal as the absolute path, against the target
        # this move is about to publish. Without it the driver turned the
        # offset into an absolute move beyond travel and nothing refused it,
        # so a jog past a limit reported success from wherever the stage
        # stopped.
        limits = self.get_axis_limits(axis)
        if limits is not None and not (limits.min_um <= target_pos <= limits.max_um):
            raise PositionOutOfRangeError(axis, target_pos, limits.min_um, limits.max_um)
        return target_pos

    def _move_xy_impl(
        self, member: str, x_um: float | None, y_um: float | None, ignore_limits: bool
    ) -> Operation:
        """Start X and Y moving to a place, each that is asked for; return the started moves.

        Every axis asked for is refused before either drives, then each is
        driven, X first, on this one lane task. An axis given as None is
        not read, refused, written or begun, whatever it is doing.
        ``member`` names the request in a refusal.

        Returns:
            One ``Operation`` for the axes driven (``Operation.all_of``),
            done at once when neither is asked for.
        """
        targets = tuple((axis, um) for axis, um in (('X', x_um), ('Y', y_um)) if um is not None)
        for axis, um in targets:
            self._refuse_absolute(member, axis, um, f'{axis.lower()}_um', ignore_limits)
        return Operation.all_of(
            tuple(
                self._drive(member, axis, um, False, f'move_abs {axis}={um:.1f}um')
                for axis, um in targets
            )
        )

    def _move_xy_by_impl(self, dx_um: float | None, dy_um: float | None) -> Operation:
        """Start X and Y moving by a distance, each that is asked for; return the started moves.

        As ``_move_xy_impl``: every axis asked for has its distance added
        to the board's target and is refused before either drives.
        """
        targets = tuple(
            (axis, d, self._refuse_relative('move_xy_by', axis, d, f'd{axis.lower()}_um'))
            for axis, d in (('X', dx_um), ('Y', dy_um))
            if d is not None
        )
        return Operation.all_of(
            tuple(
                self._drive('move_xy_by', axis, target, False, f'move_rel {axis}={d:+.1f}um')
                for axis, d, target in targets
            )
        )

    def _move_z_impl(
        self, z_um: float, overshoot_enabled: bool | None, ignore_limits: bool
    ) -> Operation:
        """Start Z moving to a place; return the started move.

        A move to a place approaches from below (``_backlash_leg``) unless
        ``overshoot_enabled`` says otherwise.
        """
        self._refuse_absolute('move_z', 'Z', z_um, 'z_um', ignore_limits)
        approach = True if overshoot_enabled is None else overshoot_enabled
        return self._drive('move_z', 'Z', z_um, approach, f'move_abs Z={z_um:.1f}um')

    def _move_z_by_impl(self, dz_um: float, overshoot_enabled: bool | None) -> Operation:
        """Start Z moving by a distance; return the started move.

        A move by a distance takes no approach from below unless
        ``overshoot_enabled`` says otherwise.
        """
        target = self._refuse_relative('move_z_by', 'Z', dz_um, 'dz_um')
        approach = False if overshoot_enabled is None else overshoot_enabled
        return self._drive('move_z_by', 'Z', target, approach, f'move_rel Z={dz_um:+.1f}um')

    # --- Public dispatch ---
    # These are what every caller reaches: an SDK script, a REST
    # handler, the GUI -- and the run, the autofocus sweep and the diagnostics,
    # which call them under their taking so the lane admits their work while
    # they hold the scope. From a task already on the lane's worker the lane
    # runs the body inline.

    # Liveness margin for a dispatched motion command that waits for no
    # motion (a move's command, a precision mode or acceleration write):
    # queue residence plus the serial round-trips, with headroom. A home and
    # a turret move wait for their body with no bound of their own: the
    # board bounds a home, and each of a turret move's moves is judged.
    _MOTION_WAIT_BASE_S = 30.0

    # How long a moving axis's microstep reading may hold one value, with
    # the board saying it has not arrived, before the monitor gives it up as
    # stalled. A healthy move holds a reading only through its ramps: 0.14 s
    # at most on an LS850T on field firmware (X, Y and the turret, 2026-10-14,
    # tests/capability/motion_p11_stop_halts_and_progress_reads.py), 0.08 s in
    # the simulator, Z included.
    _MOVE_STALL_WINDOW_S = 2.0

    def _dispatch_motion(
        self,
        impl,
        name,
        args=(),
        kwargs=None,
        *,
        timeout_s,
        slow_task_threshold_sec=None,
        falsifies_recording=False,
    ):
        """Run one motion command for an external caller, on the right thread.

        The body runs on the scope's io lane, serialized against every other
        hardware write, and this blocks until it has. A lane that will not
        accept work tells the caller so, because the alternative is `put`
        returning None and the command disappearing with nothing raised and
        nothing logged.

        The lane's ``call`` decides a refusal and raises it to the caller:
        the lane is closed, or a run, a diagnostic or a home holds the scope and this
        call is not made under its taking, or ``falsifies_recording`` is set
        and a recording holds the scope.

        slow_task_threshold_sec declares how long this command may take
        before the elapsed-time WARNING means anything. Left None the task
        inherits IOTask's default, which describes a SINGLE motion -- a
        command that physically performs several will trip it every time it
        succeeds. Distinct from timeout_s, which is the hung bound: this one
        is "unusually slow, look at it", that one is "definitively stuck".
        """
        kwargs = kwargs or {}
        return self._scope._io_executor.call(
            IOTask(
                action=impl,
                args=args,
                kwargs=kwargs,
                slow_task_threshold_sec=slow_task_threshold_sec,
                falsifies_recording=falsifies_recording,
            ),
            name,
            timeout_s,
        )

    @api(stop='Operation.stop')
    def start_move_to_plate(self, at: PlateLocation) -> Operation:
        """Start X and Y moving to ``at`` on the selected plate; return the started moves.

        ``at`` is in mm from the plate's top-left, at the live stage
        offset, the number a person types into a position box. An axis that
        is None is left where it is, a move under way on it included, so a
        caller moving X alone leaves Y alone. The axes given are asked once
        whether they know their position and are started on one task of the
        IO lane (``one_command``), so a capture asked meanwhile never takes
        the new X at the old Y. Each is checked against the plate positions
        this stage can reach, and refused in mm, before either moves.

        Returns:
            One ``Operation`` for the moves started, done once every one is
            (``Operation.all_of``); its ``result()`` raises the first move's
            raise, as ``start_move_xy``'s does. With no axis given
            nothing moves and it is done at once.

        Raises:
            ArgumentRefusedError: ``'plate_frame_axis'``, a Z given: a plate
                position has no Z; ``'not_a_number'``, an axis given is not a
                finite number. Nothing was submitted.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis;
                ``'lid_open'``, the stage's lid is open; ``'monitor_ended'``,
                the motion monitor has ended. Nothing was driven.
            PositionOutOfRangeError: ``'position_out_of_range'``, an axis
                given is outside the plate positions this stage can reach,
                in plate mm. Nothing was driven.
            AxisStateUnknownError: An axis given does not know its position.
                Nothing was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
        """
        if at.z_um is not None:
            raise ArgumentRefusedError(
                'plate_frame_axis', argument='axis', value='Z', offered=_PLATE_AXES
            )
        targets = tuple(
            (axis, value) for axis, value in (('X', at.x_mm), ('Y', at.y_mm)) if value is not None
        )
        for axis, value in targets:
            refuse_unless_finite_number(value, f'at.{axis.lower()}_mm')
        return self._start_plate_moves('start_move_to_plate', lambda: targets)

    @api
    def move_to_plate(self, at: PlateLocation) -> None:
        """Move X and Y to ``at`` on the selected plate, and return once they have arrived.

        ``start_move_to_plate`` followed by its operation's ``result()``.

        Raises:
            ArgumentRefusedError: ``'plate_frame_axis'`` or
                ``'not_a_number'``, as ``start_move_to_plate``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'lid_open'`` or ``'monitor_ended'``, as
                ``start_move_to_plate``.
            PositionOutOfRangeError: As ``start_move_to_plate``.
            AxisStateUnknownError: As ``start_move_to_plate``.
            MoveFaultError: As ``start_move_to_plate``.
            MoveNotCompletedError: An axis did not arrive; see
                ``start_move_xy``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_move_to_plate``.
                Nothing was driven.
        """
        refuse_waiting_on_ui_thread('move_to_plate')
        self.start_move_to_plate(at).result()

    @api(stop='Operation.stop')
    def start_move_to_well(self, well: str) -> Operation:
        """Start X and Y moving to the centre of ``well`` on the selected plate; return the started moves.

        ``well`` is the label the plate gives it (``'A1'``, ``'H12'``),
        read from the plate selected when the move is on the lane, so the
        plate the well is found on is the plate the move converts on.
        Otherwise as ``start_move_to_plate`` for both axes.

        Raises:
            ArgumentRefusedError: ``'well_unknown'``, the selected plate has
                no well of that label (a plate with no wells has none).
                Nothing was driven.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'lid_open'`` or ``'monitor_ended'``, as
                ``start_move_to_plate``.
            PositionOutOfRangeError: As ``start_move_to_plate``.
            AxisStateUnknownError: As ``start_move_to_plate``.
            MoveFaultError: As ``start_move_to_plate``.
        """

        def targets() -> tuple[tuple[str, float], ...]:
            plate = self._scope.runtime_state.get_labware()
            for x_mm, y_mm, label in plate.get_positions_with_labels():
                if label == well:
                    return (('X', x_mm), ('Y', y_mm))
            raise ArgumentRefusedError('well_unknown', argument='well', value=well, kind=plate.name)

        return self._start_plate_moves('start_move_to_well', targets)

    @api
    def move_to_well(self, well: str) -> None:
        """Move X and Y to the centre of ``well`` on the selected plate, and return once they have arrived.

        ``start_move_to_well`` followed by its operation's ``result()``.

        Raises:
            ArgumentRefusedError: ``'well_unknown'``, as ``start_move_to_well``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'lid_open'`` or ``'monitor_ended'``, as
                ``start_move_to_plate``.
            PositionOutOfRangeError: As ``start_move_to_plate``.
            AxisStateUnknownError: As ``start_move_to_plate``.
            MoveFaultError: As ``start_move_to_plate``.
            MoveNotCompletedError: An axis did not arrive; see
                ``start_move_xy``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_move_to_well``.
                Nothing was driven.
        """
        refuse_waiting_on_ui_thread('move_to_well')
        self.start_move_to_well(well).result()

    def _start_plate_moves(
        self, member: str, targets: Callable[[], tuple[tuple[str, float], ...]]
    ) -> Operation:
        """Put the plate moves *targets* names on one IO-lane task; the operation of the moves.

        *targets* is read on the lane, with the plate and offset the
        conversion uses.
        """
        return self._dispatch_motion(
            self._plate_moves_on_lane,
            member,
            args=(member, targets),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    def _plate_moves_on_lane(
        self, member: str, targets: Callable[[], tuple[tuple[str, float], ...]]
    ) -> Operation:
        """The lane half of a plate move: every axis converted and checked, then one XY request."""
        given = targets()
        for axis, _ in given:
            self.refuse_absent(member, axis)
        # Every target is converted before any axis moves, so a refusal of
        # one leaves the other where it was.
        stage = {axis: self._plate_target_to_stage(axis, mm) for axis, mm in given}
        with self.one_command(tuple(stage), then='move on the plate'):
            return self._move_xy_impl(member, stage.get('X'), stage.get('Y'), False)

    @api(stop='Operation.stop', in_process_arguments=('ignore_limits',))
    def start_move_xy(
        self,
        x_um: float | None = None,
        y_um: float | None = None,
        ignore_limits: bool = False,
    ) -> Operation:
        """Start X and Y moving to a place, in um, each that is given; return the started moves.

        One request: every axis given is refused before either drives, so
        a refusal of one moves neither. An axis that is None is left as it
        is, still, travelling or unknown, and is not read. With neither
        given nothing moves and the move is done at once.

        ``ignore_limits`` drives past the configured travel, never past the
        motor's safety limit. It is in-process only: a wire request that
        sends it is refused as ``invalid_request``.

        Returns:
            One ``Operation`` for the axes given, done once every one is
            (``Operation.all_of``); its ``result()`` raises the first
            axis's raise, as ``start_move_z``'s does for Z. Its ``stop()``
            is ``stop_motion`` while a move still has its axis.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, an axis given is not a
                finite number. Nothing was driven.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis;
                ``'monitor_ended'``, the motion monitor has ended, so
                nothing would judge the move. Nothing was driven.
                ``'lid_open'``, the stage's lid is open: nothing was driven
                when it was open before the request; when it opened after X
                was commanded, the stage stopped X and Y was not driven.
            PositionOutOfRangeError: ``'position_out_of_range'``, an axis
                given is outside its configured travel and ``ignore_limits``
                is False. Nothing was driven.
            AxisStateUnknownError: An axis given does not know its position.
                Nothing was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
        """
        return self._dispatch_motion(
            self._move_xy_impl,
            'start_move_xy',
            args=('move_xy', x_um, y_um, ignore_limits),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    @api(in_process_arguments=('ignore_limits',))
    def move_xy(
        self,
        x_um: float | None = None,
        y_um: float | None = None,
        ignore_limits: bool = False,
    ) -> None:
        """Move X and Y to a place, in um, each that is given, and return once they have arrived.

        ``start_move_xy`` followed by its operation's ``result()``.
        ``ignore_limits`` is in-process only, as ``start_move_xy``'s.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, as ``start_move_xy``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'monitor_ended'`` or ``'lid_open'``, as
                ``start_move_xy``.
            PositionOutOfRangeError: ``'position_out_of_range'``, as ``start_move_xy``.
            AxisStateUnknownError: As ``start_move_xy``.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
            MoveNotCompletedError: An axis did not arrive; see
                ``start_move_z``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_move_xy``. Nothing was
                driven.
        """
        refuse_waiting_on_ui_thread('move_xy')
        self._dispatch_motion(
            self._move_xy_impl,
            'move_xy',
            args=('move_xy', x_um, y_um, ignore_limits),
            timeout_s=self._MOTION_WAIT_BASE_S,
        ).result()

    @api(stop='Operation.stop')
    def start_move_xy_by(self, dx_um: float | None = None, dy_um: float | None = None) -> Operation:
        """Start X and Y moving by a distance, in um, each that is given; return the started moves.

        As ``start_move_xy``. Each distance is added to the axis's board
        target, so a move still under way on it is added to, and every
        target is read and refused before either axis drives.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, a distance given is not
                a finite number. Nothing was driven.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis;
                ``'position_unread'``, the board did not report a target to
                add to; ``'monitor_ended'``, as ``start_move_xy``. Nothing
                was driven. ``'lid_open'``, as ``start_move_xy``.
            PositionOutOfRangeError: ``'position_out_of_range'``, a distance
                is beyond the safety limit or its target is outside the
                axis's travel. Nothing was driven.
            AxisStateUnknownError: An axis given does not know its position.
                Nothing was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
        """
        return self._dispatch_motion(
            self._move_xy_by_impl,
            'start_move_xy_by',
            args=(dx_um, dy_um),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    @api
    def move_xy_by(self, dx_um: float | None = None, dy_um: float | None = None) -> None:
        """Move X and Y by a distance, in um, each that is given, and return once they have arrived.

        ``start_move_xy_by`` followed by its operation's ``result()``.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, as
                ``start_move_xy_by``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'position_unread'``,
                ``'monitor_ended'`` or ``'lid_open'``, as
                ``start_move_xy_by``.
            PositionOutOfRangeError: ``'position_out_of_range'``, as ``start_move_xy_by``.
            AxisStateUnknownError: As ``start_move_xy_by``.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
            MoveNotCompletedError: An axis did not arrive; see
                ``start_move_z``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_move_xy_by``. Nothing
                was driven.
        """
        refuse_waiting_on_ui_thread('move_xy_by')
        self._dispatch_motion(
            self._move_xy_by_impl,
            'move_xy_by',
            args=(dx_um, dy_um),
            timeout_s=self._MOTION_WAIT_BASE_S,
        ).result()

    @api(stop='Operation.stop', in_process_arguments=('ignore_limits',))
    def start_move_z(
        self,
        z_um: float,
        overshoot_enabled: bool | None = None,
        ignore_limits: bool = False,
    ) -> Operation:
        """Start Z moving to a place, in um; return the started move.

        A move of Z to a place approaches it from below, so the backlash
        is always taken the same way. ``overshoot_enabled`` is for a bench
        script that measures the backlash: None is the API's approach,
        True or False the script's own.

        ``ignore_limits`` drives past Z's configured travel, never past the
        motor's safety limit. It is in-process only: a wire request that
        sends it is refused as ``invalid_request``.

        Returns:
            The move's ``Operation``. It is done once Z has been set at
            rest, or given up, and the position listeners told: its
            ``result()`` returns None when Z arrived, and raises
            ``MoveNotCompletedError`` otherwise -- the monitor's fault
            (``'stalled'``, ``'board_lost'``, ``'position_unread'`` or
            ``'status_unread'``) or ``'faulted'``, each leaving Z UNKNOWN;
            ``MoveStoppedError`` ``'stopped'``, a stop the board took while
            it moved; ``MoveStoppedError`` ``'superseded'``, another move or
            a home took Z first. The outcome is the one the move ended
            with, however late it is read. Its ``stop()`` is
            ``stop_motion`` while this move still has Z, and nothing once
            it has ended; the board's STOP halts every axis.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the position is not a
                finite number. Nothing was driven.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no Z;
                ``'position_unread'``, an approach from below whose start
                the board did not report; ``'monitor_ended'``, the motion
                monitor has ended, so nothing would judge the move. Nothing
                was driven.
            PositionOutOfRangeError: ``'position_out_of_range'``, the target
                is outside Z's configured travel and ``ignore_limits`` is
                False. Nothing was driven.
            AxisStateUnknownError: Z does not know its position. Nothing
                was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take the
                command.
        """
        return self._dispatch_motion(
            self._move_z_impl,
            'start_move_z',
            args=(z_um, overshoot_enabled, ignore_limits),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    @api(in_process_arguments=('ignore_limits',))
    def move_z(
        self,
        z_um: float,
        overshoot_enabled: bool | None = None,
        ignore_limits: bool = False,
    ) -> None:
        """Move Z to a place, in um, and return once it has arrived.

        ``start_move_z`` followed by its operation's ``result()``.
        ``ignore_limits`` is in-process only, as ``start_move_z``'s.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, as ``start_move_z``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'position_unread'`` or
                ``'monitor_ended'``, as ``start_move_z``.
            PositionOutOfRangeError: ``'position_out_of_range'``, as ``start_move_z``.
            AxisStateUnknownError: As ``start_move_z``.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
            MoveNotCompletedError: Z did not arrive; see ``start_move_z``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_move_z``. Nothing was
                driven.
        """
        refuse_waiting_on_ui_thread('move_z')
        self._dispatch_motion(
            self._move_z_impl,
            'move_z',
            args=(z_um, overshoot_enabled, ignore_limits),
            timeout_s=self._MOTION_WAIT_BASE_S,
        ).result()

    @api(stop='Operation.stop')
    def start_move_z_by(self, dz_um: float, overshoot_enabled: bool | None = None) -> Operation:
        """Start Z moving by a distance, in um; return the started move.

        The distance is added to Z's board target, so a move still under
        way is added to. A move by a distance takes no approach from below;
        ``overshoot_enabled`` is the bench script's override, as for
        ``start_move_z``.

        Returns:
            The move's ``Operation``, as ``start_move_z``'s.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the distance is not a
                finite number. Nothing was driven.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no Z;
                ``'position_unread'``, the board did not report the target
                to add to (or, approaching from below, the position);
                ``'monitor_ended'``, as ``start_move_z``. Nothing was driven.
            PositionOutOfRangeError: ``'position_out_of_range'``, the
                distance is beyond the safety limit or its target is outside
                Z's travel. Nothing was driven.
            AxisStateUnknownError: Z does not know its position. Nothing
                was driven.
            MoveFaultError: ``'driver_failed'``, the board did not take the
                command.
        """
        return self._dispatch_motion(
            self._move_z_by_impl,
            'start_move_z_by',
            args=(dz_um, overshoot_enabled),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    @api
    def move_z_by(self, dz_um: float, overshoot_enabled: bool | None = None) -> None:
        """Move Z by a distance, in um, and return once it has arrived.

        ``start_move_z_by`` followed by its operation's ``result()``.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, as
                ``start_move_z_by``.
            HardwareCommandRefusedError: ``'not_connected'``,
                ``'axis_absent'``, ``'position_unread'`` or
                ``'monitor_ended'``, as ``start_move_z_by``.
            PositionOutOfRangeError: ``'position_out_of_range'``, as ``start_move_z_by``.
            AxisStateUnknownError: As ``start_move_z_by``.
            MoveFaultError: ``'driver_failed'``, the board did not take a
                command.
            MoveNotCompletedError: Z did not arrive; see ``start_move_z``.
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on
                the UI thread, whose form is ``start_move_z_by``. Nothing
                was driven.
        """
        refuse_waiting_on_ui_thread('move_z_by')
        self._dispatch_motion(
            self._move_z_by_impl,
            'move_z_by',
            args=(dz_um, overshoot_enabled),
            timeout_s=self._MOTION_WAIT_BASE_S,
        ).result()

    @api
    def home(self, axis: HomeTarget = HomeTarget.ALL) -> None:
        """Home the given axis set, and wait for it.

        Returns only when the home established a reference: every homed
        axis has a known position (for ``'ALL'``, the axes the board has).
        The API sets no time limit on the wait: the home ends when the board
        ends it, which bounds each of its phases and halts when one runs out,
        or when the driver stops waiting for a board that has stopped
        answering; either ends it in ``HomingFaultError``.

        Args:
            axis: ``'Z'`` homes the Z axis only. ``'T'`` homes the turret
                (parks Z at 0, homes T, restores Z). ``'ALL'`` (default)
                homes every axis the board has; the firmware routine homes
                Z, then T, then X/Y.

        Raises:
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on the
                UI thread; nothing was started (``start_home`` is the form
                for a button).
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is not
                ``'Z'``, ``'T'`` or ``'ALL'``; nothing was asked of the scope.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller, or no Z or turret to
                home (see ``refuse_absent``); ``'exclusive_activity_running'``,
                a run, a diagnostic or a recording holds the scope; or
                ``'lid_open'``, the stage's lid is open, and nothing moves.
            HomingFaultError: the home was driven and did not establish
                a reference: the driver raised, or a homed axis's position
                could not be read. The axes it names are UNKNOWN.
            HomingStoppedError: a Stop ended the home. The axes it names
                are UNKNOWN; none, when the Stop landed before the home was
                driven.
            HardwareCommandRefusedError: ``'home_in_flight'``, while a home
                asked earlier, by any caller, has not ended.

        While it runs, the home holds the session's activity claim as a run
        does: every other request to the scope is refused, naming the home,
        and the controls lock until it ends. Asked by an activity that
        already holds the scope (the support report's diagnostic), it runs
        as that activity's work and takes no claim of its own.
        """
        refuse_waiting_on_ui_thread('home')
        impl = self._home_body(axis)
        body, release_if_unrun, _stop, taking = self.claim_home(impl)
        try:
            with acting(taking):
                self._dispatch_motion(
                    body,
                    'home',
                    timeout_s=None,
                    falsifies_recording=self._home_moves_turret(impl),
                )
        except BaseException:
            release_if_unrun()
            raise

    @api(stop='Operation.stop')
    def start_home(self, axis: HomeTarget = HomeTarget.ALL) -> Operation:
        """Start the home ``home`` runs, without waiting for it; returns its ``Operation``.

        For a caller that must not block for the length of a home -- a
        button. The home is asked, and refused or put on the io lane, before
        this returns, so a second home asked while this one is in flight is
        refused at once rather than queued behind it. The operation is done
        once the home has released the scope, with what ``home`` would have
        returned or raised; its ``stop()`` is ``stop_motion`` while the home
        drives, refused ``'home_not_stoppable'`` on a board that cannot stop
        a home, and nothing once the home has ended.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is not
                ``'Z'``, ``'T'`` or ``'ALL'``; nothing was asked of the scope.
            HardwareCommandRefusedError: ``'home_in_flight'``, while a home
                asked earlier has not ended; or the home was refused as
                ``home`` is refused.
        """
        impl = self._home_body(axis)
        body, release_if_unrun, stop, taking = self.claim_home(impl)
        waiter: concurrent.futures.Future[None] = concurrent.futures.Future()
        # Marked running before the lane holds it, so only the lane settles
        # it and nothing cancels it beside the claim.
        waiter.set_running_or_notify_cancel()
        try:
            with acting(taking):
                self._scope._io_executor.submit(
                    IOTask(action=body, falsifies_recording=self._home_moves_turret(impl)),
                    'home',
                    waiter=waiter,
                )
        except BaseException:
            release_if_unrun()
            raise
        return Operation.following(waiter, stop=stop, first=release_if_unrun)

    def _home_body(self, axis: HomeTarget) -> Callable[[], None]:
        """The home body for ``axis``."""
        if axis == HomeTarget.Z:
            return self._zhome_impl
        if axis == HomeTarget.T:
            return self._home_turret_impl
        return self._home_impl

    def claim_home(
        self, impl: Callable[[], None]
    ) -> tuple[Callable[[], None], Callable[[], None], Callable[[], None], Taking | None]:
        """Take the scope for a home of ``impl``, or refuse this one.

        ``impl`` is a home, or the Session's startup sequence, which holds the
        scope as one home from its first move to its last.

        Returns the lane body, which runs ``impl`` and releases the claim when
        it ends; the release for a body the lane never ran: refused at
        admission, refused while queued, or dropped (a body that started
        releases its own, so the second is a no-op then); the home's stop,
        ``stop_motion`` until the body has ended and nothing after, so a
        stop asked as the home lets go never reaches motion begun after it;
        and the taking the home is submitted under, so the lane runs it
        while the claim holds.

        Asked under a live activity's taking, the home is that activity's
        work: it is submitted under that taking and takes nothing. Asked by a
        control session's holder, it is the holder's own home, and takes the
        scope as any home does. On a scope no session composes there is no
        claim, and the home takes nothing either.

        Raises:
            HardwareCommandRefusedError: ``'home_in_flight'`` while another
                home holds the scope; ``'exclusive_activity_running'``,
                naming the holder, while anything else does;
                ``'control_session_held'``, naming the session, while one is
                held by someone else.
        """
        current = current_taking()
        activity = activity_taking(current)
        claim = self._scope._activity_claim
        held = None
        if claim is not None and (activity is None or not activity.holds):
            try:
                held = claim.try_claim(WorkKind.HOME)
            except ControlSessionHeldError as held_by:
                raise command_refused(held_by.session, 'home') from None
            if held is None:
                holder = claim.holder
                if holder is not None and holder.kind == WorkKind.HOME:
                    raise HardwareCommandRefusedError('home_in_flight', 'home')
                raise HardwareCommandRefusedError(
                    'exclusive_activity_running', 'home', holder.kind if holder else None
                )
        started = threading.Event()
        ended = threading.Event()

        def release() -> None:
            if held is not None:
                held.release()

        # The lane names a task by its action and reads the cost it declared
        # (``@slow_task_budget``) from it, so the body carries both from the home.
        @functools.wraps(impl)
        def body() -> None:
            started.set()
            try:
                impl()
            finally:
                # Under the lock the home's stop asks it in, before the claim
                # lets anything else take the scope.
                with self._stop_lock:
                    ended.set()
                release()

        def release_if_unrun() -> None:
            if not started.is_set():
                release()

        def stop() -> None:
            self._stop_motion_while(lambda: not ended.is_set())

        return body, release_if_unrun, stop, held if held is not None else current

    @api
    def move_turret(self, slot: int, restore_z: bool = True) -> None:
        """Move the turret to a slot, and wait for it. See ``_move_turret_impl``.

        Three moves -- the Z park, the turret move itself and the Z restore
        -- each judged as every move is, so the wait sets no time limit of
        its own.

        Raises:
            BlockingCallRefusedError: ``'wait_on_ui_thread'``, called on the
                UI thread; nothing was started.
            HardwareCommandRefusedError: a recording holds the scope: its
                frames carry the pixel size of the objective it started with.
        """
        refuse_waiting_on_ui_thread('move_turret')
        return self._dispatch_motion(
            self._move_turret_impl,
            'move_turret',
            args=(slot,),
            kwargs={'restore_z': restore_z},
            timeout_s=None,
            falsifies_recording=True,
        )

    @api
    def wait_until_finished_moving(self, timeout_s: float = 120.0) -> None:
        """Block until every axis moving now has stopped; raise if one did not stop well.

        For motion the caller did not start, or started without keeping its
        operation. A move the caller started is judged by its operation's
        ``result()``, which also knows when a stop or a later move ended it;
        this wait judges only whether each axis stopped with a known
        position. An axis a stop halted is IDLE where it stopped, so after a
        stop this returns.

        The axes it waits for are those MOVING or HOMING when it is called.
        An axis that was not moving is not this wait's business -- one that
        was never homed among them -- so a scope with unhomed X and Y can
        still wait on its Z. Waits on the per-axis arrival events the motion
        monitor sets; no serial I/O from the calling thread.

        Args:
            timeout_s: Maximum seconds to wait for all of them (default 120s).

        Raises:
            MoveFaultError: An axis it waited for ended UNKNOWN (the
                monitor's own object when it gave the axis up: ``'stalled'``,
                ``'board_lost'``, ``'position_unread'`` or ``'status_unread'``;
                else ``'faulted'``).
            MoveStoppedError: An axis it waited for ended UNKNOWN after a
                STOP ended its move, as the close's does: ``'stopped'``, or
                ``'superseded'``.
            MoveWaitExpiredError: ``'still_moving'``, a ``TimeoutError``: an
                axis was still moving when ``timeout_s`` ran out; its state
                is left to its own move.
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                ``timeout_s`` is not a wait from 0 to 30 days: a NaN deadline
                would end the wait at once, and one longer than the host can
                hold would raise out of it.
        """
        refuse_unless_wait_s(timeout_s, 'timeout_s')
        with self._axis_state_lock:
            moving = [
                ax
                for ax, state in self._axis_state.items()
                if state in (AxisState.MOVING, AxisState.HOMING)
            ]
        deadline = time.monotonic() + timeout_s
        for ax in moving:
            if not self._wait_for_axis_to_stop(ax, max(0.0, deadline - time.monotonic())):
                raise MoveWaitExpiredError(ax, 'still_moving')
        for ax in moving:
            with self._axis_state_lock:
                unknown = self._axis_state.get(ax) == AxisState.UNKNOWN
                move = self._current_move.get(ax)
            ended = move.error() if move is not None else None
            if unknown:
                # A move that arrived before the axis was found UNKNOWN, or
                # none (a home's), still did not stop with a known position.
                raise ended if ended is not None else MoveFaultError(ax, 'faulted')

    def _set_axis_state(
        self,
        axis: str,
        state: str,
        *,
        verdict_for: _Move | None = None,
        armed_only: bool = False,
        move: _Move | None = None,
        fault: MoveFaultError | None = None,
        stop_generation: int | None = None,
        moving_seen_ns: int | None = None,
    ) -> bool:
        """Set the state of an axis: the one writer of it.

        The state, the axis's current move, and the arrival event (cleared
        for MOVING and HOMING, set for IDLE and UNKNOWN so waiters unblock)
        are written in one hold of ``_axis_state_lock``, so a verdict can
        never land between a state and its event. Fires position listeners
        after every write.

        A verdict -- the monitor's IDLE or its stall, the waiter's timed-out
        UNKNOWN -- passes ``verdict_for``, the move it judges: the write
        happens only if the axis is still MOVING with that move and, with
        ``armed_only``, that move is armed. Otherwise the axis has moved on
        (a later move or a home owns it, or a drive is in flight) and
        nothing is written.

        The axis's current move is settled in the same hold as the write
        that ends it (see ``_Move``), so no exit from MOVING leaves it open:
        IDLE settles it arrived, or 'stopped' when ``stop_generation`` -- the
        generation the monitor read once the board said reached -- is not
        the one the move began under; UNKNOWN settles it with ``fault``, or
        with none 'stopped' when a STOP was taken since the move began, else
        'faulted'; MOVING or HOMING settles it 'superseded', or 'stopped'
        when a STOP was taken since the move began. MOVING is
        written only with the new ``move`` (``_begin_move``), which becomes
        the current one; HOMING leaves none. The settled move's operation is
        done after the hold is released and the listeners are told
        (``_Move.publish``).

        An IDLE or UNKNOWN write stamps the axis's change in force
        (``FrameValidity.motion_settled``) with ``moving_seen_ns``, when the
        monitor's last read that found the axis still moving began; None
        from every writer that read no such thing. While a command is open
        the stamp waits for its close, which is given the latest such read.

        Returns:
            bool: Whether the write happened. False for an axis that is not
            present on this hardware: per-axis dicts are sized to
            detect_present_axes() at init, so hardcoded callers like the
            turret-home path (T) automatically degrade to no-ops on scopes
            that lack those axes.
        """
        if (state == AxisState.MOVING) != (move is not None):
            raise ValueError('MOVING is written with its new move, and only MOVING (_begin_move)')
        if axis not in self._arrival_events:
            return False
        with self._axis_state_lock:
            old_state = self._axis_state.get(axis, AxisState.UNKNOWN)
            if verdict_for is not None and (
                old_state != AxisState.MOVING
                or self._current_move.get(axis) is not verdict_for
                or (armed_only and not verdict_for.armed)
            ):
                return False
            current = self._current_move.get(axis)
            ended = None
            if old_state == AxisState.MOVING and current is not None:
                if state == AxisState.IDLE:
                    stopped = stop_generation is not None and (
                        stop_generation != current.stop_generation
                    )
                    settled = current.settle('stopped' if stopped else 'arrived')
                elif (
                    state == AxisState.UNKNOWN
                    and fault is None
                    and (self._stop_generation != current.stop_generation)
                ):
                    # An UNKNOWN with no fault of its own after a STOP, as the
                    # close writes after its STOP: the STOP ended the move.
                    settled = current.settle('stopped')
                elif state == AxisState.UNKNOWN:
                    settled = current.settle('faulted', fault or MoveFaultError(axis, 'faulted'))
                else:
                    # A STOP the board took before this write halted the
                    # move, though the monitor has not yet seen it at rest.
                    stopped = self._stop_generation != current.stop_generation
                    settled = current.settle('stopped' if stopped else 'superseded')
                if settled:
                    ended = current
            self._axis_state[axis] = state
            # One place for every route that loses the turret -- a fault, a
            # failed home, a stall, a lost board: a turret whose position is
            # unknown is in no known slot.
            if axis == 'T' and state == AxisState.UNKNOWN:
                self._last_turret_position = None
            if state in (AxisState.MOVING, AxisState.HOMING):
                self._current_move[axis] = move
                # Clear arrival event -- axis is now in motion
                self._arrival_events[axis].clear()
            elif state in (AxisState.IDLE, AxisState.UNKNOWN):
                # Signal arrival -- unblocks any wait_for_axis() callers. IDLE
                # (arrived) and UNKNOWN (no longer moving, position
                # indeterminate -- e.g. a failed home or a disconnect
                # mid-move) are both terminal not-in-motion states; a waiter
                # must unblock rather than hang on a cleared event until the
                # 120s motion timeout.
                self._arrival_events[axis].set()
                # Taken in the hold, so it is the moment the settle check
                # began to see this state, not when the stamp lands.
                settled_ns = host_clock.steady_ns()
        # The ended move's operation is the only thing that ends its waiters,
        # so it settles on every exit from here.
        try:
            if profile_trace.ENABLE_PROFILE_TRACE and old_state != state:
                profile_trace.trace(
                    'motion_trace.csv',
                    'ts_ms,duration_ms,event,axis,detail',
                    [int(time.time() * 1000), 0, 'transition', axis, f'{old_state}->{state}'],
                    recording_id=profile_trace.NO_RECORDING,
                )

            if state in (AxisState.MOVING, AxisState.HOMING):
                # Wake the motion monitor to start polling
                self._motion_wake.set()
            else:
                # Inside a command this arrival stamps nothing; its close
                # does, and needs the read.
                if moving_seen_ns is not None:
                    with self._commands_lock:
                        if self._commands_open and (
                            self._command_moving_seen_ns is None
                            or moving_seen_ns > self._command_moving_seen_ns
                        ):
                            self._command_moving_seen_ns = moving_seen_ns
                self._scope.frame_validity.motion_settled(
                    self._AXIS_VALIDITY_SOURCE.get(axis, FrameValiditySource.XY_MOVE),
                    settled_ns,
                    moving_seen_ns=moving_seen_ns,
                )

            self._fire_position_listeners(axis)
        finally:
            if ended is not None:
                ended.publish()
        return True

    def _backlash_leg(
        self, axis: str, position: float, overshoot_enabled: bool, member: str
    ) -> float | None:
        """Where a move's backlash leg goes, or None when it has none.

        On a board with a backlash, a Z move down to a target clear of the
        bottom first drives to the backlash below it, so the backlash is always taken the same way.
        Decided from the board's position, before the move disarms the
        axis: a refusal here leaves the axis as it was.

        Raises:
            HardwareCommandRefusedError: ``'position_unread'``, the board
                did not report Z, so whether to approach from below is not
                known; nothing was driven.
        """
        if not (overshoot_enabled and axis == 'Z'):
            return None
        try:
            current = self._driver.current_pos('Z')
        except HardwareError as e:
            raise HardwareCommandRefusedError('position_unread', member) from e
        backlash = self._driver.backlash_um()
        # A board with no backlash has no approach leg.
        if backlash > 0 and current > position and position > backlash + 50:
            return position - backlash
        return None

    def _drive_to(
        self, axis: str, position: float, leg: float | None, stop_generation: int
    ) -> bool:
        """Drive ``axis`` to ``position``, through the backlash leg first when there is one.

        Runs inside ``_send_drive``, with the axis MOVING and disarmed: the
        leg is part of the move, and its arrival is nobody's verdict.
        Each target goes out through ``_write_unless_stopped``, so a STOP
        that lands during the move, the leg included, ends it there.

        Returns:
            bool: Whether the target was written; False when a stop withheld it.

        Raises:
            HardwareError: the board did not answer a target write, or the
                leg did not reach its point within ``OVERSHOOT_LEG_TIMEOUT_S``.
        """
        if leg is not None:
            if not self._write_unless_stopped(axis, leg, stop_generation):
                return False
            deadline = time.monotonic() + OVERSHOOT_LEG_TIMEOUT_S
            while not self._driver.target_status(axis):
                if time.monotonic() > deadline:
                    raise MotionNotCompletedError(
                        f'move {axis} to {position}: the overshoot leg did not reach '
                        f'its point within {OVERSHOOT_LEG_TIMEOUT_S:.0f} s'
                    )
                time.sleep(self._MOTION_POLL_INTERVAL)
        return self._write_unless_stopped(axis, position, stop_generation)

    def _write_unless_stopped(self, axis: str, position: float, stop_generation: int) -> bool:
        """Write ``axis``'s target unless a stop landed since the move read ``stop_generation``.

        Under the lock ``stop_motion`` holds across its exchange, so a STOP
        lands wholly before this write, which is then withheld, or wholly
        after it, and stops it. Without the lock a STOP between the check
        and the write moved the stage after the stop. The generation is
        compared directly: ``_stopped_since`` takes the same lock, which is
        not reentrant.

        Returns:
            bool: Whether the target was written.
        """
        with self._stop_lock:
            if self._stop_generation != stop_generation:
                return False
            self._driver.move_abs_pos(axis, position)
            return True

    def _begin_move(self, axis: str, stop_generation: int) -> _Move:
        """Set ``axis`` MOVING with a new move: the one way an axis goes MOVING.

        Args:
            axis: The axis to drive.
            stop_generation: ``_stop_generation`` as the move's body read it
                before driving.

        Returns:
            The new move's record, settled when the move ends.
        """
        move = _Move(axis, stop_generation, lambda: self._stop_move(move))
        self._set_axis_state(axis, AxisState.MOVING, move=move)
        return move

    def _stop_move(self, move: _Move) -> None:
        """A started move's stop: ``stop_motion`` while ``move`` still has its axis, else nothing.

        A move that has ended -- arrived, stopped, or its axis taken by a
        later move or a home -- may still be telling its listeners before
        its operation is done; a stop asked of it then must not halt the
        motion that took its axis.
        """

        def live() -> bool:
            with self._axis_state_lock:
                return self._current_move.get(move.axis) is move and move.outcome is None

        self._stop_motion_while(live)

    def _send_drive(self, axis: str, stop_generation: int, send, member: str) -> tuple[_Move, bool]:
        """Set ``axis`` MOVING and disarmed, then send its drive.

        The one way a move body reaches the driver. The move is motion from
        its first target write, so every reader -- the waits, the state,
        the positions, frame validity -- sees it from here, a Z backlash
        leg included. Until ``_publish_drive`` arms the axis, the board's
        reached bit is the previous target's, or the leg's, and the monitor
        writes nothing for the axis: it polls and refreshes the position,
        and runs no stall clock. A raise makes the axis UNKNOWN
        (``_fail_drive``), or, for the interlock's refusal before anything
        moved, gives the axis back the state it had
        (``_refuse_for_interlock``), so no exit ends MOVING and disarmed.

        Returns:
            The move's record, and what ``send`` returned: whether the
            target was written.

        Raises:
            HardwareCommandRefusedError: the stage's interlock refused the
                drive before anything moved; see ``_refuse_for_interlock``.
                ``'monitor_ended'``: the motion monitor has ended, so
                nothing would judge the move; nothing was sent.
            MoveFaultError: the drive raised; see ``_fail_drive``.
        """
        if self._monitor_ended:
            raise HardwareCommandRefusedError('monitor_ended', member)
        state_before = self._axis_states((axis,))[axis]
        move = self._begin_move(axis, stop_generation)
        try:
            self._scope.frame_validity.invalidate(
                self._AXIS_VALIDITY_SOURCE.get(axis, FrameValiditySource.XY_MOVE), began_ns=None
            )
            return move, send()
        except MotionInterlockError as e:
            self._refuse_for_interlock(axis, move, state_before, e, member)
        except Exception as e:
            self._fail_drive(axis, move, e)

    def _publish_drive(self, axis: str, written: bool, target_pos: float | None) -> None:
        """Publish a sent drive's target, then arm ``axis``: its verdicts may land.

        The target is the move's own (``_Move.target``), so it stays the
        move's after the move ends; it is written in the hold that arms the
        axis, so no arrival can land before it. A withheld target is
        published nowhere -- the stage is where the stop left it -- but the
        axis is armed all the same, so the monitor judges it there. An axis
        given up while its drive was sent (a lost board) is left as it is:
        neither target nor arming belongs to it.
        """
        with self._axis_state_lock:
            move = self._current_move.get(axis)
            if self._axis_state.get(axis) != AxisState.MOVING or move is None:
                return
            if written:
                move.target = target_pos
            move.armed = True

    def _give_axis_up(self, axis: str, reason: str, *, verdict_for: _Move | None = None) -> bool:
        """The monitor gives a moving axis up: one fault, reported, then UNKNOWN.

        The fault is reported before the UNKNOWN write that settles the
        move with it, because that write is what wakes a waiter: the waiter
        then raises the object already reported, and the reporter shows an
        object once. Nobody waits on a jog, so the monitor's report is
        its only popup; for a waited move, a report the reporter suppressed
        (an unattended run, the dedup window) leaves it for the waiter's
        caller to show.

        A stall passes ``verdict_for``, the move whose clock ran out: the
        report and the write each happen only while the axis is still that
        move's armed MOVING one. A lost board passes none: a
        lost board is lost whichever move holds the axis.

        Returns:
            bool: Whether the axis was given up.
        """
        fault = MoveFaultError(axis, reason)
        with self._axis_state_lock:
            if verdict_for is not None and (
                self._axis_state.get(axis) != AxisState.MOVING
                or self._current_move.get(axis) is not verdict_for
                or not verdict_for.armed
            ):
                return False
        if reason == 'stalled':
            # The board is still driving the axis at whatever holds it. The
            # halt is the scope's own STOP, which refuses nothing -- the
            # public stop_motion is refused while a home drives, and a stall
            # inside a turret home's Z park must still be halted -- and is
            # not a person's Stop, so a home it ends ends in its fault.
            try:
                self._stop()
            except MotorStopFailedError as halt_failed:
                notifications.report_outcome(halt_failed, solicited=False, category='Motion')
        notifications.report_outcome(fault, solicited=False, category='Motion')
        return self._set_axis_state(
            axis,
            AxisState.UNKNOWN,
            verdict_for=verdict_for,
            armed_only=verdict_for is not None,
            fault=fault,
        )

    def _motion_monitor_loop(self):
        """Background thread: judges every move, and ends leaving none unjudged.

        The loop ends only at the scope's close. Should it end any other
        way -- a defect in its own code -- every axis it was watching is
        given up, ``'monitor_ended'``, before the thread exits with what
        ended it, and every later drive is refused: an axis left MOVING
        with no monitor would hold every waiter, run and sweep forever.
        """
        try:
            self._judge_moves()
        finally:
            if not self._motion_monitor_stop.is_set():
                self._monitor_ended = True
                with self._axis_state_lock:
                    moving = [ax for ax, st in self._axis_state.items() if st == AxisState.MOVING]
                for ax in moving:
                    self._give_axis_up(ax, 'monitor_ended')

    def _judge_moves(self):
        """Poll firmware for axis arrival, ``_MOTION_POLL_INTERVAL`` apart.

        Sleeps on ``_motion_wake`` when all axes are IDLE. Wakes when any
        axis transitions to MOVING. Polls ``_read_target_status()`` per
        MOVING axis -- the board's answer, refused by nothing, so it reads
        through the close until ``_disconnect`` stops it -- and transitions
        them to IDLE on arrival. This is the
        single place where firmware target-status queries happen during
        normal operation -- all other code reads the in-memory axis state.

        A poll is two exchanges per moving axis, so its rate is the
        board's, not the interval's: about 7 polls a second with X and Y
        moving on an LS850T over Windows (2026-10-14). Its listeners are
        told once every moving axis is read, so a reader of
        ``axis_positions`` told of one axis finds the others from the same
        poll; told as each axis was read, the stage drew X's new position
        beside Y's from the poll before, and a diagonal move crossed the
        plate in steps.
        """
        while not self._motion_monitor_stop.is_set():
            # Sleep until something starts moving (or shutdown)
            self._motion_wake.wait()
            if self._motion_monitor_stop.is_set():
                break

            # Poll moving axes until all arrive
            while not self._motion_monitor_stop.is_set():
                moving_axes = []
                with self._axis_state_lock:
                    moving_axes = [
                        ax for ax, st in self._axis_state.items() if st == AxisState.MOVING
                    ]

                # A stall clock left over from a finished move would fault
                # the axis's NEXT move at whatever time the old entry says,
                # so the clock tracks only axes moving RIGHT NOW.
                for ax in list(self._moving_since):
                    if ax not in moving_axes:
                        self._moving_since.pop(ax, None)
                for ax in list(self._moving_seen_ns):
                    if ax not in moving_axes:
                        self._moving_seen_ns.pop(ax, None)

                if not moving_axes:
                    # All axes arrived -- go back to sleep
                    self._motion_wake.clear()
                    break

                # Query firmware for each MOVING axis
                read_this_poll = []
                with profile_trace.timer(
                    'motion_trace.csv',
                    'ts_ms,duration_ms,event,axis,detail',
                    lambda ma=moving_axes: ['poll', ','.join(ma), ''],
                ):
                    for ax in moving_axes:
                        if self._motion_monitor_stop.is_set():
                            break
                        if not self._driver.is_connected():
                            # A board that vanishes mid-move would otherwise
                            # leave the axis MOVING forever -- is_moving() never
                            # clears, so autofocus and the protocol runner wedge
                            # silently. Bound the disconnect: after a short
                            # deadline, fault the axis to a terminal state
                            # (UNKNOWN fires the arrival event so waiters and
                            # is_moving() unblock) and notify the user once.
                            first = self._disconnect_since.setdefault(ax, time.monotonic())
                            if time.monotonic() - first > self._DISCONNECT_FAULT_S:
                                self._disconnect_since.pop(ax, None)
                                self._give_axis_up(ax, 'board_lost')
                            continue
                        # Reconnected (or never lost) before the deadline.
                        self._disconnect_since.pop(ax, None)
                        # Note the move being judged before asking the board:
                        # the verdict is written only if the axis is still
                        # that move's when the answer comes back.
                        with self._axis_state_lock:
                            noted = self._current_move.get(ax)
                            armed = noted is not None and noted.armed
                        # The arrival check: firmware-authoritative via the
                        # position_reached (STATUS_R bit 22) signal. The motor
                        # owns this -- it knows when XACTUAL == XTARGET at the
                        # microstep level, including final-step settling and
                        # the firmware Zstop logic. Asked BEFORE the position
                        # read below, so the position kept on arrival was read
                        # after the exchange that saw it: where the axis
                        # stopped, not where it was one exchange earlier.
                        # None while the board has not said: a failed read
                        # is neither an arrival nor a stall, so the move's
                        # clock runs on and, at the bound, the move is
                        # given up as unread.
                        arrived: bool | None = None
                        read_began_ns = host_clock.steady_ns()
                        try:
                            arrived = self._read_target_status(ax)
                            if arrived is False:
                                self._moving_seen_ns[ax] = read_began_ns
                        except Exception as e:
                            if self._status_unread_warned.get(ax) is not noted:
                                self._status_unread_warned[ax] = noted
                                _api_log.warning(
                                    f'motion monitor: {ax} arrival read failed ({e}); '
                                    f'retrying until the stall window'
                                )
                        # A STOP sets target = actual, so the board then
                        # reports reached wherever the axis halted: a reached
                        # bit is a stop, not an arrival, when a STOP was taken
                        # since the move began. Read under the lock the STOP
                        # holds until its generation moves, so a STOP that
                        # made this bit is counted.
                        if arrived:
                            with self._stop_lock:
                                stop_generation = self._stop_generation
                        # Read the motor's actual position into the cache so
                        # get_current_position (and the crosshair, through the
                        # position listener) tracks the motor instead of a
                        # prediction: a ramp-model predictor ran 5-10x ahead
                        # of the motor. On arrival the
                        # cache holds this read -- the actual motor position,
                        # which may differ from the commanded target by up to
                        # ~1 microstep (X/Y ~0.078 um, Z ~0.025 um) of
                        # quantization -- so it stays honest about where the
                        # motor physically is.
                        # The watch is the move's own, started when the
                        # monitor first sees it armed; the reading it is fed
                        # is the microstep count, since the turret's position
                        # is its slot number and holds still for most of a
                        # turn.
                        watched = None
                        if armed:
                            watched = self._moving_since.get(ax)
                            if watched is None or watched[0] is not noted:
                                watched = (noted, StallWatch(self._MOVE_STALL_WINDOW_S))
                                watched[1].stalled(_NOT_YET_READ, time.monotonic())
                                self._moving_since[ax] = watched
                        try:
                            steps = self._driver.current_pos_steps(ax)
                            actual = self._driver.steps_to_position(ax, steps)
                            with self._pos_cache_lock:
                                self._pos_cache[ax] = float(actual)
                            read = True
                            if watched is not None:
                                watched[1].stalled(steps, time.monotonic())
                        except HardwareError as e:
                            if self._unread_warned.get(ax) is not noted:
                                self._unread_warned[ax] = noted
                                _api_log.warning(
                                    f'motion monitor: {ax} position read failed ({e}); '
                                    f'retrying until the stall window'
                                )
                            read = False
                        # An arrival is written only with the position read
                        # after it: IDLE with an unread cache was an axis
                        # "known" at a number nobody read.
                        if (
                            arrived
                            and read
                            and armed
                            and self._set_axis_state(
                                ax,
                                AxisState.IDLE,
                                verdict_for=noted,
                                armed_only=True,
                                stop_generation=stop_generation,
                                moving_seen_ns=self._moving_seen_ns.pop(ax, None),
                            )
                        ):
                            # Its clock goes with it: a back-to-back move
                            # landing within one poll interval starts its own.
                            self._moving_since.pop(ax, None)
                            continue
                        # Still moving per firmware. A connected axis whose
                        # reading has held one value for the stall window
                        # is stalled: position_reached will never fire, so
                        # nothing else can ever clear it. Halt it, fault it
                        # to UNKNOWN (terminal; fires the arrival event so
                        # waiters and state-readers unblock, and the
                        # settle-check treats UNKNOWN as settled) and tell
                        # the user once -- the same shape as the disconnect
                        # fault. The verdict lives HERE, after the arrival
                        # check, so an arriving report always wins over the
                        # stall. An axis the board says arrived but whose
                        # position it will not report, or whose arrival it
                        # will not report, feeds the watch nothing, and is
                        # given up as that, not as a stall.
                        if (
                            watched is not None
                            and not (arrived and read)
                            and watched[1].quiet_for_s(time.monotonic())
                            >= self._MOVE_STALL_WINDOW_S
                        ):
                            self._moving_since.pop(ax, None)
                            reason = (
                                'status_unread'
                                if arrived is None
                                else 'position_unread'
                                if arrived
                                else 'stalled'
                            )
                            if self._give_axis_up(ax, reason, verdict_for=noted):
                                continue
                        # No verdict this poll (still moving, a drive in
                        # flight, or the reached bit was another move's):
                        # propagate the refreshed cache value to UI listeners.
                        read_this_poll.append(ax)

                for ax in read_this_poll:
                    self._fire_position_listeners(ax)
                time.sleep(self._MOTION_POLL_INTERVAL)
