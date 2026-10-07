# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""MotionAPI -- sub-API for stage / focus / turret motion.

MotionAPI owns the motion state slots (_pos_cache, _axis_state,
_arrival_events, _position_listeners, _motion_wake,
_motion_monitor_stop, _motion_monitor_thread, _homing_event,
_turreting_event) and the bodies of all stage / focus / turret
methods. Lumascope keeps a small set of one-line method-name
forwarders (home, move_absolute, etc.) for
production callers; those retire as production migrates.

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
method list and docs/WAVE7_PHASE_2_PLAN.md for the multi-commit plan.
"""

from __future__ import annotations

import contextlib
import logging as _logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar, NoReturn
from collections.abc import Iterable, Iterator, Mapping

from drivers.exceptions import HardwareError
from drivers.null_motorboard import NullMotionBoard
from lib import profile_trace
from lvp_logger import logger
from modules.exceptions import (
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    HomingFailedError,
    MissingPart,
    MotorStopFailedError,
    MoveNotCompletedError,
    PositionOutOfRangeError,
)
from modules.notification_center import notifications
from modules.sequential_io_executor import IOTask, slow_task_budget

# Declared costs, module-level because the @slow_task_budget decorators run at
# class-body time and cannot reach a class attribute defined further down.
#
# Homing legitimately takes 10-60+ seconds depending on travel distance and
# starting position. Deliberately NOT _MOTION_SETTLE_TIMEOUT_S even though both
# are 120: that one is how long ONE motion may take to settle, this is how long
# the homing command may run before the elapsed-time warning means anything.
# Two facts that happen to share a number drift apart the moment one is tuned.
_HOMING_SLOW_TASK_S = 120.0

# A turret move is three physically-waited motions (Z park, turret, Z restore),
# so it runs past a threshold written for one and warned on every successful
# move. Measured 5.2s on an LS850T between adjacent positions; this is ~3x
# that, the "unusual" bound rather than the hung bound. Budget row:
# Firmware docs/PERFORMANCE_BUDGETS.md, turret move.
_TURRET_MOVE_SLOW_TASK_S = 15.0

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

# Match _lumascope.py's module-level _api_log channel so relocated
# bodies log to the same handler chain.
_api_log = _logging.getLogger('LVP.api')

from modules.lumascope_api._constants import (
    AxisPosition,
    AxisState,
    MOTOR_POSITION_LIMIT,
    TURRET_SLOT_MAX,
    TURRET_SLOT_MIN,
    _VALID_AXIS_NAMES,
    is_turret_slot,
    refuse_acceleration_pct,
)

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import MotorBoardProtocol


class _Move:
    """One move's outcome, fixed when the move ends.

    Created in the hold that sets its axis MOVING, and settled in the hold
    that ends it -- the monitor's IDLE, an UNKNOWN, or another move or a
    home taking the axis -- so what later happens to the axis is never
    read as this move's outcome. Settled once: the first ending wins.

    Attributes:
        stop_generation: ``MotionAPI._stop_generation`` as the move's body
            read it before driving; a different one at the move's IDLE
            means a STOP the board took ended it.
        armed: Whether the move's final target is known written, so the
            board's reached bit is this move's: set by ``_publish_drive``.
            Until then a reached bit is the previous target's, or the
            backlash leg's, and the monitor writes no verdict for the move.
        done: Set when the move is settled; the fields below are read only
            after it.
        outcome: ``'arrived'``, ``'stopped'``, ``'superseded'`` or
            ``'faulted'``; None until settled.
        fault: The error an UNKNOWN ending carried, or None.
        target: The final target the move wrote to the board, set by
            ``_publish_drive``; None until written, and for good when a stop
            withheld it.
    """

    __slots__ = ('armed', 'done', 'fault', 'outcome', 'stop_generation', 'target')

    def __init__(self, stop_generation: int) -> None:
        self.stop_generation = stop_generation
        self.armed = False
        self.done = threading.Event()
        self.outcome: str | None = None
        self.fault: MoveNotCompletedError | None = None
        self.target: float | None = None

    def settle(self, outcome: str, fault: MoveNotCompletedError | None = None) -> None:
        """Fix the outcome, unless the move is already settled."""
        if self.done.is_set():
            return
        self.outcome = outcome
        self.fault = fault
        self.done.set()


class MoveInFlight:
    """A move that has started; ``wait()`` gives its outcome.

    Returned by ``MotionAPI.start_move_absolute`` and ``start_move_relative``
    once the board has taken the command and the axis is MOVING. ``wait()``
    returns when this move's axis arrived and raises otherwise -- the same
    verdict ``move_absolute`` and ``move_relative`` give, because each of
    them is a start followed by this wait. It blocks the calling thread,
    never the scope's IO lane. The outcome is the one the move ended with,
    however late the wait: a later move or a STOP after it ended does not
    change it.

    Attributes:
        axis: The axis this move drove.
    """

    def __init__(self, motion: MotionAPI, axis: str, move: _Move) -> None:
        self._motion = motion
        self.axis = axis
        self._move = move

    def wait(self) -> None:
        """Return once this move's axis has arrived at its target.

        Raises:
            MoveNotCompletedError: The axis did not arrive. ``'stalled'`` or
                ``'board_lost'``, the motion monitor gave it up;
                ``'faulted'``, something else set it UNKNOWN; ``'timed_out'``,
                the motion bound ran out (each of these leaves the axis
                UNKNOWN); ``'stopped'``, a stop the board took while it
                moved halted it; ``'superseded'``, another move or a home
                took the axis before it arrived.
        """
        self._motion._await_move(self.axis, self._move)


class MotionAPI:
    """Motion sub-API. Hosts stateless (Phase 2b) and stateful (Phase 2c) bodies."""

    _MOTION_POLL_INTERVAL = 0.02  # 50 Hz
    # How long an axis may sit MOVING with the motor board disconnected
    # before the monitor faults it to a terminal state. Well above a
    # transient USB blip, well below the 120s motion timeout.
    _DISCONNECT_FAULT_S = 3.0

    # Maps a motion axis to its frame-validity source. X and Y share
    # 'xy_move'; Z and the turret each have their own source so the
    # settle-check gates on the correct axis reaching IDLE. A turret
    # move that recorded 'xy_move' would clear the moment X/Y read idle,
    # before the turret physically finished.
    _AXIS_VALIDITY_SOURCE: ClassVar[dict] = {'Z': 'z_move', 'T': 'turret'}

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
        # The structural invariant
        # "never hold _axis_state_lock across a serial call" is enforced
        # at runtime via warn_hold_threshold_ms=1.0 -- any acquire-release
        # cycle that holds the lock for more than 1 ms emits a warning
        # log naming the lock + thread + duration, regardless of trace
        # state. Catches future code-introducers who hold across a serial
        # round-trip (typically 30-200 ms on the motor bus).
        self._axis_state_lock = profile_trace.TimedLock(
            threading.Lock(),
            name='motion._axis_state_lock',
            warn_hold_threshold_ms=1.0,
        )

        # Motion monitor wakeup -- set when any axis starts MOVING, cleared
        # when all axes are back to IDLE. The monitor thread sleeps on this.
        self._motion_wake = threading.Event()

        # Position change listeners -- push-based UI update mechanism.
        # Each listener is called with (axis, target, state) whenever a
        # position cache update or axis state transition occurs. Listeners
        # fire from the IO executor thread, so they MUST schedule UI work
        # via Clock.schedule_once.
        self._position_listeners_lock = threading.Lock()
        self._position_listeners: list = []

        # Boolean operation flags use threading.Event for wait/signal.
        self._homing_event = threading.Event()  # set => homing in progress
        self._turreting_event = threading.Event()  # set => turret move in progress

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
        # Per-axis (move, monotonic timestamp) the move was first observed
        # MOVING with the board connected; bounds a connected-but-stalled
        # axis the same way _disconnect_since bounds a vanished board. Kept
        # with its move, so a retarget never inherits the earlier move's
        # time.
        # Without it, a move whose position_reached never fires wedges
        # every state-reader -- capture settle-checks, is_moving pollers --
        # forever, while only explicit waiters carry their own timeout.
        # Only the monitor thread touches it.
        self._moving_since: dict[str, tuple[_Move, float]] = {}
        # The move whose failed position read the monitor has already
        # warned of: once per move, not once per poll, which is about fifty
        # a second. Only the monitor thread touches it.
        self._unread_warned: dict[str, _Move] = {}
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
        self._last_turret_position: int | None = None

        # The slot the last successful turret MOVE landed on, which a home
        # never writes: a home leaves the turret on slot 1 by convention, not
        # by anyone's choice. When two slots carry the same objective, this is
        # the one a person last chose, so the slot lookup prefers it over the
        # current slot. Survives a restart through the saved turret_position,
        # seeded at bring-up. None: no preference known.
        self._preferred_turret_slot: int | None = None

        # Bumped by every stop_motion. A STOP sets target = actual on every
        # axis, so a move in flight then reports "reached" at a place
        # nobody commanded; a waited move compares this against the value
        # it saw before driving to tell a stop from an arrival.
        self._stop_generation = 0
        self._stop_lock = threading.Lock()

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

        homed_axes is required rather than defaulted: a caller that
        forgets it would silently reintroduce the blind seeding, and
        that failure is invisible until a headless run is refused.

        Args:
            present_axes: List of axis names the hardware actually has.
            homed_axes: Those axes the hardware reports as already homed.
        """
        self._pos_cache = dict.fromkeys(present_axes, 0.0)
        self._axis_state = {
            ax: (AxisState.IDLE if ax in homed_axes else AxisState.UNKNOWN) for ax in present_axes
        }
        self._arrival_events = {ax: threading.Event() for ax in present_axes}
        for ev in self._arrival_events.values():
            ev.set()  # Start as "arrived" (not moving)
        self._current_move = dict.fromkeys(present_axes)

    def _start_monitor(self) -> None:
        """Spawn the motion monitor thread.

        Called from Lumascope.__init__ after _init_axes() so the thread
        always sees fully populated state dicts. Separate from __init__
        so create_diagnostic can control the spawn sequence.
        """
        self._motion_monitor_stop.clear()
        self._motion_monitor_thread = threading.Thread(
            target=self._motion_monitor_loop,
            name='motion-monitor',
            daemon=True,
        )
        self._motion_monitor_thread.start()

    def _disconnect(self) -> None:
        """Stop the motion monitor and reset axis states.

        Called from Lumascope.disconnect() before the motor driver is
        swapped to NullMotionBoard. Sets all arrival events so any blocked
        waiters unblock cleanly.
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

    def position_is_known(self, axis: str) -> bool:
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
            UNKNOWN or HOMING.
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
            MoveNotCompletedError: the fault the move ended with --
                ``'driver_failed'``, chained from the driver's error, unless
                the monitor gave the axis up first (a lost board), whose
                object it then is. The one object its caller reports.
        """
        failed = MoveNotCompletedError(axis, 'driver_failed')
        failed.__cause__ = cause
        self._set_axis_state(axis, AxisState.UNKNOWN, fault=failed)
        raise move.fault or failed

    @staticmethod
    def _refuse_turret_on_generic_door(axis: str, member: str) -> None:
        """Refuse T at a public generic mover; the turret moves only by slot.

        A turret moved by a generic door skips the Z park that keeps the
        objective off the sample and never records a slot, so afterwards
        nothing knows which objective is in the light path. ``move_turret``
        is the one door that does both.

        Raises:
            ValueError: ``axis`` is ``'T'``.
        """
        if axis == 'T':
            raise ValueError(
                f'{member} does not move the turret: use move_turret(slot), which parks '
                f'Z first and records the slot in the light path'
            )

    def _refuse_absent(self, member: str, axis: str | None = None) -> None:
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

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, naming the missing part. Nothing was sent.
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

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                motor controller. Nothing was sent.
        """
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
            AxisStateUnknownError: Naming every refused axis. Reported once
                through the one reporter before it is raised, so a caller
                that reports it again shows nothing more.
        """
        wanted = set(axes)
        with self._axis_state_lock:
            states = {axis: s for axis, s in self._axis_state.items() if axis in wanted}
        refused = {
            axis: s
            for axis, s in states.items()
            if (not self._position_known(s) if recording else self._drive_refused(s))
        }
        if not refused:
            return
        error = AxisStateUnknownError(refused, then=then)
        # Solicited: the user just asked for this, so it reaches them even
        # while a run is in flight.
        notifications.report_outcome(error, solicited=True, category='Motion')
        raise error

    # ------------------------------------------------------------------
    # Stateless method bodies.
    #
    # Order mirrors _lumascope.py source order.
    # ------------------------------------------------------------------

    def stop_motion(self) -> None:
        """Stop all in-flight motor moves.

        Idempotent. Uses the firmware-side ``STOP`` command, which the motor
        controller implements as ``motorstop`` (target=actual on all axes).
        It does not wait behind the lane, so a stop reaches the board while
        a move holds the lane.

        Raises:
            HardwareCommandRefusedError: ``'scope_disconnected'`` after
                ``disconnect()``, as every other command is refused then;
                ``'not_connected'`` or ``'axis_absent'`` with no motor
                controller (see ``_refuse_absent``). Nothing was sent.
            MotorStopFailedError: the board did not take the STOP, so the
                stage may still be moving. Chained from the driver's
                error. The stop generation has moved regardless.
        """
        if self._scope._io_executor.pending_shutdown:
            raise HardwareCommandRefusedError('scope_disconnected', 'stop_motion')
        self._refuse_absent('stop_motion')
        self._stop()

    def _stop(self) -> None:
        """Send the STOP: the scope's own teardown, which asks presence first."""
        # The generation moves inside this lock, after the board answered,
        # and a waited move reads it under the same lock: a move the STOP
        # ended cannot read the generation before the bump, and a firmware
        # that does not implement STOP (nothing stopped) never bumps it.
        with self._stop_lock:
            self._send_stop()

    def _send_stop(self) -> None:
        try:
            # Route through MotorBoard.motor_stop so field firmware
            # (2024-09-10 EL-0940-02, no STOP command) silently no-ops
            # instead of producing two FIRMWARE ERROR warnings per
            # shutdown. motor_stop returns True if STOP was accepted,
            # False if firmware doesn't implement it (cached).
            stopped = self._driver.motor_stop()
            if stopped:
                self._stop_generation += 1
                logger.info('[SCOPE API ] stop_motion: motors stopped')
            else:
                logger.debug(
                    '[SCOPE API ] stop_motion: firmware does not '
                    'implement STOP; motors will latch on disconnect'
                )
        except Exception as e:
            # The exchange may have failed after the board took the STOP,
            # so a move in flight cannot be vouched for as arrived.
            self._stop_generation += 1
            raise MotorStopFailedError() from e

    def get_turret_position_for_objective_id(self, objective_id: str) -> int | None:
        """The turret slot to use for an objective, or None when no slot carries it.

        One lookup for every caller -- a run's step and a person's step
        navigation -- so the two choose the same slot. When several slots
        carry the objective, ranked:
            1. The preferred slot (``get_preferred_turret_slot``): the last
               slot a turret move landed on, surviving a restart. After a
               home the turret sits on slot 1 by convention, which says
               nothing about which of two identical objectives a person
               uses.
            2. The turret's current slot (``get_turret_slot``).
            3. The lowest-numbered slot carrying it.

        Args:
            objective_id: Objective identifier to search for.

        Returns:
            int | None: Turret position (1-4), or None if not found.
        """
        turret_config = self._scope.runtime_state.get_turret_config()
        for slot in (self._preferred_turret_slot, self.get_turret_slot()):
            if slot is not None and turret_config.get(slot) == objective_id:
                return slot

        for (
            turret_position,
            turret_objective_id,
        ) in turret_config.items():
            if objective_id == turret_objective_id:
                return turret_position

        return None

    def is_current_turret_position_objective_set(self) -> bool:
        """Check whether the objective slot at the current turret position is set.

        Returns:
            bool: True if the turret's current slot is known and has a
                configured objective ID; False if the slot is unconfigured
                or not known -- an unknown slot has no objective anyone can
                name.
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
        homing is preserved for diagnostics.
        """
        before = self.get_limit_switch_status_all_axes()
        logger.info(f'Limit switch status before homing: {before}', extra={'force_error': True})
        yield
        after = self.get_limit_switch_status_all_axes()
        logger.info(f'Limit switch status after homing: {after}', extra={'force_error': True})

    @slow_task_budget(_HOMING_SLOW_TASK_S)
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
                ``_refuse_absent``); nothing was driven.
            HomingFailedError: the driver answered False or raised, or a
                homed axis's position could not be read.
        """
        # Asked before the driver: without it, a home with no controller
        # dispatches into the driver where exchange_command tries to
        # auto-reconnect and burns its full timeout (~10 s), and the
        # person sees a hang, then a "Homing Failed" that implies a
        # homing-mechanics problem instead of the cable.
        self._refuse_absent('home')
        present_axes = self._scope.capabilities.axes
        _api_log.info('home START')
        for ax in present_axes:
            self._set_axis_state(ax, AxisState.HOMING)
        # A homing turret is in no known slot until the home succeeds.
        self._last_turret_position = None
        stop_generation = self._stop_generation
        if 'Z' in present_axes:
            self._scope.imaging.frame_validity.invalidate('z_move')
        if 'X' in present_axes or 'Y' in present_axes:
            self._scope.imaging.frame_validity.invalidate('xy_move')
        if 'T' in present_axes:
            self._scope.imaging.frame_validity.invalidate('turret')
        self._is_homing = True
        try:
            with self._reference_position_logger():
                result = self._driver.home()
            if result is False:
                for ax in present_axes:
                    self._set_axis_state(ax, AxisState.UNKNOWN)
                raise HomingFailedError('ALL', 'failed', present_axes)
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
                self._last_turret_position = 1
        except HomingFailedError:
            raise
        except Exception as e:
            for ax in present_axes:
                self._set_axis_state(ax, AxisState.UNKNOWN)
            raise HomingFailedError('ALL', 'error', present_axes) from e
        finally:
            self._is_homing = False
            _api_log.info('home DONE')

    @contextlib.contextmanager
    def _safe_turret_move(self, restore_z: bool = True) -> Iterator[int]:
        """Context manager that lowers Z to 0 before turret motion and restores after.

        Use as ``with scope.motion._safe_turret_move() as stop_generation:
        ... move turret ...``; it yields the stop generation read before Z
        was parked, the one a caller judges the whole change by. Sets
        ``_is_turreting`` for the duration and restores the original Z
        position even if the body raises -- unless a stop landed since Z
        was parked: the person stopped the scope, so Z stays parked.

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
        stop_generation = self._stop_generation
        # force: this retract is the turret-safety move, and it is also
        # the first motion of the turret-home recovery -- the case where
        # Z is legitimately still unknown because the home that would
        # have established it is the operation being recovered. Refusing
        # here would mean an UNKNOWN Z could never be re-homed without
        # restarting the application.
        self._move_absolute_impl('Z', position=0, force=True).wait()
        self._is_turreting = True
        try:
            yield stop_generation
        finally:
            # Always clear the flag, even if the body raised (e.g. driver
            # HardwareError from the turret home). Without this, a failed turret
            # home would leave _is_turreting=True and the stage stuck at
            # Z=0.
            self._is_turreting = False
            # The restore is a new move, which reads the generation after
            # the stop and so is not withheld by it; this is what keeps a
            # stopped change from driving Z back up.
            if self._stopped_since(stop_generation):
                logger.info(
                    '[SCOPE API ] Leaving Z parked -- a stop landed during the turret change',
                    extra={'force_error': True},
                )
            elif restore_z:
                logger.info(f'[SCOPE API ] Restoring Z to {initial_z}', extra={'force_error': True})
                # force for the same reason as the retract: this is the
                # other half of one recovery, and refusing it would park
                # the stage at Z=0 with no way back.
                self._move_absolute_impl('Z', position=initial_z, force=True).wait()
            else:
                logger.info(
                    '[SCOPE API ] Skipping Z restore -- caller will overwrite Z next',
                    extra={'force_error': True},
                )

    @slow_task_budget(_HOMING_SLOW_TASK_S)
    def _home_turret_impl(self) -> None:
        """Home the turret axis. Moves Z to 0 during turret motion for safety.

        Returns when the turret is homed.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no turret (see
                ``_refuse_absent``); refused before Z is parked, and nothing
                was driven or recorded.
            HomingFailedError: the driver answered False, the home or its
                Z park raised, or the turret's position could not be read.
        """
        # Asked before the Z park: a scope with no turret would otherwise
        # park and restore Z and record slot 1 for a turret it does not
        # have, and one with no controller burn the driver's auto-reconnect
        # timeout first.
        self._refuse_absent('home', 'T')

        # T goes HOMING once Z is parked, not before: until then nothing
        # is turning it.
        _api_log.info('T home START')
        # A homing turret is in no known slot until the home succeeds.
        self._last_turret_position = None
        stop_generation = self._stop_generation
        try:
            with self._reference_position_logger(), self._safe_turret_move():
                self._set_axis_state('T', AxisState.HOMING)
                self._scope.imaging.frame_validity.invalidate('turret')
                result = False
                try:
                    result = self._driver.thome()
                finally:
                    # Transition T out of HOMING on EVERY exit, including a
                    # raised driver call. The motion monitor polls MOVING, not
                    # HOMING, so nothing else ever takes T out of it: a
                    # still-HOMING T reads as the scope moving to every
                    # reader, and holds any wait_until_finished_moving begun
                    # during the home until its timeout. Failure -> UNKNOWN,
                    # success -> IDLE; both set the arrival event.
                    self._set_axis_state('T', AxisState.IDLE if result else AxisState.UNKNOWN)
            if result is False:
                raise HomingFailedError('T', 'failed', ('T',))
            read = self._refresh_position_cache()
            self._raise_unread_axes('T', ('T',), read)
            # Turret homes to slot 1 (see home() for why it is recorded). A
            # home a stop cut short did not reach it.
            if not self._stopped_since(stop_generation):
                self._last_turret_position = 1
        except HomingFailedError:
            raise
        except Exception as e:
            self._set_axis_state('T', AxisState.UNKNOWN)
            raise HomingFailedError('T', 'error', ('T',)) from e
        finally:
            _api_log.info('T home DONE')

    @slow_task_budget(_TURRET_MOVE_SLOW_TASK_S)
    def _move_turret_impl(self, position: int, restore_z: bool = True) -> None:
        """Move the turret to a specific position. Skips if already there.

        Args:
            position: Target turret position (1-4).
            restore_z: When True (default), restore the pre-move Z
                position after the turret move completes. Set to False
                when the caller will immediately set Z to a different
                value (e.g. protocol step navigation moves T then Z to
                the new step's target -- restoring Z first is wasted
                motion).

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no turret (see
                ``_refuse_absent``); refused before Z is parked, and nothing
                was driven or recorded.
            AxisStateUnknownError: The turret position is unknown.
            PositionOutOfRangeError: The slot is not a whole number 1-4.
            MoveNotCompletedError: The Z park, the turret move or the Z
                restore did not arrive, or was stopped. The slot is unknown
                afterwards, whichever of the three it was.
        """
        self._refuse_absent('move_turret', 'T')
        # Refused here as well as at the generic door below, and both are
        # load-bearing: this one precedes the safety Z-retract and the
        # same-position short-circuit, so a nonsense slot cannot drop Z or
        # poison the position cache on its way to being refused.
        if not is_turret_slot(position):
            raise PositionOutOfRangeError(
                'T',
                position,
                TURRET_SLOT_MIN,
                TURRET_SLOT_MAX,
                bound='turret slots',
                quantity='slot',
            )

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
        # turret in is answered without moving -- and is still a choice of
        # that slot.
        if self._last_turret_position == position:
            self._preferred_turret_slot = int(position)
            return

        # Unknown from the start, and written only once the whole command --
        # park, move, restore -- returned: a raise anywhere in it leaves the
        # turret in no slot anyone can vouch for.
        self._last_turret_position = None
        with self._safe_turret_move(restore_z=restore_z) as stop_generation:
            logger.info(f'[SCOPE API ] Moving T to position {position}')
            self._move_absolute_impl('T', position).wait()
        # T's own wait judges only T; a stop after T arrived, before the
        # restore, ended the change all the same.
        if self._stopped_since(stop_generation):
            raise MoveNotCompletedError('T', 'stopped')
        self._last_turret_position = int(position)
        self._preferred_turret_slot = int(position)

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
        """
        return self._last_turret_position

    def get_preferred_turret_slot(self) -> int | None:
        """The slot the last successful turret move landed on, or None.

        Never written by a home. Seeded at bring-up from the saved turret
        position, so a person's choice between two slots carrying the same
        objective survives a restart; the slot lookup prefers it.

        Returns:
            int | None: The slot, 1-4, or None when no preference is known.
        """
        return self._preferred_turret_slot

    def seed_preferred_turret_slot(self, slot: int | None) -> None:
        """Seed the preferred slot at bring-up from the saved turret position.

        This is not part of the L2 API surface: it is bring-up's seam,
        called by ``Lumascope.initialize`` with the saved value. A caller
        that wants a slot preferred turns the turret to it with
        ``move_turret``.

        Raises:
            PositionOutOfRangeError: ``slot`` is neither None nor a slot 1-4.
        """
        if slot is not None and not is_turret_slot(slot):
            raise PositionOutOfRangeError(
                'T',
                slot,
                TURRET_SLOT_MIN,
                TURRET_SLOT_MAX,
                bound='turret slots',
                quantity='slot',
            )
        self._preferred_turret_slot = slot

    def jog_step(self, axis: str, coarse: bool) -> float:
        """The jog step for ``axis`` under the active objective.

        A jog's size scales with the objective, so the active objective's
        catalogue entry answers: ``z_coarse`` / ``z_fine`` for Z,
        ``xy_coarse`` / ``xy_fine`` for X and Y, in the units
        ``move_relative`` takes for that axis.

        Raises:
            ObjectiveUnknownError: The objective in the light path is
                unknown; no step is guessed, so nothing should move.
            ValueError: ``axis`` is not 'X', 'Y' or 'Z'.
        """
        if axis == 'Z':
            kind = 'z'
        elif axis in ('X', 'Y'):
            kind = 'xy'
        else:
            raise ValueError(f"jog_step: axis must be 'X', 'Y' or 'Z', got {axis!r}")
        _, objective = self._scope.runtime_state.resolve_current_objective()
        return objective[f'{kind}_{"coarse" if coarse else "fine"}']

    def get_actual_position(self, axis: str) -> float | None:
        """Query the actual hardware position via serial (not cached); um for X/Y/Z, turret slot for T.

        Unlike get_current_position(), which serves the in-memory cache,
        this asks the motor controller directly. Use during continuous
        motion sweeps where the stage is moving and the cache doesn't
        reflect the true position. The commanded target is a third thing
        again -- get_target_position() answers that.

        Costs one serial round-trip (~5ms).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float | None: Current position in um; None with the null board
            installed or for an axis this scope does not have.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, the installed
                motor controller is not connected (its cable pulled).
            HardwareError: the controller did not report the position.
        """
        if not self._has_position(axis):
            return None
        self._refuse_absent('get_actual_position')
        return self._driver.current_pos(axis)

    def set_precision_mode(self, axis: str, enabled: bool) -> None:
        """Set motor precision mode for an axis.

        Precision mode uses accurate but slightly slower motor stopping.
        Use before autofocus fine passes or any measurement requiring
        precise Z positioning. Disable for coarse moves where speed
        matters more than final position accuracy.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning, False for speed.

        Raises:
            ValueError: ``axis`` is not an axis name.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``_refuse_absent``); nothing was sent.
        """
        return self._dispatch_motion(
            self._set_precision_mode_impl,
            'set_precision_mode',
            args=(axis, enabled),
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    def _set_precision_mode_impl(self, axis: str, enabled: bool) -> None:
        if axis not in _VALID_AXIS_NAMES:
            raise ValueError(f'Axis must be one of {_VALID_AXIS_NAMES}, got {axis!r}')
        self._refuse_absent('set_precision_mode', axis)
        self._driver.set_precision_mode(axis, enabled)

    def get_target_status(self, axis: str) -> bool:
        """Check if an axis has reached its target position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True if at target (always True for T if no turret present).
        """
        if not self._scope.motor_connected:
            # Disconnected is an expected degradation, not a fault: the
            # motion monitor polls this on a timer, so provoking the driver
            # would trace a HardwareError on every poll after a mid-move USB
            # yank. Answer False and stay quiet.
            return False

        # Handle case where we want to know if turret has reached its target, but there is no turret
        if (axis == 'T') and (not self._driver.has_turret()):
            return True

        try:
            status = self._driver.target_status(axis)
            return status
        except HardwareError as e:
            # Typed disconnect/timeout at the moment of unplug (before
            # motor_connected flips). Expected; log without the traceback.
            logger.warning(
                f'[SCOPE API ] get_target_status({axis}): {e}; treating as not at target'
            )
            return False
        except Exception as e:
            logger.exception(
                f'[SCOPE API ] get_target_status({axis}) failed; treating as not at target: {e}'
            )
            return False

    def get_limit_switch_status(self, axis: str) -> tuple[int, int]:
        """Get the limit switch status for an axis.

        Tells a caller WHY a move stopped short: an axis that reached a limit
        reports it here rather than raising, so the difference between "the
        move finished" and "the stage ran out of travel" is only visible by
        asking.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            tuple[int, int]: ``(left, right)``, each 1 when that switch is
            engaged, 0 when clear, and -1 when the state could not be read.
        """
        return self._driver.limit_switch_status(axis=axis)

    def get_limit_switch_status_all_axes(self) -> dict:
        """Get limit switch status for all axes.

        Returns:
            dict: Axis name -> the ``(left, right)`` pair described on
            ``get_limit_switch_status``. Covers only the axes the board
            reports, so a scope without a turret has no "T" key.
        """
        resp = {}
        for axis in self._scope.capabilities.axes:
            resp[axis] = self.get_limit_switch_status(axis=axis)
        return resp

    def is_moving(self) -> bool:
        """Check if any axis is currently moving.

        Reads from in-memory axis state -- zero serial I/O. The motion
        monitor thread handles firmware queries and state transitions.

        Returns:
            bool: True if any axis is MOVING or HOMING. A Z move is MOVING
            from its first target write, its backlash leg included.
        """
        return self.is_any_axis_moving()

    def set_acceleration_limit(self, val_pct: int) -> None:
        """Set the motor controller acceleration limit (percent of max).

        Firmware that does not implement the AMAX / DMAX query needs no
        handling here: the driver's own read already answers with a default
        for it, so a legacy board completes this call rather than failing it.
        A refusal that does reach this point is therefore a real one, and it
        belongs to the caller -- swallowing it left a scope configured from a
        value its firmware had rejected, with nothing said to anyone.

        Args:
            val_pct: Acceleration limit as a percent of the firmware max.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller (see
                ``_refuse_absent``); asked before the value, and nothing was
                sent.
            AccelerationLimitRefusedError: ``val_pct`` is not a number or
                is outside ``ACCELERATION_PCT_MIN`` to ``ACCELERATION_PCT_MAX``
                (a ValueError). Refused on every board, real or simulated,
                before it is commanded.
        """
        return self._dispatch_motion(
            self._set_acceleration_limit_impl,
            'set_acceleration_limit',
            kwargs={'val_pct': val_pct},
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    def _set_acceleration_limit_impl(self, val_pct: int) -> None:
        self._refuse_absent('set_acceleration_limit')
        refuse_acceleration_pct(val_pct)
        self._driver.set_acceleration_limits(val_pct=val_pct)

    # ------------------------------------------------------------------
    # Stateful method bodies.
    #
    # State slots (_pos_cache, _axis_state, _arrival_events,
    # _position_listeners, _motion_wake, _motion_monitor_*, _homing_event,
    # _turreting_event) live on this surface.
    # ------------------------------------------------------------------

    # --- CR-2: Thread-safe properties for shared state ---

    @property
    def _is_homing(self) -> bool:
        """True while the microscope is homing.

        Returns:
            bool: True if a homing operation is in progress.
        """
        return self._homing_event.is_set()

    @_is_homing.setter
    def _is_homing(self, value: bool) -> None:
        """Set the homing-in-progress flag."""
        if value:
            self._homing_event.set()
        else:
            self._homing_event.clear()

    @property
    def _is_turreting(self) -> bool:
        """True while the turret is moving.

        Returns:
            bool: True if a turret motion is in progress.
        """
        return self._turreting_event.is_set()

    @_is_turreting.setter
    def _is_turreting(self, value: bool) -> None:
        """Set the turret-motion-in-progress flag."""
        if value:
            self._turreting_event.set()
        else:
            self._turreting_event.clear()

    def _home_moves_turret(self, action) -> bool:
        """Whether the home body ``action`` moves the turret.

        The whole-scope home homes every axis the board has, so it moves the
        turret exactly when the scope has one.
        """
        if action == self._home_turret_impl:
            return True
        return action == self._home_impl and self._scope.capabilities.has_turret

    def get_axis_state(self, axis: str) -> str:
        """Get the current state of an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            str: One of AxisState.UNKNOWN, IDLE, MOVING, HOMING.
        """
        with self._axis_state_lock:
            return self._axis_state.get(axis, AxisState.UNKNOWN)

    def add_position_listener(self, listener) -> None:
        """Register a callback for position/state changes on any axis.

        The listener is called with ``(axis, target_pos, state)`` whenever
        the position cache or axis state changes. It fires from the thread
        that caused the change (IO executor, motion monitor, etc.), so
        listeners **must** schedule any UI work via ``Clock.schedule_once``.

        Args:
            listener: ``callable(axis: str, target: float, state: str)``
        """
        with self._position_listeners_lock:
            self._position_listeners.append(listener)

    def remove_position_listener(self, listener) -> None:
        """Unregister a position listener.

        Args:
            listener: A callable previously passed to
                ``add_position_listener``. Silently ignores listeners that
                are not currently registered.
        """
        with self._position_listeners_lock:
            try:
                self._position_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_position_listeners(self, axis: str):
        """Notify all position listeners of a change on *axis*."""
        with self._pos_cache_lock:
            target = self._pos_cache.get(axis, 0.0)
        with self._axis_state_lock:
            state = self._axis_state.get(axis, AxisState.UNKNOWN)
        with self._position_listeners_lock:
            listeners = list(self._position_listeners)
        for fn in listeners:
            try:
                fn(axis, target, state)
            except Exception as ex:
                # No caller waits on a listener, so its fault stops here; the
                # other listeners are still told.
                notifications.report_outcome(ex, solicited=False, category='Motion')

    def is_any_axis_moving(self) -> bool:
        """Check if any axis is currently MOVING or HOMING.

        Reads from the in-memory state dict -- zero serial I/O.

        Returns:
            bool: True if any axis is in MOVING or HOMING state.
        """
        with self._axis_state_lock:
            return any(s in (AxisState.MOVING, AxisState.HOMING) for s in self._axis_state.values())

    def get_axis_limits(self, axis: str) -> Mapping[str, float] | None:
        """Get the travel limits for an axis, in um.

        Args:
            axis: Axis name ("X", "Y", "Z", or "T").

        Returns:
            A read-only mapping with 'min' and 'max' positions in um
            (an edit raises ``TypeError``: it is the bound moves are
            refused against), or ``None`` if the axis has no configured
            limits (typical for the turret T axis). Callers must handle
            the None case.
        """
        return self._driver.get_axis_limits(axis=axis)

    @slow_task_budget(_HOMING_SLOW_TASK_S)
    def _zhome_impl(self) -> None:
        """Home the Z axis (focus).

        Returns when Z is homed and its position read.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no Z (see
                ``_refuse_absent``); nothing was driven.
            HomingFailedError: the driver answered False or raised (e.g.
                HardwareError on no-response / firmware-error), or Z's
                position could not be read.
        """
        # Asked before the driver, as the full home does: its
        # exchange_command would burn its auto-reconnect timeout and the
        # user see a hang instead of the actual cause.
        self._refuse_absent('home', 'Z')
        _api_log.info('Z home START')
        self._set_axis_state('Z', AxisState.HOMING)
        self._scope.imaging.frame_validity.invalidate('z_move')
        try:
            with self._reference_position_logger():
                result = self._driver.zhome()
            if result is False:
                self._set_axis_state('Z', AxisState.UNKNOWN)
                raise HomingFailedError('Z', 'failed', ('Z',))
            read = self._refresh_position_cache()
            if 'Z' in read:
                self._set_axis_state('Z', AxisState.IDLE)
            self._raise_unread_axes('Z', ('Z',), read)
        except HomingFailedError:
            raise
        except Exception as e:
            self._set_axis_state('Z', AxisState.UNKNOWN)
            raise HomingFailedError('Z', 'error', ('Z',)) from e
        finally:
            _api_log.info('Z home DONE')

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
        """
        with self._axis_state_lock:
            stage_states = [state for axis, state in self._axis_state.items() if axis != 'T']
        if not stage_states:
            return False
        return all(self._position_known(state) for state in stage_states)

    def axes_without_position(self) -> dict[str, str]:
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
            dict[str, str]: Axis name to its state (``AxisState.UNKNOWN``
                or ``AxisState.HOMING``) for every axis whose position is
                not known, in the scope's axis order. Empty when every
                axis knows its position.
        """
        with self._axis_state_lock:
            return {
                axis: state
                for axis, state in self._axis_state.items()
                if not self._position_known(state)
            }

    def axis_positions(self) -> dict[str, AxisPosition]:
        """Every axis's state and its position, or None where the position is not known.

        One snapshot: the state and the cache are read together, so a
        caller writing a position into a file cannot pair a number with a
        state that changed between two reads. An axis is answered with a
        position only while it is IDLE or MOVING; UNKNOWN and HOMING
        answer None whatever the cache holds, because the cache keeps the
        last number an axis reported after its reference is lost. um for
        X/Y/Z, the slot for T. No serial I/O.

        The state lock is taken first and the cache lock inside it. No
        other path nests the two, so this order cannot meet its reverse.

        Returns:
            dict[str, AxisPosition]: Axis name to (state, position), in
                the scope's axis order.
        """
        with self._axis_state_lock:
            states = dict(self._axis_state)
            with self._pos_cache_lock:
                cache = dict(self._pos_cache)
        return {
            ax: AxisPosition(state, cache.get(ax) if self._position_known(state) else None)
            for ax, state in states.items()
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
            HomingFailedError: ``'unread'``, naming the unread axes.
        """
        present = self._scope.capabilities.axes
        unread = [ax for ax in homed if ax in present and ax not in read]
        if unread:
            raise HomingFailedError(home, 'unread', unread)

    def _refresh_position_cache(self) -> set[str]:
        """Read every axis's position from the hardware into the cache.

        Called after a home, and once at construction, to sync the cache
        with the hardware; during normal operation the cache is updated
        by move commands and the motion monitor. An axis whose read fails
        is set UNKNOWN and its cache entry is left alone:
        a number nobody read is not a position, and a caller that writes
        positions into a file would otherwise record it as one.

        Returns:
            set[str]: The axes whose position was read.
        """
        positions = {}
        for ax in self._scope.capabilities.axes:
            try:
                positions[ax] = self._driver.target_pos(axis=ax)
            except HardwareError as e:
                _api_log.warning(f'position read on {ax} failed ({e}); axis UNKNOWN')
                self._set_axis_state(ax, AxisState.UNKNOWN)

        with self._pos_cache_lock:
            self._pos_cache.update(positions)
        for ax in positions:
            self._fire_position_listeners(ax)
        return set(positions)

    def _read_position_cache(self, axis: str | None) -> float | dict:
        """Shared cache-read primitive for the position-query methods.

        Pure cache access -- no T-axis policy here; callers decide their
        own sentinel for the "axis requested but not present" case (see
        get_target_position's None for no-turret-T).

        axis=None -> dict copy of all cached axis positions
        axis=<name> -> float (0.0 if axis missing from cache)
        """
        if axis is None:
            with self._pos_cache_lock:
                return dict(self._pos_cache)
        with self._pos_cache_lock:
            return self._pos_cache.get(axis, 0.0)

    def get_target_position(self, axis: str | None = None) -> float | dict | None:
        """Get the target position for an axis (where it is commanded to go); um for X/Y/Z, turret slot for T.

        The last target this API wrote to the board for the axis, while the
        move runs and after it arrives: the commanded number, not the polled
        position, which sits up to a microstep off it. A relative move's
        target is the board's own target plus the offset, so it carries that
        target's microstep rounding. Where no target was reached it is the
        polled position: before the axis's first move, after a home, after a
        STOP ended the move, while the axis is UNKNOWN, and while a move has
        not yet written its final target (a Z backlash leg). A refused move
        writes nothing, so the previous target stands. No serial I/O. T
        answers its slot as a float.

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict | None: Position in um for a single axis, or a dict
                of the axes that have one. None for an axis with no hardware
                behind it (see ``_has_position``).
        """
        if axis is None:
            return {
                ax: self.get_target_position(ax)
                for ax in self._scope.capabilities.axes
                if self._has_position(ax)
            }
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

    def get_current_position(self, axis: str | None = None) -> float | dict | None:
        """Get the current position for an axis; um for X/Y/Z, turret slot (1-4) for T.

        Reads from the in-memory position cache. During MOVING the cache
        is refreshed by _motion_monitor_loop polling the motor's actual
        position from hardware on every cycle; during IDLE the cache
        holds the last confirmed position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict | None: Position in um for a single axis, or a dict
                of the axes that have one. None for an axis with no hardware
                behind it (see ``_has_position``); a present axis keeps its
                number, its reference lost or not.
        """
        if axis is None:
            return {
                ax: self.get_current_position(ax)
                for ax in self._scope.capabilities.axes
                if self._has_position(ax)
            }
        if not self._has_position(axis):
            return None
        return self._read_position_cache(axis)

    def _plate_target_to_stage(self, axis: str, plate_mm: float, ignore_limits: bool) -> float:
        """Check a plate-frame target against what this stage can reach, then convert.

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
        check downstream would reject. That equivalence is what lets the
        protocol paths adopt this frame without changing which moves they
        refuse -- only the sentence the refusal carries.

        Honours ``ignore_limits`` for the same reason the travel check
        does: it is one bound expressed in two units, and a hatch that
        stopped working when the caller changed frames would be a trap.
        """
        if axis not in ('X', 'Y'):
            raise ValueError(f"frame='plate' applies to the X and Y axes, got {axis!r}")

        key = axis.lower()
        stage_position = self._scope.runtime_state.plate_to_stage_axis(axis=axis, plate_mm=plate_mm)

        limits = self.get_axis_limits(axis)
        if limits is not None and not ignore_limits:
            dimension = self._scope.runtime_state.get_labware().get_dimensions()[key]
            offset_mm = self._scope.runtime_state.get_stage_offset()[key] / 1000
            # Inverting sx = (dimension - offset - px) * 1000: the map
            # decreases in px, so the travel MAXIMUM yields the plate
            # minimum and vice versa.
            band_low = round(dimension - offset_mm - limits['max'] / 1000, 2)
            band_high = round(dimension - offset_mm - limits['min'] / 1000, 2)
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
        frame: str = 'stage',
    ) -> MoveInFlight:
        """Start an axis moving to an absolute position; return the started move.

        Returns once the board has taken the command and the axis is
        MOVING. The returned handle's ``wait()`` is the move's outcome; a
        caller that needs the axis where it was sent waits on it.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            position (float): Target position -- um for X/Y/Z; turret slot (1-4) for T.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            ignore_limits: If True, skip software limit checks.
            force: Drive even when the axis position is unknown. For the
                recovery paths only -- see ``_pre_drive``.

        Raises:
            ValueError: If axis is invalid or position is not numeric.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``_refuse_absent``); nothing was driven.
            PositionOutOfRangeError: The target is outside the axis's
                configured travel and ``ignore_limits`` is False. A
                ValueError subclass.
            AxisStateUnknownError: The axis position is unknown and
                ``force`` is False.
            HardwareCommandRefusedError: ``'position_unread'``, a Z move
                with overshoot whose position the board did not report;
                nothing was driven.
            MoveNotCompletedError: ``'driver_failed'``, the board did not
                take the command.
        """
        if axis not in _VALID_AXIS_NAMES:
            raise ValueError(f'Axis must be one of {_VALID_AXIS_NAMES}, got {axis!r}')
        self._refuse_absent('move_absolute', axis)
        if not isinstance(position, (int, float)):
            raise ValueError(f'Position must be numeric, got {type(position).__name__}')

        if frame == 'plate':
            position = self._plate_target_to_stage(axis, position, ignore_limits=ignore_limits)
        elif frame != 'stage':
            raise ValueError(f"frame must be 'stage' or 'plate', got {frame!r}")

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
            if limits is not None and not (limits['min'] <= position <= limits['max']):
                raise PositionOutOfRangeError(axis, position, limits['min'], limits['max'])

        # The turret's bound, which the travel check above cannot express: a
        # slot is not a distance, so get_axis_limits returns None for T and
        # nothing there refuses anything. Without this, a T target of 99 is
        # 24.5 revolutions. The public generic doors refuse T outright; the
        # caller that still reaches this body with T is move_turret, and the
        # bound keeps that one honest too.
        #
        # Before the ceiling below for the same reason travel is: the bound
        # that knows what the number MEANS answers first, so the turret gives
        # one vocabulary for every bad slot rather than naming slots for 5 and
        # a metre for 2000000.
        if axis == 'T' and not is_turret_slot(position):
            raise PositionOutOfRangeError(
                axis,
                position,
                TURRET_SLOT_MIN,
                TURRET_SLOT_MAX,
                bound='turret slots',
                quantity='slot',
            )

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

        leg = self._backlash_leg(axis, position, overshoot_enabled, 'move_absolute')
        stop_generation = self._stop_generation
        try:
            move, written = self._send_drive(
                axis, stop_generation, lambda: self._drive_to(axis, position, leg, stop_generation)
            )
        except Exception:
            _api_log.error(f'move_abs {axis}={position:.1f}um FAILED')
            raise
        self._publish_drive(axis, written, float(position))
        # No move-init cache write: the cache holds the CURRENT position
        # until _motion_monitor_loop reads it from hardware on its first
        # cycle. The target is the move's own (_Move.target), where
        # get_target_position reads it.
        self._fire_position_listeners(axis)
        _api_log.info(f'move_abs {axis}={position:.1f}um')
        return MoveInFlight(self, axis, move)

    def _await_move(self, axis: str, move: _Move) -> None:
        """Return only once ``move`` arrived at its target; raise otherwise.

        The outcome is the one the move was settled with when it ended (see
        ``_Move``), not anything read from the axis now: a move that
        arrived reads arrived however late this runs, and one that ended
        answers at once, never after a later move on the axis.

        Only this move is waited on and judged. Another axis's outcome
        belongs to whatever moved it, and waiting on it would hold this
        caller while, say, a person scrolls the focus.

        Args:
            axis: The axis this move drove.
            move: The move's record, from its MOVING write.

        Raises:
            MoveNotCompletedError: the fault it ended with (the monitor's
                own ``'stalled'`` or ``'board_lost'`` object when it gave
                the axis up, so the person is shown it once; ``'faulted'``
                when something else set it UNKNOWN); ``'timed_out'``, it had
                not ended when the wait's bound ran out -- the axis is then
                written UNKNOWN, in the same hold as the check that the axis
                is still this move's; ``'stopped'``, a STOP the board took
                while it moved; ``'superseded'``, another move or a home
                took the axis first.
        """
        if not self._wait_for_move(move, self._MOTION_SETTLE_TIMEOUT_S):
            # Refused only when the move ended meanwhile: every write that
            # takes the axis out of this move settles it in the same hold.
            self._set_axis_state(
                axis,
                AxisState.UNKNOWN,
                verdict_for=move,
                fault=MoveNotCompletedError(axis, 'timed_out'),
            )
        if move.outcome == 'arrived':
            return
        raise move.fault or MoveNotCompletedError(axis, move.outcome)

    def _wait_for_move(self, move: _Move, timeout_s: float) -> bool:
        """Wait for ``move`` to be settled; True when it was within ``timeout_s``."""
        return move.done.wait(timeout=timeout_s)

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

    def _move_relative_impl(
        self,
        axis: str,
        distance: float,
        overshoot_enabled: bool = False,
    ) -> MoveInFlight:
        """Start an axis moving by a relative distance; return the started move.

        Returns as ``_move_absolute_impl`` does: once the axis is MOVING,
        with the handle whose ``wait()`` is the move's outcome.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            distance (float): Distance to move -- um for X/Y/Z; turret slots for T.
            overshoot_enabled: Allow Z overshoot for backlash compensation.

        Raises:
            ValueError: If axis is invalid or distance is not numeric / out of bounds.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller or no such axis (see
                ``_refuse_absent``); nothing was driven.
            AxisStateUnknownError: The axis position is unknown.
            HardwareCommandRefusedError: ``'position_unread'``, the board
                did not report the target to add to (or, for Z with
                overshoot, the position); nothing was driven.
            MoveNotCompletedError: ``'driver_failed'``, the board did not
                take the command.

        There is deliberately no ``force`` hatch here. Every caller is a
        user jog or an autofocus sweep, and none of them is a recovery
        path -- the recovery that must move an unknown axis is the
        turret-safety Z-park, which is an absolute move. A hatch with no
        caller is just an opt-out waiting to be reached for.
        """
        if axis not in _VALID_AXIS_NAMES:
            raise ValueError(f'Axis must be one of {_VALID_AXIS_NAMES}, got {axis!r}')
        self._refuse_absent('move_relative', axis)
        if not isinstance(distance, (int, float)):
            raise ValueError(f'Distance must be numeric, got {type(distance).__name__}')
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
            raise HardwareCommandRefusedError('position_unread', 'move_relative') from e
        target_pos = start_pos + float(distance)

        # The same travel refusal as the absolute path, against the target
        # this move is about to publish. Without it the driver turned the
        # offset into an absolute move beyond travel and nothing refused it,
        # so a jog past a limit reported success from wherever the stage
        # stopped.
        limits = self.get_axis_limits(axis)
        if limits is not None and not (limits['min'] <= target_pos <= limits['max']):
            raise PositionOutOfRangeError(axis, target_pos, limits['min'], limits['max'])

        leg = self._backlash_leg(axis, target_pos, overshoot_enabled, 'move_relative')
        stop_generation = self._stop_generation
        try:
            move, written = self._send_drive(
                axis,
                stop_generation,
                lambda: self._drive_to(axis, target_pos, leg, stop_generation),
            )
        except Exception:
            _api_log.error(f'move_rel {axis}={distance:+.1f}um FAILED')
            raise
        self._publish_drive(axis, written, target_pos)
        # No move-init cache write: the cache holds the CURRENT position
        # until _motion_monitor_loop reads it from hardware on its first
        # cycle. The target is the move's own (_Move.target), where
        # get_target_position reads it.
        self._fire_position_listeners(axis)
        _api_log.info(f'move_rel {axis}={distance:+.1f}um')
        return MoveInFlight(self, axis, move)

    # --- Public dispatch ---
    # These are what every caller reaches: an SDK script, a REST
    # handler, the GUI -- and the run, the autofocus sweep and the diagnostics,
    # which call them under their taking so the lane admits their work while
    # they hold the scope. From a task already on the lane's worker the lane
    # runs the body inline.

    # Base liveness margin for a dispatched motion command: queue residence
    # plus the serial round-trips, with headroom. The per-command wait adds
    # the body's own declared motion time on top, so a long but correct move
    # or home is never timed out by its own liveness bound.
    _MOTION_WAIT_BASE_S = 30.0

    # One physically-waited motion's own bound: what a started move's
    # wait allows it, and what the homing routine legitimately takes on
    # long travel.
    _MOTION_SETTLE_TIMEOUT_S = 120.0

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
        the lane is closed, or a run or a diagnostic holds the scope and this
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

    def _start(self, member: str, body, axis: str, target: float, **kwargs) -> MoveInFlight:
        """Start one X, Y or Z move through ``member``: refuse the turret, then command it on the lane.

        The lane carries only the command -- the body's checks, the board
        write and the MOVING transition -- so its bound is the command's
        alone; the wait for arrival is the handle's, in the caller's thread.
        """
        self._refuse_turret_on_generic_door(axis, member)
        return self._dispatch_motion(
            body,
            member,
            args=(axis, target),
            kwargs=kwargs,
            timeout_s=self._MOTION_WAIT_BASE_S,
        )

    def start_move_absolute(
        self,
        axis: str,
        position: float,
        overshoot_enabled: bool = True,
        ignore_limits: bool = False,
        frame: str = 'stage',
    ) -> MoveInFlight:
        """Start X, Y or Z moving to an absolute position, in um; return the started move.

        For a caller that works while the axis travels -- measuring during
        the motion, or starting several axes together. It refuses exactly
        what ``move_absolute`` refuses, before anything moves, and returns
        once the board has taken the command. ``wait()`` on the returned
        handle gives the outcome ``move_absolute`` would have given. See
        ``_move_absolute_impl`` for the argument contract.
        """
        return self._start(
            'start_move_absolute',
            self._move_absolute_impl,
            axis,
            position,
            overshoot_enabled=overshoot_enabled,
            ignore_limits=ignore_limits,
            frame=frame,
        )

    def start_move_relative(
        self,
        axis: str,
        distance: float,
        overshoot_enabled: bool = False,
    ) -> MoveInFlight:
        """Start X, Y or Z moving by a relative distance, in um; return the started move.

        As ``start_move_absolute``, for ``move_relative``. See
        ``_move_relative_impl`` for the argument contract.
        """
        return self._start(
            'start_move_relative',
            self._move_relative_impl,
            axis,
            distance,
            overshoot_enabled=overshoot_enabled,
        )

    def move_absolute(
        self,
        axis: str,
        position: float,
        overshoot_enabled: bool = True,
        ignore_limits: bool = False,
        frame: str = 'stage',
    ) -> None:
        """Move X, Y or Z to an absolute position, in um, and return once it has arrived.

        The turret moves by slot: ``move_turret``. This is
        ``start_move_absolute`` followed by the handle's ``wait()``: the
        command goes on the scope's IO lane, and the wait for arrival
        blocks this caller, not the lane. A move on an axis this scope does
        not have is refused (``_refuse_absent``) and drives nothing.

        Raises:
            MoveNotCompletedError: The axis did not arrive; see
                ``MoveInFlight.wait``.
        """
        self._start(
            'move_absolute',
            self._move_absolute_impl,
            axis,
            position,
            overshoot_enabled=overshoot_enabled,
            ignore_limits=ignore_limits,
            frame=frame,
        ).wait()

    def move_relative(
        self,
        axis: str,
        distance: float,
        overshoot_enabled: bool = False,
    ) -> None:
        """Move X, Y or Z by a relative distance, in um, and return once it has arrived.

        The turret moves by slot: ``move_turret``. As ``move_absolute``:
        ``start_move_relative`` followed by the handle's ``wait()``.

        Raises:
            MoveNotCompletedError: The axis did not arrive; see
                ``MoveInFlight.wait``.
        """
        self._start(
            'move_relative',
            self._move_relative_impl,
            axis,
            distance,
            overshoot_enabled=overshoot_enabled,
        ).wait()

    def home(self, axis: str = 'ALL') -> None:
        """Home the given axis set, and wait for it.

        Returns only when the home established a reference: every homed
        axis has a known position (for ``'ALL'``, the axes the board has).

        Args:
            axis: ``'Z'`` homes the Z axis only. ``'T'`` homes the turret
                (parks Z at 0, homes T, restores Z -- three
                physically-waited motions, so its wait bound is three
                settle windows). ``'ALL'`` (default) homes every axis the
                board has; the firmware routine homes Z, then T, then X/Y.

        Raises:
            ValueError: on an unknown axis.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'axis_absent'``, no motor controller, or no Z or turret to
                home (see ``_refuse_absent``); or the lane refused the home: a
                run or a diagnostic holds the scope, or a recording does
                and this home moves the turret (``'T'``, or ``'ALL'`` on a
                scope with one).
            HomingFailedError: the home was driven and did not establish
                a reference: the driver answered False or raised, or a
                homed axis's position could not be read. The axes it
                names are UNKNOWN.
        """
        a = axis.upper()
        if a == 'Z':
            impl, settle_windows = self._zhome_impl, 1
        elif a == 'T':
            impl, settle_windows = self._home_turret_impl, 3
        elif a == 'ALL':
            impl, settle_windows = self._home_impl, 1
        else:
            raise ValueError(f"Unknown home axis {axis!r}: expected 'Z', 'T', or 'ALL'")
        self._dispatch_motion(
            impl,
            'home',
            timeout_s=self._MOTION_WAIT_BASE_S + settle_windows * self._MOTION_SETTLE_TIMEOUT_S,
            falsifies_recording=self._home_moves_turret(impl),
        )

    def move_turret(self, position: int, restore_z: bool = True) -> None:
        """Move the turret to a position, and wait for it. See ``_move_turret_impl``.

        The wait bound covers three physically-waited motions: the Z park,
        the turret move itself, and the Z restore.

        Raises:
            HardwareCommandRefusedError: a recording holds the scope: its
                frames carry the pixel size of the objective it started with.
        """
        return self._dispatch_motion(
            self._move_turret_impl,
            'move_turret',
            args=(position,),
            kwargs={'restore_z': restore_z},
            timeout_s=self._MOTION_WAIT_BASE_S + 3 * self._MOTION_SETTLE_TIMEOUT_S,
            falsifies_recording=True,
        )

    def wait_until_finished_moving(self, timeout_s: float = 120.0) -> None:
        """Block until every axis moving now has stopped; raise if one did not stop well.

        For motion the caller did not start, or started without keeping its
        handle. A move the caller started is judged by its handle's
        ``wait()``, which also knows when a stop or a later move ended it;
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
            MoveNotCompletedError: An axis it waited for ended UNKNOWN (the
                monitor's own ``'stalled'`` or ``'board_lost'`` object when it
                gave the axis up, else ``'faulted'``), or was still moving
                when ``timeout_s`` ran out (``'still_moving'``; its state is
                left to its own move).
        """
        with self._axis_state_lock:
            moving = [
                ax
                for ax, state in self._axis_state.items()
                if state in (AxisState.MOVING, AxisState.HOMING)
            ]
        deadline = time.monotonic() + timeout_s
        for ax in moving:
            if not self._wait_for_axis_to_stop(ax, max(0.0, deadline - time.monotonic())):
                raise MoveNotCompletedError(ax, 'still_moving')
        for ax in moving:
            with self._axis_state_lock:
                unknown = self._axis_state.get(ax) == AxisState.UNKNOWN
                move = self._current_move.get(ax)
            fault = move.fault if move is not None else None
            if unknown:
                raise fault if fault is not None else MoveNotCompletedError(ax, 'faulted')

    def _set_axis_state(
        self,
        axis: str,
        state: str,
        *,
        verdict_for: _Move | None = None,
        armed_only: bool = False,
        move: _Move | None = None,
        fault: MoveNotCompletedError | None = None,
        stop_generation: int | None = None,
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
        'faulted'; MOVING or HOMING settles it 'superseded', or 'stopped'
        when a STOP was taken since the move began. MOVING is
        written only with the new ``move`` (``_begin_move``), which becomes
        the current one; HOMING leaves none.

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
            if old_state == AxisState.MOVING and current is not None:
                if state == AxisState.IDLE:
                    stopped = stop_generation is not None and (
                        stop_generation != current.stop_generation
                    )
                    current.settle('stopped' if stopped else 'arrived')
                elif state == AxisState.UNKNOWN:
                    current.settle('faulted', fault or MoveNotCompletedError(axis, 'faulted'))
                else:
                    # A STOP the board took before this write halted the
                    # move, though the monitor has not yet seen it at rest.
                    stopped = self._stop_generation != current.stop_generation
                    current.settle('stopped' if stopped else 'superseded')
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

        self._fire_position_listeners(axis)
        return True

    def _backlash_leg(
        self, axis: str, position: float, overshoot_enabled: bool, member: str
    ) -> float | None:
        """Where a move's backlash leg goes, or None when it has none.

        A Z move down to a target clear of the bottom first drives to the
        backlash below it, so the backlash is always taken the same way.
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
        if current > position and position > backlash + 50:
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
                    raise HardwareError(
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
        move = _Move(stop_generation)
        self._set_axis_state(axis, AxisState.MOVING, move=move)
        return move

    def _send_drive(self, axis: str, stop_generation: int, send) -> tuple[_Move, bool]:
        """Set ``axis`` MOVING and disarmed, then send its drive.

        The one way a move body reaches the driver. The move is motion from
        its first target write, so every reader -- the waits, the state,
        the positions, frame validity -- sees it from here, a Z backlash
        leg included. Until ``_publish_drive`` arms the axis, the board's
        reached bit is the previous target's, or the leg's, and the monitor
        writes nothing for the axis: it polls and refreshes the position,
        and runs no stall clock. A raise makes the axis UNKNOWN
        (``_fail_drive``), so no exit ends MOVING and disarmed.

        Returns:
            The move's record, and what ``send`` returned: whether the
            target was written.

        Raises:
            MoveNotCompletedError: the drive raised; see ``_fail_drive``.
        """
        move = self._begin_move(axis, stop_generation)
        try:
            self._scope.imaging.frame_validity.invalidate(
                self._AXIS_VALIDITY_SOURCE.get(axis, 'xy_move')
            )
            return move, send()
        except Exception as e:
            self._fail_drive(axis, move, e)

    def _publish_drive(self, axis: str, written: bool, target_pos: float) -> None:
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
        fault = MoveNotCompletedError(axis, reason)
        with self._axis_state_lock:
            if verdict_for is not None and (
                self._axis_state.get(axis) != AxisState.MOVING
                or self._current_move.get(axis) is not verdict_for
                or not verdict_for.armed
            ):
                return False
        notifications.report_outcome(fault, solicited=False, category='Motion')
        return self._set_axis_state(
            axis,
            AxisState.UNKNOWN,
            verdict_for=verdict_for,
            armed_only=verdict_for is not None,
            fault=fault,
        )

    def _motion_monitor_loop(self):
        """Background thread: polls firmware for axis arrival at 50 Hz.

        Sleeps on ``_motion_wake`` when all axes are IDLE. Wakes when any
        axis transitions to MOVING. Polls ``get_target_status()`` per
        MOVING axis and transitions them to IDLE on arrival. This is the
        single place where firmware target-status queries happen during
        normal operation -- all other code reads the in-memory axis state.
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

                if not moving_axes:
                    # All axes arrived -- go back to sleep
                    self._motion_wake.clear()
                    break

                # Query firmware for each MOVING axis
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
                        try:
                            arrived = self.get_target_status(ax)
                        except Exception as e:
                            logger.warning(
                                f'[SCOPE API ] Motion monitor: target_status({ax}) failed: {e}'
                            )
                            continue
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
                        try:
                            actual = self._driver.current_pos(ax)
                            with self._pos_cache_lock:
                                self._pos_cache[ax] = float(actual)
                            read = True
                        except HardwareError as e:
                            if self._unread_warned.get(ax) is not noted:
                                self._unread_warned[ax] = noted
                                _api_log.warning(
                                    f'motion monitor: {ax} position read failed ({e}); '
                                    f'retrying until the motion bound'
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
                            )
                        ):
                            # Its clock goes with it: a back-to-back move
                            # landing within one poll interval starts its own.
                            self._moving_since.pop(ax, None)
                            continue
                        if armed and not (arrived and read):
                            # Still moving per firmware. A connected axis that
                            # stays not-arrived past the published motion bound
                            # is stalled: position_reached will never fire, so
                            # nothing else can ever clear it. Fault it to
                            # UNKNOWN (terminal; fires the arrival event so
                            # waiters and state-readers unblock, and the
                            # settle-check treats UNKNOWN as settled) and tell
                            # the user once -- the same shape as the disconnect
                            # fault. The clock is the move's own, kept with the
                            # move and run only while it is armed, and
                            # it lives HERE, after the arrival check, so an
                            # arriving report always wins over the stall
                            # verdict. An axis the board says arrived but
                            # whose position it will not report runs the same
                            # clock, and is given up as that, not as a stall.
                            since = self._moving_since.get(ax)
                            if since is None or since[0] is not noted:
                                since = (noted, time.monotonic())
                                self._moving_since[ax] = since
                            if time.monotonic() - since[1] > self._MOTION_SETTLE_TIMEOUT_S:
                                self._moving_since.pop(ax, None)
                                reason = 'position_unread' if arrived else 'stalled'
                                if self._give_axis_up(ax, reason, verdict_for=noted):
                                    continue
                        # No verdict this poll (still moving, a drive in
                        # flight, or the reached bit was another move's):
                        # propagate the refreshed cache value to UI listeners.
                        self._fire_position_listeners(ax)

                time.sleep(self._MOTION_POLL_INTERVAL)
