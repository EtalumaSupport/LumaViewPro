# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""MotionAPI -- sub-API for stage / focus / turret motion.

MotionAPI owns the motion state slots (_pos_cache, _axis_state,
_arrival_events, _move_profile, _position_listeners, _motion_wake,
_motion_monitor_stop, _motion_monitor_thread, _homing_event,
_turreting_event) and the bodies of all stage / focus / turret
methods. Lumascope keeps a small set of one-line method-name
forwarders (zhome, home, thome, move_absolute_position, etc.) for
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
from typing import TYPE_CHECKING
from collections.abc import Iterator

from drivers.exceptions import HardwareError
from lib import profile_trace
from lvp_logger import logger
from modules.notification_center import notifications

# Match _lumascope.py's module-level _api_log channel so relocated
# bodies log to the same handler chain.
_api_log = _logging.getLogger('LVP.api')

from modules.lumascope_api._constants import (
    AxisState,
    MOTOR_POSITION_LIMIT,
    _VALID_AXIS_NAMES,
)

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import MotorBoardProtocol


class MotionAPI:
    """Motion sub-API. Hosts stateless (Phase 2b) and stateful (Phase 2c) bodies."""

    _MOTION_POLL_INTERVAL = 0.02  # 50 Hz

    # Maps a motion axis to its frame-validity source. X and Y share
    # 'xy_move'; Z and the turret each have their own source so the
    # settle-check gates on the correct axis reaching IDLE. A turret
    # move that recorded 'xy_move' would clear the moment X/Y read idle,
    # before the turret physically finished.
    _AXIS_VALIDITY_SOURCE = {'Z': 'z_move', 'T': 'turret'}

    def __init__(self, scope: Lumascope, driver: MotorBoardProtocol) -> None:  # noqa: ARG002
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
        # populated by init_axes() called from Lumascope.__init__ after the
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

        # Lock for motion profile dict (built by init_axes, after driver init).
        self._move_profile_lock = threading.Lock()

        # Boolean operation flags use threading.Event for wait/signal.
        self._homing_event = threading.Event()  # set => homing in progress
        self._turreting_event = threading.Event()  # set => turret move in progress

        # Motion monitor thread handle -- populated by start_monitor().
        # Not started at __init__ because the motion driver and per-axis
        # dicts aren't ready yet; Lumascope.__init__ calls start_monitor()
        # after init_axes().
        self._motion_monitor_stop = threading.Event()
        self._motion_monitor_thread: threading.Thread | None = None

        # Per-axis state dicts -- empty until init_axes() fills them.
        self._pos_cache: dict = {}
        self._axis_state: dict = {}
        self._arrival_events: dict = {}
        self._move_profile: dict = {}

        # Last turret position cache -- tmove() short-circuits a same-
        # position request to avoid a no-op move command. Defaults to None
        # so the first tmove() always goes through to the firmware.
        self._last_turret_position: int | None = None

    def init_axes(self, present_axes: list[str]) -> None:
        """Populate per-axis state dicts from the list of detected axes.

        Called from Lumascope.__init__ (and create_diagnostic) after the
        motion driver's detect_present_axes() has run. NullMotionBoard
        returns [] so a system with no motor hardware ends up with empty
        dicts -- all state-touching methods handle that via no-ops.

        Args:
            present_axes: List of axis names the hardware actually has.
        """
        self._pos_cache = dict.fromkeys(present_axes, 0.0)
        self._axis_state = dict.fromkeys(present_axes, AxisState.UNKNOWN)
        self._arrival_events = {ax: threading.Event() for ax in present_axes}
        for ev in self._arrival_events.values():
            ev.set()  # Start as "arrived" (not moving)
        self._move_profile = dict.fromkeys(present_axes)

    def start_monitor(self) -> None:
        """Spawn the motion monitor thread.

        Called from Lumascope.__init__ after init_axes() so the thread
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

    def disconnect(self) -> None:
        """Stop the motion monitor and reset axis states.

        Called from Lumascope.disconnect() before the motor driver is
        swapped to NullMotionBoard. Sets all arrival events so any blocked
        waiters unblock cleanly.
        """
        self._motion_monitor_stop.set()
        self._motion_wake.set()  # unblock if sleeping
        if self._motion_monitor_thread is not None and self._motion_monitor_thread.is_alive():
            self._motion_monitor_thread.join(timeout=1.0)

        with self._axis_state_lock:
            for ax in self._axis_state:
                self._axis_state[ax] = AxisState.UNKNOWN
        for ev in self._arrival_events.values():
            ev.set()

    @property
    def _driver(self) -> MotorBoardProtocol:
        return self._scope._motion_driver

    # ------------------------------------------------------------------
    # Stateless method bodies.
    #
    # Order mirrors _lumascope.py source order.
    # ------------------------------------------------------------------

    def move_absolute_async(
        self,
        axis,
        pos,
        *,
        wait_until_complete=False,
        overshoot_enabled=True,
        callback=None,
        cb_kwargs=None,
    ) -> None:
        """Submit ``move_absolute_position`` to the io_executor.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            pos: Target position in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        from modules.sequential_io_executor import IOTask  # local-import: avoid cycle

        ex = self._scope._require_executor(self._scope._io_executor, 'move_absolute_async')
        ex.put(
            IOTask(
                action=self.move_absolute_position,
                kwargs={
                    'axis': axis,
                    'pos': pos,
                    'wait_until_complete': wait_until_complete,
                    'overshoot_enabled': overshoot_enabled,
                },
                callback=callback,
                cb_kwargs=cb_kwargs,
            )
        )

    def stop_motion(self) -> None:
        """Stop all in-flight motor moves (LVP-A-1).

        Idempotent + safe-when-disconnected -- no-ops when the motor
        board isn't connected. Uses the firmware-side
        ``STOP`` command which the motor controller implements as
        ``motorstop`` (target=actual on all axes); same wire command the
        UI emergency-stop already uses, just routed through the API
        instead of an inline ``motion.exchange_command('STOP')``.

        Called as the first step of ``disconnect()`` so every disconnect
        path (App on_stop, REST shutdown, test teardown, future CLI
        tools) stops motors before tearing down the serial port.
        """
        if not self._scope.motor_connected:
            return
        try:
            # Route through MotorBoard.motor_stop so field firmware
            # (2024-09-10 EL-0940-02, no STOP command) silently no-ops
            # instead of producing two FIRMWARE ERROR warnings per
            # shutdown. motor_stop returns True if STOP was accepted,
            # False if firmware doesn't implement it (cached).
            stopped = self._driver.motor_stop()
            if stopped:
                logger.info('[SCOPE API ] stop_motion: motors stopped')
            else:
                logger.debug(
                    '[SCOPE API ] stop_motion: firmware does not '
                    'implement STOP; motors will latch on disconnect'
                )
        except Exception as e:
            # Log + notify, but don't re-raise: stop_motion is called
            # from shutdown paths where the caller can't meaningfully
            # recover and a raised exception would leave disconnect()
            # half-done.
            logger.warning(f'[SCOPE API ] stop_motion failed: {type(e).__name__}: {e}')
            try:
                notifications.warning(
                    'Motion',
                    'Motor stop failed',
                    f'STOP command failed during shutdown: {type(e).__name__}: {e}',
                )
            except Exception:
                pass

    def get_turret_position_for_objective_id(
        self,
        objective_id: str,
        prefer_current: bool = True,
        persisted_position: int | None = None,
    ) -> int | None:
        """Find the turret position holding a given objective.

        Lookup ranking when multiple positions hold the same objective (#488):
            1. Persisted position from settings, if it matches objective_id
               and is provided by the caller. Honors the user's most
               recent explicit choice -- survives restarts and post-home
               situations where the current physical position is an
               artifact of the home routine (T zeros to 1), not user
               intent.
            2. Current physical T position, if it matches objective_id.
               Catches the case where the user has already rotated to a
               matching slot in this session and no persisted hint exists.
            3. First-match dict iteration (lowest position with the
               objective). Used when neither hint is available -- preserves
               today's fallback behavior.

        Args:
            objective_id: Objective identifier to search for.
            prefer_current: If True (default), check the current physical
                turret position when persisted_position is unavailable
                or doesn't match.
            persisted_position: Caller-supplied hint, typically
                ``settings.get('turret_position')``. None disables this
                tier of the lookup.

        Returns:
            int | None: Turret position (1-4), or None if not found.
        """
        if persisted_position is not None:
            if self._scope.runtime_state._turret_config.get(persisted_position) == objective_id:
                return persisted_position

        if prefer_current:
            try:
                current_pos = self.get_current_position(axis='T')
                if self._scope.runtime_state._turret_config.get(current_pos) == objective_id:
                    return current_pos
            except Exception:
                pass

        for (
            turret_position,
            turret_objective_id,
        ) in self._scope.runtime_state._turret_config.items():
            if objective_id == turret_objective_id:
                return turret_position

        return None

    def is_current_turret_position_objective_set(self) -> bool:
        """Check whether the objective slot at the current turret position is set.

        Returns:
            bool: True if the current turret position has a configured
                objective ID; False if the slot is unconfigured.
        """
        position = self.get_current_position(axis='T')
        return self._scope.runtime_state._turret_config[position] is not None

    def get_axes_config(self) -> dict:
        """Get the axis configuration from the motion board.

        Returns:
            dict: Axis configuration (axes present, limits, etc.).
        """
        return self._driver.get_axes_config()

    @contextlib.contextmanager
    def reference_position_logger(self) -> Iterator[None]:
        """Context manager that logs limit-switch status before and after homing.

        Use as ``with scope.motion.reference_position_logger(): ... home ...``.
        Emits forced-INFO log lines so the limit-switch state pre/post
        homing is preserved for diagnostics.
        """
        before = self.get_limit_switch_status_all_axes()
        logger.info(f'Limit switch status before homing: {before}', extra={'force_error': True})
        yield
        after = self.get_limit_switch_status_all_axes()
        logger.info(f'Limit switch status after homing: {after}', extra={'force_error': True})

    def home(self) -> bool:
        """Home every axis the motor board has.

        This is the unified "home everything" entry point used by
        startup and the GUI Home button. The firmware's home routine
        homes Z, then T, then X/Y -- on a Z-only board (LS820) it homes
        Z and reports the missing X/Y; on a full XYZ scope it homes
        all three. The driver returns True for both cases (full and
        partial), raises HardwareError on real failure.

        Returns:
            bool: True on full or partial success. False if the motor
                is not connected, the driver returned False, or the
                driver raised (HardwareError or other). The user is
                notified on failure; programmatic callers can branch on
                the bool.
        """
        # Short-circuit on disconnected motor -- without this, home()
        # dispatches into the driver where exchange_command tries to
        # auto-reconnect and burns its full timeout (~10 s). That was
        # the user-perceived "spinning beachball" in #632. Fire ONE
        # clean notification with the right cause, instead of the
        # misleading "Homing Failed. Position is unknown" that implies
        # a homing-mechanics problem.
        if not self._scope.motor_connected:
            logger.warning('[SCOPE API ] home() called with motor not connected')
            # Suppress the per-component popup when the scope is in
            # no_hardware mode -- lumaviewpro.on_start fires a single
            # consolidated "No hardware detected" popup that covers
            # the missing motor.
            if not getattr(self._scope, 'no_hardware', False):
                notifications.error(
                    'Motion',
                    'Motor Not Connected',
                    'Cannot home -- motor controller is not connected. '
                    'Check the USB cable and that no other program '
                    '(Thonny, mpremote, etc.) is holding the port.',
                )
            return False
        present_axes = self._scope.capabilities.axes
        _api_log.info('home START')
        for ax in present_axes:
            self._set_axis_state(ax, AxisState.HOMING)
        if 'Z' in present_axes:
            self._scope.imaging.frame_validity.invalidate('z_move')
        if 'X' in present_axes or 'Y' in present_axes:
            self._scope.imaging.frame_validity.invalidate('xy_move')
        if 'T' in present_axes:
            self._scope.imaging.frame_validity.invalidate('turret')
        self.is_homing = True
        try:
            with self.reference_position_logger():
                result = self._driver.home()
            if result is False:
                logger.error('[SCOPE API ] Homing failed')
                notifications.error(
                    'Motion', 'Homing Failed', 'Homing failed. Position is unknown.'
                )
                for ax in present_axes:
                    self._set_axis_state(ax, AxisState.UNKNOWN)
                return False
            for ax in present_axes:
                self._set_axis_state(ax, AxisState.IDLE)
            self.refresh_position_cache()
            # The firmware homes the turret to position 1, so seed the cache.
            # Without this it stays None and a subsequent tmove(1) -- e.g. the
            # startup select-position-1 -- can't recognize the turret is
            # already there, and runs a redundant Z-retract / rotate / restore.
            if 'T' in present_axes:
                self._last_turret_position = 1
            return True
        except Exception:
            logger.exception('[SCOPE API ] Homing exception')
            for ax in present_axes:
                self._set_axis_state(ax, AxisState.UNKNOWN)
            notifications.error(
                'Motion', 'Homing Error', 'Homing encountered an error. Position is unknown.'
            )
            return False
        finally:
            self.is_homing = False
            _api_log.info('home DONE')

    @contextlib.contextmanager
    def safe_turret_move(self, restore_z: bool = True) -> Iterator[None]:
        """Context manager that lowers Z to 0 before turret motion and restores after.

        Use as ``with scope.motion.safe_turret_move(): ... move turret ...``.
        Sets ``is_turreting`` for the duration and restores the original
        Z position even if the body raises.

        Args:
            restore_z: When True (default), restore the original Z
                position on exit. Set to False when the immediate next
                operation will overwrite Z anyway (e.g. protocol
                step-navigation moves T then immediately moves Z to the
                step's target -- the restore is wasted motion). When
                False, Z is left at 0 and the caller is responsible for
                the next Z move. Standalone callers (UI turret button,
                thome) leave the default True.
        """
        # Save off current Z position before moving Z to 0
        logger.info('[SCOPE API ] Moving Z to 0', extra={'force_error': True})
        initial_z = self.get_current_position(axis='Z')
        self.move_absolute_position('Z', pos=0, wait_until_complete=True)
        self.is_turreting = True
        try:
            yield
        finally:
            # Always clear the flag, even if the body raised (e.g. driver
            # HardwareError from thome). Without this, a failed turret
            # home would leave is_turreting=True and the stage stuck at
            # Z=0.
            self.is_turreting = False
            if restore_z:
                logger.info(
                    f'[SCOPE API ] Restoring Z to {initial_z}', extra={'force_error': True}
                )
                self.move_absolute_position('Z', pos=initial_z, wait_until_complete=True)
            else:
                logger.info(
                    '[SCOPE API ] Skipping Z restore -- caller will overwrite Z next',
                    extra={'force_error': True},
                )

    def thome(self) -> bool:
        """Home the turret axis. Moves Z to 0 during turret motion for safety.

        Returns:
            bool: True on successful turret homing (or when the board
                reports the turret is not present). False if the motor
                is not connected, the driver returned False, or the
                driver raised (HardwareError or other). The user is
                notified on failure; programmatic callers can branch on
                the bool.
        """
        # Short-circuit on disconnected motor -- same rationale as
        # home() above. Without this, thome dispatches into the driver
        # where exchange_command burns its 15s timeout doing failed
        # auto-reconnect attempts. Fire one clean notification.
        if not self._scope.motor_connected:
            logger.warning('[SCOPE API ] thome() called with motor not connected')
            if not getattr(self._scope, 'no_hardware', False):
                notifications.error(
                    'Motion',
                    'Motor Not Connected',
                    'Cannot home turret -- motor controller is not connected. '
                    'Check the USB cable and that no other program is '
                    'holding the port.',
                )
            return False

        # Move turret -- set HOMING after Z is safe, not before.
        # Setting T to HOMING clears its arrival event, which would block
        # wait_until_finished_moving() inside safe_turret_move's Z move.
        _api_log.info('thome START')
        try:
            with self.reference_position_logger():
                with self.safe_turret_move():
                    self._set_axis_state('T', AxisState.HOMING)
                    self._scope.imaging.frame_validity.invalidate('turret')
                    result = self._driver.thome()
                    # Transition T out of HOMING BEFORE exiting
                    # safe_turret_move. The context manager's finally
                    # restores Z via wait_until_complete=True, which calls
                    # wait_until_finished_moving and iterates EVERY axis
                    # arrival event. A still-HOMING T has a cleared event
                    # and the motion monitor only polls MOVING (not
                    # HOMING) axes, so the Z-restore's wait would hang on
                    # T until the 120s default timeout fires. Setting T
                    # to its post-homing state here keeps the Z restore
                    # blocked only on the axis that's actually moving.
                    if result is False:
                        self._set_axis_state('T', AxisState.UNKNOWN)
                    else:
                        self._set_axis_state('T', AxisState.IDLE)
            if result is False:
                logger.error('[SCOPE API ] Turret homing failed')
                notifications.error(
                    'Motion', 'Homing Failed', 'Turret homing failed. Position is unknown.'
                )
                return False
            self.refresh_position_cache()
            # Turret homes to position 1; seed the cache so a following
            # tmove(1) is a no-op rather than a redundant Z-retract / rotate /
            # restore (see home() for the full rationale).
            self._last_turret_position = 1
            _api_log.info('thome DONE')
            return True
        except Exception:
            logger.exception('[SCOPE API ] Turret homing exception')
            self._set_axis_state('T', AxisState.UNKNOWN)
            notifications.error(
                'Motion', 'Homing Error', 'Turret homing encountered an error. Position is unknown.'
            )
            _api_log.info('thome DONE')
            return False

    def has_thomed(self) -> bool:
        """Check if the turret has been homed since startup.

        Returns:
            bool: True if turret homing has been performed.
        """
        return self._driver.has_thomed()

    def tmove(self, position: int, restore_z: bool = True) -> None:
        """Move the turret to a specific position. Skips if already there.

        Args:
            position: Target turret position (1-4).
            restore_z: When True (default), restore the pre-move Z
                position after the turret move completes. Set to False
                when the caller will immediately set Z to a different
                value (e.g. protocol step navigation moves T then Z to
                the new step's target -- restoring Z first is wasted
                motion).
        """
        # Commanding a move of the T axis is slow, even if the move is to the current position.
        # Use caching to determine if T is requested to move to it's current position, and bypass the
        # move altogether if it is.
        if self._last_turret_position == position:
            return

        with self.safe_turret_move(restore_z=restore_z):
            logger.info(f'[SCOPE API ] Moving T to position {position}')
            self.move_absolute_position('T', position, wait_until_complete=True)
            self._last_turret_position = position

    def has_turret(self) -> bool:
        """Check if the microscope has a turret axis.

        Thin wrapper over ``self.capabilities.has_turret``.

        Returns:
            bool: True if the scope reports a turret axis.
        """
        return self._scope.capabilities.has_turret

    def get_actual_position(self, axis: str) -> float:
        """Query the actual hardware position via serial (not cached).

        Unlike get_current_position() which returns the last commanded
        target, this queries the motor controller for where it actually is
        right now. Use during continuous motion sweeps where the stage is
        moving and the cache doesn't reflect the true position.

        Costs one serial round-trip (~5ms).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Current position in um. 0 if motor not connected.
        """
        if not self._scope.motor_connected:
            return 0.0
        pos = self._driver.current_pos(axis)
        return pos if pos is not None else 0.0

    def set_precision_mode(self, axis: str, enabled: bool) -> None:
        """Set motor precision mode for an axis.

        Precision mode uses accurate but slightly slower motor stopping.
        Use before autofocus fine passes or any measurement requiring
        precise Z positioning. Disable for coarse moves where speed
        matters more than final position accuracy.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning, False for speed.
        """
        if not self._scope.motor_connected:
            return
        self._driver.set_precision_mode(axis, enabled)

    def get_home_status(self, axis: str) -> bool:
        """Check if an axis is at its home position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True if the axis is homed, False otherwise or on error.
        """
        if not self._scope.motor_connected:
            # Disconnected is an expected degradation, not an error -- the
            # motion monitor polls this; provoking + tracing a per-poll
            # HardwareError floods the log on a mid-move USB yank (#539).
            return False
        try:
            status = self._driver.home_status(axis)
            return status
        except HardwareError as e:
            # Typed disconnect/timeout at the moment of unplug (before
            # motor_connected flips). Expected; log without the traceback.
            logger.warning(
                f'[SCOPE API ] get_home_status({axis}): {e}; treating as not home'
            )
            return False
        except Exception as e:
            logger.exception(
                f'[SCOPE API ] get_home_status({axis}) failed; treating as not home: {e}'
            )
            return False

    def get_target_status(self, axis: str) -> bool:
        """Check if an axis has reached its target position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True if at target (always True for T if no turret present).
        """
        if not self._scope.motor_connected:
            # See get_home_status: disconnected is an expected degradation;
            # the motion monitor polls this, so don't provoke + trace a
            # per-poll HardwareError on a mid-move USB yank (#539).
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

    def get_reference_status(self, axis: str) -> str:
        """Get reference status register bits for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            str: 32-character binary string of register bits (MSB first).
        """
        return self._driver.reference_status(axis=axis)

    def get_limit_switch_status(self, axis: str) -> tuple[int, int]:
        """Get the limit switch status for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            Limit switch state for the specified axis (driver-defined).
        """
        return self._driver.limit_switch_status(axis=axis)

    def get_limit_switch_status_all_axes(self) -> dict:
        """Get limit switch status for all axes.

        Returns:
            dict: Mapping of axis name to limit switch state.
        """
        resp = {}
        for axis in self._scope.capabilities.axes:
            resp[axis] = self.get_limit_switch_status(axis=axis)
        return resp

    def get_overshoot(self) -> bool:
        """Check if the Z axis is currently in overshoot (backlash compensation) mode.

        Returns:
            bool: True if overshoot is in progress.
        """
        return self._driver.overshoot

    def is_moving(self) -> bool:
        """Check if any axis is currently moving.

        Reads from in-memory axis state -- zero serial I/O. The motion
        monitor thread handles firmware queries and state transitions.

        Returns:
            bool: True if any axis is MOVING/HOMING or overshoot is active.
        """
        if self.is_any_axis_moving():
            return True
        return bool(self.get_overshoot())

    def set_acceleration_limit(self, val_pct: int) -> None:
        """Set the motor controller acceleration limit (percent of max).

        Silently ignores firmware that doesn't implement the command --
        legacy boards lack the acceleration-limits feature.

        Args:
            val_pct: Acceleration limit as a percent of the firmware max.
        """
        try:
            self._driver.set_acceleration_limits(val_pct=val_pct)
        except Exception:
            pass  # Legacy firmware doesn't support acceleration limits

    # ------------------------------------------------------------------
    # Stateful method bodies.
    #
    # State slots (_pos_cache, _axis_state, _arrival_events, _move_profile,
    # _position_listeners, _motion_wake, _motion_monitor_*, _homing_event,
    # _turreting_event) live on this surface.
    # ------------------------------------------------------------------

    # --- CR-2: Thread-safe properties for shared state ---

    @property
    def is_homing(self) -> bool:
        """True while the microscope is homing.

        Returns:
            bool: True if a homing operation is in progress.
        """
        return self._homing_event.is_set()

    @is_homing.setter
    def is_homing(self, value: bool) -> None:
        """Set the homing-in-progress flag."""
        if value:
            self._homing_event.set()
        else:
            self._homing_event.clear()

    @property
    def is_turreting(self) -> bool:
        """True while the turret is moving.

        Returns:
            bool: True if a turret motion is in progress.
        """
        return self._turreting_event.is_set()

    @is_turreting.setter
    def is_turreting(self, value: bool) -> None:
        """Set the turret-motion-in-progress flag."""
        if value:
            self._turreting_event.set()
        else:
            self._turreting_event.clear()

    def move_absolute_sync(
        self, axis, pos, *, wait_until_complete=True, overshoot_enabled=True, timeout_s=30
    ) -> None:
        """Run ``move_absolute_position`` through the io_executor and block.

        Blocks until both the IOTask completes and (when
        ``wait_until_complete``) the stage has physically arrived.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            pos: Target position in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            timeout_s: Max seconds to wait for completion.
        """
        from modules.sequential_io_executor import IOTask  # local-import: avoid cycle

        ex = self._scope._require_executor(self._scope._io_executor, 'move_absolute_sync')
        task = IOTask(
            action=self.move_absolute_position,
            kwargs={
                'axis': axis,
                'pos': pos,
                'wait_until_complete': wait_until_complete,
                'overshoot_enabled': overshoot_enabled,
            },
        )
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout_s)

    def move_relative_async(
        self,
        axis,
        um,
        *,
        wait_until_complete=False,
        overshoot_enabled=True,
        callback=None,
        cb_kwargs=None,
    ) -> None:
        """Submit ``move_relative_position`` to the io_executor.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            um: Distance to move in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        from modules.sequential_io_executor import IOTask  # local-import: avoid cycle

        ex = self._scope._require_executor(self._scope._io_executor, 'move_relative_async')
        ex.put(
            IOTask(
                action=self.move_relative_position,
                kwargs={
                    'axis': axis,
                    'um': um,
                    'wait_until_complete': wait_until_complete,
                    'overshoot_enabled': overshoot_enabled,
                },
                callback=callback,
                cb_kwargs=cb_kwargs,
            )
        )

    def move_home_async(self, axis, *, callback=None, cb_args=None) -> None:
        """Home an axis (or the whole scope) via the io_executor.

        Args:
            axis: 'Z' or 'T' homes that single axis. 'ALL' (or legacy 'XY')
                homes everything the board has via self.home() -- firmware
                homes Z and T first as part of the same routine.
            callback: Optional completion callback.
            cb_args: Optional positional args passed to the callback.
        """
        from modules.sequential_io_executor import IOTask  # local-import: avoid cycle

        ex = self._scope._require_executor(self._scope._io_executor, 'move_home_async')
        a = axis.upper()
        # Homing legitimately takes 10-60+ seconds depending on travel
        # distance and starting position -- well above the 5 sec default
        # slow-task threshold. Bump to 120s; only a true stall warrants
        # a warning here.
        HOME_THRESHOLD = 120.0
        if a == 'Z':
            ex.put(
                IOTask(
                    action=self.zhome,
                    callback=callback,
                    cb_args=cb_args,
                    slow_task_threshold_sec=HOME_THRESHOLD,
                )
            )
        elif a in ('ALL', 'XY'):
            ex.put(
                IOTask(
                    action=self.home,
                    callback=callback,
                    cb_args=cb_args,
                    slow_task_threshold_sec=HOME_THRESHOLD,
                )
            )
        elif a == 'T':
            ex.put(
                IOTask(
                    action=self.thome,
                    callback=callback,
                    cb_args=cb_args,
                    slow_task_threshold_sec=HOME_THRESHOLD,
                )
            )
        else:
            logger.warning(f'[SCOPE API ] Unknown home axis: {axis}')

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
                _api_log.debug(f'position listener error: {ex}')

    def is_any_axis_moving(self) -> bool:
        """Check if any axis is currently MOVING or HOMING.

        Reads from the in-memory state dict -- zero serial I/O.

        Returns:
            bool: True if any axis is in MOVING or HOMING state.
        """
        with self._axis_state_lock:
            return any(s in (AxisState.MOVING, AxisState.HOMING) for s in self._axis_state.values())

    def get_axis_limits(self, axis: str) -> dict | None:
        """Get the travel limits for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", or "T").

        Returns:
            dict with 'min' and 'max' positions in um, or ``None`` if
            the axis has no configured limits (typical for the turret
            T axis). Callers must handle the None case.
        """
        return self._driver.get_axis_limits(axis=axis)

    def zhome(self) -> bool:
        """Home the Z axis (focus).

        Returns:
            bool: True on successful Z homing. False if the driver
                returned False or raised (e.g. HardwareError on
                no-response / firmware-error). The user is notified on
                failure; programmatic callers can branch on the bool.
        """
        _api_log.info('zhome START')
        self._set_axis_state('Z', AxisState.HOMING)
        self._scope.imaging.frame_validity.invalidate('z_move')
        try:
            with self.reference_position_logger():
                result = self._driver.zhome()
            if result is False:
                logger.error('[SCOPE API ] Z homing failed')
                notifications.error(
                    'Motion', 'Homing Failed', 'Z axis homing failed. Position is unknown.'
                )
                self._set_axis_state('Z', AxisState.UNKNOWN)
                return False
            self._set_axis_state('Z', AxisState.IDLE)
            self.refresh_position_cache()
            _api_log.info('zhome DONE')
            return True
        except Exception:
            logger.exception('[SCOPE API ] Z homing exception')
            self._set_axis_state('Z', AxisState.UNKNOWN)
            notifications.error(
                'Motion', 'Homing Error', 'Z axis homing encountered an error. Position is unknown.'
            )
            _api_log.info('zhome DONE')
            return False

    def has_homed(self) -> bool:
        """Check if the scope has been homed since startup.

        Returns:
            bool: True if home() has succeeded at least once.
        """
        return self._driver.has_homed()

    def xycenter(self) -> None:
        """Move the XY stage to center position."""
        self._set_axis_state('X', AxisState.MOVING)
        self._set_axis_state('Y', AxisState.MOVING)
        self._driver.xycenter()
        self._set_axis_state('X', AxisState.IDLE)
        self._set_axis_state('Y', AxisState.IDLE)
        # XY field of view changed -- the camera pipeline still holds
        # frames from the old position; hold capture until they flush.
        self._scope.imaging.frame_validity.invalidate('xy_move')
        self.refresh_position_cache()

    def refresh_position_cache(self) -> None:
        """Fetch all axis positions from hardware and update the cache.

        Called after homing completes to sync the cache with actual hardware
        positions. During normal operation the cache is updated directly
        by move commands -- no polling needed.
        """
        positions = {}
        for ax in self._scope.capabilities.axes:
            try:
                pos = self._driver.target_pos(axis=ax)
                positions[ax] = pos if pos is not None else 0.0
            except Exception:
                positions[ax] = 0.0

        with self._pos_cache_lock:
            self._pos_cache.update(positions)
        for ax in positions:
            self._fire_position_listeners(ax)

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
        """Get the target position for an axis (where it is commanded to go).

        During MOVING: returns the target captured in _move_profile when the
        move was commanded. This is what the host told the chip; no serial
        I/O. During IDLE: returns the cached current position (which is the
        last polled motor position, ~1 microstep off the commanded target).

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict: Position in um for a single axis, or dict of all
                axis positions. Returns 0 if motion board inactive, None if
                axis T requested but no turret present.
        """
        if axis is None:
            result = {}
            for ax in self._scope.capabilities.axes:
                result[ax] = self.get_target_position(ax)
            return result
        if axis == 'T' and not self._driver.has_turret():
            return None
        with self._axis_state_lock:
            state = self._axis_state.get(axis, AxisState.UNKNOWN)
        if state == AxisState.MOVING:
            with self._move_profile_lock:
                profile = self._move_profile.get(axis)
            if profile is not None and profile.get('target_pos') is not None:
                return float(profile['target_pos'])
        return self._read_position_cache(axis)

    def get_current_position(self, axis: str | None = None) -> float | dict:
        """Get the current position for an axis.

        Reads from the in-memory position cache. During MOVING the cache
        is refreshed by _motion_monitor_loop polling the motor's actual
        position from hardware on every cycle; during IDLE the cache
        holds the last confirmed position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T"), or None for all axes.

        Returns:
            float | dict: Position in um for a single axis, or dict of all
                axis positions. Returns 0 if motion board inactive.
        """
        if axis is None:
            result = {}
            for ax in self._scope.capabilities.axes:
                result[ax] = self.get_current_position(ax)
            return result
        return self._read_position_cache(axis)

    def _predicted_position(self, axis: str) -> float | None:
        """Predict position during a move using the trapezoidal ramp profile.

        Returns None if no motion profile is available (falls back to cache).
        Supports simple trapezoidal (a1/v1/d1=0) and 6-point ramps.
        """
        with self._move_profile_lock:
            profile = self._move_profile.get(axis)
            if profile is None:
                return None
            start_time = profile['start_time']
            start_pos = profile['start_pos']
            target_pos = profile['target_pos']
            ramp = profile['ramp']

        elapsed = time.monotonic() - start_time
        distance = abs(target_pos - start_pos)
        if distance < 0.01:  # trivially short move
            return target_pos
        direction = 1.0 if target_pos > start_pos else -1.0

        vmax = ramp['vmax']
        amax = ramp['amax']
        dmax = ramp['dmax']
        if amax <= 0 or dmax <= 0 or vmax <= 0:
            return None  # invalid ramp params

        # Simple trapezoidal profile (a1/v1/d1 are zero)
        t_accel = vmax / amax
        t_decel = vmax / dmax
        s_accel = 0.5 * amax * t_accel * t_accel
        s_decel = 0.5 * dmax * t_decel * t_decel

        if distance <= (s_accel + s_decel):
            # Triangular profile -- never reaches VMAX
            import math

            t_peak = math.sqrt(2.0 * distance / (amax + amax * amax / dmax))
            v_peak = amax * t_peak
            s_accel_tri = 0.5 * amax * t_peak * t_peak
            t_decel_tri = v_peak / dmax
            total_time = t_peak + t_decel_tri

            if elapsed >= total_time:
                return target_pos
            elif elapsed <= t_peak:
                s = 0.5 * amax * elapsed * elapsed
            else:
                dt = elapsed - t_peak
                s = s_accel_tri + v_peak * dt - 0.5 * dmax * dt * dt
        else:
            # Full trapezoidal profile
            s_cruise = distance - s_accel - s_decel
            t_cruise = s_cruise / vmax
            total_time = t_accel + t_cruise + t_decel

            if elapsed >= total_time:
                return target_pos
            elif elapsed <= t_accel:
                s = 0.5 * amax * elapsed * elapsed
            elif elapsed <= (t_accel + t_cruise):
                dt = elapsed - t_accel
                s = s_accel + vmax * dt
            else:
                dt = elapsed - t_accel - t_cruise
                s = s_accel + s_cruise + vmax * dt - 0.5 * dmax * dt * dt

        # Clamp to [start, target] -- never overshoot in prediction
        s = max(0.0, min(s, distance))
        return start_pos + direction * s

    def move_absolute_position(
        self,
        axis: str,
        pos: float,
        wait_until_complete: bool = False,
        overshoot_enabled: bool = True,
        ignore_limits: bool = False,
    ) -> None:
        """Move an axis to an absolute position.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            pos (float): Target position in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.
            ignore_limits: If True, skip software limit checks.

        Raises:
            ValueError: If axis is invalid or pos is not numeric / out of bounds.
        """
        if axis not in _VALID_AXIS_NAMES:
            raise ValueError(f'Axis must be one of {_VALID_AXIS_NAMES}, got {axis!r}')
        if not isinstance(pos, (int, float)):
            raise ValueError(f'Position must be numeric, got {type(pos).__name__}')
        if abs(pos) > MOTOR_POSITION_LIMIT:
            raise ValueError(
                f'Position {pos} um exceeds safety limit of +/-{MOTOR_POSITION_LIMIT} um'
            )

        # Silently no-op for axes that aren't present on this hardware.
        # _arrival_events is sized to detect_present_axes() at init,
        # so this is the canonical "is this axis trackable" check.
        if axis not in self._arrival_events:
            _api_log.debug(f'move_abs ignored: {axis} not present on this scope')
            return

        # Capture start_pos + ramp before driving. start_time is captured
        # AFTER the driver call returns -- the serial round-trip to write
        # the hardware target takes ~50 ms, during which the motor has not
        # begun physical motion yet. If start_time were captured BEFORE the
        # driver call, _predicted_position's `elapsed` would lead the motor's
        # real elapsed by the full serial RT latency, and the UI crosshair
        # would visibly outrun the stage on long moves.
        with self._pos_cache_lock:
            start_pos = self._pos_cache.get(axis, 0.0)
        try:
            ramp = self._driver.motorconfig.ramp_params(axis)
        except Exception:
            ramp = None

        # Write the hardware target BEFORE transitioning the axis to MOVING.
        # Previously the order was reversed: _set_axis_state(MOVING) cleared
        # the arrival event and woke the motion monitor, then motion.move_abs_pos
        # spent ~50ms on serial I/O (current_pos read + TARGET_W write) before
        # the hardware actually received the new target. During that window
        # the motion monitor could poll STATUS_R, observe the PRIOR move's
        # still-valid position_reached bit, and falsely set the arrival
        # event -- causing wait_until_finished_moving to return before the
        # new move even began. See issue #618. With this order, by the
        # time the axis is marked MOVING the hardware XTARGET is already
        # the new value, so position_reached is reliably False and the
        # motion monitor polls until real arrival.
        try:
            self._driver.move_abs_pos(
                axis, pos, overshoot_enabled=overshoot_enabled, ignore_limits=ignore_limits
            )
        except Exception:
            _api_log.error(f'move_abs {axis}={pos:.1f}um FAILED')
            raise
        if ramp:
            with self._move_profile_lock:
                self._move_profile[axis] = {
                    'start_time': time.monotonic(),
                    'start_pos': start_pos,
                    'target_pos': float(pos),
                    'ramp': ramp,
                }
        self._set_axis_state(axis, AxisState.MOVING)
        # No move-init cache write: cache holds CURRENT position, which is
        # still start_pos until _motion_monitor_loop reads it from hardware
        # on its first cycle. Target is held in _move_profile[axis], where
        # get_target_position picks it up during MOVING.
        self._fire_position_listeners(axis)
        self._scope.imaging.frame_validity.invalidate(self._AXIS_VALIDITY_SOURCE.get(axis, 'xy_move'))
        _api_log.info(f'move_abs {axis}={pos:.1f}um{" wait" if wait_until_complete else ""}')

        if wait_until_complete is True:
            self.wait_until_finished_moving()
            self._set_axis_state(axis, AxisState.IDLE)

    def move_relative_position(
        self,
        axis: str,
        um: float,
        wait_until_complete: bool = False,
        overshoot_enabled: bool = False,
    ) -> None:
        """Move an axis by a relative distance.

        Args:
            axis (str): Axis name ("X", "Y", "Z", "T").
            um (float): Distance to move in um.
            wait_until_complete: If True, block until move finishes.
            overshoot_enabled: Allow Z overshoot for backlash compensation.

        Raises:
            ValueError: If axis is invalid or um is not numeric / out of bounds.
        """
        if axis not in _VALID_AXIS_NAMES:
            raise ValueError(f'Axis must be one of {_VALID_AXIS_NAMES}, got {axis!r}')
        if not isinstance(um, (int, float)):
            raise ValueError(f'Distance must be numeric, got {type(um).__name__}')
        if abs(um) > MOTOR_POSITION_LIMIT:
            raise ValueError(
                f'Distance {um} um exceeds safety limit of +/-{MOTOR_POSITION_LIMIT} um'
            )

        # Silently no-op for axes that aren't present on this hardware.
        # See move_absolute_position for the rationale.
        if axis not in self._arrival_events:
            _api_log.debug(f'move_rel ignored: {axis} not present on this scope')
            return

        # Capture start_pos + ramp before driving. start_time is captured
        # AFTER the driver call returns -- mirrors move_absolute_position.
        # The ~50 ms serial round-trip to write the hardware target precedes
        # any physical motion; capturing start_time before that would make
        # _predicted_position's elapsed-since-arm lead the motor's real
        # elapsed by the full serial RT, and the UI crosshair would visibly
        # outrun the stage on long moves.
        #
        # If a prior move is still in flight on this axis, accumulate against
        # the prior move's commanded target (mirrors the driver-layer
        # `move_rel_pos` semantics: it reads `target_pos()` from firmware,
        # not `current_pos()`, so chained relative moves add to the previous
        # target). At IDLE the cache holds the post-arrival current position
        # (~= previous target), so reading cache as start_pos is correct.
        with self._move_profile_lock:
            prior_profile = self._move_profile.get(axis)
        if prior_profile is not None and prior_profile.get('target_pos') is not None:
            start_pos = float(prior_profile['target_pos'])
        else:
            with self._pos_cache_lock:
                start_pos = self._pos_cache.get(axis, 0.0)
        target_pos = start_pos + float(um)
        try:
            ramp = self._driver.motorconfig.ramp_params(axis)
        except Exception:
            ramp = None

        # Write hardware target BEFORE transitioning axis to MOVING --
        # same race fix as move_absolute_position (#618).
        try:
            self._driver.move_rel_pos(axis, um, overshoot_enabled=overshoot_enabled)
        except Exception:
            _api_log.error(f'move_rel {axis}={um:+.1f}um FAILED')
            raise
        if ramp:
            with self._move_profile_lock:
                self._move_profile[axis] = {
                    'start_time': time.monotonic(),
                    'start_pos': start_pos,
                    'target_pos': target_pos,
                    'ramp': ramp,
                }
        self._set_axis_state(axis, AxisState.MOVING)
        # No move-init cache write: cache holds CURRENT position, which is
        # still start_pos until _motion_monitor_loop reads it from hardware
        # on its first cycle. Target is held in _move_profile[axis], where
        # get_target_position picks it up during MOVING.
        self._fire_position_listeners(axis)
        self._scope.imaging.frame_validity.invalidate(self._AXIS_VALIDITY_SOURCE.get(axis, 'xy_move'))
        _api_log.info(f'move_rel {axis}={um:+.1f}um{" wait" if wait_until_complete else ""}')

        if wait_until_complete is True:
            self.wait_until_finished_moving()
            self._set_axis_state(axis, AxisState.IDLE)

    def wait_until_finished_moving(self, timeout_s: float = 120.0) -> bool:
        """Block until all axes have reached their target positions.

        Waits on per-axis arrival events set by the motion monitor thread.
        Zero serial I/O from the calling thread -- all firmware queries
        happen on the monitor thread at 50 Hz.

        Args:
            timeout_s: Maximum seconds to wait (default 120s).

        Returns:
            bool: True if all axes arrived, False if timed out.
        """
        deadline = time.monotonic() + timeout_s
        # Iterate arrival events directly (not axes_present) so a transient
        # motion.detect_present_axes() failure at call time can never cause
        # this to return True without actually waiting for the in-flight
        # move. _arrival_events was sized to detect_present_axes() at init
        # and never changes shape thereafter, so iterating its keys is the
        # canonical "every axis this scope can track" set. Events for
        # non-moving axes are .set() by construction.
        for ax in self._arrival_events:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f'[SCOPE API ] wait_until_finished_moving timed out on axis {ax}')
                return False
            if not self._arrival_events[ax].wait(timeout=remaining):
                logger.warning(f'[SCOPE API ] wait_until_finished_moving timed out on axis {ax}')
                return False

        return True

    def _set_axis_state(self, axis: str, state: str):
        """Set the state of an axis (internal use only).

        When transitioning to MOVING/HOMING, clears the axis arrival event
        and wakes the motion monitor. When transitioning to IDLE, sets the
        arrival event so waiters unblock. Fires position listeners on every
        transition.

        Silently no-ops for axes that are not present on this hardware.
        Per-axis dicts are sized to detect_present_axes() at init, so
        hardcoded callers like xycenter() (X/Y) and thome() (T)
        automatically degrade to no-ops on scopes that lack those axes.
        """
        if axis not in self._arrival_events:
            return
        with self._axis_state_lock:
            old_state = self._axis_state.get(axis, AxisState.UNKNOWN)
            self._axis_state[axis] = state
        if profile_trace.ENABLE_PROFILE_TRACE and old_state != state:
            profile_trace.trace(
                'motion_trace.csv',
                'ts_ms,duration_ms,event,axis,detail',
                [int(time.time() * 1000), 0, 'transition', axis, f'{old_state}->{state}'],
            )

        if state in (AxisState.MOVING, AxisState.HOMING):
            # Clear arrival event -- axis is now in motion
            self._arrival_events[axis].clear()
            # Wake the motion monitor to start polling
            self._motion_wake.set()
        elif state == AxisState.IDLE:
            # Signal arrival -- unblocks any wait_for_axis() callers
            self._arrival_events[axis].set()
            # Clear motion profile -- predictor falls back to cache
            with self._move_profile_lock:
                self._move_profile[axis] = None

        self._fire_position_listeners(axis)

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

                if not moving_axes:
                    # Also check overshoot -- if overshoot is active,
                    # the monitor should keep running
                    if hasattr(self._driver, 'overshoot') and self._driver.overshoot:
                        time.sleep(self._MOTION_POLL_INTERVAL)
                        continue
                    # All axes arrived -- go back to sleep
                    self._motion_wake.clear()
                    break

                # Query firmware for each MOVING axis
                with profile_trace.timer(
                    'motion_trace.csv',
                    'ts_ms,duration_ms,event,axis,detail',
                    lambda: ['poll', ','.join(moving_axes), ''],
                ):
                    for ax in moving_axes:
                        if self._motion_monitor_stop.is_set():
                            break
                        if not self._driver.is_connected():
                            continue
                        # Read motor actual position from hardware and update
                        # the cache so get_current_position (and the crosshair
                        # via the position listener) tracks the motor instead
                        # of the cached target. Fixes #674 H4 -- previously,
                        # get_current_position routed through _predicted_position,
                        # whose trapezoidal model used unrealistic ramp_params
                        # (motorconfig amax=50000 vs firmware register 30000;
                        # converted to ~70-116 m/s^2 vs real stage <5 m/s^2)
                        # and raced the motor 5-10x ahead.
                        try:
                            actual = self._driver.current_pos(ax)
                            if actual is not None:
                                with self._pos_cache_lock:
                                    self._pos_cache[ax] = float(actual)
                        except Exception as e:
                            _api_log.debug(f'motion monitor current_pos({ax}) failed: {e}')
                        # Arrival check: firmware-authoritative via the
                        # position_reached (STATUS_R bit 22) signal. The motor
                        # owns this -- it knows when XACTUAL == XTARGET at the
                        # microstep level, including final-step settling and
                        # the firmware Zstop logic. On arrival, cache holds
                        # whatever current_pos read above returned -- that
                        # is the actual motor position, which may differ from
                        # the commanded target by up to ~1 microstep (X/Y
                        # ~0.078 um, Z ~0.025 um) due to microstep
                        # quantization. Reporting the polled value (not the
                        # commanded target) keeps the cache honest about
                        # where the motor physically is.
                        try:
                            if self.get_target_status(ax):
                                self._set_axis_state(ax, AxisState.IDLE)
                            else:
                                # Still moving -- propagate the refreshed
                                # cache value to UI listeners.
                                self._fire_position_listeners(ax)
                        except Exception as e:
                            logger.warning(
                                f'[SCOPE API ] Motion monitor: target_status({ax}) failed: {e}'
                            )

                time.sleep(self._MOTION_POLL_INTERVAL)
