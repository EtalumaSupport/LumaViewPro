# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""MotionAPI -- sub-API for stage / focus / turret motion.

Wave 7 Phase 2b: 22 stateless method bodies have relocated from
Lumascope to this surface. Stateful methods (move_absolute_position,
get_current_position, etc.) still forward to Lumascope as thin
*args, **kwargs placeholders -- they relocate in Phase 2c along with
the state slots (_pos_cache, _axis_state, _arrival_events, ...).

Constructor signature:
    MotionAPI(scope, driver) -- scope is the Lumascope back-ref;
    driver is the MotorBoardProtocol instance (also accessible as
    scope._motion_driver).

Within a relocated body:
    * driver calls use ``self._driver.X`` (the bound MotorBoardProtocol
      handle).
    * cross-method calls to a stateless sibling on this surface call
      directly: ``self.<sibling>()``.
    * cross-method calls to a method still on Lumascope (state-touching
      helpers like ``_set_axis_state``, ``axes_present``, ``refresh_
      position_cache``, the stateful movement methods) route via
      ``self._scope.X``.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.1 for the canonical
method list this surface implements, and
docs/WAVE7_PHASE_2_PLAN.md for the multi-commit decomposition.
"""

from __future__ import annotations

import contextlib
import logging as _logging
from typing import TYPE_CHECKING

from lvp_logger import logger
from modules.notification_center import notifications

# Match _lumascope.py's module-level _api_log channel so relocated
# bodies log to the same handler chain.
_api_log = _logging.getLogger('LVP.api')

# AxisState constants live on _lumascope.py for now (move in 2c). Module-
# top import is safe because MotionAPI is only constructed inside
# Lumascope.__init__ (via a function-local import there), so by the time
# motion.py first loads, _lumascope.py is fully initialized.
from modules.lumascope_api._lumascope import AxisState

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import MotorBoardProtocol


class MotionAPI:
    """Motion sub-API. Hosts the stateless method bodies (Wave 7 Phase 2b)."""

    def __init__(self, scope: 'Lumascope', driver: 'MotorBoardProtocol') -> None:  # noqa: ARG002
        # `driver` is in the signature for backcompat (Phase 1 Lumascope
        # passes it explicitly). It is intentionally unused: `_driver`
        # is a dynamic property that re-resolves through `_scope` on
        # every access. Lumascope reassigns `_motion_driver` during
        # `connect()` / `disconnect()` (e.g. swaps to NullMotionBoard on
        # disconnect); capturing the init-time handle would leave this
        # surface talking to a stale driver after every reconnect.
        self._scope = scope

    @property
    def _driver(self) -> 'MotorBoardProtocol':
        return self._scope._motion_driver

    # ------------------------------------------------------------------
    # Stateless method bodies (relocated in Wave 7 Phase 2b).
    #
    # Order mirrors _lumascope.py source order so a side-by-side diff
    # against the Phase 2a inventory stays readable.
    # ------------------------------------------------------------------

    def move_absolute_async(self, axis, pos, *, wait_until_complete=False,
                            overshoot_enabled=True, callback=None,
                            cb_kwargs=None) -> None:
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
        ex.put(IOTask(
            action=self._scope.move_absolute_position,
            kwargs={
                'axis': axis,
                'pos': pos,
                'wait_until_complete': wait_until_complete,
                'overshoot_enabled': overshoot_enabled,
            },
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def stop_motion(self) -> None:
        """Stop all in-flight motor moves (LVP-A-1).

        Idempotent + safe-when-disconnected per Rule 4 + Rule 8 -- no-ops
        when the motor board isn't connected. Uses the firmware-side
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
            # LVP-A-1 followup: route through MotorBoard.motor_stop so
            # field firmware (2024-09-10 EL-0940-02) silently no-ops
            # instead of producing two FIRMWARE ERROR warnings per
            # shutdown. motor_stop returns True if STOP was accepted,
            # False if firmware doesn't implement it (cached).
            stopped = self._driver.motor_stop()
            if stopped:
                logger.info('[SCOPE API ] stop_motion: motors stopped')
            else:
                logger.debug(
                    '[SCOPE API ] stop_motion: firmware does not '
                    'implement STOP; motors will latch on disconnect')
        except Exception as e:
            # Rule 14 -- log + notify, but don't re-raise: stop_motion
            # is called from shutdown paths where the caller can't
            # meaningfully recover and a raised exception would leave
            # disconnect() half-done.
            logger.warning(
                f'[SCOPE API ] stop_motion failed: {type(e).__name__}: {e}')
            try:
                notifications.warning(
                    'Motion', 'Motor stop failed',
                    f'STOP command failed during shutdown: '
                    f'{type(e).__name__}: {e}')
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
            if self._scope._turret_config.get(persisted_position) == objective_id:
                return persisted_position

        if prefer_current:
            try:
                current_pos = self._scope.get_current_position(axis='T')
                if self._scope._turret_config.get(current_pos) == objective_id:
                    return current_pos
            except Exception:
                pass

        for turret_position, turret_objective_id in self._scope._turret_config.items():
            if objective_id == turret_objective_id:
                return turret_position

        return None

    def is_current_turret_position_objective_set(self) -> bool:
        """Check whether the objective slot at the current turret position is set.

        Returns:
            bool: True if the current turret position has a configured
                objective ID; False if the slot is unconfigured.
        """
        position = self._scope.get_current_position(axis='T')
        if self._scope._turret_config[position] is None:
            return False

        return True

    def get_axes_config(self) -> dict:
        """Get the axis configuration from the motion board.

        Returns:
            dict: Axis configuration (axes present, limits, etc.).
        """
        return self._driver.get_axes_config()

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
        # clean Rule 14 notification with the right cause, instead of
        # the misleading "Homing Failed. Position is unknown" that
        # implies a homing-mechanics problem.
        if not self._scope.motor_connected:
            logger.warning('[SCOPE API ] home() called with motor not connected')
            notifications.error(
                "Motion",
                "Motor Not Connected",
                "Cannot home -- motor controller is not connected. "
                "Check the USB cable and that no other program "
                "(Thonny, mpremote, etc.) is holding the port.",
            )
            return False
        present_axes = self._scope.axes_present()
        _api_log.info('home START')
        for ax in present_axes:
            self._scope._set_axis_state(ax, AxisState.HOMING)
        if 'Z' in present_axes:
            self._scope.frame_validity.invalidate('z_move')
        if 'X' in present_axes or 'Y' in present_axes:
            self._scope.frame_validity.invalidate('xy_move')
        if 'T' in present_axes:
            self._scope.frame_validity.invalidate('turret')
        self._scope.is_homing = True
        try:
            with self._scope.reference_position_logger():
                result = self._driver.home()
            if result is False:
                logger.error('[SCOPE API ] Homing failed')
                notifications.error("Motion", "Homing Failed",
                    "Homing failed. Position is unknown.")
                for ax in present_axes:
                    self._scope._set_axis_state(ax, AxisState.UNKNOWN)
                return False
            for ax in present_axes:
                self._scope._set_axis_state(ax, AxisState.IDLE)
            self._scope.refresh_position_cache()
            return True
        except Exception:
            logger.exception('[SCOPE API ] Homing exception')
            for ax in present_axes:
                self._scope._set_axis_state(ax, AxisState.UNKNOWN)
            notifications.error("Motion", "Homing Error",
                "Homing encountered an error. Position is unknown.")
            return False
        finally:
            self._scope.is_homing = False
            _api_log.info('home DONE')

    @contextlib.contextmanager
    def safe_turret_move(self):
        """Context manager that lowers Z to 0 before turret motion and restores after.

        Use as ``with scope.motion.safe_turret_move(): ... move turret ...``.
        Sets ``is_turreting`` for the duration and restores the original
        Z position even if the body raises.
        """
        # Save off current Z position before moving Z to 0
        logger.info('[SCOPE API ] Moving Z to 0', extra={'force_error': True})
        initial_z = self._scope.get_current_position(axis='Z')
        self._scope.move_absolute_position('Z', pos=0, wait_until_complete=True)
        self._scope.is_turreting = True
        try:
            yield
        finally:
            # Always clear the flag and restore Z, even if the body raised
            # (e.g. driver HardwareError from thome). Without this, a failed
            # turret home would leave is_turreting=True and the stage stuck
            # at Z=0.
            self._scope.is_turreting = False
            logger.info(f'[SCOPE API ] Restoring Z to {initial_z}', extra={'force_error': True})
            self._scope.move_absolute_position('Z', pos=initial_z, wait_until_complete=True)

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
        # auto-reconnect attempts. Fire one clean Rule 14 notification.
        if not self._scope.motor_connected:
            logger.warning('[SCOPE API ] thome() called with motor not connected')
            notifications.error(
                "Motion",
                "Motor Not Connected",
                "Cannot home turret -- motor controller is not connected. "
                "Check the USB cable and that no other program is "
                "holding the port.",
            )
            return False

        # Move turret -- set HOMING after Z is safe, not before.
        # Setting T to HOMING clears its arrival event, which would block
        # wait_until_finished_moving() inside safe_turret_move's Z move.
        _api_log.info('thome START')
        try:
            with self._scope.reference_position_logger():
                with self.safe_turret_move():
                    self._scope._set_axis_state('T', AxisState.HOMING)
                    self._scope.frame_validity.invalidate('turret')
                    result = self._driver.thome()
            if result is False:
                logger.error('[SCOPE API ] Turret homing failed')
                notifications.error("Motion", "Homing Failed",
                    "Turret homing failed. Position is unknown.")
                self._scope._set_axis_state('T', AxisState.UNKNOWN)
                return False
            self._scope._set_axis_state('T', AxisState.IDLE)
            self._scope.refresh_position_cache()
            _api_log.info('thome DONE')
            return True
        except Exception:
            logger.exception('[SCOPE API ] Turret homing exception')
            self._scope._set_axis_state('T', AxisState.UNKNOWN)
            notifications.error("Motion", "Homing Error",
                "Turret homing encountered an error. Position is unknown.")
            _api_log.info('thome DONE')
            return False

    def has_thomed(self) -> bool:
        """Check if the turret has been homed since startup.

        Returns:
            bool: True if turret homing has been performed.
        """
        return self._driver.has_thomed()

    def tmove(self, position: int) -> None:
        """Move the turret to a specific position. Skips if already there.

        Args:
            position: Target turret position (1-4).
        """
        # Commanding a move of the T axis is slow, even if the move is to the current position.
        # Use caching to determine if T is requested to move to it's current position, and bypass the
        # move altogether if it is.
        if self._scope._last_turret_position == position:
            return

        with self.safe_turret_move():
            logger.info(f'[SCOPE API ] Moving T to position {position}')
            self._scope.move_absolute_position('T', position, wait_until_complete=True)
            self._scope._last_turret_position = position

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

    def set_motor_precision_mode(self, axis: str, enabled: bool) -> None:
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
        try:
            status = self._driver.home_status(axis)
            return status
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_home_status({axis}) failed; treating as not home: {e}")
            return False

    def get_target_status(self, axis: str) -> bool:
        """Check if an axis has reached its target position.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            bool: True if at target (always True for T if no turret present).
        """
        # Handle case where we want to know if turret has reached its target, but there is no turret
        if (axis == 'T') and (not self._driver.has_turret()):
            return True

        try:
            status = self._driver.target_status(axis)
            return status
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_target_status({axis}) failed; treating as not at target: {e}")
            return False

    def get_target_pos(self, axis: str) -> float:
        """Get the target position for an axis (error-safe version).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            float: Target position in um, or -1 on error/no turret.
        """
        if (axis == 'T') and (not self._driver.has_turret()):
            return -1

        try:
            pos = self._driver.target_pos(axis)
            return pos if pos is not None else -1
        except Exception as e:
            logger.exception(f"[SCOPE API ] get_target_pos({axis}) failed; returning -1: {e}")
            return -1

    def get_reference_status(self, axis: str) -> str:
        """Get reference status register bits for an axis.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").

        Returns:
            str: 32-character binary string of register bits (MSB first).
        """
        return self._driver.reference_status(axis=axis)

    def get_limit_switch_status(self, axis: str):
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
        for axis in self._scope.axes_present():
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
        if self._scope.is_any_axis_moving():
            return True
        if self.get_overshoot():
            return True
        return False

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
    # Phase 2c thin forwarders (still delegate to Lumascope).
    #
    # These methods touch motion state slots (_pos_cache, _axis_state,
    # _arrival_events, _move_profile, _position_listeners, _motion_wake,
    # _motion_monitor_thread). State + bodies relocate together in
    # Phase 2c so the invariant chain stays intact.
    # ------------------------------------------------------------------

    def move_relative_async(self, *args, **kwargs):
        return self._scope.move_relative_async(*args, **kwargs)

    def move_home_async(self, *args, **kwargs):
        return self._scope.move_home_async(*args, **kwargs)

    def move_absolute_sync(self, *args, **kwargs):
        return self._scope.move_absolute_sync(*args, **kwargs)

    def move_absolute_position(self, *args, **kwargs):
        return self._scope.move_absolute_position(*args, **kwargs)

    def move_relative_position(self, *args, **kwargs):
        return self._scope.move_relative_position(*args, **kwargs)

    def zhome(self, *args, **kwargs):
        return self._scope.zhome(*args, **kwargs)

    def xycenter(self, *args, **kwargs):
        return self._scope.xycenter(*args, **kwargs)

    def has_homed(self, *args, **kwargs):
        return self._scope.has_homed(*args, **kwargs)

    def get_axis_state(self, *args, **kwargs):
        return self._scope.get_axis_state(*args, **kwargs)

    def get_current_position(self, *args, **kwargs):
        return self._scope.get_current_position(*args, **kwargs)

    def get_target_position(self, *args, **kwargs):
        return self._scope.get_target_position(*args, **kwargs)

    def is_any_axis_moving(self, *args, **kwargs):
        return self._scope.is_any_axis_moving(*args, **kwargs)

    @property
    def is_homing(self) -> bool:
        return self._scope.is_homing

    @property
    def is_turreting(self) -> bool:
        return self._scope.is_turreting

    def wait_until_finished_moving(self, *args, **kwargs):
        return self._scope.wait_until_finished_moving(*args, **kwargs)

    def add_position_listener(self, *args, **kwargs):
        return self._scope.add_position_listener(*args, **kwargs)

    def remove_position_listener(self, *args, **kwargs):
        return self._scope.remove_position_listener(*args, **kwargs)

    def get_axis_limits(self, *args, **kwargs):
        return self._scope.get_axis_limits(*args, **kwargs)

    def refresh_position_cache(self, *args, **kwargs):
        return self._scope.refresh_position_cache(*args, **kwargs)

    # ------------------------------------------------------------------
    # Aliases preserved for caller compatibility.
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Alias for ``stop_motion`` -- preserves the original facade name."""
        return self.stop_motion()
