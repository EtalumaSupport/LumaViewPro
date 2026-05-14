# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""MotionAPI -- sub-API for stage / focus / turret motion.

Phase 1 of Wave 7 decomposition. This class is a thin delegating facade
over the Lumascope composition root. Method bodies still live on
Lumascope; this surface routes calls through. Later Wave 7 phases
physically relocate the bodies and migrate the ~70 caller sites.

Constructor signature:
    MotionAPI(scope, driver) -- scope is the Lumascope back-ref;
    driver is the MotorBoardProtocol instance (also accessible as
    scope._motion_driver).

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.1 for the canonical
method list this surface implements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import MotorBoardProtocol


class MotionAPI:
    """Motion sub-API. Forwards to Lumascope composition root."""

    def __init__(self, scope: 'Lumascope', driver: 'MotorBoardProtocol') -> None:
        self._scope = scope
        self._driver = driver

    # --- Movement (async) ---
    def move_absolute_async(self, *args, **kwargs):
        return self._scope.move_absolute_async(*args, **kwargs)

    def move_relative_async(self, *args, **kwargs):
        return self._scope.move_relative_async(*args, **kwargs)

    def move_home_async(self, *args, **kwargs):
        return self._scope.move_home_async(*args, **kwargs)

    def move_absolute_sync(self, *args, **kwargs):
        return self._scope.move_absolute_sync(*args, **kwargs)

    # --- Movement (blocking) ---
    def move_absolute_position(self, *args, **kwargs):
        return self._scope.move_absolute_position(*args, **kwargs)

    def move_relative_position(self, *args, **kwargs):
        return self._scope.move_relative_position(*args, **kwargs)

    # --- Homing ---
    def home(self, *args, **kwargs):
        return self._scope.home(*args, **kwargs)

    def zhome(self, *args, **kwargs):
        return self._scope.zhome(*args, **kwargs)

    def thome(self, *args, **kwargs):
        return self._scope.thome(*args, **kwargs)

    def xycenter(self, *args, **kwargs):
        return self._scope.xycenter(*args, **kwargs)

    def has_homed(self, *args, **kwargs):
        return self._scope.has_homed(*args, **kwargs)

    def has_thomed(self, *args, **kwargs):
        return self._scope.has_thomed(*args, **kwargs)

    # --- State ---
    def get_axis_state(self, *args, **kwargs):
        return self._scope.get_axis_state(*args, **kwargs)

    def get_current_position(self, *args, **kwargs):
        return self._scope.get_current_position(*args, **kwargs)

    def get_target_position(self, *args, **kwargs):
        return self._scope.get_target_position(*args, **kwargs)

    def get_actual_position(self, *args, **kwargs):
        return self._scope.get_actual_position(*args, **kwargs)

    def get_target_pos(self, *args, **kwargs):
        return self._scope.get_target_pos(*args, **kwargs)

    def is_moving(self, *args, **kwargs):
        return self._scope.is_moving(*args, **kwargs)

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

    # --- Listeners ---
    def add_position_listener(self, *args, **kwargs):
        return self._scope.add_position_listener(*args, **kwargs)

    def remove_position_listener(self, *args, **kwargs):
        return self._scope.remove_position_listener(*args, **kwargs)

    # --- Configuration ---
    def get_axes_config(self, *args, **kwargs):
        return self._scope.get_axes_config(*args, **kwargs)

    def get_axis_limits(self, *args, **kwargs):
        return self._scope.get_axis_limits(*args, **kwargs)

    def set_motor_precision_mode(self, *args, **kwargs):
        return self._scope.set_motor_precision_mode(*args, **kwargs)

    def set_acceleration_limit(self, *args, **kwargs):
        return self._scope.set_acceleration_limit(*args, **kwargs)

    def refresh_position_cache(self, *args, **kwargs):
        return self._scope.refresh_position_cache(*args, **kwargs)

    # --- Limit / status ---
    def get_home_status(self, *args, **kwargs):
        return self._scope.get_home_status(*args, **kwargs)

    def get_target_status(self, *args, **kwargs):
        return self._scope.get_target_status(*args, **kwargs)

    def get_reference_status(self, *args, **kwargs):
        return self._scope.get_reference_status(*args, **kwargs)

    def get_limit_switch_status(self, *args, **kwargs):
        return self._scope.get_limit_switch_status(*args, **kwargs)

    def get_limit_switch_status_all_axes(self, *args, **kwargs):
        return self._scope.get_limit_switch_status_all_axes(*args, **kwargs)

    def get_overshoot(self, *args, **kwargs):
        return self._scope.get_overshoot(*args, **kwargs)

    # --- Turret ---
    def safe_turret_mover(self, *args, **kwargs):
        return self._scope.safe_turret_mover(*args, **kwargs)

    def tmove(self, *args, **kwargs):
        return self._scope.tmove(*args, **kwargs)

    def has_turret(self, *args, **kwargs):
        return self._scope.has_turret(*args, **kwargs)

    def is_current_turret_position_objective_set(self, *args, **kwargs):
        return self._scope.is_current_turret_position_objective_set(*args, **kwargs)

    def get_turret_position_for_objective_id(self, *args, **kwargs):
        return self._scope.get_turret_position_for_objective_id(*args, **kwargs)

    # --- Stop ---
    def stop(self, *args, **kwargs):
        return self._scope.stop_motion(*args, **kwargs)
