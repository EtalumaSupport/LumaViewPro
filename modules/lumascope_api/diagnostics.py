# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""DiagnosticsAPI -- sub-API for hardware diagnostic probes.

Phase 1 of Wave 7 decomposition. Thin delegating facade over the
Lumascope composition root. Bodies still live on Lumascope; later
phases relocate them and migrate callers.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.4 for the canonical
method list. No persistent state -- per-call probes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class DiagnosticsAPI:
    """Diagnostics sub-API. Forwards to Lumascope composition root."""

    def __init__(self, scope: 'Lumascope') -> None:
        self._scope = scope

    # --- Camera probes ---
    def get_camera_temperatures(self, *args, **kwargs):
        return self._scope.get_camera_temperatures(*args, **kwargs)

    def get_camera_diagnostic_info(self, *args, **kwargs):
        return self._scope.get_camera_diagnostic_info(*args, **kwargs)

    def run_camera_bandwidth_test(self, *args, **kwargs):
        return self._scope.run_camera_bandwidth_test(*args, **kwargs)

    def run_grab_lifecycle_benchmark(self, *args, **kwargs):
        return self._scope.run_grab_lifecycle_benchmark(*args, **kwargs)

    def run_pylon_diagnostic_probe(self, *args, **kwargs):
        return self._scope.run_pylon_diagnostic_probe(*args, **kwargs)

    # --- Serial probes ---
    def send_diagnostic_command(self, *args, **kwargs):
        return self._scope.send_diagnostic_command(*args, **kwargs)

    def send_diagnostic_command_multiline(self, *args, **kwargs):
        return self._scope.send_diagnostic_command_multiline(*args, **kwargs)

    # --- Motor power / driver / fan diagnostics ---
    # Each returns parsed values or None when the firmware does not
    # support the command (legacy 2024-09-10 firmware did not include
    # VOLTAGE / DRVSTAT_<axis> / FANSPEED / FAN). Per Eric: the driver
    # owns firmware-version gating; callers (TSR, future REST
    # diagnostic endpoint) read None as "INCONCLUSIVE -- firmware
    # does not support this probe."

    def read_motor_voltages(self):
        """Read motor-board power rail tolerance diagnostic.

        Returns a dict mapping rail label to volts (or None per rail
        if unparseable), or None when the firmware does not implement
        the VOLTAGE command. See MotorBoard.read_voltages.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_voltages'):
            return None
        return drv.read_voltages()

    def read_motor_drv_status(self, axis: str):
        """Read TMC5072 DRV_STATUS register for an axis.

        Returns the raw register value as int (caller decodes bits),
        or None when the firmware does not implement DRVSTAT_<axis>.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_drv_status'):
            return None
        return drv.read_drv_status(axis)

    def read_motor_fanspeed(self):
        """Read motor-board fan tachometer RPM.

        Returns RPM as int (0 if no tach wire) or None when firmware
        does not implement FANSPEED.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_fanspeed'):
            return None
        return drv.read_fanspeed()

    def set_motor_fan_duty(self, duty_pct: int) -> bool:
        """Set motor-board fan PWM duty cycle (0..100).

        Returns True if firmware accepted the command, False if firmware
        does not implement FAN:<duty> or no motor driver is present.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'set_fan_duty'):
            return False
        return drv.set_fan_duty(duty_pct)
