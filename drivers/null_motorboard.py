# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Null-object motor board -- no-op implementation of the MotorBoard interface.

Used when no motor hardware is present (e.g., LS620 Lumascope Classic, or
MotorBoard connection failure). All methods return safe defaults:
positions return 0.0, moves complete immediately, homing reports done.

This eliminates the need for ``if not self.motion`` guards throughout the
codebase (the API handles missing hardware gracefully).

The Lumascope API assigns ``self.motion = NullMotionBoard()`` instead of
``self.motion = None``, so callers never need to check for None.
"""

from __future__ import annotations

import logging
import pathlib
import threading

from drivers.registry import motor_registry
from drivers.motorconfig import MotorConfig

logger = logging.getLogger('LVP.drivers.null_motorboard')


@motor_registry.register('null', priority=0)
class NullMotionBoard:
    """No-op motor board that satisfies the full MotorBoard interface.

    Attributes match what ``lumascope_api.py`` and other callers access
    directly (``driver``, ``overshoot``, ``thread_lock``, etc.).
    """

    def __init__(self, motorconfig_defaults_file: pathlib.Path | None = None):
        # Required attributes accessed directly by lumascope_api and callers
        self.driver = True  # truthy sentinel -- satisfies `not self.motion.driver`
        self.overshoot = False
        self.thread_lock = threading.RLock()
        self._lock = self.thread_lock  # alias used by SerialBoard pattern
        self._state_lock = threading.Lock()
        self.port = None
        self.found = False
        self._fullinfo = None
        self.initial_homing_complete = True  # trivially "homed" (no motors to home)
        self.initial_t_homing_complete = True
        self._has_turret = False
        self._connect_fails = 0
        self.firmware_version = ''
        self.firmware_responding = False
        self.is_v3 = False

        # Load motorconfig for coordinate transforms (uses defaults)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path('data/motorconfig_defaults.json')
        try:
            self.motorconfig = MotorConfig(defaults_file=motorconfig_defaults_file)
        except Exception:
            self.motorconfig = MotorConfig.__new__(MotorConfig)
            self.motorconfig._config = {}

        self.backlash = 0.0

        self.axes_config = {
            'Z': {'limits': {'min': 0.0, 'max': 14000.0}, 'move_func': self.z_um2ustep},
            'X': {'limits': {'min': 0.0, 'max': 120000.0}, 'move_func': self.xy_um2ustep},
            'Y': {'limits': {'min': 0.0, 'max': 80000.0}, 'move_func': self.xy_um2ustep},
            'T': {'move_func': self.t_pos2ustep},
        }

        logger.debug('[NULL Motor] NullMotionBoard initialized (no motor hardware)')

    # ------------------------------------------------------------------
    # Connection (no-ops)
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Null implementation: no-op."""
        pass

    def disconnect(self) -> None:
        """Null implementation: no-op."""
        pass

    def is_connected(self) -> bool:
        """Null implementation: never connected.

        Returns:
            bool: Always False.
        """
        return False

    # ------------------------------------------------------------------
    # Movement (no-ops)
    # ------------------------------------------------------------------
    def move(self, axis, steps) -> None:
        """Null implementation: no-op."""
        pass

    def move_abs_pos(self, axis, pos, overshoot_enabled=True, ignore_limits=False) -> None:
        """Null implementation: no-op."""
        pass

    def move_rel_pos(self, axis, um, overshoot_enabled=False) -> None:
        """Null implementation: no-op."""
        pass

    # ------------------------------------------------------------------
    # Position queries (return 0)
    # ------------------------------------------------------------------
    def target_pos(self, axis) -> float:
        """Null implementation: returns sentinel value.

        Returns:
            float: Always 0.0.
        """
        return 0.0

    def current_pos(self, axis) -> float:
        """Null implementation: returns sentinel value.

        Returns:
            float: Always 0.0.
        """
        return 0.0

    def target_pos_steps(self, axis) -> int:
        """Null implementation: returns sentinel value.

        Returns:
            int: Always 0.
        """
        return 0

    def current_pos_steps(self, axis) -> int:
        """Null implementation: returns sentinel value.

        Returns:
            int: Always 0.
        """
        return 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def target_status(self, axis) -> bool:
        """Null implementation: always reports arrived.

        Returns:
            bool: Always True.
        """
        return True

    def home_status(self, axis) -> str:
        """Null implementation: returns sentinel value.

        Returns:
            str: Always empty string.
        """
        return ''

    def reference_status(self, axis) -> str:
        """Null implementation: returns sentinel value.

        Returns:
            str: 32-char zero bitstring.
        """
        return '00000000000000000000000000000000'

    def limit_switch_status(self, axis) -> tuple:
        """Null implementation: no limits engaged.

        Returns:
            tuple: ``(0, 0)`` -- both switches clear.
        """
        return (0, 0)

    # ------------------------------------------------------------------
    # Homing (no-ops, report complete)
    # ------------------------------------------------------------------
    def zhome(self) -> bool:
        """Null implementation: trivially homed.

        Returns:
            bool: Always True.
        """
        return True

    def home(self) -> bool:
        """Null implementation: trivially homed.

        Returns:
            bool: Always True.
        """
        return True

    def thome(self) -> bool:
        """Null implementation: trivially homed.

        Returns:
            bool: Always True.
        """
        return True

    def has_turret(self) -> bool:
        """Null implementation: never has a turret.

        Returns:
            bool: Always False.
        """
        return self._has_turret

    def has_homed(self) -> bool:
        """Null implementation: trivially homed.

        Returns:
            bool: Always True.
        """
        return True

    def has_thomed(self) -> bool:
        """Null implementation: trivially homed.

        Returns:
            bool: Always True.
        """
        return True

    def detect_present_axes(self) -> list:
        """Null board has no physically present axes.

        Returns:
            list: Always empty.
        """
        return []

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_microscope_model(self) -> str | None:
        """Null implementation: returns sentinel value.

        Returns:
            str | None: Always None.
        """
        return None

    def get_serial_number(self) -> str | None:
        """Null implementation: returns sentinel value.

        Returns:
            str | None: Always None.
        """
        return None

    def fullinfo(self) -> dict:
        """Null implementation: returns a dict with all fields blank/False.

        Returns:
            dict: Sentinel dict with model/serial/firmware None and every
                axis ``_homed`` / ``_present`` False.
        """
        return {
            'model': None,
            'serial_number': None,
            'firmware_version': None,
            'x_homed': False,
            'x_present': False,
            'y_homed': False,
            'y_present': False,
            'z_homed': False,
            'z_present': False,
            't_homed': False,
            't_present': False,
        }

    def get_axes_config(self) -> dict:
        """Return the per-axis config (limits + unit-conversion func).

        Returns:
            dict: Axis-letter-keyed configuration dict.
        """
        return self.axes_config

    def get_axis_limits(self, axis) -> dict | None:
        """Return travel limits for an axis, or None if no limits defined.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            dict | None: ``{'min': float, 'max': float}`` or None.
        """
        cfg = self.axes_config.get(axis, {})
        return cfg.get('limits')

    # ------------------------------------------------------------------
    # Acceleration (no-ops)
    # ------------------------------------------------------------------
    def acceleration_limit(self, axis, parameter) -> int:
        """Null implementation: returns sentinel value.

        Returns:
            int: Always 100.
        """
        return 100

    def acceleration_limits(self) -> dict:
        """Null implementation: returns sentinel value.

        Returns:
            dict: Always empty.
        """
        return {}

    def set_acceleration_limit(self, axis, parameter, val_pct) -> None:
        """Null implementation: no-op."""
        pass

    def set_acceleration_limits(self, val_pct) -> None:
        """Null implementation: no-op."""
        pass

    # ------------------------------------------------------------------
    # Precision mode (no-op)
    # ------------------------------------------------------------------
    def set_precision_mode(self, axis, enabled) -> None:
        """Null implementation: no-op."""
        pass

    # ------------------------------------------------------------------
    # SPI (no-ops)
    # ------------------------------------------------------------------
    def spi_read(self, axis, addr) -> str:
        """Null implementation: returns sentinel value.

        Returns:
            str: Always ``'0'``.
        """
        return '0'

    def spi_write(self, axis, addr, payload) -> str:
        """Null implementation: returns sentinel value.

        Returns:
            str: Always empty string.
        """
        return ''

    # ------------------------------------------------------------------
    # Coordinate transforms (delegate to motorconfig or use defaults)
    # ------------------------------------------------------------------
    def z_ustep2um(self, ustep) -> float:
        """Convert Z microsteps to micrometers (motorconfig delegate).

        Args:
            ustep: Microstep count.

        Returns:
            float: Position in micrometers.
        """
        try:
            return self.motorconfig.z_ustep2um(ustep)
        except Exception:
            return float(ustep) * 0.049  # default

    def z_um2ustep(self, um) -> int:
        """Convert Z micrometers to microsteps (motorconfig delegate).

        Args:
            um: Position in micrometers.

        Returns:
            int: Microstep count.
        """
        try:
            return self.motorconfig.z_um2ustep(um)
        except Exception:
            return int(um / 0.049)

    def xy_ustep2um(self, ustep) -> float:
        """Convert XY microsteps to micrometers (motorconfig delegate).

        Args:
            ustep: Microstep count.

        Returns:
            float: Position in micrometers.
        """
        try:
            return self.motorconfig.xy_ustep2um(ustep)
        except Exception:
            return float(ustep) * 0.049

    def xy_um2ustep(self, um) -> int:
        """Convert XY micrometers to microsteps (motorconfig delegate).

        Args:
            um: Position in micrometers.

        Returns:
            int: Microstep count.
        """
        try:
            return self.motorconfig.xy_um2ustep(um)
        except Exception:
            return int(um / 0.049)

    def t_ustep2deg(self, ustep) -> float:
        """Null implementation: returns sentinel value.

        Returns:
            float: Always 0.0.
        """
        return 0.0

    def t_ustep2pos(self, ustep) -> float:
        """Null implementation: returns sentinel value.

        Returns:
            float: Always 0.0.
        """
        return 0.0

    def t_deg2ustep(self, degrees) -> int:
        """Null implementation: returns sentinel value.

        Returns:
            int: Always 0.
        """
        return 0

    def t_pos2ustep(self, position) -> int:
        """Null implementation: returns sentinel value.

        Returns:
            int: Always 0.
        """
        return 0

    # ------------------------------------------------------------------
    # Firmware / serial (no-ops)
    # ------------------------------------------------------------------
    def get_current_firmware(self) -> str:
        """Null implementation: returns sentinel value.

        Returns:
            str: Always empty string.
        """
        return ''

    def check_firmware(self) -> dict:
        """Null implementation: returns sentinel value.

        Returns:
            dict: ``{'status': False}``.
        """
        return {'status': False}

    def exchange_command(self, command, response_numlines=1, timeout=None):
        """Null implementation: returns sentinel value.

        Returns:
            None: Always.
        """
        return None

    def exchange_multiline(self, command, timeout=60, end_markers=None) -> list:
        """Null implementation: returns sentinel value.

        Returns:
            list: Always empty.
        """
        return []

    def motor_stop(self) -> bool:
        """No motor, no stop.

        Returns:
            bool: Always False.
        """
        return False

    def supports_motor_stop(self) -> bool:
        """No motor hardware: no command family is supported."""
        return False

    def supports_fan(self) -> bool:
        """No motor hardware: no command family is supported."""
        return False

    def supports_diagnostics(self) -> bool:
        """No motor hardware: no command family is supported."""
        return False

    # ------------------------------------------------------------------
    # Raw REPL (no-ops)
    # ------------------------------------------------------------------
    def enter_raw_repl(self, **kwargs) -> bool:
        """Null implementation: returns sentinel value.

        Returns:
            bool: Always False.
        """
        return False

    def exit_raw_repl(self, serial_port=None) -> bool:
        """Null implementation: returns sentinel value.

        Returns:
            bool: Always False.
        """
        return False

    def repl_exec(self, code, **kwargs) -> tuple[bytes, bytes]:
        """Null implementation: returns sentinel value.

        Returns:
            tuple[bytes, bytes]: Empty ``(stdout, stderr)`` tuple.
        """
        return (b'', b'')

    def repl_list_files(self, **kwargs) -> list:
        """Null implementation: returns sentinel value.

        Returns:
            list: Always empty.
        """
        return []

    def repl_read_file(self, remote_path, **kwargs) -> bytes:
        """Null implementation: returns sentinel value.

        Returns:
            bytes: Always empty.
        """
        return b''

    def repl_write_file(self, remote_path, data, **kwargs) -> bool:
        """Null implementation: returns sentinel value.

        Returns:
            bool: Always False.
        """
        return False

    def verify_firmware_running(self, **kwargs) -> bool:
        """Null implementation: returns sentinel value.

        Returns:
            bool: Always False.
        """
        return False

    # ------------------------------------------------------------------
    # Disconnect hook (no-op)
    # ------------------------------------------------------------------
    def _on_disconnect(self) -> None:
        """Null implementation: no-op."""
        pass
