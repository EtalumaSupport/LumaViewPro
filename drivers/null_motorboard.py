# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Null-object motor board — no-op implementation of the MotorBoard interface.

Used when no motor hardware is present (e.g., LS620 Lumascope Classic, or
MotorBoard connection failure). All methods return safe defaults:
positions return 0.0, moves complete immediately, homing reports done.

This eliminates the need for ``if not self.motion`` guards throughout the
codebase (Rule 8: API handles missing hardware gracefully).

The Lumascope API assigns ``self.motion = NullMotionBoard()`` instead of
``self.motion = None``, so callers never need to check for None.
"""

from __future__ import annotations

import logging
import pathlib
import threading

from modules.motorconfig import MotorConfig

logger = logging.getLogger('LVP.drivers.null_motorboard')


class NullMotionBoard:
    """No-op motor board that satisfies the full MotorBoard interface.

    Attributes match what ``lumascope_api.py`` and other callers access
    directly (``driver``, ``overshoot``, ``thread_lock``, etc.).
    """

    def __init__(self, motorconfig_defaults_file: pathlib.Path | None = None):
        # Required attributes accessed directly by lumascope_api and callers
        self.driver = True              # truthy sentinel — satisfies `not self.motion.driver`
        self.overshoot = False
        self.thread_lock = threading.RLock()
        self._lock = self.thread_lock   # alias used by SerialBoard pattern
        self._state_lock = threading.Lock()
        self.port = None
        self.found = False
        self._fullinfo = None
        self.initial_homing_complete = True   # trivially "homed" (no motors to home)
        self.initial_t_homing_complete = True
        self._has_turret = False
        self._connect_fails = 0
        self.firmware_version = ''
        self.firmware_responding = False
        self.is_v3 = False

        # Load motorconfig for coordinate transforms (uses defaults)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path("data/motorconfig_defaults.json")
        try:
            self.motorconfig = MotorConfig(defaults_file=motorconfig_defaults_file)
        except Exception:
            self.motorconfig = MotorConfig.__new__(MotorConfig)
            self.motorconfig._config = {}

        self.backlash = 0.0

        self.axes_config = {
            'Z': {'limits': {'min': 0., 'max': 20000.}, 'move_func': self.z_um2ustep},
            'X': {'limits': {'min': 0., 'max': 120000.}, 'move_func': self.xy_um2ustep},
            'Y': {'limits': {'min': 0., 'max': 80000.}, 'move_func': self.xy_um2ustep},
            'T': {'move_func': self.t_pos2ustep},
        }

        logger.debug('[NULL Motor] NullMotionBoard initialized (no motor hardware)')

    # ------------------------------------------------------------------
    # Connection (no-ops)
    # ------------------------------------------------------------------
    def connect(self): pass
    def disconnect(self): pass
    def is_connected(self) -> bool: return False

    # ------------------------------------------------------------------
    # Movement (no-ops)
    # ------------------------------------------------------------------
    def move(self, axis, steps): pass
    def move_abs_pos(self, axis, pos, overshoot_enabled=True, ignore_limits=False): pass
    def move_rel_pos(self, axis, um, overshoot_enabled=False): pass

    # ------------------------------------------------------------------
    # Position queries (return 0)
    # ------------------------------------------------------------------
    def target_pos(self, axis) -> float: return 0.0
    def current_pos(self, axis) -> float: return 0.0
    def target_pos_steps(self, axis) -> int: return 0
    def current_pos_steps(self, axis) -> int: return 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def target_status(self, axis) -> bool: return True  # always "arrived"
    def home_status(self, axis) -> str: return ''
    def reference_status(self, axis) -> str: return '00000000000000000000000000000000'
    def limit_switch_status(self, axis) -> tuple: return (0, 0)

    # ------------------------------------------------------------------
    # Homing (no-ops, report complete)
    # ------------------------------------------------------------------
    def zhome(self) -> bool: return True
    def xyhome(self) -> bool: return True
    def thome(self) -> bool: return True
    def has_turret(self) -> bool: return self._has_turret
    def has_xyhomed(self) -> bool: return True
    def has_thomed(self) -> bool: return True

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_microscope_model(self) -> str | None: return None
    def fullinfo(self) -> dict:
        return {
            'model': None,
            'serial_number': None,
            'firmware_version': None,
            'x_homed': False, 'x_present': False,
            'y_homed': False, 'y_present': False,
            'z_homed': False, 'z_present': False,
            't_homed': False, 't_present': False,
        }
    def get_axes_config(self): return self.axes_config
    def get_axis_limits(self, axis) -> dict | None:
        cfg = self.axes_config.get(axis, {})
        return cfg.get('limits')

    # ------------------------------------------------------------------
    # Acceleration (no-ops)
    # ------------------------------------------------------------------
    def acceleration_limit(self, axis, parameter) -> int: return 100
    def acceleration_limits(self) -> dict: return {}
    def set_acceleration_limit(self, axis, parameter, val_pct): pass
    def set_acceleration_limits(self, val_pct): pass

    # ------------------------------------------------------------------
    # Precision mode (no-op)
    # ------------------------------------------------------------------
    def set_precision_mode(self, axis, enabled): pass

    # ------------------------------------------------------------------
    # SPI (no-ops)
    # ------------------------------------------------------------------
    def spi_read(self, axis, addr) -> str: return '0'
    def spi_write(self, axis, addr, payload) -> str: return ''

    # ------------------------------------------------------------------
    # Coordinate transforms (delegate to motorconfig or use defaults)
    # ------------------------------------------------------------------
    def z_ustep2um(self, ustep) -> float:
        try: return self.motorconfig.z_ustep2um(ustep)
        except Exception: return float(ustep) * 0.049  # default

    def z_um2ustep(self, um) -> int:
        try: return self.motorconfig.z_um2ustep(um)
        except Exception: return int(um / 0.049)

    def xy_ustep2um(self, ustep) -> float:
        try: return self.motorconfig.xy_ustep2um(ustep)
        except Exception: return float(ustep) * 0.049

    def xy_um2ustep(self, um) -> int:
        try: return self.motorconfig.xy_um2ustep(um)
        except Exception: return int(um / 0.049)

    def t_ustep2deg(self, ustep) -> float: return 0.0
    def t_ustep2pos(self, ustep) -> float: return 0.0
    def t_deg2ustep(self, degrees) -> int: return 0
    def t_pos2ustep(self, position) -> int: return 0

    # ------------------------------------------------------------------
    # Firmware / serial (no-ops)
    # ------------------------------------------------------------------
    def get_current_firmware(self) -> str: return ''
    def check_firmware(self) -> dict: return {'status': False}
    def exchange_command(self, command, response_numlines=1, timeout=None): return None
    def exchange_multiline(self, command, timeout=60, end_markers=None): return []

    # ------------------------------------------------------------------
    # Raw REPL (no-ops)
    # ------------------------------------------------------------------
    def enter_raw_repl(self, **kwargs) -> bool: return False
    def exit_raw_repl(self, serial_port=None) -> bool: return False
    def repl_exec(self, code, **kwargs): return (b'', b'')
    def repl_list_files(self, **kwargs) -> list: return []
    def repl_read_file(self, remote_path, **kwargs) -> bytes: return b''
    def repl_write_file(self, remote_path, data, **kwargs) -> bool: return False
    def verify_firmware_running(self, **kwargs) -> bool: return False

    # ------------------------------------------------------------------
    # Disconnect hook (no-op)
    # ------------------------------------------------------------------
    def _on_disconnect(self): pass
