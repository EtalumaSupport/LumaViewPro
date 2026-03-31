# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""MotorConfig — reads per-unit hardware configuration from motorconfig.json.

Provides axis travel limits, microstep conversion factors, turret positions,
board identity (model/serial), and optics parameters. Falls back to defaults
for any missing keys.
"""

import json
import pathlib

from lvp_logger import logger


class MotorConfig:

    def __init__(self, defaults_file: pathlib.Path):
        self._config = {}
        self._defaults = self._load_json(defaults_file, label="defaults")
        # Start with defaults
        self._config = dict(self._defaults)

    def update_from_board(self, board_config: dict):
        """Merge per-unit config from the motor controller on top of defaults."""
        self._deep_merge(self._config, board_config)

    @staticmethod
    def _load_json(file_path: pathlib.Path, label: str = "") -> dict:
        if file_path is None or not file_path.is_file():
            logger.warning(f"[MotorConfig] {label} file not found: {file_path}")
            return {}
        try:
            with open(file_path, 'r') as fp:
                return json.load(fp)
        except (json.JSONDecodeError, OSError) as ex:
            logger.error(f"[MotorConfig] Failed to load {label} file {file_path}: {ex}")
            return {}

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        """Recursively merge override into base (mutates base)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                MotorConfig._deep_merge(base[key], value)
            else:
                base[key] = value

    def _axis_lookup(self, section: str, axis: str, default=0):
        sub = self._config.get(section)
        if sub is None:
            logger.warning(f"[MotorConfig] Section '{section}' not found")
            return default
        axis = axis.upper()
        val = sub.get(axis)
        if val is None:
            logger.warning(f"[MotorConfig] Axis '{axis}' not in section '{section}'")
            return default
        return val

    # --- Axis properties ---

    def usteps_per_mm(self, axis: str) -> int:
        return int(self._axis_lookup("Axis Microsteps per mm / Objective", axis, default=20157))

    def travel_limit_mm(self, axis: str) -> float:
        return float(self._axis_lookup("Axis Travel Limit", axis, default=120))

    def travel_limit_um(self, axis: str) -> float:
        return self.travel_limit_mm(axis) * 1000.0

    def axis_present(self, axis: str) -> bool:
        return bool(self._axis_lookup("Axis Present", axis, default=True))

    def antibacklash_um(self, axis: str) -> int:
        return int(self._axis_lookup("Axis antibacklash", axis, default=0))

    # --- Motion ramp parameters (TMC5072 6-point ramp) ---

    # Default ramp parameters from INI files (usteps/sec and usteps/sec²).
    # Keyed by axis. These are hardware constants set at firmware load time.
    # Future: query dynamically from board via SPI or v3.1 firmware command.
    _DEFAULT_RAMP = {
        'X': {'vstart': 0, 'a1': 0, 'v1': 0, 'amax': 50000, 'vmax': 800000, 'dmax': 50000, 'd1': 0, 'vstop': 10},
        'Y': {'vstart': 0, 'a1': 0, 'v1': 0, 'amax': 50000, 'vmax': 800000, 'dmax': 50000, 'd1': 0, 'vstop': 10},
        'Z': {'vstart': 0, 'a1': 0, 'v1': 0, 'amax': 25000, 'vmax': 400000, 'dmax': 25000, 'd1': 0, 'vstop': 100},
        'T': {'vstart': 0, 'a1': 0, 'v1': 0, 'amax': 5000, 'vmax': 128000, 'dmax': 5000, 'd1': 0, 'vstop': 10},
    }

    def ramp_params_usteps(self, axis: str) -> dict:
        """Return all TMC5072 ramp parameters in ustep units for an axis."""
        axis = axis.upper()
        return dict(self._DEFAULT_RAMP.get(axis, self._DEFAULT_RAMP['X']))

    # TMC5072 uses internal clock for velocity/acceleration registers.
    # Conversion: v_real = register * f_clk / 2^24 (in usteps/sec)
    #             a_real = register * f_clk^2 / (512 * 2^24) (in usteps/sec²)
    # f_clk = 16 MHz (internal oscillator, typical for TMC5072)
    _TMC_FCLK = 16_000_000
    _TMC_VEL_FACTOR = _TMC_FCLK / (2**24)        # register → usteps/sec
    _TMC_ACC_FACTOR = _TMC_FCLK**2 / (512 * 2**24)  # register → usteps/sec²

    def ramp_params(self, axis: str) -> dict:
        """Return ramp parameters converted to physical units (um/sec, um/sec²).

        Converts TMC5072 register values to real usteps/sec, then to um/sec
        using the axis microstep-to-mm conversion.
        Supports simple trapezoidal (a1/v1/d1 = 0) and future 6-point ramps.
        """
        raw = self.ramp_params_usteps(axis)
        usteps_mm = self.usteps_per_mm(axis)
        um_per_ustep = 1000.0 / usteps_mm

        vel_keys = ('vstart', 'v1', 'vmax', 'vstop')
        acc_keys = ('a1', 'amax', 'dmax', 'd1')

        result = {}
        for k, v in raw.items():
            if k in vel_keys:
                result[k] = v * self._TMC_VEL_FACTOR * um_per_ustep
            elif k in acc_keys:
                result[k] = v * self._TMC_ACC_FACTOR * um_per_ustep
            else:
                result[k] = v
        return result

    # --- Board identity ---

    def model(self) -> str:
        return self._config.get("Microscope", "LS850")

    def serial_number(self) -> str:
        return self._config.get("Serial Number", "Unknown")

    def hardware_rev(self) -> str:
        return self._config.get("HardwareRev", "Unknown")

    # --- Image center offset ---

    def image_center_offset(self) -> tuple[int, int]:
        """Return (X, Y) image center offset in microsteps."""
        section = self._config.get("ImageCenter")
        if section is None:
            return (0, 0)
        return (int(section.get("X", 0)), int(section.get("Y", 0)))

    # --- Turret ---

    def turret_position_usteps(self, position: int) -> int:
        """Get microstep position for turret slot (1-based)."""
        section = self._config.get("TurretPosition")
        if section is None:
            return 0
        val = section.get(str(position))
        if val is None:
            logger.warning(f"[MotorConfig] Turret position {position} not found")
            return 0
        return int(val)

    # --- Optics ---

    def lens_focal_length(self) -> float:
        try:
            return float(self._config["Optics"]["LensFocalLength"])
        except (KeyError, TypeError, ValueError):
            return 47.8

    def pixel_size(self) -> float:
        try:
            return float(self._config["Optics"]["PixelSize"])
        except (KeyError, TypeError, ValueError):
            return 2.0
