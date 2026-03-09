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

    # --- Board identity ---

    def model(self) -> str:
        return self._config.get("Microscope", "LS850")

    def serial_number(self) -> str:
        return self._config.get("Serial Number", "Unknown")

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

    # --- LED channel names ---

    def led_channel_name(self, channel_index: int) -> str:
        section = self._config.get("LEDChannel")
        if section is None:
            return "Unknown"
        val = section.get(f"Ch{channel_index}")
        if val is None:
            return "Unknown"
        return val

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
