# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Domain-specific exception classes for LumaViewPro."""


class HardwareError(Exception):
    """Hardware communication or configuration failure (motor, LED, camera)."""
    pass


class ProtocolError(Exception):
    """Protocol file parsing, validation, or execution error."""
    pass


class ConfigError(Exception):
    """Application configuration or settings error."""
    pass


class CaptureError(Exception):
    """Image capture, save, or processing failure."""
    pass
