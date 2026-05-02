# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Driver-layer exception classes (raised from drivers/, caught at module/API)."""


class HardwareError(Exception):
    """Hardware communication or configuration failure (motor, LED, camera)."""
    pass
