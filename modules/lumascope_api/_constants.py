# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Shared constants for the lumascope_api package.

Module-level home for values that both `_lumascope.Lumascope` and the
sub-API modules (motion.py, etc.) need to read. Lifting them here
breaks the import cycle that would otherwise force lazy imports in
the sub-API files.

Other modules in the package alias these onto their classes for
back-compat (e.g. `Lumascope._VALID_AXIS_NAMES = _VALID_AXIS_NAMES`)
so existing callers (`scope._VALID_AXIS_NAMES`, tests reading the
class attribute) keep working.
"""

# Structural axis-name vocabulary used only for input sanity checks
# ("did the caller pass a real axis letter?"). NOT a capability query --
# use `scope.capabilities.axes` for "what does this scope have?".
_VALID_AXIS_NAMES = ('X', 'Y', 'Z', 'T')

# Absolute position bounds in um -- generous outer limits. Per-axis
# travel limits are enforced by the motor board itself.
MOTOR_POSITION_LIMIT = 1_000_000  # 1 meter in um
