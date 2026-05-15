# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: `get_target_pos` retired from the API surface.

The driver-direct target-position read (`TARGET_R<axis>` register
roundtrip, ~30ms) had no production or test callers -- the canonical
reader is `get_target_position(axis)` which serves from the
push-based position cache (zero serial I/O). Two methods exposing the
same conceptual state via different access paths violated the single-
canonical-implementation rule and confused future readers about which
to use.

Retired API methods:
- `Lumascope.get_target_pos(axis)` (backcompat forwarder)
- `MotionAPI.get_target_pos(axis)` (driver-wrapping body)

The underlying `MotorBoard.target_pos(axis)` driver method stays --
it's used by `Lumascope.refresh_position_cache` to seed the cache
from hardware after homing. The driver method has a different
contract (raises HardwareError on serial failure, returns None on
no-data) than the retired API wrapper (try/except-swallowing,
returned -1 on any failure).
"""

from __future__ import annotations


class TestGetTargetPosRetired:
    def test_motion_api_has_no_get_target_pos(self):
        from modules.lumascope_api.motion import MotionAPI
        assert not hasattr(MotionAPI, 'get_target_pos'), (
            "MotionAPI.get_target_pos was retired (zero external "
            "callers, duplicated cache-based get_target_position). "
            "Use scope.get_target_position(axis) instead."
        )

    def test_lumascope_has_no_get_target_pos_forwarder(self):
        from modules.lumascope_api import Lumascope
        assert not hasattr(Lumascope, 'get_target_pos'), (
            "Lumascope.get_target_pos backcompat forwarder was retired "
            "alongside MotionAPI.get_target_pos."
        )

    def test_motion_api_still_has_get_target_position(self):
        """The canonical reader stays."""
        from modules.lumascope_api.motion import MotionAPI
        assert hasattr(MotionAPI, 'get_target_position'), (
            "MotionAPI.get_target_position is the canonical "
            "target-position reader (cache-based; zero serial I/O)."
        )

    def test_motorboard_target_pos_driver_method_still_exists(self):
        """The driver method that the retired API wrapper called
        stays -- `refresh_position_cache` uses it to seed the cache
        from hardware after homing."""
        from drivers.motorboard import MotorBoard
        assert hasattr(MotorBoard, 'target_pos'), (
            "MotorBoard.target_pos is used by refresh_position_cache "
            "to seed the position cache from hardware after homing."
        )

    def test_no_lingering_callers_in_source(self):
        """Source-wide source-grep guard: a future merge that re-adds
        an API call to scope.get_target_pos() or motion.get_target_pos()
        should be caught."""
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        for sub in ('modules', 'ui', 'drivers'):
            for path in (repo_root / sub).rglob('*.py'):
                text = path.read_text()
                # Skip the driver-level target_pos (legitimate); only
                # the API-shape get_target_pos is retired.
                for line in text.splitlines():
                    if 'get_target_pos(' in line:
                        raise AssertionError(
                            f"{path.relative_to(repo_root)}: line "
                            f"references retired `get_target_pos(...)`. "
                            f"Use `get_target_position(...)` (cache-"
                            f"based, zero serial I/O) instead. Line: "
                            f"{line.strip()!r}"
                        )
