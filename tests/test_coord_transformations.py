# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for `modules/coord_transformations.py` boundary contracts.

Pre-#634, callers passed labware=None into these transforms and crashed
with `AttributeError: 'NoneType' has no attribute 'get_dimensions'` deep
inside the call. Per Rule 8 the API boundary must reject invalid inputs
at the boundary, so each public method now raises NoLabwareSelectedError
when labware is None. Issue #632 / #634 cluster regression coverage.
"""

from unittest.mock import MagicMock

import pytest

from modules.coord_transformations import (
    CoordinateTransformer,
    NoLabwareSelectedError,
)


@pytest.fixture
def transformer():
    return CoordinateTransformer()


@pytest.fixture
def labware():
    plate = MagicMock()
    plate.get_dimensions.return_value = {'x': 127.76, 'y': 85.48}  # 96-well plate
    return plate


@pytest.fixture
def stage_offset():
    return {'x': 1000.0, 'y': 1000.0}


class TestRejectsNoneLabware:
    """Every public coord transform must raise NoLabwareSelectedError
    when labware is None -- boundary check (Rule 8). Without this, callers
    that propagate `(None, None)` from get_selected_labware() crash deep
    inside `labware.get_dimensions()` instead of getting a clear error.
    """

    def test_stage_to_plate_rejects_none(self, transformer, stage_offset):
        with pytest.raises(NoLabwareSelectedError):
            transformer.stage_to_plate(
                labware=None,
                stage_offset=stage_offset,
                sx=10000,
                sy=10000,
            )

    def test_plate_to_stage_rejects_none(self, transformer, stage_offset):
        with pytest.raises(NoLabwareSelectedError):
            transformer.plate_to_stage(
                labware=None,
                stage_offset=stage_offset,
                px=50,
                py=50,
            )

    def test_plate_to_pixel_rejects_none(self, transformer):
        with pytest.raises(NoLabwareSelectedError):
            transformer.plate_to_pixel(
                labware=None,
                px=50,
                py=50,
                scale_x=10,
                scale_y=10,
            )

    def test_stage_to_pixel_rejects_none(self, transformer, stage_offset):
        with pytest.raises(NoLabwareSelectedError):
            transformer.stage_to_pixel(
                labware=None,
                stage_offset=stage_offset,
                sx=10000,
                sy=10000,
                scale_x=10,
                scale_y=10,
            )

    def test_no_labware_error_is_value_error_subclass(self):
        # Existing callers may catch ValueError generically -- ensure
        # NoLabwareSelectedError is recognized as such.
        assert issubclass(NoLabwareSelectedError, ValueError)


class TestHappyPathStillWorks:
    """Regression: real labware should continue to round-trip through
    transforms unchanged. Boundary check must not break valid calls."""

    def test_stage_to_plate_returns_tuple(self, transformer, labware, stage_offset):
        result = transformer.stage_to_plate(
            labware=labware,
            stage_offset=stage_offset,
            sx=0,
            sy=0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_plate_to_stage_returns_tuple(self, transformer, labware, stage_offset):
        result = transformer.plate_to_stage(
            labware=labware,
            stage_offset=stage_offset,
            px=50,
            py=50,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_round_trip_preserves_values(self, transformer, labware, stage_offset):
        # stage -> plate -> stage should give back the original (within
        # floating-point precision).
        sx, sy = 50000.0, 30000.0
        px, py = transformer.stage_to_plate(
            labware=labware, stage_offset=stage_offset, sx=sx, sy=sy
        )
        sx2, sy2 = transformer.plate_to_stage(
            labware=labware, stage_offset=stage_offset, px=px, py=py
        )
        assert abs(sx2 - sx) < 0.001
        assert abs(sy2 - sy) < 0.001
