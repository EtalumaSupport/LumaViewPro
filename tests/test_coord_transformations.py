# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for `modules/coord_transformations.py`: valid labware round-trips."""

from unittest.mock import MagicMock

import pytest

from modules.coord_transformations import CoordinateTransformer


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


class TestHappyPathStillWorks:
    """Labware round-trips through the transforms."""

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
