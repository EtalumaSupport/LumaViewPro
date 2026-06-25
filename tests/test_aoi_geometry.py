# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for the camera-agnostic oversize-then-crop AOI geometry.

These exercise the pure framing math (no SDK, no hardware): rounding the AOI up
to the alignment grid, centering it on the optical axis, cropping back to the
exact request, the graceful degrade as the request approaches the sensor max,
and the reverse/transpose reorientation of the optical-center offset. The live
driver wiring and the bench collimator sign calibration are validated
separately on hardware.
"""

import numpy as np
import pytest

from modules.aoi_geometry import (
    AoiPlan,
    SensorOrientation,
    ceil_to,
    floor_to,
    plan_aoi,
    reorient_image_center,
)
from modules.image_utils import center_crop


# --- grid rounding ---------------------------------------------------------


@pytest.mark.parametrize(
    'value,step,expected',
    [
        (1900, 48, 1920),  # the IDS width case: 1900 is not a multiple of 48
        (1900, 4, 1900),  # the IDS height case: already legal
        (1920, 48, 1920),  # already on the grid -> unchanged
        (1, 48, 48),  # rounds up to one step
        (1900, 1, 1900),  # step 1 is a no-op
    ],
)
def test_ceil_to(value, step, expected):
    assert ceil_to(value, step) == expected


@pytest.mark.parametrize(
    'value,step,expected',
    [
        (1900, 48, 1872),  # the historical down-snap
        (1900, 4, 1900),
        (47, 48, 48),  # never below one step
        (1900, 1, 1900),  # step 1 is a no-op
    ],
)
def test_floor_to(value, step, expected):
    assert floor_to(value, step) == expected


# --- optical-center reorientation -----------------------------------------


def test_reorient_identity():
    assert reorient_image_center(10, -7, SensorOrientation()) == (10, -7)


def test_reorient_reverse_x_negates_x_only():
    o = SensorOrientation(reverse_x=True)
    assert reorient_image_center(10, -7, o) == (-10, -7)


def test_reorient_reverse_y_negates_y_only():
    o = SensorOrientation(reverse_y=True)
    assert reorient_image_center(10, -7, o) == (10, 7)


def test_reorient_transpose_swaps_axes():
    o = SensorOrientation(transpose=True)
    assert reorient_image_center(10, -7, o) == (-7, 10)


def test_reorient_transpose_then_reverses():
    # transpose first (swap), then negate per active reverse
    o = SensorOrientation(reverse_x=True, reverse_y=True, transpose=True)
    assert reorient_image_center(10, -7, o) == (7, -10)


# --- AOI planning ----------------------------------------------------------

# IMX676-like sensor: 3552x3552, width grid 48, height grid 4.
MAX = (3552, 3552)
STEP = (48, 4)


def test_plan_basic_oversize_and_centered_crop():
    plan = plan_aoi(target=(1900, 1900), step=STEP, max_size=MAX, offset_step=(1, 1))
    assert isinstance(plan, AoiPlan)
    # width oversizes 1900 -> 1920; height already legal
    assert (plan.acq_width, plan.acq_height) == (1920, 1900)
    # crop yields exactly the request, centered: 20 surplus px -> 10 each side
    assert (plan.crop_width, plan.crop_height) == (1900, 1900)
    assert (plan.crop_x0, plan.crop_y0) == (10, 0)
    # AOI centered on the sensor (no optical bias)
    assert plan.offset_x == (3552 - 1920) // 2
    assert plan.offset_y == (3552 - 1900) // 2


def test_plan_at_sensor_max_degrades_to_noop_crop():
    plan = plan_aoi(target=MAX, step=STEP, max_size=MAX, offset_step=(1, 1))
    assert (plan.acq_width, plan.acq_height) == MAX
    assert (plan.crop_width, plan.crop_height) == MAX
    assert (plan.crop_x0, plan.crop_y0) == (0, 0)
    assert (plan.offset_x, plan.offset_y) == (0, 0)


def test_plan_request_above_max_is_capped():
    plan = plan_aoi(target=(9999, 9999), step=STEP, max_size=MAX, offset_step=(1, 1))
    # capped to the largest legal AOI at/below max; crop never exceeds it
    assert plan.acq_width <= MAX[0] and plan.acq_height <= MAX[1]
    assert plan.crop_width == plan.acq_width
    assert plan.crop_height == plan.acq_height
    assert (plan.offset_x, plan.offset_y) == (0, 0)


def test_plan_optical_bias_shifts_offset_not_crop():
    base = plan_aoi(target=(1900, 1900), step=STEP, max_size=MAX, offset_step=(1, 1))
    biased = plan_aoi(
        target=(1900, 1900), step=STEP, max_size=MAX, offset_step=(1, 1), bias=(30, -12)
    )
    # the crop window is geometric and unchanged by the bias
    assert (biased.crop_x0, biased.crop_y0) == (base.crop_x0, base.crop_y0)
    # the bias moves the hardware offset
    assert biased.offset_x == base.offset_x + 30
    assert biased.offset_y == base.offset_y - 12


def test_plan_offset_snaps_to_increment():
    plan = plan_aoi(target=(1900, 1900), step=STEP, max_size=MAX, offset_step=(16, 16), bias=(0, 0))
    assert plan.offset_x % 16 == 0
    assert plan.offset_y % 16 == 0


def test_plan_bias_clamps_near_max_graceful_degrade():
    # an AOI nearly filling the sensor leaves almost no room for the bias;
    # the offset clamps into [0, max-acq] rather than reading off-sensor.
    plan = plan_aoi(
        target=(3504, 3504), step=STEP, max_size=MAX, offset_step=(1, 1), bias=(9999, 9999)
    )
    assert 0 <= plan.offset_x <= MAX[0] - plan.acq_width
    assert 0 <= plan.offset_y <= MAX[1] - plan.acq_height


# --- center_crop -----------------------------------------------------------


def test_center_crop_mono_region():
    img = np.arange(1920 * 1900, dtype=np.uint16).reshape(1900, 1920)
    out = center_crop(img, x0=10, y0=0, width=1900, height=1900)
    assert out.shape == (1900, 1900)
    assert np.array_equal(out, img[0:1900, 10:1910])


def test_center_crop_passes_channels_through():
    img = np.zeros((1900, 1920, 3), dtype=np.uint8)
    out = center_crop(img, x0=10, y0=0, width=1900, height=1900)
    assert out.shape == (1900, 1900, 3)


def test_center_crop_returns_view():
    img = np.zeros((1900, 1920), dtype=np.uint16)
    out = center_crop(img, x0=10, y0=0, width=1900, height=1900)
    out[0, 0] = 4242
    assert img[0, 10] == 4242  # a view, not a copy


def test_center_crop_rejects_out_of_bounds_window():
    img = np.zeros((1900, 1920), dtype=np.uint16)
    with pytest.raises(ValueError):
        center_crop(img, x0=30, y0=0, width=1900, height=1900)  # 30 + 1900 > 1920


# --- centering correctness + validation ------------------------------------


def test_plan_crop_centers_on_axis_with_fine_offset():
    # with a 1px offset grid the AOI offset absorbs the full bias, so the
    # delivered window centers on the optical axis to the pixel -- even on the
    # height axis here, where the AOI has zero surplus to crop.
    bx, by = 23, -17
    plan = plan_aoi(target=(1900, 1900), step=STEP, max_size=MAX, offset_step=(1, 1), bias=(bx, by))
    axis_x = MAX[0] // 2 + bx
    axis_y = MAX[1] // 2 + by
    assert plan.offset_x + plan.crop_x0 + plan.crop_width // 2 == axis_x
    assert plan.offset_y + plan.crop_y0 + plan.crop_height // 2 == axis_y


def test_plan_offset_stays_on_grid_when_bias_clamps_near_max():
    # a bias large enough to push the offset against the high bound must still
    # land on the offset grid, not pin to the off-grid (max - acq) bound.
    plan = plan_aoi(
        target=(3504, 3504), step=STEP, max_size=MAX, offset_step=(32, 32), bias=(9999, 9999)
    )
    assert plan.offset_x % 32 == 0
    assert plan.offset_y % 32 == 0
    assert 0 <= plan.offset_x <= MAX[0] - plan.acq_width
    assert 0 <= plan.offset_y <= MAX[1] - plan.acq_height


def test_plan_rejects_nonpositive_target():
    with pytest.raises(ValueError):
        plan_aoi(target=(0, 1900), step=STEP, max_size=MAX, offset_step=(1, 1))


def test_plan_rejects_max_below_one_step():
    with pytest.raises(ValueError):
        plan_aoi(target=(10, 10), step=(48, 4), max_size=(40, 40), offset_step=(1, 1))
