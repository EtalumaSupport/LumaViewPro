# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a cell-count size slider stands for is the Session's read.

The panel mapped its area and perimeter sliders, sized their reach and chose
the scale the preview counts at by calling the module below the API, so a
REST client building the same form had none of it.
``session.post_processing.cell_count_size_scale`` answers the scale and,
for each size, the slider's stops and reach; a bound it maps is a Python
float a method file can hold as it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.cell_count_method import CellCountMethod
from modules.exceptions import PostProcessingRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_SHAPE = (1000, 2000)
_PIXEL_SIZE_UM = 0.5  # 2 pixels per micron


@pytest.fixture
def post_processing(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session.post_processing
    session.shutdown()


def test_a_slider_position_maps_to_a_python_size_at_each_stop(post_processing):
    scale = post_processing.cell_count_size_scale(
        CellCountMethod(), shape=_SHAPE, pixel_size_um=_PIXEL_SIZE_UM
    )

    for stops, positions, sizes in (
        (scale.area, (30, 60), (1000.0, 10000.0)),
        (scale.perimeter, (50, 0), (100.0, 0.0)),
    ):
        bounds = stops.size_at(positions)

        assert bounds == sizes
        assert all(type(bound) is float for bound in bounds)
        back = stops.positions_of(bounds)
        assert back == positions
        assert all(type(position) is float for position in back)


def test_the_top_of_a_slider_is_the_largest_size_the_image_holds(post_processing):
    scale = post_processing.cell_count_size_scale(
        CellCountMethod(), shape=_SHAPE, pixel_size_um=_PIXEL_SIZE_UM
    )

    assert scale.area.reach == 1000 * 2000 / 2.0**2
    assert scale.perimeter.reach == 2 * (2 * 1000 + 2 * 2000) / 2.0
    assert scale.area.size_at((0, 100)) == (0.0, scale.area.reach)
    assert scale.perimeter.size_at((0, 100)) == (0.0, scale.perimeter.reach)


def test_a_methods_scale_wins_over_the_images(post_processing):
    def scale_of(method, pixel_size_um):
        return post_processing.cell_count_size_scale(
            method, shape=_SHAPE, pixel_size_um=pixel_size_um
        ).pixels_per_um

    assert scale_of(CellCountMethod(pixels_per_um=4.0), _PIXEL_SIZE_UM) == 4.0
    assert scale_of(CellCountMethod(), _PIXEL_SIZE_UM) == 2.0
    assert scale_of(CellCountMethod(), None) is None


def test_a_shape_or_scale_no_image_has_is_refused_not_answered(post_processing):
    # Over the wire these were a 500 (a scale of 0, a one-sided shape) or a
    # record a form would draw (a negative scale, a side of 0).
    refused = []
    for shape, pixel_size_um in (
        (_SHAPE, 0),
        (_SHAPE, -1.0),
        (_SHAPE, float('nan')),
        ((64,), _PIXEL_SIZE_UM),
        ((0, 64), _PIXEL_SIZE_UM),
        ((True, 64), _PIXEL_SIZE_UM),
    ):
        with pytest.raises(PostProcessingRefusedError) as raised:
            post_processing.cell_count_size_scale(
                CellCountMethod(), shape=shape, pixel_size_um=pixel_size_um
            )
        refused.append(raised.value.reason)
    with pytest.raises(PostProcessingRefusedError) as preview:
        post_processing.preview_cell_count(
            np.zeros((8, 8), dtype=np.uint8), CellCountMethod(), significant_bits=8, pixel_size_um=0
        )

    assert refused == ['invalid_setting'] * 6
    assert preview.value.reason == 'invalid_setting'


def test_with_no_preview_a_position_is_its_own_size(post_processing):
    scale = post_processing.cell_count_size_scale(CellCountMethod())

    assert scale.area.size_at((10, 90)) == (10.0, 90.0)
    assert scale.perimeter.positions_of((5, 50)) == (5.0, 50.0)
