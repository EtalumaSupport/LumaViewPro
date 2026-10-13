# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The cell-count method's area and perimeter sliders are mapped to sizes by the method's owner.

The mapping lived in the panel, where it used ``np.interp`` and wrote numpy
floats into the method (#818); the GUI displays, so the module maps, and a
bound it hands back is a Python float a method file can hold as it is.
"""

from modules import post_processing

_SHAPE = (1000, 2000)
_PIXELS_PER_UM = 2.0


def test_a_slider_position_maps_to_a_python_size_at_each_stop():
    for size, positions, sizes in (
        ('area', (30, 60), (1000.0, 10000.0)),
        ('perimeter', (50, 0), (100.0, 0.0)),
    ):
        bounds = post_processing.size_filter_from_slider(size, positions, _SHAPE, _PIXELS_PER_UM)

        assert bounds == sizes
        assert all(type(bound) is float for bound in bounds), size
        back = post_processing.size_filter_to_slider(size, bounds, _SHAPE, _PIXELS_PER_UM)
        assert back == positions
        assert all(type(position) is float for position in back), size


def test_the_top_of_a_slider_is_the_largest_size_the_image_holds():
    area = post_processing.area_filter_max(_SHAPE, _PIXELS_PER_UM)
    perimeter = post_processing.perimeter_filter_max(_SHAPE, _PIXELS_PER_UM)

    assert area == 1000 * 2000 / _PIXELS_PER_UM**2
    assert perimeter == 2 * (2 * 1000 + 2 * 2000) / _PIXELS_PER_UM
    assert post_processing.size_filter_from_slider('area', (0, 100), _SHAPE, _PIXELS_PER_UM) == (
        0.0,
        area,
    )
    assert post_processing.size_filter_from_slider(
        'perimeter', (0, 100), _SHAPE, _PIXELS_PER_UM
    ) == (0.0, perimeter)


def test_with_no_preview_the_positions_are_handed_back_as_they_are():
    assert post_processing.size_filter_from_slider('area', (10, 90), None, None) == (10, 90)
    assert post_processing.size_filter_to_slider('perimeter', (5, 50), None, None) == (5, 50)
