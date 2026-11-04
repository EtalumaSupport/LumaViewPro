# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell-count method the count cannot use cannot be built.

A pixels-per-micron of 0 or below counted no cells on an image with two, and
NaN wrote NaN areas into results.csv; an incomplete method faulted with a
KeyError; and the only check on any of it was the panel's text box, so a
method file or a REST caller's dict reached the count unchecked. The method
is now a record, ``CellCountMethod`` in modules/cell_count_method.py, that
refuses, naming the field, as it is built: a method a caller holds is one
the count can use, wherever it came from.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds
from modules.exceptions import PostProcessingRefusedError
from modules.post_processing import PostProcessing


def _blobs() -> np.ndarray:
    image = np.zeros((200, 200), np.uint8)
    image[50:80, 50:80] = 200
    image[120:160, 120:150] = 180
    return image


def test_the_default_method_counts_both_blobs():
    # The default's size filters are open, so every blob is kept.
    _, stats = PostProcessing().preview_cell_count(
        image=_blobs(), method=CellCountMethod(), significant_bits=8, pixels_per_um=None
    )
    assert stats['summary']['num_regions'] == 2


# None is no override (each image's own scale), so it is not in this list.
@pytest.mark.parametrize('scale', [0, 0.0, -2.0, float('nan'), float('inf'), '2', True])
def test_a_scale_that_is_not_a_positive_number_is_refused_naming_it(scale):
    with pytest.raises(PostProcessingRefusedError) as refused:
        CellCountMethod(pixels_per_um=scale)
    assert refused.value.reason == 'method_invalid'
    assert 'pixels_per_um' in str(refused.value)


def test_a_filter_whose_min_is_above_its_max_is_refused_naming_it():
    with pytest.raises(PostProcessingRefusedError) as refused:
        CellCountFilters(sphericity=FilterBounds(0.8, 0.2))
    assert refused.value.reason == 'method_invalid'
    assert 'filters.sphericity' in str(refused.value)


def test_an_open_bound_is_a_usable_filter():
    filters = CellCountFilters(
        area_um2=FilterBounds(None, None), perimeter_um=FilterBounds(None, 50)
    )
    assert CellCountMethod(filters=filters).filters.perimeter_um.max == 50


@pytest.mark.parametrize('flag', ['yes', 1, None])
def test_a_fluorescent_mode_that_is_not_true_or_false_is_refused(flag):
    with pytest.raises(PostProcessingRefusedError) as refused:
        CellCountMethod(fluorescent_mode=flag)
    assert refused.value.reason == 'method_invalid'
    assert 'fluorescent_mode' in str(refused.value)


@pytest.mark.parametrize('threshold', ['20', None, float('nan')])
def test_a_threshold_that_is_not_a_number_is_refused_naming_it(threshold):
    with pytest.raises(PostProcessingRefusedError) as refused:
        CellCountMethod(threshold_pct=threshold)
    assert refused.value.reason == 'method_invalid'
    assert 'threshold' in str(refused.value)


@pytest.mark.parametrize('bound', ['10', [1], True])
def test_a_filter_bound_that_is_not_a_number_is_refused_naming_it(bound):
    with pytest.raises(PostProcessingRefusedError) as refused:
        CellCountFilters(intensity_max_pct=FilterBounds(min=bound, max=100))
    assert refused.value.reason == 'method_invalid'
    assert 'filters.intensity_max_pct.min' in str(refused.value)


def test_filters_that_are_not_their_records_are_refused_naming_them():
    with pytest.raises(PostProcessingRefusedError) as as_dict:
        CellCountFilters(area_um2={'min': None, 'max': None})
    assert 'filters.area_um2' in str(as_dict.value)
    with pytest.raises(PostProcessingRefusedError) as whole:
        CellCountMethod(filters={})
    assert whole.value.reason == 'method_invalid'
    assert 'filters must be' in str(whole.value)


def test_a_change_is_a_new_method_and_is_refused_the_same():
    method = CellCountMethod()
    with pytest.raises(PostProcessingRefusedError) as refused:
        dataclasses.replace(method, pixels_per_um=-1.0)
    assert refused.value.reason == 'method_invalid'
    assert method.pixels_per_um is None


def test_the_panels_apply_to_preview_is_reported_through_the_boundary(monkeypatch):
    from types import SimpleNamespace

    import ui.post_processing as panel_module

    calls = []
    monkeypatch.setattr(
        panel_module, 'run_reported', lambda call, redraw, label: calls.append((call, label))
    )
    regenerate = object()
    panel = SimpleNamespace(_regenerate_image_preview=regenerate)
    panel_module.CellCountControls.apply_method_to_preview_image(panel)
    assert calls == [(regenerate, 'APPLY_METHOD_TO_PREVIEW')]
