# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell-count method the count cannot use is refused by its owner, at every entry.

A pixels-per-micron of 0 or below counted no cells on an image with two, and
NaN wrote NaN areas into results.csv; an incomplete method faulted with a
KeyError; and the only check on any of it was the panel's text box, so a
method file or a REST caller's dict reached the count unchecked. The owner
in modules/post_processing.py holds the method's defaults and refuses, by
type, naming the field, wherever a method enters: the one-image preview and
the session's folder count, before the count is queued.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from modules.exceptions import PostProcessingRefusedError
from modules.post_processing import (
    PostProcessing,
    check_cell_count_method,
    default_cell_count_method,
)


def _blobs() -> np.ndarray:
    image = np.zeros((200, 200), np.uint8)
    image[50:80, 50:80] = 200
    image[120:160, 120:150] = 180
    return image


def _method(**context) -> dict:
    # The default's size filters are open, so every blob is kept.
    method = default_cell_count_method()
    method['context'].update(context)
    return method


def test_the_default_method_counts_both_blobs():
    _, stats = PostProcessing().preview_cell_count(
        image=_blobs(), settings=_method(), significant_bits=8, pixels_per_um=None
    )
    assert stats['summary']['num_regions'] == 2


def test_the_default_method_passes_its_own_check():
    check_cell_count_method(default_cell_count_method())


def test_each_default_is_a_fresh_copy():
    first = default_cell_count_method()
    first['context']['pixels_per_um'] = 9.0
    assert default_cell_count_method()['context']['pixels_per_um'] is None


def test_a_method_needs_no_file_metadata():
    method = default_cell_count_method()
    assert 'metadata' not in method
    check_cell_count_method(method)


# None is no override (each image's own scale), so it is not in this list.
@pytest.mark.parametrize('scale', [0, 0.0, -2.0, float('nan'), float('inf'), '2', True])
def test_a_scale_that_is_not_a_positive_number_is_refused_naming_it(scale):
    with pytest.raises(PostProcessingRefusedError) as refused:
        PostProcessing().preview_cell_count(
            image=_blobs(),
            settings=_method(pixels_per_um=scale),
            significant_bits=8,
            pixels_per_um=None,
        )
    assert refused.value.reason == 'method_invalid'
    assert 'pixels_per_um' in str(refused.value)


@pytest.mark.parametrize(
    'remove',
    [
        ('segmentation',),
        ('segmentation', 'algorithm'),
        ('segmentation', 'parameters', 'threshold'),
        ('context', 'fluorescent_mode'),
        ('filters', 'intensity', 'mean'),
        ('filters', 'area', 'max'),
    ],
)
def test_a_method_missing_a_field_is_refused_naming_it(remove):
    method = _method()
    holder = method
    for key in remove[:-1]:
        holder = holder[key]
    del holder[remove[-1]]
    with pytest.raises(PostProcessingRefusedError) as refused:
        check_cell_count_method(method)
    assert '.'.join(remove) in str(refused.value)


def test_a_filter_whose_min_is_above_its_max_is_refused_naming_it():
    method = _method()
    method['filters']['sphericity'] = {'min': 0.8, 'max': 0.2}
    with pytest.raises(PostProcessingRefusedError) as refused:
        check_cell_count_method(method)
    assert 'filters.sphericity' in str(refused.value)


def test_an_open_bound_is_a_usable_filter():
    method = _method()
    method['filters']['area'] = {'min': None, 'max': None}
    check_cell_count_method(method)


@pytest.mark.parametrize('flag', ['yes', 1, None])
def test_a_fluorescent_mode_that_is_not_true_or_false_is_refused(flag):
    with pytest.raises(PostProcessingRefusedError):
        check_cell_count_method(_method(fluorescent_mode=flag))


@pytest.mark.parametrize('method', [['context'], 'context', None])
def test_a_method_that_is_not_a_dict_is_refused(method):
    with pytest.raises(PostProcessingRefusedError) as refused:
        check_cell_count_method(method)
    assert 'context.pixels_per_um' in str(refused.value)


def test_the_session_refuses_a_bad_method_before_the_count_is_queued(tmp_path):
    from modules.post_processing_api import PostProcessingAPI

    class _Lane:
        def __init__(self):
            self.queued = []

        def call(self, *a, **k):
            self.queued.append(a)

    lane = _Lane()
    api = PostProcessingAPI(
        lane=lane,
        tiling_configs_path=lambda: tmp_path,
        has_turret=lambda: False,
        settings_snapshot=dict,
    )
    with pytest.raises(PostProcessingRefusedError) as refused:
        api.count_cells(tmp_path, method=_method(pixels_per_um=0))
    assert refused.value.reason == 'method_invalid'
    assert lane.queued == []


def test_the_refusal_leaves_the_callers_method_as_it_was():
    method = _method(pixels_per_um=-1.0)
    before = copy.deepcopy(method)
    with pytest.raises(PostProcessingRefusedError):
        check_cell_count_method(method)
    assert method == before


@pytest.mark.parametrize('threshold', ['20', None, float('nan')])
def test_a_threshold_that_is_not_a_number_is_refused_naming_it(threshold):
    method = _method()
    method['segmentation']['parameters']['threshold'] = threshold
    with pytest.raises(PostProcessingRefusedError) as refused:
        check_cell_count_method(method)
    assert 'segmentation.parameters.threshold' in str(refused.value)


@pytest.mark.parametrize('bound', ['10', [1], True])
def test_a_filter_bound_that_is_not_a_number_is_refused_naming_it(bound):
    method = _method()
    method['filters']['intensity']['max']['min'] = bound
    with pytest.raises(PostProcessingRefusedError) as refused:
        check_cell_count_method(method)
    assert 'filters.intensity.max.min' in str(refused.value)


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
