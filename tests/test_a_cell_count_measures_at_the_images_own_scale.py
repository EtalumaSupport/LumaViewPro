# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell count measures each image at its own scale, or in pixels, and says which.

The method carried a fixed 1.0 pixels per micron that described no image, so
every count on the default reported pixels as microns. Unknown stays unknown
(Eric, 2026-10-03): a count reads the scale the image states, a person may
override it, and an image with none is counted in pixels with its results
labelled so. Area and perimeter bounds are in microns, so they are refused for
an image with no scale rather than applied as if a pixel were a micron.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from PIL import Image

from modules import image_utils, post_processing
from modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from modules.post_processing import (
    PostProcessing,
    check_cell_count_method,
    default_cell_count_method,
    load_cell_count_method,
)

_BLOB = np.zeros((64, 64), dtype=np.uint8)
_BLOB[20:40, 20:40] = 255  # one bright square, 20 x 20 pixels


def _capture(path, pixel_size_um):
    image_utils.write_tiff(
        data=_BLOB,
        file_loc=path,
        metadata={
            'objective': {},
            'pixel_size_um': pixel_size_um,
            'channel': 'Green',
            'significant_bits': 8,
            'datetime': '2026-10-04T00:00:00',
        },
        ome=False,
        color='Green',
        significant_bits=8,
        save_encoding='right_aligned',
    )
    return path


def _png(path):
    Image.fromarray(_BLOB).save(path)
    return path


def _results(folder):
    with open(folder / 'results.csv', newline='') as f:
        return {row['file']: row for row in csv.DictReader(f)}


def test_the_default_method_states_no_scale_and_filters_no_size():
    method = default_cell_count_method()

    assert method['context']['pixels_per_um'] is None
    assert method['filters']['area'] == {'min': None, 'max': None}
    assert method['filters']['perimeter'] == {'min': None, 'max': None}
    check_cell_count_method(method)


@pytest.mark.parametrize(
    'override, image_pixel_size_um, expected',
    [
        (None, 0.5, 2.0),  # the image's own: 0.5 um per pixel is 2 pixels per um
        (None, None, None),  # no scale anywhere: pixels
        (4.0, 0.5, 4.0),  # a person's override wins
        (4.0, None, 4.0),
    ],
)
def test_the_count_scale_is_the_override_else_the_images_own(
    override, image_pixel_size_um, expected
):
    from modules.post_processing import cell_count_scale

    method = default_cell_count_method()
    method['context']['pixels_per_um'] = override

    assert cell_count_scale(method, image_pixel_size_um) == expected


def test_a_folder_count_measures_each_image_at_its_own_scale(tmp_path):
    _capture(tmp_path / 'scaled.tiff', pixel_size_um=0.5)
    _png(tmp_path / 'unscaled.png')

    PostProcessing().apply_cell_count_to_folder(tmp_path, default_cell_count_method())

    rows = _results(tmp_path)
    scaled, unscaled = rows['scaled.tiff'], rows['unscaled.png']
    assert scaled['area_unit'] == 'um2'
    assert unscaled['area_unit'] == 'px2'
    # Same pixels: the micron area is the pixel area at 0.5 um per pixel.
    assert float(scaled['total_object_area']) == pytest.approx(
        float(unscaled['total_object_area']) * 0.25, abs=0.01
    )


def test_an_override_scale_applies_to_every_image(tmp_path):
    _png(tmp_path / 'unscaled.png')
    method = default_cell_count_method()
    method['context']['pixels_per_um'] = 2.0

    PostProcessing().apply_cell_count_to_folder(tmp_path, method)

    assert _results(tmp_path)['unscaled.png']['area_unit'] == 'um2'


def test_a_size_filter_is_refused_for_an_image_with_no_scale(tmp_path):
    _capture(tmp_path / 'scaled.tiff', pixel_size_um=0.5)
    _png(tmp_path / 'unscaled.png')
    method = default_cell_count_method()
    method['filters']['area'] = {'min': 10, 'max': None}

    with pytest.raises(PostProcessingFailedError) as failed:
        PostProcessing().apply_cell_count_to_folder(tmp_path, method)

    assert any('unscaled.png' in error for error in failed.value.errors)
    rows = _results(tmp_path)
    assert list(rows) == ['scaled.tiff'], 'the scaled image is still counted'


def test_a_preview_of_an_image_with_no_scale_refuses_a_size_filter():
    method = default_cell_count_method()
    method['filters']['perimeter'] = {'min': None, 'max': 50}

    with pytest.raises(PostProcessingRefusedError) as refused:
        PostProcessing().preview_cell_count(
            image=_BLOB, settings=method, significant_bits=8, pixels_per_um=None
        )

    assert refused.value.reason == 'no_scale'


def test_a_preview_without_a_scale_measures_in_pixels():
    _, stats = PostProcessing().preview_cell_count(
        image=_BLOB, settings=default_cell_count_method(), significant_bits=8, pixels_per_um=None
    )

    (region,) = stats['regions'].values()
    assert region['area']['units'] == 'px^2'
    assert region['perimeter']['units'] == 'px'
    assert stats['summary']['area_unit'] == 'px2'


@pytest.mark.parametrize('bad', [0, -1.0, 'x', True])
def test_a_scale_that_is_not_a_positive_number_is_refused(bad):
    method = default_cell_count_method()
    method['context']['pixels_per_um'] = bad

    with pytest.raises(PostProcessingRefusedError):
        check_cell_count_method(method)


def test_a_version_one_method_files_old_default_scale_is_dropped_and_said(tmp_path, monkeypatch):
    method = default_cell_count_method()
    method['context']['pixels_per_um'] = 1.0
    path = tmp_path / 'old.json'
    path.write_text(
        json.dumps({**method, 'metadata': {'type': 'cell_count_method', 'version': '1'}})
    )
    shown = []
    monkeypatch.setattr(
        post_processing.notifications,
        'report_outcome',
        lambda outcome, **kw: shown.append(outcome),
    )

    loaded = load_cell_count_method(path)

    assert loaded['context']['pixels_per_um'] is None
    (notice,) = shown
    assert notice.reason == 'cell_count_scale_dropped'


def test_a_saved_methods_scale_is_honoured(tmp_path):
    method = default_cell_count_method()
    method['context']['pixels_per_um'] = 1.0
    path = tmp_path / 'new.json'
    post_processing.save_cell_count_method(method, path)

    assert load_cell_count_method(path)['context']['pixels_per_um'] == 1.0
