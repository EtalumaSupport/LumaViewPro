# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell-count preview is the Session's, and counts an image as the folder count does.

The panel previewed through its own ``PostProcessing`` instance, choosing
the scale itself, so the preview was the GUI's alone and a script had no
way to see what a method would keep before counting a folder.
``session.post_processing.preview_cell_count`` now counts the pixels the
caller holds, at the scale ``count_cells`` would use for the image's file,
and answers the outlined image with the counts of the row ``count_cells``
writes for it.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import ui.post_processing as panel_module
from modules import image_utils
from modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds
from modules.exceptions import PostProcessingRefusedError
from modules.layer_record import release_color
from modules.post_processing import read_cell_count_image
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED1

_FIELD = np.zeros((64, 64), dtype=np.uint8)
_FIELD[8:20, 8:20] = 255  # a 12 x 12 square
_FIELD[30:56, 30:56] = 255  # a 26 x 26 square

# Keeps the larger square only, at the image's 0.5 um per pixel: 36 um2 and 169 um2.
_METHOD = CellCountMethod(filters=CellCountFilters(area_um2=FilterBounds(min=100, max=None)))


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def _capture(path, pixel_size_um):
    image_utils.write_tiff(
        data=_FIELD,
        file_loc=path,
        metadata={
            'objective': {},
            'pixel_size_um': pixel_size_um,
            'channel': LED1,
            'significant_bits': 8,
            'datetime': '2026-10-21T00:00:00',
        },
        ome=False,
        color=release_color(LED1),
        significant_bits=8,
        save_encoding='right_aligned',
        white_bits=None,
    )
    return path


def test_the_preview_counts_an_image_as_the_folder_count_counts_its_file(session, tmp_path):
    folder = tmp_path / 'cells'
    folder.mkdir()
    path = _capture(folder / 'scaled.tiff', pixel_size_um=0.5)
    session.post_processing.count_cells(folder, method=_METHOD)
    with open(folder / 'results.csv', newline='') as f:
        (row,) = csv.DictReader(f)

    pixels, significant_bits = read_cell_count_image(path)
    preview = session.post_processing.preview_cell_count(
        pixels,
        _METHOD,
        significant_bits=significant_bits,
        pixel_size_um=image_utils.read_pixel_size_um(path),
    )

    counts = preview.counts
    assert counts.num_cells == int(row['num_cells']) == 1
    assert counts.total_object_area_px2 == pytest.approx(float(row['total_object_area_px2']))
    assert counts.total_object_area_um2 == pytest.approx(float(row['total_object_area_um2']))
    assert counts.total_object_intensity == pytest.approx(float(row['total_object_intensity']))
    assert preview.image.shape[:2] == pixels.shape[:2]
    assert not np.array_equal(preview.image, pixels), 'no outline was drawn'


def test_the_methods_scale_overrides_the_images_own(session):
    method = CellCountMethod(pixels_per_um=1.0, filters=_METHOD.filters)
    counts = session.post_processing.preview_cell_count(
        _FIELD, method, significant_bits=8, pixel_size_um=0.5
    ).counts
    # At 1 pixel per um both squares are over 100 um2.
    assert counts.num_cells == 2
    assert counts.total_object_area_um2 == counts.total_object_area_px2


def test_an_image_with_no_scale_is_counted_in_pixels_and_refuses_a_size_filter(session):
    counts = session.post_processing.preview_cell_count(
        _FIELD, CellCountMethod(), significant_bits=8, pixel_size_um=None
    ).counts
    assert (counts.num_cells, counts.total_object_area_um2) == (2, None)
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.preview_cell_count(
            _FIELD, _METHOD, significant_bits=8, pixel_size_um=None
        )
    assert refused.value.reason == 'no_scale'


def test_the_panel_shows_the_sessions_preview(session, monkeypatch):
    shown = SimpleNamespace(texture=None)
    monkeypatch.setattr(
        panel_module._app_ctx,
        'ctx',
        SimpleNamespace(
            session=session, cell_count_content=SimpleNamespace(ids={'cell_count_image_id': shown})
        ),
    )
    textures = []
    monkeypatch.setattr(
        panel_module.image_utils_kivy,
        'image_to_texture',
        lambda image, existing: textures.append(image) or 'texture',
    )
    panel = SimpleNamespace(
        _preview_source_image=_FIELD,
        _preview_source_significant_bits=8,
        _preview_pixel_size_um=0.5,
        _preview_image=None,
        _method=_METHOD,
    )

    panel_module.CellCountControls._regenerate_image_preview(panel)

    expected = session.post_processing.preview_cell_count(
        _FIELD, _METHOD, significant_bits=8, pixel_size_um=0.5
    ).image
    assert np.array_equal(panel._preview_image, expected)
    assert len(textures) == 1 and np.array_equal(textures[0], expected)
    assert shown.texture == 'texture'
