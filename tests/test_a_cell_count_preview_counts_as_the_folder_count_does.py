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

import datetime
import csv
from types import SimpleNamespace

import numpy as np
import pytest

import ui.post_processing as panel_module
from _lumascope.modules import image_utils
from _lumascope.modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds
from _lumascope.modules.exceptions import ArgumentRefusedError, PostProcessingRefusedError
from _lumascope.modules.channel_record import release_color
from _lumascope.modules.post_processing import CellCountImage, read_cell_count_image
from _lumascope.modules.scope_session import ScopeSession
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
            'datetime': datetime.datetime(2026, 10, 21, 0, 0, 0),
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
    with open(folder / 'results.csv', encoding='utf-8', newline='') as f:
        (row,) = csv.DictReader(f)

    image = read_cell_count_image(path)
    preview = session.post_processing.preview_cell_count(
        image, _METHOD, pixel_size_um=image_utils.read_pixel_size_um(path)
    )

    counts = preview.counts
    assert image.channel == LED1
    assert counts.num_cells == int(row['num_cells']) == 1
    assert counts.total_object_area_px2 == pytest.approx(float(row['total_object_area_px2']))
    assert counts.total_object_area_um2 == pytest.approx(float(row['total_object_area_um2']))
    assert counts.total_object_intensity == pytest.approx(float(row['total_object_intensity']))
    assert preview.image.shape == (*image.pixels.shape, 3)
    assert _colours(preview.image) == {_BLACK, _GREEN, _MAGENTA}, 'no outline was drawn'


_BLACK, _GREY_255 = (0, 0, 0), (255, 255, 255)
_GREEN, _MAGENTA, _ORANGE = (0, 255, 0), (255, 0, 255), (255, 100, 0)


def _colours(image: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(v) for v in pixel) for pixel in np.unique(image.reshape(-1, 3), axis=0)}


def test_an_image_of_a_channel_is_shown_in_its_colour_with_its_complement_outlined(session):
    preview = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, LED1),
        CellCountMethod(),
        pixel_size_um=None,
    )

    # The squares are the channel's green; their outlines magenta, never white.
    assert _colours(preview.image) == {_BLACK, _GREEN, _MAGENTA}
    assert tuple(preview.image[14, 14]) == _GREEN


def test_an_image_that_states_no_channel_is_grey_with_orange_outlines(session):
    preview = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, None),
        CellCountMethod(),
        pixel_size_um=None,
    )

    assert _colours(preview.image) == {_BLACK, _GREY_255, _ORANGE}


def test_a_pale_colour_is_outlined_in_its_complement_brightened_to_full(session):
    session.set_channel_settings(LED1, display_color='#ffffef')

    preview = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, LED1),
        CellCountMethod(),
        pixel_size_um=None,
    )

    # The complement of #ffffef is (0, 0, 16): drawn at full, blue.
    assert (0, 0, 255) in _colours(preview.image)


def test_a_colour_image_is_shown_in_its_own_colours_outlined_against_its_mean(session):
    red = np.zeros((*_FIELD.shape, 3), dtype=np.uint8)
    red[..., 0] = _FIELD

    preview = session.post_processing.preview_cell_count(
        CellCountImage(red, 8, None),
        CellCountMethod(),
        pixel_size_um=None,
    )

    # Red objects, outlined in red's complement, cyan.
    assert _colours(preview.image) == {_BLACK, (255, 0, 0), (0, 255, 255)}


def test_a_save_baked_in_its_channels_colour_counts_as_its_mono(session):
    yellow = np.zeros((*_FIELD.shape, 3), dtype=np.uint8)
    # A yellow channel's save: the mono on its red and green planes. Read by
    # luminance its squares are 226 of 255; read as the channel's, 255.
    yellow[..., 0] = _FIELD
    yellow[..., 1] = _FIELD
    session.set_channel_settings(LED1, display_color='#ffff00')
    mono = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, LED1),
        CellCountMethod(),
        pixel_size_um=None,
    ).counts

    baked = session.post_processing.preview_cell_count(
        CellCountImage(yellow, 8, LED1),
        CellCountMethod(),
        pixel_size_um=None,
    ).counts

    assert baked == mono


def test_a_channel_this_release_does_not_know_is_refused(session):
    with pytest.raises(ArgumentRefusedError) as refused:
        session.post_processing.preview_cell_count(
            CellCountImage(_FIELD, 8, 'Ex999'),
            CellCountMethod(),
            pixel_size_um=None,
        )
    assert refused.value.reason == 'channel_unknown'


def test_the_methods_scale_overrides_the_images_own(session):
    method = CellCountMethod(pixels_per_um=1.0, filters=_METHOD.filters)
    counts = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, None),
        method,
        pixel_size_um=0.5,
    ).counts
    # At 1 pixel per um both squares are over 100 um2.
    assert counts.num_cells == 2
    assert counts.total_object_area_um2 == counts.total_object_area_px2


def test_an_image_with_no_scale_is_counted_in_pixels_and_refuses_a_size_filter(session):
    counts = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, None),
        CellCountMethod(),
        pixel_size_um=None,
    ).counts
    assert (counts.num_cells, counts.total_object_area_um2) == (2, None)
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.preview_cell_count(
            CellCountImage(_FIELD, 8, None),
            _METHOD,
            pixel_size_um=None,
        )
    assert refused.value.reason == 'no_scale'


def test_a_depth_no_frame_has_is_refused_with_its_limits(session):
    twelve_bit = _FIELD.astype(np.uint16) * 16
    refused = {}
    for depth in (-1, 0, 17):
        with pytest.raises(ArgumentRefusedError) as raised:
            session.post_processing.preview_cell_count(
                CellCountImage(twelve_bit, depth, None),
                CellCountMethod(),
                pixel_size_um=None,
            )
        refused[depth] = (raised.value.reason, raised.value.limits)

    assert refused == dict.fromkeys((-1, 0, 17), ('out_of_range', (1, 16)))
    counts = session.post_processing.preview_cell_count(
        CellCountImage(twelve_bit, 12, None),
        CellCountMethod(),
        pixel_size_um=None,
    ).counts
    assert counts.num_cells == 2


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
        _preview_source=CellCountImage(_FIELD, 8, LED1),
        _preview_pixel_size_um=0.5,
        _preview_image=None,
        _method=_METHOD,
    )

    panel_module.CellCountControls._regenerate_image_preview(panel)

    expected = session.post_processing.preview_cell_count(
        CellCountImage(_FIELD, 8, LED1),
        _METHOD,
        pixel_size_um=0.5,
    ).image
    assert np.array_equal(panel._preview_image, expected)
    assert len(textures) == 1 and np.array_equal(textures[0], expected)
    assert shown.texture == 'texture'
