# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An image the cell count cannot read is refused by one reader, naming it.

The panel's preview caught the read's failure, logged a warning and
returned, so choosing an unreadable file changed nothing and said nothing.
The folder count caught the same failure in its own words. One reader in
modules/post_processing.py now refuses the file by type, naming it, for
both; the panel reports it through the GUI boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import ui.post_processing as panel_module
from modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from modules.post_processing import (
    PostProcessing,
    default_cell_count_method,
    read_cell_count_image,
)


def test_a_missing_file_is_refused_naming_it(tmp_path):
    path = tmp_path / 'nowhere.tif'
    with pytest.raises(PostProcessingRefusedError) as refused:
        read_cell_count_image(path)
    assert refused.value.reason == 'unreadable'
    assert str(path) in str(refused.value)


def test_a_file_that_is_not_an_image_is_refused_naming_it(tmp_path):
    path = tmp_path / 'broken.tif'
    path.write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingRefusedError) as refused:
        read_cell_count_image(path)
    assert refused.value.reason == 'unreadable'
    assert str(path) in str(refused.value)


def test_an_image_reads_with_its_depth(tmp_path):
    path = tmp_path / 'cells.tif'
    tifffile.imwrite(path, np.full((8, 8), 200, np.uint8))
    image, bits = read_cell_count_image(path)
    assert image.shape == (8, 8)
    assert bits == 8


def test_the_folder_count_names_an_unreadable_image_in_the_same_words(tmp_path):
    tifffile.imwrite(tmp_path / 'good.tif', np.zeros((8, 8), np.uint8))
    broken = tmp_path / 'broken.tif'
    broken.write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingRefusedError) as refused:
        read_cell_count_image(broken)
    with pytest.raises(PostProcessingFailedError) as failed:
        PostProcessing().apply_cell_count_to_folder(str(tmp_path), default_cell_count_method())
    assert list(failed.value.errors) == [str(refused.value)]


def test_the_panel_reports_an_unreadable_preview_and_keeps_its_image(tmp_path, monkeypatch):
    outcomes = []

    def reported(call, redraw, label):
        try:
            call()
        except Exception as e:
            outcomes.append((label, e))
        if redraw is not None:
            redraw()

    monkeypatch.setattr(panel_module, 'run_reported', reported)
    shown = []
    panel = SimpleNamespace(_preview_source_significant_bits=16, set_preview_source=shown.append)
    broken = tmp_path / 'broken.tif'
    broken.write_bytes(b'not a tiff')
    panel_module.CellCountControls.set_preview_source_file(panel, str(broken))
    [(label, refused)] = outcomes
    assert label == 'LOAD_CELL_COUNT_INPUT_IMAGE'
    assert isinstance(refused, PostProcessingRefusedError)
    assert shown == []
    assert panel._preview_source_significant_bits == 16
