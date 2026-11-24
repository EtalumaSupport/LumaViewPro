# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An image the cell count cannot read is refused by one reader, naming it.

The panel's preview caught the read's failure, logged a warning and
returned, so choosing an unreadable file changed nothing and said nothing.
The folder count caught the same failure in its own words. One reader in
src/_lumascope/modules/post_processing.py now refuses the file by type, naming it, for
both; the panel reports it through the GUI boundary. A file that is not
there is refused as any path that names nothing is (``not_found``).
"""

from __future__ import annotations

import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from _lumascope.modules.cell_count_method import CellCountMethod
import ui.post_processing as panel_module
from _lumascope.modules.exceptions import (
    LiveFolderPathRefusedError,
    PostProcessingFailedError,
    PostProcessingRefusedError,
)
from _lumascope.modules.post_processing import (
    PostProcessing,
    read_cell_count_image,
)
from tests.installation_fixtures import shipped_grids


def test_a_missing_file_is_refused_naming_it(tmp_path):
    path = tmp_path / 'nowhere.tif'
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        read_cell_count_image(path)
    assert refused.value.reason == 'path_not_found'
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
        PostProcessing().apply_cell_count_to_folder(
            str(tmp_path), CellCountMethod(), tiling_grids=shipped_grids()
        )
    # The folder names the file itself, so it states the refusal's detail.
    assert refused.value.detail in str(refused.value)
    assert list(failed.value.errors) == [('broken.tif', refused.value.detail)]


@pytest.mark.skipif(
    sys.platform == 'win32' or os.geteuid() == 0,
    reason='a file mode binds neither on Windows nor for root',
)
def test_the_folder_count_refuses_an_image_the_system_will_not_open_and_counts_the_rest(
    tmp_path,
):
    tifffile.imwrite(tmp_path / 'good.tif', np.zeros((8, 8), np.uint8))
    locked = tmp_path / 'locked.tif'
    tifffile.imwrite(locked, np.zeros((8, 8), np.uint8))
    locked.chmod(0)
    try:
        with pytest.raises(PostProcessingFailedError) as failed:
            PostProcessing().apply_cell_count_to_folder(
                str(tmp_path), CellCountMethod(), tiling_grids=shipped_grids()
            )
    finally:
        locked.chmod(0o600)

    assert list(failed.value.errors) == [('locked.tif', 'Permission denied')]
    assert 'could not be counted: Permission denied (locked.tif).' in str(failed.value)
    results = pathlib.Path(failed.value.produced_paths[0]).read_text(encoding='utf-8')
    assert 'good.tif' in results and 'locked.tif' not in results, results


def test_a_folder_named_like_an_image_is_not_counted_and_named(tmp_path):
    tifffile.imwrite(tmp_path / 'good.tif', np.zeros((8, 8), np.uint8))
    (tmp_path / 'stack.tif').mkdir()

    with pytest.raises(PostProcessingFailedError) as failed:
        PostProcessing().apply_cell_count_to_folder(
            str(tmp_path), CellCountMethod(), tiling_grids=shipped_grids()
        )

    [(name, cause)] = failed.value.errors
    assert name == 'stack.tif' and 'stack.tif is a folder, not a file' in cause, cause


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
