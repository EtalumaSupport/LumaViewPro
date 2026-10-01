# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Cell count and Quick Enhance over a folder answer with what they made.

Cell count's folder pass returned nothing, so the button read every
success as "FAILED" (#763); a folder with nothing it could read rewrote
an existing ``results.csv`` with only a header (#765); a failed write was
logged and swallowed. Quick Enhance's folder pass said "complete" with
images skipped. Each now returns only a complete result: nothing to work
on is a refusal that leaves the previous results alone, and a partial
pass is a failure that carries what it did save.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from modules import image_utils
from modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from modules.post_processing import PostProcessing
from modules.quick_enhance import QuickEnhancer, QuickEnhanceSettings

OLD_RESULTS = 'file,time,num_cells\nold.tif,yesterday,7\n'


def _write_tiff(path):
    image_utils.write_tiff(
        data=np.full((8, 8), 200, dtype=np.uint8),
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'Green',
            'objective': '10x',
            'exposure_time_ms': 50.0,
            'gain_db': 0.0,
            'illumination_ma': 100.0,
            'z_pos_um': 1000.0,
            'plate_pos_mm': {'x': 10.0, 'y': 20.0},
            'datetime': '2026:06:18 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': 'A1',
            'significant_bits': 8,
        },
        ome=False,
        color='Green',
        significant_bits=8,
        save_encoding='right_aligned',
    )


@pytest.fixture
def counter(monkeypatch):
    post = PostProcessing()

    def counted(image, settings, significant_bits):
        return None, {
            'summary': {'num_regions': 3, 'total_object_area': 1.0, 'total_object_intensity': 2.0}
        }

    monkeypatch.setattr(post, 'preview_cell_count', counted)
    return post


def _old_results(folder):
    (folder / 'results.csv').write_text(OLD_RESULTS)


def test_a_counted_folder_answers_with_its_results(counter, tmp_path):
    _write_tiff(tmp_path / 'a.tif')
    _write_tiff(tmp_path / 'b.tif')
    seen = []

    result = counter.apply_cell_count_to_folder(
        str(tmp_path), {}, on_progress=lambda percent, text: seen.append(percent)
    )

    assert result['counted'] == 2
    assert result['results_path'] == os.path.join(str(tmp_path), 'results.csv')
    assert 'a.tif' in (tmp_path / 'results.csv').read_text()
    assert seen[-1] == 100


def test_an_empty_folder_is_refused_and_keeps_the_previous_results(counter, tmp_path):
    _old_results(tmp_path)
    with pytest.raises(PostProcessingRefusedError):
        counter.apply_cell_count_to_folder(str(tmp_path), {})
    assert (tmp_path / 'results.csv').read_text() == OLD_RESULTS


def test_a_folder_of_unreadable_images_is_refused_and_keeps_the_previous_results(counter, tmp_path):
    _old_results(tmp_path)
    (tmp_path / 'broken.tif').write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingRefusedError):
        counter.apply_cell_count_to_folder(str(tmp_path), {})
    assert (tmp_path / 'results.csv').read_text() == OLD_RESULTS


def test_some_unreadable_images_make_the_count_incomplete_with_its_results(counter, tmp_path):
    _write_tiff(tmp_path / 'good.tif')
    (tmp_path / 'broken.tif').write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingFailedError) as raised:
        counter.apply_cell_count_to_folder(str(tmp_path), {})
    assert raised.value.produced_paths == (os.path.join(str(tmp_path), 'results.csv'),)
    assert 'good.tif' in (tmp_path / 'results.csv').read_text()


def test_a_failed_results_write_raises_and_keeps_the_previous_results(
    counter, tmp_path, monkeypatch
):
    _old_results(tmp_path)
    _write_tiff(tmp_path / 'a.tif')

    def refuse(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(os, 'replace', refuse)
    with pytest.raises(PostProcessingFailedError):
        counter.apply_cell_count_to_folder(str(tmp_path), {})
    assert (tmp_path / 'results.csv').read_text() == OLD_RESULTS
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.tif', 'results.csv']


def test_an_enhance_of_a_folder_with_no_images_is_refused(tmp_path):
    with pytest.raises(PostProcessingRefusedError):
        QuickEnhancer().export_folder(tmp_path, QuickEnhanceSettings())


def test_an_enhance_that_skips_an_image_is_incomplete_with_what_it_saved(tmp_path):
    _write_tiff(tmp_path / 'good.tif')
    (tmp_path / 'broken.tif').write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingFailedError) as raised:
        QuickEnhancer().export_folder(tmp_path, QuickEnhanceSettings())
    assert len(raised.value.produced_paths) == 1
    assert 'good_enhanced' in raised.value.produced_paths[0]
