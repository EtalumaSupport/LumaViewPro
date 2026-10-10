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
from modules.post_processing import PostProcessing, default_cell_count_method
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

    def counted(image, settings, significant_bits, *, pixels_per_um, name):
        return None, {
            'summary': {
                'num_regions': 3,
                'total_object_area': 1.0,
                'area_unit': 'px2',
                'total_object_intensity': 2.0,
            }
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
        str(tmp_path),
        default_cell_count_method(),
        on_progress=lambda percent, text: seen.append(percent),
    )

    assert result['counted'] == 2
    assert result['results_path'] == os.path.join(str(tmp_path), 'results.csv')
    assert 'a.tif' in (tmp_path / 'results.csv').read_text()
    assert seen[-1] == 100


def test_an_empty_folder_is_refused_and_keeps_the_previous_results(counter, tmp_path):
    _old_results(tmp_path)
    with pytest.raises(PostProcessingRefusedError):
        counter.apply_cell_count_to_folder(str(tmp_path), default_cell_count_method())
    assert (tmp_path / 'results.csv').read_text() == OLD_RESULTS


def test_a_folder_of_unreadable_images_is_refused_and_keeps_the_previous_results(counter, tmp_path):
    _old_results(tmp_path)
    (tmp_path / 'broken.tif').write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingRefusedError):
        counter.apply_cell_count_to_folder(str(tmp_path), default_cell_count_method())
    assert (tmp_path / 'results.csv').read_text() == OLD_RESULTS


def test_some_unreadable_images_make_the_count_incomplete_with_its_results(counter, tmp_path):
    _write_tiff(tmp_path / 'good.tif')
    (tmp_path / 'broken.tif').write_bytes(b'not a tiff')
    with pytest.raises(PostProcessingFailedError) as raised:
        counter.apply_cell_count_to_folder(str(tmp_path), default_cell_count_method())
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
        counter.apply_cell_count_to_folder(str(tmp_path), default_cell_count_method())
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


def test_an_enhance_of_one_image_answers_with_its_folder_progress_and_display(tmp_path):
    from modules.post_processing_api import PostProcessingAPI

    _write_tiff(tmp_path / 'one.tif')
    texts = []
    shown = []

    result = PostProcessingAPI._enhance(
        tmp_path / 'one.tif',
        lambda percent, text: texts.append((percent, text)),
        lambda image, significant_bits: shown.append(significant_bits),
    )

    assert result.output_folder == tmp_path
    assert result.message == 'Enhance complete.'
    assert texts[-1] == (100, 'Image 1 of 1')
    assert shown == [8]


@pytest.mark.parametrize('raised', ['not a tiff', 'cv2'])
def test_an_enhance_of_one_image_that_fails_is_typed(tmp_path, monkeypatch, raised):
    import cv2

    from modules.post_processing_api import PostProcessingAPI

    source = tmp_path / 'broken.tif'
    source.write_bytes(b'not a tiff')
    if raised == 'cv2':

        def fail(self, *args, **kwargs):
            raise cv2.error('codec')

        monkeypatch.setattr(QuickEnhancer, 'export_file', fail)

    with pytest.raises(PostProcessingFailedError) as failure:
        PostProcessingAPI._enhance(source, None, None)
    assert failure.value.produced_paths == ()


class _ScriptedBuild:
    """A folder build whose groups all succeed, driven through the base loop."""

    @staticmethod
    def make(name):
        import pandas as pd

        from modules.common_utils import PostFunction
        from modules.protocol_post_processing_result import PostProcResult
        from modules.protocol_post_processor import ProtocolPostProcessor

        def _groups(self, df):
            return [(key, group) for key, group in df.groupby('GroupKey')]

        cls = type(
            name,
            (ProtocolPostProcessor,),
            {
                '_get_groups': _groups,
                '_generate_filename': lambda self, df, **kw: df.iloc[0]['OutName'],
                '_filter_ignored_types': lambda self, df: df,
                '_group_algorithm': lambda self, path, df, **kw: PostProcResult.ok(
                    significant_bits=8
                ),
                '_add_record': lambda self, *a, **kw: None,
            },
        )
        build = cls(post_function=PostFunction.STITCHED, has_turret=False)
        rows = []
        for g in range(2):
            for i in range(2):
                row = {'Filepath': f'g{g}_f{i}.tiff', 'GroupKey': g, 'OutName': f'out{g}.tiff'}
                row.update(dict.fromkeys(PostFunction.list_values(), False))
                rows.append(row)
        return build, pd.DataFrame(rows)


def _drive_scripted(name, tmp_path, monkeypatch, **kwargs):
    from unittest.mock import MagicMock

    build, df = _ScriptedBuild.make(name)
    record = MagicMock()
    record.file_exists_in_records.return_value = False
    monkeypatch.setattr(
        build._post_processing_helper,
        'load_folder',
        lambda **kw: {
            'status': True,
            'images_df': df,
            'root_path': tmp_path,
            'protocol_post_record': record,
            'protocol': None,
        },
    )
    seen = []
    build.load_folder(
        path=tmp_path,
        tiling_configs_file_loc=tmp_path / 'tiling.json',
        on_progress=lambda percent, text: seen.append((percent, text)),
        **kwargs,
    )
    return seen


def test_a_folder_build_reports_each_group_then_done(tmp_path, monkeypatch):
    seen = _drive_scripted('ZProjector', tmp_path, monkeypatch)
    assert [percent for percent, _ in seen] == [50.0, 100.0, 100]
    assert all(text is None for _, text in seen)


def test_a_stitch_says_which_group_it_is_on_and_what_is_left(tmp_path, monkeypatch):
    seen = _drive_scripted('Stitcher', tmp_path, monkeypatch, stitching_mode='quality')
    # After the first group lands, the line names the group now starting.
    assert 'group 2/2' in seen[0][1]
    assert 'Estimated remaining time' in seen[0][1]


def test_a_stitchs_time_estimate_counts_the_group_now_starting(tmp_path, monkeypatch):
    import itertools
    from types import SimpleNamespace

    import modules.protocol_post_processor as protocol_post_processor

    # Every group takes 10 s by the build's own clock: the module's own name
    # for time, so no other thread reads these ticks.
    ticks = itertools.count(step=10.0)
    monkeypatch.setattr(
        protocol_post_processor, 'time', SimpleNamespace(perf_counter=lambda: next(ticks))
    )

    seen = _drive_scripted('Stitcher', tmp_path, monkeypatch, stitching_mode='quality')

    # One of two groups done, so one 10 s group is still to run.
    assert 'about 10 seconds' in seen[0][1]
    # Both done: nothing left.
    assert 'about 0 seconds' in seen[1][1]


def test_a_protocol_video_hands_its_progress_to_the_encoder(tmp_path, monkeypatch):
    import pandas as pd

    from modules.video_builder import VideoBuilder

    seen = {}

    def encode(self, **kwargs):
        seen.update(kwargs)
        return {'status': True, 'error': None, 'metadata': {}, 'significant_bits': 8}

    monkeypatch.setattr(VideoBuilder, '_create_video', encode)

    def progress(percent, text):
        pass

    df = pd.DataFrame(
        {'Filepath': ['a.tiff'], 'Scan Count': [0], 'Timestamp': [''], 'Color': [None]}
    )
    try:
        VideoBuilder(has_turret=False)._group_algorithm(
            path=tmp_path,
            df=df,
            frames_per_sec=5,
            enable_timestamp_overlay=False,
            output_file_loc=tmp_path / 'out.mp4',
            on_progress=progress,
            total_groups=1,
            current_group=1,
        )
    except Exception:
        # The result shape is the encoder's business; this pins only what
        # the encoder was handed.
        pass
    assert seen['on_progress'] is progress
