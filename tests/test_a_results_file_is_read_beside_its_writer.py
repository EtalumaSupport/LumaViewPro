# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell-count results file is read back by the module that writes it.

The graph read ``results.csv`` itself: it parsed the time column with
``'%c'`` and decided a column was a time by its NAME, so a numeric column
named ``elapsed_time`` crashed the trendline on ``.dt``; it kept the first
file's axis choices when a second file was loaded, so a valid second file was
plotted against columns it did not have; and an unreadable file was logged
and never shown. The reader now sits beside the writer and shares its time
format, the axes are decided by each column's type, a load starts the choices
over, and a file that cannot be graphed is refused naming it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ui.post_processing as panel_module
from modules.exceptions import PostProcessingRefusedError
from modules.post_processing import (
    PostProcessing,
    default_cell_count_method,
    read_cell_count_results,
    results_axes,
)


def _counted_folder(tmp_path):
    """A folder the app's own count has written ``results.csv`` into."""
    import cv2

    for i in range(2):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[10 + i : 30, 10:30] = 200
        cv2.imwrite(str(tmp_path / f'image_{i}.tif'), image)
    PostProcessing().apply_cell_count_to_folder(
        path=str(tmp_path), settings=default_cell_count_method()
    )
    return tmp_path / 'results.csv'


def test_the_apps_own_results_file_reads_back_with_its_time_as_a_datetime(tmp_path):
    table = read_cell_count_results(_counted_folder(tmp_path))
    assert table['time'].dtype.kind == 'M'
    x_axes, y_axes = results_axes(table)
    assert 'time' in x_axes
    assert 'time' not in y_axes
    assert 'num_cells' in y_axes


def test_a_file_written_by_ctime_before_the_shared_format_still_reads(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text(
        'file,time,num_cells\na.tif,Thu Oct  1 17:13:20 2026,3\nb.tif,Mon Sep 21 07:13:20 2026,4\n'
    )
    assert read_cell_count_results(path)['time'].dt.day.tolist() == [1, 21]


def test_a_number_column_named_for_time_is_a_number(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('elapsed_time,num_cells\n0,3\n60,4\n')
    table = read_cell_count_results(path)
    assert table['elapsed_time'].dtype.kind in 'iuf'
    assert 'elapsed_time' in results_axes(table)[1]


def test_text_columns_are_no_axis(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('file,num_cells\na.tif,3\nb.tif,4\n')
    assert results_axes(read_cell_count_results(path)) == (['num_cells'], ['num_cells'])


@pytest.mark.parametrize(
    ('content', 'words'),
    [
        (None, 'could not be read'),
        (b'II*\x00\x08\x00\x00\x00\xff\xfe\x00\x81', 'is not a CSV file'),
        (b'file,time,num_cells\na.tif,yesterday,3\n', 'time column'),
        (b'file,label\na.tif,round\n', 'no column of numbers'),
    ],
    ids=['missing', 'not-a-csv', 'unwritten-time', 'nothing-to-plot'],
)
def test_a_file_that_cannot_be_graphed_is_refused_naming_it(tmp_path, content, words):
    path = tmp_path / 'results.csv'
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(PostProcessingRefusedError) as raised:
        read_cell_count_results(path)
    assert raised.value.operation == 'Graphing'
    assert raised.value.reason == 'results_unreadable'
    assert str(path) in str(raised.value)
    assert words in str(raised.value)


@pytest.fixture
def boundary(monkeypatch):
    """The GUI boundary, inline: run the call, keep what it raised, then redraw."""
    outcomes = []

    def reported(call, redraw, label):
        try:
            call()
        except Exception as e:
            outcomes.append((label, e))
        if redraw is not None:
            redraw()

    monkeypatch.setattr(panel_module, 'run_reported', reported)
    return outcomes


def _graph() -> SimpleNamespace:
    graph = SimpleNamespace(
        graph_df=None,
        _x_axes=[],
        _y_axes=[],
        selected_x_axis='num_cells',
        selected_y_axis='total_object_area (um2)',
        _trendline_kind='Linear',
        _trendline=object(),
        shown=0,
    )
    graph._load_source = lambda file: panel_module.GraphingControls._load_source(graph, file)

    def show():
        graph.shown += 1

    graph._redraw_graph = show
    return graph


def test_loading_a_file_starts_the_axis_choices_and_trendline_over(tmp_path, boundary):
    path = tmp_path / 'second.csv'
    path.write_text('well,count\n1,3\n2,4\n')
    graph = _graph()
    panel_module.GraphingControls.set_graphing_source(graph, str(path))
    assert boundary == []
    assert (graph.selected_x_axis, graph.selected_y_axis) == (None, None)
    assert (graph._trendline_kind, graph._trendline) == ('None', None)
    assert graph._x_axes == ['well', 'count']
    assert graph.shown == 1


def test_a_file_that_cannot_be_graphed_is_reported_and_the_graph_keeps_its_data(tmp_path, boundary):
    path = tmp_path / 'image.tif'
    path.write_bytes(b'II*\x00\x08\x00\x00\x00\xff\xfe\x00\x81')
    graph = _graph()
    panel_module.GraphingControls.set_graphing_source(graph, str(path))
    [(label, refusal)] = boundary
    assert label == 'LOAD_GRAPHING_DATA'
    assert isinstance(refusal, PostProcessingRefusedError)
    assert graph.selected_x_axis == 'num_cells'
    assert graph.shown == 1
