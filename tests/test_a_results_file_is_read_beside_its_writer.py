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

Each row's time was the file system's (``getctime``), so a copied folder
read as taken at the copy; it is now the time the image states, with its UTC
offset, read back as each cell states it.
"""

from __future__ import annotations

import csv
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modules import image_utils
from modules.cell_count_method import CellCountMethod
import ui.post_processing as panel_module
from modules.exceptions import PostProcessingRefusedError
from modules.layer_record import release_color
from modules.lumascope_api.frame_record import TimestampSource
from modules.post_processing import (
    PostProcessing,
    read_cell_count_results,
    results_axes,
)
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED1


_PST = datetime.timezone(datetime.timedelta(hours=-8))

# When each capture states it was taken: months before its file was made.
_TAKEN = {
    'image_0.tif': datetime.datetime(2026, 3, 1, 9, 0, 0, 250000, tzinfo=_PST),
    'image_1.tif': datetime.datetime(2026, 3, 1, 10, 30, 0, tzinfo=_PST),
}


def _counted_folder(tmp_path):
    """A folder the app's own count has written ``results.csv`` into: two
    captures that state when they were taken, and a PNG, which states nothing."""
    from PIL import Image

    for i, (name, taken) in enumerate(_TAKEN.items()):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[10 + i : 30, 10:30] = 200
        image_utils.write_tiff(
            data=image,
            file_loc=tmp_path / name,
            metadata={
                'objective': {},
                'pixel_size_um': None,
                'channel': LED1,
                'significant_bits': 8,
                'datetime': taken,
                'timestamp_iso': taken.isoformat(timespec='microseconds'),
                'timestamp_source': TimestampSource.CAMERA_FRAME_START.value,
            },
            ome=False,
            color=release_color(LED1),
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )
    bare = np.zeros((64, 64), dtype=np.uint8)
    bare[10:30, 10:30] = 200
    Image.fromarray(bare).save(tmp_path / 'bare.png')
    PostProcessing().apply_cell_count_to_folder(path=str(tmp_path), method=CellCountMethod())
    return tmp_path / 'results.csv'


def test_each_row_states_when_its_image_was_taken_not_when_its_file_was_made(tmp_path):
    """The file system's time is the copy's on a copied folder; a time-lapse
    counted there would read as taken all at once."""
    with open(_counted_folder(tmp_path), newline='', encoding='utf-8') as f:
        times = {row['file']: row['time'] for row in csv.DictReader(f)}
    assert times == {
        **{name: taken.isoformat() for name, taken in _TAKEN.items()},
        'bare.png': '',
    }


def test_the_apps_own_results_file_reads_back_with_its_time_as_a_datetime(tmp_path):
    table = read_cell_count_results(_counted_folder(tmp_path)).set_index('file')
    assert table['time'].dtype.kind == 'M'
    assert {name: table.at[name, 'time'] for name in _TAKEN} == _TAKEN
    assert pd.isna(table.at['bare.png', 'time'])
    x_axes, y_axes = results_axes(table.reset_index())
    assert 'time' in x_axes
    assert 'time' not in y_axes
    assert 'num_cells' in y_axes


def test_a_run_across_a_daylight_saving_change_reads_in_utc_keeping_every_instant(tmp_path):
    """The repeated hour's two 01:30s are an hour apart; read on either one
    offset they would be one instant."""
    path = tmp_path / 'results.csv'
    path.write_text(
        'file,time,num_cells\n'
        'a.tif,2026-11-01T01:30:00-07:00,3\n'
        'b.tif,2026-11-01T01:30:00-08:00,4\n'
    )
    table = read_cell_count_results(path)
    assert str(table['time'].dt.tz) == 'UTC'
    assert (table['time'][1] - table['time'][0]).total_seconds() == 3600
    assert 'time' in results_axes(table)[0]


def test_times_with_and_without_an_offset_are_no_time_axis(tmp_path):
    """A time that states no offset is never given one, so it cannot be set
    against one that does (image_utils.comparable_times); the rest of the
    file still graphs."""
    path = tmp_path / 'results.csv'
    path.write_text(
        'file,time,num_cells\na.tif,2026-03-01T09:00:00,3\nb.tif,2026-03-01T10:00:00-08:00,4\n'
    )
    assert results_axes(read_cell_count_results(path)) == (['num_cells'], ['num_cells'])


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


def test_a_time_column_of_numbers_is_plotted_as_numbers(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('time,num_cells\n0,3\n60,4\n')
    table = read_cell_count_results(path)
    assert table['time'].tolist() == [0, 60]
    assert 'time' in results_axes(table)[1]


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


@pytest.fixture
def session(monkeypatch, tmp_path):
    """A simulated session the panel asks, as the app's context holds it."""
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    monkeypatch.setattr(panel_module._app_ctx, 'ctx', SimpleNamespace(session=s))
    yield s
    s.shutdown()


def _graph() -> SimpleNamespace:
    graph = SimpleNamespace(
        _results='first.csv',
        _x_axes=[],
        _y_axes=[],
        _trendline_kinds=('Linear',),
        selected_x_axis='num_cells',
        selected_y_axis='total_object_area (um2)',
        _trendline_kind='Linear',
        _figure=object(),
        shown=0,
    )
    graph._load_source = lambda file: panel_module.GraphingControls._load_source(graph, file)

    def show():
        graph.shown += 1

    graph._redraw_graph = show
    return graph


def test_loading_a_file_starts_the_axis_choices_and_trendline_over(session, tmp_path, boundary):
    path = tmp_path / 'second.csv'
    path.write_text('well,count\n1,3\n2,4\n')
    graph = _graph()
    panel_module.GraphingControls.set_graphing_source(graph, str(path))
    assert boundary == []
    assert graph._results == str(path)
    assert (graph.selected_x_axis, graph.selected_y_axis) == (None, None)
    assert (graph._trendline_kind, graph._figure) == ('None', None)
    assert graph._x_axes == ['well', 'count']
    assert graph.shown == 1


def test_a_file_that_cannot_be_graphed_is_reported_and_the_graph_keeps_its_data(
    session, tmp_path, boundary
):
    path = tmp_path / 'image.tif'
    path.write_bytes(b'II*\x00\x08\x00\x00\x00\xff\xfe\x00\x81')
    graph = _graph()
    panel_module.GraphingControls.set_graphing_source(graph, str(path))
    [(label, refusal)] = boundary
    assert label == 'LOAD_GRAPHING_DATA'
    assert refusal.reason == 'results_unreadable'
    assert (graph._results, graph.selected_x_axis) == ('first.csv', 'num_cells')
    assert graph.shown == 1
