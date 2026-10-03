# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A trendline the data cannot take is refused in words, once per pick.

The graph fitted its trendlines inline, each in a try that logged the
exception and set the spinner back to None. The person saw the spinner reset
with no reason given. An exponential or power fit over a zero, or a NaN in
either column, gave a curve of NaN or an arbitrary one. A pick with the
trendline already on fitted twice, because the axis handlers called the fit
again. The fit now lives in ``modules/graph_analysis.py``. It refuses, naming
the values and how many, and the widget fits once per gesture and draws what
it stored.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ui.post_processing as panel_module
from modules.exceptions import PostProcessingRefusedError
from modules.graph_analysis import fit_trendline, trendline_kinds


def _series(name, values):
    return pd.Series(values, name=name, dtype=float)


X = _series('num_cells', [1, 2, 3, 4])
Y = _series('area', [2, 4, 6, 8])


@pytest.mark.parametrize(
    ('kind', 'x', 'y', 'words'),
    [
        ('Exponential', X, _series('area', [0, 4, 6, 8]), '1 of the 4 area values are 0 or below'),
        ('Power', X, _series('area', [0, 4, 6, 8]), '1 of the 4 area values are 0 or below'),
        (
            'Power',
            _series('num_cells', [0, 2, 3, 4]),
            Y,
            '1 of the 4 num_cells values are 0 or below',
        ),
        (
            'Logarithmic',
            _series('num_cells', [0, 2, 3, 4]),
            Y,
            '1 of the 4 num_cells values are 0 or below',
        ),
        ('Quadratic', _series('num_cells', [1, 1, 2, 2]), Y, 'num_cells has 2 distinct value(s)'),
        ('Linear', _series('num_cells', [3, 3, 3, 3]), Y, 'num_cells has 1 distinct value(s)'),
    ],
)
def test_values_the_curve_cannot_take_are_refused_naming_them(kind, x, y, words):
    with pytest.raises(PostProcessingRefusedError) as raised:
        fit_trendline(kind, x, y)
    assert raised.value.operation == 'Trendline'
    assert raised.value.reason == 'fit_impossible'
    assert words in str(raised.value)


@pytest.mark.parametrize('kind', ['Linear', 'Quadratic', 'Exponential', 'Power', 'Logarithmic'])
def test_a_missing_value_is_refused_for_every_kind(kind):
    with pytest.raises(PostProcessingRefusedError, match='1 of the 4 area values are missing'):
        fit_trendline(kind, X, _series('area', [2, np.nan, 6, 8]))


def test_a_time_x_takes_only_the_kinds_that_use_x_as_it_is():
    times = pd.Series(
        pd.to_datetime(['2026-10-02 10:00', '2026-10-02 10:01', '2026-10-02 10:02']), name='time'
    )
    y = _series('area', [1, 2, 3])
    assert trendline_kinds(times, y) == ('Linear', 'Quadratic', 'Exponential')
    assert trendline_kinds(X, Y) == ('Linear', 'Quadratic', 'Exponential', 'Power', 'Logarithmic')
    with pytest.raises(
        PostProcessingRefusedError, match='these columns take Linear, Quadratic, Exponential'
    ):
        fit_trendline('Power', times, y)


def test_a_time_x_is_fitted_in_seconds_from_its_first_value_and_drawn_as_times():
    times = pd.Series(
        pd.to_datetime(['2026-10-02 10:02', '2026-10-02 10:00', '2026-10-02 10:01']), name='time'
    )
    curve = fit_trendline('Linear', times, _series('area', [120, 0, 60]))
    assert curve.x.tolist() == sorted(times.tolist())
    np.testing.assert_allclose(curve.y, [0, 60, 120], atol=1e-6)


@pytest.mark.parametrize(
    ('kind', 'y'),
    [
        ('Linear', [2, 4, 6, 8]),
        ('Quadratic', [1, 4, 9, 16]),
        ('Exponential', [np.e, np.e**2, np.e**3, np.e**4]),
        ('Power', [1, 8, 27, 64]),
        ('Logarithmic', list(np.log([1, 2, 3, 4]))),
    ],
)
def test_each_kind_draws_its_curve_through_data_of_its_shape(kind, y):
    curve = fit_trendline(kind, X, _series('area', y))
    assert curve.kind == kind
    np.testing.assert_allclose(curve.y, y, rtol=1e-6, atol=1e-9)


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
def records(monkeypatch):
    seen = []
    monkeypatch.setattr(
        panel_module.gui_logger, 'select', lambda name, value: seen.append((name, value))
    )
    return seen


def _graph(monkeypatch):
    """The widget's state and handlers, with spinners that call back on a change as Kivy's do."""
    cls = panel_module.GraphingControls
    graph = SimpleNamespace(
        graph_df=pd.DataFrame(
            {'num_cells': [1.0, 2.0, 3.0], 'area': [0.0, 4.0, 6.0], 'perimeter': [1.0, 2.0, 4.0]}
        ),
        selected_x_axis='num_cells',
        selected_y_axis='area',
        _trendline_kind='None',
        _trendline=None,
        fits=0,
        redraws=0,
    )
    handlers = {
        'graphing_x_axis_spinner': lambda: cls.set_x_axis(graph),
        'graphing_y_axis_spinner': lambda: cls.set_y_axis(graph),
        'trendline_spinner': lambda: cls.update_trendline(graph),
    }

    class Spinner:
        def __init__(self, key, text):
            self.key, self._text = key, text

        @property
        def text(self):
            return self._text

        @text.setter
        def text(self, value):
            if value != self._text:
                self._text = value
                handlers[self.key]()

    class Ids(SimpleNamespace):
        def __getitem__(self, key):
            return getattr(self, key)

    graph.ids = Ids(
        graphing_x_axis_spinner=Spinner('graphing_x_axis_spinner', 'num_cells'),
        graphing_y_axis_spinner=Spinner('graphing_y_axis_spinner', 'area'),
        trendline_spinner=Spinner('trendline_spinner', 'None'),
        x_axis_label_input=SimpleNamespace(text=''),
        y_axis_label_input=SimpleNamespace(text=''),
    )

    def fit():
        graph.fits += 1
        cls._fit_trendline(graph)

    def redraw():
        graph.redraws += 1
        graph.ids.graphing_x_axis_spinner.text = graph.selected_x_axis or 'X-Axis'
        graph.ids.graphing_y_axis_spinner.text = graph.selected_y_axis or 'Y-Axis'
        graph.ids.trendline_spinner.text = graph._trendline_kind

    graph._fit_trendline = fit
    graph._redraw_graph = redraw
    return graph


def test_a_pick_the_data_cannot_take_is_refused_once_and_the_spinner_reads_none(
    monkeypatch, boundary, records
):
    graph = _graph(monkeypatch)
    graph.ids.trendline_spinner.text = 'Exponential'
    assert [label for label, _ in boundary] == ['TRENDLINE']
    assert isinstance(boundary[0][1], PostProcessingRefusedError)
    assert graph.fits == 1
    assert graph.ids.trendline_spinner.text == 'None'
    assert records == [('TRENDLINE', 'Exponential')]


def test_an_axis_change_after_a_refused_pick_asks_for_no_fit_again(monkeypatch, boundary, records):
    graph = _graph(monkeypatch)
    graph.ids.trendline_spinner.text = 'Exponential'
    graph.ids.graphing_x_axis_spinner.text = 'perimeter'
    assert len(boundary) == 1
    assert graph._trendline is None
    assert records == [('TRENDLINE', 'Exponential'), ('GRAPHING_X_AXIS', 'perimeter')]


def test_a_pick_fits_once_and_an_axis_change_fits_once_more(monkeypatch, boundary, records):
    graph = _graph(monkeypatch)
    graph.ids.trendline_spinner.text = 'Linear'
    assert (graph.fits, boundary) == (1, [])
    assert graph._trendline.kind == 'Linear'
    graph.ids.graphing_y_axis_spinner.text = 'perimeter'
    assert graph.fits == 2
    assert graph._trendline.y.tolist() == pytest.approx(
        fit_trendline('Linear', graph.graph_df['num_cells'], graph.graph_df['perimeter']).y.tolist()
    )


def test_the_redraw_showing_the_stored_choice_is_not_a_choice(monkeypatch, boundary, records):
    graph = _graph(monkeypatch)
    panel_module.GraphingControls.set_x_axis(graph)
    panel_module.GraphingControls.set_y_axis(graph)
    panel_module.GraphingControls.update_trendline(graph)
    assert (records, graph.fits, graph.redraws) == ([], 0, 0)
