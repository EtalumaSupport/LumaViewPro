# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A graph of cell-count results is drawn and saved by the API, for every caller.

The Object Plotting popup once drew its graph with pyplot and saved it with
``plt.savefig``, so only the popup could make one and a name such as
``g.v2`` reached matplotlib, which raised with nothing to report it.
``graph_analysis.draw_graph`` now draws the figure, and
``session.post_processing`` answers the form's questions (``graph_options``),
draws it for an in-process GUI (``draw_graph``) and saves it (``graph``).
Each member's valid and bad calls, in Python and over REST, are walked in
``tests/test_every_enrolled_member_answers_as_its_contract_says.py``, and the
writer's name and folder rules in
``tests/test_a_path_that_cannot_be_used_is_refused_as_a_request.py``; this
file holds what the drawing shows and the words of an incomplete save.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.dates import ConciseDateFormatter

from modules import graph_analysis
from modules.exceptions import ArgumentRefusedError, PostProcessingRefusedError
from modules.graph_analysis import Trendline
from modules.post_processing import RESULTS_TIME_FORMAT
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            'time': pd.to_datetime(
                [
                    'Mon Oct 13 10:00:00 2025',
                    'Mon Oct 13 10:05:00 2025',
                    'Mon Oct 13 10:10:00 2025',
                ],
                format=RESULTS_TIME_FORMAT,
            ),
            'count': [3, 5, 8],
        }
    )


def test_the_graph_shows_its_points_labels_and_title(table):
    figure = graph_analysis.draw_graph(
        table, 'time', 'count', None, x_label='When', y_label='Cells', title='Growth'
    )

    (axes,) = figure.axes
    (points,) = axes.collections
    assert len(points.get_offsets()) == 3
    assert (axes.get_xlabel(), axes.get_ylabel(), axes.get_title()) == ('When', 'Cells', 'Growth')
    assert isinstance(axes.xaxis.get_major_formatter(), ConciseDateFormatter)


def test_a_trendline_is_drawn_as_a_red_dashed_line(table):
    trendline = graph_analysis.fit_trendline('Linear', table['time'], table['count'])

    figure = graph_analysis.draw_graph(
        table, 'time', 'count', trendline, x_label='', y_label='', title=''
    )

    (line,) = figure.axes[0].get_lines()
    assert (line.get_color(), line.get_linestyle()) == ('r', '--')
    assert list(line.get_ydata()) == list(trendline.y)


def test_no_axis_chosen_draws_the_empty_frame(table):
    figure = graph_analysis.draw_graph(
        table, None, None, None, x_label='X-Axis', y_label='Y-Axis', title=''
    )

    (axes,) = figure.axes
    assert all(len(c.get_offsets()) == 0 for c in axes.collections)
    assert axes.get_lines() == []


def test_the_figure_is_not_held_by_pyplot(table):
    before = plt.get_fignums()

    graph_analysis.draw_graph(table, 'time', 'count', None, x_label='', y_label='', title='')

    assert plt.get_fignums() == before


@pytest.fixture
def session(tmp_path):
    (tmp_path / 'results.csv').write_text('count,mean_area\n3,1.5\n5,2.5\n')
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.mark.parametrize(
    ('results', 'x', 'y'),
    [
        (None, 'count', 'mean_area'),
        ('results.csv', None, 'mean_area'),
        ('results.csv', 'count', None),
    ],
)
def test_a_save_without_a_file_and_both_axes_is_refused_in_words(session, tmp_path, results, x, y):
    given = None if results is None else tmp_path / results

    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.graph(given, tmp_path / 'g.png', x=x, y=y)

    assert refused.value.reason == 'graph_incomplete'
    assert 'choose a results file and an X and a Y axis first' in str(refused.value)
    assert not (tmp_path / 'g.png').exists()


def test_the_in_process_drawing_matches_the_saved_choices(session, tmp_path):
    figure = session.post_processing.draw_graph(
        tmp_path / 'results.csv', x='count', y='mean_area', trendline='Linear'
    )

    axes = figure.axes[0]
    assert (axes.get_xlabel(), axes.get_ylabel()) == ('count', 'mean_area')
    assert len(axes.get_lines()) == 1


def test_the_trendlines_offered_are_those_the_axes_given_take(session, tmp_path):
    """The panel offered all five kinds until both axes were chosen, so with a
    time X it offered Power and Logarithmic, which a time X cannot take. The
    session now answers for whichever axes are given."""
    results = tmp_path / 'timed.csv'
    results.write_text(
        'time,count,mean_area\n'
        'Mon Oct 13 10:00:00 2025,3,1.5\n'
        'Mon Oct 13 10:05:00 2025,5,2.5\n'
        'Mon Oct 13 10:10:00 2025,8,4.0\n'
    )
    every = ('Linear', 'Quadratic', 'Exponential', 'Power', 'Logarithmic')
    by_time = ('Linear', 'Quadratic', 'Exponential')
    options = session.post_processing.graph_options

    assert options(results).trendlines == every
    assert options(results, y='mean_area').trendlines == every
    assert options(results, x='time').trendlines == by_time
    assert options(results, x='time', y='count').trendlines == by_time
    assert options(results, x='count').trendlines == every


def test_the_trendlines_offered_are_members_and_a_kind_outside_the_set_is_refused(
    session, tmp_path
):
    results = tmp_path / 'results.csv'
    post = session.post_processing

    assert all(type(kind) is Trendline for kind in post.graph_options(results).trendlines)
    with pytest.raises(ArgumentRefusedError) as refused:
        post.draw_graph(results, x='count', y='mean_area', trendline='Sideways')
    assert (refused.value.reason, refused.value.argument) == ('not_offered', 'trendline')
    assert refused.value.offered == tuple(Trendline)
