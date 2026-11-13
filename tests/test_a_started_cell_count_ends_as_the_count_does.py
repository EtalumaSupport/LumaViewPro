# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell count started without waiting ends as the waiting count does.

A count's time grows with its folder, so a caller that must not block, a
REST client above all, starts it: ``start_count_cells`` puts the same count
on the post-processing lane and returns an ``Operation`` done with what
``count_cells`` would have returned or raised. An unknown channel is refused
at the call, before anything is queued; what the count finds on the lane (a
folder that is not there) ends the operation.
"""

from __future__ import annotations

import pytest

from _lumascope.modules.cell_count_method import CellCountMethod
from _lumascope.modules.exceptions import ArgumentRefusedError, LiveFolderPathRefusedError
from tests.count_runs import (
    copy_run,
    count_scope,  # noqa: F401 -- the module's runs, a fixture
    results_rows,
    session,  # noqa: F401 -- a fixture
)


def test_a_started_count_ends_with_the_answer_the_waiting_count_gives(
    count_scope, session, tmp_path
):
    started_in = copy_run(count_scope, 'scan', tmp_path / 'started')
    waited_in = copy_run(count_scope, 'scan', tmp_path / 'waited')
    post = session.post_processing

    started = post.start_count_cells(started_in, method=CellCountMethod()).result(timeout=60)
    waited = post.count_cells(waited_in, method=CellCountMethod())

    assert started.results_path == started_in / 'results.csv'
    assert started.counted == waited.counted > 0
    assert len(results_rows(started_in)) == len(results_rows(waited_in))


def test_a_folder_that_is_not_there_ends_the_started_count(session, tmp_path):
    started = session.post_processing.start_count_cells(tmp_path / 'gone', method=CellCountMethod())

    with pytest.raises(LiveFolderPathRefusedError) as refused:
        started.result(timeout=60)
    assert refused.value.reason == 'path_not_found'


def test_an_unknown_channel_is_refused_at_the_call(session, tmp_path):
    with pytest.raises(ArgumentRefusedError) as refused:
        session.post_processing.start_count_cells(
            tmp_path, method=CellCountMethod(), channel='Blue-ish'
        )
    assert refused.value.reason == 'channel_unknown'
