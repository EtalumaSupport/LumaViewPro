# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build leaves a record that it ran, when, and how it ended.

A cell count writes ``results.csv`` into a run's folder, and on the bench a
build of a whole folder ran for a minute with no line in the log: when it
started, whether it overlapped a run and how it ended could not be read
back. Every build passes through one place on its lane, so that place says
when the build starts and how it ends. A build that raises is reported by
the reporter where its flight stops; the end line names only the outcome's
type, so the failure is not told twice.
"""

from __future__ import annotations

import pytest

import modules.post_processing_api as post_processing_api
from modules.exceptions import PostProcessingRefusedError
from tests.log_capture import capture_module_log, messages


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def test_a_build_logs_its_start_and_its_result(session, monkeypatch, tmp_path):
    from modules.post_processing import PostProcessing

    monkeypatch.setattr(
        PostProcessing,
        'apply_cell_count_to_folder',
        lambda self, path, settings, on_progress=None: {'message': 'Counted cells in 3 image(s).'},
    )
    records = capture_module_log(monkeypatch, post_processing_api)

    session.post_processing.count_cells(tmp_path, method={})

    lines = messages(records)
    assert any('count_cells' in m and 'started' in m and str(tmp_path) in m for m in lines), lines
    ended = [m for m in lines if 'count_cells' in m and 'ended' in m]
    assert len(ended) == 1, lines
    assert 'Counted cells in 3 image(s).' in ended[0]
    assert ' s' in ended[0]


def test_a_refused_build_logs_its_end_by_the_outcomes_type_only(session, monkeypatch, tmp_path):
    records = capture_module_log(monkeypatch, post_processing_api)

    with pytest.raises(PostProcessingRefusedError) as raised:
        session.post_processing.stitch(tmp_path)

    lines = messages(records)
    assert any('stitch' in m and 'started' in m and str(tmp_path) in m for m in lines), lines
    ended = [m for m in lines if 'stitch' in m and 'ended' in m]
    assert len(ended) == 1, lines
    assert 'PostProcessingRefusedError' in ended[0]
    assert str(raised.value) not in ended[0]
    assert all(r.levelname == 'INFO' for r in records)


def test_a_build_keeps_the_slow_task_budget_it_declares(session, monkeypatch, tmp_path):
    # The lines are written by a wrapper around the build, and the lane reads
    # a build's slow-task budget off the callable it is handed: a budget a
    # build declares must survive the wrapper.
    from modules.post_processing import PostProcessing
    from modules.sequential_io_executor import slow_task_budget

    @slow_task_budget(600.0)
    def count(self, path, settings, on_progress=None):
        return {'message': 'counted'}

    monkeypatch.setattr(PostProcessing, 'apply_cell_count_to_folder', count)
    lane = session.post_processing.lane
    seen = []
    real_call = lane.call

    def call(task, *args, **kwargs):
        seen.append(task.declared_slow_task_budget())
        return real_call(task, *args, **kwargs)

    monkeypatch.setattr(lane, 'call', call)

    session.post_processing.count_cells(tmp_path, method={})

    assert seen == [600.0]
