# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each Post Processing button asks the session's member, with what the panel holds.

The buttons used to build the modules themselves and choose the tiling
file, the turret, the lane and the composite's settings. They now hand the
panel's own values to ``session.post_processing`` and show its answer. Each
button's body is run unbound on a stand-in for its panel, without the popup
thread.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pytest

from modules.post_processing_api import BuildResult, CellCountResult, EnhanceResult

import ui.post_processing as post_processing


@pytest.fixture
def session(monkeypatch):
    """A stand-in session whose members answer, and an inline GUI boundary."""
    ctx = MagicMock()
    built = BuildResult(
        message='done',
        new_count=0,
        output_root=pathlib.Path('folder'),
        artifact_paths=(),
        degraded_outputs=(),
    )
    for member in ('stitch', 'zproject', 'composite', 'video'):
        getattr(ctx.session.post_processing, member).return_value = built
    ctx.session.post_processing.enhance.return_value = EnhanceResult(
        message='done', output_folder=pathlib.Path('folder'), created=()
    )
    ctx.session.post_processing.count_cells.return_value = CellCountResult(
        message='done', results_path=pathlib.Path('folder/results.csv'), counted=0
    )
    monkeypatch.setattr(post_processing._app_ctx, 'ctx', ctx)

    def inline(call, redraw, label, **kwargs):
        assert kwargs['lane'] is ctx.session.post_processing.lane
        call()
        redraw()

    monkeypatch.setattr(post_processing, 'submit_reported', inline)
    return ctx.session


def _body(cls, name):
    return getattr(cls, name).__wrapped__


def test_stitch_asks_for_the_mode_the_panel_shows(session):
    panel = SimpleNamespace(
        stitching_mode='Fast Preview', _MODE_VALUES=post_processing.StitchControls._MODE_VALUES
    )
    _body(post_processing.StitchControls, 'run_stitcher')(panel, MagicMock(), '/data/run')
    session.post_processing.stitch.assert_called_once_with(
        '/data/run', mode='fast_preview', on_progress=ANY
    )


def test_zprojection_asks_for_the_selected_method(session):
    panel = SimpleNamespace(ids={'zprojection_method_spinner': SimpleNamespace(text='Max')})
    _body(post_processing.ZProjectionControls, 'run_zprojection')(panel, MagicMock(), '/data/z')
    session.post_processing.zproject.assert_called_once_with(
        '/data/z', method='Max', on_progress=ANY
    )


def test_composite_asks_with_the_folder_alone(session):
    _body(post_processing.CompositeGenControls, 'run_composite_gen')(
        SimpleNamespace(), MagicMock(), '/data/c'
    )
    session.post_processing.composite.assert_called_once_with('/data/c', on_progress=ANY)


@pytest.mark.parametrize(('typed', 'asked'), [(' 12 ', '12'), ('', None), ('Auto', None)])
def test_video_hands_the_typed_rate_to_the_build_to_judge(session, typed, asked):
    panel = SimpleNamespace(
        ids={
            'video_gen_fps_id': SimpleNamespace(text=typed),
            'enable_timestamp_overlay_btn': SimpleNamespace(state='down'),
        }
    )
    _body(post_processing.VideoCreationControls, 'run_video_gen')(panel, MagicMock(), '/data/v')
    session.post_processing.video.assert_called_once_with(
        '/data/v', frames_per_sec=asked, timestamp_overlay=True, on_progress=ANY
    )


def test_cell_count_asks_with_the_panels_method(session):
    method = {'context': {'pixels_per_um': 1.0}}
    panel = SimpleNamespace(_settings=method)
    _body(post_processing.CellCountControls, 'apply_method_to_folder')(
        panel, MagicMock(), '/data/cells'
    )
    session.post_processing.count_cells.assert_called_once_with(
        '/data/cells', method=method, on_progress=ANY
    )


def test_enhance_asks_and_shows_the_answer(session):
    panel = SimpleNamespace(
        _queue_derived_image=lambda image, bits: None,
        busy=True,
        status_text='',
        last_output_folder='',
    )
    panel._export_done = lambda result: post_processing.QuickEnhanceControls._export_done(
        panel, result
    )
    popup = MagicMock()
    _body(post_processing.QuickEnhanceControls, 'export')(panel, popup, '/data/one.tif')

    session.post_processing.enhance.assert_called_once_with(
        '/data/one.tif', on_progress=ANY, on_derived_image=panel._queue_derived_image
    )
    assert panel.busy is False
    assert panel.status_text == 'done'
    assert popup.text == 'done'
