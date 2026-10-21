# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every post-processing build is a session member any caller reaches.

The builds were module classes only the GUI drove: it chose the tiling
file, the turret flag, the lane, the composite's settings and whether a
playback rate was valid, so a script or REST could reach a build only by
re-deciding all of that. Each build is now ``session.post_processing.<build>``:
it supplies those itself, runs on the post-processing lane, and answers
with its result or raises the typed outcome. No ``ui`` import here.
"""

from __future__ import annotations

import time

import pytest

import modules.sequential_io_executor as sie
from modules.exceptions import PostProcessingRefusedError
from modules.post_processing_api import CellCountResult


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def _spy_build(monkeypatch, cls, method_name):
    """Replace *cls.method_name* with a recorder; returns what it was called with."""
    seen = {}

    def record(self, path, **kwargs):
        seen.update(kwargs, path=path)
        seen['lane'] = getattr(sie._lane_worker, 'executor', None)
        seen['has_turret'] = self._has_turret
        return {
            'status': True,
            'message': 'Success.',
            'new_count': 0,
            'output_root': str(path),
            'artifact_paths': [],
            'excluded_inputs': (),
        }

    monkeypatch.setattr(cls, method_name, record)
    return seen


@pytest.mark.parametrize(
    ('member', 'module', 'cls', 'method', 'kwargs'),
    [
        ('stitch', 'modules.stitcher', 'Stitcher', 'load_folder', {}),
        ('zproject', 'modules.zprojector', 'ZProjector', 'load_folder', {}),
        ('composite', 'modules.composite_generation', 'CompositeGeneration', 'load_folder', {}),
        ('video', 'modules.video_builder', 'VideoBuilder', 'build_from_folder', {}),
    ],
)
def test_a_build_runs_on_its_lane_with_the_installations_tiling(
    session, monkeypatch, tmp_path, member, module, cls, method, kwargs
):
    import importlib

    from modules.zprojector import ZProjector

    if member == 'zproject':
        # A real method name, read from the build rather than spelled here.
        kwargs = {'method': ZProjector.methods()[0]}
    seen = _spy_build(monkeypatch, getattr(importlib.import_module(module), cls), method)

    # A turret scope, so a build told False is caught; and the caller's own
    # progress callback, so a build that drops it is caught.
    import dataclasses

    turret_scope = dataclasses.replace(session.scope.capabilities, has_turret=True)
    monkeypatch.setattr(session.scope, 'capabilities', turret_scope)

    told = []

    def progress(percent, text):
        told.append((percent, text))

    result = getattr(session.post_processing, member)(tmp_path, on_progress=progress, **kwargs)

    assert result.message == 'Success.'
    assert result.output_root == tmp_path
    assert seen['lane'] is session.post_processing.lane
    assert seen['lane'] is not session.file_io_executor
    assert seen['tiling_configs_file_loc'] == session.scope.protocols.tiling_configs_path()
    assert seen['has_turret'] is True
    # What the build is handed reaches the caller's own callback.
    seen['on_progress'](42, 'Image 42')
    assert told == [(42, 'Image 42')]


def test_the_composite_takes_the_users_format_and_thresholds(session, monkeypatch, tmp_path):
    import modules.config_helpers as config_helpers
    from modules.composite_generation import CompositeGeneration

    from modules import image_mode

    seen = _spy_build(monkeypatch, CompositeGeneration, 'load_folder')
    # Not the default, so a build that ignores the user's choice is caught.
    with session.settings_lock:
        session.settings['image_output_format']['sequenced'] = image_mode.OUTPUT_FORMAT_OME_TIFF
    session.post_processing.composite(tmp_path)

    settings = session.get_settings_snapshot()
    assert seen['output_format'] == settings['image_output_format']['sequenced']
    assert seen['output_format'] == image_mode.OUTPUT_FORMAT_OME_TIFF
    assert seen['brightness_thresholds_percent'] == (
        config_helpers.get_composite_blend_thresholds(settings)
    )


def test_a_folder_with_nothing_to_stitch_is_refused(session, tmp_path):
    with pytest.raises(PostProcessingRefusedError):
        session.post_processing.stitch(tmp_path)


@pytest.mark.parametrize(
    ('member', 'kwargs'),
    [
        ('stitch', {'mode': 'sideways'}),
        ('zproject', {'method': 'no such method'}),
        ('video', {'frames_per_sec': 0}),
        ('video', {'frames_per_sec': 0.5}),
        ('video', {'frames_per_sec': '-3'}),
        ('video', {'frames_per_sec': 'fast'}),
    ],
)
def test_a_setting_the_build_cannot_use_is_refused_not_adjusted(session, tmp_path, member, kwargs):
    with pytest.raises(PostProcessingRefusedError) as raised:
        getattr(session.post_processing, member)(tmp_path, **kwargs)
    assert raised.value.reason == 'invalid_setting'


def test_a_typed_playback_rate_reaches_the_build_as_a_number(session, monkeypatch, tmp_path):
    from modules.video_builder import VideoBuilder

    seen = _spy_build(monkeypatch, VideoBuilder, 'build_from_folder')
    session.post_processing.video(tmp_path, frames_per_sec='12', timestamp_overlay=True)
    assert seen['frames_per_sec'] == 12.0
    assert seen['enable_timestamp_overlay'] is True
    session.post_processing.video(tmp_path, frames_per_sec=1)
    assert seen['frames_per_sec'] == 1.0


def test_no_playback_rate_leaves_the_recordings_own_rate_to_the_build(
    session, monkeypatch, tmp_path
):
    from modules.video_builder import VideoBuilder

    seen = _spy_build(monkeypatch, VideoBuilder, 'build_from_folder')
    session.post_processing.video(tmp_path)
    assert seen['frames_per_sec'] is None
    assert seen['enable_timestamp_overlay'] is False


def test_a_cell_count_takes_the_callers_method_and_progress(session, monkeypatch, tmp_path):
    from modules.cell_count_method import CellCountMethod
    from modules.post_processing import PostProcessing

    seen = {}

    def record(self, path, method, on_progress=None):
        seen.update(path=path, method=method, on_progress=on_progress)
        seen['lane'] = getattr(sie._lane_worker, 'executor', None)
        return {'message': 'counted', 'results_path': str(path / 'results.csv'), 'counted': 3}

    monkeypatch.setattr(PostProcessing, 'apply_cell_count_to_folder', record)
    method = CellCountMethod(threshold=35)

    told = []

    def progress(percent, text):
        told.append((percent, text))

    assert session.post_processing.count_cells(
        tmp_path, method=method, on_progress=progress
    ) == CellCountResult(message='counted', results_path=tmp_path / 'results.csv', counted=3)
    assert seen['method'] is method
    # What the build is handed reaches the caller's own callback.
    seen['on_progress'](42, 'Image 42')
    assert told == [(42, 'Image 42')]
    assert seen['lane'] is session.post_processing.lane


def test_the_default_cell_count_method_is_the_one_the_count_and_rest_take(session, tmp_path):
    """A script or REST client starts from the method the panel starts from, and counts with it."""
    from fastapi.testclient import TestClient

    from modules.cell_count_method import CellCountMethod
    from rest.app import build_app

    method = session.post_processing.default_cell_count_method()
    assert method == CellCountMethod()

    (tmp_path / 'empty').mkdir()
    with TestClient(build_app(session)) as client:
        answer = client.get('/api/v1/post_processing/default_cell_count_method')
        assert (answer.status_code, answer.json()) == (200, method.to_dict())
        # The wire's method, sent back as it came, is built as the record:
        # refused for the empty folder, never for the method.
        counted = client.post(
            '/api/v1/post_processing/count_cells',
            json={'folder': 'empty', 'method': answer.json()},
        ).json()
        while counted.get('ended', True) is None:
            # Still running when the route stopped waiting: its job's end is the answer.
            counted = client.get(
                f'/api/v1/jobs/{counted["id"]}', headers={'Prefer': 'wait=60'}
            ).json()
    problem = counted.get('error', counted)
    assert problem.get('reason') == 'no_data', counted

    # Taken as it is in Python: the same refusal.
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.count_cells(tmp_path / 'empty', method=method)
    assert refused.value.reason == 'no_data'


def test_the_session_lists_and_stops_the_post_processing_lane(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    lane = session.post_processing.lane
    try:
        assert 'POSTPROC' in session.executor_bundle.snapshot()
        assert lane.worker_alive
    finally:
        session.shutdown()
    deadline = time.monotonic() + 5.0
    while lane.worker_alive and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not lane.worker_alive


def test_the_stitcher_plugin_stitches_through_the_session(session, monkeypatch, tmp_path):
    import functools

    from modules.plugins.builtin import stitcher_plugin
    from modules.stitcher import Stitcher

    seen = _spy_build(monkeypatch, Stitcher, 'load_folder')
    processor = functools.partial(stitcher_plugin._stitcher_processor, session.post_processing)

    result = processor(str(tmp_path), {}, '')

    assert result.success
    assert seen['lane'] is session.post_processing.lane
    assert seen['tiling_configs_file_loc'] == session.scope.protocols.tiling_configs_path()
