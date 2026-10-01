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

import pytest

import modules.sequential_io_executor as sie
from modules.exceptions import PostProcessingRefusedError


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

    def record(self, **kwargs):
        seen.update(kwargs)
        seen['lane'] = getattr(sie._lane_worker, 'executor', None)
        return {'status': True, 'message': 'Success.'}

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

    result = getattr(session.post_processing, member)(tmp_path, **kwargs)

    assert result['message'] == 'Success.'
    assert seen['lane'] is session.post_processing.lane
    assert seen['lane'] is not session.file_io_executor
    assert seen['tiling_configs_file_loc'] == session.scope.protocols.tiling_configs_path()


def test_the_composite_takes_the_users_format_and_thresholds(session, monkeypatch, tmp_path):
    import modules.config_helpers as config_helpers
    from modules.composite_generation import CompositeGeneration

    seen = _spy_build(monkeypatch, CompositeGeneration, 'load_folder')
    session.post_processing.composite(tmp_path)

    settings = session.get_settings_snapshot()
    assert seen['output_format'] == settings['image_output_format']['sequenced']
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
    session.post_processing.video(tmp_path, frames_per_sec='12')
    assert seen['frames_per_sec'] == 12.0


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
