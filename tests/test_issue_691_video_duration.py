"""Regression: #691 -- protocol video duration is not capped, and the
recording title shows seconds (not percent).

Bench feedback (2026-06-01): a protocol video-acquire step was silently
capped at 30s and its progress read as "% complete". The cap was wrong --
the global Video Time Limit is a manual-recording safety (forgot-to-stop),
not a protocol limit; a multi-minute protocol video is allowed. So:
- the per-step duration slider ceiling is 60s but the text box accepts
  longer (up to a 1-hour sanity bound),
- the protocol "video step exceeds limit" advisory is removed (its premise
  -- a protocol cap -- is gone),
- the recording title shows elapsed/total seconds, matching manual
  recording.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

for _kivy_submod in ('kivy.core', 'kivy.core.window', 'kivy.uix', 'kivy.uix.scrollview'):
    sys.modules.setdefault(_kivy_submod, MagicMock())

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Recording title shows seconds, not percent
# ---------------------------------------------------------------------------


def test_recording_title_shows_seconds_not_percent():
    import ui.ui_helpers as ui_helpers
    from modules.run_events import VideoProgress

    ui_helpers.show_video_progress(VideoProgress('recording', elapsed_s=12, total_s=30))
    title = ui_helpers._title_event_text
    assert '12s' in title and '30s' in title, title
    assert '%' not in title, f'recording title must not show percent: {title!r}'


def test_the_end_of_a_video_clears_only_its_own_title():
    import ui.ui_helpers as ui_helpers
    from modules.run_events import VideoProgress

    ui_helpers.show_video_progress(VideoProgress('recording', elapsed_s=7, total_s=30))
    ui_helpers.show_video_progress(VideoProgress('ended'))
    assert ui_helpers._title_event_text is None

    ui_helpers.show_video_progress(VideoProgress('writing', percent=40))
    ui_helpers.set_title_event_text('Homing, please wait...')
    ui_helpers.show_video_progress(VideoProgress('ended'))
    assert ui_helpers._title_event_text == 'Homing, please wait...'


# ---------------------------------------------------------------------------
# Protocol video is not capped: the advisory was removed
# ---------------------------------------------------------------------------


def test_protocol_video_advisory_removed():
    """The protocol 'video step exceeds limit' advisory is gone -- protocol
    video has no cap, so the warning premise no longer exists."""
    from modules.protocol import Protocol

    assert not hasattr(Protocol, 'video_steps_over_limit'), (
        'protocol video is uncapped now; the advisory method must not exist'
    )
    src = (REPO_ROOT / 'modules' / 'sequenced_capture_runner.py').read_text(encoding='utf-8')
    assert 'video_steps_over_limit' not in src, 'run-start video-limit advisory must be removed'


# ---------------------------------------------------------------------------
# Text box accepts a duration beyond the slider ceiling
# ---------------------------------------------------------------------------


def _method_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


def test_video_duration_text_allows_beyond_slider():
    """A typed duration past the slider's 60 s ceiling is written as typed (a
    multi-minute protocol video): the box hands the writer no ceiling, and the
    writer's own range for a step's duration runs to an hour."""
    import json

    from modules.exceptions import SettingRefusedError
    from modules.settings_paths import VIDEO_STEP_DURATION_S_MAX, check_write

    method = _method_node(REPO_ROOT / 'ui' / 'layer_control.py', 'video_duration_text')
    assert 'value_max' not in ast.unparse(method), (
        "video_duration_text must hand the writer no ceiling; the range is the writer's"
    )
    template = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text(encoding='utf-8'))
    check_write(template, 'BF.video_config.duration', 600)
    with pytest.raises(SettingRefusedError):
        check_write(template, 'BF.video_config.duration', VIDEO_STEP_DURATION_S_MAX + 1)


def test_video_duration_slider_ceiling_is_60():
    # pin-justified: kv is declarative source with no headless seam; the
    # slider ceiling in the kv text is the contract.
    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text(encoding='utf-8')
    # Find the video_duration_slider block and assert its max is 60.
    idx = kv.index('id: video_duration_slider')
    block = kv[idx : idx + 200]
    assert 'max: 60' in block, 'video_duration_slider ceiling must be 60s'
