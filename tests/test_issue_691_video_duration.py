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

for _kivy_submod in ('kivy.core', 'kivy.core.window', 'kivy.uix', 'kivy.uix.scrollview'):
    sys.modules.setdefault(_kivy_submod, MagicMock())

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Recording title shows seconds, not percent
# ---------------------------------------------------------------------------


def test_recording_title_shows_seconds_not_percent():
    import ui.ui_helpers as ui_helpers

    ui_helpers.set_recording_title(elapsed_sec=12, total_sec=30)
    title = ui_helpers._title_event_text
    assert '12s' in title and '30s' in title, title
    assert '%' not in title, f'recording title must not show percent: {title!r}'


def test_recording_title_elapsed_only_and_start():
    import ui.ui_helpers as ui_helpers

    ui_helpers.set_recording_title(elapsed_sec=7)
    assert '7s' in ui_helpers._title_event_text
    assert '%' not in ui_helpers._title_event_text

    ui_helpers.set_recording_title()
    assert ui_helpers._title_event_text == 'Recording Video...'


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
    assert 'video_steps_over_limit' not in src, (
        'run-start video-limit advisory must be removed'
    )


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
    """video_duration_text passes value_max so a typed value can exceed the
    slider's 60s ceiling (a multi-minute protocol video)."""
    method = _method_node(REPO_ROOT / 'ui' / 'layer_control.py', 'video_duration_text')
    src = ast.unparse(method)
    assert 'value_max' in src, (
        'video_duration_text must pass value_max so the text box accepts a '
        'duration longer than the slider ceiling'
    )


def test_video_duration_slider_ceiling_is_60():
    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text(encoding='utf-8')
    # Find the video_duration_slider block and assert its max is 60.
    idx = kv.index('id: video_duration_slider')
    block = kv[idx : idx + 200]
    assert 'max: 60' in block, 'video_duration_slider ceiling must be 60s'
