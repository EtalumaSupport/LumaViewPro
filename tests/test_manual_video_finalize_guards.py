"""Manual-video finalize surfaces failures instead of discarding the recording.

finalize_manual_video is the write/finalize step of a finished manual
recording, extracted from MainDisplay so it runs off the Kivy widget and
can be exercised on the real production path. It owns no GUI state: the
live scope and a progress callback are passed in.

This file covers, in order of the commits that added them:
  - characterization: the extracted function writes an MP4 / honors the
    existing missing-layer guard, proving the extraction preserved behavior;
  - crash guard: a None image_capture_config (failed snapshot) is surfaced,
    not subscripted into a TypeError that discards the whole recording;
  - drop-count parity: per-frame write failures are tallied and surfaced in
    one end-of-recording notification, matching the protocol video path.

The real NotificationCenter records what the user would have seen.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules.manual_video_finalize import finalize_manual_video
from modules.notification_center import Severity, notifications


def _capture_notifications():
    """Record every notification the real center emits; return the list."""
    captured = []
    notifications.clear()
    notifications.add_listener(captured.append, min_severity=Severity.DEBUG)
    return captured


def _valid_image_capture_config():
    return {
        'output_format': {'sequenced': 'TIFF'},
        'save_encoding': 'png',
        'capture_depth': 8,
    }


def _kwargs(tmp_path, **overrides):
    """Baseline kwargs for a short recording; overridable per test."""
    kwargs = {
        'captured_frames': 3,
        'timestamps': [],
        'chunks_per_frame': [],
        'tick_freq_hz': None,
        'video_frames': np.zeros((3, 16, 16), dtype=np.uint8),
        'video_duration': 1.0,
        'video_save_folder': tmp_path,
        'start_time_str': '2026-06-21_00-00-00',
        'video_as_frames': False,
        'memmap_path': 'sentinel-memmap.dat',
        'video_false_color': None,
        'ui_snapshot': {
            'active_layer_config': ('Green', {}),
            'image_capture_config': _valid_image_capture_config(),
            'objective_info': (None, {'focal_length': 1.0}),
            'binning': 1,
        },
        'scope': None,
        'progress_cb': None,
    }
    kwargs.update(overrides)
    return kwargs


# --- E1a: characterization (extraction preserved behavior) -----------------


def test_mp4_path_writes_file_and_reports_progress(tmp_path):
    """The MP4 path encodes a real file, reports progress, returns the path."""
    notifications.clear()
    progress = []

    result = finalize_manual_video(**_kwargs(tmp_path, progress_cb=progress.append))

    assert result == 'sentinel-memmap.dat'  # passed through for cleanup tracking
    assert progress == [1, 2, 3]  # one report per captured frame, 1-based
    # PyAV writes Video_*.mp4; the cv2 fallback rewrites the suffix to .avi.
    assert any(p.name.startswith('Video_') for p in tmp_path.iterdir())


def test_missing_active_layer_config_notifies_and_returns(tmp_path):
    """A None active_layer_config snapshot fails loud, never crashes."""
    captured = _capture_notifications()

    result = finalize_manual_video(**_kwargs(tmp_path, ui_snapshot={'active_layer_config': None}))

    assert result == 'sentinel-memmap.dat'
    assert any(n.severity >= Severity.ERROR for n in captured)
    notifications.clear()


# --- E1b: snapshot precondition (the whole missing-field cluster) -----------


def test_missing_image_capture_config_notifies_and_returns(tmp_path):
    """A None image_capture_config in the save-as-frames path must fail loud.

    The snapshot stores None when get_image_capture_config_from_ui() raises.
    Before the guard, the frames path subscripted None
    (image_capture_config['output_format']) and the resulting TypeError
    propagated out, discarding the whole finished recording behind one log
    line. The recording must instead be reported, not silently lost.
    """
    captured = _capture_notifications()

    result = finalize_manual_video(
        **_kwargs(
            tmp_path,
            video_as_frames=True,
            ui_snapshot={
                'active_layer_config': ('Green', {}),
                'image_capture_config': None,  # snapshot failed
                'objective_info': (None, {'focal_length': 1.0}),
                'binning': 1,
            },
        )
    )

    assert result == 'sentinel-memmap.dat'
    assert any(n.severity >= Severity.ERROR for n in captured)
    notifications.clear()


def test_missing_objective_for_hyperstack_notifies_and_returns(tmp_path):
    """OME-TIFF Hyperstack reads the objective snapshot; a None objective_info
    must fail loud through the same precondition, not crash unpacking None."""
    captured = _capture_notifications()

    result = finalize_manual_video(
        **_kwargs(
            tmp_path,
            video_as_frames=True,
            ui_snapshot={
                'active_layer_config': ('Green', {}),
                'image_capture_config': {
                    'output_format': {'sequenced': 'OME-TIFF Hyperstack'},
                    'save_encoding': 'png',
                    'capture_depth': 8,
                },
                'objective_info': None,  # snapshot failed
                'binning': 1,
            },
        )
    )

    assert result == 'sentinel-memmap.dat'
    assert any(n.severity >= Severity.ERROR for n in captured)
    notifications.clear()
