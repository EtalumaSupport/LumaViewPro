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
import pytest


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules.image_mode import ImageCaptureConfig
from modules.manual_video_finalize import finalize_manual_video
from modules.notification_center import Severity, notifications


def _capture_notifications():
    """Record every notification the real center emits; return the list."""
    captured = []
    notifications.clear()
    notifications.add_listener(captured.append, min_severity=Severity.DEBUG)
    return captured


def _valid_image_capture_config():
    return ImageCaptureConfig.from_image_mode('8bit')


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


# --- sub-1-fps recording keeps a real rate (no floor-to-zero) --------------


def test_sub_one_fps_recording_keeps_nonzero_rate(tmp_path, monkeypatch):
    """A slow manual recording captures fewer frames than the seconds elapsed
    (long-exposure / timelapse). Its true rate is below 1 fps; computing it with
    floor division yields 0, which the encoder rejects -- the MP4 comes out empty
    and the recording is silently lost. The writer must receive the real sub-1
    rate, preserving the recording's playback duration."""
    notifications.clear()
    captured_fps = {}

    class _FpsCapturingWriter:
        def __init__(self, **kwargs):
            captured_fps['fps'] = kwargs['fps']
            self.dropped_frames = 0
            self.output_path = kwargs.get('output_path')

        def add_frame(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr('modules.manual_video_finalize.VideoWriter', _FpsCapturingWriter)

    # 3 frames over 10 seconds -> 0.3 fps. Floor division would yield 0.
    finalize_manual_video(
        **_kwargs(
            tmp_path,
            captured_frames=3,
            video_duration=10.0,
            video_frames=np.zeros((3, 16, 16), dtype=np.uint8),
        )
    )

    assert captured_fps['fps'] != 0, 'sub-1-fps recording must not hand the writer a 0 rate'
    assert captured_fps['fps'] == pytest.approx(0.3)
    notifications.clear()


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
                'image_capture_config': ImageCaptureConfig.from_image_mode(
                    '8bit', output_format_sequenced='OME-TIFF Hyperstack'
                ),
                'objective_info': None,  # snapshot failed
                'binning': 1,
            },
        )
    )

    assert result == 'sentinel-memmap.dat'
    assert any(n.severity >= Severity.ERROR for n in captured)
    notifications.clear()


# --- E1c: drop-count parity with the protocol video path -------------------


def _dropped_warnings(captured):
    return [
        n for n in captured if n.severity == Severity.WARNING and n.title == 'Video Frames Dropped'
    ]


def test_frames_path_drops_are_counted_and_notified(tmp_path, monkeypatch):
    """Every per-frame TIFF write failure is tallied into one warning."""
    captured = _capture_notifications()

    def _raise(**kwargs):
        raise OSError('disk full')

    monkeypatch.setattr('modules.image_save.write_video_frame', _raise)

    result = finalize_manual_video(**_kwargs(tmp_path, video_as_frames=True))

    assert result == 'sentinel-memmap.dat'  # recording released, not aborted
    warnings = _dropped_warnings(captured)
    assert len(warnings) == 1
    assert '3 of 3' in warnings[0].message  # all three frames failed
    notifications.clear()


def test_mp4_add_frame_failure_is_counted_and_notified(tmp_path, monkeypatch):
    """An MP4 writer that raises on add_frame contributes to the drop total."""
    captured = _capture_notifications()

    class _RaisingWriter:
        def __init__(self, **kwargs):
            self.dropped_frames = 0
            self.output_path = kwargs.get('output_path')

        def add_frame(self, **kwargs):
            raise RuntimeError('encoder gone')

        def close(self):
            pass

    monkeypatch.setattr('modules.manual_video_finalize.VideoWriter', _RaisingWriter)

    result = finalize_manual_video(**_kwargs(tmp_path, video_as_frames=False))

    assert result == 'sentinel-memmap.dat'
    assert len(_dropped_warnings(captured)) == 1
    notifications.clear()


def test_mp4_encoder_dropped_frames_are_notified(tmp_path, monkeypatch):
    """Encode failures the writer counts internally (no raise) still notify."""
    captured = _capture_notifications()

    class _DroppingWriter:
        def __init__(self, **kwargs):
            self.dropped_frames = 2  # accepted but lost inside the encoder
            self.output_path = kwargs.get('output_path')

        def add_frame(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr('modules.manual_video_finalize.VideoWriter', _DroppingWriter)

    result = finalize_manual_video(**_kwargs(tmp_path, video_as_frames=False))

    assert result == 'sentinel-memmap.dat'
    warnings = _dropped_warnings(captured)
    assert len(warnings) == 1
    assert '2 of 3' in warnings[0].message
    notifications.clear()


def test_no_drops_emits_no_drop_warning(tmp_path):
    """A clean recording must not fire the drop warning."""
    captured = _capture_notifications()

    finalize_manual_video(**_kwargs(tmp_path, video_as_frames=False))

    assert _dropped_warnings(captured) == []
    notifications.clear()


# --- hardening: any unexpected failure is surfaced, never silently discarded --


def test_unexpected_error_notifies_recording_not_saved(tmp_path):
    """An unexpected failure while consuming the snapshot must surface
    'Recording Not Saved' and re-raise, not vanish behind a log line. Here the
    config object is present (the None precondition passes) but is a stub
    missing output_format_sequenced, so the hyperstack probe raises -- the
    class of error the top-level guard converts from a silent discard into a
    loud notification."""
    from types import SimpleNamespace

    captured = _capture_notifications()

    kwargs = _kwargs(
        tmp_path,
        video_as_frames=True,
        ui_snapshot={
            'active_layer_config': ('Green', {}),
            'image_capture_config': SimpleNamespace(save_encoding='8bit', capture_depth=8),
            'objective_info': (None, {'focal_length': 1.0}),
            'binning': 1,
        },
    )

    with pytest.raises(AttributeError):
        finalize_manual_video(**kwargs)

    assert any(n.severity >= Severity.ERROR and n.title == 'Recording Not Saved' for n in captured)
    notifications.clear()
