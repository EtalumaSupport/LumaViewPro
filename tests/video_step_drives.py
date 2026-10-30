# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Builders that drive a real ProtocolVideoStep on a simulated scope.

``video_step_on`` builds the step over a scope as a run builds it; the
simulated camera's own stream delivers the frames. ``ElapsedClock`` is the
step's elapsed clock held still, so a recording ends when the test says
its duration has passed (on the first frame written, as a rule) and no
wall time is waited.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from modules.image_mode import ImageCaptureConfig
from modules.layer_record import release_colors
from modules.protocol_recording import ProtocolVideoStep
from modules.run_events import RunEvents
from tests.frame_records import plate
from tests.protocol_drives import lent_run_claim
from tests.stock_channels import LED0

# A one-second step at 5 fps, manual gain, as the protocol editor writes it.
VIDEO_STEP = {
    'Video Config': {'fps': 5, 'duration': 1},
    'Color': LED0,
    'False_Color': False,
    'Auto_Gain': False,
    'Exposure': 10.0,
}


class ElapsedClock:
    """The step's elapsed clock, standing at 0 until ``pass_duration``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def pass_duration(self, *_args, **_kwargs) -> None:
        """Move past any step's duration; takes a write's arguments, so it can stand as one."""
        self.now = 1e6


def video_step_on(
    scope, save_folder, *, step: dict | None = None, **overrides
) -> ProtocolVideoStep:
    """A ProtocolVideoStep over *scope*, writing into *save_folder*.

    *step* overrides members of ``VIDEO_STEP``; every other keyword
    overrides the step's constructor argument of that name. The run's
    callbacks are MagicMocks a test reads or replaces.
    """
    kwargs = {
        'scope': scope,
        'step': {**VIDEO_STEP, **(step or {})},
        'save_folder': save_folder,
        'name': 'clip',
        'video_as_frames': True,
        'capture_config': ImageCaptureConfig.from_image_mode('8bit'),
        'timestamp_overlay': True,
        'global_max_fps': 0,
        'autogain_settings': {},
        'events': RunEvents(),
        'aborted_event': threading.Event(),
        'is_run_in_progress': lambda: True,
        'abort_run_fatal': MagicMock(),
        'record_step_row': MagicMock(),
        'record_dropped_capture': MagicMock(),
        'run_claim': lent_run_claim(),
        'to_plate': None,
        'labware': plate(),
        'fallback_focus': None,
        'channel_colors': release_colors(),
    }
    kwargs.update(overrides)
    return ProtocolVideoStep(**kwargs)


def record_first_frame(scope, save_folder, monkeypatch, *, write_fails=False, **overrides):
    """Run a frames-leg step on *scope* to its finish; its first frame written ends it.

    The frame's write records what arrived with it and stands in for the
    disk, failing with ``OSError`` when *write_fails*. Returns the step and
    the arrivals written, in order.
    """
    import modules.protocol_recording as protocol_recording

    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))
    elapsed = ElapsedClock()
    written = []

    def _write_frame(_scope, _image, _file_loc, arrived, **_kwargs):
        elapsed.pass_duration()
        written.append(arrived)
        if write_fails:
            raise OSError('disk said no')

    monkeypatch.setattr(protocol_recording, 'write_frame', _write_frame)
    step = video_step_on(scope, save_folder, elapsed_clock=elapsed, **overrides)
    step.run_blocking()
    assert step.wait_until_finished(timeout=10.0), 'the finish never ended'
    return step, written
