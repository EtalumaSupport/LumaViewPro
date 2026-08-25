# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
E2E: a manual recording on the real simulated stack ends within the
stall bound when the feed silently dies.

Production-path injection: stop_streaming halts the sim camera's
callback pump WITHOUT flipping active_cached -- the silent-stall shape.
The controller's tick (the GUI's poll stand-in here) must stop the
recording, keep the frames, and record camera_stalled in the manifest.
"""

import json
import time

import modules.manual_recording as manual_recording_module
from modules.manual_recording import ManualRecordingController
from tests.video_engine_harness import ClaimStub


def test_manual_recording_ends_within_the_stall_bound(sim_scope, tmp_path, monkeypatch):
    # conftest mocks psutil, so the real disk probe returns MagicMocks;
    # report ample free disk instead.
    monkeypatch.setattr(
        manual_recording_module, 'check_disk_space_ok', lambda *_: (True, 1_000_000.0)
    )
    settings = {
        'live_folder': str(tmp_path),
        'video_as_frames': True,
        'video': {'max_fps': 0, 'max_duration_seconds': 60, 'timestamp_overlay': False},
    }
    controller = ManualRecordingController(
        scope=sim_scope, settings=settings, activity_claim=ClaimStub()
    )
    controller.start()
    time.sleep(1.0)
    sim_scope.imaging.stop_streaming()

    deadline = time.monotonic() + 30.0
    while controller.is_recording and time.monotonic() < deadline:
        controller.tick()
        time.sleep(0.1)

    assert not controller.is_recording, 'the stall bound must end the recording'
    while controller.is_busy and time.monotonic() < deadline:
        time.sleep(0.1)

    manifest_path = controller.save_folder / 'recording_manifest.json'
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest['end_reason'] == 'camera_stalled'
    frames = list(controller.save_folder.glob('ManualVideo_Frame_*.tiff'))
    assert len(frames) == manifest['frames_written']
    assert frames, 'frames captured before the death must stay on disk'
