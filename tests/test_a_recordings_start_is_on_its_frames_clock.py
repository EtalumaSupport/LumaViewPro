# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording's start is read on the clock its frames are timed on.

A recording, manual or a run's video step, keeps a frame by comparing the
frame's time with the recording's start. A frame's time is its start on the
camera's clock, carried onto the wall clock through the camera's clock
mapping, so the start is read through that same mapping: the host's wall
clock stepping, or reading in coarse steps (15.6 ms on Windows under Python
3.12), between the mapping's latch and the start cannot move the start
against the frames.

Each test gives the recording a wall clock that reads 500 ms ahead, as a wall
step between the latch and the start would, keeps every frame (no rate
limit), and reads the first frame the recording saved from its file's
record. The first kept frame started within a frame or two of the moment the
recording began, as the test's own wall reads bracket it; a start read on the
stepped clock instead leaves out the first 500 ms of frames.
"""

import threading
import time

import modules.manual_recording as manual_recording_module
from modules.activity_claim import ActivityClaim
from modules.image_utils import read_file_record, stated_frame_time
from modules.layer_record import release_colors
from modules.manual_recording import ManualRecordingController
from modules.run_events import RunEvents, VideoPhase
from tests.scope_fakes import bind_settings_like_a_session
from tests.settings_fixtures import complete_settings
from tests.video_engine_harness import ManualFireScheduler
from tests.video_step_drives import video_step_on

WALL_STEP_S = 0.5
# Far below the step and far above a simulated frame period and a coarse
# wall read: how late after the recording began its first kept frame may start.
LATEST_FIRST_FRAME_S = 0.2


def _stepped_wall_clock():
    return time.time() + WALL_STEP_S


def _first_kept_frame_time(folder):
    """The earliest time any frame saved under ``folder`` states, in epoch seconds."""
    times = [stated_frame_time(read_file_record(path)) for path in folder.rglob('*.tif*')]
    assert times, 'the recording saved no frame'
    return min(times).timestamp()


def test_a_video_steps_first_frame_is_not_moved_by_a_step_of_the_wall_clock(sim_scope, tmp_path):
    began = []

    def on_progress(progress):
        if progress.phase == VideoPhase.RECORDING and not began:
            began.append(time.time())

    before = time.time()
    step = video_step_on(
        sim_scope,
        tmp_path,
        step={'Video Config': {'fps': 1000, 'duration': 1}},
        events=RunEvents(video_progress=on_progress),
        clock=_stepped_wall_clock,
    )
    assert step.run_blocking() == 'completed'
    assert step.wait_until_finished(timeout=20.0)

    first = _first_kept_frame_time(tmp_path)
    assert before <= first <= began[0] + LATEST_FIRST_FRAME_S, (
        f'the first kept frame started {first - began[0]:+.4f} s after the step began recording'
    )


def test_a_manual_recordings_first_frame_is_not_moved_by_a_step_of_the_wall_clock(
    sim_scope, tmp_path, monkeypatch, post_processing
):
    # The disk probe would report this machine's free space; ample free disk
    # keeps the recording's start the only thing under test.
    monkeypatch.setattr(
        manual_recording_module, 'check_disk_space_ok', lambda *_: (True, 1_000_000.0)
    )
    bind_settings_like_a_session(sim_scope, objective_id='4x Oly')
    controller = ManualRecordingController(
        scope=sim_scope,
        settings_snapshot=lambda: complete_settings(
            live_folder=str(tmp_path),
            video_as_frames=True,
            video={'max_fps': 0, 'max_duration_s': 60, 'timestamp_overlay': False},
        ),
        shown_layer=lambda: 'BF',
        channel_colors=release_colors,
        activity_claim=ActivityClaim(),
        scheduler=ManualFireScheduler(),
        post_processing=post_processing,
        clock=_stepped_wall_clock,
    )
    delivered = threading.Event()
    seen = []

    def on_frame(frame):
        seen.append(frame)
        if len(seen) >= 60:
            delivered.set()

    before = time.time()
    operation = controller.start()
    began = time.time()
    sim_scope.imaging.add_frame_listener(on_frame, name='test_frames_after_the_start')
    try:
        assert delivered.wait(10.0), 'the camera delivered fewer than 60 frames'
    finally:
        sim_scope.imaging.remove_frame_listener(on_frame)
    operation.stop()
    operation.result(timeout=30)

    first = _first_kept_frame_time(controller.save_folder)
    assert before <= first <= began + LATEST_FIRST_FRAME_S, (
        f'the first kept frame started {first - began:+.4f} s after the recording began'
    )
