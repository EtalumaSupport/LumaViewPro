# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording with a rate keeps its whole frame budget when its camera keeps the rate.

10 fps for 2 s is ``ceil(10 x 2)`` = 20 frames. The 20th starts at the last
slot's deadline, 2.0 s after the recording began, and reaches the recording
later by its exposure, readout and hand-over. The recording ends on its
frames' clock, so that frame is kept; the controller's steady-clock bound
waits a stall bound past the duration and never races it.

Each test holds every frame back on its way to the store by ten frames
(``delay_delivery``), as a driver's queue that has fallen behind, so the last
frame lands well after 2.0 s on the host's clock, and reads the count and the
end from the recording's manifest.
"""

import json

import _lumascope.modules.manual_recording as manual_recording_module
from tests.protocol_drives import a_claim_its_scope_asks
from _lumascope.modules.channel_record import release_colors
from _lumascope.modules.manual_recording import ManualRecordingController
from _lumascope.modules.scheduler import ThreadingTimerScheduler
from tests.scope_fakes import bind_settings_like_a_session
from tests.settings_fixtures import complete_settings
from tests.video_step_drives import video_step_on

BACKLOG_FRAMES = 10


def _the_manifest(folder):
    (path,) = folder.rglob('*manifest.json')
    return json.loads(path.read_text())


def test_a_video_steps_last_frame_held_on_its_way_is_kept(sim_scope, tmp_path, monkeypatch):
    import _lumascope.modules.protocol_recording as protocol_recording

    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))
    sim_scope.simulation.delay_delivery(BACKLOG_FRAMES)
    step = video_step_on(
        sim_scope,
        tmp_path,
        step={'Video Config': {'duration': 2}, 'Exposure': 2.0},
        global_max_fps=10,
    )
    assert step.run_blocking() == 'completed'
    assert step.wait_until_finished(timeout=20.0)

    manifest = _the_manifest(tmp_path)
    assert (manifest['end_reason'], manifest['frames_written']) == ('frame_budget_filled', 20)


def test_a_manual_recordings_last_frame_held_on_its_way_is_kept(
    sim_scope, tmp_path, monkeypatch, post_processing
):
    # The disk probe would report this machine's free space; ample free disk
    # keeps the recording's end the only thing under test.
    monkeypatch.setattr(
        manual_recording_module, 'check_disk_space_ok', lambda *_: (True, 1_000_000.0)
    )
    bind_settings_like_a_session(sim_scope, objective_id='4x Oly')
    sim_scope.simulation.delay_delivery(BACKLOG_FRAMES)
    scheduler = ThreadingTimerScheduler(name_prefix='TestWholeBudget')
    controller = ManualRecordingController(
        on_saved=lambda folder: None,
        scope=sim_scope,
        settings_snapshot=lambda: complete_settings(
            live_folder=str(tmp_path),
            video_as_frames=True,
            video={'max_fps': 10, 'max_duration_s': 2, 'timestamp_overlay': False},
        ),
        shown_channel=lambda: 'BF',
        channel_colors=release_colors,
        activity_claim=a_claim_its_scope_asks(sim_scope),
        scheduler=scheduler,
        post_processing=post_processing,
    )
    try:
        assert controller.start().result(timeout=30).end_reason == 'frame_budget_filled'
    finally:
        scheduler.shutdown()

    manifest = _the_manifest(controller.save_folder)
    assert (manifest['end_reason'], manifest['frames_written']) == ('frame_budget_filled', 20)
