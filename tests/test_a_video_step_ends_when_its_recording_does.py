# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's video step goes on the moment its recording ends.

The engine ends a recording on its frames' clock; the step waiting on it
wakes at that end, not at its next tick. The tick is only how often the
wait also reads a stop, the camera and the stall watch, so a step whose
tick is longer than the whole test still ends with its recording.
"""

import threading

import _lumascope.modules.protocol_recording as protocol_recording
from tests.video_step_drives import ElapsedClock, video_step_on

WAIT_S = 10.0


def test_the_step_returns_when_the_engine_ends_the_recording(tmp_path, monkeypatch, sim_scope):
    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))
    monkeypatch.setattr(protocol_recording, 'write_frame', lambda *a, **k: None)
    # A tick longer than the test: only the engine's end can end this wait.
    monkeypatch.setattr(protocol_recording, '_WAIT_TICK_S', 3 * WAIT_S)
    # The step's own clock stands still, so its backstop past the duration
    # never ends the wait either; the engine's frames' clock ends it.
    step = video_step_on(
        sim_scope,
        tmp_path,
        step={'Video Config': {'duration': 0.2}},
        elapsed_clock=ElapsedClock(),
    )

    outcome = []
    returned = threading.Event()

    def _run():
        outcome.append(step.run_blocking())
        returned.set()

    threading.Thread(target=_run, name='video-step-under-test', daemon=True).start()

    assert returned.wait(WAIT_S), 'the step waited out its tick after the recording ended'
    assert outcome == [protocol_recording.COMPLETED]
    assert step.wait_until_finished(timeout=WAIT_S), 'the finish never ended'
