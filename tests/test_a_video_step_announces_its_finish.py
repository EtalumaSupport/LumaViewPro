# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's video step tells run-state listeners when its finish ends.

A video step's finish -- the MP4 closed, its execution-record row written --
runs on its own thread after the drain, while the run goes on. Until it
ends the Session's live_work lists the run's video; when it ends that
item is gone. The listener contract is that every change to the
Session's run state is announced, so a REST or headless listener re-reads
at the moment it changes rather than at some later, unrelated edge.

The manual recording announces its finish the same way. The step's busy
state is its own flag, cleared before it announces: the finish thread is
still alive while it tells the listener, and must not read busy to it.
"""

import threading

import modules.protocol_recording as protocol_recording
from modules.activity_claim import ActivityClaim
from tests.protocol_drives import run_identity
from tests.video_step_drives import ElapsedClock, video_step_on

WAIT_S = 10.0


def test_the_finish_end_is_announced_and_reads_not_busy(tmp_path, monkeypatch, sim_scope):
    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))
    # The first frame written ends the 1 s recording, so no wall time is waited.
    elapsed = ElapsedClock()
    monkeypatch.setattr(protocol_recording, 'write_frame', elapsed.pass_duration)

    heard = []
    step_box = {}
    in_finish = threading.Event()
    release_finish = threading.Event()

    def _record_step_row(**_kwargs):
        in_finish.set()
        release_finish.wait(WAIT_S)

    def _on_transition():
        step = step_box.get('step')
        if step is not None:
            heard.append(step.is_busy)

    run = ActivityClaim(on_transition=_on_transition).try_claim('protocol', run=run_identity())
    step = video_step_on(
        sim_scope,
        tmp_path,
        timestamp_overlay=False,
        record_step_row=_record_step_row,
        elapsed_clock=elapsed,
        run_claim=run.lend(),
    )
    step_box['step'] = step

    assert step.run_blocking() == protocol_recording.COMPLETED

    assert in_finish.wait(WAIT_S), 'the finish never reached its row'
    assert step.is_busy, 'the step read not busy while its finish was still running'
    release_finish.set()
    assert step.wait_until_finished(timeout=WAIT_S), 'the finish never ended'

    assert not step.is_busy
    assert heard and heard[-1] is False, (
        f'the finish ended without an announce a listener reads as not busy: {heard}'
    )
    assert run.holds, 'the step must leave the run holding its claim'
    run.release()
