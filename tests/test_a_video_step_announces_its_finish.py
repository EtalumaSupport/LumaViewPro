# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's video step tells run-state listeners when its finish ends.

A video step's finish -- the MP4 closed, its execution-record row written --
runs on its own thread after the drain, while the run goes on. Until it
ends the Session's close_drain_pending answers True; when it ends that
answer turns False. The listener contract is that every change to the
Session's run state is announced, so a REST or headless listener re-reads
at the moment it changes rather than at some later, unrelated edge.

The manual recording announces its finish the same way. The step's busy
state is its own flag, cleared before it announces: the finish thread is
still alive while it tells the listener, and must not read busy to it.
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np

import modules.protocol_recording as protocol_recording
from modules.activity_claim import ActivityClaim
from modules.protocol_recording import ProtocolVideoStep
from tests.scope_fakes import answer_auto_gain_like_the_api, spec_scope
from tests.protocol_drives import run_identity

WAIT_S = 10.0


def _scope(listeners):
    scope = spec_scope()
    answer_auto_gain_like_the_api(scope.imaging)
    scope.runtime_state.resolve_current_objective.return_value = (
        '4x Oly',
        {'focal_length': 45.0},
    )
    scope.motion.axis_positions = lambda: {}
    scope.runtime_state.plate_transform = lambda: None
    scope.illumination.get_led_states = lambda: {}
    scope.imaging.frames_until_valid.return_value = 0
    scope.imaging.active_cached = True
    scope.capabilities.camera_model = 'sim'
    scope.capabilities.camera_serial_number = '0'
    scope.capabilities.camera_timestamp_tick_hz = None
    scope.imaging.frame_size_cached = {'width': 8, 'height': 8}
    scope.imaging.add_frame_listener = lambda cb, name=None: listeners.update(cb=cb)
    return scope


def test_the_finish_end_is_announced_and_reads_not_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))
    monkeypatch.setattr(protocol_recording.image_save, 'write_video_frame', lambda **kw: None)

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

    run = ActivityClaim(on_transition=_on_transition).try_claim(
        'protocol', run=run_identity('test')
    )
    listeners = {}
    clock = {'t': 1000.0}
    step = ProtocolVideoStep(
        scope=_scope(listeners),
        step={
            'Video Config': {'fps': 5, 'duration': 1},
            'Color': 'Blue',
            'False_Color': False,
            'Auto_Gain': False,
            'Exposure': 10.0,
        },
        save_folder=tmp_path,
        name='clip',
        video_as_frames=True,
        capture_config=MagicMock(capture_depth=8, save_encoding='8bit'),
        timestamp_overlay=False,
        global_max_fps=0,
        autogain_settings={},
        callbacks={},
        aborted_event=threading.Event(),
        is_run_in_progress=lambda: True,
        abort_run_fatal=MagicMock(),
        record_step_row=_record_step_row,
        record_dropped_capture=MagicMock(),
        clock=lambda: clock['t'],
        run_claim=run.lend(),
        to_plate=None,
    )
    step_box['step'] = step

    worker = threading.Thread(target=step.run_blocking)
    worker.start()
    deadline = time.monotonic() + WAIT_S
    while 'cb' not in listeners and time.monotonic() < deadline:
        time.sleep(0.01)
    listeners['cb'](np.zeros((8, 8), dtype=np.uint8), 1000.3, None)
    clock['t'] = 1002.0
    worker.join(timeout=WAIT_S)

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
