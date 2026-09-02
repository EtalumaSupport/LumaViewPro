# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Builders that drive the protocol stack headlessly on a real
SequencedCaptureRunner with MagicMock deps.

Three layers of readiness:
- bare_capture_runner(): a constructed runner; enough for prepare()'s
  refusal gates and start()'s snapshot phase.
- scan_ready_runner(): the scan-ready state prepare()+start() normally
  establish, with a single-step protocol mock -- drive
  runner._step_executor.scan_iterate() / scan_loop() directly.
- run_loop_ready_runner(): additionally RUNNING state, zero period,
  go_to_step callback, and a mocked _cleanup -- drive
  runner._run_loop_executor.run_loop() synchronously on the test
  thread (cleanup behavior is covered separately on run_cleanup).
"""

from __future__ import annotations

import datetime
import time
from unittest.mock import MagicMock

from modules.image_mode import ImageCaptureConfig


def wait_until_not_running(session, timeout: float = 5.0) -> bool:
    """Wait for a finished run to release the activity claim.

    `run_complete` fires DURING cleanup; the claim -- and with it
    `session.is_protocol_running` -- releases at cleanup END, a moment
    later. A test that waits on the callback and then asserts the state
    immediately is asserting mid-teardown, and passes or fails on timing.

    Shared because two test modules assert this same state after a
    completed run, and a second copy is a second thing to drift.
    """
    deadline = time.monotonic() + timeout
    while session.is_protocol_running:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def protocol_step(**overrides):
    """A scan_iterate-shaped step dict (plain dict, not pandas Series)."""
    step = {
        'Auto_Focus': False,
        'Auto_Gain': False,
        'Color': 'BF',
        'Illumination': 50.0,
        'Gain': 2.0,
        'Exposure': 10.0,
        'Z': 100.0,
        'X': 1.0,
        'Y': 2.0,
        'Sum': 1,
        'Objective': 'objective-under-test',
    }
    step.update(overrides)
    return step


def bare_capture_runner(**overrides):
    """SequencedCaptureRunner with MagicMock deps."""
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    kwargs = {
        'scope': MagicMock(),
        'stage_offset': {},
        'io_executor': MagicMock(),
        'protocol_thread': MagicMock(),
        'file_io_executor': MagicMock(),
        'camera_executor': MagicMock(),
        'autofocus_thread': MagicMock(is_running=False),
        'autofocus_runner': MagicMock(),
    }
    kwargs.update(overrides)
    runner = SequencedCaptureRunner(**kwargs)
    runner.file_io_executor.is_protocol_queue_active.return_value = False
    # The real executor returns an int drop count (0 on a clean run); the mock
    # must too, or run-end cleanup compares a MagicMock against an int.
    runner.file_io_executor.protocol_dropped_count.return_value = 0
    runner.file_io_executor.protocol_backpressure_blocked_s.return_value = 0.0
    return runner


def scr_run_kwargs(**overrides):
    """Keyword args for SequencedCaptureRunner.prepare() with a protocol
    mock that passes every refusal gate; tests override the gate or
    snapshot under test."""
    from modules.sequenced_capture_runner import SequencedCaptureRunMode

    protocol = MagicMock()
    protocol.num_steps.return_value = 1
    protocol.validate_for_run.return_value = []
    # Real timedeltas: Protocol.period() never returns an int, and a stub
    # that did once let a zero-period guard pass green against a type the
    # production path never produces.
    protocol.period.return_value = datetime.timedelta(0)
    protocol.duration.return_value = datetime.timedelta(hours=1)
    protocol.copy_for_execution.return_value = protocol
    kwargs = {
        'protocol': protocol,
        'run_trigger_source': 'test',
        'run_mode': SequencedCaptureRunMode.FULL_PROTOCOL,
        'sequence_name': 'seq',
        'image_capture_config': ImageCaptureConfig.from_image_mode('8bit'),
        'autogain_settings': {'target_brightness': 0.3},
        'parent_dir': None,
        'disable_saving_artifacts': True,
        'initial_autofocus_states': {},
    }
    kwargs.update(overrides)
    return kwargs


def scan_ready_runner(step, **state):
    """Runner advanced to the scan-ready state prepare()+start()
    normally establish, with a single-step protocol mock returning *step*.
    Keyword args land as runner attributes (e.g. _n_scans=2)."""
    runner = bare_capture_runner()
    runner._scope.motion.is_moving.return_value = False
    runner._scope.led_connected = False
    protocol = MagicMock()
    protocol.step.return_value = step
    protocol.num_steps.return_value = 1
    runner._protocol = protocol
    runner._n_scans = 1
    runner._scan_in_progress.set()
    runner._run_in_progress_event.set()
    runner._autogain_settings = {}
    runner._image_writer = MagicMock()
    runner._disable_saving_artifacts = True
    runner._enable_image_saving = False
    runner._image_capture_config = ImageCaptureConfig.from_image_mode('8bit')
    runner._separate_folder_per_channel = False
    runner._video_as_frames = False
    runner._leds_state_at_end = 'off'
    runner._keep_led_between_steps = False
    runner._update_z_pos_from_autofocus = False
    runner._save_autofocus_data = False
    runner._parent_dir = None
    runner._run_trigger_source = 'test'
    for key, value in state.items():
        setattr(runner, key, value)
    return runner


def run_loop_ready_runner(step, n_scans=1, **state):
    """Runner ready for a synchronous run_loop() drive: RUNNING state,
    zero-period protocol, go_to_step handled by a callback mock, and
    _cleanup mocked out (its behavior is covered on run_cleanup)."""
    from modules.protocol_callbacks import ProtocolCallbacks
    from modules.protocol_state_machine import ProtocolState

    runner = scan_ready_runner(step, **state)
    runner._scan_in_progress.clear()
    runner._n_scans = n_scans
    runner._protocol.period.return_value = datetime.timedelta(0)
    # _start_t is a monotonic timestamp (seconds), matching the run loop's pacing.
    runner._start_t = time.monotonic()
    runner._callbacks = ProtocolCallbacks(go_to_step=MagicMock())
    runner._cleanup = MagicMock()
    runner._set_state(ProtocolState.RUNNING)
    return runner
