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
  runner._run_loop_executor.run_loop(runner.run_outcome()) synchronously
  on the test thread (cleanup behavior is covered separately on
  run_cleanup).
"""

from __future__ import annotations

import datetime
import threading
import time
from unittest.mock import MagicMock

from modules.activity_claim import ActivityClaim, RunIdentity
from modules.image_mode import ImageCaptureConfig
from modules.run_outcome import CaptureTally
from tests.scope_fakes import swap_lanes


def held_run_claim():
    """A run's activity claim, as the run holds it: what a top-level LED
    lease is taken under. Each call is a fresh claim, so two leases never
    share one; release() it to strand a lease taken under it."""
    return ActivityClaim().try_claim('protocol', run=run_identity('test'))


def lent_run_claim():
    """A run's activity claim, lent: what the run's writer and its video
    steps receive, so they record under the run's claim."""
    return held_run_claim().lend()


def wait_until_not_running(session, timeout: float = 5.0) -> bool:
    """Wait for a finished run to release the activity claim.

    `run_ended` and `handle.wait()` come only once the claim -- and
    with it `session.is_protocol_running` -- is released, so after either
    this returns at once; it confirms the release for a test that asserts
    the state without having waited on the run itself.

    Shared because two test modules assert this same state after a
    completed run, and a second copy is a second thing to drift.
    """
    deadline = time.monotonic() + timeout
    while session.is_protocol_running:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


# The longest a run may go without starting a step before it is called
# stalled. One simulated step takes about 150 ms alone; this is far past a
# step slowed by a loaded host, and short enough that a hang fails promptly.
STEP_STALL_S = 15.0


class StepHeartbeat:
    """A run's step_started handler that also notes when each step starts.

    The runner sends step_started once per step, so the time since the last
    one says whether the run is still moving. Wraps the test's own handler,
    if it has one.
    """

    def __init__(self, inner=None):
        self._inner = inner
        self._last = time.monotonic()

    def __call__(self, step_idx):
        self._last = time.monotonic()
        if self._inner is not None:
            self._inner(step_idx)

    def idle_s(self) -> float:
        return time.monotonic() - self._last


def wait_for_run_end(done: threading.Event, heartbeat: StepHeartbeat) -> bool:
    """Wait for a run to end; False only when it stopped starting steps.

    A bound on the whole run fails a long run on a loaded host: the 50-step
    runs take about 8 s alone and failed their 15 s bound whenever another
    suite shared the machine. A run that keeps starting steps is not hung,
    however slowly it goes, so only a stall of STEP_STALL_S fails it.
    """
    while not done.wait(timeout=0.25):
        if heartbeat.idle_s() > STEP_STALL_S:
            return done.is_set()
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
        # scan_iterate is handed a full schema row in production and reads the
        # grouping to decide both LED hold and z-stack focus; -1 is the
        # not-part-of-a-stack sentinel, so an unstacked step is the default.
        'Z-Stack Group ID': -1,
    }
    step.update(overrides)
    return step


def run_identity(trigger: str = 'test', words: str = 'scan') -> RunIdentity:
    """A run's identity for a test that takes or dispatches as a run."""
    return RunIdentity(trigger=trigger, words=words)


def bare_capture_runner(**overrides):
    """SequencedCaptureRunner with MagicMock deps."""
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    kwargs = {
        'scope': MagicMock(),
        'protocol_thread': MagicMock(),
        'file_io_executor': MagicMock(),
        'autofocus_thread': MagicMock(in_flight_sweep=None),
        'activity_claim': ActivityClaim(),
        'autofocus_runner': MagicMock(),
    }
    kwargs.update(overrides)
    if 'scope' not in overrides:
        # A run is refused, and a capture raises, while any axis position is
        # unknown; a bare mock answers that question with a truthy mock, so
        # the default scope states the homed answer. A test about position
        # passes its own scope.
        kwargs['scope'].motion.axes_without_position.return_value = {}
    # The engine reads IO and CAMERA from its scope; a test that passes its
    # own lane puts it there.
    swap_lanes(
        kwargs['scope'],
        io=kwargs.pop('io_executor', None),
        camera=kwargs.pop('camera_executor', None),
    )
    runner = SequencedCaptureRunner(**kwargs)
    # A run takes the camera only once the camera lane is idle; a bare mock
    # answers "busy" and "stalled" with truthy mocks, so the default lane
    # states the idle answer. A test about the lane passes its own executor.
    runner.camera_executor.is_busy.return_value = False
    runner.camera_executor.in_flight_task_stalled.return_value = False
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
    # The run hands its writer the plate the protocol names, so the plate
    # is a catalogue name, as a real protocol's is after validate_for_run.
    protocol.labware.return_value = '96 well microplate'
    kwargs = {
        'protocol': protocol,
        'run_trigger_source': 'test',
        'run_mode': SequencedCaptureRunMode.FULL_PROTOCOL,
        'sequence_name': 'seq',
        'image_capture_config': ImageCaptureConfig.from_image_mode('8bit'),
        'autogain_settings': {'target_brightness': 0.3},
        'parent_dir': None,
        'disable_saving_artifacts': True,
    }
    kwargs.update(overrides)
    return kwargs


def stand_in_step_targets(scope, *, plate_to_stage=(0.0, 0.0)):
    """Answer ``scope.protocols``' step conversion from the protocol's own rows.

    A mock scope's protocols API cannot convert a plate position (its plate
    catalogue is a mock); this answers every step's X/Y with
    *plate_to_stage* and its Z from the step, with no turret slot.
    """
    from modules.lumascope_api.protocols import StepTargets

    scope.protocols.plate_to_stage.side_effect = lambda protocol, px, py, stage_offset=None: (
        plate_to_stage
    )
    scope.protocols.step_targets.side_effect = lambda protocol, step_idx, stage_offset=None: (
        StepTargets(
            turret_slot=None,
            x=plate_to_stage[0],
            y=plate_to_stage[1],
            z=protocol.step(idx=step_idx)['Z'],
        )
    )


def scan_ready_runner(step, **state):
    """Runner advanced to the scan-ready state prepare()+start()
    normally establish, with a single-step protocol mock returning *step*.
    Keyword args land as runner attributes (e.g. _n_scans=2)."""
    from modules.protocol_state_machine import ProtocolState, SequencedCaptureRunMode

    runner = bare_capture_runner()
    runner._scope.motion.is_moving.return_value = False
    runner._scope.led_connected = False
    protocol = MagicMock()
    protocol.step.return_value = step
    protocol.num_steps.return_value = 1
    runner._protocol = protocol
    runner._n_scans = 1
    runner._scan_in_progress.set()
    runner._state = ProtocolState.RUNNING
    # prepare() always carries the target the takeover writes to the camera.
    runner._autogain_settings = {'target_brightness': 0.5}
    runner._image_writer = MagicMock()
    # This drive saves nothing, so its run is asked for no captures.
    runner._image_writer.capture_tally = CaptureTally(asked=0, captured=0, failed=())
    runner._disable_saving_artifacts = True
    runner._enable_image_saving = False
    runner._image_capture_config = ImageCaptureConfig.from_image_mode('8bit')
    runner._separate_folder_per_channel = False
    runner._video_as_frames = False
    runner._run_mode = SequencedCaptureRunMode.FULL_PROTOCOL
    runner._keep_led_between_steps = False
    runner._ag_ae_max_exposure_ms = {}
    runner._write_focus_to = None
    runner._save_autofocus_data = False
    runner._parent_dir = None
    runner._run_identity = run_identity()
    for key, value in state.items():
        setattr(runner, key, value)
    return runner


def run_loop_ready_runner(step, n_scans=1, **state):
    """Runner ready for a synchronous run_loop() drive: RUNNING state,
    zero-period protocol, step_started handled by a mock, and
    _cleanup mocked out (its behavior is covered on run_cleanup)."""
    from modules.run_events import RunEvents
    from modules.protocol_state_machine import ProtocolState
    from modules.run_outcome import PendingRunOutcome

    runner = scan_ready_runner(step, **state)
    runner._scan_in_progress.clear()
    runner._n_scans = n_scans
    runner._protocol.period.return_value = datetime.timedelta(0)
    # _start_t is a monotonic timestamp (seconds), matching the run loop's pacing.
    runner._start_t = time.monotonic()
    runner._events = RunEvents(step_started=MagicMock())
    # The run moves every step itself: a turretless scope on a flat plate
    # frame, its moves queued on the io executor mock.
    runner._scope.capabilities.has_turret = False
    stand_in_step_targets(runner._scope, plate_to_stage=(0.0, 0.0))
    runner._cleanup = MagicMock()
    # The loop's first act takes the camera: it snapshots the camera and
    # takes the auto-gain arm out of that snapshot. A bare mock snapshot
    # answers the arm with a mock, which the arm take cannot apply, so the
    # default snapshot states the common case: no standing arm.
    runner._scope.imaging.save_camera_state.return_value = {'auto_gain_arm': None}
    # The run the loop is dispatched for: every cleanup it asks for names it.
    runner._run_outcome = PendingRunOutcome()
    runner._set_state(ProtocolState.RUNNING)
    return runner
