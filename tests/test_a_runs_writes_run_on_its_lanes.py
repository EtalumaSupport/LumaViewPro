# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's instrument writes run on the lanes, under the run's taking.

Before, run code bound the private bodies of the hardware members and wrote
the instrument from its own threads -- the protocol thread, the autofocus
thread, cleanup's -- because the public members refused the run's own work
while it held the lanes. Now the lanes ask the activity claim, so the run
calls the public members under its taking and each write executes on the
lane's worker: LED and motion on IO, the camera on CAMERA.

What that must not cost, each driven on the simulator:
- a camera setting the camera rejects mid-run is reported and the run goes on;
- the safety darken does not wait behind whatever the IO lane is busy with;
- a camera task queued as Run is pressed is refused to its waiter, and the
  run does not hang behind it, now that the camera is fenced like IO.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import modules.sequenced_capture_runner as scr
from modules.exceptions import HardwareCommandRefusedError
from modules.run_outcome import RunEnding
from modules.sequential_io_executor import IOTask
from tests.protocol_drives import held_run_claim
from tests.test_a_run_waits_for_the_camera_lane import RESULT_TIMEOUT_S, _LaneHold, lane_session  # noqa: F401
from tests.test_protocol_cleanup_extinguish import _make_runner_stub

# The driver calls that change the instrument, per driver. Reads, grabs and
# the simulator's own focus coupling are not writes.
_WRITES = {
    '_led_driver': (
        'led_on',
        'led_off',
        'leds_off',
        'led_on_fast',
        'led_off_fast',
        'leds_off_fast',
    ),
    '_motion_driver': ('move_abs_pos', 'move_rel_pos', 'move', 'set_precision_mode'),
    '_camera_driver': (
        'gain',
        'exposure_t',
        'auto_gain',
        'auto_gain_once',
        'update_auto_gain_target_brightness',
    ),
}
_LANE_OF = {'_led_driver': 'io', '_motion_driver': 'io', '_camera_driver': 'camera'}


def _record_write_threads(scope, monkeypatch) -> list:
    """Wrap every instrument write so it records the thread it executed on."""
    seen = []
    for driver_attr, names in _WRITES.items():
        driver = getattr(scope, driver_attr)
        for name in names:
            real = getattr(driver, name, None)
            if real is None:
                continue

            def _spy(*args, _real=real, _where=(driver_attr, name), **kwargs):
                seen.append((*_where, threading.current_thread().name))
                return _real(*args, **kwargs)

            monkeypatch.setattr(driver, name, _spy)
    return seen


def _run_protocol(session, tmp_path, steps) -> None:
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
    from tests.test_run_refusal_contract import _build_real_protocol

    runner = session.create_protocol_runner()
    files_written = threading.Event()
    runner.run_single_scan(
        protocol=_build_real_protocol(steps),
        sequence_name='lanes',
        parent_dir=str(tmp_path),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        callbacks={
            'run_complete': lambda **kw: None,
            'files_complete': lambda **kw: files_written.set(),
        },
    )
    assert files_written.wait(COMPLETION_TIMEOUT), 'the run never finished its files'
    outcome = runner.wait_for_completion(timeout=COMPLETION_TIMEOUT)
    assert outcome.status == 'completed', outcome


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        yield session
    finally:
        session.shutdown()


def _steps():
    from tests.test_run_refusal_contract import _make_single_step_protocol

    base = _make_single_step_protocol().step(idx=0)
    return base, [
        {**base, 'Name': 'A1_af', 'Label': 'A1_af', 'Auto_Focus': True, 'Gain': 3.0},
        {
            **base,
            'Name': 'A1_ag',
            'Label': 'A1_ag',
            'Auto_Gain': True,
            'Step Index': 1,
        },
    ]


def test_every_instrument_write_of_a_run_executes_on_its_lane(sim_session, tmp_path, monkeypatch):
    lanes = {
        'io': sim_session.io_executor.executor_name,
        'camera': sim_session.camera_executor.executor_name,
    }
    seen = _record_write_threads(sim_session.scope, monkeypatch)

    _run_protocol(sim_session, tmp_path, _steps()[1])

    kinds = {driver for driver, _, _ in seen}
    assert kinds == set(_WRITES), f'the run must write LEDs, motion and the camera; saw {kinds}'
    off_lane = [
        (driver, name, thread) for driver, name, thread in seen if thread != lanes[_LANE_OF[driver]]
    ]
    assert off_lane == [], (
        'every write the run makes -- its steps, its autofocus sweep, its cleanup -- '
        f'must execute on its lane worker; these did not: {off_lane}'
    )


def test_a_rejected_camera_setting_is_reported_and_the_run_goes_on(
    sim_session, tmp_path, monkeypatch
):
    from modules.notification_center import notifications

    rejected_gain = 7.5
    camera = sim_session.scope._camera_driver
    real_gain = camera.gain

    def _refuse_one_gain(value):
        if value == rejected_gain:
            return False
        return real_gain(value)

    monkeypatch.setattr(camera, 'gain', _refuse_one_gain)
    errors = []
    monkeypatch.setattr(notifications, 'error', lambda *a, **k: errors.append(a))
    base, _ = _steps()

    _run_protocol(sim_session, tmp_path, [{**base, 'Gain': rejected_gain}])

    images = [p for p in tmp_path.rglob('*') if p.suffix.lower() in ('.tif', '.tiff')]
    assert len(images) == 1, f'the step must still capture once; wrote {images}'
    assert any(a[0] == 'Camera' for a in errors), (
        f'the rejected gain must be reported to the user; notified {errors}'
    )


def test_the_safety_darken_does_not_wait_for_a_busy_io_lane(sim_session, monkeypatch):
    """An undecided LED end-state is darkened owner-blind at once, while the
    IO lane is still busy -- a move holds the worker until it arrives, and
    a darken queued behind it would leave the sample lit for as long."""
    scope = sim_session.scope
    layer = 'Blue'
    scope.illumination.led_on(layer, 10.0)
    assert scope.illumination.get_led_state(layer)['enabled'], 'precondition: lit'
    lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
    assert lease is not None
    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(side_effect=RuntimeError('cleanup died')))
    stub = _make_runner_stub(scope, lease=lease)
    # The owner-blind safety off, which says in the log that it bypassed
    # the lease: the darken a reader of a post-mortem looks for.
    forced = []
    real_force_off = scope.illumination.force_off
    monkeypatch.setattr(
        scope.illumination, 'force_off', lambda: forced.append(1) or real_force_off()
    )

    busy_release = threading.Event()
    busy_running = threading.Event()

    def _busy():
        busy_running.set()
        busy_release.wait(RESULT_TIMEOUT_S)

    sim_session.io_executor.put(IOTask(action=_busy))
    assert busy_running.wait(5.0), 'the IO lane never started the busy task'
    finished = threading.Event()

    def _cleanup():
        try:
            scr.SequencedCaptureRunner._cleanup_inner(
                stub, RunEnding('failed', 'run_loop_crashed', 'Protocol Crashed', 'died')
            )
        except RuntimeError as died:
            # The cleanup this test makes raise, so the undecided path runs.
            assert str(died) == 'cleanup died'
        finally:
            finished.set()

    threading.Thread(target=_cleanup, daemon=True).start()
    try:
        assert finished.wait(5.0), 'cleanup waited on the busy IO lane'
        assert not scope.illumination.get_led_state(layer)['enabled'], (
            'the undecided end-state must be dark before the busy lane frees'
        )
        assert forced == [1], 'the undecided end-state must be darkened by the safety off'
        assert busy_running.is_set() and not busy_release.is_set()
    finally:
        busy_release.set()


def test_a_camera_task_queued_as_run_is_pressed_is_refused_and_the_run_completes(lane_session):
    session, runner, tmp_path = lane_session
    hold = _LaneHold(session)
    ran = threading.Event()
    queued = session.camera_executor.put(IOTask(action=ran.set), return_future=True)
    assert queued is not None, 'the camera task was not queued before the run'

    outcome = runner.start_composite(sequence_name='queued_camera_task', parent_dir=str(tmp_path))
    hold.release()

    result = outcome.wait(timeout_s=RESULT_TIMEOUT_S)
    assert result is not None and result.status == 'completed', result
    with pytest.raises(HardwareCommandRefusedError):
        queued.result(timeout=RESULT_TIMEOUT_S)
    assert not ran.is_set(), 'a task queued before the run ran inside it'


def test_a_write_under_an_ended_run_is_refused_on_an_idle_scope():
    """An autofocus unwind can outlive cleanup's wait for it. Its late write
    is made under the run's taking after the run has released the scope,
    and nothing else holds it -- the write must still be refused, not move
    the stage or light the sample of a scope the run no longer has."""
    from modules.activity_claim import ActivityClaim, acting
    from modules.sequential_io_executor import SequentialIOExecutor

    claim = ActivityClaim()
    lane = SequentialIOExecutor(name='TEST_IO')
    lane.ask_claim(claim)
    lane.start()
    try:
        run = claim.try_claim('protocol')
        run.release()
        ran = threading.Event()
        outcome = {}

        def _late_write():
            with acting(run):
                try:
                    lane.call(IOTask(action=ran.set), 'late_autofocus_write', 2.0)
                except HardwareCommandRefusedError as refused:
                    outcome['refused'] = refused

        worker = threading.Thread(target=_late_write)
        worker.start()
        worker.join(5.0)
        assert not ran.is_set(), 'a write under an ended run reached the hardware'
        refused = outcome['refused']
        assert refused.reason == 'activity_ended', outcome
        assert str(refused) == (
            'The activity that sent this command has ended, so the command was not sent.'
        )
    finally:
        lane.shutdown()


def test_a_lease_released_dark_darkens_on_the_io_lane(sim_session, monkeypatch):
    """A lease released without leave_on darkens what it lit on the io lane,
    like every other LED write -- not on the releasing thread, where it
    would race the lane's own LED commands."""
    illumination = sim_session.scope.illumination
    threads = []
    real = illumination._leds_off_lit_by

    def _spy(lease):
        threads.append(threading.current_thread().name)
        return real(lease)

    monkeypatch.setattr(illumination, '_leds_off_lit_by', _spy)
    lease = illumination.acquire_led_lease('protocol', claim=held_run_claim())

    lease.release()

    assert threads == [sim_session.io_executor.executor_name], threads


def test_the_autofocus_dark_frame_retry_is_the_runs_own_grab(monkeypatch):
    """A dark frame mid-sweep is grabbed again through the public member,
    under the sweep's taking, and the retry is what gets scored."""
    import numpy as np

    from tests.af_drives import af_runner_and_scope, drive_af

    scored = []

    def _score(image):
        scored.append(float(np.mean(image)))
        return 7.0

    monkeypatch.setattr('modules.autofocus_functions.focus_function', _score)
    runner, scope = af_runner_and_scope()
    lit = np.full((40, 40), 50, dtype=np.uint8)
    frames = iter([np.zeros((40, 40), dtype=np.uint8)])
    scope.imaging.capture_and_wait.side_effect = lambda **kw: next(frames, lit)

    drive_af(runner)

    assert scored and scored[0] == 50.0, (
        f'the first position must be scored on the retry, not the dark frame; scored {scored}'
    )


@pytest.mark.parametrize('door', ['protocol_put', 'protocol_put_wait'])
def test_the_run_door_refuses_the_lender_of_a_borrowed_run(door):
    """A run borrowed inside a diagnostic holds the lane's protocol door. The
    diagnostic still holds the scope, but its own work is not the run's
    and is refused at the door until the run ends."""
    from modules.activity_claim import ActivityClaim, acting
    from modules.sequential_io_executor import SequentialIOExecutor

    claim = ActivityClaim()
    lane = SequentialIOExecutor(name='TEST_IO')
    lane.ask_claim(claim)
    lane.start()
    diagnostic = claim.try_claim('diagnostic')
    run = diagnostic.lend().try_claim('protocol', run_trigger_source='api_autofocus')
    lane.protocol_start(run)
    ran = threading.Event()
    try:
        with acting(diagnostic):
            if door == 'protocol_put':
                fut = lane.protocol_put(
                    IOTask(action=ran.set, silent_on_failure=True), return_future=True
                )
                with pytest.raises(HardwareCommandRefusedError):
                    fut.result(timeout=RESULT_TIMEOUT_S)
            else:
                refused = lane.protocol_put_wait(
                    IOTask(action=ran.set, silent_on_failure=True),
                    should_abort=lambda: False,
                    stall_timeout_s=1.0,
                )
                assert isinstance(refused, HardwareCommandRefusedError), refused
        with acting(run):
            ok = lane.protocol_put(IOTask(action=lambda: 'ran'), return_future=True)
        assert ok.result(timeout=RESULT_TIMEOUT_S) == 'ran'
        assert not ran.is_set(), "the lender's work ran through the run's door"
    finally:
        lane.protocol_end()
        run.release()
        diagnostic.release()
        lane.shutdown()


def test_a_step_darken_the_lane_refuses_falls_to_the_safety_off():
    from types import SimpleNamespace

    from modules.protocol_step_runner import ProtocolStepRunner

    illumination = MagicMock()
    parent = SimpleNamespace(
        LOGGER_NAME='test',
        _io_executor=MagicMock(),
        _scope=SimpleNamespace(illumination=illumination),
    )
    runner = ProtocolStepRunner(parent)

    runner.leds_off()
    illumination.leds_off.assert_called_once_with()
    illumination.force_off.assert_not_called()

    illumination.leds_off.side_effect = HardwareCommandRefusedError('activity_ended', 'leds_off')
    runner.leds_off()
    illumination.force_off.assert_called_once_with()
    parent._io_executor.protocol_put.assert_not_called()


def test_the_run_takes_a_standing_arm_through_the_public_member():
    from modules.protocol_state_machine import SequencedCaptureRunMode
    from tests.protocol_drives import bare_capture_runner

    runner = bare_capture_runner()
    arm = MagicMock(settings={'target_brightness': 0.5})
    runner._saved_camera_state = {'auto_gain_arm': arm}
    runner._run_mode = SequencedCaptureRunMode.FULL_PROTOCOL

    runner._take_auto_gain_arm_for_run()

    runner._scope.imaging.set_auto_gain.assert_called_once_with(False, {'target_brightness': 0.5})
