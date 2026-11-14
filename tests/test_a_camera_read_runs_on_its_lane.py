# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera read runs on the camera's lane, admitted whatever holds the scope.

Every published read that goes to the camera -- the fresh ``read_*`` reads,
the diagnostics' temperature and link reads, the snapshot a run or an
autofocus restores -- is one task on the camera lane, so it never lands
inside a camera write, and it skips the claim, so a run, a home or a control
session does not refuse it. It waits behind what is queued there, up to the
restore's bound, so a read put behind a slow camera command answers once that
command is done rather than giving up at the old 5 s.

Driven on the simulated scope.
"""

import threading
import time

import pytest

from _lumascope.modules.lumascope_api.camera import CameraAPI
from _lumascope.modules.sequential_io_executor import IOTask

from tests.test_a_late_write_records_its_frame import _step
from tests.test_a_run_reports_what_it_captured import _run
from tests.test_composite_run_e2e import headless_settings, open_composite_session

# A bound on every wait here, far past what the simulator takes.
WAIT_S = 30.0


def _until(condition, what: str) -> None:
    """Wait for ``condition``, which no event announces: a task queued on a lane."""
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, f'never: {what}'
        time.sleep(0.01)


def _reading(read) -> tuple[threading.Thread, dict]:
    """``read`` on a thread of its own, its answer or its raise kept."""
    out = {}

    def run():
        try:
            out['answer'] = read()
        except BaseException as ex:
            out['raised'] = ex

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def _lane_held(session) -> threading.Event:
    """A camera lane running a task that ends when the returned event is set."""
    gate = threading.Event()
    session.camera_executor.put(IOTask(action=lambda: gate.wait(WAIT_S)))
    _until(lambda: session.camera_executor.running_task is not None, 'the held task runs')
    return gate


def test_a_fresh_read_waits_for_the_lanes_task_and_is_not_refused_by_a_control_session(
    sim_session,
):
    gate = _lane_held(sim_session)
    session_id = sim_session.take_control('bench')
    try:
        reader, out = _reading(sim_session.scope.camera.read_gain_db)
        _until(lambda: sim_session.camera_executor.queue_size() == 1, 'the read queues')
        assert reader.is_alive(), 'the read answered with the lane running another task'
        gate.set()
        reader.join(WAIT_S)
    finally:
        gate.set()
        sim_session.end_control(session_id)

    assert 'raised' not in out, out
    assert out['answer'] == sim_session.scope.camera.gain_db


@pytest.mark.slow
def test_a_camera_read_behind_a_task_longer_than_five_seconds_answers(sim_session):
    gate = _lane_held(sim_session)
    try:
        reader, out = _reading(sim_session.scope.diagnostics.read_camera_temperatures_degc)
        _until(lambda: sim_session.camera_executor.queue_size() == 1, 'the read queues')
        # Past the 5 s the diagnostics' reads once waited.
        time.sleep(5.5)
        gate.set()
        reader.join(WAIT_S)
    finally:
        gate.set()

    assert 'raised' not in out, out
    assert out['answer'] is not None


def test_a_fresh_read_during_a_run_is_answered_and_leaves_the_run_alone(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        stop = threading.Event()
        answers, raised = [], []

        def poll():
            while not stop.is_set():
                try:
                    answers.append(session.scope.camera.read_gain_db())
                except BaseException as ex:
                    raised.append(ex)
                    return

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        try:
            outcome = _run(
                runner,
                tmp_path / 'runs',
                [_step(f'C{i}', i, x=20.0 + 10 * i, gain=1.0) for i in range(2)],
            )
        finally:
            stop.set()
            poller.join(WAIT_S)

    assert (outcome.status, outcome.reason) == ('completed', 'completed'), outcome
    assert raised == []
    assert answers


def test_the_snapshot_reads_on_the_lane_so_a_frame_write_waits_for_it(sim_session):
    """A frame size written while the snapshot reads the camera lands after it, so the
    size held is the size the camera delivers, never the one the snapshot read before."""
    scope = sim_session.scope
    camera = scope.camera
    driver = scope._camera_driver
    real_read = driver.get_frame_size
    read_on = []
    reading = threading.Event()
    release = threading.Event()

    # A stand-in by design: a camera read held mid-flight is not a fault the
    # simulator offers (its read faults fail, they do not wait), a simulator gap.
    def held_first_read():
        if not reading.is_set():
            read_on.append(threading.current_thread())
            reading.set()
            release.wait(WAIT_S)
        return real_read()

    driver.get_frame_size = held_first_read
    try:
        saver, saved = _reading(lambda: camera.save_camera_state('test'))
        assert reading.wait(WAIT_S), 'the snapshot never read the frame size'
        writer, written = _reading(lambda: camera.set_frame_size(400, 300))
        _until(
            lambda: sim_session.camera_executor.queue_size() == 1 or not writer.is_alive(),
            'the frame write queues or lands',
        )
        release.set()
        saver.join(WAIT_S)
        writer.join(WAIT_S)
    finally:
        release.set()
        driver.get_frame_size = real_read

    assert 'raised' not in saved and 'raised' not in written, (saved, written)
    assert read_on[0] is not threading.current_thread()
    assert read_on[0] is not saver, 'the snapshot read the camera on its caller'
    assert camera.frame_size == camera.read_frame_size()


def test_a_snapshot_past_its_bound_ends_the_run_start_camera_lane_stalled(tmp_path, monkeypatch):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        driver = session.scope._camera_driver
        real_read = driver.get_frame_size
        stalled = threading.Event()
        stalled.set()

        # A stand-in by design: a camera read that hangs past its bound is not a
        # fault the simulator offers, a simulator gap.
        def slow_read():
            if stalled.is_set():
                time.sleep(1.5)
            return real_read()

        monkeypatch.setattr(CameraAPI, '_CAMERA_RESTORE_TIMEOUT_S', 0.5)
        driver.get_frame_size = slow_read
        try:
            outcome = _run(
                runner,
                tmp_path / 'runs',
                [_step('C0', 0, x=20.0, gain=1.0)],
            )
        finally:
            stalled.clear()
            # The snapshot the run gave up on is still running: the session
            # closes once it has ended, so its lane leaves nothing behind.
            _until(lambda: not session.camera_executor.is_busy(), 'the snapshot ends')
            driver.get_frame_size = real_read

    assert (outcome.status, outcome.reason) == ('failed_at_start', 'camera_lane_stalled'), outcome


def test_a_scope_disconnected_under_the_run_starts_snapshot_is_a_disconnect(tmp_path):
    """The snapshot's lane closing under it is the scope disconnecting, never a stalled camera."""
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        driver = session.scope._camera_driver
        real_read = driver.get_frame_size
        unplugging = threading.Event()

        # A stand-in by design: a disconnect landing inside a camera read is
        # not a fault the simulator offers, a simulator gap.
        def read_as_the_scope_disconnects():
            if not unplugging.is_set():
                unplugging.set()
                threading.Thread(target=session.scope.disconnect, daemon=True).start()
                _until(lambda: session.camera_executor.pending_shutdown, 'the lane closes')
            return real_read()

        driver.get_frame_size = read_as_the_scope_disconnects
        outcome = _run(runner, tmp_path / 'runs', [_step('C0', 0, x=20.0, gain=1.0)])

    assert unplugging.is_set(), 'the run never read the camera'
    assert (outcome.status, outcome.reason) == ('failed_at_start', 'hardware_disconnected'), outcome
