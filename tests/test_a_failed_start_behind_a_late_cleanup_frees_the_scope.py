# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run started behind the previous run's late cleanup runs, or is unwound.

A finished run's loop runs one more cleanup pass on its way out -- the
safety net -- after the run is already IDLE and its claim released. A run
started in that gap is admitted and takes the claim. Before, the protocol
thread then refused its dispatch because the previous loop had not
returned, and its failed-start cleanup, taking the cleanup lock without
waiting, found the late pass holding it and was skipped: the caller was
never told, the runner stayed live and the scope stayed held until restart.

Now the run waits for the previous loop to return and runs; and a start
that fails for its own reason in that gap is still unwound.
"""

import threading
import time

import modules.sequenced_capture_runner as sequenced_capture_runner
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0
_SAFETY_NET = 'The run loop exited without cleaning up.'


def _run(runner, run_parent, name):
    return runner.run_single_scan(
        protocol=_protocol([_step(name, 0, x=20.0, gain=1.0)]),
        parent_dir=str(run_parent),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )


def _hold_the_late_pass(monkeypatch):
    """Hold the first run's safety-net pass inside the cleanup lock."""
    in_late_pass = threading.Event()
    release = threading.Event()
    real_inner = sequenced_capture_runner.SequencedCaptureRunner._cleanup_inner

    def _held(self, ending, run):
        if ending.message == _SAFETY_NET and not in_late_pass.is_set():
            in_late_pass.set()
            release.wait(WAIT_S)
        return real_inner(self, ending, run)

    monkeypatch.setattr(sequenced_capture_runner.SequencedCaptureRunner, '_cleanup_inner', _held)
    return in_late_pass, release


def _wait_until_drained(session):
    deadline = time.monotonic() + WAIT_S
    while session.protocol_files_draining and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not session.protocol_files_draining


def _first_run_in_its_late_pass(session, runner, run_parent, in_late_pass):
    first = _run(runner, run_parent, 'A1')
    assert first.wait(timeout_s=WAIT_S) is not None
    assert in_late_pass.wait(WAIT_S), "the first run's late cleanup pass never ran"
    # A run is refused while the last one's files drain; they land on their
    # own lane while the late pass is held.
    _wait_until_drained(session)


def _scope_left(session, runner, outcome):
    told = outcome.wait(timeout_s=10.0)
    idle = runner.sequenced_capture_runner.wait_for_run_idle(10.0)
    status = told.status if told is not None else None
    return status, idle, session.activity_claim.holder


def test_a_run_started_behind_the_late_pass_runs(tmp_path, monkeypatch):
    in_late_pass, release = _hold_the_late_pass(monkeypatch)
    run_parent = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        _first_run_in_its_late_pass(session, runner, run_parent, in_late_pass)

        second = _run(runner, run_parent, 'B1')
        release.set()
        status, idle, holder = _scope_left(session, runner, second)

        assert (status, idle, holder) == ('completed', True, None), (
            f'the second run: status {status!r}, runner idle {idle}, scope held by {holder!r}'
        )
        _wait_until_drained(session)
        third = _run(runner, run_parent, 'C1')
        result = third.wait(timeout_s=WAIT_S)
        assert result is not None and result.status == 'completed', result


def test_a_start_that_fails_behind_the_late_pass_is_unwound(tmp_path, monkeypatch):
    in_late_pass, release = _hold_the_late_pass(monkeypatch)
    run_parent = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        _first_run_in_its_late_pass(session, runner, run_parent, in_late_pass)
        engine = runner.sequenced_capture_runner

        def _no_folder():
            raise OSError('save folder vanished')

        monkeypatch.setattr(engine, '_setup_run_dir', _no_folder)
        # The failed start's unwind waits for the held pass, so the start
        # does not return until the pass is released.
        started = []
        starter = threading.Thread(
            target=lambda: started.append(_run(runner, run_parent, 'B1')), daemon=True
        )
        starter.start()
        time.sleep(0.3)
        release.set()
        starter.join(WAIT_S)
        assert started, 'the failed start never returned'
        status, idle, holder = _scope_left(session, runner, started[0])

        assert (status, idle, holder) == ('failed_at_start', True, None), (
            f'the failed start: status {status!r}, runner idle {idle}, scope held by {holder!r}'
        )


def test_a_cleanup_that_raises_still_takes_its_lanes_out_of_run_mode(tmp_path, monkeypatch):
    """The run's own cleanup ends its lanes' run modes on every path out.

    A later pass may not: once the run is IDLE its lanes may be a
    successor's. So a cleanup that raised before ending them must end them
    itself, or the IO and CAMERA workers serve a run queue nothing fills and
    every later move and camera task starves.
    """

    def _raising_cleanup(**kwargs):
        raise RuntimeError('cleanup fell over before ending the lanes')

    monkeypatch.setattr(sequenced_capture_runner, 'run_cleanup', _raising_cleanup)
    run_parent = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        first = _run(runner, run_parent, 'A1')
        assert first.wait(timeout_s=WAIT_S) is not None
        assert runner.sequenced_capture_runner.wait_for_run_idle(WAIT_S)

        in_run_mode = [
            lane.name
            for lane in (session.io_executor, session.camera_executor)
            if lane.is_protocol_running() and not lane.protocol_finish.is_set()
        ]
        assert not in_run_mode, f'left in run mode after the run ended: {in_run_mode}'

        monkeypatch.undo()
        _wait_until_drained(session)
        second = _run(runner, run_parent, 'B1')
        result = second.wait(timeout_s=WAIT_S)
        assert result is not None and result.status == 'completed', result


def test_a_run_started_while_the_late_pass_decides_is_not_torn_down(tmp_path, monkeypatch):
    """The late pass reads "is this run the runner's" and "is it live" as
    one answer. Read apart, a successor started between the two reads is
    both, and the late pass tears it down as the run it came to end."""
    runner_cls = sequenced_capture_runner.SequencedCaptureRunner
    real_inner = runner_cls._cleanup_inner
    real_live = runner_cls._is_run_live
    late = {}
    in_gap = threading.Event()
    successor_committed = threading.Event()

    def _inner(self, ending, run):
        if ending.message == _SAFETY_NET and 'thread' not in late:
            late['thread'] = threading.current_thread()
        return real_inner(self, ending, run)

    def _live(self):
        if late.get('thread') is threading.current_thread() and not in_gap.is_set():
            in_gap.set()
            successor_committed.wait(5.0)
        return real_live(self)

    monkeypatch.setattr(runner_cls, '_cleanup_inner', _inner)
    monkeypatch.setattr(runner_cls, '_is_run_live', _live)
    run_parent = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        engine = runner.sequenced_capture_runner
        first = _run(runner, run_parent, 'A1')
        assert first.wait(timeout_s=WAIT_S) is not None
        assert in_gap.wait(WAIT_S), "the first run's late pass never asked whether it was live"
        _wait_until_drained(session)

        started = []
        threading.Thread(
            target=lambda: started.append(_run(runner, run_parent, 'B1')), daemon=True
        ).start()
        # Committed once the runner names the successor's outcome -- or,
        # where the late pass holds the lock start() commits under, not
        # until that pass has decided.
        deadline = time.monotonic() + 1.0
        while engine.run_outcome() is first and time.monotonic() < deadline:
            time.sleep(0.01)
        successor_committed.set()

        deadline = time.monotonic() + WAIT_S
        while not started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started, 'the successor never returned from its start'
        status, idle, holder = _scope_left(session, runner, started[0])

        assert (status, idle, holder) == ('completed', True, None), (
            f'the successor: status {status!r}, runner idle {idle}, scope held by {holder!r}'
        )
