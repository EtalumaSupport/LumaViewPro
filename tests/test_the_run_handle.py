# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's operation: what every start returns, and all a caller needs about its run.

Its answers are about the run that returned it, never whichever run is
live: its folder stays its own after another run starts, its progress is
None once it has ended. Its result is given once the run no longer holds the
scope, so the caller's next act needs no second wait.
"""

import ast
import threading

from tests.ast_seams import iter_package_modules
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 60.0


def _scan(runner, run_parent, name):
    return runner.start_single_scan(
        protocol=_protocol([_step(name, 0, x=20.0, gain=1.0)]),
        parent_dir=str(run_parent),
    )


def test_a_woken_waiter_finds_the_scope_free_and_starts_the_next_run(tmp_path, monkeypatch):
    """The plugin's case: autofocus, then at once the next autofocus.

    The claim release is slowed so that a wait that returned when the
    outcome settled, before the run let go of the scope, is caught.
    """
    import time

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        engine = session.sequenced_capture_runner
        release = engine._release_activity_claim

        def slow_release():
            time.sleep(0.3)
            release()

        monkeypatch.setattr(engine, '_release_activity_claim', slow_release)

        first = runner.start_autofocus(layer='BF')
        assert first.result(timeout=WAIT_S) is not None
        assert session.activity_claim.owner is None, (
            f'the wait returned while {session.activity_claim.owner!r} still held the scope'
        )
        second = runner.start_autofocus(layer='BF')
        assert second.result(timeout=WAIT_S) is not None


def test_a_run_keeps_its_own_folder_after_the_next_run_starts(tmp_path):
    run_parent = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        first = _scan(runner, run_parent, 'C1')
        assert first.result(timeout=WAIT_S) is not None
        assert session.sequenced_capture_runner.write_batch().wait_complete(WAIT_S)
        first_dir = first.progress().run_dir

        second = _scan(runner, run_parent, 'C2')
        assert second.result(timeout=WAIT_S) is not None

    assert first_dir is not None and first_dir.is_dir()
    second_dir = second.progress().run_dir
    assert second_dir is not None and second_dir != first_dir
    assert first.progress().run_dir == first_dir, "a run answered with the next run's folder"


class _RunLockThatRefusesReentry:
    """The engine's run lock, raising where the real one would wait forever.

    The real lock is not reentrant, so a read that takes it while its own
    call already holds it never returns; this one says so instead.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._holder = None

    def __enter__(self):
        if self._holder == threading.get_ident():
            raise AssertionError('the run lock was taken by the thread already holding it')
        self._lock.acquire()
        self._holder = threading.get_ident()
        return self

    def __exit__(self, *_exc):
        self._holder = None
        self._lock.release()


def test_a_live_runs_progress_returns(tmp_path, monkeypatch):
    """A live run's progress is read; it never waits on the run lock it holds.

    The run is held live in its teardown, before it lets go of the scope.
    """
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        engine = session.sequenced_capture_runner
        monkeypatch.setattr(engine, '_run_lock', _RunLockThatRefusesReentry())
        release = engine._release_activity_claim
        held = threading.Event()
        let_go = threading.Event()

        def held_release():
            held.set()
            let_go.wait(WAIT_S)
            release()

        monkeypatch.setattr(engine, '_release_activity_claim', held_release)

        run = _scan(runner, tmp_path / 'runs', 'C1')
        try:
            assert held.wait(WAIT_S), 'the run never reached its teardown'
            progress = run.progress()
        finally:
            let_go.set()
        assert run.result(timeout=WAIT_S) is not None

    assert progress.is_live is True
    assert (progress.step_number, progress.num_steps) == (1, 1)


def test_an_ended_runs_progress_is_none(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        run = _scan(runner, tmp_path / 'runs', 'C1')
        assert run.result(timeout=WAIT_S) is not None

    progress = run.progress()
    assert not progress.is_live
    assert (
        progress.step_number,
        progress.num_steps,
        progress.remaining_scans,
        progress.period_s,
    ) == (
        None,
        None,
        None,
        None,
    )


_ENGINE_RUN_MEMBERS = {
    '_progress_of',
    '_last_run',
    '_step_number',
    '_num_steps',
    '_remaining_scans',
    '_protocol_period_s',
}


def test_the_gui_asks_its_runs_through_their_operations():
    """No ui/ file asks the engine about a run, or stops one through it.

    The GUI holds the operation each start returned; every question about
    that run and its Stop go through it. force_reset, app close's stop of
    whatever is live, holds no run and stays the engine's.
    """
    offenders = []
    for rel_path, tree in iter_package_modules(('ui',)):
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name = node.func.attr
            receiver = ast.unparse(node.func.value)
            asks_the_engine = name in _ENGINE_RUN_MEMBERS and (
                'runner' in receiver or 'engine' in receiver
            )
            stops_through_the_engine = name == '_stop_run' and (
                'runner' in receiver or 'engine' in receiver
            )
            if asks_the_engine or stops_through_the_engine:
                offenders.append(f'{rel_path}:{node.lineno} {receiver}.{name}()')
    assert offenders == [], offenders
