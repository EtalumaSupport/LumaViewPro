# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Stop that lands while start() is still setting the run up.

start() commits the run (RUNNING) under the run lock, then sets it up
outside the lock -- the run folder, the image writer, both lanes into run
mode -- and only then hands the loop to the protocol thread. A Stop is
accepted anywhere in that window: the run is live. The contract is the
same as for a Stop of a running run: the run ends 'aborted' / 'stopped',
nothing of it is left set up, and the person who pressed Stop is not told
the run failed.

Two injection points, the two shapes the window has: before the run folder
is made (the run's claim is still needed by the setup after it) and just
before the lanes enter run mode (the setup after it re-enters run mode).
"""

import threading
import time

from unittest.mock import MagicMock

import pytest

from modules import sequenced_capture_runner
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    COMPLETION_TIMEOUT,
    _make_autogain_settings,
    _make_image_capture_config,
    _make_multi_step_protocol,
    _make_tile_grid_steps,
    executor,
    executors,
    scope,
)


def _start_run(executor, tmp_path):
    """Start a scan and return the handle start() gave back, without
    waiting for it to be live: a run stopped during setup never is."""
    plan = executor.prepare(
        protocol=_make_multi_step_protocol(_make_tile_grid_steps(rows=6, cols=8)),
        run_trigger_source='scan',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='stop_during_start',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks={
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        },
    )
    return executor.start(plan)


def _stop_the_run_being_started(executor):
    executor._reset(executor._last_run())


def _inject_at_setup_run_dir(executor, monkeypatch):
    setup = executor._setup_run_dir

    def stop_then_setup():
        _stop_the_run_being_started(executor)
        setup()

    monkeypatch.setattr(executor, '_setup_run_dir', stop_then_setup)


def _inject_before_the_lanes_enter_run_mode(executor, monkeypatch):
    lane = executor.camera_executor
    protocol_start = lane.protocol_start

    def stop_then_protocol_start(*args, **kwargs):
        _stop_the_run_being_started(executor)
        return protocol_start(*args, **kwargs)

    monkeypatch.setattr(lane, 'protocol_start', stop_then_protocol_start)


@pytest.mark.parametrize(
    'inject',
    [_inject_at_setup_run_dir, _inject_before_the_lanes_enter_run_mode],
    ids=['at_setup_run_dir', 'before_the_lanes_enter_run_mode'],
)
def test_a_stop_during_setup_ends_the_run_stopped_with_nothing_left_set_up(
    executor, tmp_path, monkeypatch, centre_posts, inject
):
    # The test environment's logger is a mock, so its calls are read
    # directly; caplog would see nothing.
    engine_log = MagicMock()
    monkeypatch.setattr(sequenced_capture_runner, 'logger', engine_log)
    loop_entries = []
    run_loop = executor._run_loop_executor.run_loop

    def counted_run_loop(run):
        loop_entries.append(run)
        return run_loop(run)

    monkeypatch.setattr(executor._run_loop_executor, 'run_loop', counted_run_loop)
    inject(executor, monkeypatch)

    run = _start_run(executor, tmp_path)
    outcome = run.wait(timeout_s=COMPLETION_TIMEOUT)
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT), 'the stopped run never went idle'
    deadline = time.monotonic() + COMPLETION_TIMEOUT
    while executor.protocol_thread.is_running:
        assert time.monotonic() < deadline, 'a run loop is still running'
        time.sleep(0.01)

    assert outcome is not None, 'the stopped run never reported'
    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')
    assert loop_entries == [], 'the run loop ran for a run stopped before it began'
    assert not executor.camera_executor.is_protocol_running(), 'the camera lane is left in run mode'
    assert not executor._io_executor.is_protocol_running(), 'the IO lane is left in run mode'
    assert executor._activity_claim.holder is None, 'the scope is still held'
    errors = [str(call) for call in engine_log.error.call_args_list]
    assert errors == [], f'a Stop during setup logged an error: {errors}'
    told = [n.title for n in centre_posts if n.title == 'Run failed to start']
    assert told == [], 'a person who pressed Stop was told the run failed to start'


@pytest.mark.slow
def test_a_shutdown_during_a_later_runs_setup_leaves_the_unwind_to_start(
    executor, tmp_path, monkeypatch
):
    """force_reset unwinds on its own thread only a run whose loop ended
    without unwinding it. During the setup of a run that follows another,
    the loop it finds must be this run's (none yet), never the previous
    run's finished one, or the shutdown tears the run down mid-setup."""
    first = _start_run(executor, tmp_path / 'first')
    assert first.wait(timeout_s=COMPLETION_TIMEOUT) is not None
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT)
    assert executor.write_batch().wait_complete(COMPLETION_TIMEOUT)

    teardowns = []
    teardown = executor._cleanup_inner

    def counted(ending, run, after_end):
        teardowns.append(threading.current_thread().name)
        return teardown(ending, run, after_end)

    monkeypatch.setattr(executor, '_cleanup_inner', counted)
    lane = executor.camera_executor
    protocol_start = lane.protocol_start

    def shutdown_then_protocol_start(*args, **kwargs):
        executor.force_reset(reason='app shutdown')
        return protocol_start(*args, **kwargs)

    monkeypatch.setattr(lane, 'protocol_start', shutdown_then_protocol_start)

    second = _start_run(executor, tmp_path / 'second')
    outcome = second.wait(timeout_s=COMPLETION_TIMEOUT)

    assert outcome is not None
    assert (outcome.status, outcome.reason) == ('aborted', 'force_reset')
    assert teardowns == [threading.current_thread().name], (
        f'the run was torn down {len(teardowns)} times, on {teardowns}; only '
        'start() on this thread may unwind a run it is still setting up'
    )
    assert not executor.camera_executor.is_protocol_running(), 'the camera lane is left in run mode'
    assert executor._activity_claim.holder is None, 'the scope is still held'
