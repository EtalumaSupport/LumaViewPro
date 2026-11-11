# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a run keeps of its captures is bounded by its protocol, not by its length.

A run kept a record of every capture it took, and every video step it
recorded, until it ended: about 3 pymalloc blocks a saved file, so a
month-long run on an 8 GB laptop grew by gigabytes, and each capture walked
every record before it. It keeps its protocol's steps and the video steps
still finishing; a scan run again after a transient failure still counts each
of its captures once, and the run's end still says whether every video frame
was kept.

Driven end to end on the simulated scope.
"""

import contextlib
import datetime
import gc
import logging
import threading
import tracemalloc

import pandas as pd
import pytest

import _lumascope.modules.protocol_recording as protocol_recording
from _lumascope.modules.exceptions import MoveWaitExpiredError
from _lumascope.modules.protocol import Protocol
from _lumascope.modules.protocol_recording import ProtocolVideoStep
from _lumascope.modules.run_events import RunEvents
from tests.protocol_fixtures import protocol_of
from tests.test_a_late_write_records_its_frame import _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 120.0


def _protocol(steps, scans):
    period = datetime.timedelta(seconds=1.0)
    return protocol_of(
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(steps),
            'period': period,
            'duration': period * scans,
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _stills(count):
    return [_step(f'C{i}', i, x=20.0 + 10.0 * (i % 2), gain=1.0) for i in range(count)]


def _videos(count):
    steps = [_step(f'V{i}', i, x=20.0 + 10.0 * (i % 2), gain=1.0) for i in range(count)]
    for step in steps:
        step['Acquire'] = 'video'
        step['Video Config'] = {'duration': 0.4}
    return steps


def _run(runner, tmp_path, protocol, events):
    pending = runner.start_protocol(
        protocol=protocol, parent_dir=str(tmp_path / 'runs'), events=events
    )
    outcome = pending.result(timeout=WAIT_S)
    assert pending.files.result(timeout=WAIT_S) is not None, 'the run never finished its files'
    return outcome


def _writer_blocks():
    """The blocks the run's capture writer holds, traced to the line that allocated them."""
    return tracemalloc.take_snapshot().filter_traces(
        (tracemalloc.Filter(True, '*protocol_image_writer.py'),)
    )


def _blocks(snapshot):
    return sum(stat.count for stat in snapshot.statistics('filename'))


@pytest.mark.slow
def test_a_runs_held_captures_do_not_grow_with_its_scans(tmp_path, caplog):
    steps, scans = 8, 6
    held = {}

    def scan_ended(scan_index, _remaining, _period):
        held[scan_index] = _writer_blocks()

    # pytest's log capture keeps every record it is passed, each message
    # built on the writer's line: a capture's VERBOSE line would be counted
    # as the writer's. No handler keeps them in the app.
    caplog.set_level(logging.INFO, logger='LVP.protocol')

    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        tracemalloc.start()
        try:
            outcome = _run(
                runner,
                tmp_path,
                _protocol(_stills(steps), scans),
                RunEvents(scan_ended=scan_ended),
            )
        finally:
            tracemalloc.stop()

    assert outcome.status == 'completed', outcome
    assert (outcome.captures.asked, outcome.captures.captured) == (steps * scans, steps * scans)
    # From scan 1 the run holds every step's record of the scan in progress;
    # four scans more add nothing that stays. Before, two blocks a capture.
    grown = _blocks(held[scans - 1]) - _blocks(held[1])
    growing = [
        f'{stat.traceback}: {stat.count_diff:+d}'
        for stat in held[scans - 1].compare_to(held[1], 'lineno')
        if stat.count_diff > 0
    ]
    assert grown < steps, f'the writer grew {grown} blocks over 4 scans of {steps} steps: {growing}'


@pytest.mark.slow
def test_a_video_run_keeps_at_most_one_scans_video_steps(tmp_path):
    settings = headless_settings(tmp_path)
    settings['video_as_frames'] = True
    video_steps, scans = 2, 4
    alive = {}

    def scan_ended(scan_index, _remaining, _period):
        gc.collect()
        alive[scan_index] = sum(isinstance(o, ProtocolVideoStep) for o in gc.get_objects())

    with open_composite_session(settings) as (_session, runner):
        outcome = _run(
            runner,
            tmp_path,
            _protocol(_videos(video_steps), scans),
            RunEvents(scan_ended=scan_ended),
        )

    assert outcome.status == 'completed', outcome
    assert max(alive.values()) <= video_steps, f'video steps alive at each scan end: {alive}'


class _LastScanHeld(logging.Handler):
    """The last scan's frame writes wait until the run's end waits for them.

    So the run's end has a video step still saving, and states how long it
    waited, or that not every frame was kept. Released by the run's end
    saying it waits (its log line, the one signal it gives); bounded, so a
    run that never waits fails the test instead of hanging it. With
    ``lose_first``, the run's first frame write fails.
    """

    def __init__(self, monkeypatch, scans, *, lose_first=False):
        super().__init__()
        self.scans_started = 0
        self.end_waits = threading.Event()
        self.lost = threading.Event()
        real = protocol_recording.write_frame

        def write_frame(*args, **kwargs):
            # A stand-in by design: the simulator cannot fail one frame's
            # write, nor hold one until a run's end.
            if lose_first and not self.lost.is_set():
                self.lost.set()
                raise OSError('the disk refused the frame')
            if self.scans_started == scans:
                assert self.end_waits.wait(WAIT_S), 'the run ended without waiting for its video'
            return real(*args, **kwargs)

        monkeypatch.setattr(protocol_recording, 'write_frame', write_frame)

    def scan_started(self, _scan_index, _remaining, _period):
        self.scans_started += 1

    def emit(self, record):
        if 'Waiting for video write drain' in record.getMessage():
            self.end_waits.set()

    @contextlib.contextmanager
    def listening(self):
        lvp = logging.getLogger('LVP')
        lvp.addHandler(self)
        try:
            yield RunEvents(scan_started=self.scan_started)
        finally:
            lvp.removeHandler(self)


def test_a_video_step_read_before_it_records_is_still_waited_for(tmp_path, monkeypatch):
    """The Session's live work is read from any thread while a run records,
    a step's prologue included, when it is not yet busy; the step read
    there is still waited for at the run's end and counted as kept."""
    settings = headless_settings(tmp_path)
    settings['video_as_frames'] = True
    held = _LastScanHeld(monkeypatch, scans=2)
    with open_composite_session(settings) as (session, runner), held.listening() as events:
        # Polled on every camera frame, as a REST client or the GUI polls it.
        session.scope.imaging.add_frame_listener(
            lambda _frame: session.live_work, name='live_work_poll'
        )
        outcome = _run(runner, tmp_path, _protocol(_videos(2), 2), events)

    assert held.end_waits.is_set()
    assert outcome.status == 'completed', outcome
    assert outcome.video_save_wait_s is not None, outcome


def test_a_video_step_that_lost_a_frame_is_on_the_runs_end_after_it_finished(tmp_path, monkeypatch):
    """A step that lost a frame in the first scan has finished long before
    the run's end waits for the last scan's; the run's end still says not
    every frame was kept."""
    settings = headless_settings(tmp_path)
    settings['video_as_frames'] = True
    held = _LastScanHeld(monkeypatch, scans=2, lose_first=True)
    with open_composite_session(settings) as (_session, runner), held.listening() as events:
        outcome = _run(runner, tmp_path, _protocol(_videos(1), 2), events)

    assert held.lost.is_set()
    assert held.end_waits.is_set()
    assert outcome.status == 'completed', outcome
    assert outcome.video_save_wait_s is None, outcome


def test_a_scan_run_again_counts_each_capture_once(tmp_path, monkeypatch):
    """A transient fault mid-scan re-runs the scan from its first step: the
    steps it captured before the fault are taken again, and counted once."""
    attempts = []
    faulted = threading.Event()

    def scan_started(scan_index, _remaining, _period):
        attempts.append(scan_index)

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        protocols = session.scope.protocols
        real_targets = protocols.step_targets

        def step_targets(protocol, step_index, **kwargs):
            # A stand-in by design: the simulator cannot fail a scan and leave
            # every axis positioned and every board connected, the transient
            # case; a move's wait running out is one such fault.
            if len(attempts) == 2 and step_index == 2 and not faulted.is_set():
                faulted.set()
                raise MoveWaitExpiredError('X', 'still_moving')
            return real_targets(protocol, step_index, **kwargs)

        monkeypatch.setattr(protocols, 'step_targets', step_targets)
        outcome = _run(
            runner,
            tmp_path,
            _protocol(_stills(3), 2),
            RunEvents(scan_started=scan_started),
        )

    assert faulted.is_set()
    assert attempts == [0, 1, 1], attempts
    assert outcome.status == 'completed', outcome
    assert (outcome.captures.asked, outcome.captures.captured) == (6, 6), outcome.captures
    assert outcome.captures.failed == (), outcome.captures
