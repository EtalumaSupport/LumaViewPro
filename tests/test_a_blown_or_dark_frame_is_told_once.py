# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A blown or dark frame is kept, recorded on its frame's record, and told once.

The capture records what the frame was (``FrameRecord.saturated_fraction``,
``blown``, ``dark``) and tells no one: a soak at one warning per capture
logged 135,793 of them. Whoever took the capture tells it -- a run once per
step, counting each step's on its outcome (``captures.saturated``,
``captures.dark``) with one line at its end; a still once, itself.

Driven on the simulated scope; only the frames' pixels are replaced.
"""

import dataclasses
import datetime
import logging

import numpy as np
import pandas as pd
import pytest

from _lumascope.modules.protocol import Protocol
from tests.test_a_late_write_records_its_frame import _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.protocol_fixtures import protocol_of

WAIT_S = 60.0


def _frames_made(monkeypatch, camera, pixels_of):
    """Every frame the camera hands a capture carries ``pixels_of(frame)``.

    A stand-in by design: the simulator cannot be told to make a blown or a
    dark frame. The frame is still the camera's own -- its arrival number,
    depth and chunks -- with only its image swapped.
    """
    real = camera.grab_new_capture

    def made(timeout_s):
        frame = real(timeout_s)
        if frame is None:
            return None
        image = pixels_of(frame)
        image.flags.writeable = False
        return dataclasses.replace(frame, image=image)

    monkeypatch.setattr(camera, 'grab_new_capture', made)


def _blown(frame):
    return np.full_like(frame.image, (1 << frame.significant_bits) - 1)


def _dark(frame):
    return np.zeros_like(frame.image)


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


def _run(tmp_path, monkeypatch, caplog, steps, scans, pixels_of):
    """Run the protocol; return its outcome and what the run said at WARNING."""
    with (
        caplog.at_level(logging.WARNING),
        open_composite_session(headless_settings(tmp_path)) as (session, runner),
    ):
        _frames_made(monkeypatch, session.scope._camera_driver, pixels_of)
        pending = runner.start_protocol(
            protocol=_protocol(steps, scans), parent_dir=str(tmp_path / 'runs')
        )
        outcome = pending.result(timeout=WAIT_S)
        assert pending.files.result(timeout=WAIT_S) is not None, 'the run never finished'
    return outcome, [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_runs_blown_frames_are_told_once_per_step_and_counted(tmp_path, monkeypatch, caplog):
    steps = [_step('C1', 0, x=20.0, gain=1.0), _step('C2', 1, x=30.0, gain=1.0)]

    outcome, told = _run(tmp_path, monkeypatch, caplog, steps, scans=2, pixels_of=_blown)

    captures = outcome.captures
    assert captures.captured == 4, outcome
    assert [
        (s.step_index, s.step_name, s.count, s.first_scan_index) for s in captures.saturated
    ] == [(0, 'C1', 2, 0), (1, 'C2', 2, 0)], captures
    per_step = [line for line in told if 'saturated --' in line]
    assert len(per_step) == 2, per_step
    assert any('Step 1 (C1), scan 1' in line for line in per_step), per_step
    assert any('Step 2 (C2), scan 1' in line for line in per_step), per_step
    ends = [line for line in told if 'The run kept frames' in line]
    assert ends == ['[PROTOCOL] The run kept frames that were 4 saturated (steps: C1, C2)'], told


def test_a_runs_dark_frames_are_told_once_per_step_and_counted(tmp_path, monkeypatch, caplog):
    # The step's light is on, so a frame with no lit pixel is judged dark.
    steps = [_step('C1', 0, x=20.0, gain=1.0)]

    outcome, told = _run(tmp_path, monkeypatch, caplog, steps, scans=2, pixels_of=_dark)

    captures = outcome.captures
    assert [(s.step_index, s.step_name, s.count, s.first_scan_index) for s in captures.dark] == [
        (0, 'C1', 2, 0)
    ], captures
    assert captures.saturated == (), captures
    per_step = [line for line in told if 'the frame is dark' in line]
    assert len(per_step) == 1, per_step
    ends = [line for line in told if 'The run kept frames' in line]
    assert ends == ['[PROTOCOL] The run kept frames that were 2 dark (steps: C1)'], told


def test_a_step_is_told_once_a_run_though_its_scan_was_run_again(tmp_path, monkeypatch, caplog):
    """Step C0 is blown, its scan fails and is run again with C0 clean, then
    C0 is blown in the next scan: told once in the run, and counted once,
    for the scan whose capture stands."""
    from _lumascope.modules.exceptions import MoveWaitExpiredError
    from _lumascope.modules.run_events import RunEvents

    steps = [_step('C0', 0, x=20.0, gain=1.0), _step('C1', 1, x=30.0, gain=1.0)]
    attempts = []
    step_now = []

    def blown_in_attempts_1_and_3(frame):
        if step_now and step_now[-1] == 0 and len(attempts) in (1, 3):
            return _blown(frame)
        return frame.image.copy()

    with (
        caplog.at_level(logging.WARNING),
        open_composite_session(headless_settings(tmp_path)) as (session, runner),
    ):
        _frames_made(monkeypatch, session.scope._camera_driver, blown_in_attempts_1_and_3)
        protocols = session.scope.protocols
        real_targets = protocols.step_targets

        def step_targets(protocol, step_index, **kwargs):
            # A stand-in by design: the simulator cannot fail a scan and leave
            # the scope positioned and connected, the transient case.
            if len(attempts) == 1 and step_index == 1:
                raise MoveWaitExpiredError('X', 'still_moving')
            return real_targets(protocol, step_index, **kwargs)

        monkeypatch.setattr(protocols, 'step_targets', step_targets)
        pending = runner.start_protocol(
            protocol=_protocol(steps, 2),
            parent_dir=str(tmp_path / 'runs'),
            events=RunEvents(
                scan_started=lambda index, _left, _period: attempts.append(index),
                step_started=step_now.append,
            ),
        )
        outcome = pending.result(timeout=WAIT_S)
        assert pending.files.result(timeout=WAIT_S) is not None, 'the run never finished'
    told = [r.getMessage() for r in caplog.records if 'saturated --' in r.getMessage()]

    assert attempts == [0, 0, 1], attempts
    assert [
        (s.step_index, s.step_name, s.count, s.first_scan_index) for s in outcome.captures.saturated
    ] == [(0, 'C0', 1, 1)], outcome.captures
    assert len(told) == 1, told
    assert 'Step 1 (C0), scan 1' in told[0], told


def test_a_blown_still_is_told_once_by_the_still(tmp_path, monkeypatch, caplog):
    from tests.test_manual_capture_member import _capture, _open_session, _settings

    with _open_session(_settings(tmp_path)) as session:
        _frames_made(monkeypatch, session.scope._camera_driver, _blown)
        with caplog.at_level(logging.WARNING, logger='LVP.modules.manual_capture'):
            (still,) = _capture(session)

    assert still.file.is_file(), 'a blown still is saved'
    assert still.record.blown is True
    told = [r.getMessage() for r in caplog.records if 'The still is' in r.getMessage()]
    assert told == [
        '[ManualCapture] The still is 100% saturated -- likely over-exposure or a '
        'stale camera gain; the frame may be unusable.'
    ], told


@pytest.mark.slow
def test_a_saved_file_states_what_its_frame_was(tmp_path, monkeypatch):
    from _lumascope.modules.image_utils import read_file_record
    from tests.test_manual_capture_member import _files, _open_session, _settings

    with _open_session(_settings(tmp_path)) as session:
        # The still's light is on, so a frame with no lit pixel is judged dark.
        session.scope.illumination.led_on('BF', 50)
        _frames_made(monkeypatch, session.scope._camera_driver, _dark)
        paths = _files(session)

    saved = read_file_record(paths[0])
    assert (saved.get('dark'), saved.get('saturated_fraction')) == (True, 0.0), saved
