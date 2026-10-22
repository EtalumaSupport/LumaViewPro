# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A blown or dark frame is kept, recorded on its frame's record, and told once.

The capture records what the frame was (``FrameRecord.saturated_fraction``,
``blown``, ``dark``) and tells no one: a soak at one warning per capture
logged 135,793 of them. Whoever took the capture tells it -- a run once per
step, counting the rest on its outcome (``captures.saturated``,
``captures.dark``) with one line at its end; a still once, itself.

Driven on the simulated scope; only the frames' pixels are replaced.
"""

import contextlib
import dataclasses
import datetime
import logging

import numpy as np
import pandas as pd

from modules.protocol import Protocol
from tests.test_a_late_write_records_its_frame import REPO_ROOT, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

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


@contextlib.contextmanager
def _warnings():
    """What the run's two loggers say at WARNING while the block runs.

    ``caplog`` cannot see them: the suite replaces ``lvp_logger`` with a
    stand-in (``conftest.install_mock_deps``). Wrapping the one call is
    observation only; the log line is the product here.
    """
    import lvp_logger

    lines = []
    loggers = (lvp_logger.logger, lvp_logger.protocol_logger)
    originals = [log.warning for log in loggers]
    for log in loggers:
        log.warning = lambda msg, *a, **kw: lines.append(str(msg))
    try:
        yield lines
    finally:
        for log, original in zip(loggers, originals, strict=True):
            log.warning = original


def _protocol(steps, scans):
    period = datetime.timedelta(seconds=1.0)
    return Protocol(
        tiling_configs_file_loc=REPO_ROOT / 'data' / 'tiling.json',
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


def _run(tmp_path, monkeypatch, steps, scans, pixels_of):
    with (
        _warnings() as told,
        open_composite_session(headless_settings(tmp_path)) as (session, runner),
    ):
        _frames_made(monkeypatch, session.scope._camera_driver, pixels_of)
        pending = runner.start_protocol(
            protocol=_protocol(steps, scans), parent_dir=str(tmp_path / 'runs')
        )
        outcome = pending.result(timeout=WAIT_S)
        assert pending.files.result(timeout=WAIT_S) is not None, 'the run never finished'
    return outcome, told


def test_a_runs_blown_frames_are_told_once_per_step_and_counted(tmp_path, monkeypatch):
    steps = [_step('C1', 0, x=20.0, gain=1.0), _step('C2', 1, x=30.0, gain=1.0)]

    outcome, told = _run(tmp_path, monkeypatch, steps, scans=2, pixels_of=_blown)

    captures = outcome.captures
    assert captures.captured == 4, outcome
    assert sorted((c.scan_index, c.step_name) for c in captures.saturated) == [
        (0, 'C1'),
        (0, 'C2'),
        (1, 'C1'),
        (1, 'C2'),
    ], captures
    per_step = [line for line in told if 'saturated --' in line]
    assert len(per_step) == 2, per_step
    assert any('Step 1 (C1), scan 1' in line for line in per_step), per_step
    assert any('Step 2 (C2), scan 1' in line for line in per_step), per_step
    ends = [line for line in told if 'The run kept frames' in line]
    assert ends == ['[PROTOCOL] The run kept frames that were 4 saturated (steps: C1, C2)'], told


def test_a_runs_dark_frames_are_told_once_per_step_and_counted(tmp_path, monkeypatch):
    # The step's light is on, so a frame with no lit pixel is judged dark.
    steps = [_step('C1', 0, x=20.0, gain=1.0)]

    outcome, told = _run(tmp_path, monkeypatch, steps, scans=2, pixels_of=_dark)

    captures = outcome.captures
    assert [(c.scan_index, c.step_name) for c in captures.dark] == [(0, 'C1'), (1, 'C1')], captures
    assert captures.saturated == (), captures
    per_step = [line for line in told if 'the frame is dark' in line]
    assert len(per_step) == 1, per_step
    ends = [line for line in told if 'The run kept frames' in line]
    assert ends == ['[PROTOCOL] The run kept frames that were 2 dark (steps: C1)'], told


def test_a_blown_still_is_told_once_by_the_still(tmp_path, monkeypatch, caplog):
    from tests.test_manual_capture_member import _files, _open_session, _settings

    with _open_session(_settings(tmp_path)) as session:
        _frames_made(monkeypatch, session.scope._camera_driver, _blown)
        with caplog.at_level(logging.WARNING, logger='LVP.modules.manual_capture'):
            paths = _files(session)
        record = session.scope.imaging.last_capture_info['frame_record']

    assert paths and paths[0].is_file(), 'a blown still is saved'
    assert record.blown is True
    told = [r.getMessage() for r in caplog.records if 'The still is' in r.getMessage()]
    assert told == [
        '[ManualCapture] The still is 100% saturated -- likely over-exposure or a '
        'stale camera gain; the frame may be unusable.'
    ], told


def test_a_saved_file_states_what_its_frame_was(tmp_path, monkeypatch):
    from modules.image_utils import read_postproc_input_metadata
    from tests.test_manual_capture_member import _files, _open_session, _settings

    with _open_session(_settings(tmp_path)) as session:
        # The still's light is on, so a frame with no lit pixel is judged dark.
        session.scope.illumination.led_on('BF', 50)
        _frames_made(monkeypatch, session.scope._camera_driver, _dark)
        paths = _files(session)

    saved = read_postproc_input_metadata(paths[0])
    assert (saved.get('dark'), saved.get('saturated_fraction')) == (True, 0.0), saved
