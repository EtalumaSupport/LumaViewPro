# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each reader of a failed capture keeps its own policy, with the true reason.

A capture that gets no frame raises ``CaptureFailedError``; a camera that is
gone raises the ``not_connected`` refusal. A protocol run still counts a
failure as a strike, now naming the capture's own reason on its row; a
removal ends the run ``hardware_disconnected`` in words naming the camera.
Autofocus retries a capture failure and then reports it as itself, and
keeps a dark frame whose retry gets none. The bandwidth test counts its
failures by reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.exceptions import CaptureFailedError
from modules.run_events import RunEvents
from tests.test_a_late_write_records_its_frame import _step
from tests.test_a_run_reports_what_it_captured import _run
from tests.test_an_autofocus_scores_a_frame_made_at_its_z import (
    FOCAL_Z_UM,
    _autofocus,
    _session_over_the_focus_target,
)
from tests.test_composite_run_e2e import headless_settings, open_composite_session


def _steps(n):
    return [_step(f'C{i}', i, x=20.0 + 10 * i, gain=1.0) for i in range(n)]


class TestAProtocolRun:
    def test_three_stalled_captures_end_it_each_row_naming_no_frame(self, tmp_path):
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            session.scope.simulation.stall_stream(0.0, 600.0)
            outcome = _run(runner, tmp_path / 'runs', _steps(4))

        assert (outcome.status, outcome.reason) == ('failed', 'camera_failure'), outcome
        assert [failed.cause for failed in outcome.captures.failed] == [
            str(CaptureFailedError('no_frame'))
        ] * 3

    def test_a_removed_camera_ends_it_in_words_naming_the_camera(self, tmp_path):
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):

            def step_started(step_index):
                # Past the run's start, which refuses a missing camera.
                if step_index == 1:
                    session.scope.simulation.remove_camera()

            outcome = _run(
                runner, tmp_path / 'runs', _steps(3), events=RunEvents(step_started=step_started)
            )

        assert (outcome.status, outcome.reason) == ('failed', 'hardware_disconnected'), outcome
        assert 'camera' in outcome.message.lower(), outcome.message


class TestAnAutofocus:
    @pytest.mark.slow
    def test_a_stalled_camera_is_reported_as_no_frame_after_its_retries(self):
        with _session_over_the_focus_target() as session:
            session.scope.simulation.stall_stream(0.0, 600.0)
            with pytest.raises(CaptureFailedError) as failed:
                _autofocus(session)

        assert failed.value.reason == 'no_frame'

    def test_a_dark_frame_whose_retry_gets_none_is_kept_and_the_sweep_focuses(self):
        # The sweep's first capture, at z_min, far from the focus.
        dark_at = 1
        with _session_over_the_focus_target() as session:
            imaging = session.scope.imaging
            capture = imaging.capture_and_wait
            calls = []

            # A stand-in by design: the simulator cannot deliver a dark frame
            # and then no frame on the next capture, in that order; its
            # stall is timed, not counted.
            def dark_then_none(*args, **kwargs):
                calls.append(len(calls) + 1)
                if len(calls) == dark_at:
                    return np.zeros_like(capture(*args, **kwargs))
                if len(calls) == dark_at + 1:
                    raise CaptureFailedError('no_frame')
                return capture(*args, **kwargs)

            imaging.capture_and_wait = dark_then_none
            focus = _autofocus(session)

        assert len(calls) > dark_at + 1, 'the dark frame and its failed retry were never served'
        assert focus == pytest.approx(FOCAL_Z_UM, abs=50.0)


@pytest.mark.slow
def test_the_bandwidth_test_counts_its_failures_by_reason(sim_scope):
    sim_scope.simulation.stall_stream(0.0, 600.0)

    results = sim_scope.diagnostics.run_camera_bandwidth_test(1)

    assert results['num_frames_failed'] == 1
    assert results['failed_by_reason'] == {'no_frame': 1}
    assert results['passed'] is False
