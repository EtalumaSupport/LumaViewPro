# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run takes the camera only once the camera lane has finished what it holds.

A run closes the camera lane to new work and then drives the camera from its
own thread. What the lane already held -- a still mid-grab, a gain write, a
settings widget's task -- keeps running on the lane's worker. Before this
the run touched the camera anyway: two threads in the driver at once, and
the run's first LED landing under a still's grab, which frame validity
turned into a failed still after its budget. Now the run loop's first act
waits for the lane to go idle, and only then snapshots the camera and
writes to it.

The tests here drive the real engine on simulated hardware. The lane is
held two ways: by a real still with a long exposure, and by a task that
holds the worker until released, which makes the wait as wide as a test
needs without reaching into the engine.
"""

import threading
import time
from unittest.mock import patch

import pytest

import modules.sequenced_capture_runner as runner_module
from modules.sequential_io_executor import IOTask
from tests.test_composite_run_e2e import headless_settings, open_composite_session

RESULT_TIMEOUT_S = 30.0
STILL_EXPOSURE_MS = 800.0


@pytest.fixture
def lane_session(tmp_path):
    settings = headless_settings(tmp_path)
    settings['separate_folder_per_channel'] = False
    # The simulator honours exposure as the frame interval, so a long
    # exposure is a still that holds the lane long enough to see.
    settings['BF']['exposure_ms'] = STILL_EXPOSURE_MS
    with open_composite_session(settings) as (session, runner):
        yield session, runner, tmp_path


class _LaneHold:
    """Hold the camera lane with a task of its own until released."""

    def __init__(self, session):
        self._release = threading.Event()
        self._running = threading.Event()
        session.camera_executor.put(IOTask(action=self._hold))
        assert self._running.wait(5.0), 'the hold never reached the camera lane'

    def _hold(self):
        self._running.set()
        self._release.wait(RESULT_TIMEOUT_S)

    def release(self):
        self._release.set()


def _first_step_observer(session):
    """Record what the lane and the still looked like at the run's first step."""
    seen = {}

    def _on_step(_step_number):
        if 'lane_busy' not in seen:
            seen['lane_busy'] = session.camera_executor.is_busy()
            seen['still_in_flight'] = session.manual_capture.in_flight
            seen['at'] = time.monotonic()

    return seen, {'update_step_number': _on_step}


class TestAStillInFlightFinishesFirst:
    def test_the_still_saves_and_the_run_starts_after_it(self, lane_session):
        session, runner, tmp_path = lane_session
        still = session.manual_capture.capture(layer='BF', false_color_on=False)
        assert session.manual_capture.in_flight, 'the still never reached the lane'
        seen, callbacks = _first_step_observer(session)

        outcome = runner.start_composite(
            sequence_name='after_still', parent_dir=str(tmp_path), callbacks=callbacks
        )

        paths = still.result(timeout=RESULT_TIMEOUT_S)
        still_done_at = time.monotonic()
        assert paths and paths[0].exists(), 'the still under a starting run saved nothing'
        assert session.sequenced_capture_runner.wait_for_run_idle(timeout_s=RESULT_TIMEOUT_S)
        result = outcome.wait(timeout_s=RESULT_TIMEOUT_S)
        assert result is not None and result.status == 'completed', result
        assert seen['still_in_flight'] is False, (
            "the run's first step ran while the still was still on the lane"
        )
        assert seen['lane_busy'] is False, "the run's first step ran while the lane held work"
        assert seen['at'] >= still_done_at - 0.05, 'the first step came before the still finished'


class TestAHeldLane:
    def test_a_stop_during_the_wait_ends_the_run_stopped_and_restores_nothing(self, lane_session):
        session, runner, tmp_path = lane_session
        scope = session.scope
        # A channel lit by the user before the click: cleanup must leave it
        # lit, not read the missing snapshot as "nothing was lit" and
        # darken the sample.
        scope.illumination.led_on('BF', 20.0)
        gain_before = scope.imaging.get_gain_db()
        leds_before = scope.illumination.get_led_states()
        assert leds_before['BF']['enabled'], 'the fixture could not light a channel'
        hold = _LaneHold(session)
        try:
            outcome = runner.start_composite(sequence_name='stopped', parent_dir=str(tmp_path))
            assert session.is_protocol_running
            # The run is waiting on the protocol thread; Stop it there.
            session.sequenced_capture_runner.reset(outcome)
            assert session.sequenced_capture_runner.wait_for_run_idle(timeout_s=RESULT_TIMEOUT_S)
        finally:
            hold.release()
        result = outcome.wait(timeout_s=RESULT_TIMEOUT_S)
        assert result is not None and result.status == 'aborted', result
        assert result.reason == 'stopped'
        assert scope.imaging.get_gain_db() == gain_before, 'cleanup wrote the camera'
        assert scope.illumination.get_led_states() == leds_before, 'cleanup changed an LED'

    def test_a_stuck_lane_ends_the_run_failed_and_names_the_lane(self, lane_session):
        session, runner, tmp_path = lane_session
        completions = []
        hold = _LaneHold(session)
        try:
            # The bound is the file lane's wedge threshold; a test cannot
            # wait it out, so it is shortened at the one name the takeover reads.
            with patch.object(runner_module, 'WRITE_STALL_FATAL_S', 0.2):
                outcome = runner.start_composite(
                    sequence_name='stuck',
                    parent_dir=str(tmp_path),
                    callbacks={'run_complete': lambda **kw: completions.append(kw)},
                )
                assert session.sequenced_capture_runner.wait_for_run_idle(
                    timeout_s=RESULT_TIMEOUT_S
                )
        finally:
            hold.release()
        result = outcome.wait(timeout_s=RESULT_TIMEOUT_S)
        assert result is not None and result.status == 'failed', result
        assert result.reason == 'camera_lane_stalled'
        assert len(completions) == 1, completions


class TestARunThatAlreadyEnded:
    def test_takes_nothing(self):
        # A reset in the gap between the run's commit and its loop's
        # dispatch tears the run down inline; the loop then still runs,
        # and its first act must not write the camera for a run that is
        # gone or snapshot a state nothing will restore.
        from modules.protocol_state_machine import ProtocolState
        from tests.protocol_drives import bare_capture_runner

        runner = bare_capture_runner()
        runner._state = ProtocolState.IDLE

        assert runner._take_camera() is None
        assert runner._scope.imaging.save_camera_state.call_count == 0
        assert runner._scope.imaging.update_auto_gain_target_brightness.call_count == 0


class TestATakeoverWriteThatRaises:
    def test_ends_the_run_failed_with_one_completion(self, lane_session):
        session, runner, tmp_path = lane_session
        completions = []

        def _boom(*args, **kwargs):
            raise RuntimeError('camera rejected target brightness')

        with patch.object(session.scope.imaging, '_update_auto_gain_target_brightness_impl', _boom):
            outcome = runner.start_composite(
                sequence_name='raises',
                parent_dir=str(tmp_path),
                callbacks={'run_complete': lambda **kw: completions.append(kw)},
            )
            assert session.sequenced_capture_runner.wait_for_run_idle(timeout_s=RESULT_TIMEOUT_S)
        result = outcome.wait(timeout_s=RESULT_TIMEOUT_S)
        assert result is not None and result.status == 'failed', result
        assert result.reason == 'run_loop_crashed'
        assert len(completions) == 1, completions
        assert not session.is_protocol_running
