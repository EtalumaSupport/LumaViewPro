# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ScopeSession composition and lifecycle, against a real scope.

`ScopeSession` is the session-composition root: it wires a `Lumascope`,
settings, executors and the protocol runner into one object per client,
and L2 callers (REST / SDK / MATLAB / micromanager) reach the hardware
only through it. It had no test file of its own, so what coverage existed
was incidental -- including one test that asserted
`start_application_session` exists and that its name appears in a
docstring, which passes for a method that does nothing.

Everything here composes a REAL `Lumascope(simulate=True)` into a REAL
session and asserts observable behavior. Where a lifecycle call is meant
to start a worker, the test proves work actually runs on it rather than
inspecting a thread handle -- a started thread that cannot execute a
task is the failure worth catching.

Deliberately NOT covered: the ~20 zero-caller convenience forwarders
(`led_*_async`, `move_*_async`, `set_gain_*`, `capture_and_wait_*`,
`get_*_configs`). Pinning them would make the API-shrink work fight this
file for no gain, and a forwarder's behavior is its target's behavior,
tested where that lives.

`start_application_session` is covered where its motion lives:
`test_motion_state_gate.py` (the home-then-turret sequence and the
failed-home gate) and `test_session_bringup.py` (homing disabled means
no startup motion).
"""

from __future__ import annotations

import threading

import pytest

from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask


@pytest.fixture
def headless_session():
    """A real headless session, torn down whatever the test does."""
    session = ScopeSession.create_headless()
    try:
        yield session
    finally:
        try:
            session.shutdown_executors()
        except Exception:
            # Teardown must not mask the test's own failure; a session
            # whose executors never started is the normal case here.
            pass
        session.scope.disconnect()


class TestCreateHeadlessComposesARealSession:
    """`create_headless()` must return a session that can actually work.

    A factory that returns a half-wired object is the failure mode here:
    every field it forgets shows up much later as a None-deref or a
    capture that times out, far from this line.
    """

    def test_scope_is_a_real_lumascope_on_simulated_drivers(self, headless_session):
        from modules.lumascope_api import Lumascope
        from drivers.simulated_ledboard import SimulatedLEDBoard

        assert isinstance(headless_session.scope, Lumascope)
        assert isinstance(headless_session.scope._led_driver, SimulatedLEDBoard)

    def test_streaming_is_released(self, headless_session):
        """connect() leaves the sim camera configured but not grabbing.

        This factory is the entire bring-up for a headless session, so if
        it does not release the start gate every capture times out --
        which is the reason the release is in the factory at all.
        """
        assert headless_session.scope.imaging.is_streaming()

    def test_all_three_executor_handles_are_registered_on_the_scope(self, headless_session):
        """`scope.X_async` lands on a real queue, not None.

        The session holding an executor is not enough: the SCOPE has to
        know about it too, or the async paths silently have nowhere to
        put work.
        """
        assert headless_session.scope._io_executor is not None
        assert headless_session.scope._camera_executor is not None
        assert headless_session.scope._file_io_executor is not None

    def test_file_io_executor_is_the_bundle_s_one_instance(self, headless_session):
        """One FILE executor, not a duplicate per consumer.

        The session exposes `file_io_executor` so callers source the
        shared one; if it ever became a second instance, writes would be
        ordered against the wrong queue.
        """
        assert headless_session.file_io_executor is not None
        assert headless_session.file_io_executor is (
            headless_session.executor_bundle.file_io_executor
        )

    def test_source_path_is_registered(self, headless_session):
        assert headless_session.source_path == '.'

    def test_settings_is_a_dict(self, headless_session):
        """Resolved from current.json, then settings.json, then empty.

        Only the type is asserted: the CONTENT depends on what the
        machine running the suite has on disk, so asserting values here
        would make the test pass or fail on local state.
        """
        assert isinstance(headless_session.settings, dict)


class TestExecutorLifecycle:
    """`start_executors` / `shutdown_executors` must really start and stop.

    Proven by running work, not by reading a thread flag.
    """

    def test_started_io_executor_runs_queued_work(self, headless_session):
        headless_session.start_executors()
        ran = threading.Event()
        headless_session.io_executor.put(IOTask(action=ran.set))
        assert ran.wait(timeout=5.0), 'io_executor did not execute a queued task'

    def test_started_camera_executor_runs_queued_work(self, headless_session):
        headless_session.start_executors()
        ran = threading.Event()
        headless_session.camera_executor.put(IOTask(action=ran.set))
        assert ran.wait(timeout=5.0), 'camera_executor did not execute a queued task'

    def test_shutdown_stops_the_worker_threads(self, headless_session):
        headless_session.start_executors()
        ran = threading.Event()
        headless_session.io_executor.put(IOTask(action=ran.set))
        assert ran.wait(timeout=5.0)

        headless_session.shutdown_executors()
        for executor in (headless_session.io_executor, headless_session.camera_executor):
            worker = executor._worker_thread
            if worker is not None:
                worker.join(timeout=5.0)
                assert not worker.is_alive(), f'{executor.executor_name} still running'

    def test_shutdown_without_start_does_not_raise(self, headless_session):
        """Teardown paths call this without knowing whether start ran.

        An abort during bring-up reaches shutdown with executors that
        were never started, and raising there would mask the original
        failure.
        """
        headless_session.shutdown_executors()


class TestIsProtocolRunning:
    """The canonical read for callers holding a session handle.

    It must DERIVE from the exclusive-activity claim, not hold a second
    copy of the answer -- a duplicated flag is how "is a run in
    progress" ends up with two disagreeing sources.
    """

    def test_false_on_a_fresh_session(self, headless_session):
        assert headless_session.is_protocol_running is False

    def test_tracks_the_claim_in_both_directions(self, headless_session):
        assert headless_session.activity_claim.try_claim('protocol')
        assert headless_session.is_protocol_running is True
        headless_session.activity_claim.release('protocol')
        assert headless_session.is_protocol_running is False

    def test_a_recording_claim_is_not_a_run(self, headless_session):
        assert headless_session.activity_claim.try_claim('recording')
        try:
            assert headless_session.is_protocol_running is False
        finally:
            headless_session.activity_claim.release('recording')

    def test_is_read_only(self, headless_session):
        """No setter, so no second writer for run state."""
        with pytest.raises(AttributeError):
            headless_session.is_protocol_running = True
