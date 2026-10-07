# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Autofocus All Steps is an API call, and the API writes the focus it found.

The scan and its write-back were composed only in the GUI: the button
copied the protocol, turned every step's autofocus on, asked the engine
for a Z write-back by a flag no other caller could reach, and copied the
focused Z column into the protocol from a callback scheduled after the
run had released the scope. A script had to rebuild all of it by hand,
and still found the focus only on the protocol the run_complete callback
handed it.

Now ``ProtocolRunner.run_autofocus_all_steps`` is the scan, and the run
writes the focused Z into the caller's protocol itself, on its own
thread, before run_complete is sent and before it lets go of the scope.
A protocol whose steps no longer match the scanned copy -- a different
number of steps, or a step at a different position, channel or objective
-- is left unchanged, and the refusal is reported once in its own words.

Driven end to end on the simulated scope through the Session.
"""

import contextlib
from unittest.mock import MagicMock

import pytest

from modules.exceptions import FocusNotWrittenError, Refusal
from modules.run_outcome import PendingRunOutcome, RunEnding
from modules.protocol_image_writer import RunWriteBatch
from modules.sequenced_capture_runner import RunHandle

from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 120.0
FOCAL_Z_UM = 5050.0
STEP_Z_UM = 5000.0
# The 10x objective's sweep around a step at STEP_Z_UM.
SWEEP_LOW_UM, SWEEP_HIGH_UM = 4925.0, 5075.0


def _two_steps():
    return _protocol([_step('S1', 0, x=20.0, gain=1.0), _step('S2', 1, x=30.0, gain=1.0)])


@contextlib.contextmanager
def _focusing_session(tmp_path):
    """A headless session whose simulated sample comes into focus inside the sweep."""
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope._camera_driver.set_test_pattern(True, 'focus_target')
        session.scope.imaging.start_streaming()
        session.scope._camera_driver.set_focal_z(FOCAL_Z_UM)
        yield session, runner


@contextlib.contextmanager
def _reports_of(outcome_type):
    """Every outcome of *outcome_type* reported while the block runs."""
    from modules.notification_center import notifications

    reported = []
    report = notifications.report_outcome

    def counted(outcome, *a, **kw):
        if isinstance(outcome, outcome_type):
            reported.append(outcome)
        return report(outcome, *a, **kw)

    notifications.report_outcome = counted
    try:
        yield reported
    finally:
        notifications.report_outcome = report


class TestTheScanWritesTheFocusIntoTheCallersProtocol:
    def test_every_step_takes_the_focus_found_for_it(self, tmp_path):
        protocol = _two_steps()
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.run_autofocus_all_steps(protocol).wait(timeout_s=WAIT_S)

        assert outcome.status == 'completed', outcome
        assert outcome.focus_written is True, outcome
        # A sweep that chose no focus puts the stage back and leaves the
        # step's Z as it was, so a Z that moved inside the sweep's range is
        # one the sweep chose.
        for z in protocol.steps()['Z']:
            assert z != STEP_Z_UM, 'a step kept the Z it had before the scan'
            assert SWEEP_LOW_UM <= z <= SWEEP_HIGH_UM, protocol.steps()['Z'].tolist()

    def test_only_the_z_column_changes(self, tmp_path):
        protocol = _two_steps()
        before = protocol.steps().drop(columns=['Z']).copy()
        with _focusing_session(tmp_path) as (_session, runner):
            runner.run_autofocus_all_steps(protocol).wait(timeout_s=WAIT_S)

        assert protocol.steps().drop(columns=['Z']).equals(before), (
            "the scan turned every step's autofocus on in its own copy, not the caller's"
        )

    def test_the_focus_is_written_before_run_complete_is_sent(self, tmp_path):
        protocol = _two_steps()
        seen_at_run_complete = []
        with _focusing_session(tmp_path) as (_session, runner):
            runner.run_autofocus_all_steps(
                protocol,
                callbacks={
                    'run_complete': lambda **kw: seen_at_run_complete.append(
                        protocol.steps()['Z'].tolist()
                    )
                },
            ).wait(timeout_s=WAIT_S)

        assert seen_at_run_complete == [protocol.steps()['Z'].tolist()]
        assert STEP_Z_UM not in seen_at_run_complete[0], (
            'run_complete was sent before the focus was written'
        )

    def test_the_outcome_names_no_single_focus(self, tmp_path):
        # A run that focuses at several steps has no one focus to report;
        # the per-step answer is the protocol's Z column.
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.run_autofocus_all_steps(_two_steps()).wait(timeout_s=WAIT_S)

        assert outcome.af_focus_z_um is None, outcome


class TestAProtocolThatChangedDuringTheScanIsLeftAlone:
    def test_a_step_moved_during_the_scan_refuses_the_write_and_says_so_once(self, tmp_path):
        protocol = _two_steps()

        def _move_a_step(_step):
            # As the scan reaches its second step, behind the writers: the
            # protocol has no one-cell position writer.
            protocol._config['steps'].at[1, 'X'] = 40.0

        with (
            _reports_of(FocusNotWrittenError) as reported,
            _focusing_session(tmp_path) as (_session, runner),
        ):
            outcome = runner.run_autofocus_all_steps(
                protocol, callbacks={'update_step_number': _move_a_step}
            ).wait(timeout_s=WAIT_S)

        assert outcome.status == 'completed', outcome
        assert outcome.focus_written is False, outcome
        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM, STEP_Z_UM]
        assert len(reported) == 1, reported
        assert isinstance(reported[0], Refusal)

    def test_a_stopped_scan_writes_nothing(self, tmp_path):
        import threading

        protocol = _two_steps()
        with _focusing_session(tmp_path) as (_session, runner):
            # The callback can fire before the start returns the handle, so
            # the Stop waits for the handle the caller holds.
            started = []
            have_handle = threading.Event()

            def _stop_once_held():
                assert have_handle.wait(WAIT_S), 'the start never returned its handle'
                started[0].stop()

            def _stop(_step):
                threading.Thread(target=_stop_once_held).start()

            run = runner.run_autofocus_all_steps(protocol, callbacks={'update_step_number': _stop})
            started.append(run)
            have_handle.set()
            outcome = run.wait(timeout_s=WAIT_S)

        assert outcome.status == 'aborted', outcome
        assert outcome.focus_written is False, outcome
        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM, STEP_Z_UM]


class TestOnlyACompletedScanWritesItsFocus:
    """A scan that did not complete focused some of its steps and left the
    rest at their pre-scan Z, so writing its column would overwrite the
    caller's protocol with steps that never ran."""

    @staticmethod
    def _engine_after_a_scan(protocol):
        from tests.protocol_drives import bare_capture_runner

        engine = bare_capture_runner()
        scanned = protocol.copy_for_execution()
        for idx, z in enumerate([5111.0, 5222.0]):
            scanned.modify_step_z_height(idx, z)
        engine._protocol = scanned
        engine._write_focus_to = protocol
        return engine

    def test_a_completed_scan_writes_it(self):
        protocol = _two_steps()
        pending = PendingRunOutcome()
        engine = self._engine_after_a_scan(protocol)

        engine._write_focus(
            RunEnding('completed', 'completed', 't', 'm'),
            RunHandle(engine, pending, RunWriteBatch(MagicMock())),
        )

        assert protocol.steps()['Z'].tolist() == [5111.0, 5222.0]
        pending.resolve_if_pending(RunEnding('completed', 'completed', 't', 'm'))
        assert pending.wait(timeout_s=1.0).focus_written is True

    @pytest.mark.parametrize('status', ['aborted', 'failed', 'failed_at_start'])
    def test_an_unfinished_scan_writes_nothing(self, status):
        protocol = _two_steps()
        pending = PendingRunOutcome()
        pending.record_focus_written(False)  # as start() records it
        engine = self._engine_after_a_scan(protocol)

        engine._write_focus(RunEnding(status, 'stopped', 't', 'm'), pending)

        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM, STEP_Z_UM]
        pending.resolve_if_pending(RunEnding(status, 'stopped', 't', 'm'))
        assert pending.wait(timeout_s=1.0).focus_written is False

    def test_a_scan_torn_down_before_its_cleanup_says_it_wrote_nothing(self):
        # A shutdown settles the run without its cleanup, so the write never
        # runs; the outcome must say False, not None ('was not asked').
        from modules.sequenced_capture_runner import SequencedCaptureRunMode
        from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

        engine = bare_capture_runner()
        plan = engine.prepare(
            **scr_run_kwargs(
                run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                max_scans=1,
                write_focus_to=_two_steps(),
            )
        )
        run = engine.start(plan)

        # Session shutdown's settle. The run is never unwound here, so the
        # handle's wait (which waits for the scope to be free) would only
        # time out; the outcome the engine settled is read directly.
        engine.settle_unfinished_run(
            'shutdown', fallback=RunEnding('aborted', 'shutdown', 't', 'm')
        )

        assert run._pending.wait(timeout_s=1.0).focus_written is False


class TestARunThatWritesNoFocusSaysNone:
    def test_a_standalone_autofocus(self, tmp_path):
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.run_autofocus(layer='BF').wait(timeout_s=WAIT_S)

        assert outcome.focus_written is None, outcome

    def test_a_scan(self, tmp_path):
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.run_single_scan(
                protocol=_two_steps(),
                parent_dir=str(tmp_path / 'runs'),
            ).wait(timeout_s=WAIT_S)

        assert outcome.focus_written is None, outcome


class TestTheProtocolsOneFocusWriter:
    """``Protocol.adopt_focus_from`` writes Z only into the steps that were scanned."""

    def _scanned(self, protocol):
        scanned = protocol.copy_for_execution()
        for idx, z in enumerate([5111.0, 5222.0]):
            scanned.modify_step_z_height(idx, z)
        return scanned

    def test_matching_steps_take_the_scanned_z(self):
        protocol = _two_steps()
        protocol.adopt_focus_from(self._scanned(protocol))
        assert protocol.steps()['Z'].tolist() == [5111.0, 5222.0]

    @pytest.mark.parametrize(
        'column, value',
        [('X', 40.0), ('Y', 40.0), ('Color', 'Red'), ('Objective', '4x Oly')],
    )
    def test_a_step_that_differs_refuses_and_changes_nothing(self, column, value):
        protocol = _two_steps()
        scanned = self._scanned(protocol)
        # Behind the writers: the protocol has no one-cell writer for these.
        protocol._config['steps'].at[1, column] = value

        with pytest.raises(FocusNotWrittenError) as refused:
            protocol.adopt_focus_from(scanned)

        assert isinstance(refused.value, Refusal)
        assert refused.value.title == 'Focus Not Saved'
        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM, STEP_Z_UM]

    def test_a_different_number_of_steps_refuses_and_changes_nothing(self):
        protocol = _two_steps()
        scanned = self._scanned(protocol)
        protocol.delete_step(1)

        with pytest.raises(FocusNotWrittenError):
            protocol.adopt_focus_from(scanned)

        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM]


class TestEverySettlePathCarriesWhetherTheFocusWasWritten:
    @staticmethod
    def _ending():
        return RunEnding(status='completed', reason='', title='t', message='m')

    def test_cleanups_resolver_carries_it(self):
        pending = PendingRunOutcome()
        pending.record_focus_written(True)
        assert pending.resolve_if_pending(self._ending()) is True
        assert pending.wait(timeout_s=1.0).focus_written is True

    def test_teardowns_force_resolve_carries_it(self):
        pending = PendingRunOutcome()
        pending.record_focus_written(False)
        assert pending.force_resolve('shutdown', fallback=self._ending()) is True
        assert pending.wait(timeout_s=1.0).focus_written is False

    def test_the_merge_threads_resolver_carries_it(self):
        pending = PendingRunOutcome()
        pending.record_focus_written(True)
        token = pending.arm(self._ending())
        assert pending.resolve(token, merged=False, artifact_path=None, merge_reason='') is True
        assert pending.wait(timeout_s=1.0).focus_written is True

    def test_a_run_that_records_nothing_reports_none(self):
        pending = PendingRunOutcome()
        assert pending.resolve_if_pending(self._ending()) is True
        assert pending.wait(timeout_s=1.0).focus_written is None
