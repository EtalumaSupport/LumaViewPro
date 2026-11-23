# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Autofocus All Steps is an API call, and the API writes the focus it found.

The scan and its write-back were composed only in the GUI: the button
copied the protocol, turned every step's autofocus on, asked the engine
for a Z write-back by a flag no other caller could reach, and copied the
focused Z column into the protocol from a callback scheduled after the
run had released the scope. A script had to rebuild all of it by hand,
and still found the focus only on the protocol the run_ended event handed
it.

Now ``ProtocolRunner.start_autofocus_all_steps`` is the scan, and the run
writes the focused Z into the caller's protocol itself, on its own
thread, before run_ended is sent and before it lets go of the scope.
A protocol whose steps no longer match the scanned copy -- a different
number of steps, or a step at a different position, channel or objective
-- is left unchanged, and the refusal is reported once in its own words.
A scan stopped or failed partway keeps the focus of every step it focused,
and tells how many once.

Driven end to end on the simulated scope through the Session.
"""

import contextlib
from unittest.mock import MagicMock

import pytest

from _lumascope.modules.exceptions import FocusKeptNotice, FocusNotWrittenError, Refusal
from _lumascope.modules.run_outcome import PendingRunOutcome, RunEnding
from _lumascope.modules.protocol_image_writer import RunWriteBatch
from _lumascope.modules.run_events import RunEvents
from _lumascope.modules.sequenced_capture_runner import EngineRun

from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.stock_channels import LED2

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
        session.scope.imaging.enable_streaming()
        session.scope._camera_driver.set_focal_z(FOCAL_Z_UM)
        yield session, runner


@contextlib.contextmanager
def _reports_of(outcome_type):
    """Every outcome of *outcome_type* reported while the block runs."""
    from _lumascope.modules.notification_center import notifications

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
    @pytest.mark.slow
    def test_every_step_takes_the_focus_found_for_it(self, tmp_path):
        protocol = _two_steps()
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.start_autofocus_all_steps(protocol).result(timeout=WAIT_S)

        assert outcome.status == 'completed', outcome
        assert outcome.focus_written is True, outcome
        # A sweep that chose no focus puts the stage back and leaves the
        # step's Z as it was, so a Z that moved inside the sweep's range is
        # one the sweep chose.
        for z in protocol.steps()['Z']:
            assert z != STEP_Z_UM, 'a step kept the Z it had before the scan'
            assert SWEEP_LOW_UM <= z <= SWEEP_HIGH_UM, protocol.steps()['Z'].tolist()

    @pytest.mark.slow
    def test_only_the_z_column_changes(self, tmp_path):
        protocol = _two_steps()
        before = protocol.steps().drop(columns=['Z']).copy()
        with _focusing_session(tmp_path) as (_session, runner):
            runner.start_autofocus_all_steps(protocol).result(timeout=WAIT_S)

        assert protocol.steps().drop(columns=['Z']).equals(before), (
            "the scan turned every step's autofocus on in its own copy, not the caller's"
        )

    @pytest.mark.slow
    def test_the_focus_is_written_before_run_ended_is_sent(self, tmp_path):
        protocol = _two_steps()
        seen_at_run_ended = []
        with _focusing_session(tmp_path) as (_session, runner):
            runner.start_autofocus_all_steps(
                protocol,
                events=RunEvents(
                    run_ended=lambda *ended: seen_at_run_ended.append(
                        protocol.steps()['Z'].tolist()
                    )
                ),
            ).result(timeout=WAIT_S)

        assert seen_at_run_ended == [protocol.steps()['Z'].tolist()]
        assert STEP_Z_UM not in seen_at_run_ended[0], (
            'run_ended was sent before the focus was written'
        )

    @pytest.mark.slow
    def test_the_outcome_names_no_single_focus(self, tmp_path):
        # A run that focuses at several steps has no one focus to report;
        # the per-step answer is the protocol's Z column.
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.start_autofocus_all_steps(_two_steps()).result(timeout=WAIT_S)

        assert outcome.af_focus_z_um is None, outcome


class TestAProtocolThatChangedDuringTheScanIsLeftAlone:
    @pytest.mark.slow
    def test_a_step_moved_during_the_scan_refuses_the_write_and_says_so_once(self, tmp_path):
        protocol = _two_steps()

        def _move_a_step(step_index):
            # As the scan reaches its second step, behind the writers: the
            # protocol has no one-cell position writer.
            if step_index == 1:
                protocol._config['steps'].at[1, 'X'] = 40.0

        with (
            _reports_of(FocusNotWrittenError) as reported,
            _focusing_session(tmp_path) as (_session, runner),
        ):
            outcome = runner.start_autofocus_all_steps(
                protocol, events=RunEvents(step_started=_move_a_step)
            ).result(timeout=WAIT_S)

        assert outcome.status == 'completed', outcome
        assert outcome.focus_written is False, outcome
        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM, STEP_Z_UM]
        assert len(reported) == 1, reported
        assert isinstance(reported[0], Refusal)


def _three_steps():
    return _protocol([_step(f'S{i + 1}', i, x=20.0 + 10.0 * i, gain=1.0) for i in range(3)])


def _a_step_and_a_stack_and_a_step():
    """S1, a three-slice z-stack at one position (its reference slice at
    STEP_Z_UM, 10 um apart), then S5."""
    stack = []
    for slice_index, offset in enumerate((-10.0, 0.0, 10.0)):
        row = _step(f'Z{slice_index + 1}', 1 + slice_index, x=30.0, gain=1.0)
        row.update(
            {
                'Z': STEP_Z_UM + offset,
                'Z-Slice': slice_index,
                'Z-Stack Group ID': 0,
                'Z-Stack Offset': offset,
                'Well': 'Z',
            }
        )
        stack.append(row)
    return _protocol([_step('S1', 0, x=20.0, gain=1.0), *stack, _step('S5', 4, x=40.0, gain=1.0)])


def _scan_stopped_at(runner, protocol, step_index, *, before_the_stop=None):
    """Autofocus All Steps on *protocol*, Stop pressed as step *step_index*
    (0-based) starts, after *before_the_stop* runs there; returns the outcome."""
    import threading

    # The event can fire before the start returns the handle, so the Stop
    # waits for the handle the caller holds.
    started = []
    have_handle = threading.Event()

    def _stop_once_held():
        assert have_handle.wait(WAIT_S), 'the start never returned its operation'
        started[0].stop()

    def _stop(index):
        if index == step_index:
            if before_the_stop is not None:
                before_the_stop()
            threading.Thread(target=_stop_once_held).start()

    run = runner.start_autofocus_all_steps(protocol, events=RunEvents(step_started=_stop))
    started.append(run)
    have_handle.set()
    return run.result(timeout=WAIT_S)


class TestAStoppedScanKeepsWhatItFound:
    """Autofocus All Steps stopped partway: each step whose autofocus found a
    focus keeps it, a step not reached keeps its Z, and the person is told
    once how many steps were focused."""

    @pytest.mark.slow
    def test_the_steps_focused_before_the_stop_keep_their_focus(self, tmp_path):
        protocol = _three_steps()
        with (
            _reports_of(FocusKeptNotice) as told,
            _focusing_session(tmp_path) as (_session, runner),
        ):
            outcome = _scan_stopped_at(runner, protocol, 1)

        assert outcome.status == 'aborted', outcome
        assert outcome.focus_written is True, outcome
        first, *rest = protocol.steps()['Z'].tolist()
        assert first != STEP_Z_UM and SWEEP_LOW_UM <= first <= SWEEP_HIGH_UM, first
        assert rest == [STEP_Z_UM, STEP_Z_UM]
        assert [(notice.focused, notice.steps) for notice in told] == [(1, 3)]
        assert '1 of 3 steps' in str(told[0])

    @pytest.mark.slow
    def test_a_stop_during_the_first_autofocus_writes_nothing_and_says_so(self, tmp_path):
        protocol = _three_steps()
        with (
            _reports_of(FocusKeptNotice) as told,
            _focusing_session(tmp_path) as (_session, runner),
        ):
            outcome = _scan_stopped_at(runner, protocol, 0)

        assert outcome.status == 'aborted', outcome
        assert outcome.focus_written is False, outcome
        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM] * 3
        assert [(notice.focused, notice.steps) for notice in told] == [(0, 3)]
        assert 'No step was focused' in str(told[0])

    @pytest.mark.slow
    def test_a_placed_stack_keeps_every_slice_and_counts_them_all(self, tmp_path):
        protocol = _a_step_and_a_stack_and_a_step()
        with (
            _reports_of(FocusKeptNotice) as told,
            _focusing_session(tmp_path) as (_session, runner),
        ):
            # Stopped as S5 starts: S1 and the stack were focused.
            outcome = _scan_stopped_at(runner, protocol, 4)

        assert outcome.status == 'aborted', outcome
        z = protocol.steps()['Z'].tolist()
        assert z[0] != STEP_Z_UM, z
        assert z[2] != STEP_Z_UM and SWEEP_LOW_UM <= z[2] <= SWEEP_HIGH_UM, z
        assert (z[1], z[3]) == pytest.approx((z[2] - 10.0, z[2] + 10.0)), z
        assert z[4] == STEP_Z_UM, z
        assert [(notice.focused, notice.steps) for notice in told] == [(4, 5)]

    @pytest.mark.slow
    def test_a_step_changed_under_a_stopped_scan_refuses_once_and_tells_nothing_else(
        self, tmp_path
    ):
        protocol = _three_steps()

        def _move_step_3():
            # Behind the writers: the protocol has no one-cell position writer.
            protocol._config['steps'].at[2, 'X'] = 80.0

        with (
            _reports_of(FocusNotWrittenError) as refused,
            _reports_of(FocusKeptNotice) as told,
            _focusing_session(tmp_path) as (_session, runner),
        ):
            outcome = _scan_stopped_at(runner, protocol, 1, before_the_stop=_move_step_3)

        assert outcome.status == 'aborted', outcome
        assert outcome.focus_written is False, outcome
        assert protocol.steps()['Z'].tolist() == [STEP_Z_UM] * 3
        assert len(refused) == 1, refused
        assert told == []


class TestTheScanWritesWhatItFoundHoweverItEnded:
    """The run's copy holds a found focus at each step that has one and the
    caller's Z everywhere else, so its column is written whenever a step was
    focused, whatever ended the run; a stopped or failed scan tells what it
    kept, and a scan that focused no step writes nothing."""

    @staticmethod
    def _engine_after_a_scan(protocol, *, focused):
        from tests.protocol_drives import bare_capture_runner

        engine = bare_capture_runner()
        scanned = protocol.copy_for_execution()
        for idx, z in enumerate([5111.0, 5222.0][:focused]):
            scanned.modify_step_z_height(idx, z)
        engine._protocol = scanned
        engine._write_focus_to = protocol
        # As the step runner counts each focus it writes into the copy.
        engine._steps_focused = focused
        return engine

    @staticmethod
    def _write(engine, status):
        pending = PendingRunOutcome()
        pending.record_focus_written(False)  # as start() records it
        engine._write_focus(
            RunEnding(status, 'stopped', 'm'),
            EngineRun(engine, pending, RunWriteBatch(MagicMock())),
        )
        pending.resolve_if_pending(RunEnding(status, 'stopped', 'm'))
        return pending.wait(timeout_s=1.0)

    def test_every_ending_writes_the_steps_it_focused_and_a_partway_one_tells_it(self):
        told_by_status = {}
        for status in ('completed', 'incomplete', 'aborted', 'failed'):
            protocol = _two_steps()
            engine = self._engine_after_a_scan(protocol, focused=1)
            with _reports_of(FocusKeptNotice) as told:
                outcome = self._write(engine, status)
            assert protocol.steps()['Z'].tolist() == [5111.0, STEP_Z_UM], status
            assert outcome.focus_written is True, status
            told_by_status[status] = [(notice.focused, notice.steps) for notice in told]

        assert told_by_status == {
            'completed': [],
            'incomplete': [],
            'aborted': [(1, 2)],
            'failed': [(1, 2)],
        }

    def test_a_scan_that_focused_no_step_writes_nothing(self):
        told_by_status = {}
        for status in ('aborted', 'failed', 'failed_at_start'):
            protocol = _two_steps()
            # The copy's Z moved, but by no found focus: nothing is written.
            engine = self._engine_after_a_scan(protocol, focused=0)
            engine._protocol.modify_step_z_height(0, 5333.0)
            with _reports_of(FocusKeptNotice) as told:
                outcome = self._write(engine, status)
            assert protocol.steps()['Z'].tolist() == [STEP_Z_UM, STEP_Z_UM], status
            assert outcome.focus_written is False, status
            told_by_status[status] = [(notice.focused, notice.steps) for notice in told]

        # A run that failed at start is told by its start failure.
        assert told_by_status == {'aborted': [(0, 2)], 'failed': [(0, 2)], 'failed_at_start': []}


class TestARunThatWritesNoFocusSaysNone:
    def test_a_scan(self, tmp_path):
        with _focusing_session(tmp_path) as (_session, runner):
            outcome = runner.start_single_scan(
                protocol=_two_steps(),
                parent_dir=str(tmp_path / 'runs'),
            ).result(timeout=WAIT_S)

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
        [('X', 40.0), ('Y', 40.0), ('channel', LED2), ('Objective', '4x Oly')],
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

    def test_the_refusal_names_the_first_step_and_its_first_identity_that_differ(self):
        """Step 1's position and objective and step 2's position differ: step 1's position is named."""
        protocol = _two_steps()
        scanned = self._scanned(protocol)
        # Behind the writers: the protocol has no one-cell writer for these.
        protocol._config['steps'].at[0, 'Objective'] = '4x Oly'
        protocol._config['steps'].at[0, 'Y'] = 40.0
        protocol._config['steps'].at[1, 'X'] = 40.0

        with pytest.raises(FocusNotWrittenError) as refused:
            protocol.adopt_focus_from(scanned)

        assert "step 1's position changed" in str(refused.value), refused.value

    def test_a_step_with_no_position_matches_one_with_none_and_not_one_with_a_position(self):
        protocol = _two_steps()
        # Behind the writers: step 1 is made a step with no stage position.
        protocol._config['steps'].loc[0, ['X', 'Y']] = float('nan')
        scanned = self._scanned(protocol)

        protocol.adopt_focus_from(scanned)
        assert protocol.steps()['Z'].tolist() == [5111.0, 5222.0]

        protocol._config['steps'].at[0, 'X'] = 20.0
        with pytest.raises(FocusNotWrittenError) as refused:
            protocol.adopt_focus_from(scanned)
        assert "step 1's position changed" in str(refused.value), refused.value

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
        return RunEnding(status='completed', reason='completed', message='m')

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
