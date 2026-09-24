# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's outcome says whether autofocus characterization data landed.

"The run completed" and "the characterization data is on disk" are
different answers, and a headless caller can only see the second if the
outcome carries it.  The trap this pins is that every cheaper source for
that answer is a lie waiting to happen: the results folder is allocated
eagerly when the sweep STARTS, the CSV write is queued rather than
performed, the queue is discarded wholesale by an aborting run, and the
save early-returns when the sweep collected nothing.  All four leave a
folder on disk with nothing in it, so a folder-sourced answer reports a
delivery that never happened -- the same "queued is not delivered" shape
that made a standalone AF silently produce empty folders.

So the outcome is sourced from the write itself, and the pair that says
so cannot disagree: the flag is derived from the path.

The end-to-end harness mirrors
tests/test_standalone_af_characterization_delivery.py deliberately --
real simulated scope, real executors, a real AutofocusRunner on a real
AutofocusThread.  A mock file executor makes "nothing was written" pass
vacuously, which is precisely the blind spot being pinned.
"""

from __future__ import annotations

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Mock settings_init before
# sequenced_capture_runner imports it. (Harness mirrors
# tests/test_standalone_af_characterization_delivery.py.)
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {
    'BF': {'autofocus': False},
    'PC': {'autofocus': False},
    'DF': {'autofocus': False},
    'Red': {'autofocus': False},
    'Green': {'autofocus': False},
    'Blue': {'autofocus': False},
    'Lumi': {'autofocus': False},
}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from modules.activity_claim import ActivityClaim
from modules.image_mode import ImageCaptureConfig
from modules.lumascope_api import Lumascope
from modules.protocol import Protocol
from modules.run_outcome import PendingRunOutcome, RunEnding, RunOutcome
from modules.sequenced_capture_runner import SequencedCaptureRunner
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import SequentialIOExecutor
from tests.protocol_drives import autofocus_snapshot
from tests.scope_fakes import home_sim_scope
from tests.scope_fakes import configure_turret_like_bringup

COMPLETION_TIMEOUT = 60  # seconds -- a real AF sweep runs in sim time

TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


def _ending(status: str = 'completed') -> RunEnding:
    return RunEnding(status=status, reason='stopped', title='t', message='m')


def _make_af_step_protocol():
    import pandas as pd

    step = {
        'Name': 'AF_test',
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': True,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': '10x Oly',
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {'duration': 1, 'fps': 5},
        'Stim_Config': {},
        'Step Index': 0,
        'Label': 'AF_test',
        'Auto_Named': False,
    }
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': pd.DataFrame([step]),
        'period': datetime.timedelta(minutes=1.0),
        'duration': datetime.timedelta(hours=1.0),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)


class _AfRig:
    """A real runner over real executors, reusable for two runs in a row."""

    def __init__(self):
        from modules.autofocus_runner import AutofocusRunner
        from modules.autofocus_thread import AutofocusThread
        from modules.coord_transformations import CoordinateTransformer
        from modules.labware_loader import WellPlateLoader
        from modules.protocol_thread import ProtocolThread

        self.scope = home_sim_scope(Lumascope(simulate=True))
        # A bare scope skipped bring-up, which fills the turret from the
        # persisted slots; an empty turret addresses no glass at all.
        configure_turret_like_bringup(self.scope)
        # The session registers the data root at bring-up; a runner over a
        # bare scope needs it too, or the run refuses at start.
        self.scope.protocols.register_source_path('.')
        self.scope._led_driver.set_timing_mode('fast')
        self.scope._motion_driver.set_timing_mode('fast')
        self.scope._camera_driver.set_timing_mode('fast')
        self.scope.imaging.start_streaming()

        self.io_executor = SequentialIOExecutor(name='AFOUT_IO')
        self.file_io_executor = SequentialIOExecutor(name='AFOUT_FILE')
        self.camera_executor = SequentialIOExecutor(name='AFOUT_CAMERA')
        for e in (self.io_executor, self.file_io_executor, self.camera_executor):
            e.start()
        self.protocol_thread = ProtocolThread()
        self.protocol_thread.start()

        self.af_runner = AutofocusRunner(
            scope=self.scope,
            camera_executor=self.camera_executor,
            io_executor=self.io_executor,
            file_io_executor=self.file_io_executor,
        )
        self.af_thread = AutofocusThread(afe=self.af_runner)
        self.af_thread.start()

        self.runner = SequencedCaptureRunner(
            scope=self.scope,
            stage_offset={'x': 0.0, 'y': 0.0},
            io_executor=self.io_executor,
            protocol_thread=self.protocol_thread,
            file_io_executor=self.file_io_executor,
            camera_executor=self.camera_executor,
            autofocus_thread=self.af_thread,
            activity_claim=ActivityClaim(),
            autofocus_runner=self.af_runner,
        )
        self.runner._wellplate_loader = WellPlateLoader()
        self.runner._coordinate_transformer = CoordinateTransformer()

    def run_autofocus(
        self, parent_dir: pathlib.Path, *, save_data: bool, borrowed_claim=None
    ) -> RunOutcome:
        """Drive one standalone AF run to its settled outcome."""
        pending = self.start_autofocus(
            parent_dir, save_data=save_data, borrowed_claim=borrowed_claim
        )
        assert self._done.wait(timeout=COMPLETION_TIMEOUT), 'AF run did not complete'
        assert self._files_done.wait(timeout=COMPLETION_TIMEOUT), 'files_complete did not fire'
        outcome = pending.wait(timeout_s=COMPLETION_TIMEOUT)
        assert outcome is not None, 'the AF run never settled its outcome'
        return outcome

    def prepare_autofocus(self, parent_dir: pathlib.Path, *, save_data: bool, borrowed_claim=None):
        """Prepare one standalone AF run; the plan, not yet started."""
        done = self._done = threading.Event()
        files_done = self._files_done = threading.Event()
        return self.runner.prepare(
            protocol=_make_af_step_protocol(),
            run_trigger_source='autofocus',
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            sequence_name='autofocus',
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            autogain_settings={
                'target_brightness': 0.3,
                'min_gain_db': 0.0,
                'max_gain_db': 20.0,
                'max_duration': datetime.timedelta(seconds=1),
            },
            parent_dir=parent_dir,
            enable_image_saving=False,
            disable_saving_artifacts=True,
            save_autofocus_data=save_data,
            max_scans=1,
            callbacks={
                'go_to_step': lambda **kw: None,
                'move_position': lambda axis: None,
                'run_complete': lambda **kw: done.set(),
                'files_complete': lambda **kw: files_done.set(),
            },
            leds_state_at_end='off',
            autofocus_snapshot=autofocus_snapshot(
                states={
                    'BF': True,
                    'PC': False,
                    'DF': False,
                    'Red': False,
                    'Green': False,
                    'Blue': False,
                    'Lumi': False,
                },
            ),
            borrowed_claim=borrowed_claim,
        )

    def start_autofocus(
        self, parent_dir: pathlib.Path, *, save_data: bool, borrowed_claim=None
    ) -> PendingRunOutcome:
        """Start one standalone AF run and return without waiting."""
        plan = self.prepare_autofocus(
            parent_dir, save_data=save_data, borrowed_claim=borrowed_claim
        )
        return self.runner.start(plan)

    def close(self):
        self.af_thread.stop()
        self.protocol_thread.stop(timeout=2.0)
        for e in (self.io_executor, self.file_io_executor, self.camera_executor):
            try:
                e.shutdown()
            except Exception:
                # Teardown only, and deliberately silent: a failing shutdown
                # here would replace whichever assertion actually failed with
                # a teardown error, hiding the result this test exists to
                # report. Nothing under test is observed after this point.
                pass
        self.scope.imaging.stop_streaming()
        self.scope.disconnect()


class TestOutcomeReportsDeliveredAutofocusData:
    def test_requested_data_is_reported_with_the_file_that_was_written(self, tmp_path):
        """The outcome names the CSV, and the CSV is readable."""
        rig = _AfRig()
        char_dir = tmp_path / 'Autofocus Characterization'
        try:
            outcome = rig.run_autofocus(char_dir, save_data=True)
        finally:
            rig.close()

        assert outcome.af_data_saved is True, (
            'an AF run asked to save characterization data must report that it did'
        )
        assert outcome.af_data_path is not None
        written = pathlib.Path(outcome.af_data_path)
        # Reading the file is the assertion: a path that merely exists as
        # a string, or one naming the eagerly-created folder, is the exact
        # answer this field exists to replace.
        assert written.is_file(), f'af_data_path must name a file that exists: {written}'
        assert written.suffix == '.csv'
        assert written.read_text().strip(), 'the reported characterization file is empty'
        assert char_dir in written.parents

    def test_a_later_run_that_saves_nothing_reports_nothing(self, tmp_path):
        """A populated folder from an earlier run is not this run's answer.

        The folder from run one is still on disk and still full when run
        two settles.  An implementation sourced from the results directory
        -- or one that forgets to clear the recorded path between runs --
        reports run one's delivery as run two's.
        """
        rig = _AfRig()
        char_dir = tmp_path / 'Autofocus Characterization'
        try:
            first = rig.run_autofocus(char_dir, save_data=True)
            assert first.af_data_saved is True, 'precondition: run one delivers'
            second = rig.run_autofocus(char_dir, save_data=False)
        finally:
            rig.close()

        assert list(char_dir.rglob('*.csv')), (
            "precondition: run one's data is still on disk when run two settles"
        )
        assert second.af_data_saved is False, (
            'a run that saved no characterization data must not report the '
            "previous run's file as its own"
        )
        assert second.af_data_path is None


class TestTheTwoFieldsCannotDisagree:
    """The flag is derived from the path, so 'saved with nowhere to look'
    is unconstructible rather than merely undocumented."""

    def test_a_path_makes_it_saved(self):
        outcome = RunOutcome.from_ending(
            _ending(), merged=False, artifact_path=None, merge_reason='', af_data_path='/tmp/a.csv'
        )
        assert outcome.af_data_saved is True
        assert outcome.af_data_path == '/tmp/a.csv'

    def test_no_path_makes_it_unsaved(self):
        outcome = RunOutcome.from_ending(
            _ending(), merged=False, artifact_path=None, merge_reason=''
        )
        assert outcome.af_data_saved is False
        assert outcome.af_data_path is None


class TestEverySettlePathCarriesTheRecordedData:
    """A run settles down one of three paths and the data landed
    regardless of which; all three must answer the same."""

    def test_cleanups_resolver_carries_it(self):
        pending = PendingRunOutcome()
        pending.record_autofocus_data('/tmp/af.csv')
        assert pending.resolve_if_pending(_ending()) is True
        outcome = pending.wait(timeout_s=1.0)
        assert (outcome.af_data_saved, outcome.af_data_path) == (True, '/tmp/af.csv')

    def test_teardowns_force_resolve_carries_it(self):
        pending = PendingRunOutcome()
        pending.record_autofocus_data('/tmp/af.csv')
        assert pending.force_resolve('shutdown', fallback=_ending('failed')) is True
        outcome = pending.wait(timeout_s=1.0)
        assert (outcome.af_data_saved, outcome.af_data_path) == (True, '/tmp/af.csv')

    def test_the_merge_threads_resolver_carries_it(self):
        pending = PendingRunOutcome()
        pending.record_autofocus_data('/tmp/af.csv')
        token = pending.arm(_ending())
        assert token is not None
        assert (
            pending.resolve(token, merged=True, artifact_path='/tmp/merged.tiff', merge_reason='')
            is True
        )
        outcome = pending.wait(timeout_s=1.0)
        assert (outcome.af_data_saved, outcome.af_data_path) == (True, '/tmp/af.csv')
        assert outcome.merged is True

    def test_a_run_that_records_nothing_reports_nothing(self):
        pending = PendingRunOutcome()
        assert pending.resolve_if_pending(_ending()) is True
        outcome = pending.wait(timeout_s=1.0)
        assert (outcome.af_data_saved, outcome.af_data_path) == (False, None)


class TestTheSweepDoesNotReturnBeforeItsWriteLands:
    """Queueing the save is not writing it.

    The sweep's answer to "what did I write" is read after the sweep ends,
    by the run that settles its outcome. The save rides the sequential file
    lane and run cleanup does not wait for it -- it defers a files-complete
    callback instead -- so a sweep that returned as soon as the task was
    queued would let a reader be told nothing was written by a sweep whose
    file lands moments later.
    """

    def _bare_runner(self):
        """A runner whose only exercised surface is the wait itself.

        The scope is specced rather than bare so a name this test does not
        use, but a future edit might, fails loudly instead of answering
        with a mock. The executors stay plain doubles: nothing here
        submits work, and the waiter is handed in directly.
        """
        from modules.autofocus_runner import AutofocusRunner
        from tests.scope_fakes import spec_scope

        return AutofocusRunner(
            scope=spec_scope(),
            camera_executor=MagicMock(),
            io_executor=MagicMock(),
            file_io_executor=MagicMock(),
        )

    def test_the_wait_blocks_until_the_write_completes(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter

        runner = self._bare_runner()
        waiter = _ReusableTaskWaiter()
        runner._data_write_future = waiter

        HOLD_S = 0.25
        threading.Timer(HOLD_S, lambda: waiter.set_result(None)).start()
        started = time.monotonic()
        runner._await_data_write()
        elapsed = time.monotonic() - started

        assert elapsed >= HOLD_S, (
            'the sweep returned before its queued write completed; a reader '
            'would be told nothing was written by a sweep whose file lands '
            f'moments later (waited {elapsed:.3f}s for a {HOLD_S}s write)'
        )

    def test_a_cancelled_write_does_not_cost_the_bound(self):
        """An aborting run discards the queued save on purpose.

        clear_protocol_pending cancels the task rather than running it, so
        the wait must end on the cancellation and not sit out the timeout --
        an abort exists to give the user control back.
        """
        from modules.autofocus_runner import AF_DATA_WRITE_WAIT_S
        from modules.sequential_io_executor import _ReusableTaskWaiter

        runner = self._bare_runner()
        waiter = _ReusableTaskWaiter()
        waiter.cancel()
        runner._data_write_future = waiter

        started = time.monotonic()
        runner._await_data_write()
        elapsed = time.monotonic() - started

        assert elapsed < AF_DATA_WRITE_WAIT_S / 2, (
            f'a cancelled write should end the wait at once, not sit out the '
            f'{AF_DATA_WRITE_WAIT_S}s bound (took {elapsed:.3f}s)'
        )

    def test_nothing_queued_needs_no_wait(self):
        """A refused submit returns no waiter, and saved_data_path stays None."""
        runner = self._bare_runner()
        runner._data_write_future = None

        started = time.monotonic()
        runner._await_data_write()

        assert time.monotonic() - started < 0.1
        assert runner.saved_data_path() is None

    def test_the_waiter_is_consumed_so_a_later_sweep_cannot_inherit_it(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter

        runner = self._bare_runner()
        waiter = _ReusableTaskWaiter()
        waiter.set_result(None)
        runner._data_write_future = waiter

        runner._await_data_write()

        assert runner._data_write_future is None


class TestARunUnderALentClaim:
    """A run inside a diagnostic acts under the diagnostic's claim.

    The diagnostic holds the session's claim for its whole length; a run it
    starts must neither be refused by it nor release it when the run ends.
    """

    def test_it_runs_to_completion_and_leaves_the_diagnostic_holding(self, tmp_path):
        rig = _AfRig()
        claim = rig.runner._activity_claim
        diagnostic = claim.try_claim('diagnostic')
        try:
            outcome = rig.run_autofocus(tmp_path, save_data=False, borrowed_claim=diagnostic.lend())
            assert rig.runner.wait_for_run_idle(COMPLETION_TIMEOUT)
            assert outcome.status == 'completed', (outcome.status, outcome.reason, outcome.message)
            assert diagnostic.holds, "the run released the diagnostic's claim when it ended"
            assert claim.owner == 'diagnostic'
            assert rig.scope.illumination.led_lease_purpose is None, (
                'the run left its LED lease on the stack'
            )
        finally:
            diagnostic.release()
            rig.close()

    def test_a_second_run_during_it_is_refused_naming_the_diagnostic(self, tmp_path):
        from modules.exceptions import ProtocolRunRefusedError

        rig = _AfRig()
        diagnostic = rig.runner._activity_claim.try_claim('diagnostic')
        try:
            pending = rig.start_autofocus(
                tmp_path, save_data=False, borrowed_claim=diagnostic.lend()
            )
            with pytest.raises(ProtocolRunRefusedError) as excinfo:
                rig.prepare_autofocus(tmp_path, save_data=False)
            assert excinfo.value.reason == 'exclusive_activity_running', excinfo.value.reason
            assert excinfo.value.holder == 'diagnostic'
            pending.wait(timeout_s=COMPLETION_TIMEOUT)
            assert rig.runner.wait_for_run_idle(COMPLETION_TIMEOUT)
        finally:
            diagnostic.release()
            rig.close()

    def test_a_borrow_whose_lender_has_released_is_refused(self, tmp_path):
        from modules.exceptions import ProtocolRunRefusedError

        rig = _AfRig()
        diagnostic = rig.runner._activity_claim.try_claim('diagnostic')
        borrow = diagnostic.lend()
        diagnostic.release()
        try:
            with pytest.raises(ProtocolRunRefusedError) as excinfo:
                rig.start_autofocus(tmp_path, save_data=False, borrowed_claim=borrow)
            assert excinfo.value.reason == 'exclusive_activity_running'
            assert rig.runner._activity_claim.owner is None, (
                'a refused borrowed start took the claim'
            )
        finally:
            rig.close()

    def test_a_lease_failure_at_start_leaves_the_diagnostic_holding(self, tmp_path, monkeypatch):
        rig = _AfRig()
        diagnostic = rig.runner._activity_claim.try_claim('diagnostic')

        def _refuse(*args, **kwargs):
            raise RuntimeError('lease refused')

        try:
            plan = rig.prepare_autofocus(
                tmp_path, save_data=False, borrowed_claim=diagnostic.lend()
            )
            monkeypatch.setattr(rig.scope.illumination, 'acquire_led_lease', _refuse)
            with pytest.raises(RuntimeError, match='lease refused'):
                rig.runner.start(plan)
            assert diagnostic.holds, (
                "the start's lease-failure path released the diagnostic's claim"
            )
        finally:
            diagnostic.release()
            rig.close()
