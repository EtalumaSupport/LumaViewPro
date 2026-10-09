# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for decomposed protocol modules.

Tests the 5 modules extracted from sequenced_capture_runner.py:
  - protocol_state_machine
  - run_events
  - kivy_utils
  - protocol_cleanup
  - protocol_image_writer
"""

import datetime
import threading
from unittest.mock import MagicMock

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time.

import pytest

from tests.protocol_drives import lent_run_claim
from tests.frame_records import plate
from modules.protocol_image_writer import RunWriteBatch
from modules.sequential_io_executor import ENQUEUED
from modules.protocol_state_machine import (
    ProtocolState,
    SequencedCaptureRunMode,
    PROTOCOL_STATE_TRANSITIONS,
    validate_transition,
)
from modules.run_events import RunEvents


# ===========================================================================
# protocol_state_machine.py
# ===========================================================================


from modules.run_outcome import EndingLatch, RunEnding
from tests.scope_fakes import swap_lanes


class TestSequencedCaptureRunMode:
    """Verify enum values match expected protocol run modes."""

    def test_all_run_modes_present(self):
        modes = {m.value for m in SequencedCaptureRunMode}
        assert modes == {
            'full_protocol',
            'single_scan',
            'single_zstack',
            'single_autofocus',
            'single_autofocus_scan',
            'single_composite',
        }

    def test_enum_access_by_name(self):
        assert SequencedCaptureRunMode.FULL_PROTOCOL.value == 'full_protocol'
        assert SequencedCaptureRunMode.SINGLE_SCAN.value == 'single_scan'


class TestProtocolState:
    """Verify the state enum and transition table."""

    def test_all_states_present(self):
        states = {s.value for s in ProtocolState}
        assert states == {'idle', 'running', 'scanning', 'completing', 'error'}

    def test_every_state_has_transition_entry(self):
        for state in ProtocolState:
            assert state in PROTOCOL_STATE_TRANSITIONS, f'{state} missing from transition table'


class TestValidateTransition:
    """Test all valid and invalid state transitions."""

    # --- Valid transitions ---

    def test_idle_to_running(self):
        validate_transition(ProtocolState.IDLE, ProtocolState.RUNNING)

    def test_running_to_scanning(self):
        validate_transition(ProtocolState.RUNNING, ProtocolState.SCANNING)

    def test_running_to_completing(self):
        validate_transition(ProtocolState.RUNNING, ProtocolState.COMPLETING)

    def test_running_to_error(self):
        validate_transition(ProtocolState.RUNNING, ProtocolState.ERROR)

    def test_scanning_to_running(self):
        validate_transition(ProtocolState.SCANNING, ProtocolState.RUNNING)

    def test_scanning_to_completing(self):
        validate_transition(ProtocolState.SCANNING, ProtocolState.COMPLETING)

    def test_scanning_to_error(self):
        validate_transition(ProtocolState.SCANNING, ProtocolState.ERROR)

    def test_completing_to_idle(self):
        validate_transition(ProtocolState.COMPLETING, ProtocolState.IDLE)

    def test_error_to_idle(self):
        validate_transition(ProtocolState.ERROR, ProtocolState.IDLE)

    # --- Same-state no-op ---

    def test_same_state_is_noop(self):
        for state in ProtocolState:
            validate_transition(state, state)  # should not raise

    # --- Invalid transitions ---

    @pytest.mark.parametrize(
        'old,new',
        [
            (ProtocolState.IDLE, ProtocolState.SCANNING),
            (ProtocolState.IDLE, ProtocolState.COMPLETING),
            (ProtocolState.IDLE, ProtocolState.ERROR),
            (ProtocolState.COMPLETING, ProtocolState.RUNNING),
            (ProtocolState.COMPLETING, ProtocolState.SCANNING),
            (ProtocolState.COMPLETING, ProtocolState.ERROR),
            (ProtocolState.ERROR, ProtocolState.RUNNING),
            (ProtocolState.ERROR, ProtocolState.SCANNING),
            (ProtocolState.ERROR, ProtocolState.COMPLETING),
        ],
    )
    def test_invalid_transition_raises(self, old, new):
        with pytest.raises(ValueError, match='Invalid state transition'):
            validate_transition(old, new)

    def test_custom_logger_name_in_error_message(self):
        with pytest.raises(ValueError, match='MyExecutor'):
            validate_transition(
                ProtocolState.IDLE,
                ProtocolState.SCANNING,
                logger_name='MyExecutor',
            )


# ===========================================================================
# run_events.py
# ===========================================================================


class TestRunEvents:
    """``RunEvents``: optional handlers, each named; a misspelt one fails at construction."""

    def test_unset_handlers_are_none(self):
        fn = lambda *a: None
        events = RunEvents(run_ended=fn, step_started=fn)
        assert events.run_ended is fn
        assert events.step_started is fn
        assert events.files_written is None
        assert events.video_progress is None

    def test_a_misspelt_handler_is_refused(self):
        with pytest.raises(TypeError):
            RunEvents(run_complete=lambda *a: None)

    def test_a_positional_handler_is_refused(self):
        with pytest.raises(TypeError):
            RunEvents(lambda *a: None)

    def test_the_events_cannot_be_rewritten_after_construction(self):
        import dataclasses

        events = RunEvents()
        with pytest.raises(dataclasses.FrozenInstanceError):
            events.run_ended = lambda *a: None


class TestRunEventsHaveNoAutofocusRestore:
    """A run restores no layer's autofocus flag, so the events carry no restorer.

    Nothing in a run reads the flags; a restore at its end could only
    revert a change the user made during it. A field for one would bring
    that back. Assert on dataclasses.fields so it cannot come back under a
    different construction site.
    """

    def test_no_restore_autofocus_state_field(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(RunEvents)}
        assert 'restore_autofocus_state' not in names, (
            'a run restores no autofocus flag, so the events carry no '
            f'restorer. fields={sorted(names)}'
        )

    def test_construction_with_the_removed_field_is_refused(self):
        with pytest.raises(TypeError):
            RunEvents(restore_autofocus_state=lambda **kw: None)


# ===========================================================================
# kivy_utils.py
# ===========================================================================


class TestScheduleUI:
    """Test schedule_ui falls back to direct call when no Kivy event loop."""

    def test_schedule_ui_calls_directly_without_kivy(self):
        from modules.kivy_utils import schedule_ui

        called_with = []
        schedule_ui(lambda dt: called_with.append(dt))
        assert called_with == [0]

    def test_schedule_ui_with_timeout(self):
        from modules.kivy_utils import schedule_ui

        called = []
        schedule_ui(lambda dt: called.append(dt), timeout=0.5)
        assert len(called) == 1

    def test_schedule_ui_passes_dt_zero(self):
        """schedule_ui passes dt=0 to the function (matching Clock convention)."""
        from modules.kivy_utils import schedule_ui

        received_dt = []
        schedule_ui(lambda dt: received_dt.append(dt))
        assert received_dt == [0]

    def test_schedule_ui_multiple_calls(self):
        """schedule_ui can be called multiple times."""
        from modules.kivy_utils import schedule_ui

        count = []
        for _ in range(5):
            schedule_ui(lambda dt: count.append(1))
        assert len(count) == 5


# ===========================================================================
# protocol_cleanup.py
# ===========================================================================


def test_cleanup_never_reaches_a_settings_module_global():
    """Cleanup touches no settings.

    The module-level settings dict is None in every headless process, so
    a cleanup that read it would fail there and bury the failure in the
    cleanup-error summary; and a cleanup that wrote the session's settings
    would revert what the user changed while the run went.
    """
    import pathlib as _pathlib

    source = (
        _pathlib.Path(__file__).resolve().parent.parent / 'modules' / 'protocol_cleanup.py'
    ).read_text()
    offenders = [
        f'{i}: {line.strip()}'
        for i, line in enumerate(source.splitlines(), 1)
        if 'settings[' in line or 'settings_init' in line
    ]
    assert not offenders, f'protocol_cleanup must touch no settings; found {offenders}'


class _FakeExecutor:
    """Minimal stand-in for SequentialIOExecutor used in cleanup tests.

    One instance per lane. IO and CAMERA use the run-mode members; FILE
    uses ``put``, which a run's write batch submits its writes through.
    """

    def __init__(self):
        self.protocol_ended = False
        self.protocol_pending_cleared = False

    def protocol_end(self):
        self.protocol_ended = True

    def clear_protocol_pending(self):
        self.protocol_pending_cleared = True

    def wait_for_idle(self, timeout: float = 1.0) -> bool:
        # No worker thread in this stub; nothing in flight to wait for.
        # Return True so the cleanup path proceeds without thinking it
        # timed out and skipping the rest of the teardown.
        return True

    def protocol_put(self, task):
        task.action()

    def put(self, task, return_future=False):
        # No queue in this stub: the task runs inline, so a write handed
        # to a batch on this lane has landed by the time put returns.
        task.action(*task.args, **task.kwargs)
        return ENQUEUED


class _HeldFileLane:
    """A FILE lane that takes a write and holds it until the test runs it."""

    def __init__(self):
        self.held = []

    def put(self, task, return_future=False):
        self.held.append(task)
        return ENQUEUED

    def run_held(self):
        while self.held:
            task = self.held.pop(0)
            task.action(*task.args, **task.kwargs)


class TestRunCleanup:
    """Test protocol_cleanup.run_cleanup logic."""

    def _make_cleanup_args(self, **overrides):
        """Build a full keyword-argument dict for run_cleanup with sane defaults."""
        state = [ProtocolState.RUNNING]

        def get_state():
            return state[0]

        def set_state(s):
            validate_transition(state[0], s)
            state[0] = s

        io_exec = _FakeExecutor()
        # autofocus_thread replaces autofocus_io_executor in Stage B2;
        # MagicMock so the cleanup tests can assert abort() was called.
        af_thread = MagicMock()
        file_exec = _FakeExecutor()
        camera_exec = _FakeExecutor()
        ending = overrides.pop(
            'ending',
            RunEnding('completed', 'completed', 'Protocol Complete', 'The run finished normally.'),
        )

        defaults = {
            'get_state_fn': get_state,
            'set_state_fn': set_state,
            'scan_in_progress': threading.Event(),
            'forced_dark': False,
            'leds_state_at_end': 'off',
            'original_led_states': {},
            'saved_camera_state': None,
            'return_to_position': None,
            'scope': MagicMock(),
            'apply_led_transition_fn': lambda transition, ctx: None,
            'default_move_fn': lambda **kw: None,
            'cancel_scheduled_events_fn': lambda: None,
            'autofocus_thread': af_thread,
            'write_batch': RunWriteBatch(file_exec),
            'ending': ending,
            'record_cleanup_failures': lambda steps: None,
        }
        defaults.update(overrides)
        # Cleanup ends the scope's own IO and CAMERA lanes; the fakes, or a
        # test's own, stand in for them there.
        swap_lanes(
            defaults['scope'],
            io=defaults.pop('io_executor', io_exec),
            camera=defaults.pop('camera_executor', camera_exec),
        )
        return defaults, state

    def test_cleanup_hands_the_run_back_completing_and_does_not_end_it(self):
        """Cleanup moves the run into COMPLETING and leaves it there.

        Ending the run belongs to the caller's finally, which runs even
        when a step in here raises. A second writer at the end of this
        function is what used to admit the next run onto resources the
        teardown was still handing back.
        """
        from modules.protocol_cleanup import run_cleanup

        args, state = self._make_cleanup_args()
        run_cleanup(**args)
        assert state[0] == ProtocolState.COMPLETING

    def test_the_camera_restore_is_the_public_member(self):
        """Cleanup restores the camera through ``restore_camera_state``, whose
        own dispatch puts it on the camera lane under the run's taking --
        cleanup submits nothing to the camera executor itself, so there is
        no inline branch for a closed lane to route it onto."""
        from modules.protocol_cleanup import run_cleanup
        from unittest.mock import MagicMock

        camera = MagicMock()
        scope = MagicMock()
        args, _ = self._make_cleanup_args(
            camera_executor=camera,
            saved_camera_state={'gain': 1.0, 'exposure': 10.0},
            scope=scope,
        )
        run_cleanup(**args)

        scope.imaging.restore_camera_state.assert_called_once_with({'gain': 1.0, 'exposure': 10.0})
        camera.put.assert_not_called()
        camera.protocol_put.assert_not_called()

    def test_files_written_is_handed_over_after_the_close(self):
        """run_cleanup sends neither run_ended nor files_written: the run
        sends both once it has let go of the scope. The runner closes the
        batch before the run ends, so a next run reads it as draining, and a
        batch with nothing outstanding completes at the close -- but its
        files_written goes out only when the run's end hands the actions
        over.
        """
        from types import SimpleNamespace

        from modules.protocol_cleanup import run_cleanup
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        fired = []
        events = RunEvents(
            run_ended=lambda *a: fired.append('run_ended'),
            files_written=lambda *a: fired.append('files_written'),
        )
        args, _ = self._make_cleanup_args()
        run_cleanup(**args)
        assert fired == [], "cleanup must leave both events to the run's end"

        # The runner's close, on a runner that saves no record.
        runner = SimpleNamespace(
            _disable_saving_artifacts=True,
            _protocol_execution_record=None,
            _events=events,
            _protocol=None,
            _run_dir=None,
            _on_run_idle=None,
            _run_mode=None,
            LOGGER_NAME='TEST',
        )
        batch = args['write_batch']
        files_written = SequencedCaptureRunner._close_run_writes(
            runner, batch, args['ending'], MagicMock()
        )
        assert batch.wait_complete(0), 'a batch with nothing outstanding completes at the close'
        assert fired == [], 'closing the batch must not send files_written'
        batch.when_complete(files_written)
        assert fired == ['files_written']

    def test_cleanup_ends_all_executors(self):
        from modules.protocol_cleanup import run_cleanup

        args, _ = self._make_cleanup_args()
        run_cleanup(**args)
        assert args['scope'].io_lane().protocol_ended
        assert args['autofocus_thread'].abort.called
        assert args['scope'].camera_lane().protocol_ended
        assert args['scope'].camera_lane().protocol_pending_cleared

    def test_cleanup_clears_scan_in_progress(self):
        from modules.protocol_cleanup import run_cleanup

        args, _ = self._make_cleanup_args()
        args['scan_in_progress'].set()
        run_cleanup(**args)
        assert not args['scan_in_progress'].is_set()

    def test_cleanup_returns_to_position(self):
        from modules.protocol_cleanup import run_cleanup

        moved_to = []
        pos = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        args, _ = self._make_cleanup_args(
            return_to_position=pos,
            default_move_fn=lambda px=0, py=0, z=0: moved_to.append((px, py, z)),
        )
        run_cleanup(**args)
        assert moved_to == [(1.0, 2.0, 3.0)]

    def test_cleanup_holds_error_through_the_teardown(self):
        """A run that died holds ERROR for the whole teardown.

        ERROR is what tells the rest of cleanup this was a fault, and the
        run still holds the scope while it unwinds. The caller's finally is what returns it
        to IDLE, once everything has been handed back.
        """
        from modules.protocol_cleanup import run_cleanup

        state = [ProtocolState.ERROR]
        args, _ = self._make_cleanup_args()
        args['get_state_fn'] = lambda: state[0]

        def set_state(s):
            validate_transition(state[0], s)
            state[0] = s

        args['set_state_fn'] = set_state
        run_cleanup(**args)
        assert state[0] == ProtocolState.ERROR

    def test_pending_writes_are_written_however_the_run_aborts(self):
        """An image the run captured is written, however the run ends: an
        ERROR abort (hardware fault) keeps its pending writes exactly as a
        user Stop does. Cleanup ends no write; the held write is still the
        batch's after cleanup, and lands when the lane reaches it.
        """
        from modules.protocol_cleanup import run_cleanup

        # ERROR abort (hardware fault): the pending write is kept.
        err_state = [ProtocolState.ERROR]
        err_lane = _HeldFileLane()
        written = []
        args, _ = self._make_cleanup_args(write_batch=RunWriteBatch(err_lane))
        args['write_batch'].submit(
            lambda: written.append('error'), {}, what='The image e', pace_until=None
        )
        args['get_state_fn'] = lambda: err_state[0]

        def set_err_state(s):
            if err_state[0] == ProtocolState.ERROR and s == ProtocolState.IDLE:
                err_state[0] = s
            elif err_state[0] == s:
                pass
            else:
                validate_transition(err_state[0], s)
                err_state[0] = s

        args['set_state_fn'] = set_err_state
        run_cleanup(**args)
        assert args['write_batch'].pending == 1, (
            'an ERROR-state abort must keep the pending file write, not drop it'
        )
        err_lane.run_held()
        assert written == ['error'] and args['write_batch'].pending == 0

        # Non-ERROR end/abort (user Stop, RUNNING -> COMPLETING -> IDLE): pending
        # writes drain, not dropped.
        stop_lane = _HeldFileLane()
        args2, _ = self._make_cleanup_args(write_batch=RunWriteBatch(stop_lane))
        args2['write_batch'].submit(
            lambda: written.append('stop'), {}, what='The image s', pace_until=None
        )
        run_cleanup(**args2)
        assert args2['write_batch'].pending == 1, (
            'a non-ERROR (user Stop) abort must drain pending writes, not drop them'
        )
        stop_lane.run_held()
        assert written == ['error', 'stop'] and args2['write_batch'].pending == 0


# ===========================================================================
# protocol_image_writer.py
# ===========================================================================


class TestProtocolImageWriterWriteCapture:
    """Test ProtocolImageWriter.write_capture -- the file-IO thread method."""

    def _make_writer(self, execution_record=None):
        """Create a ProtocolImageWriter with minimal stubs."""
        from modules.image_mode import ImageCaptureConfig
        from modules.protocol_image_writer import ProtocolImageWriter

        writer = ProtocolImageWriter(
            scope=MagicMock(),
            events=RunEvents(),
            aborted=threading.Event(),
            write_batch=RunWriteBatch(_FakeExecutor()),
            abort_fn=lambda: None,
            fatal_abort_event=threading.Event(),
            ending=EndingLatch(),
            execution_record=execution_record,
            leds_off_fn=lambda: None,
            is_run_in_progress_fn=lambda: True,
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            timestamp_overlay=True,
            video_max_fps=0,
            engineering_mode=False,
            run_claim=lent_run_claim(),
            labware=plate(),
            to_plate=None,
            captures_asked=1,
        )
        return writer

    def test_write_capture_saving_disabled_records_unsaved(self):
        record = MagicMock()
        writer = self._make_writer(execution_record=record)
        writer.write_capture(
            enable_image_saving=False,
            step={'Name': 'test'},
        )
        record.add_step.assert_called_once()
        call_kwargs = record.add_step.call_args
        assert (
            call_kwargs[1]['capture_result_file_name'] == 'unsaved'
            or call_kwargs[0][0] == 'unsaved'
            if call_kwargs[0]
            else 'capture_result_file_name' in call_kwargs[1]
        )

    def test_write_capture_saving_disabled_correct_unsaved_value(self):
        record = MagicMock()
        writer = self._make_writer(execution_record=record)
        writer.write_capture(
            enable_image_saving=False,
            step={'Name': 'test'},
            step_index=0,
            scan_count=1,
        )
        record.add_step.assert_called_once()
        # Check the keyword arguments
        _, kwargs = record.add_step.call_args
        assert kwargs['capture_result_file_name'] == 'unsaved'

    def test_write_capture_none_execution_record_no_crash(self):
        writer = self._make_writer(execution_record=None)
        # Should not raise even with no execution record
        writer.write_capture(
            enable_image_saving=False,
            step={'Name': 'test'},
        )

    def test_write_capture_none_record_with_failed_image_no_crash(self):
        writer = self._make_writer(execution_record=None)
        writer.write_capture(
            enable_image_saving=True,
            captured_image=None,
            step={'Name': 'test'},
        )
        # Should not crash -- just returns


class TestRunCleanupCancelledHandoff:
    """A superseding run/abort cycle cancels queued cleanup tasks (LED
    restore, return-to-position) via the executor's pending-clear. That is
    a normal ownership hand-off: the new cycle sets LED state and stage
    position itself. Treating the CancelledError as a cleanup failure
    fired an error popup per cycle when the run button was clicked
    rapidly -- nine popups in four seconds on the bench.
    """

    def _args(self, **overrides):
        helper = TestRunCleanup()
        return helper._make_cleanup_args(**overrides)

    def test_cancelled_led_restore_is_not_a_cleanup_error(self, centre_posts):
        from concurrent.futures import CancelledError
        from modules.notification_center import Severity
        from modules.protocol_cleanup import run_cleanup

        def cancelled_apply(transition, ctx):
            raise CancelledError()

        args, _ = self._args(apply_led_transition_fn=cancelled_apply)
        run_cleanup(**args)
        assert [n for n in centre_posts if n.severity >= Severity.WARNING] == []

    def test_cancelled_return_move_is_not_a_cleanup_error(self, centre_posts):
        from concurrent.futures import CancelledError
        from modules.notification_center import Severity
        from modules.protocol_cleanup import run_cleanup

        def cancelled_move(**kw):
            raise CancelledError()

        args, _ = self._args(
            return_to_position={'x': 1.0, 'y': 2.0, 'z': 3.0},
            default_move_fn=cancelled_move,
        )
        run_cleanup(**args)
        assert [n for n in centre_posts if n.severity >= Severity.WARNING] == []


class TestFinalStepKeepsLedWhenCleanupRestoresIt:
    """The final step of the final scan must not turn its LED off when
    cleanup is about to re-light the same channel (it was lit before the
    run): the off->on pair is a visible end-of-acquire flicker on a
    z-stack started from a live-view-lit channel. Non-final scans keep
    the LED off so inter-scan waits stay dark (sample safety).

    The runner drives the boundary decision through the LED authority
    (apply_led_transition(STEP_BOUNDARY, ctx)) after a completed capture,
    so the assertion is on the authority's target set: a non-empty target
    holds the channel lit, an empty target lets it go dark.
    """

    def _boundary_target_for(self, *, run_mode, original_led_states, n_scans=1):
        """Drive the final step of a scan and return the STEP_BOUNDARY target
        the runner asks the authority for (empty set = goes dark, non-empty =
        held lit)."""
        from unittest.mock import MagicMock

        from modules.lumascope_api.illumination import LedLease, LedTransition
        from tests.protocol_drives import protocol_step, scan_ready_runner

        runner = scan_ready_runner(
            protocol_step(),
            _disable_saving_artifacts=False,
            _run_dir=MagicMock(),
            _run_mode=run_mode,
            _original_led_states=original_led_states,
            _n_scans=n_scans,
        )
        captured = {}

        def _spy(transition, ctx):
            if transition is LedTransition.STEP_BOUNDARY:
                captured['ctx'] = ctx

        # Identity and driver resolvers agree on a real channel number, as
        # they do on real hardware: the hold decision compares the step's
        # channel against the run-end snapshot, and a bare MagicMock would
        # hand each resolver a different sentinel object.
        def resolver(c):
            return {'BF': 3}.get(c)

        runner._scope.illumination.color2ch.side_effect = resolver
        runner._scope.illumination.state_color2ch.side_effect = resolver

        runner._step_executor.apply_led_transition = _spy
        runner._step_executor.scan_iterate()
        assert runner._image_writer.capture.called, 'the step must reach capture'
        assert 'ctx' in captured, 'the runner must drive the STEP_BOUNDARY decision'
        return LedLease.target_leds(LedTransition.STEP_BOUNDARY, captured['ctx'])

    @staticmethod
    def _lit_before_run():
        return {'BF': {'enabled': True, 'illumination_ma': 100.0}}

    def test_final_scan_restore_keeps_lit_channel(self):
        assert self._boundary_target_for(
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            original_led_states=self._lit_before_run(),
        ), 'cleanup is about to re-light this channel; turning it off here blinks'

    def test_leds_off_at_end_never_keeps(self):
        assert not self._boundary_target_for(
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            original_led_states=self._lit_before_run(),
        ), 'a run that ends dark must always end dark'

    def test_non_final_scan_stays_dark(self):
        assert not self._boundary_target_for(
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            original_led_states=self._lit_before_run(),
            n_scans=2,
        ), 'inter-scan waits must run dark (sample safety) on non-final scans'


# ===========================================================================
# protocol_execution_record.py -- end-of-run reconciliation
# ===========================================================================


class TestProtocolRecordReconciliation:
    """Every capture the protocol attempts must leave exactly one row in the
    execution record. At protocol end the record reconciles the count of
    attempts against the count of rows actually written; a shortfall means a
    capture vanished without a row (a silent data gap) OR a row write itself
    failed, and the user is warned once."""

    def _make_record(self, tmp_path):
        from modules.protocol_execution_record import ProtocolExecutionRecord

        return ProtocolExecutionRecord(
            protocol_file_loc=tmp_path / 'protocol.tsv',
            outfile=tmp_path / 'record.tsv',
        )

    def _add_row(self, rec, name):
        rec.add_step(
            capture_result_file_name=name,
            step_name='step',
            step_index=0,
            scan_count=0,
            timestamp=datetime.datetime(2026, 6, 15, 12, 0, 0),
        )

    def _capture_warnings(self, monkeypatch):
        # The shortfall is raised to the run's files completion, whose
        # reporter tells the person; _complete collects what was raised.
        return []

    def _complete(self, rec, fired):
        from modules.exceptions import RecordIncompleteError

        try:
            rec.complete()
        except RecordIncompleteError as shortfall:
            fired.append(((shortfall.title, str(shortfall)), {}))

    def test_shortfall_fires_one_warning(self, tmp_path, monkeypatch):
        fired = self._capture_warnings(monkeypatch)
        rec = self._make_record(tmp_path)
        # Three captures attempted; only two leave a row -- one silent gap.
        rec.note_capture_attempt()
        rec.note_capture_attempt()
        rec.note_capture_attempt()
        self._add_row(rec, 'a')
        self._add_row(rec, 'b')
        self._complete(rec, fired)
        assert len(fired) == 1, 'a record shortfall must fire exactly one warning'
        # The body names the gap count so the L1 reader knows the magnitude.
        body = ' '.join(str(x) for x in fired[0][0])
        assert '1' in body and '3' in body

    def test_clean_run_fires_nothing(self, tmp_path, monkeypatch):
        fired = self._capture_warnings(monkeypatch)
        rec = self._make_record(tmp_path)
        rec.note_capture_attempt()
        rec.note_capture_attempt()
        self._add_row(rec, 'a')
        self._add_row(rec, 'b')
        self._complete(rec, fired)
        assert fired == [], 'attempts == rows recorded -> no warning'

    def test_failed_row_write_is_a_shortfall(self, tmp_path, monkeypatch):
        # EXC-M-5: if add_step's own disk write raises, the row is lost but the
        # attempt counted -- reconciliation must still catch it.
        fired = self._capture_warnings(monkeypatch)
        rec = self._make_record(tmp_path)
        rec.note_capture_attempt()
        rec.note_capture_attempt()
        self._add_row(rec, 'a')
        monkeypatch.setattr(rec, '_outfile', tmp_path / 'nonexistent_dir' / 'record.tsv')
        self._add_row(rec, 'b')  # write raises inside add_step, swallowed + logged
        self._complete(rec, fired)
        assert len(fired) == 1, 'a failed row write must register as a shortfall'

    def test_an_aborted_run_is_reconciled_too(self, tmp_path, monkeypatch):
        # An aborted run writes every image it captured, as any other run
        # does, so a gap on abort is a real shortfall, not an expected one:
        # the record reconciles however the run ended.
        fired = self._capture_warnings(monkeypatch)
        rec = self._make_record(tmp_path)
        rec.note_capture_attempt()
        rec.note_capture_attempt()
        rec.note_capture_attempt()
        self._add_row(rec, 'a')
        self._complete(rec, fired)
        assert len(fired) == 1, 'an aborted run with a gap must warn like any other run'
