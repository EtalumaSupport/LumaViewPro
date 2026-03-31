# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for decomposed protocol modules.

Tests the 5 modules extracted from sequenced_capture_executor.py:
  - protocol_state_machine
  - protocol_callbacks
  - kivy_utils
  - protocol_cleanup
  - protocol_image_writer
"""

import sys
import threading
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies before any module imports
# ---------------------------------------------------------------------------
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = MagicMock()
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('kivy', MagicMock())
sys.modules.setdefault('kivy.clock', MagicMock())
sys.modules.setdefault('kivy.base', MagicMock())
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('psutil', MagicMock())
sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

import pytest

from modules.protocol_state_machine import (
    ProtocolState,
    SequencedCaptureRunMode,
    PROTOCOL_STATE_TRANSITIONS,
    validate_transition,
)
from modules.protocol_callbacks import ProtocolCallbacks


# ===========================================================================
# protocol_state_machine.py
# ===========================================================================

class TestSequencedCaptureRunMode:
    """Verify enum values match expected protocol run modes."""

    def test_all_run_modes_present(self):
        modes = {m.value for m in SequencedCaptureRunMode}
        assert modes == {
            'full_protocol',
            'single_scan',
            'single_zstack',
            'single_autofocus_scan',
            'single_autofocus',
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
            assert state in PROTOCOL_STATE_TRANSITIONS, (
                f"{state} missing from transition table"
            )


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

    @pytest.mark.parametrize("old,new", [
        (ProtocolState.IDLE, ProtocolState.SCANNING),
        (ProtocolState.IDLE, ProtocolState.COMPLETING),
        (ProtocolState.IDLE, ProtocolState.ERROR),
        (ProtocolState.COMPLETING, ProtocolState.RUNNING),
        (ProtocolState.COMPLETING, ProtocolState.SCANNING),
        (ProtocolState.COMPLETING, ProtocolState.ERROR),
        (ProtocolState.ERROR, ProtocolState.RUNNING),
        (ProtocolState.ERROR, ProtocolState.SCANNING),
        (ProtocolState.ERROR, ProtocolState.COMPLETING),
    ])
    def test_invalid_transition_raises(self, old, new):
        with pytest.raises(ValueError, match="Invalid state transition"):
            validate_transition(old, new)

    def test_custom_logger_name_in_error_message(self):
        with pytest.raises(ValueError, match="MyExecutor"):
            validate_transition(
                ProtocolState.IDLE,
                ProtocolState.SCANNING,
                logger_name="MyExecutor",
            )


# ===========================================================================
# protocol_callbacks.py
# ===========================================================================

class TestProtocolCallbacksFromDict:
    """Test ProtocolCallbacks.from_dict() factory."""

    def test_from_dict_full(self):
        fn = lambda: None
        d = {
            'run_complete': fn,
            'leds_off': fn,
            'move_position': fn,
        }
        cb = ProtocolCallbacks.from_dict(d)
        assert cb.run_complete is fn
        assert cb.leds_off is fn
        assert cb.move_position is fn
        # Unset fields stay None
        assert cb.files_complete is None

    def test_from_dict_empty(self):
        cb = ProtocolCallbacks.from_dict({})
        assert cb.run_complete is None
        assert cb.leds_off is None

    def test_from_dict_none(self):
        cb = ProtocolCallbacks.from_dict(None)
        assert cb.run_complete is None

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            'run_complete': lambda: None,
            'totally_bogus_key': 42,
            'another_unknown': 'hello',
        }
        cb = ProtocolCallbacks.from_dict(d)
        assert cb.run_complete is not None
        assert not hasattr(cb, 'totally_bogus_key')


class TestProtocolCallbacksToDict:
    """Test ProtocolCallbacks.to_dict() — must NOT use dataclasses.asdict()."""

    def test_to_dict_only_non_none(self):
        fn = lambda: None
        cb = ProtocolCallbacks(run_complete=fn, leds_off=fn)
        d = cb.to_dict()
        assert 'run_complete' in d
        assert 'leds_off' in d
        # None fields omitted
        assert 'files_complete' not in d
        assert 'move_position' not in d

    def test_to_dict_no_callbacks_set(self):
        cb = ProtocolCallbacks()
        d = cb.to_dict()
        assert d == {}

    def test_to_dict_all_callbacks_set(self):
        import dataclasses
        fields = dataclasses.fields(ProtocolCallbacks)
        fn = lambda: None
        kwargs = {f.name: fn for f in fields}
        cb = ProtocolCallbacks(**kwargs)
        d = cb.to_dict()
        assert len(d) == len(fields)
        for f in fields:
            assert f.name in d

    def test_to_dict_does_not_deepcopy(self):
        """Verify to_dict() returns the original callable references,
        not deep copies. dataclasses.asdict() would deep-copy and crash
        on Kivy bound methods."""
        fn = lambda: None
        cb = ProtocolCallbacks(run_complete=fn)
        d = cb.to_dict()
        assert d['run_complete'] is fn  # same object, not a copy

    def test_roundtrip_dict(self):
        fn_a = lambda: None
        fn_b = lambda: None
        original = {'run_complete': fn_a, 'leds_off': fn_b}
        cb = ProtocolCallbacks.from_dict(original)
        result = cb.to_dict()
        assert result['run_complete'] is fn_a
        assert result['leds_off'] is fn_b


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

class _FakeExecutor:
    """Minimal stand-in for SequentialIOExecutor used in cleanup tests."""

    def __init__(self):
        self.protocol_ended = False
        self.protocol_pending_cleared = False
        self.enabled = False
        self._protocol_queue_active = False
        self._complete_callback = None
        self._finish_called = False

    def protocol_end(self):
        self.protocol_ended = True

    def clear_protocol_pending(self):
        self.protocol_pending_cleared = True

    def enable(self):
        self.enabled = True

    def is_protocol_queue_active(self):
        return self._protocol_queue_active

    def set_protocol_complete_callback(self, callback, cb_args=None, cb_kwargs=None):
        self._complete_callback = callback

    def protocol_finish_then_end(self):
        self._finish_called = True

    def protocol_put(self, task):
        task.action()


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

        run_in_progress = [True]
        io_exec = _FakeExecutor()
        proto_exec = _FakeExecutor()
        af_exec = _FakeExecutor()
        file_exec = _FakeExecutor()
        camera_exec = _FakeExecutor()

        defaults = dict(
            get_state_fn=get_state,
            set_state_fn=set_state,
            run_lock=threading.Lock(),
            protocol_ended=threading.Event(),
            scan_in_progress=threading.Event(),
            leds_state_at_end="off",
            original_led_states={},
            original_autofocus_states={},
            original_gain=-1.0,
            original_exposure=-1.0,
            return_to_position=None,
            disable_saving_artifacts=True,
            protocol=None,
            protocol_execution_record=None,
            scope=MagicMock(),
            callbacks=ProtocolCallbacks(),
            leds_off_fn=lambda: None,
            led_on_fn=lambda **kw: None,
            default_move_fn=lambda **kw: None,
            cancel_scheduled_events_fn=lambda: None,
            io_executor=io_exec,
            protocol_executor=proto_exec,
            autofocus_io_executor=af_exec,
            file_io_executor=file_exec,
            camera_executor=camera_exec,
            set_run_in_progress_fn=lambda v: run_in_progress.__setitem__(0, v),
        )
        defaults.update(overrides)
        return defaults, state, run_in_progress

    def test_cleanup_transitions_to_idle(self):
        from modules.protocol_cleanup import run_cleanup
        args, state, _ = self._make_cleanup_args()
        run_cleanup(**args)
        assert state[0] == ProtocolState.IDLE

    def test_cleanup_sets_run_not_in_progress(self):
        from modules.protocol_cleanup import run_cleanup
        args, _, run_in_progress = self._make_cleanup_args()
        run_cleanup(**args)
        assert run_in_progress[0] is False

    def test_cleanup_fires_run_complete_callback(self):
        from modules.protocol_cleanup import run_cleanup
        completed = []
        cb = ProtocolCallbacks(run_complete=lambda protocol=None: completed.append(True))
        args, _, _ = self._make_cleanup_args(callbacks=cb)
        run_cleanup(**args)
        assert len(completed) == 1

    def test_cleanup_fires_files_complete_when_no_queue(self):
        from modules.protocol_cleanup import run_cleanup
        files_done = []
        cb = ProtocolCallbacks(
            run_complete=lambda protocol=None: None,
            files_complete=lambda protocol=None: files_done.append(True),
        )
        args, _, _ = self._make_cleanup_args(callbacks=cb)
        run_cleanup(**args)
        assert len(files_done) == 1

    def test_cleanup_handles_missing_callbacks_gracefully(self):
        from modules.protocol_cleanup import run_cleanup
        cb = ProtocolCallbacks()  # all None
        args, state, _ = self._make_cleanup_args(callbacks=cb)
        run_cleanup(**args)  # should not raise
        assert state[0] == ProtocolState.IDLE

    def test_cleanup_calls_leds_off(self):
        from modules.protocol_cleanup import run_cleanup
        leds_off_called = []
        args, _, _ = self._make_cleanup_args(
            leds_off_fn=lambda: leds_off_called.append(True),
            leds_state_at_end="off",
        )
        run_cleanup(**args)
        assert len(leds_off_called) == 1

    def test_cleanup_restores_leds_to_original(self):
        from modules.protocol_cleanup import run_cleanup
        restored = []
        original_leds = {
            'Red': {'enabled': True, 'illumination': 50},
            'Green': {'enabled': False, 'illumination': 0},
        }
        args, _, _ = self._make_cleanup_args(
            leds_state_at_end="return_to_original",
            original_led_states=original_leds,
            led_on_fn=lambda color=None, illumination=None, block=True, force=True: restored.append((color, illumination)),
        )
        run_cleanup(**args)
        assert ('Red', 50) in restored
        # Green was not enabled, so should not be restored
        assert ('Green', 0) not in restored

    def test_cleanup_ends_all_executors(self):
        from modules.protocol_cleanup import run_cleanup
        args, _, _ = self._make_cleanup_args()
        run_cleanup(**args)
        assert args['io_executor'].protocol_ended
        assert args['protocol_executor'].protocol_ended
        assert args['autofocus_io_executor'].protocol_ended
        assert args['camera_executor'].enabled

    def test_cleanup_sets_protocol_ended_event(self):
        from modules.protocol_cleanup import run_cleanup
        args, _, _ = self._make_cleanup_args()
        run_cleanup(**args)
        assert args['protocol_ended'].is_set()

    def test_cleanup_clears_scan_in_progress(self):
        from modules.protocol_cleanup import run_cleanup
        args, _, _ = self._make_cleanup_args()
        args['scan_in_progress'].set()
        run_cleanup(**args)
        assert not args['scan_in_progress'].is_set()

    def test_cleanup_returns_to_position(self):
        from modules.protocol_cleanup import run_cleanup
        moved_to = []
        pos = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        args, _, _ = self._make_cleanup_args(
            return_to_position=pos,
            default_move_fn=lambda px=0, py=0, z=0: moved_to.append((px, py, z)),
        )
        run_cleanup(**args)
        assert moved_to == [(1.0, 2.0, 3.0)]

    def test_cleanup_from_error_state(self):
        """Cleanup from ERROR state should transition ERROR -> IDLE."""
        from modules.protocol_cleanup import run_cleanup
        state = [ProtocolState.ERROR]
        args, _, _ = self._make_cleanup_args()
        # Override state functions to use ERROR as starting state
        args['get_state_fn'] = lambda: state[0]
        def set_state(s):
            # ERROR -> IDLE is valid
            if state[0] == ProtocolState.ERROR and s == ProtocolState.IDLE:
                state[0] = s
            elif state[0] == s:
                pass
            else:
                validate_transition(state[0], s)
                state[0] = s
        args['set_state_fn'] = set_state
        run_cleanup(**args)
        assert state[0] == ProtocolState.IDLE


# ===========================================================================
# protocol_image_writer.py
# ===========================================================================

class TestProtocolImageWriterWriteCapture:
    """Test ProtocolImageWriter.write_capture — the file-IO thread method."""

    def _make_writer(self, execution_record=None):
        """Create a ProtocolImageWriter with minimal stubs."""
        from modules.protocol_image_writer import ProtocolImageWriter
        writer = ProtocolImageWriter(
            scope=MagicMock(),
            callbacks=ProtocolCallbacks(),
            protocol_ended=threading.Event(),
            video_write_finished=threading.Event(),
            file_io_executor=_FakeExecutor(),
            protocol_executor=_FakeExecutor(),
            execution_record=execution_record,
            leds_off_fn=lambda: None,
            led_on_fn=lambda **kw: None,
            is_run_in_progress_fn=lambda: True,
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
        assert call_kwargs[1]['capture_result_file_name'] == 'unsaved' or \
               call_kwargs[0][0] == 'unsaved' if call_kwargs[0] else \
               'capture_result_file_name' in call_kwargs[1]

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

    def test_write_capture_failed_image_records_failure(self):
        record = MagicMock()
        writer = self._make_writer(execution_record=record)
        writer.write_capture(
            enable_image_saving=True,
            is_video=False,
            captured_image=False,
            step={'Name': 'test_step'},
            name='test_name',
            step_index=3,
            scan_count=2,
        )
        record.add_step.assert_called_once()
        _, kwargs = record.add_step.call_args
        assert kwargs['capture_result_file_name'] == 'capture_failed'
        assert kwargs['frame_count'] == 0

    def test_write_capture_none_record_with_failed_image_no_crash(self):
        writer = self._make_writer(execution_record=None)
        writer.write_capture(
            enable_image_saving=True,
            is_video=False,
            captured_image=False,
            step={'Name': 'test'},
        )
        # Should not crash — just returns
