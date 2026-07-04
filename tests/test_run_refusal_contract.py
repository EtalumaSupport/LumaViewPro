# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the sequenced-run start contract.

The start sequence is prepare() -> start(plan):

- prepare() performs every refusal gate and raises
  ProtocolRunRefusedError (after notifying the user exactly once)
  without committing any runner state or touching disk. A refused
  prepare is observationally a no-op: every getter still answers for
  the previous run.
- start(plan) is the commitment point: once entered, the terminal
  run_complete callback fires exactly once on every path -- normal
  completion, abort, or a setup failure that unwinds as an
  immediately-failed run (status 'failed_at_start'). No caller can
  wait forever on a run that never started.

Covered here:
1. A refused headless run raises instead of silently wedging the
   session in protocol-running state.
2. A failure after the commitment point (run-dir init) unwinds as a
   failed-at-start run: terminal callback with failed status, no
   orphan empty run directory, runner reusable. The prepare-side twin
   confirms a refusal preserves the previous run's state and disk.
3. A failure in the last commit step before dispatch cannot
   half-start the runner.
4. Every refusal reason notifies exactly once and the raised error
   carries the matching reason code.

The caller-ordering half of the contract (UI starters commit
running-state only between prepare and start, inside the shared
refusal boundary) is locked by tests/test_protocol_start_refusal_ui_gate.py.
"""

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Mock settings_init before
# sequenced_capture_runner imports it.
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

from modules.exceptions import ProtocolRunRefusedError
from modules.lumascope_api import Lumascope
from modules.protocol import Protocol
from modules.sequenced_capture_runner import SequencedCaptureRunner
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import SequentialIOExecutor

COMPLETION_TIMEOUT = 15  # seconds -- generous for CI

TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


# ---------------------------------------------------------------------------
# Harness (mirrors tests/test_protocol_execution.py)
# ---------------------------------------------------------------------------


def _make_simulated_scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s.imaging.start_streaming()
    return s


def _make_executors():
    from modules.protocol_thread import ProtocolThread

    execs = {
        'io': SequentialIOExecutor(name='REFUSAL_IO'),
        'file_io': SequentialIOExecutor(name='REFUSAL_FILE'),
        'camera': SequentialIOExecutor(name='REFUSAL_CAMERA'),
        'autofocus': SequentialIOExecutor(name='REFUSAL_AF'),
    }
    for e in execs.values():
        e.start()
    pt = ProtocolThread()
    pt.start()
    execs['protocol'] = pt
    return execs


def _shutdown_executors(execs):
    for name, e in execs.items():
        try:
            if name == 'protocol':
                e.stop(timeout=2.0)
            else:
                e.shutdown()
        except Exception:
            pass


def _make_autogain_settings():
    return {
        'target_brightness': 0.3,
        'min_gain_db': 0.0,
        'max_gain_db': 20.0,
        'max_duration': datetime.timedelta(seconds=1),
    }


def _make_image_capture_config():
    return {
        'output_format': {
            'live': 'TIFF',
            'sequenced': 'TIFF',
        },
        'capture_depth': 8,
        'save_encoding': '8bit',
    }


def _build_real_protocol(rows, period_min=1.0, duration_hrs=1.0):
    import pandas as pd

    df = pd.DataFrame(rows)
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': df,
        'period': datetime.timedelta(minutes=period_min),
        'duration': datetime.timedelta(hours=duration_hrs),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)


def _make_single_step_protocol(color='BF'):
    step = {
        'Name': 'A1_test',
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': False,
        'Color': color,
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
    }
    return _build_real_protocol([step])


@pytest.fixture
def scope():
    s = _make_simulated_scope()
    yield s
    s.imaging.stop_streaming()
    s.disconnect()


@pytest.fixture
def executors():
    execs = _make_executors()
    yield execs
    _shutdown_executors(execs)


@pytest.fixture
def executor(scope, executors):
    from modules.coord_transformations import CoordinateTransformer
    from modules.labware_loader import WellPlateLoader

    mock_af = MagicMock()
    mock_af.reset = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.complete = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.result = MagicMock(return_value=None)
    mock_af.best_focus_position = MagicMock(return_value=5000.0)
    mock_af.run_in_progress = MagicMock(return_value=False)

    exc = SequencedCaptureRunner(
        scope=scope,
        stage_offset={'x': 0.0, 'y': 0.0},
        io_executor=executors['io'],
        protocol_thread=executors['protocol'],
        file_io_executor=executors['file_io'],
        camera_executor=executors['camera'],
        autofocus_thread=MagicMock(is_running=False),
        autofocus_runner=mock_af,
    )
    exc._wellplate_loader = WellPlateLoader()
    exc._coordinate_transformer = CoordinateTransformer()
    return exc


def _prepare(executor, protocol, tmp_path, callbacks=None):
    cbs = {
        'go_to_step': lambda **kw: None,
        'move_position': lambda axis: None,
        # Restore AF states through a callback so cleanup does not
        # depend on the global settings module, which other test files
        # replace with import-order-dependent stand-ins.
        'restore_autofocus_state': lambda **kw: None,
    }
    if callbacks:
        cbs.update(callbacks)
    return executor.prepare(
        protocol=protocol,
        run_trigger_source='test',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='refusal_contract',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks=cbs,
        leds_state_at_end='off',
        initial_autofocus_states={
            'BF': False,
            'PC': False,
            'DF': False,
            'Red': False,
            'Green': False,
            'Blue': False,
            'Lumi': False,
        },
    )


def _run_to_completion(executor, protocol, tmp_path):
    done = threading.Event()
    plan = _prepare(
        executor, protocol, tmp_path, callbacks={'run_complete': lambda **kw: done.set()}
    )
    executor.start(plan)
    assert done.wait(timeout=COMPLETION_TIMEOUT), 'run did not complete within timeout'


def _wait_for_file_queue_drain(executor, timeout=5.0):
    deadline = time.monotonic() + timeout
    while executor.file_io_executor.is_protocol_queue_active():
        if time.monotonic() > deadline:
            raise TimeoutError('file_io_executor did not drain in time')
        time.sleep(0.05)


def _wait_for_executors_out_of_protocol_mode(executor, timeout=5.0):
    """protocol_finish drains asynchronously (dispatcher cycle), so poll."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not executor._io_executor.is_protocol_running() and not (
            executor.file_io_executor.is_protocol_running()
        ):
            return True
        time.sleep(0.05)
    return False


def _capture_notifications(monkeypatch):
    """Route both severities of the notification singleton to one list."""
    import modules.notification_center as notification_center

    captured = []
    monkeypatch.setattr(
        notification_center.notifications,
        'error',
        lambda *args, **kwargs: captured.append(('error', args)),
    )
    monkeypatch.setattr(
        notification_center.notifications,
        'warning',
        lambda *args, **kwargs: captured.append(('warning', args)),
    )
    return captured


# ---------------------------------------------------------------------------
# 1. A refused headless run raises; the session never wedges in
#    protocol-running state and stays usable.
# ---------------------------------------------------------------------------


class TestHeadlessRefusalDoesNotHang:
    def test_refused_single_scan_raises_and_session_stays_usable(self, tmp_path):
        from modules.scope_session import ScopeSession

        settings = {
            'BF': {'autofocus': False},
            'PC': {'autofocus': False},
            'DF': {'autofocus': False},
            'Red': {'autofocus': False},
            'Green': {'autofocus': False},
            'Blue': {'autofocus': False},
            'Lumi': {'autofocus': False},
            'stage_offset': {'x': 0.0, 'y': 0.0},
            'live_folder': str(tmp_path),
            'protocol': {
                'autogain': {
                    'target_brightness': 0.3,
                    'max_duration_seconds': 1.0,
                    'min_gain_db': 0.0,
                    'max_gain_db': 20.0,
                },
            },
        }
        session = ScopeSession.create_headless(settings=settings)
        session.start_executors()
        runner = session.create_protocol_runner()
        try:
            # First: a valid run completes and arms wait_for_completion.
            done = threading.Event()
            runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='refusal_headless_first',
                parent_dir=str(tmp_path),
                callbacks={
                    'run_complete': lambda **kw: done.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert done.wait(timeout=COMPLETION_TIMEOUT), 'valid first run did not complete'
            assert runner.wait_for_completion(timeout=COMPLETION_TIMEOUT), (
                'wait_for_completion must report the completed first run'
            )
            assert not session.is_protocol_running

            # A refused run raises out of run_single_scan; nothing waits.
            with pytest.raises(ProtocolRunRefusedError):
                runner.run_single_scan(
                    protocol=_build_real_protocol([]),
                    sequence_name='refusal_headless_refused',
                    parent_dir=str(tmp_path),
                )
            assert not session.is_protocol_running, (
                'a refused run must not leave session.protocol_running set'
            )
            # The completion event was never re-armed for the refused run,
            # so a caller polling wait_for_completion returns immediately
            # instead of hanging until timeout.
            t0 = time.monotonic()
            assert runner.wait_for_completion(timeout=2), (
                'a refusal must not clear the completion event; callers '
                'polling wait_for_completion would hang on a run that '
                'never started'
            )
            assert time.monotonic() - t0 < 1.0, (
                'wait_for_completion should return immediately after a refusal'
            )

            # The session is not wedged: a subsequent valid run works.
            done2 = threading.Event()
            runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='refusal_headless_second',
                parent_dir=str(tmp_path),
                callbacks={
                    'run_complete': lambda **kw: done2.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert done2.wait(timeout=COMPLETION_TIMEOUT), (
                'a valid run after a refusal must start and complete'
            )
            assert not session.is_protocol_running
        finally:
            runner.shutdown()
            session.shutdown_executors()


# ---------------------------------------------------------------------------
# 2. Late failure (run-dir init) unwinds as failed-at-start; refusal
#    (prepare) preserves the previous run entirely.
# ---------------------------------------------------------------------------


class TestLateFailurePreservesNothingAndLeavesNoOrphan:
    def test_run_dir_init_failure_fails_at_start_no_orphan_dir(
        self, executor, tmp_path, monkeypatch
    ):
        _run_to_completion(executor, _make_single_step_protocol(), tmp_path)
        _wait_for_file_queue_drain(executor)
        output_dir = tmp_path / 'output'
        listing_after_first = sorted(p.name for p in output_dir.iterdir())

        captured = _capture_notifications(monkeypatch)

        def _boom():
            raise OSError('protocol file write failed')

        monkeypatch.setattr(executor, '_initialize_run_dir', _boom)
        completions = []
        plan = _prepare(
            executor,
            _make_single_step_protocol(),
            tmp_path,
            callbacks={'run_complete': lambda **kw: completions.append(kw)},
        )
        executor.start(plan)

        assert len(completions) == 1, (
            f'run_complete must fire exactly once for a failed-at-start run; got {completions}'
        )
        assert completions[0].get('status') == 'failed_at_start', (
            f'run_complete must carry the failed-at-start status; got {completions[0]}'
        )
        assert len(captured) == 1, f'a failed-at-start run must notify exactly once; got {captured}'
        assert not executor.run_in_progress()

        # The half-created run directory was empty and must not remain as
        # an orphan in the capture location.
        listing_after_failure = sorted(p.name for p in output_dir.iterdir())
        assert listing_after_failure == listing_after_first, (
            'a failed-at-start run must not leave an orphan directory; '
            f'before={listing_after_first} after={listing_after_failure}'
        )
        empty_dirs = [p for p in output_dir.iterdir() if p.is_dir() and not any(p.iterdir())]
        assert not empty_dirs, f'empty orphan run dirs remain: {empty_dirs}'

        # Executors are out of protocol mode, so the next run can start.
        assert _wait_for_executors_out_of_protocol_mode(executor), (
            'executors stuck in protocol mode after a failed-at-start run'
        )
        monkeypatch.undo()
        _run_to_completion(executor, _make_single_step_protocol(), tmp_path)

    def test_refused_prepare_preserves_previous_run_state(self, executor, tmp_path, monkeypatch):
        _run_to_completion(executor, _make_single_step_protocol(), tmp_path)
        _wait_for_file_queue_drain(executor)
        output_dir = tmp_path / 'output'
        first_run_dir = executor.run_dir()
        first_num_scans = executor.num_scans()
        first_trigger = executor.run_trigger_source()
        listing_after_first = sorted(p.name for p in output_dir.iterdir())

        protocol2 = _make_single_step_protocol()

        def _crash(**kwargs):
            raise OSError('labware config unreadable')

        monkeypatch.setattr(protocol2, 'validate_for_run', _crash)
        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _prepare(executor, protocol2, tmp_path)
        assert excinfo.value.reason == 'validation_crashed'

        # Observationally a no-op: every getter still answers for the
        # FIRST run and the disk is untouched.
        assert executor.run_dir() == first_run_dir, (
            'a refused prepare must not disturb run_dir(); callers saving '
            'into it would target the wrong location'
        )
        assert executor.num_scans() == first_num_scans
        assert executor.run_trigger_source() == first_trigger
        assert sorted(p.name for p in output_dir.iterdir()) == listing_after_first, (
            'a refused prepare must not touch the capture location'
        )
        assert not executor.run_in_progress()


# ---------------------------------------------------------------------------
# 3. start() cannot silently half-start.
# ---------------------------------------------------------------------------


class TestStartCannotSilentlyHalfStart:
    def test_commit_step_failure_fires_terminal_callback_and_recovers(
        self, executor, scope, tmp_path, monkeypatch
    ):
        def _boom(*args, **kwargs):
            raise RuntimeError('camera rejected target brightness')

        completions = []
        with monkeypatch.context() as mp:
            mp.setattr(scope.imaging, 'update_auto_gain_target_brightness', _boom)
            plan = _prepare(
                executor,
                _make_single_step_protocol(),
                tmp_path,
                callbacks={'run_complete': lambda **kw: completions.append(kw)},
            )
            executor.start(plan)

        assert len(completions) == 1, (
            'a failure in the last commit step must fire the terminal '
            f'callback exactly once; got {completions}'
        )
        assert completions[0].get('status') == 'failed_at_start', (
            f'the terminal callback must carry the failed status; got {completions[0]}'
        )
        assert not executor.run_in_progress(), (
            'the run-in-progress latch must not stay set after a start failure'
        )
        assert _wait_for_executors_out_of_protocol_mode(executor), (
            'executors stuck in protocol mode after a start failure'
        )

        # Not wedged: the next prepare/start succeeds end to end.
        _run_to_completion(executor, _make_single_step_protocol(), tmp_path)


# ---------------------------------------------------------------------------
# 4. Notify-once funnel: each refusal reason notifies exactly once and
#    the raised error carries the matching reason.
# ---------------------------------------------------------------------------


class TestRefusalNotifyOnceFunnel:
    def _scenarios(self, executor, scope):
        """(reason, setup_fn(mp) -> protocol) for each reachable gate."""

        def already_running(mp):
            executor._run_in_progress_event.set()
            return _make_single_step_protocol()

        def files_writing(mp):
            mp.setattr(executor.file_io_executor, 'is_protocol_queue_active', lambda: True)
            return _make_single_step_protocol()

        def empty_protocol(mp):
            return _build_real_protocol([])

        def validation_failed(mp):
            protocol = _make_single_step_protocol()
            mp.setattr(protocol, 'validate_for_run', lambda **kw: ['step 1: out of bounds'])
            return protocol

        def validation_crashed(mp):
            protocol = _make_single_step_protocol()

            def _crash(**kw):
                raise OSError('objectives.json missing')

            mp.setattr(protocol, 'validate_for_run', _crash)
            return protocol

        def hardware_state_unknown(mp):
            def _crash():
                raise RuntimeError('usb enumeration failed')

            mp.setattr(scope, 'are_all_connected', _crash)
            return _make_single_step_protocol()

        def hardware_disconnected(mp):
            mp.setattr(scope, 'are_all_connected', lambda: False)
            return _make_single_step_protocol()

        def autofocus_running(mp):
            # A live interactive autofocus owns Z and the LED lease; a run
            # prepared under it must be refused before any commitment.
            mp.setattr(executor.autofocus_thread, 'is_running', True)
            return _make_single_step_protocol()

        return [
            ('already_running', already_running),
            ('files_writing', files_writing),
            ('autofocus_running', autofocus_running),
            ('empty_protocol', empty_protocol),
            ('validation_failed', validation_failed),
            ('validation_crashed', validation_crashed),
            ('hardware_state_unknown', hardware_state_unknown),
            ('hardware_disconnected', hardware_disconnected),
        ]

    def test_each_refusal_reason_notifies_once_with_matching_reason(
        self, executor, scope, tmp_path, monkeypatch
    ):
        for reason, setup in self._scenarios(executor, scope):
            with monkeypatch.context() as mp:
                captured = _capture_notifications(mp)
                protocol = setup(mp)
                try:
                    with pytest.raises(ProtocolRunRefusedError) as excinfo:
                        _prepare(executor, protocol, tmp_path)
                finally:
                    executor._run_in_progress_event.clear()
                assert excinfo.value.reason == reason, (
                    f'expected refusal reason {reason!r}, got {excinfo.value.reason!r}'
                )
                assert len(captured) == 1, (
                    f'refusal {reason!r} must notify exactly once; got {captured}'
                )
                assert not executor.run_in_progress(), (
                    f'refusal {reason!r} must leave the runner idle'
                )
