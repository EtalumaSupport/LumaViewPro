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
5. A run prepared without the autofocus snapshot it would restore from
   is refused at the signature, so no partial run reaches disk.
6. A run whose data root cannot be resolved fails at start -- one
   notification, the terminal callback, nothing left on disk -- rather
   than proceeding and failing later on a post-run background thread.

The caller-ordering half of the contract (UI starters commit
running-state only between prepare and start, inside the shared
refusal boundary) is locked by tests/test_protocol_start_refusal_ui_gate.py.
"""

import ast
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
from tests.protocol_drives import autofocus_snapshot, wait_until_not_running
from modules.image_mode import ImageCaptureConfig
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
    # The session registers the data root at bring-up; a runner over a
    # bare scope needs it too, or the run refuses at start.
    s.protocols.register_source_path('.')
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
    return ImageCaptureConfig.from_image_mode('8bit')


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
        # The snapshot carries its own restorer, so cleanup never reaches
        # the global settings module that other test files replace with
        # import-order-dependent stand-ins.
        autofocus_snapshot=autofocus_snapshot(),
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
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks={
                    'run_complete': lambda **kw: done.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert done.wait(timeout=COMPLETION_TIMEOUT), 'valid first run did not complete'
            assert runner.wait_for_completion(timeout=COMPLETION_TIMEOUT), (
                'wait_for_completion must report the completed first run'
            )
            assert wait_until_not_running(session)

            # A refused run raises out of run_single_scan; nothing waits.
            with pytest.raises(ProtocolRunRefusedError):
                runner.run_single_scan(
                    protocol=_build_real_protocol([]),
                    sequence_name='refusal_headless_refused',
                    parent_dir=str(tmp_path),
                    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                )
            assert not session.is_protocol_running, (
                'a refused run must not leave the session reporting a live run'
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
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks={
                    'run_complete': lambda **kw: done2.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert done2.wait(timeout=COMPLETION_TIMEOUT), (
                'a valid run after a refusal must start and complete'
            )
            assert wait_until_not_running(session)
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
            # The runner's commit-step write binds the impl (run-internal
            # machinery never goes through the external dispatcher), so
            # the fault is injected at the seam the runner actually calls.
            mp.setattr(scope.imaging, '_update_auto_gain_target_brightness_impl', _boom)
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

_FUNNEL_LOOP = 'test_each_refusal_reason_notifies_once_with_matching_reason'

# Every refusal reason the runner can raise, mapped to the test that pins
# its notify-once contract. test_every_runner_refusal_reason_is_covered
# diffs this map against the runner's source, so a reason added to the
# runner reds the suite until it gets a scenario or a dedicated test AND
# a row here; a reason retired from the runner reds until its row is
# removed. The map cannot silently drift from production.
RUNNER_REFUSAL_COVERAGE = {
    'already_running': _FUNNEL_LOOP,
    'files_writing': _FUNNEL_LOOP,
    'files_writing_stalled': _FUNNEL_LOOP,
    'autofocus_running': _FUNNEL_LOOP,
    'empty_protocol': _FUNNEL_LOOP,
    'validation_failed': _FUNNEL_LOOP,
    'validation_crashed': _FUNNEL_LOOP,
    'hardware_state_unknown': _FUNNEL_LOOP,
    'hardware_disconnected': _FUNNEL_LOOP,
    # Raised at start(), not prepare(), so it cannot ride the scenario
    # loop (which drives _prepare); it gets the start-tier twin below.
    'exclusive_activity_running': ('test_start_refused_while_recording_holds_activity_claim'),
    # Raised at composite config assembly, before the engine is reached at
    # all -- a composite the merge could not produce is refused where the
    # channel count is known.
    'composite_needs_two_channels': ('tests/test_composite_run_config.py::TestTwoChannelFloor'),
}

# Every module that raises a run refusal. The census below reads all of
# them, so a reason introduced outside the runner is covered too -- the
# vocabulary is the contract, not the file it happens to live in.
REFUSING_MODULES = (
    'modules/sequenced_capture_runner.py',
    'modules/config_helpers.py',
)


class TestRefusalNotifyOnceFunnel:
    def _scenarios(self, executor, scope):
        """(reason, setup_fn(mp) -> protocol) for each reachable gate."""

        def already_running(mp):
            executor._run_in_progress_event.set()
            return _make_single_step_protocol()

        def files_writing(mp):
            mp.setattr(executor.file_io_executor, 'is_protocol_queue_active', lambda: True)
            return _make_single_step_protocol()

        def files_writing_stalled(mp):
            # The stalled branch nests under the queue-active gate: both
            # probes must fire for the stalled refusal to be reachable.
            mp.setattr(executor.file_io_executor, 'is_protocol_queue_active', lambda: True)
            mp.setattr(
                executor.file_io_executor,
                'protocol_drain_stalled',
                lambda threshold_s: True,
            )
            mp.setattr(
                executor.file_io_executor,
                'describe_running_task',
                lambda: "write_capture 'A1_BF' 45s in flight",
            )
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
            ('files_writing_stalled', files_writing_stalled),
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
        scenarios = self._scenarios(executor, scope)
        assert {reason for reason, _ in scenarios} == {
            reason
            for reason, covered_by in RUNNER_REFUSAL_COVERAGE.items()
            if covered_by == _FUNNEL_LOOP
        }, 'the scenario list and RUNNER_REFUSAL_COVERAGE drifted apart'
        for reason, setup in scenarios:
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

    def test_start_refused_while_recording_holds_activity_claim(
        self, executor, tmp_path, monkeypatch
    ):
        """start()-tier twin of the loop above.

        The session's activity claim is checked at the commitment point,
        after prepare() has already succeeded, so a live video recording
        refuses the run through the same notify-once funnel. The refusal
        must not disturb the recording's claim, and the runner must stay
        fully usable once the recording releases it.
        """
        with monkeypatch.context() as mp:
            captured = _capture_notifications(mp)
            plan = _prepare(executor, _make_single_step_protocol(), tmp_path)
            assert executor._activity_claim.try_claim('recording')
            try:
                with pytest.raises(ProtocolRunRefusedError) as excinfo:
                    executor.start(plan)
                assert executor._activity_claim.owner == 'recording', (
                    "a refused start must not steal or release the recording's claim"
                )
            finally:
                executor._activity_claim.release('recording')
        assert excinfo.value.reason == 'exclusive_activity_running'
        assert 'recording' in excinfo.value.message.lower(), (
            'the recording-holder branch must name the recording, not the '
            f'generic other-activity copy; got {excinfo.value.message!r}'
        )
        assert len(captured) == 1, f'the refusal must notify exactly once; got {captured}'
        assert not executor.run_in_progress(), 'a start()-tier refusal must leave the runner idle'
        # Not wedged: with the claim released, the next run completes.
        _run_to_completion(executor, _make_single_step_protocol(), tmp_path)


# ---------------------------------------------------------------------------
# 5. A run cannot be prepared without the autofocus snapshot it restores from.
# ---------------------------------------------------------------------------


class TestAutofocusSnapshotIsRequiredToPrepare:
    """The run's autofocus snapshot is a required prepare() argument.

    A run that starts without one has no record of the per-layer
    autofocus flags it is about to overwrite, so cleanup cannot put them
    back. Refusing at the signature keeps that run from ever committing:
    no directory on the capture disk, no terminal callback, no
    run-in-progress state to unwind.
    """

    def test_prepare_without_the_snapshot_raises(self):
        from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

        runner = bare_capture_runner()
        # None tells the drive builder not to construct one; the key then
        # comes out entirely, which is the call shape under test.
        kwargs = scr_run_kwargs(autofocus_snapshot=None)
        kwargs.pop('autofocus_snapshot')

        with pytest.raises(TypeError) as excinfo:
            runner.prepare(**kwargs)
        assert 'autofocus_snapshot' in str(excinfo.value), (
            f'the binding error must name the missing argument; got {excinfo.value}'
        )

    def test_omitting_the_snapshot_performs_no_run_at_all(self, tmp_path):
        from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

        output_dir = tmp_path / 'output'
        run_complete = MagicMock()
        runner = bare_capture_runner()
        kwargs = scr_run_kwargs(
            parent_dir=output_dir,
            disable_saving_artifacts=False,
            callbacks={'run_complete': run_complete, 'go_to_step': lambda **kw: None},
            autofocus_snapshot=None,
        )
        kwargs.pop('autofocus_snapshot')

        # The signature declines the run; what follows asserts that the
        # decline left nothing behind -- no directory, no callback, no
        # run-in-progress state.
        with pytest.raises(TypeError):
            runner.prepare(**kwargs)

        assert runner.run_dir() is None, (
            f'a run that never had a snapshot must claim no run directory; got {runner.run_dir()}'
        )
        assert not output_dir.exists() or not list(output_dir.iterdir()), (
            'a run that never had a snapshot must leave the capture location '
            f'untouched; found {sorted(p.name for p in output_dir.iterdir())}'
        )
        assert run_complete.call_count == 0, (
            f'no run started, so no terminal callback may fire; got {run_complete.call_args_list}'
        )
        assert not runner.run_in_progress()


# ---------------------------------------------------------------------------
# 6. A run that cannot resolve its data root fails at start.
# ---------------------------------------------------------------------------


class TestARunThatCannotResolveItsDataRootFailsAtStart:
    """The run's data root is resolved at start, on the caller's thread.

    The post-run merge and stack build both read data/tiling.json, and
    both run on a daemon thread long after start() returned. A scope
    whose source path was never registered cannot answer for that file
    at all -- so resolving it late means the run captures a full plate
    and then fails where nobody is waiting. Resolving it at start makes
    the failure the caller's: the run never commits to disk, the
    terminal callback carries failed_at_start, and the user is told
    once.
    """

    def test_an_unresolvable_data_root_fails_the_run_at_start(self, tmp_path, monkeypatch):
        import modules.notification_center as notification_center
        from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

        notified = []
        monkeypatch.setattr(
            notification_center.notifications,
            'error',
            lambda *args, **kwargs: notified.append(args),
        )

        output_dir = tmp_path / 'out'
        runner = bare_capture_runner()
        runner._scope.protocols.tiling_configs_path.side_effect = RuntimeError(
            'scope.protocols.load_protocol/create_protocol require '
            'scope.protocols.register_source_path() to have been called.'
        )
        # A post-run step spawned here would carry the failure onto a
        # daemon thread, which is the shape this contract replaces.
        spawned = []
        monkeypatch.setattr(
            runner, '_spawn_post_run_step', lambda **kwargs: spawned.append(kwargs['name'])
        )

        completions = []
        plan = runner.prepare(
            **scr_run_kwargs(
                parent_dir=output_dir,
                disable_saving_artifacts=False,
                callbacks={
                    'run_complete': lambda **kw: completions.append(kw),
                    'go_to_step': lambda **kw: None,
                },
            )
        )
        # start() resolves the outcome rather than raising, so the caller
        # is released with an answer on this path like every other.
        runner.start(plan)

        runner._scope.protocols.tiling_configs_path.assert_called_once_with()
        assert len(completions) == 1, (
            'the terminal callback must fire exactly once for a run that '
            f'could not resolve its data root; got {completions}'
        )
        assert completions[0].get('status') == 'failed_at_start', (
            f'run_complete must carry the failed-at-start status; got {completions[0]}'
        )
        failed_to_start = [args for args in notified if 'Run failed to start' in args]
        assert len(failed_to_start) == 1, (
            f'the user must be told once that the run failed to start; got {notified}'
        )
        assert notified == failed_to_start, (
            f'a failed-at-start run raises exactly that one error; got {notified}'
        )
        # The path is resolved BEFORE the run directory is created, so a
        # run that cannot resolve it leaves no half-made directory behind.
        assert not output_dir.exists() or not list(output_dir.iterdir()), (
            'a run that failed before it could resolve its data root must '
            f'leave the capture location untouched; found {list(output_dir.iterdir())}'
        )
        assert not runner.run_in_progress(), (
            'a run that failed at start must not stay marked in progress'
        )
        assert not spawned, (
            f'a run that never captured anything has nothing to post-process; got {spawned}'
        )


def test_every_runner_refusal_reason_is_covered():
    """Census guard: the coverage map above matches production source.

    Collects every reason literal fed to a _refuse funnel (or a direct
    ProtocolRunRefusedError construction) across REFUSING_MODULES and
    diffs the set against RUNNER_REFUSAL_COVERAGE, in both directions.
    This is what makes the notify-once suite's enumeration unmissable
    instead of remembered: exclusive_activity_running shipped with zero
    coverage because nothing coupled the scenario list to the refusal
    vocabulary.
    """
    from tests import ast_seams

    raised = set()

    class ReasonCollector(ast.NodeVisitor):
        def visit_Call(self, node):
            callee = getattr(node.func, 'attr', None) or getattr(node.func, 'id', None)
            # Any _refuse* funnel, not one exact spelling: a second refusal
            # site naming its funnel for what it refuses would otherwise be
            # invisible to this census, which is the drift it exists to stop.
            is_funnel = callee is not None and callee.startswith('_refuse')
            if is_funnel or callee == 'ProtocolRunRefusedError':
                for kw in node.keywords:
                    if kw.arg == 'reason' and isinstance(kw.value, ast.Constant):
                        raised.add(kw.value.value)
            self.generic_visit(node)

    for rel_path in REFUSING_MODULES:
        ReasonCollector().visit(ast_seams.parse_module(rel_path))
    assert raised, 'found no refusal reasons in production source; the scan is broken'

    uncovered = raised - set(RUNNER_REFUSAL_COVERAGE)
    assert not uncovered, (
        f'refusal reasons raised by the runner with no coverage row: {sorted(uncovered)}. '
        'Pin the notify-once contract for each (scenario or dedicated test), '
        'then add its row to RUNNER_REFUSAL_COVERAGE.'
    )
    stale = set(RUNNER_REFUSAL_COVERAGE) - raised
    assert not stale, (
        f'coverage rows for reasons the runner no longer raises: {sorted(stale)}. '
        'Retire each row together with its test.'
    )
