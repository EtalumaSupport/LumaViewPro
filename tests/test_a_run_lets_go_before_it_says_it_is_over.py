"""A run tells its callers it is over only once it has let go of the scope.

Completion means ready for the next thing: when ``run_ended`` or
``files_written`` reaches a subscriber, the run is IDLE and its claim is
released, so the subscriber can act on the scope at once. Both were sent
from inside the run's cleanup, before the release -- inline on the
cleanup's thread under REST, headless and in tests -- so a subscriber
woken by either found the run still holding the scope.

The next run is the one thing that also waits for the run's files: a run
does not start over another's unwritten files, and a run that saves no
images still writes its run record's rows. So a client chaining runs starts
the next from ``files_written``; from ``run_ended`` it can be refused
``files_writing`` while those writes land.
"""

import threading
import time

import modules.protocol_image_writer as protocol_image_writer
from modules.exceptions import ProtocolRunRefusedError
from modules.protocol_state_machine import SequencedCaptureRunMode
from modules.run_events import RunEvents
from tests.test_run_refusal_contract import (  # noqa: F401 -- fixtures
    COMPLETION_TIMEOUT,
    _make_autogain_settings,
    _make_image_capture_config,
    _make_single_step_protocol,
    executor,
    executors,
    scope,
)


def _prepare(executor, tmp_path, events, **overrides):
    plan_args = {
        'protocol': _make_single_step_protocol(),
        'run_mode': SequencedCaptureRunMode.SINGLE_SCAN,
        'sequence_name': 'lets_go',
        'image_capture_config': _make_image_capture_config(),
        'autogain_settings': _make_autogain_settings(),
        'parent_dir': tmp_path / 'output',
        'max_scans': 1,
        'events': events,
    }
    return executor.prepare(**{**plan_args, **overrides})


def _refuse_the_protocol_copy(monkeypatch, plan):
    """Make ``plan``'s run fail at start: the file system refuses its protocol copy."""

    def _refused(**kwargs):
        raise OSError('No space left on device')

    monkeypatch.setattr(plan.protocol, 'to_file', _refused)
    return plan


def _what_the_subscriber_saw(executor, told: threading.Event, seen: list):
    def record(*_args):
        seen.append((executor.run_in_progress(), executor.run_started_by()))
        told.set()

    return record


def test_run_ended_reaches_its_subscriber_after_the_run_let_go(executor, tmp_path):
    told, seen = threading.Event(), []
    plan = _prepare(
        executor, tmp_path, RunEvents(run_ended=_what_the_subscriber_saw(executor, told, seen))
    )
    executor.start(plan)
    assert told.wait(COMPLETION_TIMEOUT), 'run_ended never came'
    assert seen == [(False, None)], 'the run still held the scope when it said it was over'


def test_files_written_reaches_its_subscriber_after_the_run_let_go(executor, tmp_path):
    # Saving nothing, the run's writes are all in when it closes them: the
    # completion that once ran there, before the release.
    told, seen = threading.Event(), []
    plan = _prepare(
        executor,
        tmp_path,
        RunEvents(files_written=_what_the_subscriber_saw(executor, told, seen)),
        enable_image_saving=False,
    )
    executor.start(plan)
    assert told.wait(COMPLETION_TIMEOUT), 'files_written never came'
    assert seen == [(False, None)], 'the run still held the scope when its files were reported'


def test_a_failed_start_says_so_after_it_let_go(executor, tmp_path, monkeypatch):
    # The copy of the protocol cannot be written, so the run fails at start
    # and unwinds on start()'s thread.
    told, seen = threading.Event(), []
    plan = _prepare(
        executor,
        tmp_path,
        RunEvents(run_ended=_what_the_subscriber_saw(executor, told, seen)),
    )
    executor.start(_refuse_the_protocol_copy(monkeypatch, plan))
    assert told.wait(COMPLETION_TIMEOUT), 'run_ended never came'
    assert seen == [(False, None)], 'the failed start still held the scope when it said so'


def test_a_files_written_subscriber_can_start_the_next_run(executor, tmp_path):
    # Ready for the next run: the run's last write has landed, so the next run
    # starts from the subscriber, on the thread that told it, with no refusal.
    started, refused = threading.Event(), []
    second_done = threading.Event()

    def start_the_next(*_args):
        try:
            second = _prepare(
                executor,
                tmp_path,
                RunEvents(run_ended=lambda *_: second_done.set()),
                enable_image_saving=False,
            )
            executor.start(second)
        except Exception as ex:
            refused.append(ex)
        started.set()

    first = _prepare(
        executor, tmp_path, RunEvents(files_written=start_the_next), enable_image_saving=False
    )
    executor.start(first)
    assert started.wait(COMPLETION_TIMEOUT), 'files_written never came'
    assert refused == [], f'the next run was refused from files_written: {refused}'
    assert second_done.wait(COMPLETION_TIMEOUT), 'the next run never finished'


def test_a_run_started_from_run_ended_while_its_writes_land_is_refused(
    executor, tmp_path, monkeypatch
):
    # A run that saves no images still writes its run record's rows. While
    # one is still landing when run_ended comes, the next run is refused for
    # it, by name, as a refusal the caller may retry.
    original = protocol_image_writer.ProtocolImageWriter.write_capture

    # a stand-in by design: a slow save drive, which the simulator cannot
    # produce (its file lane writes as fast as the host's disk).
    def slow(self, *args, **kwargs):
        time.sleep(0.5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(protocol_image_writer.ProtocolImageWriter, 'write_capture', slow)
    told, refused = threading.Event(), []

    def start_the_next(*_args):
        try:
            executor.start(_prepare(executor, tmp_path, RunEvents(), enable_image_saving=False))
        except ProtocolRunRefusedError as ex:
            refused.append(ex.reason)
        told.set()

    executor.start(
        _prepare(executor, tmp_path, RunEvents(run_ended=start_the_next), enable_image_saving=False)
    )
    assert told.wait(COMPLETION_TIMEOUT), 'run_ended never came'
    assert refused == ['files_writing']


def test_a_failed_starts_handle_keeps_no_folder_when_its_subscriber_starts_the_next(
    executor, tmp_path, monkeypatch
):
    # A failed start ends on start()'s own thread, so its run_ended runs
    # there before start() returns; a subscriber that starts a run which
    # saves sets the runner's folder to that run's. The failed run's handle
    # answers for itself: it never had a folder.
    second = []
    second_done = threading.Event()

    def start_the_next(*_args):
        plan = _prepare(executor, tmp_path, RunEvents(run_ended=lambda *_: second_done.set()))
        second.append(executor.start(plan))

    failed = executor.start(
        _refuse_the_protocol_copy(
            monkeypatch, _prepare(executor, tmp_path, RunEvents(run_ended=start_the_next))
        )
    )
    assert second, 'the subscriber never started the next run'
    assert second_done.wait(COMPLETION_TIMEOUT), 'the next run never finished'
    assert failed.progress().run_dir is None, (
        f"the failed start's run took {failed.progress().run_dir}"
    )
    assert second[0].progress().run_dir is not None and second[0].progress().run_dir.is_dir()


def test_a_raising_run_state_listener_does_not_cost_the_events(executor, tmp_path):
    # The run's end tells each listener on its own: the Session's levels
    # failing must not leave run_ended unsent.
    def _broken():
        raise RuntimeError('the levels could not be read')

    executor._on_run_idle = _broken
    told = threading.Event()
    executor.start(
        _prepare(
            executor,
            tmp_path,
            RunEvents(run_ended=lambda *_: told.set()),
            enable_image_saving=False,
        )
    )
    assert told.wait(COMPLETION_TIMEOUT), 'run_ended was skipped after the listener raised'
