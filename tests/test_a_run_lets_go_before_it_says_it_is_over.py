"""A run tells its callers it is over only once it has let go of the scope.

Completion means ready for the next thing: when ``run_complete`` or
``files_complete`` reaches a subscriber, the run is IDLE and its claim is
released, so the subscriber can act on the scope at once. Both were sent
from inside the run's cleanup, before the release -- inline on the
cleanup's thread under REST, headless and in tests -- so a subscriber
woken by either found the run still holding the scope.
"""

import threading

from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.test_run_refusal_contract import (  # noqa: F401 -- fixtures
    COMPLETION_TIMEOUT,
    _make_autogain_settings,
    _make_image_capture_config,
    _make_single_step_protocol,
    autofocus_snapshot,
    executor,
    executors,
    scope,
)


def _prepare(executor, tmp_path, callbacks, **overrides):
    plan_args = {
        'protocol': _make_single_step_protocol(),
        'run_trigger_source': 'test',
        'run_mode': SequencedCaptureRunMode.SINGLE_SCAN,
        'sequence_name': 'lets_go',
        'image_capture_config': _make_image_capture_config(),
        'autogain_settings': _make_autogain_settings(),
        'parent_dir': tmp_path / 'output',
        'max_scans': 1,
        'callbacks': {
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
            **callbacks,
        },
        'autofocus_snapshot': autofocus_snapshot(),
    }
    return executor.prepare(**{**plan_args, **overrides})


def _what_the_subscriber_saw(executor, told: threading.Event, seen: list):
    def record(**kwargs):
        seen.append((executor.run_in_progress(), executor.run_trigger_source()))
        told.set()

    return record


def test_run_complete_reaches_its_subscriber_after_the_run_let_go(executor, tmp_path):
    told, seen = threading.Event(), []
    plan = _prepare(
        executor, tmp_path, {'run_complete': _what_the_subscriber_saw(executor, told, seen)}
    )
    executor.start(plan)
    assert told.wait(COMPLETION_TIMEOUT), 'run_complete never came'
    assert seen == [(False, None)], 'the run still held the scope when it said it was over'


def test_files_complete_reaches_its_subscriber_after_the_run_let_go(executor, tmp_path):
    # Saving nothing, the run's writes are all in when it closes them: the
    # completion that once ran there, before the release.
    told, seen = threading.Event(), []
    plan = _prepare(
        executor,
        tmp_path,
        {'files_complete': _what_the_subscriber_saw(executor, told, seen)},
        enable_image_saving=False,
    )
    executor.start(plan)
    assert told.wait(COMPLETION_TIMEOUT), 'files_complete never came'
    assert seen == [(False, None)], 'the run still held the scope when its files were reported'


def test_a_failed_start_says_so_after_it_let_go(executor, tmp_path):
    # The copy of the protocol names a folder that does not exist, so the
    # run fails at start and unwinds on start()'s thread.
    told, seen = threading.Event(), []
    plan = _prepare(
        executor,
        tmp_path,
        {'run_complete': _what_the_subscriber_saw(executor, told, seen)},
        sequence_name='no_such_folder/lets_go',
    )
    executor.start(plan)
    assert told.wait(COMPLETION_TIMEOUT), 'run_complete never came'
    assert seen == [(False, None)], 'the failed start still held the scope when it said so'


def test_a_run_complete_subscriber_can_start_the_next_run(executor, tmp_path):
    # Ready for the next thing: a run that saved nothing has no files to
    # drain, so the next run starts from the subscriber, on the thread that
    # told it, with no refusal.
    started, refused = threading.Event(), []
    second_done = threading.Event()

    def start_the_next(**kwargs):
        try:
            second = _prepare(
                executor,
                tmp_path,
                {'run_complete': lambda **kw: second_done.set()},
                enable_image_saving=False,
            )
            executor.start(second)
        except Exception as ex:
            refused.append(ex)
        started.set()

    first = _prepare(
        executor, tmp_path, {'run_complete': start_the_next}, enable_image_saving=False
    )
    executor.start(first)
    assert started.wait(COMPLETION_TIMEOUT), 'run_complete never came'
    assert refused == [], f'the next run was refused from run_complete: {refused}'
    assert second_done.wait(COMPLETION_TIMEOUT), 'the next run never finished'


def test_a_failed_starts_handle_keeps_no_folder_when_its_subscriber_starts_the_next(
    executor, tmp_path
):
    # A failed start ends on start()'s own thread, so its run_complete runs
    # there before start() returns; a subscriber that starts a run which
    # saves sets the runner's folder to that run's. The failed run's handle
    # answers for itself: it never had a folder.
    second = []
    second_done = threading.Event()

    def start_the_next(**kwargs):
        plan = _prepare(executor, tmp_path, {'run_complete': lambda **kw: second_done.set()})
        second.append(executor.start(plan))

    failed = executor.start(
        _prepare(
            executor,
            tmp_path,
            {'run_complete': start_the_next},
            sequence_name='no_such_folder/lets_go',
        )
    )
    assert second, 'the subscriber never started the next run'
    assert second_done.wait(COMPLETION_TIMEOUT), 'the next run never finished'
    assert failed.run_dir is None, f"the failed start's handle took {failed.run_dir}"
    assert second[0].run_dir is not None and second[0].run_dir.is_dir()


def test_a_raising_run_state_listener_does_not_cost_the_callbacks(executor, tmp_path):
    # The run's end tells each listener on its own: the Session's levels
    # failing must not leave run_complete unsent.
    def _broken():
        raise RuntimeError('the levels could not be read')

    executor._on_run_idle = _broken
    told = threading.Event()
    executor.start(
        _prepare(
            executor, tmp_path, {'run_complete': lambda **kw: told.set()}, enable_image_saving=False
        )
    )
    assert told.wait(COMPLETION_TIMEOUT), 'run_complete was skipped after the listener raised'
