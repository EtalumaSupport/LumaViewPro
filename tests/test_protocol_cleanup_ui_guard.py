# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A failing run-cleanup callback must not take the application down.

Cleanup is deliberately fault-tolerant: every step runs regardless of any
other failing, and the failures are collected into one summary. Callbacks
handed to the UI scheduler escaped that contract, because they run on a
later Clock tick -- the `try` that scheduled them has already returned,
and the app's crash guard re-raises anything it cannot attribute to a
plugin. So a cosmetic panel-restore step could, and did, terminate
LumaViewPro at the end of every protocol run.

The re-raise is the right DEFAULT and is left alone everywhere else; a
core bug should be loud. It is wrong for these six callbacks for the same
reason the code around them is fault-tolerant: the run's images are
already on disk by the time cleanup starts, and an unattended run must
not lose its application to a restore step.

The assertion shape: conftest replaces the whole `lvp_logger` module with
a MagicMock, so log records are asserted through the mock rather than
caplog.
"""

import threading

import pytest
from lvp_logger import logger

from modules import kivy_utils, protocol_cleanup


def _sent():
    """A summary flag that has already fired -- the deferred GUI case."""
    flag = threading.Event()
    flag.set()
    return flag


@pytest.fixture
def immediate_gui_dispatcher():
    """Stand in for Clock.schedule_once, which runs callbacks unguarded.

    The GUI branch is the one that matters here: the headless branch of
    schedule_ui has caught and logged for a while, so a headless-only
    test would pass against the very bug this file exists for.
    """
    previous = kivy_utils._ui_dispatcher
    kivy_utils.set_ui_dispatcher(lambda func, timeout: func(timeout))
    logger.reset_mock()
    yield
    kivy_utils.set_ui_dispatcher(previous)


def _boom(_dt):
    raise RuntimeError('cleanup callback exploded')


def test_a_raising_cleanup_callback_does_not_reach_the_event_loop(immediate_gui_dispatcher):
    """This is the crash: the exception used to escape to Kivy and exit."""
    protocol_cleanup._schedule_cleanup_ui(_boom, 'Sync layer panel', [], _sent())


def test_a_raising_cleanup_callback_is_logged_with_its_step_name(immediate_gui_dispatcher):
    protocol_cleanup._schedule_cleanup_ui(_boom, 'Sync layer panel', [], _sent())

    assert logger.exception.called, (
        'a cleanup callback that raised produced no log record; silent '
        'swallowing is the failure mode this guard must not introduce'
    )
    logged = ' '.join(str(call) for call in logger.exception.call_args_list)
    assert 'Sync layer panel' in logged, (
        'the log must name WHICH cleanup step failed -- a bare traceback '
        'from a deferred callback gives the reader no run context'
    )


def test_a_healthy_cleanup_callback_still_runs(immediate_gui_dispatcher):
    seen = []
    protocol_cleanup._schedule_cleanup_ui(lambda dt: seen.append(dt), 'Harmless step', [], _sent())

    assert seen == [0], 'the guard must not change what a working callback does'
    assert not logger.exception.called, (
        'a callback that returned normally must not produce an exception record'
    )


def test_the_unguarded_scheduler_still_propagates(immediate_gui_dispatcher):
    """Why the guard has to exist, and that it stays narrowly scoped.

    Raw schedule_ui on the GUI branch propagates -- that is the documented
    app-wide policy and this change deliberately does not touch it. If
    this test ever fails, the global policy moved and the guard above may
    no longer be needed.
    """
    with pytest.raises(RuntimeError):
        kivy_utils.schedule_ui(_boom, 0)


def test_a_failure_before_the_summary_is_collected_not_self_reported(immediate_gui_dispatcher):
    """Inline execution -- headless, REST, or a test dispatcher.

    The summary has not gone out yet, so the failure belongs in it, worded
    exactly as the surrounding except blocks word their own. Reporting it
    separately here would drop the summary's count and split one run's
    story across two messages.
    """
    errors: list[str] = []
    not_sent = threading.Event()

    protocol_cleanup._schedule_cleanup_ui(_boom, 'Restore layer shader', errors, not_sent)

    assert len(errors) == 1, f'the failure must be collected for the summary; got {errors}'
    assert errors[0].startswith('Restore layer shader: RuntimeError'), (
        f'collected wording must match the surrounding except blocks; got {errors[0]!r}'
    )
