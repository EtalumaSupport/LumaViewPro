# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A UI callback that raises off the GUI path must still be reported.

`schedule_ui` has two branches, and they are NOT symmetric.

The GUI branch hands the callback to `Clock.schedule_once`. A raising
callback there reaches the app's crash guard, which returns
`ExceptionManager.RAISE` for any frame it cannot attribute to a
plugin -- which is everything this function dispatches. So the GUI
RE-RAISES: it is the harsher host, and it is loud.

The no-GUI branch -- the ONLY branch a headless, REST or test caller
takes -- used to `pass` on the exception. A throwing protocol or
recording callback produced no record at all and the run looked like
it had succeeded.

These tests pin that the headless failure is VISIBLE, not that the two
branches match. Logging rather than raising is a deliberate choice to
be more forgiving than the GUI: a REST caller must not be killed by
one bad UI callback. Silence is the only option with no argument for
it.

Note on the assertion shape: the suite replaces the whole `lvp_logger`
module with a MagicMock (`conftest.py`), so log records are asserted
through the mock. Reading stdout/caplog for `lvp_logger` output proves
nothing here.
"""

from unittest.mock import MagicMock

import pytest
from lvp_logger import logger

from modules import kivy_utils


@pytest.fixture(autouse=True)
def _headless_dispatcher():
    """Force the no-GUI branch and restore whatever was there before."""
    previous = kivy_utils._ui_dispatcher
    kivy_utils.set_ui_dispatcher(None)
    logger.reset_mock()
    yield
    kivy_utils.set_ui_dispatcher(previous)


def _boom(_dt):
    raise RuntimeError('callback exploded')


def test_headless_callback_failure_is_logged_with_traceback():
    kivy_utils.schedule_ui(_boom)

    assert logger.exception.called, (
        'a callback that raised on the headless branch produced no log '
        'record; the GUI path logs it, so headless silence is a '
        'failure-parity gap'
    )


def test_headless_callback_failure_does_not_reach_the_caller():
    # Loud-log, not raise: the GUI branch does not crash its caller, and
    # a headless caller that suddenly does would be a new failure mode.
    kivy_utils.schedule_ui(_boom)


def test_a_healthy_headless_callback_still_runs_and_logs_nothing():
    called = MagicMock()

    kivy_utils.schedule_ui(called)

    called.assert_called_once_with(0)
    assert not logger.exception.called, (
        'a callback that returned normally must not produce an exception '
        'record; a logger that fires on the happy path is noise, not signal'
    )
