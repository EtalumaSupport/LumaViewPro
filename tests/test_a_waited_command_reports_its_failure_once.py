# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A waited hardware command that fails is reported once: to its caller.

The camera, LED and motion dispatchers run a command on a lane and block on
its result, so the exception reaches the caller. The lane also posted its
generic 'Background operation failed' notice for the same exception, so a
failed command reached the user twice -- once from whatever the caller does
with the raise, once from the executor, which does not know the caller is
already holding it. A fire-and-forget task has no caller waiting, and the
lane's notice stays its only report.
"""

import logging

import pytest

from modules.sequential_io_executor import IOTask

NOTIFICATION_LOGGER = 'LVP.notifications'


class _BoomError(RuntimeError):
    pass


def _boom():
    raise _BoomError('the command failed')


@pytest.fixture
def scope():
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        yield session.scope
    finally:
        session.shutdown()


def _drain(lane):
    """Everything queued before this has finished, epilogue included."""
    lane.put(IOTask(action=lambda: None), return_future=True).result(timeout=5.0)


def _notices(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == NOTIFICATION_LOGGER and '_boom' in r.getMessage()
    ]


def _dispatchers(scope):
    return {
        'camera': (
            lambda: scope.imaging._dispatch_camera(_boom, 'boom', timeout_s=5.0),
            scope._camera_executor,
        ),
        'led': (lambda: scope.illumination._dispatch_led(_boom, 'boom'), scope._io_executor),
        'motion': (
            lambda: scope.motion._dispatch_motion(_boom, 'boom', timeout_s=5.0),
            scope._io_executor,
        ),
    }


@pytest.mark.parametrize('kind', ['camera', 'led', 'motion'])
def test_the_caller_gets_the_failure_and_the_lane_posts_nothing(scope, caplog, kind):
    call, lane = _dispatchers(scope)[kind]

    with caplog.at_level(logging.DEBUG, logger=NOTIFICATION_LOGGER):
        with pytest.raises(_BoomError):
            call()
        _drain(lane)

    assert _notices(caplog) == [], f'the {kind} lane also posted the failure its caller holds'


def test_a_task_nobody_waits_on_is_still_reported_by_its_lane(scope, caplog):
    lane = scope._io_executor

    with caplog.at_level(logging.DEBUG, logger=NOTIFICATION_LOGGER):
        lane.put(IOTask(action=_boom))
        _drain(lane)

    # Shown or deduplicated, the lane is what reported it.
    assert _notices(caplog), 'a failure nobody waits on went unreported'
