# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The startup motion holds the scope, and not the caller that started it.

The GUI started the startup motion on its main thread and waited for it, so
the window could not draw for the whole home -- 39 s on an LS720. The
Session now starts it without waiting: the scope is the startup's from its
first move to its last, so a turret pick made meanwhile is refused rather
than undone by the startup's move to slot 1, and the claim is released
before the caller hears the motion has ended. The blocking member stays for
integrators, and refuses a call from an executor's worker, whose own lane
its sequence would wait on.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.settings_fixtures import complete_settings

HOME_S = 60.0
# Only a failure waits this long.
ASKED_S = 5.0


@pytest.fixture
def session():
    s = ScopeSession.create(
        complete_settings(microscope='LS850T'), simulate=True, warn_pre_release=False
    )
    yield s
    s.shutdown()


class _HeldHome:
    """The board's home, held inside the driver until released."""

    def __init__(self, scope, monkeypatch):
        self.entered = threading.Event()
        self.release = threading.Event()
        real = scope._motion_driver.home

        def held(*args, **kwargs):
            self.entered.set()
            assert self.release.wait(HOME_S), 'the held home was never released'
            return real(*args, **kwargs)

        monkeypatch.setattr(scope._motion_driver, 'home', held)


@pytest.mark.slow
def test_the_startup_motion_holds_the_scope_until_it_ends(session, monkeypatch):
    scope = session.scope
    held = _HeldHome(scope, monkeypatch)

    started = session.begin_application_session()
    locked_when_settled = []
    started.add_done_callback(lambda _done: locked_when_settled.append(session.controls_locked))
    try:
        assert held.entered.wait(HOME_S)
        assert not started.done()
        assert session.controls_locked
        with pytest.raises(HardwareCommandRefusedError) as pick:
            scope.motion.move_turret(3)
        assert 'home' in str(pick.value).lower(), str(pick.value)
    finally:
        held.release.set()
        started.exception(timeout=HOME_S)
    started.result(timeout=HOME_S)

    assert locked_when_settled == [False]
    assert not session.controls_locked
    assert scope.motion.has_homed()
    assert scope.motion.get_turret_slot() == 1


def test_a_blocking_startup_asked_from_a_lane_task_is_refused(session):
    outcome: concurrent.futures.Future = concurrent.futures.Future()

    def asked_on_the_lane():
        try:
            session.start_application_session()
        except BaseException as e:
            outcome.set_result(e)
        else:
            outcome.set_result(None)

    scope = session.scope
    scope.io_lane().put(IOTask(action=asked_on_the_lane))
    refusal = outcome.result(timeout=ASKED_S)
    assert isinstance(refusal, RuntimeError), refusal
