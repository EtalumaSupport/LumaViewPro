# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope is torn down once, whoever asks and whenever.

The session's close tears the scope down on its own thread. When the process
ended mid-close (a crash in the window, a signal, a platform quit), the exit
hook Lumascope registers ran a second disconnect() beside the close's, with
nothing between them: both threads tore the same camera driver down, and the
second met a handle the first had already released (the LS850T, 2026-10-15:
NoneType on DetachDevice). One disconnect runs at a time; the exit hook waits
for one in flight, up to its bound, and finds nothing left.
"""

import threading
import time

import pytest

from modules.lumascope_api import Lumascope
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_DRIVERS = ('_led_driver', '_motion_driver', '_camera_driver')


@pytest.fixture
def session(tmp_path):
    built = ScopeSession.create(
        complete_settings(microscope='LS850T', live_folder=str(tmp_path)), simulate=True
    )
    home_sim_scope(built.scope)
    yield built
    built.shutdown()


def _count_disconnects(scope, *, hold_camera):
    """Record (driver, thread) for each driver disconnect; a camera disconnect
    waits on ``hold_camera`` (an Event) before it runs."""
    calls = []
    in_camera = threading.Event()
    for slot in _DRIVERS:
        driver = getattr(scope, slot)
        disconnect = driver.disconnect

        def counted(*args, _disconnect=disconnect, _slot=slot, **kwargs):
            calls.append((_slot, threading.current_thread().name))
            if _slot == '_camera_driver':
                in_camera.set()
                hold_camera.wait(10)
            return _disconnect(*args, **kwargs)

        driver.disconnect = counted
    return calls, in_camera


def test_the_exit_hook_waits_for_the_close_and_tears_nothing_down_again(session):
    release = threading.Event()
    calls, in_camera = _count_disconnects(session.scope, hold_camera=release)
    closing = threading.Thread(target=session.shutdown, name='session-close')
    closing.start()
    assert in_camera.wait(10)
    threading.Timer(0.3, release.set).start()

    session.scope._emergency_shutdown()
    closing.join(10)

    assert sorted(calls) == sorted((slot, 'session-close') for slot in _DRIVERS)


def test_the_exit_hook_gives_up_after_its_bound(session, monkeypatch):
    monkeypatch.setattr(Lumascope, '_EXIT_WAIT_S', 0.2)
    release = threading.Event()
    calls, in_camera = _count_disconnects(session.scope, hold_camera=release)
    closing = threading.Thread(target=session.shutdown, name='session-close')
    closing.start()
    assert in_camera.wait(10)

    started = time.monotonic()
    session.scope._emergency_shutdown()
    waited = time.monotonic() - started
    release.set()
    closing.join(10)

    assert 0.2 <= waited < 5.0, waited
    assert sorted(calls) == sorted((slot, 'session-close') for slot in _DRIVERS)
