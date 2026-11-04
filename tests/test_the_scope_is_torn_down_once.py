# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope is torn down once, whoever asks and whenever.

The session's close tears the scope down on its own thread. When the process
ended mid-close (a crash in the window, a signal, a platform quit), the exit
hook Lumascope registers ran a second disconnect() beside the close's, with
nothing between them: both threads tore the same camera driver down, and the
second met a handle the first had already released (the LS850T, 2026-10-15:
NoneType on DetachDevice). One disconnect runs at a time; the exit hook waits
for one in flight, up to its bound, and finds nothing left.

A camera has a second teardown of its own: on a definite removal it tears
itself down on a thread of its own. That and the scope's camera step are one
flight too: whichever comes second waits for the first and finds nothing
left, so the SDK's Close, DetachDevice and DestroyDevice each run once.
"""

import threading
import time

import pytest

from modules.lumascope_api import Lumascope
from modules.scope_session import ScopeSession
from tests.camera_fakes import disconnectable_pylon_camera
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


@pytest.fixture
def pylon_camera():
    """A Pylon driver on a fake SDK handle whose Close holds until released."""
    cam = disconnectable_pylon_camera()
    sdk = cam.active
    in_close = threading.Event()
    second_close = threading.Event()
    release = threading.Event()

    def held_close():
        if in_close.is_set():
            second_close.set()
        in_close.set()
        release.wait(10)

    sdk.Close.side_effect = held_close
    cam.second_close = second_close
    yield cam, sdk, in_close, release
    release.set()


def _removal_teardowns_ended(cam):
    for thread in threading.enumerate():
        if thread.name == f'{type(cam).__name__}RemovalTeardown':
            thread.join(10)


def _released_once(sdk):
    return (sdk.Close.call_count, sdk.DetachDevice.call_count, sdk.DestroyDevice.call_count)


def test_a_removal_during_the_close_tears_the_camera_down_once(pylon_camera):
    cam, sdk, in_close, release = pylon_camera
    closing = threading.Thread(target=cam.disconnect, name='session-close')
    closing.start()
    assert in_close.wait(10)
    disconnect = cam.disconnect
    removal_asked = threading.Event()

    def asked_by_the_removal():
        removal_asked.set()
        return disconnect()

    cam.disconnect = asked_by_the_removal

    cam._schedule_async_teardown()
    assert removal_asked.wait(10)
    # Unflighted, the removal's teardown reaches the SDK's Close in about
    # 50 ms; in one flight it waits for the close and never does.
    cam.second_close.wait(0.5)
    release.set()
    closing.join(10)
    _removal_teardowns_ended(cam)

    assert _released_once(sdk) == (1, 1, 1)


def test_the_close_during_a_removal_teardown_tears_the_camera_down_once(pylon_camera):
    cam, sdk, in_close, release = pylon_camera
    cam._schedule_async_teardown()
    assert in_close.wait(10)

    closing = threading.Thread(target=cam.disconnect, name='session-close')
    closing.start()
    release.set()
    closing.join(10)
    _removal_teardowns_ended(cam)

    assert _released_once(sdk) == (1, 1, 1)
