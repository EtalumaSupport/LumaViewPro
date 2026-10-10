# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 stream the parser cannot frame says so, once, and stays connected.

A device whose frames are all the wrong length for the window keeps sending
bytes, so it is not unplugged, but it stores no frame either: the live view
freezes and captures time out. Before, the only trace was the shifted count
climbing in the INFO stream line every 10 s, and the line written when the
stream stopped left the shifted count out. Now the grab loop logs one WARNING
when bytes have kept arriving with no frame stored, naming what it discarded
and the window, and a stored frame ends that episode.
"""

from __future__ import annotations

import logging
import time

import pytest

from drivers import fx2driver
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_UNFRAMEABLE = 'no frame stored for'


def _wait_until(condition, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return condition()


@pytest.fixture
def session():
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    session.scope.imaging.start_streaming()
    assert _wait_until(lambda: session.scope.imaging.get_image() is not None, 5.0)
    session.scope._camera_driver.FRAMING_STALL_S = 0.5
    yield session
    session.shutdown()


@pytest.fixture
def log(caplog, monkeypatch):
    # The suite replaces lvp_logger with a mock; the driver's lines go to a
    # real logger here so the test can read them.
    monkeypatch.setattr(fx2driver, 'logger', logging.getLogger('fx2_under_test'))
    caplog.set_level(logging.INFO, logger='fx2_under_test')
    return caplog


def _transport(session):
    return session.scope._led_driver._fx2._transport


def _unframeable(log):
    return [
        r for r in log.records if r.levelno == logging.WARNING and _UNFRAMEABLE in r.getMessage()
    ]


def test_a_misaligned_stream_warns_once_and_stays_connected(session, log):
    _transport(session).misalign()

    assert _wait_until(lambda: _unframeable(log), 3.0)
    time.sleep(1.5)

    assert len(_unframeable(log)) == 1
    message = _unframeable(log)[0].getMessage()
    assert 'shifted' in message
    assert 'window' in message
    assert session.scope.camera_connected is True


def test_a_stored_frame_ends_the_episode(session, log):
    camera = session.scope._camera_driver
    transport = _transport(session)
    transport.misalign()
    assert _wait_until(lambda: _unframeable(log), 3.0)

    transport.misalign(0)
    good = camera.stream_stats.summary()['good_frames']
    assert _wait_until(lambda: camera.stream_stats.summary()['good_frames'] > good, 3.0)
    transport.misalign()

    assert _wait_until(lambda: len(_unframeable(log)) == 2, 3.0)


def test_the_stop_line_carries_the_shifted_count(session, log):
    camera = session.scope._camera_driver
    _transport(session).misalign()
    assert _wait_until(lambda: camera.stream_stats.summary()['shifted_frames'] > 0, 3.0)

    camera.stop_grabbing()

    shifted = camera.stream_stats.summary()['shifted_frames']
    stop_lines = [r.getMessage() for r in log.records if 'streaming stopped' in r.getMessage()]
    assert stop_lines
    assert f'{shifted} shifted' in stop_lines[-1]
