# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 unplugged while it streams is removed, as any other camera is.

An unplug sends the FX2 driver no error: the bytes just stop. Before, nothing
measured the silence, so the camera and the LED -- one USB device -- both
went on answering "connected", a run could start, get_image() handed back the
last frame, and only the GUI's live-view stall notice ever fired. Now the
driver notices (the bytes stop and the device is gone from the bus, or a
transfer reports the device gone), marks the camera removed and tears it down
off its own thread, as the IDS and Pylon drivers do; the LED reads the same
removal from the connection they share. A device still on the bus that goes
quiet is not called unplugged until it has been silent for the ceiling.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from drivers import fx2driver
from drivers.fx2driver import _ByteStream
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


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
    yield session
    session.shutdown()


def _transport(session):
    return session.scope._led_driver._fx2._transport


def test_an_unplug_while_streaming_disconnects_the_camera_and_the_led(session):
    scope = session.scope
    camera = scope._camera_driver
    grab_thread = camera._grab_thread

    _transport(session).unplug()

    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    assert scope.led_connected is False
    assert scope.are_all_connected() is False
    assert scope.imaging.get_image() is None
    assert _wait_until(lambda: not grab_thread.is_alive(), 6.0)
    assert _wait_until(lambda: scope._camera_driver.active is None, 6.0)


def test_a_transfer_that_finds_the_device_gone_removes_it_before_the_silence_bound(session):
    scope = session.scope
    start = time.monotonic()

    _transport(session).unplug(resubmit_failures=4)

    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    assert time.monotonic() - start < fx2driver._UnplugWatch.SILENCE_S


@pytest.mark.slow
def test_a_silent_device_still_on_the_bus_is_removed_only_at_the_ceiling(session, monkeypatch):
    monkeypatch.setattr(fx2driver._UnplugWatch, 'CEILING_S', 4.0)
    scope = session.scope
    start = time.monotonic()

    _transport(session).go_silent()

    assert _wait_until(lambda: time.monotonic() - start > 3.0, 4.0)
    assert scope.camera_connected is True
    assert _wait_until(lambda: not scope.camera_connected, 6.0)


def test_a_gain_written_after_the_removal_is_not_recorded(session):
    scope = session.scope
    before = scope.imaging.gain_db_cached

    _transport(session).unplug()
    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    scope.imaging.set_gain_db(before + 6.0)

    assert scope.imaging.gain_db_cached == before


def test_a_gain_written_before_the_teardown_ends_is_not_recorded(session):
    # Between the camera marking itself removed and its teardown releasing
    # it, the camera is still active: its gain() must answer refused, since
    # the API reads any other answer as applied.
    scope = session.scope
    before = scope.imaging.gain_db_cached
    scope._camera_driver._mark_disconnected()

    scope.imaging.set_gain_db(before + 6.0)

    assert scope.imaging.gain_db_cached == before


def test_the_settings_are_saved_after_the_scope_is_unplugged(session, tmp_path):
    scope = session.scope
    _transport(session).unplug()
    assert _wait_until(lambda: not scope.camera_connected, 6.0)

    out = tmp_path / 'current.json'
    session.save_settings(file=str(out))

    assert out.exists()


def test_the_shutdown_after_a_removal_raises_nothing():
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    session.scope.imaging.start_streaming()
    session.scope._led_driver._fx2._transport.unplug()
    assert _wait_until(lambda: not session.scope.camera_connected, 6.0)

    session.shutdown()


def test_a_second_stop_of_the_stream_does_not_reach_the_transport(session, monkeypatch):
    # The libusb transport's stop closes its context, so a second stop raised
    # AttributeError; the removal's teardown and the app's own shutdown can
    # both stop the stream.
    connection = session.scope._led_driver._fx2
    transport = connection._transport
    stops = []
    real_stop = transport.stop_stream
    monkeypatch.setattr(transport, 'stop_stream', lambda: (stops.append(1), real_stop()))

    connection.stop_stream()
    connection.stop_stream()

    assert len(stops) == 1


def test_the_stream_says_how_long_since_a_byte_arrived():
    stream = _ByteStream()
    stream.restart()
    time.sleep(0.2)
    assert stream.seconds_since_arrival(time.monotonic()) >= 0.2
    stream.packet(b'z')
    assert stream.seconds_since_arrival(time.monotonic()) < 0.1


def test_the_removal_is_not_run_on_the_grab_thread(session):
    # The grab loop cannot join itself; the teardown is the base Camera's thread.
    seen = []
    camera = session.scope._camera_driver
    original = camera.disconnect

    def recording_disconnect():
        seen.append(threading.current_thread().name)
        return original()

    camera.disconnect = recording_disconnect
    _transport(session).unplug()
    assert _wait_until(lambda: seen, 6.0)
    assert seen[0].endswith('RemovalTeardown')


def _streaming_transport(monkeypatch, gone):
    class NoDeviceError(Exception):
        pass

    class NotFoundError(Exception):
        pass

    monkeypatch.setattr(fx2driver.usb1, 'USBErrorNoDevice', NoDeviceError, raising=False)
    monkeypatch.setattr(fx2driver.usb1, 'USBErrorNotFound', NotFoundError, raising=False)
    transport = fx2driver._LibusbTransport()
    transport._stream = _ByteStream()
    transport._on_error = lambda: None
    transport._on_gone = lambda: gone.append(1)
    transport._streaming = True
    return transport, NoDeviceError, NotFoundError


def _cancelled_transfer_whose_resubmit_raises(error):
    def submit():
        raise error

    return SimpleNamespace(getStatus=lambda: object(), iterISO=lambda: iter(()), submit=submit)


def test_a_resubmit_refused_for_a_gone_device_reports_it(monkeypatch):
    gone = []
    transport, NoDeviceError, NotFoundError = _streaming_transport(monkeypatch, gone)

    transport._iso_callback(_cancelled_transfer_whose_resubmit_raises(NoDeviceError('gone')))
    transport._iso_callback(_cancelled_transfer_whose_resubmit_raises(NotFoundError('gone')))

    assert len(gone) == 2


def test_a_resubmit_that_fails_otherwise_reports_no_removal(monkeypatch):
    gone = []
    transport, _, _ = _streaming_transport(monkeypatch, gone)

    transport._iso_callback(_cancelled_transfer_whose_resubmit_raises(RuntimeError('busy')))

    assert gone == []


def test_the_shutdown_after_a_removal_writes_nothing_to_the_gone_led():
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    session.scope.imaging.start_streaming()
    session.scope._led_driver._fx2._transport.unplug()
    assert _wait_until(lambda: not session.scope.camera_connected, 6.0)
    led = session.scope._led_driver
    writes = []
    led.leds_off = lambda: writes.append('leds_off')
    led.led_off = lambda channel: writes.append(('led_off', channel))

    session.shutdown()

    assert writes == []
