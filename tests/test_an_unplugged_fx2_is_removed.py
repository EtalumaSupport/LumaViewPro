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
quiet is never called unplugged: at the ceiling its camera is torn down, as a
stream that stopped and did not recover, and its LED, which the bus still
lists, is not declared gone.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from drivers import fx2driver
from drivers.exceptions import BoardPortFailedError, CameraWriteFailedError
from modules.exceptions import CaptureFailedError, HardwareCommandRefusedError, MissingPart
from modules.lumascope_api.illumination import LedState, LedTransition, LedTransitionCtx
from modules.scope_session import ScopeSession
from tests.camera_fakes import fx2_stream
from tests.protocol_drives import held_run_claim
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
    session.scope.imaging.enable_streaming()
    assert _wait_until(lambda: _delivers(session.scope), 5.0)
    yield session
    session.shutdown()


def _delivers(scope) -> bool:
    """Whether a grab now returns a frame; one that got none raises."""
    try:
        scope.imaging.get_image()
    except CaptureFailedError:
        return False
    return True


def _transport(session):
    return session.scope._led_driver._fx2._transport


def test_an_unplug_while_streaming_disconnects_the_camera_and_the_led(session):
    scope = session.scope
    camera = scope._camera_driver
    grab_thread = camera._grab_thread

    session.scope.simulation.unplug('led')

    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    assert scope.led_connected is False
    assert scope.are_all_connected() is False
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.imaging.get_image()
    assert refused.value.reason == 'not_connected'
    assert _wait_until(lambda: not grab_thread.is_alive(), 6.0)
    assert _wait_until(lambda: scope._camera_driver.active is None, 6.0)


def test_a_transfer_that_finds_the_device_gone_removes_it_before_the_silence_bound(session):
    scope = session.scope
    start = time.monotonic()

    _transport(session).unplug(resubmit_failures=4)

    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    assert time.monotonic() - start < fx2driver._UnplugWatch.SILENCE_S


@pytest.mark.slow
def test_a_silent_device_still_on_the_bus_loses_its_camera_at_the_ceiling_and_keeps_its_led(
    session, monkeypatch
):
    monkeypatch.setattr(fx2driver._UnplugWatch, 'CEILING_S', 4.0)
    scope = session.scope
    start = time.monotonic()

    session.scope.simulation.silence('led')

    assert _wait_until(lambda: time.monotonic() - start > 3.0, 4.0)
    assert scope.camera_connected is True
    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    assert scope.led_connected is True


def test_a_gain_written_after_the_removal_is_refused_and_reads_unknown(session):
    scope = session.scope
    before = scope.camera.gain_db

    session.scope.simulation.unplug('led')
    assert _wait_until(lambda: not scope.camera_connected, 6.0)
    with pytest.raises(HardwareCommandRefusedError) as exc:
        scope.camera.set_gain_db(before + 6.0)

    assert exc.value.missing == MissingPart.CAMERA
    assert scope.camera.gain_db is None


def test_a_write_to_a_camera_removed_while_it_is_not_streaming_is_refused_not_connected():
    # No stream, so no grab loop hears the silence: the failed write is the
    # driver's only word that the device is gone, and the bus decides it.
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    scope = session.scope
    try:
        scope.imaging.disable_streaming()
        assert scope.imaging.is_streaming() is False
        before = scope.camera.gain_db
        scope.simulation.remove_camera()

        with pytest.raises(HardwareCommandRefusedError) as refused:
            scope.camera.set_gain_db(before + 6.0)

        assert refused.value.reason == 'not_connected'
        assert refused.value.missing == MissingPart.CAMERA
        assert scope.camera_connected is False
        assert scope.led_connected is False
        assert scope.camera.gain_db is None
    finally:
        session.shutdown()


def _ls620_not_streaming_then_unplugged():
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    session.scope.imaging.disable_streaming()
    assert session.scope.imaging.is_streaming() is False
    session.scope.simulation.remove_camera()
    return session


def test_a_stream_start_on_a_camera_removed_while_it_is_not_streaming_is_refused_not_connected():
    # The start is the first transfer since the pull, so it is what finds
    # the device gone; its fault is not the answer for a camera not there.
    session = _ls620_not_streaming_then_unplugged()
    scope = session.scope
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            scope.imaging.enable_streaming()

        assert refused.value.reason == 'not_connected'
        assert refused.value.missing == MissingPart.CAMERA
        assert scope.camera_connected is False
    finally:
        session.shutdown()


def test_the_grab_benchmark_on_a_camera_removed_under_it_is_refused_not_connected():
    # The benchmark keeps cycling past a failed cycle, so a pull it finds
    # must still end it, not be returned as one cycle's error.
    session = _ls620_not_streaming_then_unplugged()
    scope = session.scope
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            scope.diagnostics.run_grab_lifecycle_benchmark(num_cycles=2)

        assert refused.value.reason == 'not_connected'
        assert refused.value.missing == MissingPart.CAMERA
    finally:
        session.shutdown()


def test_a_stream_stop_that_finds_its_camera_removed_is_satisfied(session, monkeypatch):
    # An off holds without a camera, so finding the camera gone satisfies it.
    scope = session.scope
    driver = scope._camera_driver

    def stop_that_finds_the_device_gone():
        # a stand-in by design: the simulated FX2's stop never fails, so
        # the transfer that finds the device gone is staged here.
        driver._fx2.report_removed('off the bus during the stop')
        raise CameraWriteFailedError('The camera did not take the stop')

    monkeypatch.setattr(driver, 'stop_grabbing', stop_that_finds_the_device_gone)

    assert scope.imaging.disable_streaming() is None
    assert scope.camera.camera_removed is True


def test_an_led_on_that_finds_the_device_unplugged_is_refused_not_connected():
    # The LED shares the camera's USB device; its write is the first
    # transfer since the pull, so it is what finds the device gone.
    session = _ls620_not_streaming_then_unplugged()
    scope = session.scope
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            scope.illumination.led_on('BF', 10.0)

        assert refused.value.reason == 'not_connected'
        assert refused.value.missing == MissingPart.LED_CONTROLLER
        assert scope.led_connected is False
    finally:
        session.shutdown()


def test_an_all_off_that_finds_the_device_unplugged_with_nothing_lit_is_satisfied():
    session = _ls620_not_streaming_then_unplugged()
    scope = session.scope
    try:
        assert scope.illumination.leds_off() is None
        assert scope.led_connected is False
    finally:
        session.shutdown()


def test_the_scopes_own_off_that_finds_the_device_unplugged_reports_no_controller():
    # Bring-up, shutdown and the safety off call it directly; its answer
    # is the only thing it tells, so the answer is what is asserted.
    session = _ls620_not_streaming_then_unplugged()
    scope = session.scope
    try:
        assert scope.illumination._leds_off_if_present() is False
    finally:
        session.shutdown()


def test_the_disconnect_off_that_finds_the_device_unplugged_writes_nothing_and_does_not_raise():
    # The disconnect reaches it first when the lane's own off never ran.
    session = _ls620_not_streaming_then_unplugged()
    scope = session.scope
    try:
        assert scope.illumination._leds_off_emergency() is None
    finally:
        session.shutdown()


def test_a_lease_release_that_finds_the_device_unplugged_is_refused_and_its_led_stays_lit():
    # The release could not darken the LED, so it must not be believed dark.
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    scope = session.scope
    try:
        scope.imaging.disable_streaming()
        lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
        channel = scope.illumination.color2ch(color='BF')
        lease.apply(
            LedTransition.MANUAL_STEP,
            LedTransitionCtx(channel=channel, illumination_ma=10.0, preview_on=True),
        )
        assert scope.illumination.get_led_state('BF').state is LedState.LIT
        scope.simulation.remove_camera()

        with pytest.raises(HardwareCommandRefusedError) as refused:
            lease.release()

        assert refused.value.reason == 'not_connected'
        assert refused.value.missing == MissingPart.LED_CONTROLLER
        assert scope.illumination.get_led_state('BF').state is LedState.LIT
    finally:
        session.shutdown()


def test_an_led_write_refused_with_the_device_still_present_keeps_its_fault(session):
    # Only a controller gone turns a fault into not_connected: one still
    # on the bus refused this write, and that is what the caller hears.
    scope = session.scope
    scope.simulation.refuse_led_write(1)

    with pytest.raises(BoardPortFailedError) as failed:
        scope.illumination.led_on('BF', 10.0)

    assert failed.value.reason == 'board_port_failed'
    assert scope.led_connected is True


def test_a_gain_written_before_the_teardown_ends_is_refused_and_reads_unknown(session):
    # Between the camera marking itself removed and its teardown releasing
    # it, the camera is still active, but it no longer answers connected:
    # the camera lane refuses the write before it reaches gain(), and the
    # held gain reads unknown from the mark on.
    scope = session.scope
    before = scope.camera.gain_db
    scope._camera_driver._mark_disconnected()

    with pytest.raises(HardwareCommandRefusedError) as exc:
        scope.camera.set_gain_db(before + 6.0)

    assert exc.value.missing == MissingPart.CAMERA
    assert scope.camera.gain_db is None


def test_the_settings_are_saved_after_the_scope_is_unplugged(tmp_path):
    # Found at bring-up, not connected now: the session's values came from
    # the scope, so its close still saves them.
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    out = tmp_path / 'current.json'
    session = ScopeSession.create(settings, simulate=True, settings_file=str(out))
    session.scope.imaging.enable_streaming()
    session.scope.simulation.unplug('led')
    assert _wait_until(lambda: not session.scope.camera_connected, 6.0)

    session.shutdown()

    assert out.exists()


def test_the_shutdown_after_a_removal_raises_nothing():
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    session.scope.imaging.enable_streaming()
    session.scope.simulation.unplug('led')
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
    stream = fx2_stream()
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
    session.scope.simulation.unplug('led')
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
    transport._stream = fx2_stream()
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
    session.scope.imaging.enable_streaming()
    session.scope.simulation.unplug('led')
    assert _wait_until(lambda: not session.scope.camera_connected, 6.0)
    led = session.scope._led_driver
    writes = []
    led.leds_off = lambda: writes.append('leds_off')
    led.led_off = lambda channel: writes.append(('led_off', channel))

    session.shutdown()

    assert writes == []
