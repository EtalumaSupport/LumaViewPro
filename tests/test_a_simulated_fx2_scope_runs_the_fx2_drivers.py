# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated LS620 or LS560 runs the production FX2 drivers over a simulated FX2.

Before, the simulator stood an EL-0940-shaped Python LED board and a generic
simulated camera in for every FX2 model: 1920x1200 with auto gain, so a
simulated LS560 showed an Auto Gain checkbox the real one does not have, and
no FX2 code path ran without hardware. Now both tiers build ``FX2Camera`` and
``FX2LEDController`` on one simulated device, which takes the firmware upload,
answers the sensor and LED writes as the wire carries them, and streams
frames at the bench's period that are black unless its LED peripheral holds a
channel lit.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivers import fx2driver
from drivers.fx2driver import FX2Camera, FX2LEDController
from drivers.simulated_fx2 import (
    TRANSFER_S,
    SimulatedFX2Device,
    bytes_per_transfer,
    frame_period_s,
)
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

SCOPES = [
    pytest.param((model, tier), id=f'{model}-{tier}')
    for model in ('LS620', 'LS560')
    for tier in ('fast', 'firmware')
]


@pytest.fixture(scope='module', params=SCOPES)
def session(request):
    model, tier = request.param
    settings = complete_settings()
    settings['microscope'] = model
    settings['simulator_tier'] = tier
    session = ScopeSession.create(settings, simulate=True)
    yield session
    session.shutdown()


def _device(session) -> SimulatedFX2Device:
    return session.scope._led_driver._fx2._transport.device


def _fresh_frame(scope) -> np.ndarray:
    # A lit or darkened LED shows from the next frame the sensor starts, so
    # the frame in flight at the change is passed over.
    scope.imaging.get_image(force_new_capture=True)
    return scope.imaging.get_image(force_new_capture=True)


def test_the_scope_runs_the_fx2_drivers_on_the_device_it_uploaded(session):
    scope = session.scope
    assert isinstance(scope._camera_driver, FX2Camera)
    assert isinstance(scope._led_driver, FX2LEDController)
    assert scope._camera_driver._fx2 is scope._led_driver._fx2
    device = _device(session)
    assert device.uploads == 1
    assert device.pid == fx2driver.PID_APP


def test_the_camera_is_the_fx2s_mt9p031(session):
    caps = session.scope.capabilities
    camera = session.scope._camera_driver
    assert caps.camera_model == 'MT9P031-LS620'
    assert camera.get_max_frame_size() == {'width': 1900, 'height': 1900}
    assert camera.get_supported_pixel_formats() == ('Mono8',)
    assert camera.max_gain == 42.1
    assert camera.max_exposure == 178.0
    assert caps.camera_supports_auto_gain is False


def test_a_frame_is_black_with_nothing_lit_and_shows_the_field_with_bf_lit(session):
    scope = session.scope
    scope.illumination.leds_off()
    dark = _fresh_frame(scope)
    assert dark is not None and dark.shape == (1900, 1900)
    assert dark.max() == 0

    scope.illumination.led_on('BF', 100)
    try:
        lit = _fresh_frame(scope)
    finally:
        scope.illumination.leds_off()
    assert lit.mean() > 20


def test_840_ma_reaches_the_peripheral_as_0xfe(session):
    session.scope._led_driver.led_on(3, 840)
    try:
        assert _device(session).leds.commands[-1] == ('D', 0xFE)
    finally:
        session.scope._led_driver.leds_off()


def test_blue_reaches_the_peripheral_as_c(session):
    session.scope._led_driver.led_on(0, 100)
    try:
        assert _device(session).leds.commands[-1][0] == 'C'
    finally:
        session.scope._led_driver.leds_off()


# ---------------------------------------------------------------------------
# The device model, without a scope
# ---------------------------------------------------------------------------


def _running_device() -> SimulatedFX2Device:
    from drivers.simulated_fx2 import SimulatedFX2

    return SimulatedFX2().device


def test_a_0xff_brightness_is_a_new_preamble_and_the_command_is_lost():
    device = _running_device()
    for byte in (0xFF, ord('D'), 0x40, 0xFF, ord('D'), 0xFF):
        device.vendor_out(fx2driver.VR_I2C_WRITE, 0, fx2driver.I2C_LED, bytes([byte]))
    assert device.leds.commands == [('D', 0x40)]
    assert device.leds.lost_commands == 1
    assert device.leds.brightness[ord('D')] == 0x40


@pytest.mark.parametrize(
    'window, fps',
    [((1900, 1900), 4.5), ((1000, 1000), 12.0)],
    ids=['1900x1900', '1000x1000'],
)
def test_the_frame_period_is_the_benchs(window, fps):
    assert frame_period_s(*window, exposure_s=0.050) == pytest.approx(1 / fps)


def test_a_long_exposure_stretches_the_frame():
    assert frame_period_s(1000, 1000, exposure_s=0.178) == pytest.approx(0.178)


def test_the_stream_arrives_a_transfer_of_the_wires_bytes_at_a_time_frames_back_to_back():
    # A whole frame delivered at once left the grab loop spinning on a
    # frame's worth of bytes with no delimiter after it, a core at 100%; a
    # transfer always full bunched a small window's frames seconds apart.
    import threading

    device = _running_device()
    device.sensor.write(bytes([fx2driver.REG_COL_SIZE, 0, 101]))  # the driver writes w + 1
    device.sensor.write(bytes([fx2driver.REG_ROW_SIZE, 0, 81]))
    frame_bytes = len(fx2driver.FRAME_DELIM) + fx2driver.frame_layout(100, 80).frame_bytes
    chunks: list[bytes] = []
    two_frames = threading.Event()

    def sink(data: bytes) -> None:
        chunks.append(data)
        if sum(map(len, chunks)) > 2 * frame_bytes:
            two_frames.set()

    device.attach(sink)
    device.vendor_out(fx2driver.VR_START_STREAMING, 0, 0, b'')
    try:
        assert two_frames.wait(5.0)
    finally:
        device.stop()
    per_transfer = bytes_per_transfer(100, 80, device.sensor.exposure_s())
    assert per_transfer < frame_bytes
    assert all(abs(len(c) - per_transfer) <= 1 for c in chunks)
    stream = b''.join(chunks)
    assert stream.find(fx2driver.FRAME_DELIM, 1) == frame_bytes


@pytest.mark.parametrize('window', [(1900, 1900), (1000, 1000)], ids=['1900x1900', '1000x1000'])
def test_a_frames_bytes_take_its_period_on_the_wire(window):
    w, h = window
    frame_bytes = len(fx2driver.FRAME_DELIM) + fx2driver.frame_layout(w, h).frame_bytes
    transfers = frame_bytes / bytes_per_transfer(w, h, 0.050)
    assert transfers * TRANSFER_S == pytest.approx(frame_period_s(w, h, 0.050))


def test_a_frame_is_as_long_as_the_parser_accepts():
    device = _running_device()
    device.sensor.write(bytes([fx2driver.REG_COL_SIZE, 0, 101]))  # the driver writes w + 1
    device.sensor.write(bytes([fx2driver.REG_ROW_SIZE, 0, 81]))
    frame = device.frame()
    assert frame.startswith(fx2driver.FRAME_DELIM)
    assert len(frame) - len(fx2driver.FRAME_DELIM) == fx2driver.frame_layout(100, 80).frame_bytes


def test_a_request_the_device_does_not_model_raises():
    device = _running_device()
    with pytest.raises(ValueError, match='no request'):
        device.vendor_out(fx2driver.VR_INIT_GPIF, 0, 0, b'')
    with pytest.raises(ValueError, match='no I2C device'):
        device.vendor_out(fx2driver.VR_I2C_WRITE, 0, 0x50, b'\x00')


def test_an_image_that_is_not_the_shipped_firmware_is_refused():
    device = SimulatedFX2Device()
    device.vendor_out(fx2driver.VR_ANCHOR_DLD, 0xE600, 0, b'\x01')
    device.vendor_out(fx2driver.VR_ANCHOR_DLD, 0, 0, b'\x02\x00\x00')
    with pytest.raises(RuntimeError, match='not the shipped firmware'):
        device.vendor_out(fx2driver.VR_ANCHOR_DLD, 0xE600, 0, b'\x00')
    assert device.pid == fx2driver.PID_BOOT


def test_the_support_report_says_the_led_diagnostics_are_not_supported(session):
    from modules.tech_support_report import FirmwareDiagnostics

    assert session.scope.diagnostics.get_led_info()['command_set'] is None
    answer = FirmwareDiagnostics(scope=session.scope).get_led_info()
    assert answer.startswith('Not supported on this LED board'), answer
