# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2's window is placed and read out as the MT9P031 documents give.

The driver sets Mirror_Row and leaves Mirror_Column off, so a target reads as
it does on the LS850. Without the column mirror the register
reference requires Column_Start in the form 4n, and the columns read out in
numerical order from Column_Start, so the columns beyond the window trail it.
Row_Start is LumaView Classic's. And the simulated FX2 sends sensor data in
its first and last output rows: the LS620's test patterns show both carry it.

The expected values are the documents' and the bench's, written as literals.
"""

from __future__ import annotations

import pytest

from drivers import fx2driver
from drivers.fx2driver import FX2Camera

# DS: the active array is 2592 columns from column 16, so its centre is
# column 16 + 2592 / 2.
ARRAY_CENTRE = 1312


@pytest.fixture
def sim():
    from drivers.simulated_fx2 import SimulatedFX2

    return SimulatedFX2()


@pytest.fixture
def camera(sim):
    cam = FX2Camera(connection=sim.connection)
    yield cam
    cam.disconnect()


def test_every_window_starts_on_a_4n_column_and_is_centred(sim, camera):
    registers = sim.device.sensor.registers
    for w in range(100, 1901, 4):
        camera.set_frame_size(w, 100)
        start = registers[fx2driver.REG_COL_START]
        # RR R0x02: Column_Start is 4n with Mirror_Column clear.
        assert start % 4 == 0, (w, start)
        # DS Table 8: the sensor outputs W = Column_Size + 1 = w + 4 columns.
        assert abs(start + (w + 4) / 2 - ARRAY_CENTRE) <= 2, (w, start)


@pytest.mark.parametrize(
    'w, start',
    [
        (1900, 360),
        # The centred start is 810, halfway between 808 and 812: the lower.
        (1000, 808),
        (500, 1060),
        (100, 1260),
    ],
)
def test_the_column_start_is_the_nearest_4n_to_the_centre(sim, camera, w, start):
    camera.set_frame_size(w, w)
    assert sim.device.sensor.registers[fx2driver.REG_COL_START] == start


def test_every_window_writes_column_size_4n_minus_1(sim, camera):
    # RR R0x04: Column_Size in the form 4n - 1; w + 3, since w - 1 leaves too
    # few pixels a row.
    registers = sim.device.sensor.registers
    for w in range(100, 1901, 4):
        camera.set_frame_size(w, 100)
        assert registers[fx2driver.REG_COL_SIZE] == w + 3, w


def _classic_row_start(h):
    # LumaView Classic, AptinaMT9P031_Control3.cs SetWindowSize: the centred
    # start of an h-row window on the 1944-row array from row 54, made even
    # by adding one.
    start = (1944 - h) // 2 + 54
    return start + start % 2


def test_every_window_starts_on_classics_row(sim, camera):
    registers = sim.device.sensor.registers
    for h in range(100, 1901, 4):
        camera.set_frame_size(100, h)
        assert registers[fx2driver.REG_ROW_START] == _classic_row_start(h), h
    camera.set_frame_size(1900, 1900)
    assert registers[fx2driver.REG_ROW_START] == 76


def test_the_stored_pixels_are_the_first_w_a_row_carries():
    # With Mirror_Column clear the columns Column_Size adds beyond the window
    # come last on the wire (RR R0x020); storing the last w would move the
    # image two sensor columns. Each wire byte here carries its own column index.
    import threading
    import time
    from types import SimpleNamespace

    w, h = 100, 80
    layout = fx2driver.frame_layout(w, h)
    row = bytes(range(layout.stride - 1)) + b'\x00'
    body = bytes(layout.skip - layout.stride) + row * (h + 2)
    assert len(body) == layout.frame_bytes
    t = fx2driver.ISO_TRANSACTION_SIZE
    whole = len(body) - len(body) % t
    stream = fx2driver._ByteStream()
    stream.packet(fx2driver.FRAME_DELIM)
    for i in range(0, whole, t):
        stream.packet(body[i : i + t])
    stream.packet(body[whole:])

    cam = object.__new__(FX2Camera)
    cam._fx2 = SimpleNamespace(
        stream=stream, take_gone_report=lambda: False, device_present=lambda: True
    )
    cam._grabbing = True
    cam._width, cam._height = w, h
    cam.stream_stats = fx2driver.StreamStats()
    stored = []
    cam.cam_image_handler = SimpleNamespace(
        _store_frame=lambda image, ts, significant_bits: stored.append(image)
    )
    loop = threading.Thread(target=cam._grab_loop, daemon=True)
    loop.start()
    deadline = time.monotonic() + 2.0
    while not stored and time.monotonic() < deadline:
        time.sleep(0.01)
    cam._grabbing = False
    loop.join(2.0)

    assert len(stored) == 1
    image = stored[0]
    assert image.shape == (h, w)
    assert (image == list(range(w))).all()


def test_the_simulated_frame_puts_the_window_where_the_parser_stores_it(sim, monkeypatch):
    # The wire trails each row with the columns beyond the window; a simulator
    # that put its pixels last would have the parser store an image moved
    # two columns, with the window's first two columns lost.
    import numpy as np

    device = sim.device
    device.sensor.write(bytes([fx2driver.REG_COL_SIZE, 0, 103]))
    device.sensor.write(bytes([fx2driver.REG_ROW_SIZE, 0, 81]))
    pattern = np.tile(np.arange(1, 101, dtype=np.uint8), (82, 1))
    monkeypatch.setattr(device, '_pixels', lambda w, h: pattern)
    layout = fx2driver.frame_layout(100, 80)
    body = np.frombuffer(device.frame(), dtype=np.uint8)
    rows = body[layout.skip : layout.needed].reshape(80, layout.stride)
    assert (rows[:, :100] == pattern[1:81]).all()


def test_the_simulated_frame_carries_sensor_rows_where_the_parser_stores_none(sim):
    device = sim.device
    device.sensor.write(bytes([fx2driver.REG_COL_SIZE, 0, 103]))
    device.sensor.write(bytes([fx2driver.REG_ROW_SIZE, 0, 81]))
    device.leds.brightness[ord('A')] = 255
    layout = fx2driver.frame_layout(100, 80)
    body = device.frame()
    first, last = body[: layout.skip], body[layout.needed :]
    assert any(first[:100]), 'the first output row is sensor data, not zeros'
    assert any(last[:100]), 'the last output row is sensor data, not zeros'
    stored = [
        body[layout.skip + r * layout.stride : layout.skip + (r + 1) * layout.stride]
        for r in range(80)
    ]
    assert all(row[layout.stride - 1] == 0 for row in stored), 'each row ends in the 0 sync byte'


@pytest.mark.parametrize(
    'mode, order',
    [
        (0x0040, (slice(None, None, 1), slice(None, None, 1))),
        (0x4040, (slice(None, None, 1), slice(None, None, -1))),
        (0x8040, (slice(None, None, -1), slice(None, None, 1))),
        (0xC040, (slice(None, None, -1), slice(None, None, -1))),
    ],
)
def test_the_simulated_sensor_reads_out_in_the_order_read_mode_2_sets(sim, mode, order):
    # RR R0x020: Mirror_Row (bit 15) and Mirror_Column (bit 14) each reverse
    # their axis of the active image.
    import numpy as np

    sensor = sim.device.sensor
    sensor.write(bytes([fx2driver.REG_READ_MODE2, mode >> 8, mode & 0xFF]))
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert (sensor.read_out(pixels) == pixels[order]).all()


def test_the_simulated_frame_carries_the_specimen_upright_at_the_drivers_read_mode(
    sim, camera, monkeypatch
):
    # The simulated FX2's optics put the specimen on the sensor upside down,
    # as the FX2 scopes' do; the driver's Mirror_Row delivers it the way up
    # every simulated camera delivers it. Each specimen row carries its index.
    import numpy as np

    device = sim.device
    w, h = device.sensor.window()
    pattern = np.repeat((np.arange(h + 2) % 251).astype(np.uint8)[:, None], w, axis=1)
    monkeypatch.setattr(device, '_pixels', lambda w, h: pattern)
    layout = fx2driver.frame_layout(w, h)
    body = np.frombuffer(device.frame(), dtype=np.uint8)
    rows = body[layout.skip : layout.needed].reshape(h, layout.stride)
    assert (rows[:, :w] == pattern[1 : h + 1]).all()
