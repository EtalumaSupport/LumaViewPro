# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2's window is placed and read out as the MT9P031 documents give.

Before, Column_Start was the centred value rounded to even, which lands on
4n at about half the widths (360 at 1900, 1060 at 500), where the register
reference requires 4n + 2 under Mirror_Column, which the driver sets. And the
simulated FX2 sent zeros where the sensor sends its first and last output
rows: the LS620's test patterns show both carry sensor data.

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


def test_every_window_starts_on_a_mirrored_column_and_is_centred(sim, camera):
    registers = sim.device.sensor.registers
    for w in range(100, 1901, 4):
        camera.set_frame_size(w, 100)
        start = registers[fx2driver.REG_COL_START]
        # RR R0x02: Column_Start is 4n + 2 under Mirror_Column.
        assert start % 4 == 2, (w, start)
        # DS Table 8: the sensor outputs W = Column_Size + 1 = w + 2 columns.
        assert abs(start + (w + 2) / 2 - ARRAY_CENTRE) <= 2, (w, start)


@pytest.mark.parametrize(
    'w, start', [(1900, 362), (1000, 810), (500, 1062), (1896, 362), (1880, 370)]
)
def test_the_column_start_at_the_benched_widths(sim, camera, w, start):
    camera.set_frame_size(w, w)
    assert sim.device.sensor.registers[fx2driver.REG_COL_START] == start


def test_every_window_writes_column_size_4n_minus_1_and_keeps_its_column_start(sim, camera):
    # RR R0x04: Column_Size in the form 4n - 1; w + 3, since w - 1 leaves too
    # few pixels a row. Column_Start stays what a Column_Size of w + 1 gave,
    # so the stored image is the same sensor columns.
    registers = sim.device.sensor.registers
    for w in range(100, 1901, 4):
        camera.set_frame_size(w, 100)
        assert registers[fx2driver.REG_COL_SIZE] == w + 3, w
        start_at_w_plus_1 = ((2592 - (w + 1)) // 2 + 16) // 4 * 4 + 2
        assert registers[fx2driver.REG_COL_START] == start_at_w_plus_1, w


def test_the_stored_pixels_are_the_last_w_a_row_carries():
    # Under Mirror_Column the columns Column_Size adds beyond the window come
    # first on the wire; storing the first w would move the image two sensor
    # columns. Each wire byte here carries its own column index.
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
    assert (image == list(range(2, w + 2))).all()


def test_the_simulated_frame_puts_the_window_where_the_parser_stores_it(sim, monkeypatch):
    # The wire leads each row with the columns beyond the window; a simulator
    # that put its pixels first would have the parser store an image moved
    # two columns, with the window's last two columns lost.
    import numpy as np

    device = sim.device
    device.sensor.write(bytes([fx2driver.REG_COL_SIZE, 0, 103]))
    device.sensor.write(bytes([fx2driver.REG_ROW_SIZE, 0, 81]))
    pattern = np.tile(np.arange(1, 101, dtype=np.uint8), (82, 1))
    monkeypatch.setattr(device, '_pixels', lambda w, h: pattern)
    layout = fx2driver.frame_layout(100, 80)
    body = np.frombuffer(device.frame(), dtype=np.uint8)
    rows = body[layout.skip : layout.needed].reshape(80, layout.stride)
    assert (rows[:, layout.column : layout.column + 100] == pattern[1:81]).all()


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
