# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 driver reaches USB through one transport, and its stream has one buffer.

Before, the camera called the USB libraries itself and wrote the connection's
private routing state, so nothing short of libusb and a real device could run
it. On Windows the camera read the WinUSB reader's own bytearray and took from
it by replacing it with a new one, so after the first take the reader kept
filling an object the parser no longer read: no frame was stored after the
first, and the reader's buffer grew by the stream's rate. Now the connection
owns the stream, every reader hands its packets to it, and the transport
moves control onto the stream's handle and back.
"""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from drivers import fx2driver

DRIVER = Path(__file__).resolve().parent.parent / 'drivers' / 'fx2driver.py'


# ---------------------------------------------------------------------------
# The grab loop stores the frames a reader appends
# ---------------------------------------------------------------------------

W = H = 100
_BODY = bytes(fx2driver.frame_layout(W, H).frame_bytes)
# One well-formed frame as the device sends it: the delimiter, then the frame
# in packets of two transactions, the last one short.
_STEP = 2 * fx2driver.ISO_TRANSACTION_SIZE
_FRAME = [fx2driver.FRAME_DELIM] + [_BODY[i : i + _STEP] for i in range(0, len(_BODY), _STEP)]


def test_the_grab_loop_stores_every_frame_a_reader_hands_on_after_its_first_take():
    stream = fx2driver._ByteStream()
    cam = object.__new__(fx2driver.FX2Camera)
    cam._fx2 = SimpleNamespace(
        stream=stream, take_gone_report=lambda: False, device_present=lambda: True
    )
    cam._grabbing = True
    cam._width, cam._height = W, H
    cam.stream_stats = fx2driver.StreamStats()
    stored = []
    cam.cam_image_handler = SimpleNamespace(
        _store_frame=lambda image, ts, significant_bits, wire_bytes: stored.append(image.shape)
    )

    loop = threading.Thread(target=cam._grab_loop, daemon=True)
    loop.start()
    for _ in range(40):
        for packet in _FRAME:
            stream.packet(packet)  # the reader's only way in
        time.sleep(0.005)
    deadline = time.monotonic() + 2.0
    while len(stored) < 40 and time.monotonic() < deadline:
        time.sleep(0.01)
    cam._grabbing = False
    loop.join(2.0)

    # Every one: a frame ends in its own last packet, not at the next delimiter.
    assert len(stored) == 40
    assert set(stored) == {(H, W)}


# ---------------------------------------------------------------------------
# Control moves onto the stream's handle while it runs, and back after
# ---------------------------------------------------------------------------


class _PyusbDevice:
    """A pyusb device: records the vendor requests sent to it."""

    def __init__(self, events, name):
        self._events, self._name = events, name

    def is_kernel_driver_active(self, interface):
        return False

    def set_configuration(self):
        pass

    def ctrl_transfer(self, request_type, request, value, index, data, timeout=None):
        self._events.append((self._name, request))
        return len(data)


class _IsoHandle:
    """A python-libusb1 device handle: records control writes and transfer submits."""

    def __init__(self, events):
        self._events = events
        self.drained = False

    def kernelDriverActive(self, interface):
        return False

    def claimInterface(self, interface):
        pass

    def setInterfaceAltSetting(self, interface, alt):
        pass

    def getTransfer(self, iso_packets):
        def cancel():
            self.drained = True

        return SimpleNamespace(
            setIsochronous=lambda *a, **kw: None,
            submit=lambda: self._events.append(('iso', 'submit')),
            cancel=cancel,
        )

    def controlWrite(self, request_type, request, value, index, data, timeout=None):
        self._events.append(('iso', request))
        return len(data)

    def releaseInterface(self, interface):
        pass

    def close(self):
        pass


class _Context:
    def __init__(self, handle):
        self._handle = handle

    def open(self):
        return self

    def openByVendorIDAndProductID(self, vid, pid):
        return self._handle

    def handleEventsTimeout(self, tv):
        if self._handle.drained:
            raise RuntimeError('no events left')  # ends stop_stream's drain
        time.sleep(0.005)

    def close(self):
        pass


def _connection_on_libusb(monkeypatch, events):
    """A real _FX2Connection on a real _LibusbTransport, over stand-in libraries.

    The first device found is ``first``; the one the transport reopens after
    the stream stops is ``second``.
    """
    found = iter([_PyusbDevice(events, 'first'), _PyusbDevice(events, 'second')])
    handle = _IsoHandle(events)
    monkeypatch.setattr(
        fx2driver,
        'usb',
        SimpleNamespace(
            core=SimpleNamespace(find=lambda **kw: next(found), USBError=OSError),
            util=SimpleNamespace(
                claim_interface=lambda dev, i: None, dispose_resources=lambda dev: None
            ),
        ),
    )
    monkeypatch.setattr(
        fx2driver,
        'usb1',
        SimpleNamespace(
            USBContext=lambda: _Context(handle),
            TRANSFER_COMPLETED=object(),
            TRANSFER_CANCELLED=object(),
        ),
    )
    return fx2driver._FX2Connection(fx2driver._LibusbTransport())


def test_control_goes_through_the_stream_while_it_runs_and_back_after(monkeypatch):
    events = []
    conn = _connection_on_libusb(monkeypatch, events)
    write = fx2driver.VR_I2C_WRITE

    conn.i2c_write(fx2driver.I2C_LED, [0xFF])
    conn.start_stream()
    conn.i2c_write(fx2driver.I2C_LED, [0xFF])
    conn.stop_stream()
    conn.i2c_write(fx2driver.I2C_LED, [0xFF])

    controls = [e for e in events if e[1] != 'submit']
    assert controls == [
        ('first', write),
        ('iso', fx2driver.VR_START_STREAMING),
        ('iso', write),
        ('iso', fx2driver.VR_STOP_STREAMING),
        ('second', write),
    ]


def test_the_iso_transfers_are_pending_before_the_device_starts_streaming(monkeypatch):
    events = []
    conn = _connection_on_libusb(monkeypatch, events)
    conn.start_stream()
    conn.stop_stream()

    start = events.index(('iso', fx2driver.VR_START_STREAMING))
    submits = [i for i, e in enumerate(events) if e == ('iso', 'submit')]
    assert len(submits) == fx2driver.ISO_NUM_TRANSFERS
    assert max(submits) < start


# ---------------------------------------------------------------------------
# The drivers reach no USB library and no connection private state
# ---------------------------------------------------------------------------


def test_a_connection_that_cannot_connect_closes_its_transport_and_raises_its_own_error():
    import pytest

    class NoDevice:
        closed = False

        def find(self, pid):
            raise RuntimeError('no FX2 on the bus')

        def close(self):
            NoDevice.closed = True

    with pytest.raises(RuntimeError, match='no FX2 on the bus'):
        fx2driver._FX2Connection(NoDevice())
    assert NoDevice.closed


def test_a_windows_start_that_fails_leaves_its_reader_where_the_stop_stops_it(monkeypatch):
    import sys
    import types

    class Device:
        def control_transfer(self, request_type, request, value, index, data=b''):
            if request == fx2driver.VR_START_STREAMING:
                raise RuntimeError('ControlTransfer OUT failed: 31')
            return len(data)

    readers = []

    class Reader:
        def __init__(self, vid, pid, **kwargs):
            self.device = Device()
            self.running = False
            readers.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    monkeypatch.setitem(
        sys.modules, 'drivers.winusb_iso', types.SimpleNamespace(WinUsbIsoReader=Reader)
    )
    monkeypatch.setattr(
        fx2driver,
        'usb',
        types.SimpleNamespace(
            core=types.SimpleNamespace(find=lambda **kw: None),
            util=types.SimpleNamespace(dispose_resources=lambda dev: None),
        ),
        raising=False,
    )
    transport = fx2driver._WinUsbTransport()
    try:
        transport.start_stream(fx2driver._ByteStream(), on_gone=lambda: None)
    except RuntimeError:
        pass
    transport.stop_stream()
    assert len(readers) == 1
    assert readers[0].running is False


_USB_NAMES = {'usb', 'usb1', 'WinUsbIsoReader', 'winusb_iso'}


def _driver_classes():
    tree = ast.parse(DRIVER.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in ('FX2Camera', 'FX2LEDController')
    }


def test_the_guard_sees_both_driver_classes():
    assert set(_driver_classes()) == {'FX2Camera', 'FX2LEDController'}


def test_neither_fx2_driver_names_a_usb_library():
    found = []
    for name, cls in _driver_classes().items():
        for node in ast.walk(cls):
            if isinstance(node, ast.Name) and node.id in _USB_NAMES:
                found.append(f'{name}:{node.lineno} {node.id}')
            elif isinstance(node, ast.ImportFrom | ast.Import):
                found.append(f'{name}:{node.lineno} import')
    assert found == []


def test_neither_fx2_driver_reaches_the_connections_private_state():
    found = []
    for name, cls in _driver_classes().items():
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith('_')
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == '_fx2'
            ):
                found.append(f'{name}:{node.lineno} _fx2.{node.attr}')
    assert found == []
