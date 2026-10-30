# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The WinUSB control-transfer failure contract -- WINDOWS ONLY.

`drivers/winusb_iso.py` loads its DLLs with ``ctypes.WinDLL`` at module scope, so
it cannot be imported on macOS or Linux at all. This whole module is skipped
there, which means the defect these tests cover is NOT guarded by any CI run
on a dev machine -- only by running this file on a Windows box.

The defect: ``WinUsbDevice.control_transfer`` discarded the BOOL that
``WinUsb_ControlTransfer`` returns. A failed vendor request was then
indistinguishable from a successful one -- the OUT branch returned None either
way, and the IN branch returned b'' because the transferred count stays 0 on
failure. Downstream, ``_led_write``'s short-write detector skips its check when
the count is None, so that None silently disabled a safety check added
specifically to catch silent short-writes.

Run on the bench:  python3 -m pytest tests/test_winusb_iso_transfer_contract.py -q
"""

from __future__ import annotations

import ctypes
import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != 'win32',
    reason='winusb_iso loads its DLLs with ctypes.WinDLL, which exists only on Windows',
)


def _device():
    """A WinUsbDevice with no hardware behind it.

    __new__ rather than the constructor: the real one enumerates USB, opens a
    file handle and calls WinUsb_Initialize. Only the interface handle the
    transfer passes through is needed.
    """
    from drivers import winusb_iso

    dev = object.__new__(winusb_iso.WinUsbDevice)
    dev._iface_handle = ctypes.c_void_p(1)
    return dev


def _fake_winusb(ok, wrote=0):
    """Stand in for the winusb DLL, reporting `ok` and writing `wrote` into
    the caller's transferred-count out-parameter (passed by byref)."""

    class _Fake:
        def WinUsb_ControlTransfer(self, iface, pkt, buf, length, transferred_ref, overlapped):
            transferred_ref._obj.value = wrote
            return ok

    return _Fake()


def test_a_failed_out_transfer_raises_and_names_the_error_code():
    from drivers import winusb_iso

    with (
        patch.object(winusb_iso, 'winusb', _fake_winusb(ok=0)),
        patch.object(winusb_iso.ctypes, 'get_last_error', lambda: 31),
        pytest.raises(OSError, match='31'),
    ):
        _device().control_transfer(0x40, 0xB3, 0, 0x42, data=b'\x01')


def test_a_failed_in_transfer_raises_instead_of_returning_empty_bytes():
    """Pre-fix this returned b'', which a caller cannot tell from a real
    zero-length read."""
    from drivers import winusb_iso

    with (
        patch.object(winusb_iso, 'winusb', _fake_winusb(ok=0)),
        patch.object(winusb_iso.ctypes, 'get_last_error', lambda: 31),
        pytest.raises(OSError, match='31'),
    ):
        _device().control_transfer(0xC0, 0xB2, 0, 0x42, length=2)


def test_a_successful_out_transfer_returns_the_byte_count():
    """The half that re-arms the short-write detector: pre-fix this returned
    None, and `_led_write` skips its check when the count is None."""
    from drivers import winusb_iso

    with patch.object(winusb_iso, 'winusb', _fake_winusb(ok=1, wrote=1)):
        result = _device().control_transfer(0x40, 0xB3, 0, 0x42, data=b'\x01')

    assert result == 1


def test_a_short_out_transfer_reports_the_short_count_not_the_requested_one():
    """A transfer that succeeds but writes fewer bytes than asked must report
    what actually went out -- that difference is the whole signal."""
    from drivers import winusb_iso

    with patch.object(winusb_iso, 'winusb', _fake_winusb(ok=1, wrote=0)):
        result = _device().control_transfer(0x40, 0xB3, 0, 0x42, data=b'\x01')

    assert result == 0


def test_a_successful_in_transfer_returns_the_buffer_slice():
    from drivers import winusb_iso

    with patch.object(winusb_iso, 'winusb', _fake_winusb(ok=1, wrote=2)):
        result = _device().control_transfer(0xC0, 0xB2, 0, 0x42, length=2)

    assert isinstance(result, bytes)
    assert len(result) == 2


@pytest.mark.parametrize('request_type, kwargs', [(0x40, {'data': b'\x01'}), (0xC0, {'length': 2})])
def test_a_timed_out_transfer_raises_a_timeout_naming_the_code(request_type, kwargs):
    """ERROR_SEM_TIMEOUT is the device not answering, which the FX2 driver
    tells from every other failure by the type."""
    from drivers import winusb_iso

    with (
        patch.object(winusb_iso, 'winusb', _fake_winusb(ok=0)),
        patch.object(winusb_iso.ctypes, 'get_last_error', lambda: winusb_iso.ERROR_SEM_TIMEOUT),
        pytest.raises(TimeoutError, match='121'),
    ):
        _device().control_transfer(request_type, 0xB3, 0, 0x42, **kwargs)


def test_a_reader_whose_pipe_query_fails_closes_the_device_it_opened():
    """The reader is held by no one until its start returns, so a start that
    raised after opening the device left the WinUSB handle open with nothing
    to close it, and the next start could not open the device."""
    from drivers import winusb_iso

    closed = []

    class Device:
        def __init__(self, vid, pid):
            pass

        def set_alt_interface(self, setting):
            pass

        def query_pipe(self, alt, index):
            raise RuntimeError('QueryPipe failed: 31')

        def close(self):
            closed.append(True)

    reader = winusb_iso.WinUsbIsoReader(0x04B4, 0xEA17, on_data=None, on_error=None)
    with (
        patch.object(winusb_iso, 'WinUsbDevice', Device),
        pytest.raises(RuntimeError, match='QueryPipe failed'),
    ):
        reader.start()

    assert closed == [True]
    assert reader.device is None


def _slot_device(closed):
    """A WinUSB device class whose pipe answers, for a reader's start; each
    close is appended to ``closed``."""

    class Device:
        def __init__(self, vid, pid):
            self.handle = ctypes.c_void_p(1)

        def set_alt_interface(self, setting):
            pass

        def query_pipe(self, alt, index):
            return type('Pipe', (), {'MaximumPacketSize': 1024, 'PipeId': 0x82})()

        def close(self):
            closed.append(True)

    return Device


def test_a_reader_whose_buffer_registration_fails_fails_its_start():
    """The slots were built on the reader's thread, so a buffer WinUSB
    refused to register raised there, after the start had returned and the
    stream was reported started: the camera then saw only silence. Built
    in the start, the refusal is the start's, and what was opened is
    released."""
    from drivers import winusb_iso

    closed_devices, registered, closed_events = [], [], []

    class Kernel32:
        def CreateEventW(self, *args):
            return 7

        def CloseHandle(self, handle):
            closed_events.append(handle)

    class WinUsb:
        def WinUsb_RegisterIsochBuffer(self, *args):
            registered.append(True)
            return len(registered) < 3  # the third buffer is refused

        def WinUsb_UnregisterIsochBuffer(self, handle):
            pass

    reader = winusb_iso.WinUsbIsoReader(0x04B4, 0xEA17, on_data=None, on_error=None, num_slots=4)
    with (
        patch.object(winusb_iso, 'WinUsbDevice', _slot_device(closed_devices)),
        patch.object(winusb_iso, 'kernel32', Kernel32()),
        patch.object(winusb_iso, 'winusb', WinUsb()),
        patch.object(winusb_iso.ctypes, 'get_last_error', lambda: 8),
        pytest.raises(winusb_iso.CameraUsbOpenFailedError, match='RegisterIsochBuffer failed: 8'),
    ):
        reader.start()

    assert closed_devices == [True]
    assert closed_events == [7, 7, 7], 'each slot built, and the refused one, closes its event'
    assert reader.device is None


def test_a_slot_whose_event_cannot_be_made_is_refused():
    """A NULL event makes every overlapped read look complete at once."""
    from drivers import winusb_iso

    class Kernel32:
        def CreateEventW(self, *args):
            return None

    with (
        patch.object(winusb_iso, 'kernel32', Kernel32()),
        patch.object(winusb_iso.ctypes, 'get_last_error', lambda: 1450),
        pytest.raises(winusb_iso.CameraUsbOpenFailedError, match='CreateEventW failed: 1450'),
    ):
        winusb_iso.IsoSlot(ctypes.c_void_p(1), 0x82, 1024, 8)
