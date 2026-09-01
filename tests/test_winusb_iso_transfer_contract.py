# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The WinUSB control-transfer failure contract -- WINDOWS ONLY.

`drivers/winusb_iso.py` does ``from ctypes import windll`` at module scope, so
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
    reason='winusb_iso imports ctypes.windll, which exists only on Windows',
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
        patch.object(winusb_iso.kernel32, 'GetLastError', lambda: 31),
        pytest.raises(RuntimeError, match='31'),
    ):
        _device().control_transfer(0x40, 0xB3, 0, 0x42, data=b'\x01')


def test_a_failed_in_transfer_raises_instead_of_returning_empty_bytes():
    """Pre-fix this returned b'', which a caller cannot tell from a real
    zero-length read."""
    from drivers import winusb_iso

    with (
        patch.object(winusb_iso, 'winusb', _fake_winusb(ok=0)),
        patch.object(winusb_iso.kernel32, 'GetLastError', lambda: 31),
        pytest.raises(RuntimeError, match='31'),
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
