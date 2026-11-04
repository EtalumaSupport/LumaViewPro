# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Typed stand-in for pyusb and python-libusb1 in unit tests.

The FX2 driver tells a USB timeout from a device that has left the bus by the
libraries' exception classes, so those classes must be real: a blanket
MagicMock attribute in an ``except`` clause or an ``isinstance`` raises
``TypeError``. The classes here mirror the libraries' hierarchies (checked
against pyusb 1.3.1 and libusb1 3.4.0): pyusb's timeout is a subclass of its
``USBError``, an ``OSError``; python-libusb1's timeout and its two removal
errors are siblings under its ``USBError``. Everything else stays a
MagicMock, so untouched call sites behave as they did under the blanket mock.

``usb.core`` is the same object as ``usb``'s ``core`` attribute: with the
module pre-installed in ``sys.modules``, ``import usb.core`` binds ``usb``
and the driver reaches ``usb.core`` through it.

Install via tests/conftest.py install_mock_deps; never import the real
libraries here -- the stub must work on machines without them.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class USBError(OSError):
    """pyusb's ``usb.core.USBError``."""


class USBTimeoutError(USBError):
    """pyusb's ``usb.core.USBTimeoutError``."""


class Libusb1Error(Exception):
    """python-libusb1's ``usb1.USBError``."""


class USBErrorTimeout(Libusb1Error):  # noqa: N818 -- mirrors the library's class name
    pass


class USBErrorNoDevice(Libusb1Error):  # noqa: N818 -- mirrors the library's class name
    pass


class USBErrorNotFound(Libusb1Error):  # noqa: N818 -- mirrors the library's class name
    pass


core = MagicMock(name='usb.core', USBError=USBError, USBTimeoutError=USBTimeoutError)
util = MagicMock(name='usb.util')
usb = MagicMock(name='usb', core=core, util=util)
usb1 = MagicMock(
    name='usb1',
    USBError=Libusb1Error,
    USBErrorTimeout=USBErrorTimeout,
    USBErrorNoDevice=USBErrorNoDevice,
    USBErrorNotFound=USBErrorNotFound,
)
