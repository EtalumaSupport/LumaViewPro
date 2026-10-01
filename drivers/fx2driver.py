# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""FX2 driver for Lumascope Classic (LS620 / LS560 / LS720).

Cypress FX2 USB2 chip + Aptina MT9P031 image sensor, with LED control via
I2C through the same USB device. One physical device, two LumaViewPro
driver roles (camera + LED board).

Architecture
------------

Three objects live in this file, over one seam:

1. ``_FX2Connection`` -- module-level singleton that owns the device:
   discovery, firmware upload, control transfers, I2C, sensor register
   writes, and the stream (``start_stream`` / ``stop_stream``, the bytes
   arriving in its ``stream``). Every USB library call it makes goes
   through its transport, chosen once per host by ``_platform_transport``:
   ``_LibusbTransport`` (macOS / Linux) or ``_WinUsbTransport`` (Windows).
   Constructed lazily the first time any driver calls
   ``_FX2Connection.get()``. Raises on any failure; the registry treats a
   raise as "this driver isn't available" and falls through to the next
   candidate. Private to the FX2 drivers and the simulated FX2
   (``drivers/simulated_fx2.py``), which builds one on its own transport.

2. ``FX2Camera`` -- registered as ``@camera_registry.register('fx2', ...)``.
   Implements the Camera ABC. Pulls ``_FX2Connection.get()`` in ``__init__``
   so the camera and LED end up sharing the same USB handle, unless it is
   handed a connection: a simulated FX2 (``drivers/simulated_fx2.py``) hands
   both drivers the connection on its device.

3. ``FX2LEDController`` -- registered as ``@led_registry.register('fx2', ...)``.
   Satisfies LEDBoardProtocol. Thin command translator: no state tracking,
   no ``led_ma`` dict, no state queries. Source of truth for LED state is
   ``IlluminationAPI._led_state``. The class exists only to convert LVP's
   (channel, mA) calls into FX2 I2C byte sequences.

The camera and LED objects both hold a reference to the same connection:
on hardware the ``_FX2Connection._instance`` singleton -- proven viable by
``TestRegistryAccommodatesCompositeHardware`` in tests/test_driver_registry.py
-- and in the simulator the one connection the simulated FX2 hands both.
Neither driver touches a USB library or the connection's private state.

Dependencies
------------

- ``pyusb``  (firmware upload + control transfers)
- ``libusb1`` (isochronous streaming on macOS/Linux -- python-libusb1 binding)
- ``drivers.winusb_iso`` (isochronous streaming on Windows -- ctypes wrapper)
- ``libusb-package`` (the native ``libusb-1.0`` library itself, one
  versioned copy on every platform; never a host copy)

Import-time safety
------------------

All USB / libusb1 imports are wrapped in try/except. When a prerequisite is
missing the module still imports, logs which one, and registers neither
driver, so auto-detect never offers an FX2 that cannot run and non-FX2
scopes on dev machines without pyusb are unaffected.

The driver reads no application settings: the one runtime toggle, the LED
wire trace, arrives as the ``debug_wire`` constructor argument.

References
----------

- Reference port: ``git show 4.0.0-LVCtest:drivers/fx2driver.py``
- MT9P031 datasheet (DS_F, pages 35-36) for gain register layout
- MT9P031 developer guide (DG_A, page 7) for blue-strip workaround
- FX2 Technical Reference Manual for vendor request 0xA0 (VR_ANCHOR_DLD)
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import math
import os
import sys
import threading
import time
import weakref
from typing import Any, NamedTuple, NoReturn
from collections import deque
from collections.abc import Callable
from datetime import datetime

import numpy as np

from lvp_logger import logger

try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    # Fall back to the main logger so every _cam_log call site stays
    # safe -- the dedicated camera log is an enhancement, not a
    # dependency, and dozens of call sites use _cam_log unguarded.
    _cam_log = logger
from drivers.camera import Camera, ImageHandlerBase, no_hardware_auto_mode
from drivers.registry import camera_registry, led_registry

# Wire-level logging for the FX2 (LumaviewClassic LS560/620/720) USB
# control-transfer + I2C path. Without this, every i2c_write to set LED
# brightness, every objective turret move, every sensor register write
# happens with zero serial.log trace -- invisible from the supported
# debug surface. Adds the same `serial.log` line shape SerialBoard uses
# (`{label} {op} -> {result} ({elapsed_ms}ms)`), so a single grep on
# `serial.log` covers RP2040 LED + RP2040 motor + sim + FX2 in one
# place.
_serial_log = logging.getLogger('LVP.serial')

# USB device-descriptor fields worth recording for a found device. These
# are read from the descriptor the backend already holds, so naming them
# costs no bus traffic.
#
# manufacturer / product / serial_number are deliberately ABSENT: pyusb
# fetches those lazily via string-descriptor control transfers, and a
# control transfer that does not return is the exact failure this log
# line exists to help attribute -- so reading them here could hang the
# connect path on the machine we most need a log from.
_USB_DESCRIPTOR_FIELDS = (
    'address',
    'bDeviceClass',
    'bDeviceProtocol',
    'bDeviceSubClass',
    'bMaxPacketSize0',
    'bNumConfigurations',
    'bcdDevice',
    'bcdUSB',
    'bus',
    'idProduct',
    'idVendor',
    'port_number',
    'port_numbers',
    'speed',
)


def describe_usb_device(dev: Any) -> str:
    """Render what libusb can tell us about a found FX2 device.

    Bus, address, port chain and negotiated speed are the fields that
    place the device on a particular host controller -- the attribution
    a support bundle cannot otherwise carry, since nobody can inspect
    the machine afterwards.

    Degradation is per-field: a field a backend does not populate is
    named with its failure reason rather than dropping the line.

    Args:
        dev: A ``usb.core.Device`` returned by ``usb.core.find``.

    Returns:
        str: Space-separated ``Name=value`` pairs, sorted by name so two
            bundles from one machine diff cleanly.
    """
    parts = []
    for name in _USB_DESCRIPTOR_FIELDS:
        try:
            value = getattr(dev, name)
        except Exception as e:
            parts.append(f'{name}=<unreadable: {type(e).__name__}>')
            continue
        if value is None:
            continue
        # IDs and the BCD fields are conventionally read in hex: bcdUSB
        # 0x0200 is "USB 2.0" at a glance where 512 is not, and the USB
        # generation a device negotiated is part of placing it on a host
        # controller.
        if name in ('idVendor', 'idProduct', 'bcdUSB', 'bcdDevice'):
            parts.append(f'{name}=0x{value:04X}')
        else:
            parts.append(f'{name}={value!r}')
    return ' '.join(parts)


# Vendor-request integer -> human-readable name. Populated lazily after
# the VR_* constants below are defined.
_VR_NAMES: dict[int, str] = {}


def _vr_name(req: int) -> str:
    """Return a human-readable name for a vendor request, or hex fallback."""
    return _VR_NAMES.get(req, f'VR_0x{req:02X}')


try:
    import usb.core
    import usb.util

    _HAS_USB = True
except ImportError:
    _HAS_USB = False

try:
    import usb1

    _HAS_USB1 = True
except ImportError:
    _HAS_USB1 = False


def _load_bundled_libusb():
    """Bind pyusb and python-libusb1 to libusb-package's library.

    Returns ``(path, None)`` when both bindings run on the bundled file, or
    ``(None, reason)`` when they cannot. Which libusb a host happens to have
    (Homebrew's, a distro's, a stray DLL) must never decide what the driver
    runs on, so there is no fallback: ``libusb_package.get_libusb1_backend``
    would fall back to the system search when its own file is missing, so
    the path is taken and checked here instead.

    Ordering invariant: pyusb keeps one backend per process, bound by the
    first ``get_backend`` call, and python-libusb1 loads its library once
    (``loadLibrary`` returns False, it does not raise, when another is
    already loaded). This must run before anything else in the process
    touches either binding; the checks below catch it if something did.
    """
    try:
        import libusb_package
    except ImportError:
        return None, (
            'libusb-package is not installed (pip install -r requirements.txt; '
            'it has no wheel for 32-bit ARM Linux or Windows on ARM)'
        )
    lib_path = libusb_package.get_library_path()
    if lib_path is None:
        return None, 'libusb-package holds no library file in this install'
    path = str(lib_path)
    try:
        import usb.backend.libusb1

        backend = usb.backend.libusb1.get_backend(find_library=lambda _name: path)
    except Exception as ex:
        return None, f'the bundled libusb at {path} did not load: {ex}'
    if backend is None:
        return None, f'the bundled libusb at {path} did not load'
    if backend.lib._name != path:
        return None, f'another libusb was loaded before the bundled one: {backend.lib._name}'
    # Its own handle to the same file: each binding declares argument types
    # on the functions of the handle it holds, so one shared handle leaves
    # pyusb calling with python-libusb1's declarations and every descriptor
    # read fails. The OS loader still maps the file once.
    if _HAS_USB1:
        try:
            usb1_handle = ctypes.CDLL(path)
        except OSError as ex:
            return None, f'the bundled libusb at {path} did not load for python-libusb1: {ex}'
        if not usb1.loadLibrary(usb1_handle):
            return None, 'python-libusb1 had already loaded another libusb'
    return path, None


# Resolved at import so a host without a usable libusb is classified as
# "FX2 not applicable to this install" here, instead of the first
# usb.core.find() raising NoBackendError mid-connect on every startup.
_LIBUSB_PATH = None
_LIBUSB_REFUSAL = 'pyusb is not installed'
if _HAS_USB:
    _LIBUSB_PATH, _LIBUSB_REFUSAL = _load_bundled_libusb()
_HAS_USB_BACKEND = _LIBUSB_PATH is not None


# FX2 (LumaviewClassic) drivers require pyusb plus a loadable libusb-1.0
# native backend on every platform, plus libusb1 on macOS / Linux for ISO
# streaming. On systems missing those, FX2 hardware fundamentally cannot
# be reached -- registering the drivers anyway causes the registry to
# attempt instantiation on every auto-detect run, raise NoBackendError /
# ImportError, and dump a confusing traceback on systems that simply
# don't have LVC hardware. Skip-with-INFO-log is the right
# classification: this driver class isn't applicable to this install.
_FX2_AVAILABLE = _HAS_USB and _HAS_USB_BACKEND and (sys.platform == 'win32' or _HAS_USB1)
if not _FX2_AVAILABLE:
    if not _HAS_USB:
        logger.info(
            '[FX2 Driver] pyusb not installed -- FX2 (LumaviewClassic) '
            'drivers will not be registered. Install pyusb to enable '
            'LVC hardware support: pip install pyusb'
        )
    elif not _HAS_USB_BACKEND:
        logger.info(
            f'[FX2 Driver] bundled libusb not in use: {_LIBUSB_REFUSAL} -- '
            'FX2 (LumaviewClassic) drivers will not be registered.'
        )
    elif not _HAS_USB1:
        logger.info(
            '[FX2 Driver] libusb1 not installed -- FX2 (LumaviewClassic) '
            'drivers will not be registered on macOS/Linux. Install '
            'libusb1 (pip install -r requirements.txt) to enable LVC '
            'hardware support.'
        )
if _HAS_USB_BACKEND:
    logger.info(
        f'[FX2 Driver] libusb {usb1.getVersion() if _HAS_USB1 else "(version unread)"} '
        f'loaded from {_LIBUSB_PATH}'
    )


def fx2_readiness() -> dict[str, bool | None]:
    """Each term of the availability gate, as this process found it at import.

    Keys name what an installer installs: ``pyusb``, ``libusb-package``
    (the native library both bindings run on) and ``libusb1`` (the binding that streams
    frames off Windows). ``libusb1`` is ``None`` on Windows, where the gate
    does not need it. ``scripts/install_mac.sh`` reports these rather than
    probing on its own, so what it prints is what the driver will do.
    """
    return {
        'pyusb': _HAS_USB,
        'libusb-package': _HAS_USB_BACKEND,
        'libusb1': None if sys.platform == 'win32' else _HAS_USB1,
    }


def fx2_readiness_line() -> str:
    """One line for an installer: ready or not, and each gate term's state."""
    terms = ', '.join(
        f'{name} {"not needed" if present is None else "present" if present else "missing"}'
        for name, present in fx2_readiness().items()
    )
    verdict = 'ready' if _FX2_AVAILABLE else 'NOT ready'
    line = f'FX2 (LS560/LS620/LS720) support: {verdict} -- {terms}'
    if _HAS_USB and not _HAS_USB_BACKEND:
        line += f' ({_LIBUSB_REFUSAL})'
    return line


def _register_if_fx2_available(registry, name, **kwargs):
    """Register the decorated class only if FX2 prerequisites are met.

    No-op (returns identity decorator) when pyusb/libusb1 unavailable so
    the registry never sees FX2 as a candidate on non-LVC installs.
    """
    if _FX2_AVAILABLE:
        return registry.register(name, **kwargs)
    return lambda cls: cls


# ---------------------------------------------------------------------------
# USB constants
# ---------------------------------------------------------------------------

VID = 0x04B4
PID_BOOT = 0x8613  # FX2 bootloader (before firmware upload)
PID_APP = 0xEA17  # Running firmware

# Vendor request codes -- FX2 firmware vendor command handler
VR_ANCHOR_DLD = 0xA0  # Cypress standard: firmware upload
VR_I2C_WRITE = 0xB3
VR_I2C_MT9P031_READ = 0xB4  # Async MT9P031 register read (5s timeout OK)
VR_INIT_GPIF = 0xB9
# VR_IMAGE_SENSOR_CLK_MANAGED_WRITE (0xBA) was defined here as
# VR_SENSOR_CLK_WRITE and used by sensor_reg_write. It switches IFCLK
# to internal, does the I2C write, then switches back -- which disrupts
# ISO streaming because the GPIF pixel clock depends on IFCLK. Removed
# 2026-04-15 after finding it was the root cause of visible image
# corruption on every gain/exposure slider drag. LVC defines the same
# constant in `I2C_Control.cs:52` but never calls it; LVC's production
# path uses VR_I2C_WRITE (0xB3) for sensor writes via
# `AptinaMT9P031_Control.WriteWord16 -> I2C_Control.Write`. We now
# match LVC.
VR_SET_IFCLK_SRC = 0xBB
VR_CODE_VERSION = 0xBC
VR_START_STREAMING = 0xBD
VR_STOP_STREAMING = 0xBE

# Populate _VR_NAMES once constants are defined (used by serial.log
# emission below for human-readable trace lines).
_VR_NAMES.update(
    {
        VR_ANCHOR_DLD: 'VR_ANCHOR_DLD',
        VR_I2C_WRITE: 'VR_I2C_WRITE',
        VR_I2C_MT9P031_READ: 'VR_I2C_MT9P031_READ',
        VR_INIT_GPIF: 'VR_INIT_GPIF',
        VR_SET_IFCLK_SRC: 'VR_SET_IFCLK_SRC',
        VR_CODE_VERSION: 'VR_CODE_VERSION',
        VR_START_STREAMING: 'VR_START_STREAMING',
        VR_STOP_STREAMING: 'VR_STOP_STREAMING',
    }
)

# Vendor requests that the i2c_write wrapper routes through
# control_transfer_*. Logged at the i2c_write layer (with addr/data); the
# control_transfer_* layer skips them to avoid double-emission.
_I2C_VR_REQUESTS = frozenset({VR_I2C_WRITE})

# I2C addresses
I2C_SENSOR = 0x5D  # MT9P031 image sensor
I2C_LED = 0x2A  # Peripheral controller (LEDs)


# ---------------------------------------------------------------------------
# Image sensor constants
# ---------------------------------------------------------------------------

IMG_WIDTH = 1900
IMG_HEIGHT = 1900
FRAME_BYTES = IMG_WIDTH * IMG_HEIGHT  # raw pixel count (8-bit mono)
FRAME_DELIM = b'\x01\xfe\x00\xff'  # injected between frames by GpifWaveform_Isr


class FrameLayout(NamedTuple):
    """Where a frame's pixels sit in the bytes streamed between two delimiters.

    After ``FRAME_DELIM`` comes a row the parser skips (``skip`` bytes), then
    ``h`` rows of ``stride`` bytes -- ``w`` pixels and the sync byte the GPIF
    puts between rows -- then one more row of padding. A whole frame is
    ``frame_bytes`` long; anything else between two delimiters is damaged.
    """

    stride: int
    skip: int
    needed: int
    frame_bytes: int


def frame_layout(w: int, h: int) -> FrameLayout:
    """The wire layout of a ``w`` x ``h`` window: what the parser reads and a device sends."""
    stride = w + 1
    skip = stride + 1
    needed = skip + h * stride
    return FrameLayout(stride, skip, needed, needed + stride)


# MT9P031 register addresses
REG_ROW_START = 0x01
REG_COL_START = 0x02
REG_ROW_SIZE = 0x03
REG_COL_SIZE = 0x04
REG_EXPOSURE = 0x09
REG_PLL_CTRL = 0x10
REG_PLL_CFG1 = 0x11
REG_READ_MODE2 = 0x20
REG_GLOBAL_GAIN = 0x35
REG_ROW_BLACK = 0x49
REG_BLC = 0x62

REG_PLL_CFG2 = 0x12

# The largest value Shutter_Width_Lower (R0x09) holds. It is not the sensor's
# limit: Shutter_Width_Upper (R0x08) extends the shutter width past it, and the
# driver does not write R0x08.
MAX_EXPOSURE_ROWS = 65535


# ---------------------------------------------------------------------------
# The sensor's timing, as the MT9P031 data sheet states it
# ---------------------------------------------------------------------------
# Every time the driver records or publishes is computed here from the
# registers it writes, so an exposure means the same integration at every
# window and the simulator's timing is the sensor's, not a copy of it.

# The sensor's EXTCLK: 24 MHz from the main board (the Series 600 image sensor
# interface spec, "Clock signal (24 MHz) from main to sensor"). The LS620
# bench agrees: frame periods at three widths and three clock settings land
# within 0.4% of the data sheet at 24 MHz, and 7.7% off at 12 MHz.
EXTCLK_HZ = 24_000_000

# The PLL fields connect() writes. The data sheet's divisors are one more than
# the register fields: f_PIXCLK = f_EXTCLK x M / ((N_divider + 1) x
# (P1_divider + 1)) = 24 x 27 / (2 x 14) = 23.14 MHz, with the VCO at 324 MHz
# and f_EXTCLK / N at 12 MHz, both inside their ranges. P1_divider is odd:
# an even value gives a system clock that is not 50:50.
_PLL_M = 27
_PLL_N_DIVIDER = 1
_PLL_P1_DIVIDER = 13

# Horizontal timing at Row_Bin 0 and Column_Bin 0, with Row_BLC on (the
# driver's Read Mode 2): HBMIN = 346 x (Row_Bin + 1) + 64 + WDC / 2, WDC = 80
# dark columns; a row is never shorter than 41 + 346 x (Row_Bin + 1) + 99
# clocks per half. Horizontal_Blank (R0x05) is never written, so its default
# 0 gives HB = 1, under HBMIN.
_HBMIN = 450
_HALF_ROW_MIN = 486
_HB = 1
# Vertical_Blank (R0x06) is never written: its default 25 gives VB = 26 rows.
_VB = 26
# Shutter overhead SO = 208 x (Row_Bin + 1) + 98 + min(SD, SDmax) - 94 = 213
# with Shutter_Delay (R0x0C) at its default 0 (SD = 1); it costs 2 x SO
# pixel clocks.
_SHUTTER_OVERHEAD_CLOCKS = 2 * 213


def pixel_clock_hz(m: int, n_divider: int, p1_divider: int) -> float:
    """f_PIXCLK for the PLL register fields, as the data sheet's PLL section gives it."""
    return EXTCLK_HZ * m / ((n_divider + 1) * (p1_divider + 1))


_PIXEL_CLOCK_HZ = pixel_clock_hz(_PLL_M, _PLL_N_DIVIDER, _PLL_P1_DIVIDER)


def _output_size(size_register: int) -> int:
    """The pixels the sensor outputs for a Column_Size or Row_Size (no skip): W or H."""
    return 2 * -(-(size_register + 1) // 2)


def row_time_s(column_size: int) -> float:
    """tROW for a Column_Size: 2 x tPIXCLK x max(W/2 + max(HB, HBMIN), 486)."""
    half_row = max(_output_size(column_size) // 2 + max(_HB, _HBMIN), _HALF_ROW_MIN)
    return 2 * half_row / _PIXEL_CLOCK_HZ


def exposure_s(shutter_width: int, column_size: int) -> float:
    """tEXP = SW x tROW - SO x 2 x tPIXCLK, the integration a shutter width gives."""
    return (
        max(1, shutter_width) * row_time_s(column_size) - _SHUTTER_OVERHEAD_CLOCKS / _PIXEL_CLOCK_HZ
    )


def shutter_width_for(exposure_seconds: float, column_size: int) -> int:
    """The Shutter_Width_Lower whose integration is nearest ``exposure_seconds``."""
    rows = round(
        (exposure_seconds + _SHUTTER_OVERHEAD_CLOCKS / _PIXEL_CLOCK_HZ) / row_time_s(column_size)
    )
    return max(1, min(MAX_EXPOSURE_ROWS, rows))


def frame_time_s(column_size: int, row_size: int, shutter_width: int) -> float:
    """tFRAME = (H + max(VB, VBMIN)) x tROW, VBMIN = max(8, SW - H) + 1.

    A shutter width past H + 25 rows stretches the frame: the sensor adds
    blanking rows until the integration fits.
    """
    h = _output_size(row_size)
    vbmin = max(8, shutter_width - h) + 1
    return (h + max(_VB, vbmin)) * row_time_s(column_size)


# ---------------------------------------------------------------------------
# LED channel mapping
# ---------------------------------------------------------------------------
# LVP convention uses integer channels. The FX2 peripheral controller at
# I2C 0x2A takes ASCII bytes A-D. This mapping lives INSIDE the driver by
# design (see project memory `project_lvc_product_line.md`): the ASCII
# byte format must never leak above the driver layer.

_COLOR_TO_CH = {'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3}
_CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}
_CH_TO_I2C = {
    0: 0x43,  # Blue  -> 'C'
    1: 0x42,  # Green -> 'B'
    2: 0x41,  # Red   -> 'A'
    3: 0x44,  # BF    -> 'D'
}


# ---------------------------------------------------------------------------
# ISO streaming parameters (matches C# ReadISOStream_WinUsb reference config)
# ---------------------------------------------------------------------------

ISO_ALT_INTERFACE = 3  # Alt interface 3 = ISO IN, 3x1024/microframe
ISO_NUM_TRANSFERS = 16  # Pending transfers in flight
ISO_NUM_PACKETS = 256  # ISO packets per transfer (C# reference uses 256)
ISO_MAX_PACKET_SIZE = 3072  # 3 x 1024 bytes per microframe


# ---------------------------------------------------------------------------
# Intel HEX parser
# ---------------------------------------------------------------------------


def parse_intel_hex(hex_path: str) -> tuple[bytes, int]:
    """Parse an Intel HEX file into a flat byte array.

    The FX2 8051 program space is 16 KB (0x4000). Unwritten locations stay
    0xFF. Only record type 0 (data) and type 1 (EOF) are handled; other
    record types are skipped.

    Returns:
        (data, end_addr): data is a 16 KB bytes object, end_addr is the
        highest address that was actually written (used to size the
        firmware upload -- everything past end_addr stays 0xFF).
    """
    buf = bytearray(0x4000)
    for i in range(len(buf)):
        buf[i] = 0xFF
    end_addr = 0

    with open(hex_path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] != ':':
                continue
            count = int(line[1:3], 16)
            addr = int(line[3:7], 16)
            record_type = int(line[7:9], 16)
            if record_type == 1:  # EOF
                break
            if record_type != 0:  # only handle data records
                continue
            for i in range(count):
                byte_val = int(line[9 + i * 2 : 11 + i * 2], 16)
                buf[addr] = byte_val
                addr += 1
            if addr > end_addr:
                end_addr = addr

    return bytes(buf), end_addr


# ---------------------------------------------------------------------------
# Gain conversion -- MT9P031 datasheet (DS_F, pages 35-36) and register
# reference (RR_A pages 16-17).
# ---------------------------------------------------------------------------
# Global Gain register (0x35) bit fields:
#   Bits [14:8] = Digital_Gain     -- legal values [0, 120] per RR_A
#   Bit  [6]    = Analog_Multiplier (0 or 1)
#   Bits [5:0]  = Analog_Gain      -- legal values [8, 63] per RR_A
#
# Analog gain:  AG = (1 + Analog_Multiplier) x (Analog_Gain / 8)
# Digital gain: DG = 1 + (Digital_Gain / 8)
# Total gain:   AG x DG
#
# Strategy (datasheet recommended):
#   <= 4x:  analog only (multiplier=0) -- best noise performance
#   <= 8x:  analog with multiplier=1
#   > 8x:  max analog (8x) + digital for the rest
#
# Range: 1x (0 dB) to 128x (42.1 dB). The LumaviewClassic LVC driver
# reference originally had `min(127, ...)` on the digital clamp and a
# comment claiming ~135x max -- that was outside the documented legal
# range per RR_A. The corrected legal max is 120 / 128x. See the
# docstring on `_gain_db_to_register` for the conversion derivation.


def _gain_db_to_register(db: float) -> int:
    """Convert gain in dB to MT9P031 global gain register value."""
    mult = 10 ** (float(db) / 20.0)
    mult = max(1.0, mult)

    if mult <= 4.0:
        # Analog only, no multiplier
        analog_val = min(63, max(8, round(mult * 8)))
        analog_mult = 0
        digital_val = 0
    elif mult <= 8.0:
        # Analog with multiplier
        analog_val = min(63, max(8, round(mult / 2 * 8)))
        analog_mult = 1
        digital_val = 0
    else:
        # Max analog (8x) + digital
        analog_val = 32  # AG = 2 x 32/8 = 8.0
        analog_mult = 1
        dg_needed = mult / 8.0
        digital_val = min(120, max(0, round((dg_needed - 1) * 8)))

    return (digital_val << 8) | (analog_mult << 6) | analog_val


def _register_to_gain_db(reg: int) -> tuple[float, float]:
    """Convert MT9P031 global gain register value to (linear_multiplier, dB)."""
    digital_val = (reg >> 8) & 0x7F
    analog_mult = (reg >> 6) & 1
    analog_val = reg & 0x3F
    ag = (1 + analog_mult) * (analog_val / 8)
    dg = 1 + digital_val / 8
    total = ag * dg
    db = 20 * math.log10(total) if total > 0 else 0.0
    return total, db


# ---------------------------------------------------------------------------
# StreamStats -- frame rate / throughput diagnostics
# ---------------------------------------------------------------------------


class StreamStats:
    """Accumulates streaming diagnostics. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Clear every counter and the deque histories. Thread-safe."""
        with self._lock:
            self._frame_times: deque = deque(maxlen=120)
            self._partial_count = 0
            self._partial_sizes: deque = deque(maxlen=32)
            self._shifted_count = 0
            self._shifted_sizes: deque = deque(maxlen=32)
            self._good_count = 0
            self._total_bytes = 0
            self._usb_errors = 0
            self._start_time = time.monotonic()
            self._delimiters_seen = 0

    def record_good_frame(self) -> None:
        """Record a successfully-parsed full frame. Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._frame_times.append(now)
            self._good_count += 1
            self._delimiters_seen += 1

    def record_partial_frame(self, size: int) -> None:
        """Frame between two delimiters was undersized (bytes dropped before
        next delimiter). Discarded by the grab loop.

        Args:
            size: Observed inter-delimiter buffer size in bytes.
        """
        with self._lock:
            self._partial_count += 1
            self._partial_sizes.append(size)
            self._delimiters_seen += 1

    def record_shifted_frame(self, size: int) -> None:
        """Frame between two delimiters was the wrong size -- either oversized
        (likely a missed delimiter caused two frames to be concatenated, or a
        false-positive delimiter elsewhere in pixel data inflated the buffer)
        OR sized between `needed` and `expected` (off-by-some-rows, also wrong).
        Distinct from `partial` because the failure mechanism is different: a
        partial frame is lost bytes BEFORE the next delimiter; a shifted frame
        is wrong-but-still-bigger-than-minimum, indicating the parser found
        bytes from outside the intended frame. Both are discarded.

        Args:
            size: Observed inter-delimiter buffer size in bytes.
        """
        with self._lock:
            self._shifted_count += 1
            self._shifted_sizes.append(size)
            self._delimiters_seen += 1

    def rejected(self) -> tuple[int, int]:
        """Return ``(shifted, partial)``: the frames discarded so far. Thread-safe."""
        with self._lock:
            return self._shifted_count, self._partial_count

    def record_bytes(self, n: int) -> None:
        """Add ``n`` to the total-bytes counter. Thread-safe.

        Args:
            n: Number of bytes received.
        """
        with self._lock:
            self._total_bytes += n

    def record_usb_error(self) -> None:
        """Increment the USB error counter. Thread-safe."""
        with self._lock:
            self._usb_errors += 1

    def get_fps(self) -> tuple[float, float]:
        """Return ``(current_fps, avg_fps)``. Current = last 2 seconds.

        Returns:
            tuple[float, float]: ``(current_fps, average_fps)``.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._start_time
            avg_fps = self._good_count / elapsed if elapsed > 0 else 0.0
            cutoff = now - 2.0
            recent = sum(1 for t in self._frame_times if t > cutoff)
            cur_fps = recent / 2.0
            return cur_fps, avg_fps

    def summary(self) -> dict:
        """Return a snapshot dict of every counter + derived rates.

        Returns:
            dict: Keys include ``elapsed_s``, ``good_frames``,
                ``partial_frames``, ``shifted_frames``, ``total_MB``,
                ``throughput_MBps``, ``fps_current``, ``fps_average``,
                ``usb_errors``.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._start_time
            recent_partials = list(self._partial_sizes)
            recent_shifted = list(self._shifted_sizes)
        cur_fps, avg_fps = self.get_fps()
        return {
            'elapsed_s': round(elapsed, 1),
            'good_frames': self._good_count,
            'partial_frames': self._partial_count,
            'partial_sizes': recent_partials,
            'shifted_frames': self._shifted_count,
            'shifted_sizes': recent_shifted,
            'delimiters_seen': self._delimiters_seen,
            'total_MB': round(self._total_bytes / (1024 * 1024), 1),
            'throughput_MBps': round(self._total_bytes / (1024 * 1024) / elapsed, 2)
            if elapsed > 0
            else 0,
            'fps_current': round(cur_fps, 1),
            'fps_average': round(avg_fps, 2),
            'usb_errors': self._usb_errors,
        }


# ---------------------------------------------------------------------------
# _ByteStream -- the streamed bytes, between the transport and the parser
# ---------------------------------------------------------------------------


class _ByteStream:
    """The bytes the device streams, owned in one place. Thread-safe.

    The transport's reader appends; the camera's grab loop takes what has
    arrived, puts back what it could not parse, and flushes at a window
    change. Every access goes through these methods, so a reader can never
    be left extending a buffer the parser no longer reads -- the failure of
    handing the parser the reader's own bytearray, which the parser then
    replaced with a new one on its first take.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._arrived = 0
        # When bytes last arrived: an unplug stops them without any error,
        # so the silence is how the driver hears it.
        self._last_arrival = time.monotonic()
        # Whether anything arrived since the grab loop last put its unparsed
        # tail back. Without new bytes that tail holds no delimiter it did not
        # already look for, and taking it again only spins the loop.
        self._fresh = False

    def append(self, data: bytes | bytearray) -> None:
        """Add bytes that arrived from the device."""
        with self._lock:
            self._buf.extend(data)
            self._arrived += len(data)
            self._last_arrival = time.monotonic()
            self._fresh = True

    def seconds_since_arrival(self, now: float) -> float:
        """How long, at ``now`` (``time.monotonic()``), since a byte last arrived."""
        with self._lock:
            return now - self._last_arrival

    def take_arrived_count(self) -> int:
        """How many bytes arrived since the last call.

        Counted where they arrive, because the grab loop takes the same
        bytes more than once: what it puts back while waiting for a frame's
        closing delimiter comes back in its next take.
        """
        with self._lock:
            arrived, self._arrived = self._arrived, 0
            return arrived

    def take(self, at_least: int) -> bytearray | None:
        """Everything buffered, or None while fewer than ``at_least`` bytes are
        or nothing has arrived since the last ``put_back``."""
        with self._lock:
            if len(self._buf) < at_least or not self._fresh:
                return None
            taken = self._buf
            self._buf = bytearray()
            self._fresh = False
            return taken

    def put_back(self, data: bytes | bytearray, *, limit: int, keep: int) -> None:
        """Return unparsed bytes ahead of what arrived since the take.

        Past ``limit`` bytes only the newest ``keep`` are kept. One lock
        acquisition covers both, so the reader cannot append between them.
        Bytes that arrived since the take keep the buffer due another take.
        """
        with self._lock:
            self._buf[:0] = data
            if len(self._buf) > limit:
                del self._buf[:-keep]

    def flush(self) -> None:
        """Drop everything buffered. What arrived stays counted: it did arrive."""
        with self._lock:
            self._buf.clear()
            self._fresh = False

    def restart(self) -> None:
        """A new stream: nothing buffered and nothing counted from the last one.

        The silence is timed from here, so a stream is not heard as silent
        before its first bytes have had time to come.
        """
        with self._lock:
            self._buf.clear()
            self._arrived = 0
            self._last_arrival = time.monotonic()
            self._fresh = False


# ---------------------------------------------------------------------------
# Transports -- every USB library call the driver makes
# ---------------------------------------------------------------------------


class _PyusbTransport:
    """Discovery, the firmware upload and idle control, through pyusb.

    The base of both platform transports; each adds the stream. While a
    stream runs, the pyusb handle is released -- only one handle on the
    device at a time -- and control goes through the stream's handle, then
    comes back to a reopened pyusb handle when the stream stops.
    """

    def __init__(self):
        self._dev = None

    def find(self, pid: int) -> Any:
        """The FX2 enumerated under ``pid``, or None."""
        return usb.core.find(idVendor=VID, idProduct=pid)

    def describe(self, dev: Any) -> str:
        """The found device's place on the host, for the log."""
        return describe_usb_device(dev)

    def write_to(self, dev: Any, request: int, value: int, index: int, data: bytes) -> None:
        """A vendor OUT request to a found device that is not opened: the bootloader."""
        dev.ctrl_transfer(0x40, request, value, index, data)

    def open(self, dev: Any) -> None:
        """Detach the kernel driver, configure, claim interface 0; idle control goes here."""
        # On macOS/Linux, detach the kernel driver if it grabbed the
        # interface. Windows pyusb raises NotImplementedError here -- ignore.
        try:
            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
                logger.info('[FX2 Conn  ] detached kernel driver from interface 0')
        except (usb.core.USBError, NotImplementedError):
            pass

        try:
            dev.set_configuration()
        except usb.core.USBError:
            pass  # may already be configured

        try:
            usb.util.claim_interface(dev, 0)
        except usb.core.USBError:
            pass  # may already be claimed

        self._dev = dev
        logger.info('[FX2 Conn  ] USB device configured, interface 0 claimed')

    def control_out(self, request: int, value: int, index: int, data: bytes, timeout: int) -> int:
        """A vendor OUT request on the opened device; returns the bytes written."""
        return self._dev.ctrl_transfer(0x40, request, value, index, data, timeout=timeout)

    def _release_idle(self) -> None:
        try:
            usb.util.dispose_resources(self._dev)
        except Exception:
            pass

    def _reopen_idle(self) -> None:
        try:
            dev = self.find(PID_APP)
            if dev is not None:
                self.open(dev)
        except Exception as e:
            logger.warning('[FX2 Conn  ] pyusb handle reopen failed: %s', e)

    def close(self) -> None:
        """Release the pyusb handle. Idempotent, swallows errors."""
        if self._dev is not None:
            try:
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
            self._dev = None


class _LibusbTransport(_PyusbTransport):
    """macOS / Linux: the ISO stream, and control while it runs, through python-libusb1."""

    def __init__(self):
        super().__init__()
        self._ctx = None
        self._handle = None
        self._transfers: list = []
        self._event_thread: threading.Thread | None = None
        self._streaming = False
        self._stream: _ByteStream | None = None
        self._on_error = None
        self._on_gone = None

    def control_out(self, request: int, value: int, index: int, data: bytes, timeout: int) -> int:
        if self._handle is not None:
            return self._handle.controlWrite(0x40, request, value, index, data, timeout=timeout)
        return super().control_out(request, value, index, data, timeout)

    def start_stream(
        self, stream: _ByteStream, on_error: Callable[[], None], on_gone: Callable[[], None]
    ) -> None:
        """Stream ISO data into ``stream``.

        ``on_error`` is called per failed transfer or packet; ``on_gone`` when a
        transfer cannot be resubmitted because the device has left the bus.
        """
        self._release_idle()

        # Explicit open: usb1's lazy auto-open on first use is deprecated
        # (warns at every stream start) and skips the library's shutdown
        # cleanup registration. open() returns the context; the paired
        # explicit close() lives in stop_stream.
        self._ctx = usb1.USBContext().open()
        handle = self._ctx.openByVendorIDAndProductID(VID, PID_APP)
        if handle is None:
            raise RuntimeError('FX2 USB device disappeared before ISO streaming could start')
        try:
            if handle.kernelDriverActive(0):
                handle.detachKernelDriver(0)
        except Exception:
            pass
        handle.claimInterface(0)
        handle.setInterfaceAltSetting(0, ISO_ALT_INTERFACE)

        # Control goes through this handle while streaming -- the pyusb
        # handle is released.
        self._handle = handle
        self._stream = stream
        self._on_error = on_error
        self._on_gone = on_gone
        self._streaming = True

        # Submit ISO transfers BEFORE sending VR_START_STREAMING. Transfers
        # must be pending when data starts flowing or the FIFO overflows
        # while we're still queuing up.
        self._transfers = []
        for _ in range(ISO_NUM_TRANSFERS):
            xfer = handle.getTransfer(iso_packets=ISO_NUM_PACKETS)
            xfer.setIsochronous(
                0x82,
                ISO_MAX_PACKET_SIZE * ISO_NUM_PACKETS,
                callback=self._iso_callback,
                timeout=5000,
                iso_transfer_length_list=[ISO_MAX_PACKET_SIZE] * ISO_NUM_PACKETS,
            )
            xfer.submit()
            self._transfers.append(xfer)

        # USB event pump in a dedicated thread -- libusb1 needs someone
        # to call handleEventsTimeout() to process ISO completions.
        self._event_thread = threading.Thread(target=self._usb_event_loop, daemon=True)
        self._event_thread.start()

        # Now start streaming -- transfers are ready to receive data.
        handle.controlWrite(0x40, VR_START_STREAMING, 0, 0, b'')

        logger.info(
            '[FX2 Conn  ] streaming started (ISO alt %d, EP 0x82, %d transfers x %d packets)',
            ISO_ALT_INTERFACE,
            ISO_NUM_TRANSFERS,
            ISO_NUM_PACKETS,
        )

    def stop_stream(self) -> None:
        """Stop the ISO stream and give control back to a reopened pyusb handle.

        Matches the LVC reference: cancel transfers, drain events for ~2s,
        join the event thread, send STOP, close the handle.
        """
        self._streaming = False
        for xfer in self._transfers:
            try:
                xfer.cancel()
            except Exception:
                pass

        # Drain cancelled transfers.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                self._ctx.handleEventsTimeout(tv=0.1)
            except Exception:
                break

        if self._event_thread is not None:
            self._event_thread.join(timeout=3.0)
            self._event_thread = None

        try:
            self._handle.controlWrite(0x40, VR_STOP_STREAMING, 0, 0, b'')
        except Exception:
            pass
        try:
            self._handle.releaseInterface(0)
            self._handle.close()
        except Exception:
            pass
        self._transfers = []
        # Paired with the explicit open() at stream start: dropping the
        # reference without close() leaks the libusb context until GC. The
        # transfers are cancelled and the handle closed above, so close()
        # is safe here.
        self._ctx.close()
        self._ctx = None
        self._handle = None
        self._stream = None
        self._on_error = None
        self._on_gone = None

        self._reopen_idle()

    def _iso_callback(self, transfer):
        """libusb1 callback -- called when an ISO transfer completes.

        A transfer that fails, and a failed packet inside one that completed,
        are each counted as a USB error: the packet's bytes are missing from
        the stream, which is what turns the frame around it into a partial.
        """
        status = transfer.getStatus()
        if status == usb1.TRANSFER_CANCELLED:
            return
        if status == usb1.TRANSFER_COMPLETED:
            failed_packets = 0
            received = bytearray()
            for packet_status, buf in transfer.iterISO():
                if packet_status != usb1.TRANSFER_COMPLETED:
                    failed_packets += 1
                elif len(buf) > 0:
                    received.extend(buf)
            if received:
                self._stream.append(received)
            for _ in range(failed_packets):
                self._on_error()
        else:
            self._on_error()
        # Resubmit for continuous streaming.
        if self._streaming:
            try:
                transfer.submit()
            except Exception as e:
                # A dead transfer is one fewer in flight; when all are
                # gone the stream silently freezes (the preview keeps
                # showing the last frame). ERROR level so a frozen-
                # preview post-mortem finds the cause next to the
                # display-stall watchdog warning. Bounded by the
                # transfer count -- this is not a per-frame loop.
                logger.error(
                    '[FX2 Conn  ] _iso_callback: transfer resubmit failed; '
                    'grab loop will stall if this persists: %s: %s',
                    type(e).__name__,
                    e,
                )
                # Unplugged, the resubmit is refused at once: the fastest
                # word the driver gets that the device is gone, and the one
                # LumaView Classic stopped its stream on.
                if isinstance(e, (usb1.USBErrorNoDevice, usb1.USBErrorNotFound)):
                    self._on_gone()

    def _usb_event_loop(self):
        """Pump libusb1 events in a dedicated thread.

        handleEventsTimeout(tv=0.1) blocks up to 100 ms per call, so even
        when the device dies and every call raises, this loop degrades to
        a ~10 Hz idle poll -- it does not hot-spin.
        """
        while self._streaming:
            try:
                self._ctx.handleEventsTimeout(tv=0.1)
            except Exception:
                if not self._streaming:
                    break


class _WinUsbTransport(_PyusbTransport):
    """Windows: the ISO stream, and control while it runs, through WinUSB (``drivers/winusb_iso.py``)."""

    def __init__(self):
        super().__init__()
        self._reader = None

    def control_out(self, request: int, value: int, index: int, data: bytes, timeout: int) -> int:
        if self._reader is not None:
            return self._reader.device.control_transfer(0x40, request, value, index, data=data)
        return super().control_out(request, value, index, data, timeout)

    def start_stream(
        self, stream: _ByteStream, on_error: Callable[[], None], on_gone: Callable[[], None]
    ) -> None:
        """Stream ISO data into ``stream``; ``on_error`` is called per failed read or packet.

        ``on_gone`` is not called: the WinUSB reader reports no removal of its
        own, so an unplug is heard as the stream's silence.
        """
        from drivers.winusb_iso import WinUsbIsoReader

        # Release the pyusb handle -- WinUSB needs exclusive device access.
        self._release_idle()

        reader = WinUsbIsoReader(
            VID,
            PID_APP,
            pipe_id=0x82,
            alt_interface=ISO_ALT_INTERFACE,
            num_slots=ISO_NUM_TRANSFERS,
            packets_per_xfer=ISO_NUM_PACKETS,
            on_data=stream.append,
            on_error=on_error,
        )
        reader.start()
        # Held before START, so a START that raises leaves the running
        # reader where stop_stream stops it. It also routes control
        # transfers through the reader while streaming: without it, any LED
        # command or exposure/gain change during streaming would fail on
        # Windows (the branch the 4.0.0-LVCtest integration dropped from the
        # LVC upstream).
        self._reader = reader

        # Send VR_START_STREAMING through the WinUSB reader (can't use
        # the pyusb handle -- it's released).
        reader.device.control_transfer(0x40, VR_START_STREAMING, 0, 0)

        logger.info(
            '[FX2 Conn  ] streaming started (WinUSB ISO alt %d, EP 0x82)',
            ISO_ALT_INTERFACE,
        )

    def stop_stream(self) -> None:
        """Stop the WinUSB stream and give control back to a reopened pyusb handle."""
        if self._reader is not None:
            try:
                self._reader.device.control_transfer(0x40, VR_STOP_STREAMING, 0, 0)
            except Exception:
                pass
            self._reader.stop()
            self._reader = None
        self._reopen_idle()


def _platform_transport() -> _PyusbTransport:
    """The transport for this host. The platform is decided here and nowhere else.

    Raises:
        ImportError: pyusb, or python-libusb1 off Windows, is not installed.
    """
    if not _HAS_USB:
        raise ImportError(
            'pyusb is required for FX2 hardware access. Install with: pip install pyusb'
        )
    if sys.platform == 'win32':
        return _WinUsbTransport()
    # libusb1 is needed for ISO streaming on macOS/Linux. Fail fast so an
    # LS620 user on macOS without libusb1 gets a clear install hint, not a
    # confusing runtime error 30 seconds in when they hit "start streaming".
    if not _HAS_USB1:
        raise ImportError(
            'libusb1 (python-libusb1) is required for FX2 ISO streaming '
            'on macOS / Linux. Install with: pip install -r requirements.txt'
        )
    return _LibusbTransport()


# ---------------------------------------------------------------------------
# _FX2Connection -- module-level singleton owning the device
# ---------------------------------------------------------------------------


class _FX2Connection:
    """Singleton owning the FX2 USB device, through a platform transport.

    Lazily constructed on first ``_FX2Connection.get()``. Private -- outside
    the FX2 drivers only the simulated FX2 builds one, on its own transport.
    ``FX2Camera`` and ``FX2LEDController`` reach it through ``get()`` in their
    ``__init__``, or through the connection they are handed, and are the only
    objects that talk to it: every USB library call is the transport's, and
    the bytes the device streams arrive in ``stream``.

    Why a singleton:
        The FX2 chip is one USB device with two functional sub-devices
        (camera + LED). pyusb cannot share a device handle across two
        independent driver objects safely, and the B2 driver registry
        constructs camera and LED separately -- so both registered
        drivers share the same underlying connection via this module-level
        instance. Proven viable by
        ``tests/test_driver_registry.py::TestRegistryAccommodatesCompositeHardware``.
    """

    _instance: _FX2Connection | None = None
    _instance_lock = threading.Lock()

    FIRMWARE_RE_ENUM_TIMEOUT = 15.0  # seconds to wait for re-enumeration
    FIRMWARE_CHUNK_SIZE = 0x800  # vendor req 0xA0 upload chunk

    @classmethod
    def get(cls) -> _FX2Connection:
        """Return the singleton, constructing it on first call.

        Raises on construction failure (no FX2 hardware, no pyusb, firmware
        upload timeout, etc.) -- the registry catches the exception and
        falls through to the next candidate driver.

        Returns:
            _FX2Connection: The shared singleton instance.

        Raises:
            ImportError: pyusb (or libusb1 on macOS/Linux) is not installed.
            RuntimeError: No Lumascope FX2 device was found, or firmware
                upload did not re-enumerate within the timeout window.
        """
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(_platform_transport())
            return cls._instance

    @classmethod
    def _reset_for_test(cls):
        """Drop the singleton so the next ``get()`` re-enumerates.

        Test-only. Never call from production code. Used by
        ``tests/test_fx2_driver.py`` between tests that mock pyusb
        differently.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance._teardown()
                except Exception:
                    pass
            cls._instance = None

    def __init__(self, transport: _PyusbTransport):
        self._transport = transport
        # Held around every control transfer and around the stream's start
        # and stop, the moments the transport moves control between the
        # pyusb handle and the stream's, so no transfer meets a half-made
        # switch.
        self._lock = threading.Lock()
        self.stream = _ByteStream()
        # Whether the stream is running, decided under _lock: the removal's
        # teardown and the application's own shutdown can both stop it, and
        # a transport stopped twice fails on state its first stop released.
        self._streaming = False
        # The device has left the bus. Written once, by the camera's grab
        # loop when it concludes an unplug; read by the LED, which shares
        # the device. Never cleared: a replugged scope is a new process.
        self.removed = False
        # Set from the transport's event thread when a transfer finds the
        # device gone; taken by the grab loop, which confirms it on the bus.
        self._gone_reported = threading.Event()

        try:
            self._connect()
        except Exception:
            self._teardown()
            raise

    # -- connection ---------------------------------------------------------

    def _connect(self):
        """Find the device, upload firmware if needed, open it."""
        transport = self._transport
        dev = transport.find(PID_APP)
        if dev is not None:
            logger.info('[FX2 Conn  ] device: %s', transport.describe(dev))
            transport.open(dev)
            logger.info(
                '[FX2 Conn  ] device found running firmware (PID 0x%04X)',
                PID_APP,
            )
            return

        dev = transport.find(PID_BOOT)
        if dev is None:
            raise RuntimeError(
                f'No Lumascope FX2 device found (checked PID 0x{PID_APP:04X} and 0x{PID_BOOT:04X})'
            )

        logger.info(
            '[FX2 Conn  ] bootloader found (PID 0x%04X), uploading firmware...',
            PID_BOOT,
        )
        self._upload_firmware(dev, _FX2Connection.find_firmware_path())

        # Wait for re-enumeration under the new application PID.
        deadline = time.monotonic() + self.FIRMWARE_RE_ENUM_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(0.5)
            dev = transport.find(PID_APP)
            if dev is not None:
                logger.info('[FX2 Conn  ] device: %s', transport.describe(dev))
                transport.open(dev)
                logger.info(
                    '[FX2 Conn  ] firmware loaded, re-enumerated as PID 0x%04X',
                    PID_APP,
                )
                return

        raise RuntimeError(
            f'FX2 did not re-enumerate after firmware upload '
            f'(waited {self.FIRMWARE_RE_ENUM_TIMEOUT:.0f}s)'
        )

    @staticmethod
    def find_firmware_path() -> str:
        """Locate the FX2 firmware hex file in a PyInstaller bundle or source tree.

        Two hex files ship with the driver:

        - ``LumascopeClassic.hex`` (45068 bytes) -- **patched** variant
          with a modified product string ("LS Classic") to improve
          device enumeration after firmware upload. Confirmed identical
          (SHA256 ``c15a9294...``) to ``LumascopeClassic_patched.hex`` in
          the ``LumaviewClassic`` development repo. This is the primary
          production firmware.
        - ``Lumascope600.hex`` (28434 bytes) -- original smaller firmware
          (AUTOIN=1024). Kept as a fallback for any unit that refuses
          to re-enumerate with the patched variant.

        History note: the ``4.0.0-LVCtest`` integration branch in LVP
        shipped the **unpatched** ``LumascopeClassic_original.hex``
        (SHA256 ``4d457a86...``) under the name ``LumascopeClassic.hex``
        -- a packaging mismatch that the LVC upstream later corrected.
        Stage 3 of the 4.1.0-dev port intentionally copies the patched
        variant from ``LumaviewClassic/firmware/LumascopeClassic.hex``,
        NOT from the ``4.0.0-LVCtest`` branch. If this file is ever
        re-copied from anywhere, verify its SHA256 matches.

        Search order (first hit wins):
          1. ``<base>/firmware/LumascopeClassic.hex``   (patched -- preferred)
          2. ``<base>/firmware/Lumascope600.hex``       (original -- fallback)
          3. ``<base>/../firmware/<...>``               (dev tree: drivers/ is one level below repo root)

        Under PyInstaller, ``<base>`` is ``sys._MEIPASS``; otherwise it's
        the directory containing this module file.
        """
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            base = os.path.dirname(os.path.abspath(__file__))

        names = ['LumascopeClassic.hex', 'Lumascope600.hex']

        candidates: list[str] = []
        for name in names:
            candidates.append(os.path.join(base, 'firmware', name))
            candidates.append(os.path.join(base, '..', 'firmware', name))
        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f'FX2 firmware hex file not found. Searched: {", ".join(candidates)}'
        )

    def _upload_firmware(self, dev, hex_path: str):
        """Upload Intel HEX firmware to FX2 via vendor request 0xA0."""
        data, end_addr = parse_intel_hex(hex_path)
        logger.info('[FX2 Conn  ] firmware: %s (%d bytes)', hex_path, end_addr)

        write_to = self._transport.write_to

        # Put 8051 into reset
        write_to(dev, VR_ANCHOR_DLD, 0xE600, 0, b'\x01')

        # Send firmware data in chunks
        addr = 0
        chunk = self.FIRMWARE_CHUNK_SIZE
        while addr < end_addr:
            remaining = end_addr - addr
            length = min(chunk, remaining)
            write_to(dev, VR_ANCHOR_DLD, addr, 0, data[addr : addr + length])
            addr += length

        # Release 8051 from reset -- firmware boots and the device re-enumerates
        write_to(dev, VR_ANCHOR_DLD, 0xE600, 0, b'\x00')
        logger.info('[FX2 Conn  ] firmware upload complete, 8051 released')

    # -- control transfers --------------------------------------------------

    def control_transfer_out(
        self,
        request: int,
        value: int = 0,
        index: int = 0,
        data: bytes = b'',
        timeout: int = 5000,
    ) -> int:
        """Thread-safe vendor OUT control transfer.

        While streaming, the pyusb handle is closed -- only one handle
        on the device at a time. The transport routes through whichever
        streaming handle is currently live (libusb1 on macOS/Linux, WinUSB
        reader on Windows), or the pyusb handle otherwise. Callers don't
        need to care which path is active.

        Args:
            request: USB vendor request code.
            value: 16-bit ``wValue`` field.
            index: 16-bit ``wIndex`` field.
            data: Payload bytes.
            timeout: Timeout in milliseconds.

        Returns:
            int: Bytes written, as reported by the underlying USB layer.
        """
        t_start = time.monotonic()
        try:
            with self._lock:
                result = self._transport.control_out(request, value, index, data, timeout)
        except Exception as e:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            if request not in _I2C_VR_REQUESTS:
                _serial_log.error(
                    f'[FX2] {_vr_name(request)} OUT value=0x{value:04X} '
                    f'index=0x{index:04X} len={len(data)} -> EXCEPTION: '
                    f'{type(e).__name__}: {e} ({elapsed_ms:.1f}ms)'
                )
            raise
        elapsed_ms = (time.monotonic() - t_start) * 1000
        # Skip log emission for I2C ops -- the i2c_write wrapper logs
        # with richer detail (addr + data bytes).
        if request not in _I2C_VR_REQUESTS:
            _serial_log.info(
                f'[FX2] {_vr_name(request)} OUT value=0x{value:04X} '
                f'index=0x{index:04X} len={len(data)} -> result={result} '
                f'({elapsed_ms:.1f}ms)'
            )
        return result

    def i2c_write(self, addr: int, data) -> int:
        """Write bytes to the I2C bus via vendor request 0xB3.

        Returns the result of the underlying control transfer (number of
        bytes written from pyusb / libusb1). Callers that want to detect
        short writes (e.g., LED command diagnostics) can compare to
        `len(data)`. Pre-2026-04-15 this method discarded the result,
        which masked silent short-write failures in `_led_write`.

        Args:
            addr: I2C device address (used as ``wIndex``).
            data: Bytes to write.

        Returns:
            int: Number of bytes written, as reported by the USB layer.
        """
        t_start = time.monotonic()
        try:
            result = self.control_transfer_out(VR_I2C_WRITE, value=0, index=addr, data=bytes(data))
        except Exception as e:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            _serial_log.error(
                f'[FX2 I2C] WRITE addr=0x{addr:02X} data={bytes(data)!r} '
                f'-> EXCEPTION: {type(e).__name__}: {e} ({elapsed_ms:.1f}ms)'
            )
            raise
        elapsed_ms = (time.monotonic() - t_start) * 1000
        _serial_log.info(
            f'[FX2 I2C] WRITE addr=0x{addr:02X} data={bytes(data)!r} '
            f'-> result={result} ({elapsed_ms:.1f}ms)'
        )
        return result

    def sensor_reg_write(self, reg: int, value: int) -> None:
        """Write 16-bit value to an MT9P031 register via VR_I2C_WRITE (0xB3).

        Wire format matches LVC exactly (`AptinaMT9P031_Control.cs::Write`):
        3 bytes `[reg, high, low]` sent as a VR_I2C_WRITE control transfer
        with `index = I2C_SENSOR` (0x5d). The FX2 firmware's `VR_I2C_WRITEb3`
        handler (vendor_req_parse.c:129) parses `wIndexL` as the I2C address,
        `wLengthL` as the byte count, truncates to 3 bytes max, and writes
        the received payload to I2C without touching IFCLK.

        WARNING -- do NOT route sensor writes through 0xBA
        (VR_IMAGE_SENSOR_CLK_MANAGED_WRITE). That variant switches IFCLK to
        internal, does the I2C write, then switches back. The GPIF pixel
        clock depends on IFCLK, so every 0xBA call disrupts streaming and
        produces visible image corruption on the next ISO frame. Our Stage
        3 port originally used 0xBA because the firmware comment for it
        says "sensor clock managed write" -- a misleading name. LVC defines
        the constant but never calls it from any production code path; the
        real production path is plain VR_I2C_WRITE (0xB3). Fixed
        2026-04-15.

        Args:
            reg: MT9P031 register address (one byte).
            value: 16-bit value to write.
        """
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        data = bytes([reg, high, low])
        self.control_transfer_out(VR_I2C_WRITE, value=0, index=I2C_SENSOR, data=data)

    # -- the stream ---------------------------------------------------------

    def start_stream(self, on_error: Callable[[], None]) -> None:
        """Start the device streaming into ``stream``, emptied first.

        Args:
            on_error: Called once per failed transfer or failed packet.
        """
        with self._lock:
            self.stream.restart()
            self._gone_reported.clear()
            # Marked before the transport starts: a start that raises part
            # way leaves what it opened for stop_stream to release.
            self._streaming = True
            self._transport.start_stream(self.stream, on_error, self._gone_reported.set)

    def stop_stream(self) -> None:
        """Stop the stream; control returns to the idle handle. Does nothing when stopped."""
        with self._lock:
            if not self._streaming:
                return
            self._streaming = False
            self._transport.stop_stream()

    # -- presence -----------------------------------------------------------

    def take_gone_report(self) -> bool:
        """Whether a transfer found the device gone since the last call."""
        reported = self._gone_reported.is_set()
        self._gone_reported.clear()
        return reported

    def device_present(self) -> bool | None:
        """Whether the device is enumerated on the bus; None when the bus could not be read.

        Enumeration only: nothing is opened, so this is safe while the stream
        runs on its own handle.
        """
        try:
            return self._transport.find(PID_APP) is not None
        except Exception as e:
            logger.warning('[FX2 Conn  ] bus enumeration failed: %s: %s', type(e).__name__, e)
            return None

    # -- teardown ----------------------------------------------------------

    def _teardown(self):
        """Release USB resources. Idempotent, swallows errors.

        Called from ``__init__`` on construction failure and from
        ``_reset_for_test``. Does NOT null out ``_instance`` -- that's
        ``_reset_for_test``'s job.
        """
        self._transport.close()


# ---------------------------------------------------------------------------
# _FX2ImageHandler -- frame buffering (inherits LVP's ImageHandlerBase)
# ---------------------------------------------------------------------------


class _FX2ImageHandler(ImageHandlerBase):
    """Thread-safe frame buffer for FX2 camera.

    The LVC reference driver carried a standalone fallback implementation
    with its own ``_new``/``_failure_count`` state for running outside
    LVP. We drop that here -- the 4.1.0-dev module only runs inside LVP,
    so ``ImageHandlerBase`` is always available and its behavior is what
    the rest of the camera stack expects.

    The base class implements ``_store_frame`` / ``get_last_image`` /
    ``_record_failure`` / ``reset``; this handler only says when its camera
    has been removed, so a frame buffered before an unplug is not handed out
    as current.
    """

    def __init__(self, camera: FX2Camera):
        super().__init__()
        self._camera = camera

    def _detached(self) -> bool:
        return self._camera._device_removed


# ---------------------------------------------------------------------------
# _UnplugWatch -- hearing an unplug in a stream that sends no error
# ---------------------------------------------------------------------------


class _UnplugWatch:
    """Decides, from the grab loop, when the FX2 has been unplugged.

    An unplug sends the driver no error: the bytes stop. Two things raise a
    suspicion -- no byte for ``SILENCE_S``, or a transfer that found the
    device gone -- and the bus decides it: absent on two probes
    ``CONFIRM_GAP_S`` apart is an unplug. A device still enumerated is
    probed at most every ``PROBE_INTERVAL_S`` while the silence lasts, and a
    stream silent for ``CEILING_S`` is dead whatever the bus says, since an
    enumeration can keep listing a device that has gone.

    Silence, not missing frames: a device whose frames all misalign stores
    none but keeps sending bytes, and it is not unplugged.
    """

    # Every gap a working stream shows is well under this: the first frame
    # arrives 0.3-0.5 s after a start, and a window change drops buffered
    # bytes without stopping their arrival.
    SILENCE_S = 2.0
    CONFIRM_GAP_S = 0.5
    PROBE_INTERVAL_S = 1.0
    CEILING_S = 30.0

    def __init__(self, connection: _FX2Connection):
        self._connection = connection
        self._suspected = False
        self._absences = 0
        self._last_probe: float | None = None

    def verdict(self, now: float) -> str | None:
        """Why the device is judged unplugged at ``now``, or None."""
        silent_s = self._connection.stream.seconds_since_arrival(now)
        if self._connection.take_gone_report():
            self._suspected = True
        if silent_s >= self.SILENCE_S:
            self._suspected = True
        if not self._suspected:
            return None
        if silent_s >= self.CEILING_S:
            return f'no byte for {silent_s:.0f} s'
        wait_s = self.CONFIRM_GAP_S if self._absences else self.PROBE_INTERVAL_S
        if self._last_probe is not None and now - self._last_probe < wait_s:
            return None
        self._last_probe = now
        present = self._connection.device_present()
        if present is False:
            self._absences += 1
            if self._absences >= 2:
                return f'off the bus on two probes, {silent_s:.1f} s after the last byte'
        elif present is True:
            self._absences = 0
            self._suspected = silent_s >= self.SILENCE_S
        return None


# ---------------------------------------------------------------------------
# FX2Camera -- Camera ABC implementation
# ---------------------------------------------------------------------------


@_register_if_fx2_available(camera_registry, 'fx2', priority=80)
class FX2Camera(Camera):
    """Camera driver for Lumascope Classic (MT9P031 via FX2 USB).

    Registered at priority 80 (below pylon/ids/sim at 100) so LS820/LS850
    units with a Basler camera are preferred on auto-detect. On LS620/LS720
    where no pylon/ids camera is present, those drivers raise and the
    registry falls through to FX2.

    Shares its USB connection with ``FX2LEDController`` via the module-
    level ``_FX2Connection`` singleton -- both drivers call
    ``_FX2Connection.get()`` in their constructors and end up pointing
    at the same handle without any coordination from Lumascope.__init__.
    """

    # The FX2 sensor delivers Mono8 only (set_pixel_format accepts no other
    # format; the grab loop builds uint8 buffers). Override the base 16-bit
    # default so capability consumers (buffer sizing, save-format selection)
    # treat FX2 frames as 8-bit from the start, matching IDSCamera.
    native_bit_depth = 8
    significant_bits = 8

    def significant_bits_for_format(self, pixel_format: str | None) -> int:
        """FX2 delivers 8-bit payloads regardless of any format string."""
        return 8

    # How often to log streaming stats (seconds). Set to 0 to disable.
    STATS_LOG_INTERVAL = 10.0

    # How long bytes may arrive with no frame stored before the stream is
    # reported unframeable. A working stream at the full window stores about
    # four frames a second (measured at 50 ms exposure), and a window change
    # costs it a few, so 5 s is about twenty frames' worth.
    FRAMING_STALL_S = 5.0

    # Frame size bounds
    FRAME_SIZE_MIN = 100
    FRAME_SIZE_STEP = 4

    # A typical microscopy starting point.
    DEFAULT_EXPOSURE_MS = 50.0

    def __init__(self, *, connection: _FX2Connection | None = None, **kwargs):
        # Take the FX2 connection BEFORE super().__init__() -- the Camera
        # base class calls self.connect() at the end of its __init__,
        # and that needs self._fx2 live. The registry passes none, so the
        # camera shares the process's device through _FX2Connection.get();
        # if that raises (no FX2 hardware, no pyusb, firmware upload
        # fails), the exception propagates and the registry falls through
        # to the next camera driver candidate. A simulated FX2 hands its
        # own connection to both drivers instead.
        self._fx2 = connection if connection is not None else _FX2Connection.get()

        # Streaming state -- initialized here so connect() can see them
        # even though connect() runs inside super().__init__().
        self._grabbing = False
        self._grab_thread: threading.Thread | None = None
        self._width = IMG_WIDTH
        self._height = IMG_HEIGHT
        # The exposure asked for, kept so every window gets the shutter width
        # that integrates it; the driver's default until init_camera_config()
        # applies it, since connect() sets the window first.
        self._exposure_ms = self.DEFAULT_EXPOSURE_MS
        self._gain_reg = 0x0008  # default = 1.0x = 0 dB
        self._pixel_format = 'Mono8'

        self.stream_stats = StreamStats()

        # Camera base class calls self.connect() at the end of its init.
        super().__init__()

        # Register an atexit hook to drain ISO streaming state before
        # Python interpreter shutdown collects the libusb1 context.
        # Background: any unhandled exception in user code while
        # streaming triggers Python interpreter shutdown. Daemon threads
        # (our usb_event_thread + grab_thread) keep running. Module-
        # level globals get GC'd, including the libusb1 USBContext,
        # which destroys its internal mutexes. The daemon event thread
        # is still inside `handleEventsTimeout` -- its next mutex lock
        # hits a destroyed mutex and crashes Python with
        # ``Assertion failed: pthread_mutex_destroy(mutex) == 0``
        # in libusb1's ``usbi_mutex_destroy``. Reproduced 2026-04-15
        # during Stage 3.5 hardware validation by a test script with
        # a wrong-arity unpacking error.
        #
        # Fix: atexit hook calls ``stop_grabbing`` before any Python
        # GC happens. atexit runs during normal interpreter shutdown
        # in LIFO order. The weakref ensures the hook doesn't pin the
        # camera object in memory -- if user code releases its FX2Camera
        # reference earlier, the camera can still be GC'd normally and
        # the atexit hook becomes a no-op.
        self_ref = weakref.ref(self)

        def _atexit_drain():
            inst = self_ref()
            if inst is None:
                return  # camera was already GC'd, nothing to drain
            if not inst.is_grabbing():
                return  # not streaming, libusb1 context not active
            try:
                inst.stop_grabbing()
            except Exception:
                pass  # swallow -- interpreter shutdown is in progress

        atexit.register(_atexit_drain)

    # -- Context manager support ------------------------------------------
    # `with FX2Camera() as cam:` ensures stop_grabbing + disconnect run
    # via __exit__ regardless of whether the body raises. This is the
    # primary recommended cleanup pattern for ad-hoc scripts and tests
    # -- the atexit hook above is a safety net for code that doesn't use
    # the context manager (e.g., long-lived LVP UI sessions where the
    # camera is held by the Lumascope object for the lifetime of the
    # app, not inside a `with` block).

    def __enter__(self) -> FX2Camera:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Best-effort cleanup. Swallow exceptions in cleanup so the
        # original exception (if any) propagates correctly.
        try:
            if self.is_grabbing():
                self.stop_grabbing()
        except Exception as e:
            logger.warning('[FX2 Cam   ] __exit__ stop_grabbing failed: %s', e)
        try:
            self.disconnect()
        except Exception as e:
            logger.warning('[FX2 Cam   ] __exit__ disconnect failed: %s', e)
        # Returning None / False propagates any exception from the with-body.

    # -- Connection --------------------------------------------------------

    def connect(self) -> bool:
        """Called by Camera base class during construction.

        Initializes the MT9P031 sensor, creates the frame handler, loads
        the camera profile, and applies default exposure/gain via
        ``init_camera_config()``. Returns the camera CONFIGURED but NOT
        grabbing -- the camera-lifecycle split: streaming begins exactly
        once via ``open_and_start()`` (the start gate). ScopeDisplay polls
        assuming the camera is grabbing, so the live view stays blank until
        the gate is released; the bring-up sites release it via
        ``scope.imaging.start_streaming()`` after configuration (the
        blank-view failure that bit the first LS620 GUI launch 2026-04-15).
        """
        self.model_name = 'MT9P031-LS620'
        self._init_sensor()
        self.cam_image_handler = _FX2ImageHandler(self)
        # Fresh handler starts with an empty dispatch list; re-push any durable
        # listeners so a reconnect keeps delivering frames to recording / plugins.
        self._reapply_frame_callbacks()
        self._active = True
        self._load_profile()
        self._query_dynamic_capabilities()
        self.init_camera_config()
        logger.info('[FX2 Cam   ] connected: %s', self.model_name)
        return True

    def disconnect(self) -> bool:
        self.stop_grabbing()
        self._active = None
        # Clear the start gate + last-frame buffer so a same-instance reconnect
        # starts clean (re-grabs, no stale image).
        self._reset_lifecycle_state()
        logger.info('[FX2 Cam   ] disconnected')
        return True

    def is_connected(self) -> bool:
        return self._active is not None and bool(self._active) and not self._device_removed

    def _query_dynamic_capabilities(self):
        """Populate profile's dynamic gain / exposure fields.

        FX2 has no SDK to query -- these are hardcoded from the MT9P031
        datasheet and the driver's row-time constant. We fill them in
        here so the rest of LVP can read ``scope.capabilities`` or
        ``camera.profile.gain.total_max_db`` and get real numbers
        instead of the ``None`` defaults.
        """
        try:
            self.profile.gain.total_min_db = 0.0
            self.profile.gain.total_max_db = 42.1  # 128x, per audit-corrected math
            # One shutter row at the full window: the shortest exposure
            # every window the driver allows can give (a narrower window's
            # row is shorter, so it reaches this to within its own row).
            self.profile.exposure_min_us = exposure_s(1, self._column_size(IMG_WIDTH)) * 1e6
            # Cap exposure at the legacy LVC 178 ms value (matches what
            # was known-safe in the original LumaviewClassic UI). Once the
            # shutter width passes H + 25 rows the sensor adds blanking rows
            # to stretch the frame (frame_time_s): about 233 ms at 1900
            # wide, but about 84 ms at 1000 and 32 ms at 500, so at narrow
            # windows exposures inside this cap already stretch the frame.
            # The cap was hardware-validated 2026-04-15 at 1900 only, on the
            # first LS620 GUI run, where dragging the exposure slider above
            # ~200 ms corrupted the image.
            SAFE_EXPOSURE_MAX_MS = 178
            self.profile.exposure_max_us = SAFE_EXPOSURE_MAX_MS * 1000
            logger.debug(
                '[FX2 Cam   ] profile capabilities: gain 0.0-42.1 dB, exposure %.3f-%.3f ms',
                self.profile.exposure_min_us / 1000,
                self.profile.exposure_max_us / 1000,
            )
        except Exception as e:
            logger.warning('[FX2 Cam   ] _query_dynamic_capabilities failed: %s', e)

    # -- Sensor init -------------------------------------------------------

    def _init_sensor(self):
        """Initialize MT9P031 sensor: PLL, window, black level calibration.

        Uses individual 3-byte register writes because the FX2 firmware's
        I2C handler truncates writes longer than 3 bytes. 10 ms sleep
        between writes is conservative -- the sensor responds much faster
        but this matches the LVC reference that hardware-validated at
        63/63 frames.

        WARNING: do NOT send VR_INIT_GPIF here. Per the firmware
        disassembly (see LumaviewClassic/docs/STREAMING_ANALYSIS.md
        sec.3.2), VR_INIT_GPIF calls TD_Init() -> Init_GPIF() -> SetISOInterface()
        internally, which resets EP2 configuration. The sensor writes here
        go through plain VR_I2C_WRITE, which never touches IFCLK, so no
        GPIF re-init is needed.
        """
        fx2 = self._fx2

        # Initial window -- set a default BEFORE PLL config. Overwritten
        # by the set_frame_size() call at the end of this method.
        fx2.sensor_reg_write(REG_ROW_START, 0x0036)  # sensor default row_start
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_COL_START, 0x0010)  # sensor default col_start
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_ROW_SIZE, 0x0797)  # sensor default 1943
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_COL_SIZE, 0x0A1F)  # sensor default 2591
        time.sleep(0.01)

        # PLL power on
        fx2.sensor_reg_write(REG_PLL_CTRL, 0x0051)
        time.sleep(0.01)

        # PLL config: the fields the timing model reads (_PIXEL_CLOCK_HZ,
        # 23.14 MHz from the 24 MHz EXTCLK).
        fx2.sensor_reg_write(REG_PLL_CFG1, (_PLL_M << 8) | _PLL_N_DIVIDER)
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_PLL_CFG2, _PLL_P1_DIVIDER)
        time.sleep(0.01)

        # PLL activate
        fx2.sensor_reg_write(REG_PLL_CTRL, 0x0053)
        time.sleep(0.2)  # datasheet requires 1ms for VCO lock; 200ms is defensive
        # Do NOT send VR_INIT_GPIF here -- see docstring warning.

        # Blue-strip fix per MT9P031 developer guide (DG_A page 7).
        # Prevents a blue strip artifact when bright light hits the top
        # or bottom of the sensor array. Recommended even at slower
        # pixel clocks where it may not be strictly necessary.
        fx2.sensor_reg_write(0x7F, 0x0000)
        time.sleep(0.01)

        # Black level calibration
        fx2.sensor_reg_write(REG_BLC, 0x6000)  # lock green + red/blue BLC channels
        time.sleep(0.01)
        # Read Mode 2 bits we set:
        #   bit  6 (0x0040) -- Row_BLC enabled (sensor default)
        #   bit 14 (0x4000) -- Mirror_Column = horizontal flip. Per
        #                     Linux kernel mt9p031.c register defs.
        #                     LS620 optic path delivers a left/right-
        #                     reversed view through the eyepiece vs the
        #                     sensor's native readout; this bit corrects
        #                     it at the sensor (free, no CPU cost,
        #                     applies to live view + captures uniformly).
        # If the image ends up upside down instead of mirrored, swap
        # bit 14 -> bit 15 (0x4000 -> 0x8000) for Mirror_Row instead.
        fx2.sensor_reg_write(REG_READ_MODE2, 0x4040)
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_ROW_BLACK, 0x0000)  # black target = 0 (microscopy optimization)
        time.sleep(0.01)

        # Set default window to full 1900x1900 -- also configures the
        # col_size/row_size registers correctly with centering.
        if self.set_frame_size(IMG_WIDTH, IMG_HEIGHT) is False:
            # The neighboring init register writes raise on a USB failure and
            # abort connect(); the window apply must not be quieter. Swallowed
            # here, connect() would report success with the sensor still at
            # its power-on window and the ISO frame parser desynced on the
            # mismatched byte count -- garbage live view with no error.
            raise RuntimeError('MT9P031 initial window apply failed during sensor init')

        logger.info('[FX2 Cam   ] MT9P031 sensor initialized (PLL + BLC)')

    # -- Streaming start / stop --------------------------------------------

    def start_grabbing(self) -> None:
        if self._grabbing:
            if _cam_log is not None:
                _cam_log.info('fx2 start_grabbing SKIPPED: already grabbing')
            return
        if _cam_log is not None:
            _cam_log.info('fx2 start_grabbing')
        self.stream_stats.reset()
        self._grabbing = True  # set BEFORE starting threads that check it
        self._fx2.start_stream(on_error=self.stream_stats.record_usb_error)
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def stop_grabbing(self) -> None:
        if not self._grabbing:
            if _cam_log is not None:
                _cam_log.info('fx2 stop_grabbing SKIPPED: not grabbing')
            return
        if _cam_log is not None:
            _cam_log.info('fx2 stop_grabbing')
        self._grabbing = False
        self._fx2.stop_stream()

        if self._grab_thread is not None:
            self._grab_thread.join(timeout=3.0)
            self._grab_thread = None
        # Bytes that arrived after the grab loop's last take belong to this stream.
        self.stream_stats.record_bytes(self._fx2.stream.take_arrived_count())

        s = self.stream_stats.summary()
        logger.info(
            '[FX2 Cam   ] streaming stopped: %d frames in %.1fs (%.1f fps avg), '
            '%d partial, %d shifted, %d USB errors, %.1f MB total',
            s['good_frames'],
            s['elapsed_s'],
            s['fps_average'],
            s['partial_frames'],
            s['shifted_frames'],
            s['usb_errors'],
            s['total_MB'],
        )

    def is_grabbing(self) -> bool:
        return self._grabbing and self._grab_thread is not None and self._grab_thread.is_alive()

    # -- Grab loop ---------------------------------------------------------

    def _grab_loop(self):
        """Extract frames from the connection's byte stream.

        Departures from the LVC reference:
        - ``local_buf`` is explicitly initialized before the loop instead
          of relying on ``'local_buf' not in dir()`` (fragile, un-Pythonic).
        - The trim-after-prepend is one operation on the stream with the
          prepend (``put_back``) instead of two lock acquisitions, which
          could race against the reader appending.
        """
        stats = self.stream_stats
        stream = self._fx2.stream
        last_stats_log = time.monotonic()
        first_frame_logged = False
        local_buf: bytearray | None = None  # explicit init
        watch = _UnplugWatch(self._fx2)
        # When the last frame was stored, and the discards counted by then:
        # bytes that keep arriving while nothing is stored are a stream the
        # parser cannot frame, said once per episode.
        last_stored = time.monotonic()
        rejected_at_store = stats.rejected()
        framing_reported = False

        while self._grabbing:
            unplugged = watch.verdict(time.monotonic())
            if unplugged is not None:
                self._on_unplugged(unplugged)
                return

            # Re-read dimensions every iteration -- the UI can call
            # set_frame_size() between frames.
            w = self._width
            h = self._height
            layout = frame_layout(w, h)
            stride, skip_first_row, needed = layout.stride, layout.skip, layout.needed

            local_buf = stream.take(needed)

            if local_buf is None:
                time.sleep(0.005)
                continue

            stats.record_bytes(stream.take_arrived_count())

            # Scan for frame delimiters.
            buf = local_buf
            while True:
                idx = buf.find(FRAME_DELIM)
                if idx < 0:
                    # No complete frame -- put unconsumed data back and
                    # trim if it's gotten out of hand.
                    stream.put_back(buf, limit=needed * 3, keep=needed * 2)
                    break

                frame_data = buf[:idx]
                buf = buf[idx + len(FRAME_DELIM) :]

                # Strict frame validation. The MT9P031 + FX2 GPIF emits
                # frames with EXACTLY one extra row of stride padding
                # beyond the math (`needed`). Measured 2026-04-15 on
                # 175 samples of clean streaming: 173/175 (98.9%) were
                # exactly `needed + stride` bytes, the other 2 were
                # corrupt (1 partial, 1 oversized). The +stride extra
                # is hardware-constant for fixed frame size; the
                # `as_strided` block below silently truncates it.
                #
                # PRIOR BEHAVIOR: the check was
                # `len(frame_data) >= needed`, which silently accepted
                # arbitrary oversized frames as "good" and reshaped
                # them from a misaligned offset -> visually corrupt
                # frames flagged as good, no telemetry. The partial-
                # frame counter only fires on undersize and missed
                # this entirely.
                #
                # CURRENT BEHAVIOR: strict equality on `expected`.
                # Anything else is discarded, distinct shifted/partial
                # counters give honest telemetry on which failure mode
                # dominates. If frame size or readout config ever
                # changes such that the +stride invariant breaks, the
                # shifted counter will spike and we re-measure.
                expected = layout.frame_bytes

                if len(frame_data) == expected:
                    raw = np.frombuffer(frame_data, dtype=np.uint8)
                    remaining = raw[skip_first_row:]
                    raw_2d = np.lib.stride_tricks.as_strided(
                        remaining, shape=(h, stride), strides=(stride, 1)
                    )
                    image = raw_2d[:, :w].copy()
                    # The FX2 sensor is 8-bit only, so the delivered array's
                    # container width IS its payload depth; stamp it from the
                    # frame so depth and pixels stay paired.
                    self.cam_image_handler._store_frame(
                        image, datetime.now(), significant_bits=image.dtype.itemsize * 8
                    )
                    stats.record_good_frame()
                    last_stored = time.monotonic()
                    rejected_at_store = stats.rejected()
                    framing_reported = False

                    if not first_frame_logged:
                        first_frame_logged = True
                        logger.info(
                            '[FX2 Cam   ] first frame: %dx%d, stride=%d, %d bytes, mean=%.1f',
                            w,
                            h,
                            stride,
                            len(frame_data),
                            float(image.mean()),
                        )
                elif len(frame_data) > needed:
                    # Wrong size but bigger than minimum -- either
                    # oversized (missed delimiter, two frames glued)
                    # or sized between `needed` and `expected`
                    # (off-by-rows). Either way, the bytes are
                    # misaligned and would render as garbage.
                    stats.record_shifted_frame(len(frame_data))
                elif len(frame_data) > 0:
                    # Severely undersized -- bytes dropped before the
                    # next delimiter was found.
                    stats.record_partial_frame(len(frame_data))

            now = time.monotonic()
            if not framing_reported and now - last_stored >= self.FRAMING_STALL_S:
                framing_reported = True
                shifted, partial = stats.rejected()
                logger.warning(
                    '[FX2 Cam   ] no frame stored for %.0f s while bytes keep arriving: '
                    '%d shifted and %d partial frames discarded, window %dx%d',
                    now - last_stored,
                    shifted - rejected_at_store[0],
                    partial - rejected_at_store[1],
                    w,
                    h,
                )

            # Periodic stats logging.
            if self.STATS_LOG_INTERVAL > 0 and (now - last_stats_log) >= self.STATS_LOG_INTERVAL:
                last_stats_log = now
                s = stats.summary()
                logger.info(
                    '[FX2 Cam   ] stream: %.1f fps (avg %.2f), '
                    '%d good / %d partial / %d shifted, '
                    '%.1f MB/s, %d errors',
                    s['fps_current'],
                    s['fps_average'],
                    s['good_frames'],
                    s['partial_frames'],
                    s['shifted_frames'],
                    s['throughput_MBps'],
                    s['usb_errors'],
                )

    def _on_unplugged(self, reason: str) -> None:
        """The device has left the bus: record it where it is read, then tear down.

        Runs on the grab loop, which returns right after. The teardown runs
        on the base Camera's thread, because stopping the stream joins this
        one; the connection's record is what the LED, on the same device,
        reads.
        """
        self._fx2.removed = True
        self._mark_disconnected()
        logger.warning(
            '[FX2 Cam   ] the FX2 was unplugged (%s); replug it and restart LumaViewPro',
            reason,
        )
        self._schedule_async_teardown()

    # -- Grab API (mostly inherits from Camera; override for clarity) ------

    def grab_new_capture(self, timeout_s: float = 5.0) -> tuple:
        """Block until a NEW frame arrives. Used by autofocus / protocols."""
        self.cam_image_handler.reset()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok, img, ts, _significant_bits, seq = self.cam_image_handler.get_last_image()
            if ok:
                with self._array_lock:
                    self.array = img
                return True, ts, seq
            time.sleep(0.01)
        return False, None, None

    # -- Frame size --------------------------------------------------------

    def set_frame_size(self, w: int, h: int) -> dict | bool:
        """Set the sensor readout window.

        The sensor is configured to output (display + 1) x (display + 1)
        pixels. The extra column becomes a 0x00 sync byte between rows
        after GPIF processing; the extra row is discarded by the grab
        loop (``skip_first_row``). Dimensions are rounded down to
        multiples of FRAME_SIZE_STEP (4) and clamped to [100, 1900].

        Returns the delivered size ``{'width': int, 'height': int}`` after
        rounding and clamping, so the caller knows what was actually applied
        without a read-back; ``False`` when a sensor-register write fails --
        the same failure contract the Camera base class documents and the
        pylon / IDS drivers implement, so the camera-write authority
        upstream sees one rejection signal from every driver. There is no
        up-front active-flag guard: connect() configures the initial window
        through this method BEFORE the active flag is set, and with no SDK
        to consult, a failing USB register write IS the disconnected signal
        (routed to False by the handler below).

        The row time follows the window's width, so the same shutter width
        integrates a different time at each window. The shutter width is
        written again after the window, from the exposure asked for, so the
        exposure stays the setting at every window, as on every other camera.
        It takes effect two frames after the window (the data sheet's shutter
        latency), inside the frames a window change already discards.
        """
        step = self.FRAME_SIZE_STEP
        w = max(self.FRAME_SIZE_MIN, min(IMG_WIDTH, int(w)))
        h = max(self.FRAME_SIZE_MIN, min(IMG_HEIGHT, int(h)))
        w = (w // step) * step
        h = (h // step) * step

        # Sensor registers want (display + 1) per LVC reference.
        sensor_w = self._column_size(w)
        sensor_h = h + 1
        # Center the window on the active pixel area (2592 x 1944 with
        # offsets 16 col / 54 row) and force even alignment.
        col_start = max(0, (2592 - sensor_w) // 2 + 16) & ~1
        row_start = max(0, (1944 - sensor_h) // 2 + 54) & ~1

        try:
            # Individual 3-byte writes -- firmware truncates multi-byte I2C.
            self._fx2.sensor_reg_write(REG_ROW_START, row_start)
            self._fx2.sensor_reg_write(REG_COL_START, col_start)
            self._fx2.sensor_reg_write(REG_ROW_SIZE, sensor_h)
            self._fx2.sensor_reg_write(REG_COL_SIZE, sensor_w)
            self._fx2.sensor_reg_write(
                REG_EXPOSURE, shutter_width_for(self._exposure_ms / 1000.0, sensor_w)
            )
        except Exception as e:
            # Translate a USB write failure into the base contract's explicit
            # False -- the rejection signal the pylon and IDS set_frame_size
            # already return, which the camera-write authority upstream turns
            # into its keep-prior-cache branch. The window fields mutate only
            # after all five writes land, so a failed apply never lets
            # get_frame_size() report geometry the sensor never took.
            logger.error(
                '[FX2 Cam   ] set_frame_size(%dx%d) register write failed: %s: %s '
                '(sensor window may be partially applied until the next '
                'successful set_frame_size)',
                w,
                h,
                type(e).__name__,
                e,
            )
            # A partial apply (some of the four registers landed) leaves the
            # sensor window indeterminate; buffered stream data may match no
            # known geometry and would desync the frame parser, so drop it on
            # the failure path too.
            self._fx2.stream.flush()
            return False

        self._width = w
        self._height = h

        # Flush the stream -- data captured with the old window is
        # now misaligned and would desync the frame parser.
        self._fx2.stream.flush()

        logger.info(
            '[FX2 Cam   ] frame size %dx%d (sensor %dx%d, row_start=%d, col_start=%d)',
            w,
            h,
            sensor_w,
            sensor_h,
            row_start,
            col_start,
        )
        return {'width': w, 'height': h}

    def get_frame_size(self):
        return {'width': self._width, 'height': self._height}

    def get_min_frame_size(self):
        return {'width': self.FRAME_SIZE_MIN, 'height': self.FRAME_SIZE_MIN}

    def get_max_frame_size(self):
        return {'width': IMG_WIDTH, 'height': IMG_HEIGHT}

    # -- Pixel format ------------------------------------------------------

    def set_pixel_format(self, pixel_format: str) -> bool:
        # MT9P031 is 12-bit but the FX2 firmware streams top 8 bits only.
        return pixel_format == 'Mono8'

    def get_pixel_format(self) -> str:
        return 'Mono8'

    def get_supported_pixel_formats(self) -> tuple:
        return ('Mono8',)

    # -- Exposure ----------------------------------------------------------

    def exposure_t(self, exposure_ms: float) -> float:
        """Set exposure time in milliseconds, returning the microseconds
        actually in effect.

        The request is quantized onto the sensor's row-time grid below, so
        the applied value routinely differs from what was asked for -- by up
        to a full row. The caller records a chunk-match target from this
        return; a request-derived target would not describe any exposure this
        sensor can produce.

        The shutter width is the one whose integration at the current
        window is nearest the request (``shutter_width_for``, from the data
        sheet's tEXP). The request is kept, so a window change writes the
        width that integrates it there.

        NOTE on effect timing: MT9P031 has a 2-frame pipeline delay
        between writing the shutter width register and seeing the new
        exposure in output frames. Callers that depend on exact timing
        (autofocus, protocol captures) must wait >=2 frames after an
        exposure change before relying on the new value.
        """
        # Refused while disconnected, as the other drivers do: no register
        # write is attempted and nothing is recorded. connect() sets the
        # active flag before init_camera_config() writes the defaults.
        if not self.is_connected():
            return False
        target_ms = float(exposure_ms)
        rows = shutter_width_for(target_ms / 1000.0, self._column_size(self._width))
        if _cam_log is not None:
            _cam_log.info(
                f'fx2 sensor_reg_write(REG_EXPOSURE={REG_EXPOSURE:#x}, rows={rows}) (={target_ms}ms)'
            )
        self._fx2.sensor_reg_write(REG_EXPOSURE, rows)
        self._exposure_ms = target_ms
        return self.get_exposure_t() * 1000.0

    def get_exposure_t(self) -> float:
        """The integration the sensor holds, in ms: the request to within a row."""
        column_size = self._column_size(self._width)
        shutter_width = shutter_width_for(self._exposure_ms / 1000.0, column_size)
        return exposure_s(shutter_width, column_size) * 1000.0

    @staticmethod
    def _column_size(w: int) -> int:
        """The Column_Size the driver writes for a ``w``-wide window."""
        return w + 1

    def auto_exposure_t(self, state: bool = True) -> NoReturn:
        raise no_hardware_auto_mode('FX2', 'auto_exposure_t', 'auto-exposure')

    # -- Gain --------------------------------------------------------------

    def gain(self, g: float) -> float | bool | None:
        """Set gain in dB. Clamped to [0.0, 42.1], then quantized onto the
        global gain register.

        A failed register write RAISES out of ``sensor_reg_write`` rather than
        returning, so this never answers refused. It answers with the gain the
        register now encodes, which differs from the request by the
        quantization step and by the clamp.

        Returns:
            float | bool | None: See ``Camera.gain``. False while disconnected:
            refused, nothing written, as ``exposure_t`` answers -- a None there
            reads to the API as applied, and it would record a gain the camera
            never received.
        """
        if not self.is_connected():
            return False
        db = max(0.0, min(42.1, float(g)))
        reg = _gain_db_to_register(db)
        self._gain_reg = reg
        if _cam_log is not None:
            _cam_log.info(
                f'fx2 sensor_reg_write(REG_GLOBAL_GAIN={REG_GLOBAL_GAIN:#x}, reg={reg:#x}) (={db}dB)'
            )
        self._fx2.sensor_reg_write(REG_GLOBAL_GAIN, reg)
        return _register_to_gain_db(reg)[1]

    def get_gain(self):
        _, db = _register_to_gain_db(self._gain_reg)
        return db

    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> NoReturn:
        raise no_hardware_auto_mode('FX2', 'auto_gain', 'auto-gain')

    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> NoReturn:
        raise no_hardware_auto_mode('FX2', 'auto_gain_once', 'auto-gain')

    def update_auto_gain_target_brightness(self, auto_target_brightness: float) -> NoReturn:
        raise no_hardware_auto_mode('FX2', 'update_auto_gain_target_brightness', 'auto-gain')

    def update_auto_gain_min_max(
        self, min_gain_db: float | None = None, max_gain_db: float | None = None
    ) -> NoReturn:
        raise no_hardware_auto_mode('FX2', 'update_auto_gain_min_max', 'auto-gain')

    # -- Misc (no-op or trivial) ------------------------------------------

    def init_camera_config(self) -> None:
        """Apply sensible defaults for exposure and gain on startup."""
        self.exposure_t(self.DEFAULT_EXPOSURE_MS)
        self.gain(0.0)  # 0 dB = 1x gain

    def get_all_temperatures(self) -> dict:
        return {}  # MT9P031 has no temperature sensor

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0):
        pass  # Frame rate is determined by PLL / exposure, not a software cap

    def set_binning_size(self, size: int) -> bool:
        return size == 1  # only 1x1 supported in this port

    def get_binning_size(self) -> int:
        return 1

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        pass  # MT9P031 has a test pattern register but it's not wired up


# ---------------------------------------------------------------------------
# FX2LEDController -- thin command translator, no state
# ---------------------------------------------------------------------------


@_register_if_fx2_available(led_registry, 'fx2', priority=80)
class FX2LEDController:
    """LED controller for Lumascope Classic via FX2 I2C at address 0x2A.

    **Thin command translator.** The LVC reference carried a ``led_ma``
    dict and client-side state tracking read back from it. That existed
    because the pre-4.1 GUI owned LED state. In 4.1 the API owns state via
    ``IlluminationAPI._led_state`` / ``save_led_state`` / ``restore_led_state``,
    so this driver keeps no LED state and answers no state queries.

    **LED channel -> I2C ASCII byte mapping** lives in this class only:
    LVP integer channels 0/1/2/3 -> ASCII bytes 0x43/0x42/0x41/0x44
    ('C'/'B'/'A'/'D') which the FX2 peripheral controller uses on the
    wire. Never leak the ASCII form above the driver -- that's a
    project-memory directive.

    **Fast variants.** The LED protocol requires ``led_on_fast`` /
    ``led_off_fast`` / ``leds_off_fast`` for time-critical toggling.
    These FX2 variants are IDENTICAL to the normal versions -- the I2C
    writes have no serial-handshake latency to skip, so "fast" is the
    same as "normal".
    """

    # Channel count is fixed at 4 by the hardware. Per project memory,
    # per-model LED presence (e.g. LS560 having only BF + Green) is
    # handled via scopes.json Layers filtering, NOT by hiding channels
    # here. The driver reports what the FX2 protocol supports.
    _COLOR_TO_CH = _COLOR_TO_CH  # module-level dict
    _CH_TO_COLOR = _CH_TO_COLOR

    # Full-scale drive current of the Classic LED peripheral, shared by
    # all four channels; the brightness byte is linear in current up to
    # it. 840 mA is the hardware owner's figure (2026-09-12), pending an
    # ammeter confirm on an LS620 -- if the sweep disagrees, this one
    # constant is the fix. The FX2 has no current readback, so nothing in
    # software can check delivered against requested.
    _MAX_MA = 840

    # The peripheral frames a command as [0xFF, channel, brightness]. A
    # brightness byte of 0xFF repeats the preamble and the frame is
    # dropped -- the channel goes DARK at the top of the scale, not
    # bright. Full scale therefore encodes as 0xFE (~837 mA), under one
    # LSB of the wire's resolution.
    _BRIGHTNESS_MAX = 0xFE

    # ------------------------------------------------------------------
    # Byte-level wire trace for bench investigation of the "illumination
    # slider > ~150 mA silently fails to light LED on LS620 FX2"
    # report (2026-04-16). Captures:
    #   * LED-toggle entry at FX2LEDController.led_on (mA + type)
    #   * mA->brightness conversion input/output
    #   * Each of the 3 I2C bytes written (hex dump)
    #   * PREAMBLE-COLLISION flag when brightness == 0xFF (historically the
    #     saturated byte; _ma_to_brightness now stops at _BRIGHTNESS_MAX, so
    #     the flag firing means the ceiling has been bypassed)
    # Companion gates live in modules/lumascope_api/illumination.py
    # (cache-equality check) and ui/layer_control.py (slider vs text entry
    # points). Toggle by either:
    #   * set fx2_debug_wire_enabled: true in the settings, which the
    #     session passes in as ``debug_wire``
    #   * flip _FX2_DEBUG_WIRE = True  below
    # ------------------------------------------------------------------
    _FX2_DEBUG_WIRE = False

    def _wire_debug_enabled(self) -> bool:
        return self._FX2_DEBUG_WIRE or self._debug_wire

    def __init__(self, *, connection: _FX2Connection | None = None, debug_wire: bool = False):
        # The registry passes no connection, so this takes the singleton --
        # raises if no FX2 hardware, registry fallthrough handles that case
        # cleanly. A simulated FX2 hands in the connection its camera shares.
        self._fx2 = connection if connection is not None else _FX2Connection.get()
        self._enabled = True
        self._debug_wire = debug_wire

        # Attributes the Lumascope API / SerialBoard pattern expects to
        # be able to read directly without method calls. ``driver`` is
        # a truthy sentinel; ``found`` means construction succeeded;
        # ``port`` is a human-readable tag for the settings UI.
        self.driver = True
        self.found = True
        self.port = 'FX2-USB'
        self.firmware_version = 'FX2-Classic'
        self.is_v2 = False

    # -- I2C write primitive ----------------------------------------------

    def _led_write(self, channel: int, brightness: int):
        """Send the 3-byte I2C LED command: 0xFF, ASCII channel, brightness.

        The FX2 firmware's I2C handler truncates writes longer than 3
        bytes, so we split into three single-byte writes with a 10 ms
        sleep between each. This matches the LVC reference that was
        hardware-validated on LS620 macOS.

        Each byte is wrapped in try/except and the i2c_write return
        value is checked against the expected 1-byte count. A silent
        short-write would mask LED-state corruption at the I2C layer;
        propagating the return value catches it at the driver layer
        instead.

        NOTE: the 3x 10 ms delays (30 ms total per LED command) are
        the known root cause of the slider-corruption effect documented
        in project memory. During streaming, each LED command holds
        the FX2 USB connection for 30 ms, during which ISO data keeps
        arriving but cannot be drained. If the UI sends dozens of LED
        writes per second (Kivy slider drags), the ISO buffer can
        desync the frame parser. Stage 3.5 will verify and fix --
        leading candidate is a debounce on the UI side (16 ms minimum
        between writes).
        """
        i2c_channel = _CH_TO_I2C.get(channel, channel)
        writes = [
            (0xFF, 'preamble'),
            (i2c_channel, 'channel'),
            (brightness & 0xFF, 'brightness'),
        ]
        # Byte-level wire trace for the slider > ~150 mA silent-fail
        # bench investigation. See _FX2_DEBUG_WIRE block above.
        if self._wire_debug_enabled():
            wire_hex = ' '.join(f'0x{b:02X}' for b, _ in writes)
            logger.info(
                '[FX2 LED diag] _led_write ch=%d i2c_ch=0x%02X brightness=0x%02X wire=[%s]%s',
                channel,
                i2c_channel,
                brightness,
                wire_hex,
                ' PREAMBLE-COLLISION' if brightness == 0xFF else '',
            )
        for byte_val, label in writes:
            try:
                result = self._fx2.i2c_write(I2C_LED, [byte_val])
            except Exception as e:
                logger.error(
                    '[FX2 LED  ] i2c_write raised on %s byte=0x%02x (ch=%d mA_equiv=%d): %s: %s',
                    label,
                    byte_val,
                    channel,
                    brightness,
                    type(e).__name__,
                    e,
                )
                raise
            if result is not None and result != 1:
                logger.warning(
                    '[FX2 LED  ] short write on %s byte=0x%02x '
                    '(ch=%d mA_equiv=%d): wrote %r of 1 byte expected',
                    label,
                    byte_val,
                    channel,
                    brightness,
                    result,
                )
            time.sleep(0.01)

    def _ma_to_brightness(self, mA) -> int:
        """Convert mA to the 0-0xFE brightness byte (0xFF is the preamble)."""
        brightness = max(0, min(self._BRIGHTNESS_MAX, round(float(mA) * 255.0 / self._MAX_MA)))
        # Workaround trace for mA->byte conversion. See the
        # _FX2_DEBUG_WIRE block above for the full instrumentation
        # rationale (LED driver brightness curve verification).
        if self._wire_debug_enabled():
            logger.debug(
                '[FX2 LED diag] _ma_to_brightness mA=%r type=%s -> '
                'brightness=%d (0x%02X) _MAX_MA=%d',
                mA,
                type(mA).__name__,
                brightness,
                brightness,
                self._MAX_MA,
            )
        return brightness

    # -- Channel discovery (B3) --------------------------------------------

    def available_channels(self) -> tuple:
        return tuple(self._COLOR_TO_CH.values())  # (0, 1, 2, 3)

    def max_ma(self) -> int:
        return self._MAX_MA

    def available_colors(self) -> tuple:
        return tuple(self._COLOR_TO_CH.keys())  # ('Blue', 'Green', 'Red', 'BF')

    def color2ch(self, color: str) -> int | None:
        return self._COLOR_TO_CH.get(color)

    def ch2color(self, channel: int) -> str | None:
        return self._CH_TO_COLOR.get(channel)

    # -- Core LED control --------------------------------------------------

    def led_on(self, channel: int, mA: int, block: bool = False, timeout: float = 5.0):
        if not self._enabled:
            return
        # Driver-entry trace (mA + type) for the slider > ~150 mA
        # silent-fail bench investigation. See _FX2_DEBUG_WIRE block
        # above. INFO level intentionally -- this is one of the two
        # key divergence points (int vs float entry).
        if self._wire_debug_enabled():
            logger.info(
                '[FX2 LED diag] led_on ENTRY ch=%d mA=%r type=%s block=%s',
                channel,
                mA,
                type(mA).__name__,
                block,
            )
        brightness = self._ma_to_brightness(mA)
        self._led_write(channel, brightness)

    def led_off(self, channel: int):
        self._led_write(channel, 0)

    def leds_off(self):
        for ch in _CH_TO_I2C:
            self._led_write(ch, 0)

    def leds_enable(self):
        self._enabled = True

    def leds_disable(self):
        self.leds_off()
        self._enabled = False

    # -- Fast variants (same as normal -- I2C has no serial handshake) -----

    def led_on_fast(self, channel: int, mA: int):
        self.led_on(channel, mA)

    def led_off_fast(self, channel: int):
        self.led_off(channel)

    def leds_off_fast(self):
        self.leds_off()

    # -- Connection no-ops (USB owned by _FX2Connection) ------------------

    def connect(self):
        pass

    def disconnect(self):
        pass

    def is_connected(self) -> bool:
        # The LED is on the camera's USB device: when the camera's grab loop
        # finds the device unplugged, the LED is gone with it.
        return self._fx2 is not None and not self._fx2.removed

    # -- Diagnostics / protocol completeness ------------------------------

    def get_status(self):
        return 'FX2-Classic LED controller'

    def wait_until_on(self, timeout: float = 5.0):
        pass  # no serial handshake to wait for

    def read_led_current(self, channel: int):
        return None  # no ADC feedback on FX2 LED peripheral

    def exchange_command(self, command: str, **kwargs):
        # LEDBoardProtocol requires this method for drivers that speak
        # a serial command protocol. FX2 uses I2C, not ASCII commands,
        # so there's nothing to exchange. Return None to match the
        # NullLEDBoard sentinel pattern.
        return None
