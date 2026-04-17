# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""FX2 driver for Lumascope Classic (LS620 / LS560 / LS720).

Cypress FX2 USB2 chip + Aptina MT9P031 image sensor, with LED control via
I2C through the same USB device. One physical device, two LumaViewPro
driver roles (camera + LED board).

Architecture
------------

Three objects live in this file:

1. ``_FX2Connection`` — module-level singleton that owns the USB handle,
   firmware upload, control transfers, I2C, sensor register R/W, and ISO
   streaming. Constructed lazily the first time any driver calls
   ``_FX2Connection.get()``. Raises on any failure; the registry treats a
   raise as "this driver isn't available" and falls through to the next
   candidate. Private — never touched from outside this module.

2. ``FX2Camera`` — registered as ``@camera_registry.register('fx2', ...)``.
   Implements the Camera ABC. Pulls ``_FX2Connection.get()`` in ``__init__``
   so the camera and LED end up sharing the same USB handle.

3. ``FX2LEDController`` — registered as ``@led_registry.register('fx2', ...)``.
   Satisfies LEDBoardProtocol. Thin command translator: no state tracking,
   no ``led_ma`` dict, no ``is_led_on`` bookkeeping. Source of truth for
   LED state is ``Lumascope._led_owners`` (post-B3 / Stage 2 architecture).
   The class exists only to convert LVP's (channel, mA) calls into FX2
   I2C byte sequences. State-query protocol methods return sentinel
   defaults (-1 / False / dict-of-False) — matching NullLEDBoard.

The camera and LED objects both hold a reference to the same
``_FX2Connection._instance`` — proven viable by
``TestRegistryAccommodatesCompositeHardware`` in tests/test_driver_registry.py.
No special casing required in ``Lumascope.__init__``.

Dependencies
------------

- ``pyusb``  (firmware upload + control transfers)
- ``libusb1`` (isochronous streaming on macOS/Linux — python-libusb1 binding)
- ``drivers.winusb_iso`` (isochronous streaming on Windows — ctypes wrapper)
- Native ``libusb-1.0`` library (macOS: ``brew install libusb``;
  Windows: vendored ``libusb-1.0.dll`` via PyInstaller binaries list)

Import-time safety
------------------

All USB / libusb1 imports are wrapped in try/except. If neither library is
installed, the module still imports and registers with the camera and LED
registries — but ``_FX2Connection.__init__`` raises ``ImportError`` on
first use, so the registry's auto-detect fallthrough handles it cleanly
without breaking non-FX2 scopes on dev machines without pyusb.

References
----------

- Reference port: ``git show 4.0.0-LVCtest:drivers/fx2driver.py``
- MT9P031 datasheet (DS_F, pages 35-36) for gain register layout
- MT9P031 developer guide (DG_A, page 7) for blue-strip workaround
- FX2 Technical Reference Manual for vendor request 0xA0 (VR_ANCHOR_DLD)
"""

from __future__ import annotations

import atexit
import math
import os
import sys
import threading
import time
import weakref
from collections import deque
from datetime import datetime

import numpy as np

from lvp_logger import logger
from drivers.camera import Camera, ImageHandlerBase
from drivers.registry import camera_registry, led_registry

try:
    import usb.core
    import usb.util
    _HAS_USB = True
    _USBError = usb.core.USBError
    _USBTimeoutError = usb.core.USBTimeoutError
except ImportError:
    _HAS_USB = False
    _USBError = OSError
    _USBTimeoutError = TimeoutError

try:
    import usb1
    _HAS_USB1 = True
except ImportError:
    _HAS_USB1 = False


# ---------------------------------------------------------------------------
# USB constants
# ---------------------------------------------------------------------------

VID = 0x04B4
PID_BOOT = 0x8613   # FX2 bootloader (before firmware upload)
PID_APP = 0xEA17    # Running firmware

# Vendor request codes — FX2 firmware vendor command handler
VR_ANCHOR_DLD = 0xA0          # Cypress standard: firmware upload
VR_I2C_READ = 0xB2
VR_I2C_WRITE = 0xB3
VR_I2C_MT9P031_READ = 0xB4    # Async MT9P031 register read (5s timeout OK)
VR_INIT_GPIF = 0xB9
# VR_IMAGE_SENSOR_CLK_MANAGED_WRITE (0xBA) was defined here as
# VR_SENSOR_CLK_WRITE and used by sensor_reg_write. It switches IFCLK
# to internal, does the I2C write, then switches back — which disrupts
# ISO streaming because the GPIF pixel clock depends on IFCLK. Removed
# 2026-04-15 after finding it was the root cause of visible image
# corruption on every gain/exposure slider drag. LVC defines the same
# constant in `I2C_Control.cs:52` but never calls it; LVC's production
# path uses VR_I2C_WRITE (0xB3) for sensor writes via
# `AptinaMT9P031_Control.WriteWord16 → I2C_Control.Write`. We now
# match LVC. See docs/AUDIT_FX2_RUNTIME.md Bug 6.
VR_SET_IFCLK_SRC = 0xBB
VR_CODE_VERSION = 0xBC
VR_START_STREAMING = 0xBD
VR_STOP_STREAMING = 0xBE

# I2C addresses
I2C_SENSOR = 0x5D   # MT9P031 image sensor
I2C_LED = 0x2A      # Peripheral controller (LEDs)


# ---------------------------------------------------------------------------
# Image sensor constants
# ---------------------------------------------------------------------------

IMG_WIDTH = 1900
IMG_HEIGHT = 1900
FRAME_BYTES = IMG_WIDTH * IMG_HEIGHT              # raw pixel count (8-bit mono)
FRAME_DELIM = b'\x01\xfe\x00\xff'                 # injected between frames by GpifWaveform_Isr

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

# Shutter width register (0x09) is 16-bit: max 65535 rows = ~7.4 seconds
MAX_EXPOSURE_ROWS = 65535

# Row time from MT9P031 datasheet (Table 8):
#   EXTCLK = 12 MHz (FX2 24 MHz crystal / 2)
#   PLL: M=27, N=1, P1=13 → pixel_clock = 24.923 MHz
#   Row period = 2 × max(W/2 + HBMIN, 486) = 2 × 1401 = 2802 pixel clocks
#   (W=1902, HBMIN=450 with Row_BLC enabled)
#   tROW = 2802 / 24.923 MHz = 112.4 μs = 0.1124 ms
_ROW_TIME_MS = 0.1124
# Shutter overhead SO = 426 pixel clocks = 0.0171 ms
_SHUTTER_OVERHEAD_MS = 0.0171


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
    0: 0x43,  # Blue  → 'C'
    1: 0x42,  # Green → 'B'
    2: 0x41,  # Red   → 'A'
    3: 0x44,  # BF    → 'D'
}


# ---------------------------------------------------------------------------
# ISO streaming parameters (matches C# ReadISOStream_WinUsb reference config)
# ---------------------------------------------------------------------------

ISO_ALT_INTERFACE = 3     # Alt interface 3 = ISO IN, 3x1024/microframe
ISO_NUM_TRANSFERS = 16    # Pending transfers in flight
ISO_NUM_PACKETS = 256     # ISO packets per transfer (C# reference uses 256)
ISO_MAX_PACKET_SIZE = 3072  # 3 × 1024 bytes per microframe


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
        firmware upload — everything past end_addr stays 0xFF).
    """
    buf = bytearray(0x4000)
    for i in range(len(buf)):
        buf[i] = 0xFF
    end_addr = 0

    with open(hex_path, 'r') as f:
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
                byte_val = int(line[9 + i * 2:11 + i * 2], 16)
                buf[addr] = byte_val
                addr += 1
            if addr > end_addr:
                end_addr = addr

    return bytes(buf), end_addr


# ---------------------------------------------------------------------------
# Gain conversion — MT9P031 datasheet (DS_F, pages 35-36) and register
# reference (RR_A pages 16-17).
# ---------------------------------------------------------------------------
# Global Gain register (0x35) bit fields:
#   Bits [14:8] = Digital_Gain     — legal values [0, 120] per RR_A
#   Bit  [6]    = Analog_Multiplier (0 or 1)
#   Bits [5:0]  = Analog_Gain      — legal values [8, 63] per RR_A
#
# Analog gain:  AG = (1 + Analog_Multiplier) × (Analog_Gain / 8)
# Digital gain: DG = 1 + (Digital_Gain / 8)
# Total gain:   AG × DG
#
# Strategy (datasheet recommended):
#   ≤ 4x:  analog only (multiplier=0) — best noise performance
#   ≤ 8x:  analog with multiplier=1
#   > 8x:  max analog (8x) + digital for the rest
#
# Range: 1x (0 dB) to 128x (42.1 dB). The LumaviewClassic LVC driver
# reference originally had `min(127, ...)` on the digital clamp and a
# comment claiming ~135x max — that was outside the documented legal
# range per RR_A. The audit in
# LumaviewClassic/docs/DATASHEET_VERIFICATION.md §5 corrected this to
# 120 / 128x. See also the docstring on `_gain_db_to_register`.

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
        analog_val = 32  # AG = 2 × 32/8 = 8.0
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
# StreamStats — frame rate / throughput diagnostics
# ---------------------------------------------------------------------------

class StreamStats:
    """Accumulates streaming diagnostics. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._frame_times: deque = deque(maxlen=120)
            self._partial_count = 0
            self._partial_sizes: deque = deque(maxlen=32)
            self._shifted_count = 0
            self._shifted_sizes: deque = deque(maxlen=32)
            self._good_count = 0
            self._total_bytes = 0
            self._usb_errors = 0
            self._usb_timeouts = 0
            self._start_time = time.monotonic()
            self._delimiters_seen = 0

    def record_good_frame(self):
        with self._lock:
            now = time.monotonic()
            self._frame_times.append(now)
            self._good_count += 1
            self._delimiters_seen += 1

    def record_partial_frame(self, size: int):
        """Frame between two delimiters was undersized (bytes dropped before
        next delimiter). Discarded by the grab loop."""
        with self._lock:
            self._partial_count += 1
            self._partial_sizes.append(size)
            self._delimiters_seen += 1

    def record_shifted_frame(self, size: int):
        """Frame between two delimiters was the wrong size — either oversized
        (likely a missed delimiter caused two frames to be concatenated, or a
        false-positive delimiter elsewhere in pixel data inflated the buffer)
        OR sized between `needed` and `expected` (off-by-some-rows, also wrong).
        Distinct from `partial` because the failure mechanism is different: a
        partial frame is lost bytes BEFORE the next delimiter; a shifted frame
        is wrong-but-still-bigger-than-minimum, indicating the parser found
        bytes from outside the intended frame. Both are discarded.
        See docs/AUDIT_FX2_RUNTIME.md Fix 1 (2026-04-15)."""
        with self._lock:
            self._shifted_count += 1
            self._shifted_sizes.append(size)
            self._delimiters_seen += 1

    def record_bytes(self, n: int):
        with self._lock:
            self._total_bytes += n

    def record_usb_error(self):
        with self._lock:
            self._usb_errors += 1

    def record_usb_timeout(self):
        with self._lock:
            self._usb_timeouts += 1

    def get_fps(self) -> tuple[float, float]:
        """Returns (current_fps, avg_fps). Current = last 2 seconds."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._start_time
            avg_fps = self._good_count / elapsed if elapsed > 0 else 0.0
            cutoff = now - 2.0
            recent = sum(1 for t in self._frame_times if t > cutoff)
            cur_fps = recent / 2.0
            return cur_fps, avg_fps

    def summary(self) -> dict:
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
            'throughput_MBps': round(self._total_bytes / (1024 * 1024) / elapsed, 2) if elapsed > 0 else 0,
            'fps_current': round(cur_fps, 1),
            'fps_average': round(avg_fps, 2),
            'usb_errors': self._usb_errors,
            'usb_timeouts': self._usb_timeouts,
        }


# ---------------------------------------------------------------------------
# _FX2Connection — module-level singleton owning the USB handle
# ---------------------------------------------------------------------------

class _FX2Connection:
    """Singleton owning the FX2 USB device.

    Lazily constructed on first ``_FX2Connection.get()``. Private — external
    callers should never reference this class directly. ``FX2Camera`` and
    ``FX2LEDController`` reach it only via ``get()`` in their ``__init__``.

    Why a singleton:
        The FX2 chip is one USB device with two functional sub-devices
        (camera + LED). pyusb cannot share a device handle across two
        independent driver objects safely, and the B2 driver registry
        constructs camera and LED separately — so both registered
        drivers share the same underlying connection via this module-level
        instance. Proven viable by
        ``tests/test_driver_registry.py::TestRegistryAccommodatesCompositeHardware``.
    """

    _instance: '_FX2Connection | None' = None
    _instance_lock = threading.Lock()

    FIRMWARE_RE_ENUM_TIMEOUT = 15.0   # seconds to wait for re-enumeration
    FIRMWARE_CHUNK_SIZE = 0x800       # vendor req 0xA0 upload chunk

    @classmethod
    def get(cls) -> '_FX2Connection':
        """Return the singleton, constructing it on first call.

        Raises on construction failure (no FX2 hardware, no pyusb, firmware
        upload timeout, etc.) — the registry catches the exception and
        falls through to the next candidate driver.
        """
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
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

    def __init__(self):
        if not _HAS_USB:
            raise ImportError(
                "pyusb is required for FX2 hardware access. "
                "Install with: pip install pyusb"
            )
        # libusb1 is needed for ISO streaming on macOS/Linux. Windows uses
        # the native WinUSB path (drivers/winusb_iso.py) and doesn't need
        # libusb1. Fail fast so an LS620 user on macOS without libusb1
        # gets a clear install hint, not a confusing runtime error 30
        # seconds in when they hit "start streaming".
        if sys.platform != 'win32' and not _HAS_USB1:
            raise ImportError(
                "libusb1 (python-libusb1) is required for FX2 ISO streaming "
                "on macOS / Linux. Install with: pip install libusb1. "
                "On macOS you also need the native library: brew install libusb."
            )
        self._dev = None
        self._lock = threading.Lock()
        # During streaming, the pyusb handle is closed — only one handle
        # at a time. Control transfers issued while streaming route
        # through whichever of these is live:
        #   _iso_handle_for_ctrl      → usb1 handle (macOS/Linux ISO path)
        #   _winusb_reader_for_ctrl   → WinUsbIsoReader (Windows WinUSB path)
        # Both default to None; FX2Camera's streaming start/stop code sets
        # and clears them in lockstep with its own handles.
        # NOTE: losing the winusb branch was the bug in 4.0.0-LVCtest vs
        # the LVC upstream — restoring it here per the LVC reference.
        self._iso_handle_for_ctrl = None
        self._winusb_reader_for_ctrl = None

        try:
            self._connect()
        except Exception:
            self._teardown()
            raise

    # -- connection ---------------------------------------------------------

    def _connect(self):
        """Find the device, upload firmware if needed, claim interface."""
        dev = usb.core.find(idVendor=VID, idProduct=PID_APP)
        if dev is not None:
            self._dev = dev
            self._setup_device()
            logger.info(
                '[FX2 Conn  ] device found running firmware (PID 0x%04X)',
                PID_APP,
            )
            return

        dev = usb.core.find(idVendor=VID, idProduct=PID_BOOT)
        if dev is None:
            raise RuntimeError(
                f"No Lumascope FX2 device found "
                f"(checked PID 0x{PID_APP:04X} and 0x{PID_BOOT:04X})"
            )

        logger.info(
            '[FX2 Conn  ] bootloader found (PID 0x%04X), uploading firmware...',
            PID_BOOT,
        )
        self._upload_firmware(dev, self._find_firmware_path())

        # Wait for re-enumeration under the new application PID.
        deadline = time.monotonic() + self.FIRMWARE_RE_ENUM_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(0.5)
            dev = usb.core.find(idVendor=VID, idProduct=PID_APP)
            if dev is not None:
                self._dev = dev
                self._setup_device()
                logger.info(
                    '[FX2 Conn  ] firmware loaded, re-enumerated as PID 0x%04X',
                    PID_APP,
                )
                return

        raise RuntimeError(
            f"FX2 did not re-enumerate after firmware upload "
            f"(waited {self.FIRMWARE_RE_ENUM_TIMEOUT:.0f}s)"
        )

    def _setup_device(self):
        """Detach kernel driver, configure, claim interface 0."""
        dev = self._dev

        # On macOS/Linux, detach the kernel driver if it grabbed the
        # interface. Windows pyusb raises NotImplementedError here — ignore.
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

        logger.info('[FX2 Conn  ] USB device configured, interface 0 claimed')

    def _find_firmware_path(self) -> str:
        """Locate the FX2 firmware hex file in a PyInstaller bundle or source tree.

        Two hex files ship with the driver:

        - ``LumascopeClassic.hex`` (45068 bytes) — **patched** variant
          with a modified product string ("LS Classic") to improve
          device enumeration after firmware upload. Confirmed identical
          (SHA256 ``c15a9294…``) to ``LumascopeClassic_patched.hex`` in
          the ``LumaviewClassic`` development repo. This is the primary
          production firmware.
        - ``Lumascope600.hex`` (28434 bytes) — original smaller firmware
          (AUTOIN=1024). Kept as a fallback for any unit that refuses
          to re-enumerate with the patched variant.

        History note: the ``4.0.0-LVCtest`` integration branch in LVP
        shipped the **unpatched** ``LumascopeClassic_original.hex``
        (SHA256 ``4d457a86…``) under the name ``LumascopeClassic.hex``
        — a packaging mismatch that the LVC upstream later corrected.
        Stage 3 of the 4.1.0-dev port intentionally copies the patched
        variant from ``LumaviewClassic/firmware/LumascopeClassic.hex``,
        NOT from the ``4.0.0-LVCtest`` branch. If this file is ever
        re-copied from anywhere, verify its SHA256 matches.

        Search order (first hit wins):
          1. ``<base>/firmware/LumascopeClassic.hex``   (patched — preferred)
          2. ``<base>/firmware/Lumascope600.hex``       (original — fallback)
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
            f"FX2 firmware hex file not found. Searched: {', '.join(candidates)}"
        )

    def _upload_firmware(self, dev, hex_path: str):
        """Upload Intel HEX firmware to FX2 via vendor request 0xA0."""
        data, end_addr = parse_intel_hex(hex_path)
        logger.info('[FX2 Conn  ] firmware: %s (%d bytes)', hex_path, end_addr)

        # Put 8051 into reset
        dev.ctrl_transfer(0x40, VR_ANCHOR_DLD, 0xE600, 0, b'\x01')

        # Send firmware data in chunks
        addr = 0
        chunk = self.FIRMWARE_CHUNK_SIZE
        while addr < end_addr:
            remaining = end_addr - addr
            length = min(chunk, remaining)
            dev.ctrl_transfer(
                0x40, VR_ANCHOR_DLD, addr, 0, data[addr:addr + length]
            )
            addr += length

        # Release 8051 from reset — firmware boots and the device re-enumerates
        dev.ctrl_transfer(0x40, VR_ANCHOR_DLD, 0xE600, 0, b'\x00')
        logger.info('[FX2 Conn  ] firmware upload complete, 8051 released')

    # -- control transfers --------------------------------------------------

    def control_transfer_out(
        self,
        request: int,
        value: int = 0,
        index: int = 0,
        data: bytes = b'',
        timeout: int = 5000,
    ):
        """Thread-safe vendor OUT control transfer.

        While streaming, the pyusb handle is closed — only one handle
        on the device at a time. Routes through whichever streaming
        handle is currently live (libusb1 on macOS/Linux, WinUSB reader
        on Windows), or the pyusb handle otherwise. Callers don't need
        to care which path is active.
        """
        with self._lock:
            if self._iso_handle_for_ctrl is not None:
                return self._iso_handle_for_ctrl.controlWrite(
                    0x40, request, value, index, data, timeout=timeout
                )
            if self._winusb_reader_for_ctrl is not None:
                return self._winusb_reader_for_ctrl.device.control_transfer(
                    0x40, request, value, index, data=data
                )
            return self._dev.ctrl_transfer(
                0x40, request, value, index, data, timeout=timeout
            )

    def control_transfer_in(
        self,
        request: int,
        value: int = 0,
        index: int = 0,
        length: int = 0,
        timeout: int = 5000,
    ):
        """Thread-safe vendor IN control transfer (same routing as OUT)."""
        with self._lock:
            if self._iso_handle_for_ctrl is not None:
                return self._iso_handle_for_ctrl.controlRead(
                    0xC0, request, value, index, length, timeout=timeout
                )
            if self._winusb_reader_for_ctrl is not None:
                return self._winusb_reader_for_ctrl.device.control_transfer(
                    0xC0, request, value, index, length=length
                )
            return self._dev.ctrl_transfer(
                0xC0, request, value, index, length, timeout=timeout
            )

    def i2c_write(self, addr: int, data):
        """Write bytes to the I2C bus via vendor request 0xB3.

        Returns the result of the underlying control transfer (number of
        bytes written from pyusb / libusb1). Callers that want to detect
        short writes (e.g., LED command diagnostics) can compare to
        `len(data)`. Pre-2026-04-15 this method discarded the result,
        which masked silent short-write failures in `_led_write`.
        """
        return self.control_transfer_out(
            VR_I2C_WRITE, value=0, index=addr, data=bytes(data)
        )

    def i2c_read(self, addr: int, length: int):
        """Read bytes from the I2C bus via vendor request 0xB2."""
        return self.control_transfer_in(VR_I2C_READ, value=0, index=addr, length=length)

    def sensor_reg_write(self, reg: int, value: int):
        """Write 16-bit value to an MT9P031 register via VR_I2C_WRITE (0xB3).

        Wire format matches LVC exactly (`AptinaMT9P031_Control.cs::Write`):
        3 bytes `[reg, high, low]` sent as a VR_I2C_WRITE control transfer
        with `index = I2C_SENSOR` (0x5d). The FX2 firmware's `VR_I2C_WRITEb3`
        handler (vendor_req_parse.c:129) parses `wIndexL` as the I2C address,
        `wLengthL` as the byte count, truncates to 3 bytes max, and writes
        the received payload to I2C without touching IFCLK.

        WARNING — do NOT route sensor writes through 0xBA
        (VR_IMAGE_SENSOR_CLK_MANAGED_WRITE). That variant switches IFCLK to
        internal, does the I2C write, then switches back. The GPIF pixel
        clock depends on IFCLK, so every 0xBA call disrupts streaming and
        produces visible image corruption on the next ISO frame. Our Stage
        3 port originally used 0xBA because the firmware comment for it
        says "sensor clock managed write" — a misleading name. LVC defines
        the constant but never calls it from any production code path; the
        real production path is plain VR_I2C_WRITE (0xB3). Fixed
        2026-04-15. See docs/AUDIT_FX2_RUNTIME.md Bug 6.
        """
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        data = bytes([reg, high, low])
        self.control_transfer_out(
            VR_I2C_WRITE, value=0, index=I2C_SENSOR, data=data
        )

    def sensor_reg_read(self, reg: int) -> int:
        """Read 16-bit value from MT9P031 register via vendor request 0xB4.

        The FX2 firmware processes this asynchronously: it sets a flag in the
        vendor request handler, the main loop does the I2C read, then sends
        the data back on EP0 IN. A generous 5s timeout covers the async gap.
        """
        result = self.control_transfer_in(
            VR_I2C_MT9P031_READ,
            value=reg,
            index=I2C_SENSOR,
            length=2,
            timeout=5000,
        )
        return (result[0] << 8) | result[1]

    def init_gpif(self):
        """Initialize GPIF. Required after pixel clock changes via VR_INIT_GPIF.

        WARNING: do NOT call this from ``FX2Camera._init_sensor`` after the
        PLL config write. The firmware's TD_Init() / Init_GPIF() /
        SetISOInterface() path disrupts the EP2 configuration. The clock-
        managed write (0xBA) already handles IFCLK switching without a
        separate init_gpif call.
        """
        self.control_transfer_out(VR_INIT_GPIF)

    def start_streaming(self):
        """Send vendor request to start image data output."""
        self.control_transfer_out(VR_START_STREAMING)

    def stop_streaming(self):
        """Send vendor request to stop image data output."""
        self.control_transfer_out(VR_STOP_STREAMING)

    def get_firmware_version(self) -> int:
        """Read 2-byte firmware version register."""
        result = self.control_transfer_in(VR_CODE_VERSION, length=2)
        return (result[0] << 8) | result[1]

    # -- bulk / alt-interface / low-level ----------------------------------

    def set_alt_interface(self, alt: int):
        """Switch USB alternate interface setting (0=bulk, 3=iso)."""
        with self._lock:
            self._dev.set_interface_altsetting(interface=0, alternate_setting=alt)
            # Clear any halt/stall on EP 0x82 after switching alt interface
            try:
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
            logger.info('[FX2 Conn  ] alt interface set to %d', alt)

    def clear_halt(self, endpoint: int = 0x82):
        """Clear halt/stall condition on an endpoint."""
        with self._lock:
            try:
                self._dev.clear_halt(endpoint)
            except usb.core.USBError as e:
                logger.debug('[FX2 Conn  ] clear_halt(0x%02X): %s', endpoint, e)

    def bulk_read(self, size: int, timeout: int = 1000):
        """Read from bulk endpoint 0x82. NOT locked — caller manages timing."""
        return self._dev.read(0x82, size, timeout=timeout)

    # -- teardown ----------------------------------------------------------

    def _teardown(self):
        """Release USB resources. Idempotent, swallows errors.

        Called from ``__init__`` on construction failure and from
        ``_reset_for_test``. Does NOT null out ``_instance`` — that's
        ``_reset_for_test``'s job.
        """
        if self._dev is not None:
            try:
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
            self._dev = None
        self._iso_handle_for_ctrl = None
        self._winusb_reader_for_ctrl = None

    def disconnect(self):
        """Public cleanup hook. Same as ``_teardown`` but named for callers."""
        self._teardown()


# ---------------------------------------------------------------------------
# _FX2ImageHandler — frame buffering (inherits LVP's ImageHandlerBase)
# ---------------------------------------------------------------------------

class _FX2ImageHandler(ImageHandlerBase):
    """Thread-safe frame buffer for FX2 camera.

    The LVC reference driver carried a standalone fallback implementation
    with its own ``_new``/``_failure_count`` state for running outside
    LVP. We drop that here — the 4.1.0-dev module only runs inside LVP,
    so ``ImageHandlerBase`` is always available and its behavior is what
    the rest of the camera stack expects.

    No overrides needed; the base class already implements
    ``_store_frame`` / ``get_last_image`` / ``_record_failure`` / ``reset``.
    """


# ---------------------------------------------------------------------------
# FX2Camera — Camera ABC implementation
# ---------------------------------------------------------------------------

@camera_registry.register('fx2', priority=80)
class FX2Camera(Camera):
    """Camera driver for Lumascope Classic (MT9P031 via FX2 USB).

    Registered at priority 80 (below pylon/ids/sim at 100) so LS820/LS850
    units with a Basler camera are preferred on auto-detect. On LS620/LS720
    where no pylon/ids camera is present, those drivers raise and the
    registry falls through to FX2.

    Shares its USB connection with ``FX2LEDController`` via the module-
    level ``_FX2Connection`` singleton — both drivers call
    ``_FX2Connection.get()`` in their constructors and end up pointing
    at the same handle without any coordination from Lumascope.__init__.
    """

    # How often to log streaming stats (seconds). Set to 0 to disable.
    STATS_LOG_INTERVAL = 10.0

    # Frame size bounds
    FRAME_SIZE_MIN = 100
    FRAME_SIZE_STEP = 4

    def __init__(self, **kwargs):
        # Grab the FX2 connection BEFORE super().__init__() — the Camera
        # base class calls self.connect() at the end of its __init__,
        # and that needs self._fx2 live. If _FX2Connection.get() raises
        # (no FX2 hardware, no pyusb, firmware upload fails), the
        # exception propagates and the registry falls through to the
        # next camera driver candidate.
        self._fx2 = _FX2Connection.get()

        # Streaming state — initialized here so connect() can see them
        # even though connect() runs inside super().__init__().
        self._grabbing = False
        self._grab_thread: threading.Thread | None = None
        self._width = IMG_WIDTH
        self._height = IMG_HEIGHT
        self._exposure_rows = 100
        self._gain_reg = 0x0008  # default = 1.0x = 0 dB
        self._pixel_format = 'Mono8'

        # Platform-specific streaming state (set by _start_*_streaming)
        self._use_iso = False
        self._use_winusb_iso = False
        self._iso_ctx = None
        self._iso_handle = None
        self._iso_transfers: list = []
        self._iso_buf = bytearray()
        self._iso_buf_lock = threading.Lock()
        self._usb_event_thread: threading.Thread | None = None
        self._bulk_reader_thread: threading.Thread | None = None
        self._winusb_reader = None

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
        # is still inside `handleEventsTimeout` — its next mutex lock
        # hits a destroyed mutex and crashes Python with
        # ``Assertion failed: pthread_mutex_destroy(mutex) == 0``
        # in libusb1's ``usbi_mutex_destroy``. Reproduced 2026-04-15
        # during Stage 3.5 hardware validation by a test script with
        # a wrong-arity unpacking error.
        #
        # Fix: atexit hook calls ``stop_grabbing`` before any Python
        # GC happens. atexit runs during normal interpreter shutdown
        # in LIFO order. The weakref ensures the hook doesn't pin the
        # camera object in memory — if user code releases its FX2Camera
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
                pass  # swallow — interpreter shutdown is in progress

        atexit.register(_atexit_drain)

    # -- Context manager support ------------------------------------------
    # `with FX2Camera() as cam:` ensures stop_grabbing + disconnect run
    # via __exit__ regardless of whether the body raises. This is the
    # primary recommended cleanup pattern for ad-hoc scripts and tests
    # — the atexit hook above is a safety net for code that doesn't use
    # the context manager (e.g., long-lived LVP UI sessions where the
    # camera is held by the Lumascope object for the lifetime of the
    # app, not inside a `with` block).

    def __enter__(self) -> 'FX2Camera':
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
        the camera profile, applies default exposure/gain via
        ``init_camera_config()``, and **starts ISO streaming**. The
        start-grabbing-in-connect convention matches PylonCamera
        (drivers/pyloncamera.py:191), IDSCamera (drivers/idscamera.py:76),
        and SimulatedCamera (drivers/simulated_camera.py:176). LVP's
        ScopeDisplay polls the camera assuming it's already grabbing; if
        connect() returns without starting streaming, the live view stays
        blank. This bug bit on the first LS620 GUI launch (2026-04-15) —
        manual Stage 3.5 scripts didn't notice because they all called
        cam.start_grabbing() explicitly.
        """
        self.model_name = 'MT9P031-LS620'
        self._init_sensor()
        self.cam_image_handler = _FX2ImageHandler()
        self._active = True
        self._load_profile()
        self._query_dynamic_capabilities()
        self.init_camera_config()
        self.start_grabbing()
        logger.info('[FX2 Cam   ] connected: %s', self.model_name)
        return True

    def disconnect(self) -> bool:
        self.stop_grabbing()
        self._active = None
        logger.info('[FX2 Cam   ] disconnected')
        return True

    def is_connected(self) -> bool:
        return self._active is not None and bool(self._active)

    def _query_dynamic_capabilities(self):
        """Populate profile's dynamic gain / exposure fields.

        FX2 has no SDK to query — these are hardcoded from the MT9P031
        datasheet and the driver's row-time constant. We fill them in
        here so the rest of LVP can read ``scope.capabilities`` or
        ``camera.profile.gain.total_max_db`` and get real numbers
        instead of the ``None`` defaults.
        """
        try:
            self.profile.gain.total_min_db = 0.0
            self.profile.gain.total_max_db = 42.1  # 128x, per audit-corrected math
            self.profile.exposure_min_us = _ROW_TIME_MS * 1000  # 1 row = 112.4 μs
            # Cap exposure at the legacy LVC 178 ms value (matches what
            # was known-safe in the original LumaviewClassic UI). The
            # MT9P031 register itself supports up to MAX_EXPOSURE_ROWS ×
            # row_time = 7,366 ms, BUT above the per-frame readout time
            # (~214 ms at 1900 rows × 0.1124 ms/row) the sensor inserts
            # vertical blanking rows to extend the frame period, which
            # changes the bytes/sec rate mid-stream and desyncs the FX2
            # frame parser. Visible as image corruption when the user
            # drags the exposure slider above ~200 ms. Raising this
            # requires fixing the frame parser to handle variable frame
            # timing OR doing stop-grab / set / start-grab on every
            # exposure change — both Stage 3.6+ work, both non-trivial.
            # Hardware-validated 2026-04-15 on the first LS620 GUI run.
            SAFE_EXPOSURE_MAX_MS = 178
            self.profile.exposure_max_us = SAFE_EXPOSURE_MAX_MS * 1000
            logger.debug(
                '[FX2 Cam   ] profile capabilities: gain 0.0-42.1 dB, '
                'exposure %.3f-%.3f ms',
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
        between writes is conservative — the sensor responds much faster
        but this matches the LVC reference that hardware-validated at
        63/63 frames.

        WARNING: do NOT call ``self._fx2.init_gpif()`` here. Per the
        firmware disassembly (see LumaviewClassic/docs/STREAMING_ANALYSIS.md
        §3.2), VR_INIT_GPIF calls TD_Init() → Init_GPIF() → SetISOInterface()
        internally, which resets EP2 configuration. The clock-managed
        write (0xBA) already handles IFCLK switching without calling
        init_gpif.
        """
        fx2 = self._fx2

        # Initial window — set a default BEFORE PLL config. Overwritten
        # by the set_frame_size() call at the end of this method.
        fx2.sensor_reg_write(REG_ROW_START, 0x0036)  # sensor default row_start
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_COL_START, 0x0010)  # sensor default col_start
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_ROW_SIZE, 0x0797)   # sensor default 1943
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_COL_SIZE, 0x0A1F)   # sensor default 2591
        time.sleep(0.01)

        # PLL power on
        fx2.sensor_reg_write(REG_PLL_CTRL, 0x0051)
        time.sleep(0.01)

        # PLL config: M=0x1B=27, N_divider=0x01, P1_divider=0x0D=13.
        # EXTCLK = 12 MHz → pixel_clock ≈ 24.92 MHz → ~4.5 fps at 1900×1900.
        # NOTE on register interpretation: the MT9P031 datasheet formula
        # says N = N_divider + 1 and P1 = P1_divider + 1, but the
        # working silicon uses the raw register values directly (M, N,
        # P1 as written). The VCO constraint (180-360 MHz) only passes
        # with raw interpretation (12*27/1 = 324 MHz), not with +1
        # (12*27/2 = 162 MHz). See
        # LumaviewClassic/docs/DATASHEET_VERIFICATION.md §1 for the full
        # audit. The comment used to say M=27/N=1/P1=13; we keep that
        # convention but note that it's raw-register math, not
        # datasheet-formula math.
        fx2.sensor_reg_write(REG_PLL_CFG1, 0x1B01)
        time.sleep(0.01)
        fx2.sensor_reg_write(0x12, 0x000D)  # PLL Config 2: P1_divider = 13
        time.sleep(0.01)

        # PLL activate
        fx2.sensor_reg_write(REG_PLL_CTRL, 0x0053)
        time.sleep(0.2)  # datasheet requires 1ms for VCO lock; 200ms is defensive
        # Do NOT call init_gpif() here — see docstring warning.

        # Blue-strip fix per MT9P031 developer guide (DG_A page 7).
        # Prevents a blue strip artifact when bright light hits the top
        # or bottom of the sensor array. Recommended even at slower
        # pixel clocks where it may not be strictly necessary.
        fx2.sensor_reg_write(0x7F, 0x0000)
        time.sleep(0.01)

        # Black level calibration
        fx2.sensor_reg_write(REG_BLC, 0x6000)        # lock green + red/blue BLC channels
        time.sleep(0.01)
        # Read Mode 2 bits we set:
        #   bit  6 (0x0040) — Row_BLC enabled (sensor default)
        #   bit 14 (0x4000) — Mirror_Column = horizontal flip. Per
        #                     Linux kernel mt9p031.c register defs.
        #                     LS620 optic path delivers a left/right-
        #                     reversed view through the eyepiece vs the
        #                     sensor's native readout; this bit corrects
        #                     it at the sensor (free, no CPU cost,
        #                     applies to live view + captures uniformly).
        # If the image ends up upside down instead of mirrored, swap
        # bit 14 → bit 15 (0x4000 → 0x8000) for Mirror_Row instead.
        fx2.sensor_reg_write(REG_READ_MODE2, 0x4040)
        time.sleep(0.01)
        fx2.sensor_reg_write(REG_ROW_BLACK, 0x0000)  # black target = 0 (microscopy optimization)
        time.sleep(0.01)

        # Set default window to full 1900×1900 — also configures the
        # col_size/row_size registers correctly with centering.
        self.set_frame_size(IMG_WIDTH, IMG_HEIGHT)

        logger.info('[FX2 Cam   ] MT9P031 sensor initialized (PLL + BLC)')

    # -- Streaming start / stop --------------------------------------------

    def start_grabbing(self):
        if self._grabbing:
            return
        self.stream_stats.reset()
        self._grabbing = True  # set BEFORE starting threads that check it
        self._use_winusb_iso = False
        self._use_iso = False

        if sys.platform == 'win32':
            # Windows: WinUSB native ISO API (not libusb1).
            self._use_winusb_iso = True
            self._start_winusb_iso_streaming()
        elif _HAS_USB1:
            # macOS / Linux: libusb1 async ISO.
            self._use_iso = True
            self._start_iso_streaming()
        else:
            # Fallback: bulk transfers. ~0.7 fps, useful only for bring-up.
            self._start_bulk_streaming()

        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def _start_iso_streaming(self):
        """macOS / Linux ISO path via python-libusb1."""
        # Close the pyusb handle — only one handle on the device at a time.
        try:
            usb.util.dispose_resources(self._fx2._dev)
        except Exception:
            pass

        self._iso_ctx = usb1.USBContext()
        self._iso_handle = self._iso_ctx.openByVendorIDAndProductID(VID, PID_APP)
        if self._iso_handle is None:
            raise RuntimeError('FX2 USB device disappeared before ISO streaming could start')
        try:
            if self._iso_handle.kernelDriverActive(0):
                self._iso_handle.detachKernelDriver(0)
        except Exception:
            pass
        self._iso_handle.claimInterface(0)
        self._iso_handle.setInterfaceAltSetting(0, ISO_ALT_INTERFACE)

        # Route control transfers through this handle while streaming —
        # the pyusb handle is closed, so the connection's normal
        # control_transfer_out/in path would fail without this swap.
        self._fx2._iso_handle_for_ctrl = self._iso_handle

        # Fresh buffer for the ISO callback to fill.
        with self._iso_buf_lock:
            self._iso_buf = bytearray()

        # Submit ISO transfers BEFORE sending VR_START_STREAMING. Transfers
        # must be pending when data starts flowing or the FIFO overflows
        # while we're still queuing up.
        self._iso_transfers = []
        for _ in range(ISO_NUM_TRANSFERS):
            xfer = self._iso_handle.getTransfer(iso_packets=ISO_NUM_PACKETS)
            xfer.setIsochronous(
                0x82,
                ISO_MAX_PACKET_SIZE * ISO_NUM_PACKETS,
                callback=self._iso_callback,
                timeout=5000,
                iso_transfer_length_list=[ISO_MAX_PACKET_SIZE] * ISO_NUM_PACKETS,
            )
            xfer.submit()
            self._iso_transfers.append(xfer)

        # USB event pump in a dedicated thread — libusb1 needs someone
        # to call handleEventsTimeout() to process ISO completions.
        self._usb_event_thread = threading.Thread(
            target=self._usb_event_loop, daemon=True
        )
        self._usb_event_thread.start()

        # Now start streaming — transfers are ready to receive data.
        self._iso_handle.controlWrite(0x40, VR_START_STREAMING, 0, 0, b'')

        logger.info(
            '[FX2 Cam   ] streaming started (ISO alt %d, EP 0x82, %d transfers × %d packets)',
            ISO_ALT_INTERFACE, ISO_NUM_TRANSFERS, ISO_NUM_PACKETS,
        )

    def _start_winusb_iso_streaming(self):
        """Windows ISO path via WinUSB native API."""
        from drivers.winusb_iso import WinUsbIsoReader

        # Close the pyusb handle — WinUSB needs exclusive device access.
        try:
            usb.util.dispose_resources(self._fx2._dev)
        except Exception:
            pass

        self._winusb_reader = WinUsbIsoReader(
            VID, PID_APP,
            pipe_id=0x82,
            alt_interface=ISO_ALT_INTERFACE,
            num_slots=ISO_NUM_TRANSFERS,
            packets_per_xfer=ISO_NUM_PACKETS,
        )
        self._winusb_reader.start()

        # Send VR_START_STREAMING through the WinUSB reader (can't use
        # the pyusb handle — it's closed).
        self._winusb_reader.device.control_transfer(0x40, VR_START_STREAMING, 0, 0)

        # Route control transfers through the WinUSB reader while
        # streaming. Restoring this branch (which the 4.0.0-LVCtest
        # integration branch had dropped) is the entire reason we went
        # back to the LVC upstream as the port source — without it,
        # any LED command or exposure/gain change during streaming
        # would fail on Windows.
        self._fx2._winusb_reader_for_ctrl = self._winusb_reader

        # Share the reader's data buffer with our grab loop.
        self._iso_buf = self._winusb_reader.data_buf
        self._iso_buf_lock = self._winusb_reader.data_lock

        logger.info(
            '[FX2 Cam   ] streaming started (WinUSB ISO alt %d, EP 0x82)',
            ISO_ALT_INTERFACE,
        )

    def _start_bulk_streaming(self):
        """Bulk fallback via pyusb. Tops out at ~0.7 fps on macOS — only
        useful for hardware bring-up on systems without libusb1 installed.
        """
        self._fx2.set_alt_interface(0)
        self._fx2.clear_halt(0x82)
        self._fx2.start_streaming()
        with self._iso_buf_lock:
            self._iso_buf = bytearray()

        self._bulk_reader_thread = threading.Thread(
            target=self._bulk_reader_loop, daemon=True
        )
        self._bulk_reader_thread.start()

        logger.info('[FX2 Cam   ] streaming started (bulk alt 0, EP 0x82) — fallback mode')

    def stop_grabbing(self):
        if not self._grabbing:
            return
        self._grabbing = False

        if self._use_winusb_iso:
            self._stop_winusb_iso_streaming()
        elif self._use_iso:
            self._stop_iso_streaming()
        else:
            self._stop_bulk_streaming()

        if self._grab_thread is not None:
            self._grab_thread.join(timeout=3.0)
            self._grab_thread = None

        s = self.stream_stats.summary()
        logger.info(
            '[FX2 Cam   ] streaming stopped: %d frames in %.1fs (%.1f fps avg), '
            '%d partial, %d USB errors, %.1f MB total',
            s['good_frames'], s['elapsed_s'], s['fps_average'],
            s['partial_frames'], s['usb_errors'], s['total_MB'],
        )

    def _stop_iso_streaming(self):
        """Stop libusb1 ISO streaming and restore the pyusb handle.

        Matches the LVC reference: cancel transfers, drain events for
        ~2s on the main thread, join the event thread, send STOP, close
        the handle. Hardware-validated at Stage 3.5.

        **Known robustness gap (Stage 3.6 followup):** if user code on
        the main thread raises an unhandled exception while streaming,
        Python interpreter shutdown will GC the libusb1 context while
        the daemon event thread is still inside handleEventsTimeout,
        which crashes with a libusb1 native ``pthread_mutex_destroy``
        assertion. The fix is an atexit hook (or context-manager
        ``__exit__``) on FX2Camera that calls ``stop_grabbing`` before
        the interpreter tears down threads. Tracked in TODO.
        """
        for xfer in self._iso_transfers:
            try:
                xfer.cancel()
            except Exception:
                pass

        # Drain cancelled transfers.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                self._iso_ctx.handleEventsTimeout(tv=0.1)
            except Exception:
                break

        if self._usb_event_thread is not None:
            self._usb_event_thread.join(timeout=3.0)
            self._usb_event_thread = None

        try:
            self._iso_handle.controlWrite(0x40, VR_STOP_STREAMING, 0, 0, b'')
        except Exception:
            pass
        try:
            self._iso_handle.releaseInterface(0)
            self._iso_handle.close()
        except Exception:
            pass
        self._iso_transfers = []
        self._iso_ctx = None
        self._iso_handle = None
        self._fx2._iso_handle_for_ctrl = None

        # Reopen the pyusb handle so control transfers work again.
        try:
            dev = usb.core.find(idVendor=VID, idProduct=PID_APP)
            if dev is not None:
                self._fx2._dev = dev
                self._fx2._setup_device()
        except Exception as e:
            logger.warning('[FX2 Cam   ] pyusb handle reopen failed: %s', e)

    def _stop_winusb_iso_streaming(self):
        """Stop WinUSB ISO streaming and restore the pyusb handle."""
        if self._winusb_reader is not None:
            try:
                self._winusb_reader.device.control_transfer(
                    0x40, VR_STOP_STREAMING, 0, 0
                )
            except Exception:
                pass
            self._winusb_reader.stop()
            self._winusb_reader = None
        self._fx2._winusb_reader_for_ctrl = None

        try:
            dev = usb.core.find(idVendor=VID, idProduct=PID_APP)
            if dev is not None:
                self._fx2._dev = dev
                self._fx2._setup_device()
        except Exception as e:
            logger.warning('[FX2 Cam   ] pyusb handle reopen failed: %s', e)

    def _stop_bulk_streaming(self):
        """Stop bulk streaming."""
        if self._bulk_reader_thread is not None:
            self._bulk_reader_thread.join(timeout=3.0)
            self._bulk_reader_thread = None
        try:
            self._fx2.stop_streaming()
        except Exception:
            pass

    def is_grabbing(self) -> bool:
        return (
            self._grabbing
            and self._grab_thread is not None
            and self._grab_thread.is_alive()
        )

    # -- Reader threads ----------------------------------------------------

    def _iso_callback(self, transfer):
        """libusb1 callback — called when an ISO transfer completes."""
        if transfer.getStatus() == usb1.TRANSFER_COMPLETED:
            with self._iso_buf_lock:
                for status, buf in transfer.iterISO():
                    if status == usb1.TRANSFER_COMPLETED and len(buf) > 0:
                        self._iso_buf.extend(buf)
        elif transfer.getStatus() == usb1.TRANSFER_CANCELLED:
            return
        # Resubmit for continuous streaming.
        if self._grabbing:
            try:
                transfer.submit()
            except Exception:
                pass

    def _bulk_reader_loop(self):
        """Read bulk EP 0x82 and feed into `_iso_buf` (fallback path)."""
        while self._grabbing:
            try:
                data = self._fx2.bulk_read(16384, timeout=1000)
                with self._iso_buf_lock:
                    self._iso_buf.extend(data)
            except _USBTimeoutError:
                continue
            except _USBError:
                if not self._grabbing:
                    break
                self.stream_stats.record_usb_error()
                time.sleep(0.01)

    def _usb_event_loop(self):
        """Pump libusb1 events in a dedicated thread."""
        while self._grabbing:
            try:
                self._iso_ctx.handleEventsTimeout(tv=0.1)
            except Exception:
                if not self._grabbing:
                    break

    # -- Grab loop ---------------------------------------------------------

    def _grab_loop(self):
        """Extract frames from the ISO / bulk data buffer.

        Audit fixes applied vs. the LVC reference (per
        LumaviewClassic/docs/OPTIMIZATION_ANALYSIS.md §8):
        - ``local_buf`` is explicitly initialized before the loop instead
          of relying on ``'local_buf' not in dir()`` (fragile, un-Pythonic).
        - The trim-after-prepend operation shares a single lock acquisition
          with the prepend instead of splitting into two `with` blocks
          (which could race against the ISO callback).
        """
        stats = self.stream_stats
        last_stats_log = time.monotonic()
        first_frame_logged = False
        local_buf: bytearray | None = None  # audit fix: explicit init

        while self._grabbing:
            # Re-read dimensions every iteration — the UI can call
            # set_frame_size() between frames.
            w = self._width
            h = self._height
            stride = w + 1
            skip_first_row = stride + 1
            needed = skip_first_row + h * stride

            with self._iso_buf_lock:
                if len(self._iso_buf) >= needed:
                    local_buf = self._iso_buf
                    self._iso_buf = bytearray()
                else:
                    local_buf = None

            if local_buf is None:
                time.sleep(0.005)
                continue

            stats.record_bytes(len(local_buf))

            # Scan for frame delimiters.
            buf = local_buf
            while True:
                idx = buf.find(FRAME_DELIM)
                if idx < 0:
                    # No complete frame — put unconsumed data back and
                    # trim if it's gotten out of hand. Single lock
                    # acquisition covers both (audit fix).
                    with self._iso_buf_lock:
                        self._iso_buf = buf + self._iso_buf
                        if len(self._iso_buf) > needed * 3:
                            self._iso_buf = self._iso_buf[-(needed * 2):]
                    break

                frame_data = buf[:idx]
                buf = buf[idx + len(FRAME_DELIM):]

                # Strict frame validation. The MT9P031 + FX2 GPIF emits
                # frames with EXACTLY one extra row of stride padding
                # beyond the math (`needed`). Measured 2026-04-15 on
                # 175 samples of clean streaming: 173/175 (98.9%) were
                # exactly `needed + stride` bytes, the other 2 were
                # corrupt (1 partial, 1 oversized). The +stride extra
                # is hardware-constant for fixed frame size; the
                # `as_strided` block below silently truncates it.
                #
                # PRE-FIX (AUDIT_FX2_RUNTIME.md): the check was
                # `len(frame_data) >= needed`, which silently accepted
                # arbitrary oversized frames as "good" and reshaped
                # them from a misaligned offset → visually corrupt
                # frames flagged as good, no telemetry. Stage 3.5 Phase
                # 8 missed this entirely because the partial-frame
                # counter only fires on undersize.
                #
                # POST-FIX: strict equality on `expected`. Anything
                # else is discarded, distinct shifted/partial counters
                # give honest telemetry on which failure mode dominates.
                # If frame size or readout config ever changes such
                # that the +stride invariant breaks, the shifted
                # counter will spike and we re-measure. See
                # docs/AUDIT_FX2_RUNTIME.md Fix 1.
                expected = needed + stride

                if len(frame_data) == expected:
                    raw = np.frombuffer(frame_data, dtype=np.uint8)
                    remaining = raw[skip_first_row:]
                    raw_2d = np.lib.stride_tricks.as_strided(
                        remaining, shape=(h, stride), strides=(stride, 1)
                    )
                    image = raw_2d[:, :w].copy()
                    self.cam_image_handler._store_frame(image, datetime.now())
                    stats.record_good_frame()

                    if not first_frame_logged:
                        first_frame_logged = True
                        logger.info(
                            '[FX2 Cam   ] first frame: %dx%d, stride=%d, '
                            '%d bytes, mean=%.1f',
                            w, h, stride, len(frame_data), float(image.mean()),
                        )
                elif len(frame_data) > needed:
                    # Wrong size but bigger than minimum — either
                    # oversized (missed delimiter, two frames glued)
                    # or sized between `needed` and `expected`
                    # (off-by-rows). Either way, the bytes are
                    # misaligned and would render as garbage.
                    stats.record_shifted_frame(len(frame_data))
                elif len(frame_data) > 0:
                    # Severely undersized — bytes dropped before the
                    # next delimiter was found.
                    stats.record_partial_frame(len(frame_data))

            # Periodic stats logging.
            now = time.monotonic()
            if self.STATS_LOG_INTERVAL > 0 and (now - last_stats_log) >= self.STATS_LOG_INTERVAL:
                last_stats_log = now
                s = stats.summary()
                logger.info(
                    '[FX2 Cam   ] stream: %.1f fps (avg %.2f), '
                    '%d good / %d partial / %d shifted, '
                    '%.1f MB/s, %d errors, %d timeouts',
                    s['fps_current'], s['fps_average'],
                    s['good_frames'], s['partial_frames'], s['shifted_frames'],
                    s['throughput_MBps'], s['usb_errors'], s['usb_timeouts'],
                )

    # -- Grab API (mostly inherits from Camera; override for clarity) ------

    def grab_new_capture(self, timeout: float = 5.0):
        """Block until a NEW frame arrives. Used by autofocus / protocols."""
        self.cam_image_handler.reset()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok, img, ts = self.cam_image_handler.get_last_image()
            if ok:
                with self._array_lock:
                    self.array = img
                return True, ts
            time.sleep(0.01)
        return False, None

    # -- Frame size --------------------------------------------------------

    def set_frame_size(self, w, h):
        """Set the sensor readout window.

        The sensor is configured to output (display + 1) × (display + 1)
        pixels. The extra column becomes a 0x00 sync byte between rows
        after GPIF processing; the extra row is discarded by the grab
        loop (``skip_first_row``). Dimensions are rounded down to
        multiples of FRAME_SIZE_STEP (4) and clamped to [100, 1900].
        """
        step = self.FRAME_SIZE_STEP
        w = max(self.FRAME_SIZE_MIN, min(IMG_WIDTH, int(w)))
        h = max(self.FRAME_SIZE_MIN, min(IMG_HEIGHT, int(h)))
        w = (w // step) * step
        h = (h // step) * step
        self._width = w
        self._height = h

        # Sensor registers want (display + 1) per LVC reference.
        sensor_w = w + 1
        sensor_h = h + 1
        # Center the window on the active pixel area (2592 × 1944 with
        # offsets 16 col / 54 row) and force even alignment.
        col_start = max(0, (2592 - sensor_w) // 2 + 16) & ~1
        row_start = max(0, (1944 - sensor_h) // 2 + 54) & ~1

        # Individual 3-byte writes — firmware truncates multi-byte I2C.
        self._fx2.sensor_reg_write(REG_ROW_START, row_start)
        self._fx2.sensor_reg_write(REG_COL_START, col_start)
        self._fx2.sensor_reg_write(REG_ROW_SIZE, sensor_h)
        self._fx2.sensor_reg_write(REG_COL_SIZE, sensor_w)

        # Flush the ISO buffer — data captured with the old window is
        # now misaligned and would desync the frame parser.
        with self._iso_buf_lock:
            self._iso_buf = bytearray()

        logger.info(
            '[FX2 Cam   ] frame size %dx%d (sensor %dx%d, row_start=%d, col_start=%d)',
            w, h, sensor_w, sensor_h, row_start, col_start,
        )

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

    def exposure_t(self, t):
        """Set exposure time in milliseconds.

        Formula from MT9P031 datasheet DS_F p31:
            tEXP = SW × tROW - SO × 2 × tPIXCLK
        Inverted:
            SW = (tEXP + SO_ms) / tROW_ms

        NOTE on accuracy: ``_ROW_TIME_MS = 0.1124`` assumes EXTCLK=12 MHz.
        The LVC OPTIMIZATION_ANALYSIS doc measured actual throughput
        and computed EXTCLK ≈ 7.6 MHz instead, which would put the row
        time at 0.1205 ms (7% higher). Hardware validation passed with
        0.1124 ms so we keep it, but precise exposure calibration for
        brightness-matched captures may be ±7% off. Stage 3.5 bench
        work can measure row time directly with a pulsed reference
        LED and a known-duration trigger.

        NOTE on effect timing: MT9P031 has a 2-frame pipeline delay
        between writing the shutter width register and seeing the new
        exposure in output frames. Callers that depend on exact timing
        (autofocus, protocol captures) must wait ≥2 frames after an
        exposure change before relying on the new value.
        """
        target_ms = float(t)
        rows = max(
            1,
            min(
                MAX_EXPOSURE_ROWS,
                round((target_ms + _SHUTTER_OVERHEAD_MS) / _ROW_TIME_MS),
            ),
        )
        self._exposure_rows = rows
        self._fx2.sensor_reg_write(REG_EXPOSURE, rows)

    def get_exposure_t(self):
        return max(0.0, self._exposure_rows * _ROW_TIME_MS - _SHUTTER_OVERHEAD_MS)

    def auto_exposure_t(self, state=True):
        pass  # MT9P031 has no hardware auto-exposure

    # -- Gain --------------------------------------------------------------

    def gain(self, g):
        """Set gain in dB. Clamped to [0.0, 42.1] (audit-corrected max)."""
        db = max(0.0, min(42.1, float(g)))
        reg = _gain_db_to_register(db)
        self._gain_reg = reg
        self._fx2.sensor_reg_write(REG_GLOBAL_GAIN, reg)

    def get_gain(self):
        _, db = _register_to_gain_db(self._gain_reg)
        return db

    def auto_gain(self, state=True, target_brightness=0.5, min_gain=None, max_gain=None):
        pass  # no hardware auto-gain

    def auto_gain_once(self, state=True, target_brightness=0.5, min_gain=None, max_gain=None):
        pass

    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        pass

    def update_auto_gain_min_max(self, min_gain=None, max_gain=None):
        pass

    # -- Misc (no-op or trivial) ------------------------------------------

    def init_camera_config(self):
        """Apply sensible defaults for exposure and gain on startup."""
        self.exposure_t(50.0)  # 50 ms default — typical microscopy starting point
        self.gain(0.0)         # 0 dB = 1x gain

    def find_model_name(self):
        self.model_name = 'MT9P031-LS620'

    def get_all_temperatures(self) -> dict:
        return {}  # MT9P031 has no temperature sensor

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0):
        pass  # Frame rate is determined by PLL / exposure, not a software cap

    def set_binning_size(self, size: int) -> bool:
        return size == 1  # only 1×1 supported in this port

    def get_binning_size(self) -> int:
        return 1

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        pass  # MT9P031 has a test pattern register but it's not wired up


# ---------------------------------------------------------------------------
# FX2LEDController — thin command translator, no state
# ---------------------------------------------------------------------------

@led_registry.register('fx2', priority=80)
class FX2LEDController:
    """LED controller for Lumascope Classic via FX2 I2C at address 0x2A.

    **Thin command translator.** The LVC reference carried a ``led_ma``
    dict and client-side state tracking (``get_led_ma`` / ``is_led_on`` /
    etc. read back from the dict). That existed because the pre-4.1 GUI
    owned LED state. In 4.1 the API owns state via
    ``Lumascope._led_owners`` / ``save_led_state`` / ``restore_led_state``,
    so this driver drops all state bookkeeping. The LEDBoardProtocol
    state-query methods still exist (the protocol requires them) but
    return sentinel defaults matching NullLEDBoard — the real truth
    lives above the driver layer.

    **LED channel → I2C ASCII byte mapping** lives in this class only:
    LVP integer channels 0/1/2/3 → ASCII bytes 0x43/0x42/0x41/0x44
    ('C'/'B'/'A'/'D') which the FX2 peripheral controller uses on the
    wire. Never leak the ASCII form above the driver — that's a
    project-memory directive.

    **Fast variants.** The LED protocol requires ``led_on_fast`` /
    ``led_off_fast`` / ``leds_off_fast`` for time-critical toggling.
    These FX2 variants are IDENTICAL to the normal versions — the I2C
    writes have no serial-handshake latency to skip, so "fast" is the
    same as "normal".
    """

    # Channel count is fixed at 4 by the hardware. Per project memory,
    # per-model LED presence (e.g. LS560 having only BF + Green) is
    # handled via scopes.json Layers filtering, NOT by hiding channels
    # here. The driver reports what the FX2 protocol supports.
    _COLOR_TO_CH = _COLOR_TO_CH  # module-level dict
    _CH_TO_COLOR = _CH_TO_COLOR

    # Max mA per channel — used to scale (0, _max_ma) → (0, 255)
    # brightness. 200 mA is approximate; real max depends on LED model
    # and series resistor value. Not safety-critical because the FX2
    # hardware enforces its own current limit via the peripheral
    # controller.
    _MAX_MA = 200

    def __init__(self, **kwargs):
        # Grab the singleton — raises if no FX2 hardware, registry
        # fallthrough handles that case cleanly.
        self._fx2 = _FX2Connection.get()
        self._enabled = True

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
        short-write would have masked Bug 2 in AUDIT_FX2_RUNTIME.md
        (2026-04-15); propagating the return value catches it at the
        driver layer instead.

        NOTE: the 3× 10 ms delays (30 ms total per LED command) are
        the known root cause of the slider-corruption effect documented
        in project memory. During streaming, each LED command holds
        the FX2 USB connection for 30 ms, during which ISO data keeps
        arriving but cannot be drained. If the UI sends dozens of LED
        writes per second (Kivy slider drags), the ISO buffer can
        desync the frame parser. Stage 3.5 will verify and fix —
        leading candidate is a debounce on the UI side (16 ms minimum
        between writes).
        """
        i2c_channel = _CH_TO_I2C.get(channel, channel)
        # TEMP diag: byte-level trace of FX2 LED I2C writes. Bench
        # investigation of slider > ~150 mA failing to light LED on LS620.
        # Revert together with the led_on diag below once data collected.
        logger.info(
            '[FX2 LED diag] _led_write ch=%d i2c_ch=0x%02X brightness=0x%02X',
            channel, i2c_channel, brightness,
        )
        writes = [
            (0xFF, 'preamble'),
            (i2c_channel, 'channel'),
            (brightness & 0xFF, 'brightness'),
        ]
        for byte_val, label in writes:
            try:
                result = self._fx2.i2c_write(I2C_LED, [byte_val])
            except Exception as e:
                logger.error(
                    '[FX2 LED  ] i2c_write raised on %s byte=0x%02x '
                    '(ch=%d mA_equiv=%d): %s: %s',
                    label, byte_val, channel, brightness,
                    type(e).__name__, e,
                )
                raise
            if result is not None and result != 1:
                logger.warning(
                    '[FX2 LED  ] short write on %s byte=0x%02x '
                    '(ch=%d mA_equiv=%d): wrote %r of 1 byte expected',
                    label, byte_val, channel, brightness, result,
                )
            time.sleep(0.01)

    def _ma_to_brightness(self, mA) -> int:
        """Convert mA to 0-255 brightness value."""
        return max(0, min(255, round(float(mA) * 255.0 / self._MAX_MA)))

    # -- Channel discovery (B3) --------------------------------------------

    def available_channels(self) -> tuple:
        return tuple(self._COLOR_TO_CH.values())  # (0, 1, 2, 3)

    def available_colors(self) -> tuple:
        return tuple(self._COLOR_TO_CH.keys())    # ('Blue', 'Green', 'Red', 'BF')

    def color2ch(self, color: str) -> int:
        return self._COLOR_TO_CH.get(color, -1)

    def ch2color(self, channel: int) -> str:
        return self._CH_TO_COLOR.get(channel, '')

    # -- Core LED control --------------------------------------------------

    def led_on(self, channel: int, mA: int, block: bool = False, timeout: float = 5.0):
        if not self._enabled:
            return
        brightness = self._ma_to_brightness(mA)
        # TEMP diag: capture mA value + type at driver entry. Investigating
        # slider > ~150 mA LED-dark report on LS620. Revert with the
        # _led_write diag above.
        logger.info(
            '[FX2 LED diag] led_on ch=%d mA=%r type=%s -> brightness=%d (0x%02X)%s',
            channel, mA, type(mA).__name__, brightness, brightness,
            ' PREAMBLE-COLLISION' if brightness == 0xFF else '',
        )
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

    # -- Fast variants (same as normal — I2C has no serial handshake) -----

    def led_on_fast(self, channel: int, mA: int):
        self.led_on(channel, mA)

    def led_off_fast(self, channel: int):
        self.led_off(channel)

    def leds_off_fast(self):
        self.leds_off()

    # -- State queries (sentinel defaults — real state is in API) ---------
    # These methods exist because LEDBoardProtocol requires them. The
    # driver has no idea what's currently lit — that's owned by
    # Lumascope._led_owners. Callers should read state through the API
    # (scope.get_led_state(color)), never by reaching into the driver.

    def get_led_ma(self, color: str) -> int:
        return -1

    def is_led_on(self, color: str) -> bool:
        return False

    def get_led_state(self, color: str) -> dict:
        return {'enabled': False, 'illumination': -1}

    def get_led_states(self) -> dict:
        return {c: {'enabled': False, 'illumination': -1} for c in self._COLOR_TO_CH}

    # -- Connection no-ops (USB owned by _FX2Connection) ------------------

    def connect(self):
        pass

    def disconnect(self):
        pass

    def is_connected(self) -> bool:
        return self._fx2 is not None

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


