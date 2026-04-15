"""
WinUSB isochronous transfer support via ctypes.

Mirrors the C# ReadISOStream_WinUsb implementation exactly.
Windows only — uses winusb.dll, setupapi.dll, kernel32.dll directly.
"""

import ctypes
import ctypes.wintypes as wt
import logging
import struct
import threading
import time
from ctypes import (
    POINTER, Structure, byref, c_bool, c_byte, c_ubyte, c_uint, c_ulong,
    c_ushort, c_void_p, create_string_buffer, sizeof, windll,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENERIC_WRITE = 0x40000000
GENERIC_READ = 0x80000000
FILE_SHARE_WRITE = 0x02
FILE_SHARE_READ = 0x01
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_OVERLAPPED = 0x40000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
ERROR_IO_PENDING = 997
DIGCF_PRESENT = 0x02
DIGCF_DEVICEINTERFACE = 0x10
SPDRP_HARDWAREID = 1

# WinUSB GUID (standard for WinUSB devices)
GUID_DEVINTERFACE_USB_DEVICE = (0xA5DCBF10, 0x6530, 0x11D2,
    (0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED))


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
class GUID(Structure):
    _fields_ = [
        ('Data1', c_ulong),
        ('Data2', c_ushort),
        ('Data3', c_ushort),
        ('Data4', c_ubyte * 8),
    ]

class SP_DEVICE_INTERFACE_DATA(Structure):
    _fields_ = [
        ('cbSize', c_ulong),
        ('InterfaceClassGuid', GUID),
        ('Flags', c_ulong),
        ('Reserved', c_void_p),
    ]

class SP_DEVICE_INTERFACE_DETAIL_DATA(Structure):
    _fields_ = [
        ('cbSize', c_ulong),
        ('DevicePath', ctypes.c_wchar * 260),
    ]

class SP_DEVINFO_DATA(Structure):
    _fields_ = [
        ('cbSize', c_ulong),
        ('ClassGuid', GUID),
        ('DevInst', c_ulong),
        ('Reserved', c_void_p),
    ]

class OVERLAPPED(Structure):
    _fields_ = [
        ('Internal', c_void_p),
        ('InternalHigh', c_void_p),
        ('Offset', c_ulong),
        ('OffsetHigh', c_ulong),
        ('hEvent', c_void_p),
    ]

class WINUSB_PIPE_INFORMATION(Structure):
    _fields_ = [
        ('PipeType', c_ulong),
        ('PipeId', c_ubyte),
        ('MaximumPacketSize', c_ushort),
        ('Interval', c_ubyte),
    ]

class WINUSB_SETUP_PACKET(Structure):
    _fields_ = [
        ('RequestType', c_ubyte),
        ('Request', c_ubyte),
        ('Value', c_ushort),
        ('Index', c_ushort),
        ('Length', c_ushort),
    ]

class USBD_ISO_PACKET_DESCRIPTOR(Structure):
    _fields_ = [
        ('Offset', c_ulong),
        ('Length', c_ulong),
        ('Status', c_ulong),
    ]


# ---------------------------------------------------------------------------
# DLL loading
# ---------------------------------------------------------------------------
kernel32 = windll.kernel32
setupapi = windll.setupapi
winusb = windll.winusb


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------
def find_device_path(vid, pid):
    """Find the WinUSB device path for a given VID/PID."""
    guid = GUID(*GUID_DEVINTERFACE_USB_DEVICE)
    dev_info = setupapi.SetupDiGetClassDevsW(
        byref(guid), None, None,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)

    if dev_info == INVALID_HANDLE_VALUE:
        return None

    vid_pid = "vid_%04x&pid_%04x" % (vid, pid)
    iface_data = SP_DEVICE_INTERFACE_DATA()
    iface_data.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA)
    idx = 0

    while setupapi.SetupDiEnumDeviceInterfaces(
            dev_info, None, byref(guid), idx, byref(iface_data)):
        idx += 1

        detail = SP_DEVICE_INTERFACE_DETAIL_DATA()
        detail.cbSize = 8 if sizeof(c_void_p) == 8 else 6  # 64-bit vs 32-bit
        devinfo = SP_DEVINFO_DATA()
        devinfo.cbSize = sizeof(SP_DEVINFO_DATA)
        required = c_ulong(0)

        setupapi.SetupDiGetDeviceInterfaceDetailW(
            dev_info, byref(iface_data), byref(detail),
            sizeof(detail), byref(required), byref(devinfo))

        path = detail.DevicePath
        if vid_pid in path.lower():
            setupapi.SetupDiDestroyDeviceInfoList(dev_info)
            return path

    setupapi.SetupDiDestroyDeviceInfoList(dev_info)
    return None


# ---------------------------------------------------------------------------
# WinUSB device wrapper
# ---------------------------------------------------------------------------
class WinUsbDevice:
    """Wraps a WinUSB device handle for control transfers and ISO streaming."""

    def __init__(self, vid, pid):
        self.path = find_device_path(vid, pid)
        if self.path is None:
            raise RuntimeError(
                "WinUSB device not found (VID=0x%04X PID=0x%04X)" % (vid, pid))

        # Open device file
        self._file_handle = kernel32.CreateFileW(
            self.path,
            GENERIC_WRITE | GENERIC_READ,
            FILE_SHARE_WRITE | FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            None)

        if self._file_handle == INVALID_HANDLE_VALUE:
            raise RuntimeError("CreateFile failed: %d" % kernel32.GetLastError())

        # Initialize WinUSB
        self._iface_handle = c_void_p()
        if not winusb.WinUsb_Initialize(self._file_handle, byref(self._iface_handle)):
            kernel32.CloseHandle(self._file_handle)
            raise RuntimeError("WinUsb_Initialize failed: %d" % kernel32.GetLastError())

        log.info("WinUSB device opened: %s", self.path)

    def close(self):
        if self._iface_handle:
            winusb.WinUsb_Free(self._iface_handle)
            self._iface_handle = None
        if self._file_handle:
            kernel32.CloseHandle(self._file_handle)
            self._file_handle = None

    @property
    def handle(self):
        return self._iface_handle

    def set_alt_interface(self, setting):
        if not winusb.WinUsb_SetCurrentAlternateSetting(self._iface_handle, c_ubyte(setting)):
            raise RuntimeError("SetAlternateSetting(%d) failed: %d" % (setting, kernel32.GetLastError()))

    def query_pipe(self, alt, index):
        pipe_info = WINUSB_PIPE_INFORMATION()
        if not winusb.WinUsb_QueryPipe(self._iface_handle, c_ubyte(alt), c_ubyte(index), byref(pipe_info)):
            raise RuntimeError("QueryPipe failed: %d" % kernel32.GetLastError())
        return pipe_info

    def control_transfer(self, request_type, request, value, index, data=b'', length=0):
        pkt = WINUSB_SETUP_PACKET()
        pkt.RequestType = request_type
        pkt.Request = request
        pkt.Value = value
        pkt.Index = index

        transferred = c_ulong(0)
        if request_type & 0x80:  # IN transfer
            buf = create_string_buffer(length)
            pkt.Length = length
            winusb.WinUsb_ControlTransfer(
                self._iface_handle, pkt, buf, length, byref(transferred), None)
            return bytes(buf[:transferred.value])
        else:  # OUT transfer
            pkt.Length = len(data)
            buf = ctypes.create_string_buffer(data) if data else None
            winusb.WinUsb_ControlTransfer(
                self._iface_handle, pkt,
                buf if buf else None,
                len(data),
                byref(transferred), None)

    def abort_pipe(self, pipe_id):
        winusb.WinUsb_AbortPipe(self._iface_handle, c_ubyte(pipe_id))


# ---------------------------------------------------------------------------
# ISO streaming slot
# ---------------------------------------------------------------------------
class IsoSlot:
    """One async ISO transfer buffer + overlapped structure."""

    def __init__(self, iface_handle, pipe_id, buffer_size, packet_count):
        self.buffer = (c_ubyte * buffer_size)()
        self.buffer_size = buffer_size
        self.packet_count = packet_count
        self.packets = (USBD_ISO_PACKET_DESCRIPTOR * packet_count)()
        self.submitted = False

        # Create manual-reset event
        self.event = kernel32.CreateEventW(None, True, False, None)
        self.overlapped = OVERLAPPED()
        self.overlapped.hEvent = self.event

        # Register isoch buffer
        self.isoch_handle = c_void_p()
        ok = winusb.WinUsb_RegisterIsochBuffer(
            iface_handle,
            c_ubyte(pipe_id),
            ctypes.cast(self.buffer, c_void_p),
            c_ulong(buffer_size),
            byref(self.isoch_handle))

        if not ok:
            raise RuntimeError("WinUsb_RegisterIsochBuffer failed: %d" % kernel32.GetLastError())

    def dispose(self):
        if self.isoch_handle:
            winusb.WinUsb_UnregisterIsochBuffer(self.isoch_handle)
            self.isoch_handle = None
        if self.event:
            kernel32.CloseHandle(self.event)
            self.event = None


# ---------------------------------------------------------------------------
# ISO stream reader
# ---------------------------------------------------------------------------
class WinUsbIsoReader:
    """Reads isochronous data from a WinUSB device.

    Mirrors the C# ReadISOStream_WinUsb implementation.
    Runs in a dedicated thread; fills a shared bytearray.
    """

    def __init__(self, vid, pid, pipe_id=0x82, alt_interface=3,
                 num_slots=16, packets_per_xfer=256):
        self.vid = vid
        self.pid = pid
        self.pipe_id = pipe_id
        self.alt_interface = alt_interface
        self.num_slots = num_slots
        self.packets_per_xfer = packets_per_xfer

        self._dev = None
        self._running = False
        self._thread = None
        self.data_buf = bytearray()
        self.data_lock = threading.Lock()

    def start(self):
        """Open device, configure for ISO, start streaming thread."""
        self._dev = WinUsbDevice(self.vid, self.pid)
        self._dev.set_alt_interface(self.alt_interface)

        pipe_info = self._dev.query_pipe(self.alt_interface, 0)
        self._max_packet_size = pipe_info.MaximumPacketSize
        log.info("ISO pipe: PipeId=0x%02X, MaxPacketSize=%d",
                 pipe_info.PipeId, pipe_info.MaximumPacketSize)

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop streaming and close device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._dev:
            self._dev.abort_pipe(self.pipe_id)
            self._dev.close()
            self._dev = None

    @property
    def device(self):
        return self._dev

    def _read_loop(self):
        """Main ISO read loop — runs in dedicated thread."""
        dev = self._dev
        iface = dev.handle
        buffer_size = self._max_packet_size * self.packets_per_xfer
        n_packets = self.packets_per_xfer

        # Allocate slots
        slots = []
        try:
            for i in range(self.num_slots):
                slots.append(IsoSlot(iface, self.pipe_id, buffer_size, n_packets))

            # Prime: submit all slots
            continue_stream = False
            for slot in slots:
                self._submit_read(slot, buffer_size, n_packets, continue_stream)
                continue_stream = True

            # Round-robin: wait → extract → resubmit
            next_slot = 0
            while self._running:
                slot = slots[next_slot]

                transferred = c_ulong(0)
                ok = winusb.WinUsb_GetOverlappedResult(
                    iface,
                    byref(slot.overlapped),
                    byref(transferred),
                    True)  # wait=True

                if ok and transferred.value > 0:
                    # Extract data from ISO packets
                    for i in range(slot.packet_count):
                        pkt = slot.packets[i]
                        if pkt.Length > 0:
                            start = pkt.Offset
                            end = start + pkt.Length
                            chunk = bytes(slot.buffer[start:end])
                            with self.data_lock:
                                self.data_buf.extend(chunk)

                # Reset event and resubmit
                kernel32.ResetEvent(slot.event)
                if self._running:
                    self._submit_read(slot, buffer_size, n_packets, True)

                next_slot = (next_slot + 1) % self.num_slots

        finally:
            # Drain outstanding IO
            dev.abort_pipe(self.pipe_id)
            for slot in slots:
                if slot.submitted:
                    dummy = c_ulong(0)
                    winusb.WinUsb_GetOverlappedResult(
                        iface, byref(slot.overlapped), byref(dummy), True)
            for slot in slots:
                slot.dispose()

    def _submit_read(self, slot, length, packet_count, continue_stream):
        ok = winusb.WinUsb_ReadIsochPipeAsap(
            slot.isoch_handle,
            c_ulong(0),         # offset
            c_ulong(length),
            c_bool(continue_stream),
            c_ulong(packet_count),
            ctypes.cast(slot.packets, c_void_p),
            byref(slot.overlapped))

        if not ok:
            err = kernel32.GetLastError()
            if err != ERROR_IO_PENDING:
                log.warning("WinUsb_ReadIsochPipeAsap failed: %d", err)
                slot.submitted = False
                return False
        slot.submitted = True
        return True
