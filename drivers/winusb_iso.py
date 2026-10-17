"""
WinUSB isochronous transfer support via ctypes.

Mirrors the C# ReadISOStream_WinUsb implementation exactly.
Windows only -- uses winusb.dll, setupapi.dll, kernel32.dll directly.
"""

import ctypes
import logging
import threading
from ctypes import (
    Structure,
    byref,
    c_bool,
    c_int,
    c_ubyte,
    c_ulong,
    c_ushort,
    c_void_p,
    c_wchar_p,
    create_string_buffer,
    sizeof,
)

# Windows only: off Windows this import raises ImportError, which the test
# conftest's host-absent stand-in relies on.
from ctypes import WinDLL


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
# What GetLastError reports when a transfer's pipe timeout expires.
ERROR_SEM_TIMEOUT = 121
# What SetupDiEnumDeviceInterfaces reports past the last interface.
ERROR_NO_MORE_ITEMS = 259
DIGCF_PRESENT = 0x02
DIGCF_DEVICEINTERFACE = 0x10
SPDRP_HARDWAREID = 1

# WinUSB GUID (standard for WinUSB devices)
GUID_DEVINTERFACE_USB_DEVICE = (
    0xA5DCBF10,
    0x6530,
    0x11D2,
    (0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED),
)


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


# A packet descriptor's Status after a successful transfer (usb.h).
USBD_STATUS_SUCCESS = 0


class USBD_ISO_PACKET_DESCRIPTOR(Structure):
    _fields_ = [
        ('Offset', c_ulong),
        ('Length', c_ulong),
        ('Status', c_ulong),
    ]


# ---------------------------------------------------------------------------
# DLL loading
# ---------------------------------------------------------------------------
# use_last_error: each call's last error is saved as it returns, and
# ctypes.get_last_error() reads that copy; GetLastError read later sees
# whatever the interpreter set in between.
kernel32 = WinDLL('kernel32', use_last_error=True)
setupapi = WinDLL('setupapi', use_last_error=True)
winusb = WinDLL('winusb', use_last_error=True)

# Every function declares its types. Undeclared, ctypes passes and returns
# C ints, so a 64-bit handle (a device-information set, a file, an event)
# is cut to 32 bits both ways and INVALID_HANDLE_VALUE never compares equal.
# Pointers are declared c_void_p, which takes byref(), an array or None.
_BOOL = c_int
_HANDLE = c_void_p
_POINTER = c_void_p


def _declare(function, restype, *argtypes):
    function.restype = restype
    function.argtypes = list(argtypes)


_declare(setupapi.SetupDiGetClassDevsW, _HANDLE, _POINTER, c_wchar_p, _HANDLE, c_ulong)
_declare(
    setupapi.SetupDiEnumDeviceInterfaces, _BOOL, _HANDLE, _POINTER, _POINTER, c_ulong, _POINTER
)
_declare(
    setupapi.SetupDiGetDeviceInterfaceDetailW,
    _BOOL,
    _HANDLE,
    _POINTER,
    _POINTER,
    c_ulong,
    _POINTER,
    _POINTER,
)
_declare(setupapi.SetupDiDestroyDeviceInfoList, _BOOL, _HANDLE)
_declare(
    kernel32.CreateFileW, _HANDLE, c_wchar_p, c_ulong, c_ulong, _POINTER, c_ulong, c_ulong, _HANDLE
)
_declare(kernel32.CloseHandle, _BOOL, _HANDLE)
_declare(kernel32.CreateEventW, _HANDLE, _POINTER, _BOOL, _BOOL, c_wchar_p)
_declare(kernel32.ResetEvent, _BOOL, _HANDLE)
_declare(winusb.WinUsb_Initialize, _BOOL, _HANDLE, _POINTER)
_declare(winusb.WinUsb_Free, _BOOL, _HANDLE)
_declare(winusb.WinUsb_SetCurrentAlternateSetting, _BOOL, _HANDLE, c_ubyte)
_declare(winusb.WinUsb_QueryPipe, _BOOL, _HANDLE, c_ubyte, c_ubyte, _POINTER)
_declare(
    winusb.WinUsb_ControlTransfer,
    _BOOL,
    _HANDLE,
    WINUSB_SETUP_PACKET,
    _POINTER,
    c_ulong,
    _POINTER,
    _POINTER,
)
_declare(winusb.WinUsb_AbortPipe, _BOOL, _HANDLE, c_ubyte)
_declare(winusb.WinUsb_GetOverlappedResult, _BOOL, _HANDLE, _POINTER, _POINTER, _BOOL)
_declare(winusb.WinUsb_RegisterIsochBuffer, _BOOL, _HANDLE, c_ubyte, _POINTER, c_ulong, _POINTER)
_declare(winusb.WinUsb_UnregisterIsochBuffer, _BOOL, _HANDLE)
_declare(
    winusb.WinUsb_ReadIsochPipeAsap,
    _BOOL,
    _HANDLE,
    c_ulong,
    c_ulong,
    _BOOL,
    c_ulong,
    _POINTER,
    _POINTER,
)


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------
def find_device_path(vid: int, pid: int) -> str | None:
    """Find the WinUSB device path for a given VID/PID.

    Returns None when no present device interface names it. A lookup that
    failed is not an absence: it raises, naming the call and its last error.
    """
    guid = GUID(*GUID_DEVINTERFACE_USB_DEVICE)
    dev_info = setupapi.SetupDiGetClassDevsW(
        byref(guid), None, None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    )

    if dev_info == INVALID_HANDLE_VALUE:
        raise RuntimeError(f'SetupDiGetClassDevsW failed: {ctypes.get_last_error()}')

    vid_pid = f'vid_{vid:04x}&pid_{pid:04x}'
    iface_data = SP_DEVICE_INTERFACE_DATA()
    iface_data.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA)
    idx = 0

    try:
        while True:
            if not setupapi.SetupDiEnumDeviceInterfaces(
                dev_info, None, byref(guid), idx, byref(iface_data)
            ):
                code = ctypes.get_last_error()
                if code == ERROR_NO_MORE_ITEMS:
                    return None
                raise RuntimeError(f'SetupDiEnumDeviceInterfaces({idx}) failed: {code}')
            idx += 1

            detail = SP_DEVICE_INTERFACE_DETAIL_DATA()
            detail.cbSize = 8 if sizeof(c_void_p) == 8 else 6  # 64-bit vs 32-bit
            devinfo = SP_DEVINFO_DATA()
            devinfo.cbSize = sizeof(SP_DEVINFO_DATA)
            required = c_ulong(0)

            if not setupapi.SetupDiGetDeviceInterfaceDetailW(
                dev_info,
                byref(iface_data),
                byref(detail),
                sizeof(detail),
                byref(required),
                byref(devinfo),
            ):
                raise RuntimeError(
                    f'SetupDiGetDeviceInterfaceDetailW({idx - 1}) failed: {ctypes.get_last_error()}'
                )

            path = detail.DevicePath
            if vid_pid in path.lower():
                return path
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(dev_info)


def _transfer_failed(transfer: str) -> Exception:
    """The exception for a ``transfer`` WinUSB reported failed, by its last error.

    A timeout is a ``TimeoutError``, the type the FX2 driver reads as a
    device that did not answer, as it reads pyusb's and python-libusb1's
    timeouts; any other failure is a ``RuntimeError``. Both name the code.
    """
    code = ctypes.get_last_error()
    if code == ERROR_SEM_TIMEOUT:
        return TimeoutError(f'{transfer} timed out: {code}')
    return RuntimeError(f'{transfer} failed: {code}')


# ---------------------------------------------------------------------------
# WinUSB device wrapper
# ---------------------------------------------------------------------------
class WinUsbDevice:
    """Wraps a WinUSB device handle for control transfers and ISO streaming."""

    def __init__(self, vid, pid):
        self.path = find_device_path(vid, pid)
        if self.path is None:
            raise RuntimeError(f'WinUSB device not found (VID=0x{vid:04X} PID=0x{pid:04X})')

        # Open device file
        self._file_handle = kernel32.CreateFileW(
            self.path,
            GENERIC_WRITE | GENERIC_READ,
            FILE_SHARE_WRITE | FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            None,
        )

        if self._file_handle == INVALID_HANDLE_VALUE:
            raise RuntimeError(f'CreateFile failed: {ctypes.get_last_error()}')

        # Initialize WinUSB
        self._iface_handle = c_void_p()
        if not winusb.WinUsb_Initialize(self._file_handle, byref(self._iface_handle)):
            kernel32.CloseHandle(self._file_handle)
            raise RuntimeError(f'WinUsb_Initialize failed: {ctypes.get_last_error()}')

        log.info('WinUSB device opened: %s', self.path)

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

    def set_alt_interface(self, setting: int) -> None:
        if not winusb.WinUsb_SetCurrentAlternateSetting(self._iface_handle, c_ubyte(setting)):
            raise RuntimeError(f'SetAlternateSetting({setting}) failed: {ctypes.get_last_error()}')

    def query_pipe(self, alt: int, index: int) -> WINUSB_PIPE_INFORMATION:
        pipe_info = WINUSB_PIPE_INFORMATION()
        if not winusb.WinUsb_QueryPipe(
            self._iface_handle, c_ubyte(alt), c_ubyte(index), byref(pipe_info)
        ):
            raise RuntimeError(f'QueryPipe failed: {ctypes.get_last_error()}')
        return pipe_info

    def control_transfer(
        self,
        request_type: int,
        request: int,
        value: int,
        index: int,
        data: bytes = b'',
        length: int = 0,
    ) -> bytes | int:
        pkt = WINUSB_SETUP_PACKET()
        pkt.RequestType = request_type
        pkt.Request = request
        pkt.Value = value
        pkt.Index = index

        transferred = c_ulong(0)
        # The BOOL is the only failure signal this API gives. Discarding it
        # made a failed transfer indistinguishable from a successful one --
        # the OUT branch returned None either way, and the IN branch returned
        # b'' because `transferred` stays 0 on failure. That silence reached
        # the driver's short-write detector, which skips its check when the
        # count is None, so a safety check disabled itself on this path.
        # The two sibling transports (pyusb, libusb1) raise; matching them
        # keeps one failure contract for all three rather than one per
        # platform.
        if request_type & 0x80:  # IN transfer
            buf = create_string_buffer(length)
            pkt.Length = length
            ok = winusb.WinUsb_ControlTransfer(
                self._iface_handle, pkt, buf, length, byref(transferred), None
            )
            if not ok:
                raise _transfer_failed(
                    f'ControlTransfer IN (request=0x{request:02X}, '
                    f'value=0x{value:04X}, index=0x{index:04X}, '
                    f'length={length})'
                )
            return bytes(buf[: transferred.value])
        else:  # OUT transfer
            pkt.Length = len(data)
            buf = ctypes.create_string_buffer(data) if data else None
            ok = winusb.WinUsb_ControlTransfer(
                self._iface_handle, pkt, buf if buf else None, len(data), byref(transferred), None
            )
            if not ok:
                raise _transfer_failed(
                    f'ControlTransfer OUT (request=0x{request:02X}, '
                    f'value=0x{value:04X}, index=0x{index:04X}, '
                    f'len={len(data)})'
                )
            # Bytes written, so the caller's `-> int` contract holds on this
            # transport too. Returning None here is what silently switched
            # off the short-write check downstream.
            return transferred.value

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
            byref(self.isoch_handle),
        )

        if not ok:
            raise RuntimeError(f'WinUsb_RegisterIsochBuffer failed: {ctypes.get_last_error()}')

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
    Runs in a dedicated thread; hands each packet's bytes to ``on_data``.

    ``on_data`` receives the bytes in arrival order; the stream's owner
    keeps them, so there is one buffer and the reader holds none.
    ``on_error`` is called once for each read that fails and for each
    failed packet inside a read that completed, so the stream's owner can
    count what never reached the buffer.
    """

    def __init__(
        self,
        vid,
        pid,
        *,
        on_data,
        on_error,
        pipe_id=0x82,
        alt_interface=3,
        num_slots=16,
        packets_per_xfer=256,
    ):
        self.vid = vid
        self.pid = pid
        self.pipe_id = pipe_id
        self.alt_interface = alt_interface
        self.num_slots = num_slots
        self.packets_per_xfer = packets_per_xfer
        self._on_data = on_data
        self._on_error = on_error

        self._dev = None
        self._running = False
        self._thread = None

    def start(self):
        """Open device, configure for ISO, start streaming thread."""
        self._dev = WinUsbDevice(self.vid, self.pid)
        self._dev.set_alt_interface(self.alt_interface)

        pipe_info = self._dev.query_pipe(self.alt_interface, 0)
        self._max_packet_size = pipe_info.MaximumPacketSize
        log.info(
            'ISO pipe: PipeId=0x%02X, MaxPacketSize=%d',
            pipe_info.PipeId,
            pipe_info.MaximumPacketSize,
        )

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
        """Main ISO read loop -- runs in dedicated thread."""
        dev = self._dev
        iface = dev.handle
        buffer_size = self._max_packet_size * self.packets_per_xfer
        n_packets = self.packets_per_xfer

        # Allocate slots
        slots = []
        try:
            for _ in range(self.num_slots):
                slots.append(IsoSlot(iface, self.pipe_id, buffer_size, n_packets))

            # Prime: submit all slots
            continue_stream = False
            for slot in slots:
                self._submit_read(slot, buffer_size, n_packets, continue_stream)
                continue_stream = True

            # Round-robin: wait -> extract -> resubmit
            next_slot = 0
            while self._running:
                slot = slots[next_slot]

                transferred = c_ulong(0)
                ok = winusb.WinUsb_GetOverlappedResult(
                    iface, byref(slot.overlapped), byref(transferred), True
                )  # wait=True

                if not ok:
                    self._on_error()
                elif transferred.value > 0:
                    # Extract data from ISO packets
                    for i in range(slot.packet_count):
                        pkt = slot.packets[i]
                        if pkt.Status != USBD_STATUS_SUCCESS:
                            self._on_error()
                        elif pkt.Length > 0:
                            start = pkt.Offset
                            end = start + pkt.Length
                            self._on_data(bytes(slot.buffer[start:end]))

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
                        iface, byref(slot.overlapped), byref(dummy), True
                    )
            for slot in slots:
                slot.dispose()

    def _submit_read(self, slot, length, packet_count, continue_stream):
        ok = winusb.WinUsb_ReadIsochPipeAsap(
            slot.isoch_handle,
            c_ulong(0),  # offset
            c_ulong(length),
            c_bool(continue_stream),
            c_ulong(packet_count),
            ctypes.cast(slot.packets, c_void_p),
            byref(slot.overlapped),
        )

        if not ok:
            err = ctypes.get_last_error()
            if err != ERROR_IO_PENDING:
                log.warning('WinUsb_ReadIsochPipeAsap failed: %d', err)
                slot.submitted = False
                return False
        slot.submitted = True
        return True
