"""
What Windows knows of a USB device, through SetupAPI and the hub: each
device of an identity and the driver bound to it, its properties, and the
port it is on.

The stream and every transfer go through libusb, on Windows as elsewhere
(``fx2driver._LibusbTransport``); this module reads only what Windows
alone holds. Windows only -- uses setupapi.dll and kernel32.dll directly.
"""

import ctypes
import datetime
import logging
from collections.abc import Callable, Iterator
from ctypes import (
    Structure,
    byref,
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


from _lumascope.drivers.exceptions import CameraUsbLookupFailedError
from _lumascope.drivers.usb_census import HubPort
from _lumascope.drivers.usb_driver_binding import (
    DeviceBinding,
    DeviceProperties,
    binding_from,
    store_inf_folder,
    store_packages,
)

# Under LVP, never ``__name__``: a session's logging hangs the bundle on
# LumaViewPro's own logger, LVP, so a logger named for this
# module would write nowhere a support bundle collects.
log = logging.getLogger('LVP.windows_usb')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENERIC_WRITE = 0x40000000
FILE_SHARE_WRITE = 0x02
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
# What SetupDiEnumDeviceInterfaces reports past the last interface.
ERROR_NO_MORE_ITEMS = 259
DIGCF_PRESENT = 0x02
DIGCF_ALLCLASSES = 0x04
DIGCF_DEVICEINTERFACE = 0x10
# What SetupDiGetDevicePropertyW reports for a property the device lacks
# (no driver bound: no service, provider or INF).
ERROR_NOT_FOUND = 1168
DEVPROP_TYPE_UINT32 = 0x07
DEVPROP_TYPE_FILETIME = 0x10
DEVPROP_TYPE_BOOLEAN = 0x11
DEVPROP_TYPE_STRING = 0x12
DEVPROP_TYPE_NTSTATUS = 0x18
SPDRP_HARDWAREID = 1
# SetupDiOpenDevRegKey: the device's hardware key, read only, where its
# package's AddReg writes the interface GUIDs.
DICS_FLAG_GLOBAL = 0x1
DIREG_DEV = 0x1
KEY_READ = 0x20019

# Every USB hub's interface (usbiodef.h), which answers for the devices on
# its ports.
GUID_DEVINTERFACE_USB_HUB = (
    0xF18A0E88,
    0xC30C,
    0x11D0,
    (0x88, 0x15, 0x00, 0xA0, 0xC9, 0x06, 0xBE, 0xD8),
)
# usbioctl.h: CTL_CODE(FILE_DEVICE_USB, function, METHOD_BUFFERED, FILE_ANY_ACCESS).
IOCTL_USB_GET_NODE_CONNECTION_INFORMATION_EX = 0x220448
IOCTL_USB_GET_NODE_CONNECTION_INFORMATION_EX_V2 = 0x22045C
# The protocols a V2 query asks the hub about: USB 1.1, 2.0 and 3.x.
_ASK_EVERY_PROTOCOL = 0x7
# usbioctl.h's USB_PIPE_INFO is 11 bytes packed; the connection query
# answers its open pipes after the fixed part, and USBView sizes for 30.
_PIPE_INFO_BYTES = 11 * 30


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


class DEVPROPKEY(Structure):
    _fields_ = [
        ('fmtid', GUID),
        ('pid', c_ulong),
    ]


def _devpropkey(data1, data2, data3, data4, pid) -> DEVPROPKEY:
    return DEVPROPKEY(GUID(data1, data2, data3, (c_ubyte * 8)(*data4)), pid)


# devpkey.h
_DRIVER_FMTID = (0xA8B865DD, 0x2E3D, 0x4094, (0xAD, 0x97, 0xE5, 0x93, 0xA7, 0x0C, 0x75, 0xD6))
DEVPKEY_Device_DriverDate = _devpropkey(*_DRIVER_FMTID, 2)
DEVPKEY_Device_DriverVersion = _devpropkey(*_DRIVER_FMTID, 3)
DEVPKEY_Device_DriverInfPath = _devpropkey(*_DRIVER_FMTID, 5)
DEVPKEY_Device_DriverProvider = _devpropkey(*_DRIVER_FMTID, 9)
DEVPKEY_Device_Service = _devpropkey(
    0xA45C254E, 0xDF1C, 0x4EFD, (0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0), 6
)
DEVPKEY_Device_ProblemCode = _devpropkey(
    0x4340A6C5, 0x93FA, 0x4706, (0x97, 0x2C, 0x7B, 0x64, 0x80, 0x08, 0xA5, 0xA7), 3
)
DEVPKEY_Device_ProblemStatus = _devpropkey(
    0x4340A6C5, 0x93FA, 0x4706, (0x97, 0x2C, 0x7B, 0x64, 0x80, 0x08, 0xA5, 0xA7), 12
)
DEVPKEY_Device_Parent = _devpropkey(
    0x4340A6C5, 0x93FA, 0x4706, (0x97, 0x2C, 0x7B, 0x64, 0x80, 0x08, 0xA5, 0xA7), 8
)
DEVPKEY_Device_LocationInfo = _devpropkey(
    0xA45C254E, 0xDF1C, 0x4EFD, (0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0), 15
)
DEVPKEY_Device_IsPresent = _devpropkey(
    0x540B947E, 0x8B40, 0x45BC, (0xA8, 0xA2, 0x6A, 0x0B, 0x89, 0x4C, 0xBD, 0xA2), 5
)
# The name Device Manager shows: the friendly name, else the description.
DEVPKEY_NAME = _devpropkey(
    0xB725F130, 0x47EF, 0x101A, (0xA5, 0xF1, 0x02, 0x60, 0x8C, 0x9E, 0xEB, 0xAC), 10
)
DEVPKEY_Device_Class = _devpropkey(
    0xA45C254E, 0xDF1C, 0x4EFD, (0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0), 9
)
# The device node's status bits (cfg.h's DN_*).
DEVPKEY_Device_DevNodeStatus = _devpropkey(
    0x4340A6C5, 0x93FA, 0x4706, (0x97, 0x2C, 0x7B, 0x64, 0x80, 0x08, 0xA5, 0xA7), 2
)
# A USB device's address on its bus driver: the number of the hub port it is on.
DEVPKEY_Device_Address = _devpropkey(
    0xA45C254E, 0xDF1C, 0x4EFD, (0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0), 30
)


class USB_NODE_CONNECTION_INFORMATION_EX(Structure):
    """usbioctl.h, packed: a hub port's device, its descriptor, and the speed it runs at."""

    _pack_ = 1
    _fields_ = [
        ('ConnectionIndex', c_ulong),
        ('DeviceDescriptor', c_ubyte * 18),
        ('CurrentConfigurationValue', c_ubyte),
        ('Speed', c_ubyte),
        ('DeviceIsHub', c_ubyte),
        ('DeviceAddress', c_ushort),
        ('NumberOfOpenPipes', c_ulong),
        ('ConnectionStatus', c_ulong),
        ('PipeList', c_ubyte * _PIPE_INFO_BYTES),
    ]


class USB_NODE_CONNECTION_INFORMATION_EX_V2(Structure):
    """usbioctl.h: the protocols a hub port supports, and how its device runs."""

    _fields_ = [
        ('ConnectionIndex', c_ulong),
        ('Length', c_ulong),
        ('SupportedUsbProtocols', c_ulong),
        ('Flags', c_ulong),
    ]


# ---------------------------------------------------------------------------
# DLL loading
# ---------------------------------------------------------------------------
# use_last_error: each call's last error is saved as it returns, and
# ctypes.get_last_error() reads that copy; GetLastError read later sees
# whatever the interpreter set in between.
kernel32 = WinDLL('kernel32', use_last_error=True)
setupapi = WinDLL('setupapi', use_last_error=True)

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
    setupapi.SetupDiOpenDevRegKey, _HANDLE, _HANDLE, _POINTER, c_ulong, c_ulong, c_ulong, c_ulong
)
_declare(setupapi.SetupDiEnumDeviceInfo, _BOOL, _HANDLE, c_ulong, _POINTER)
_declare(
    setupapi.SetupDiGetDeviceInstanceIdW, _BOOL, _HANDLE, _POINTER, _POINTER, c_ulong, _POINTER
)
_declare(
    setupapi.SetupDiGetDevicePropertyW,
    _BOOL,
    _HANDLE,
    _POINTER,
    _POINTER,
    _POINTER,
    _POINTER,
    c_ulong,
    _POINTER,
    c_ulong,
)
_declare(
    kernel32.CreateFileW, _HANDLE, c_wchar_p, c_ulong, c_ulong, _POINTER, c_ulong, c_ulong, _HANDLE
)
_declare(kernel32.CloseHandle, _BOOL, _HANDLE)
_declare(
    kernel32.DeviceIoControl,
    _BOOL,
    _HANDLE,
    c_ulong,
    _POINTER,
    c_ulong,
    _POINTER,
    c_ulong,
    _POINTER,
    _POINTER,
)


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------
def _interface_paths(interface: tuple, instance_id: str | None = None) -> Iterator[str]:
    """The path of each present device interface of the class ``interface``, in turn.

    A generator, so a caller that stops at a match reads no interface past it.

    Args:
        interface: The interface class GUID, as its four parts.
        instance_id: Only the interfaces of this device; every device's when None.

    Raises:
        CameraUsbLookupFailedError: A SetupAPI call failed, naming it and its
            last error.
    """
    guid = GUID(*interface)
    dev_info = setupapi.SetupDiGetClassDevsW(
        byref(guid), instance_id, None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    )
    if dev_info == INVALID_HANDLE_VALUE:
        raise CameraUsbLookupFailedError(f'SetupDiGetClassDevsW failed: {ctypes.get_last_error()}')
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
                    return
                raise CameraUsbLookupFailedError(
                    f'SetupDiEnumDeviceInterfaces({idx}) failed: {code}'
                )
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
                raise CameraUsbLookupFailedError(
                    f'SetupDiGetDeviceInterfaceDetailW({idx - 1}) failed: {ctypes.get_last_error()}',
                )
            yield detail.DevicePath
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(dev_info)


def hub_port(device: DeviceProperties) -> HubPort | None:
    """The device's connection as the hub it is on reports it: its speed, its port's protocols.

    Asked of the hub's own interface, by the device's port number, without
    elevation where Windows allows a hub to be opened for it.

    Returns:
        HubPort | None: None when Windows names no parent or port for the
            device, or its parent is not a hub (a function of a composite
            device, whose speed is its parent's).

    Raises:
        CameraUsbLookupFailedError: The hub could not be opened or asked,
            naming the call and its last error.
    """
    parent = device.values.get('parent')
    port = device.values.get('address')
    if parent is None or port is None:
        return None
    hub = next(_interface_paths(GUID_DEVINTERFACE_USB_HUB, parent), None)
    if hub is None:
        return None
    handle = kernel32.CreateFileW(
        hub, GENERIC_WRITE, FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None
    )
    if handle is None or handle == INVALID_HANDLE_VALUE:
        raise CameraUsbLookupFailedError(f'CreateFileW({parent}) failed: {ctypes.get_last_error()}')
    try:
        connection = USB_NODE_CONNECTION_INFORMATION_EX()
        connection.ConnectionIndex = port
        _ask_hub(handle, IOCTL_USB_GET_NODE_CONNECTION_INFORMATION_EX, connection, parent, port)
        protocols = USB_NODE_CONNECTION_INFORMATION_EX_V2()
        protocols.ConnectionIndex = port
        protocols.Length = sizeof(USB_NODE_CONNECTION_INFORMATION_EX_V2)
        protocols.SupportedUsbProtocols = _ASK_EVERY_PROTOCOL
        _ask_hub(handle, IOCTL_USB_GET_NODE_CONNECTION_INFORMATION_EX_V2, protocols, parent, port)
    finally:
        kernel32.CloseHandle(handle)
    return HubPort(
        speed=connection.Speed, protocols=protocols.SupportedUsbProtocols, flags=protocols.Flags
    )


def _ask_hub(handle, code: int, answer: Structure, parent: str, port: int) -> None:
    """One query of a hub, ``answer`` sent and filled in place.

    Raises:
        CameraUsbLookupFailedError: The hub refused it, naming the query and
            its last error.
    """
    returned = c_ulong(0)
    if not kernel32.DeviceIoControl(
        handle,
        code,
        byref(answer),
        sizeof(answer),
        byref(answer),
        sizeof(answer),
        byref(returned),
        None,
    ):
        raise CameraUsbLookupFailedError(
            f'DeviceIoControl({code:#x}) of {parent} port {port} failed: {ctypes.get_last_error()}'
        )


def device_bindings(hardware_id: str) -> list[DeviceBinding]:
    """Every device of ``hardware_id``, present or seen before, and its driver.

    Read without elevation. A device seen before on another port keeps the
    binding it had there until it is matched afresh, so the past ones are
    read too. A device is on ours when its bound INF is one of our packages
    in the driver store.

    Args:
        hardware_id: ``USB\\VID_xxxx&PID_xxxx``; a device's instance ID is
            the identity, a backslash, then its serial or port.

    Raises:
        CameraUsbLookupFailedError: A SetupAPI call failed, naming it and its
            last error, or a property the binding is decided by came in a
            type this does not read; whether the devices are there, or
            usable, is then not known.
    """
    prefix = hardware_id.upper() + '\\'
    ours = {p.inf.lower() for p in store_packages(hardware_id, store_inf_folder()) if p.ours}
    return [
        binding_from(device, ours)
        for device in usb_devices(lambda instance_id: instance_id.upper().startswith(prefix))
    ]


# Each property the walk reads of a device, by the name ``DeviceProperties``
# holds it under.
_PROPERTIES = {
    'present': DEVPKEY_Device_IsPresent,
    'service': DEVPKEY_Device_Service,
    'provider': DEVPKEY_Device_DriverProvider,
    'inf': DEVPKEY_Device_DriverInfPath,
    'version': DEVPKEY_Device_DriverVersion,
    'date': DEVPKEY_Device_DriverDate,
    'problem_code': DEVPKEY_Device_ProblemCode,
    'problem_status': DEVPKEY_Device_ProblemStatus,
    'location': DEVPKEY_Device_LocationInfo,
    'parent': DEVPKEY_Device_Parent,
    'address': DEVPKEY_Device_Address,
}
# Each property of Device Manager's whole list, by the name ``DeviceProperties``
# holds it under.
_LIST_PROPERTIES = {
    'name': DEVPKEY_NAME,
    'class': DEVPKEY_Device_Class,
    'status': DEVPKEY_Device_DevNodeStatus,
    'problem_code': DEVPKEY_Device_ProblemCode,
}


def usb_devices(wanted: Callable[[str], bool]) -> list[DeviceProperties]:
    """Each USB device Windows lists, present or seen before, whose instance ID is ``wanted``.

    Every reader of a USB device's driver reads it here. A device's
    properties are read only once its instance ID is wanted, so another
    device's never fail the read.

    Args:
        wanted: Whether a device, by its instance ID, is read.

    Raises:
        CameraUsbLookupFailedError: A SetupAPI call failed, naming it and its
            last error.
    """
    return _walk('USB', DIGCF_ALLCLASSES, wanted, _PROPERTIES, with_interface_guids=True)


def present_devices() -> list[DeviceProperties]:
    """Every device Windows has present, of every class: its name, class, status and problem code.

    Device Manager's whole list, for a support report: what else is on the
    machine, and which of it Windows did not start.

    Raises:
        CameraUsbLookupFailedError: A SetupAPI call failed, naming it and its
            last error.
    """
    return _walk(
        None,
        DIGCF_ALLCLASSES | DIGCF_PRESENT,
        lambda _instance_id: True,
        _LIST_PROPERTIES,
        with_interface_guids=False,
    )


def _walk(
    enumerator: str | None,
    flags: int,
    wanted: Callable[[str], bool],
    properties: dict[str, DEVPROPKEY],
    *,
    with_interface_guids: bool,
) -> list[DeviceProperties]:
    """The one walk of Windows' device list: each wanted device and its ``properties``.

    Args:
        enumerator: Only the devices this enumerator lists (``'USB'``); every
            device when None.
        flags: SetupDiGetClassDevsW's flags: every class, and present only or not.
        wanted: Whether a device, by its instance ID, is read.
        properties: The properties read of each, by the name it is held under.
        with_interface_guids: Whether the device's interface GUIDs are read
            from its hardware key.

    Raises:
        CameraUsbLookupFailedError: A SetupAPI call failed, naming it and its
            last error.
    """
    dev_info = setupapi.SetupDiGetClassDevsW(None, enumerator, None, flags)
    if dev_info == INVALID_HANDLE_VALUE:
        raise CameraUsbLookupFailedError(f'SetupDiGetClassDevsW failed: {ctypes.get_last_error()}')
    devices = []
    try:
        index = 0
        while True:
            devinfo = SP_DEVINFO_DATA()
            devinfo.cbSize = sizeof(SP_DEVINFO_DATA)
            if not setupapi.SetupDiEnumDeviceInfo(dev_info, index, byref(devinfo)):
                code = ctypes.get_last_error()
                if code == ERROR_NO_MORE_ITEMS:
                    return devices
                raise CameraUsbLookupFailedError(f'SetupDiEnumDeviceInfo({index}) failed: {code}')
            index += 1
            instance_id = _instance_id(dev_info, devinfo)
            if wanted(instance_id):
                devices.append(
                    _device_properties(
                        dev_info, devinfo, instance_id, properties, with_interface_guids
                    )
                )
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(dev_info)


def _device_properties(
    dev_info,
    devinfo,
    instance_id: str,
    properties: dict[str, DEVPROPKEY],
    with_interface_guids: bool,
) -> DeviceProperties:
    """Each of ``properties``, and the interface GUIDs when asked, that Windows gave.

    A property Windows could not give, or gave in a type this does not read,
    is recorded in ``unreadable`` with why, never read as absent: one
    device's property never fails the walk of the others, and the reader
    decides whether it fails its own.
    """
    reads: dict[str, Callable[[], object]] = {
        name: (lambda key=key: _device_property(dev_info, devinfo, key))
        for name, key in properties.items()
    }
    if with_interface_guids:
        reads['interface_guids'] = lambda: _interface_guids(dev_info, devinfo)
    values: dict[str, object] = {}
    unreadable: dict[str, str] = {}
    for name, read in reads.items():
        try:
            value = read()
        except _UnreadTypeError as e:
            unreadable[name] = e.type_name
            continue
        except CameraUsbLookupFailedError as e:
            unreadable[name] = str(e)
            continue
        if value is not None:
            values[name] = value
    return DeviceProperties(instance_id=instance_id, values=values, unreadable=unreadable)


def _instance_id(dev_info, devinfo) -> str:
    buffer = ctypes.create_unicode_buffer(512)
    if not setupapi.SetupDiGetDeviceInstanceIdW(
        dev_info, byref(devinfo), buffer, len(buffer), None
    ):
        raise CameraUsbLookupFailedError(
            f'SetupDiGetDeviceInstanceIdW failed: {ctypes.get_last_error()}'
        )
    return buffer.value


def _interface_guids(dev_info, devinfo) -> tuple[str, ...] | None:
    """The device interface GUIDs the device's bound package registered for it.

    Read from the device's hardware key, where an INF's AddReg writes
    ``DeviceInterfaceGUIDs``: the value libusb's WinUSB backend reads to
    open the device, and on 2026-11-11 the cause of a code 31. None when the
    device has no hardware key (no driver bound) or no such value.

    Raises:
        _UnreadTypeError: The value is of another registry type, named.
    """
    import winreg  # local-import: Windows' own module, and this one loads only there

    key = setupapi.SetupDiOpenDevRegKey(
        dev_info, byref(devinfo), DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ
    )
    if key is None or key == INVALID_HANDLE_VALUE:
        return None
    try:
        value, value_type = winreg.QueryValueEx(key, 'DeviceInterfaceGUIDs')
    except OSError:
        return None
    finally:
        winreg.CloseKey(key)
    if value_type == winreg.REG_MULTI_SZ:
        return tuple(value)
    if value_type == winreg.REG_SZ:
        return (value,)
    raise _UnreadTypeError(f'registry type {value_type}')


class _UnreadTypeError(Exception):
    """Windows gave a device property, or a registry value, in a type this module does not read.

    Caught where the walk reads the device, which records it on the device
    by ``type_name``; whether that fails the read is the reader's to decide.
    """

    def __init__(self, type_name: str):
        self.type_name = type_name
        super().__init__(f'{type_name}, not a type this reads')


def _device_property(dev_info, devinfo, key: DEVPROPKEY):
    """A device property as Python: a string, an int, a bool, an ISO date.

    None when the device lacks it, as a device with no driver lacks its
    service and package. An NTSTATUS is its unsigned 32 bits, as Windows
    prints it in hex.

    Raises:
        CameraUsbLookupFailedError: The read failed.
        _UnreadTypeError: The property is of a type this does not read.
    """
    prop_type = c_ulong(0)
    required = c_ulong(0)
    buffer = create_string_buffer(1024)
    if not setupapi.SetupDiGetDevicePropertyW(
        dev_info,
        byref(devinfo),
        byref(key),
        byref(prop_type),
        buffer,
        sizeof(buffer),
        byref(required),
        0,
    ):
        code = ctypes.get_last_error()
        if code == ERROR_NOT_FOUND:
            return None
        raise CameraUsbLookupFailedError(f'SetupDiGetDevicePropertyW failed: {code}')
    raw = buffer.raw[: required.value]
    if prop_type.value == DEVPROP_TYPE_STRING:
        return raw.decode('utf-16-le').rstrip('\x00')
    if prop_type.value in (DEVPROP_TYPE_UINT32, DEVPROP_TYPE_NTSTATUS):
        return int.from_bytes(raw[:4], 'little')
    if prop_type.value == DEVPROP_TYPE_BOOLEAN:
        return raw[:1] != b'\x00'
    if prop_type.value == DEVPROP_TYPE_FILETIME:
        # 100 ns ticks since 1601-01-01.
        ticks = int.from_bytes(raw[:8], 'little')
        moment = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=ticks // 10)
        return moment.date().isoformat()
    raise _UnreadTypeError(f'type {prop_type.value:#x}')
