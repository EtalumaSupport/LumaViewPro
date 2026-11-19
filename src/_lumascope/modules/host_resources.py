# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""This PC's resources over a session's startup, read once and judged once.

A PC short of memory, of free space on the save drive, or of CPU because
another program holds it looks to the person like a LumaViewPro defect:
slow, freezing, a camera that seems to drop. So does a window drawn on a
PC's power-saving GPU while it has a high-performance one: a laptop with
an Intel GPU beside an NVIDIA one showed the live view at 12 fps on the
Intel and 42 fps on the NVIDIA. The session reads the PC over a window
from its ``create`` to its startup motion's end, keeps the reading
(``ScopeSession.host_reading``), and tells ``LowSystemResourcesNotice``
once when anything is past its line.

The lines are one byte count and one percent on every platform: sizes are
binary gigabytes, labelled GB as Windows' Explorer and Task Manager show
them. The GPU is judged on Windows alone, the one platform with a
system-wide ranking of a PC's GPUs to judge against.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import sys
import time
import uuid
from typing import NamedTuple

import psutil

from _lumascope.modules import common_utils
from _lumascope.modules.api_surface import api_fields
from _lumascope.modules.exceptions import LowSystemResourcesNotice

# Under these, the notice is told.
RAM_LINE_BYTES = 4 * 2**30
DISK_LINE_BYTES = 50 * 2**30
# Over this share of the whole machine, another program is named. Per
# program: two at 29% each are not past it.
PROGRAM_CPU_LINE_PCT = 30.0
# The other programs kept on the reading, largest share first. No more than
# three can be over the line, so every one past it is kept.
PROGRAMS_KEPT = 3

# The GPU drawing names its maker in its OpenGL vendor string; Windows names
# each GPU's maker by its PCI vendor id. Matched by whole word, first row
# first: "ati" is a word in "ATI Technologies Inc." and a substring of
# "Corporation", which Intel's and NVIDIA's strings carry. A maker not here
# is not judged.
GPU_MAKERS: tuple[tuple[str, int], ...] = (
    ('nvidia', 0x10DE),
    ('intel', 0x8086),
    ('amd', 0x1002),
    ('ati', 0x1002),
)


@api_fields('vendor', 'renderer')
@dataclasses.dataclass(frozen=True)
class DrawingGpu:
    """The GPU LumaViewPro's window draws on, as its OpenGL context names it.

    Handed to ``ScopeSession.create`` by the host that draws; a host with
    no window hands none.

    Attributes:
        vendor: The context's ``GL_VENDOR``, such as ``Intel`` or
            ``NVIDIA Corporation``.
        renderer: The context's ``GL_RENDERER``, such as
            ``Intel(R) HD Graphics 530``.
    """

    vendor: str
    renderer: str


@api_fields('description', 'vendor_id')
@dataclasses.dataclass(frozen=True)
class GpuAdapter:
    """One of this PC's GPUs, as Windows lists it.

    Attributes:
        description: Windows' name for it, such as
            ``NVIDIA GeForce GTX 970M``.
        vendor_id: Its maker's PCI vendor id: 0x8086 Intel, 0x10DE NVIDIA,
            0x1002 AMD.
    """

    description: str
    vendor_id: int


@dataclasses.dataclass(frozen=True)
class HostSnapshot:
    """One end of the window: what ``read_host`` read, when.

    Attributes:
        at_s: When the read began, in ``time.monotonic`` seconds.
        cpu_times: The host's ``psutil.cpu_times()``.
        processes: Every readable process's CPU seconds
            (``common_utils.process_cpu_seconds``).
        unread_processes: The processes the OS would not let this one read.
        cores: The host's logical cores.
        ram_available_bytes: The memory available.
        disk_free_bytes: The save folder's free space; None when no folder
            was asked for, or it does not exist or could not be read.
        gpus: This PC's GPUs, the one Windows ranks for high performance
            first; None when they were not asked for, off Windows, or on a
            Windows that cannot rank them (before Windows 10 1803).
        read_s: How long the read took.
    """

    at_s: float
    cpu_times: NamedTuple
    processes: common_utils.ProcessCpuSeconds
    unread_processes: int
    cores: int
    ram_available_bytes: int
    disk_free_bytes: int | None
    gpus: tuple[GpuAdapter, ...] | None
    read_s: float


def read_host(save_folder: str | None, *, read_gpus: bool) -> HostSnapshot:
    """Read this PC now: CPU times, every process, memory, the save folder's free space and its GPUs.

    The one reader of the host for the startup reading, at both ends of the
    window; the test suite replaces it, so no test reads the PC it runs on.

    Args:
        save_folder: The folder whose drive's free space is read; None reads none.
        read_gpus: Whether the PC's GPUs are read: once, at the window's
            end, where they are judged.

    Raises:
        OSError: Windows refused to list the GPUs.
    """
    started = time.monotonic()
    processes, unread = common_utils.process_cpu_seconds()
    return HostSnapshot(
        at_s=started,
        cpu_times=psutil.cpu_times(),
        processes=processes,
        unread_processes=unread,
        cores=psutil.cpu_count(),
        ram_available_bytes=psutil.virtual_memory().available,
        disk_free_bytes=None if save_folder is None else _free_space(save_folder),
        gpus=_windows_gpus() if read_gpus and sys.platform == 'win32' else None,
        read_s=time.monotonic() - started,
    )


# DXGI, the one Windows interface that ranks a PC's GPUs as its Graphics
# setting does: IDXGIFactory6.EnumAdapterByGpuPreference. Called through
# each object's table of methods, numbered as dxgi1_6.h declares them.
_IID_IDXGI_FACTORY6 = uuid.UUID('c1b6694f-ff09-44a9-b03c-77900a0a1d17')
_IID_IDXGI_ADAPTER1 = uuid.UUID('29038f61-3839-4626-91fd-086879011a05')
_RELEASE = 2
_ADAPTER1_GET_DESC1 = 10
_FACTORY6_ENUM_ADAPTER_BY_GPU_PREFERENCE = 29
_GPU_PREFERENCE_HIGH_PERFORMANCE = 2
_ADAPTER_FLAG_SOFTWARE = 2
_S_OK = 0
_E_NOINTERFACE = 0x80004002
_DXGI_ERROR_NOT_FOUND = 0x887A0002


def _windows_gpus() -> tuple[GpuAdapter, ...] | None:
    """This PC's hardware GPUs, the one Windows ranks for high performance first; None before Windows 10 1803.

    Windows' software renderer is not one.

    Raises:
        OSError: DXGI failed other than for a Windows too old to rank.
    """
    import ctypes

    class _AdapterDesc1(ctypes.Structure):
        _fields_ = [
            ('Description', ctypes.c_wchar * 128),
            ('VendorId', ctypes.c_uint),
            ('DeviceId', ctypes.c_uint),
            ('SubSysId', ctypes.c_uint),
            ('Revision', ctypes.c_uint),
            ('DedicatedVideoMemory', ctypes.c_size_t),
            ('DedicatedSystemMemory', ctypes.c_size_t),
            ('SharedSystemMemory', ctypes.c_size_t),
            ('AdapterLuidLowPart', ctypes.c_ulong),
            ('AdapterLuidHighPart', ctypes.c_long),
            ('Flags', ctypes.c_uint),
        ]

    def guid(iid: uuid.UUID):
        return (ctypes.c_ubyte * 16).from_buffer_copy(iid.bytes_le)

    def method(obj: ctypes.c_void_p, slot: int, *argtypes):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(table[slot])

    def checked(hresult: int, call: str) -> None:
        if hresult & 0xFFFFFFFF != _S_OK:
            raise OSError(f'{call} failed: 0x{hresult & 0xFFFFFFFF:08x}')

    factory = ctypes.c_void_p()
    created = ctypes.WinDLL('dxgi').CreateDXGIFactory1(
        ctypes.byref(guid(_IID_IDXGI_FACTORY6)), ctypes.byref(factory)
    )
    if created & 0xFFFFFFFF == _E_NOINTERFACE:
        return None
    checked(created, 'CreateDXGIFactory1')
    adapters: list[GpuAdapter] = []
    try:
        enumerate_by_preference = method(
            factory,
            _FACTORY6_ENUM_ADAPTER_BY_GPU_PREFERENCE,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        iid_adapter1 = guid(_IID_IDXGI_ADAPTER1)
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            listed = enumerate_by_preference(
                factory,
                index,
                _GPU_PREFERENCE_HIGH_PERFORMANCE,
                ctypes.byref(iid_adapter1),
                ctypes.byref(adapter),
            )
            if listed & 0xFFFFFFFF == _DXGI_ERROR_NOT_FOUND:
                break
            checked(listed, 'EnumAdapterByGpuPreference')
            try:
                desc = _AdapterDesc1()
                get_desc1 = method(adapter, _ADAPTER1_GET_DESC1, ctypes.POINTER(_AdapterDesc1))
                checked(get_desc1(adapter, ctypes.byref(desc)), 'GetDesc1')
            finally:
                method(adapter, _RELEASE)(adapter)
            if not desc.Flags & _ADAPTER_FLAG_SOFTWARE:
                adapters.append(GpuAdapter(desc.Description, desc.VendorId))
            index += 1
    finally:
        method(factory, _RELEASE)(factory)
    return tuple(adapters)


def _free_space(save_folder: str) -> int | None:
    """The free space of the drive ``save_folder`` is on, or None when the folder is not there to read.

    A folder that does not exist is never measured by the folder above it,
    which can be another drive's: an unplugged drive's folder would read
    the boot drive's space under the save drive's name.
    """
    try:
        if not pathlib.Path(save_folder).is_dir():
            return None
        _, free_mb = common_utils.check_disk_space_ok(save_folder, 0)
    except OSError:
        return None
    return round(free_mb * 2**20)


@api_fields('name', 'pid', 'cpu_pct')
@dataclasses.dataclass(frozen=True)
class ProgramCpu:
    """Another program's share of the CPU over the startup.

    Attributes:
        name: The program's process name.
        pid: Its process id.
        cpu_pct: Its share of the whole machine, in percent: 100 is every
            logical core busy for the whole window, as Task Manager counts.
    """

    name: str
    pid: int
    cpu_pct: float


@api_fields(
    'ram_available_bytes',
    'save_folder',
    'disk_free_bytes',
    'cpu_pct',
    'programs',
    'unread_processes',
    'window_s',
    'drawing_gpu',
    'gpus',
)
@dataclasses.dataclass(frozen=True)
class HostReading:
    """This PC's resources, read once over the session's startup.

    The window opens when the session is created and closes when its
    startup motion ends: after homing, or at once on a host that does not
    home.

    Attributes:
        ram_available_bytes: The memory available at the window's end.
        save_folder: The save folder (``live_folder``) whose drive was read.
        disk_free_bytes: Its drive's free space at the window's end; None
            when the folder did not exist or could not be read.
        cpu_pct: The whole machine's CPU busy over the window, LumaViewPro's
            own startup included; None when no CPU time passed.
        programs: The other programs with the largest shares of the CPU over
            the window, up to three, largest first. LumaViewPro is not one;
            a process the OS would not let LumaViewPro read is not either.
        unread_processes: How many processes the OS would not let
            LumaViewPro read.
        window_s: The window's length.
        drawing_gpu: The GPU LumaViewPro's window draws on; None when
            nothing draws (no window: the REST server, a script).
        gpus: This PC's GPUs, the one Windows ranks for high performance
            first; None off Windows, or on a Windows that cannot rank them
            (before Windows 10 1803).
    """

    ram_available_bytes: int
    save_folder: str
    disk_free_bytes: int | None
    cpu_pct: float | None
    programs: tuple[ProgramCpu, ...]
    unread_processes: int
    window_s: float
    drawing_gpu: DrawingGpu | None
    gpus: tuple[GpuAdapter, ...] | None


def reading_between(
    opened: HostSnapshot,
    settled: HostSnapshot,
    save_folder: str,
    drawing_gpu: DrawingGpu | None,
) -> HostReading:
    """The reading of the window from ``opened`` to ``settled``.

    Args:
        opened: The read at the window's start.
        settled: The read at its end, of ``save_folder`` and the GPUs.
        save_folder: The folder ``settled`` read the free space of.
        drawing_gpu: The GPU the host's window draws on; None with no window.
    """
    # This process is LumaViewPro; on Windows pid 0 is the System Idle
    # Process, whose CPU time is the idle time. On macOS pid 0 is
    # kernel_task, a real load, kept where the OS lets it be read.
    left_out = frozenset({os.getpid(), 0} if sys.platform == 'win32' else {os.getpid()})
    window_s = settled.at_s - opened.at_s
    shares = common_utils.program_cpu_shares(
        opened.processes, settled.processes, window_s, settled.cores, left_out
    )
    return HostReading(
        ram_available_bytes=settled.ram_available_bytes,
        save_folder=save_folder,
        disk_free_bytes=settled.disk_free_bytes,
        cpu_pct=common_utils.cpu_busy_percent(opened.cpu_times, settled.cpu_times),
        programs=tuple(ProgramCpu(name, pid, pct) for name, pid, pct in shares[:PROGRAMS_KEPT]),
        unread_processes=settled.unread_processes,
        window_s=window_s,
        drawing_gpu=drawing_gpu,
        gpus=settled.gpus,
    )


class _GpuCase(NamedTuple):
    """How the reading's GPUs stand: ``preferred`` is the high-performance GPU LumaViewPro is not drawing on."""

    words: str
    preferred: GpuAdapter | None = None


def _maker(gl_vendor: str) -> int | None:
    """The PCI vendor id ``gl_vendor`` names by whole word (``GPU_MAKERS``); None for a maker not there."""
    words = set(re.findall(r'[a-z]+', gl_vendor.lower()))
    return next((vendor_id for name, vendor_id in GPU_MAKERS if name in words), None)


def _gpu_case(reading: HostReading) -> _GpuCase:
    """Whether LumaViewPro draws on the GPU Windows ranks for high performance, or why that is not judged.

    Judged only between makers: two GPUs of one maker, or a maker the
    window's vendor string does not name, cannot be told apart.
    """
    if reading.drawing_gpu is None:
        return _GpuCase('not judged: no window draws')
    if reading.gpus is None:
        return _GpuCase('not judged: no ranking of the GPUs on this system')
    if len({gpu.vendor_id for gpu in reading.gpus}) < 2:
        return _GpuCase('not judged: fewer than two makers among the GPUs')
    drawing = _maker(reading.drawing_gpu.vendor)
    if drawing is None:
        return _GpuCase(f'not judged: no known maker in {reading.drawing_gpu.vendor!r}')
    preferred = reading.gpus[0]
    if drawing == preferred.vendor_id:
        return _GpuCase('on the high-performance GPU')
    return _GpuCase('not on the high-performance GPU', preferred)


def judge(reading: HostReading) -> LowSystemResourcesNotice | None:
    """The notice for what in ``reading`` is past its line; None when nothing is.

    A save folder that could not be read is not judged; memory and the CPU
    still are. The GPU is past when LumaViewPro draws on a GPU of another
    maker than the one Windows ranks for high performance.
    """
    ram_low = reading.ram_available_bytes < RAM_LINE_BYTES
    disk_low = reading.disk_free_bytes is not None and reading.disk_free_bytes < DISK_LINE_BYTES
    busy = tuple(p for p in reading.programs if p.cpu_pct > PROGRAM_CPU_LINE_PCT)
    preferred_gpu = _gpu_case(reading).preferred
    if not (ram_low or disk_low or busy or preferred_gpu):
        return None
    return LowSystemResourcesNotice(
        reading, ram_low=ram_low, disk_low=disk_low, busy=busy, preferred_gpu=preferred_gpu
    )


def log_line(reading: HostReading, read_s: float) -> str:
    """The reading as the one INFO line the session logs at the window's end."""
    disk = (
        'unread'
        if reading.disk_free_bytes is None
        else f'{reading.disk_free_bytes / 2**30:.1f} GB free'
    )
    cpu = 'unread' if reading.cpu_pct is None else f'{reading.cpu_pct:.1f}%'
    programs = ', '.join(f'{p.name} (pid {p.pid}) {p.cpu_pct:.1f}%' for p in reading.programs)
    drawing = (
        'none'
        if reading.drawing_gpu is None
        else f'{reading.drawing_gpu.renderer} ({reading.drawing_gpu.vendor})'
    )
    gpus = (
        'unread'
        if reading.gpus is None
        else ', '.join(f'{g.description} (0x{g.vendor_id:04X})' for g in reading.gpus) or 'none'
    )
    return (
        f'[HOST RESOURCES] RAM available: {reading.ram_available_bytes / 2**30:.1f} GB | '
        f'save drive ({reading.save_folder}): {disk} | '
        f'CPU over {reading.window_s:.1f} s, LumaViewPro included: {cpu} | '
        f'top other programs: {programs or "none"} | '
        f'{reading.unread_processes} processes unread | '
        f'drawing on: {drawing} | GPUs, high performance first: {gpus} | '
        f'{_gpu_case(reading).words} | read in {read_s * 1000:.0f} ms'
    )
