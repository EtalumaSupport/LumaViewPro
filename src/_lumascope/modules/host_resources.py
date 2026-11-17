# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""This PC's resources over a session's startup, read once and judged once.

A PC short of memory, of free space on the save drive, or of CPU because
another program holds it looks to the person like a LumaViewPro defect:
slow, freezing, a camera that seems to drop. The session reads the PC
over a window from its ``create`` to its startup motion's end, keeps the
reading (``ScopeSession.host_reading``), and tells
``LowSystemResourcesNotice`` once when anything is past its line.

The lines are one byte count and one percent on every platform: sizes are
binary gigabytes, labelled GB as Windows' Explorer and Task Manager show
them.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import time
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
        read_s: How long the read took.
    """

    at_s: float
    cpu_times: NamedTuple
    processes: common_utils.ProcessCpuSeconds
    unread_processes: int
    cores: int
    ram_available_bytes: int
    disk_free_bytes: int | None
    read_s: float


def read_host(save_folder: str | None) -> HostSnapshot:
    """Read this PC now: CPU times, every process, memory and the save folder's free space.

    The one reader of the host for the startup reading, at both ends of the
    window; the test suite replaces it, so no test reads the PC it runs on.

    Args:
        save_folder: The folder whose drive's free space is read; None reads none.
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
        read_s=time.monotonic() - started,
    )


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
    """

    ram_available_bytes: int
    save_folder: str
    disk_free_bytes: int | None
    cpu_pct: float | None
    programs: tuple[ProgramCpu, ...]
    unread_processes: int
    window_s: float


def reading_between(opened: HostSnapshot, settled: HostSnapshot, save_folder: str) -> HostReading:
    """The reading of the window from ``opened`` to ``settled``.

    Args:
        opened: The read at the window's start.
        settled: The read at its end, of ``save_folder``.
        save_folder: The folder ``settled`` read the free space of.
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
    )


def judge(reading: HostReading) -> LowSystemResourcesNotice | None:
    """The notice for what in ``reading`` is past its line; None when nothing is.

    A save folder that could not be read is not judged; memory and the CPU
    still are.
    """
    ram_low = reading.ram_available_bytes < RAM_LINE_BYTES
    disk_low = reading.disk_free_bytes is not None and reading.disk_free_bytes < DISK_LINE_BYTES
    busy = tuple(p for p in reading.programs if p.cpu_pct > PROGRAM_CPU_LINE_PCT)
    if not (ram_low or disk_low or busy):
        return None
    return LowSystemResourcesNotice(reading, ram_low=ram_low, disk_low=disk_low, busy=busy)


def log_line(reading: HostReading, read_s: float) -> str:
    """The reading as the one INFO line the session logs at the window's end."""
    disk = (
        'unread'
        if reading.disk_free_bytes is None
        else f'{reading.disk_free_bytes / 2**30:.1f} GB free'
    )
    cpu = 'unread' if reading.cpu_pct is None else f'{reading.cpu_pct:.1f}%'
    programs = ', '.join(f'{p.name} (pid {p.pid}) {p.cpu_pct:.1f}%' for p in reading.programs)
    return (
        f'[HOST RESOURCES] RAM available: {reading.ram_available_bytes / 2**30:.1f} GB | '
        f'save drive ({reading.save_folder}): {disk} | '
        f'CPU over {reading.window_s:.1f} s, LumaViewPro included: {cpu} | '
        f'top other programs: {programs or "none"} | '
        f'{reading.unread_processes} processes unread | read in {read_s * 1000:.0f} ms'
    )
