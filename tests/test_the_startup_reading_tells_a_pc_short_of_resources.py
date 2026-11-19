# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A PC short of resources at startup is told so, once, when the startup motion ends.

A distributor's demo PC with 1.5 GB of memory available, 9.3 GB free and
another program holding the CPU reported LumaViewPro as slow, freezing
and dropping USB, and was told to check the cable. The session now reads
the PC over a window from its ``create`` to its startup motion's end,
keeps the reading (``ScopeSession.host_reading``, and ``status``), and
tells ``LowSystemResourcesNotice`` naming each thing past its line: under
4 GB of memory, under 50 GB free on the save drive, another program over
30% of the machine, or a window drawn on another maker's GPU than the one
Windows ranks for high performance (a laptop's Intel beside its NVIDIA
showed the live view at 12 fps against 42). Inside every line, nothing is
told.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import namedtuple

import pytest

from _lumascope.modules import common_utils, host_resources
from _lumascope.modules.exceptions import (
    HardwareCommandRefusedError,
    HomingFaultError,
    MoveStoppedError,
    SessionClosingError,
)
from _lumascope.modules.host_resources import (
    DrawingGpu,
    GpuAdapter,
    HostReading,
    ProgramCpu,
    judge,
    log_line,
)
from _lumascope.modules.scope_session import ScopeSession
from tests.conftest import a_roomy_host
from tests.settings_fixtures import complete_settings

GB = 2**30
TITLE = 'This PC may slow LumaViewPro'
# The laptop's two GPUs, the NVIDIA ranked for high performance.
INTEL = GpuAdapter('Intel(R) HD Graphics 530', 0x8086)
NVIDIA = GpuAdapter('NVIDIA GeForce GTX 970M', 0x10DE)
ON_INTEL = DrawingGpu('Intel', 'Intel(R) HD Graphics 530')
ON_NVIDIA = DrawingGpu('NVIDIA Corporation', 'NVIDIA GeForce GTX 970M/PCIe/SSE2')
SETTLED_S = 30.0

# The real reader, bound before the suite's fixture replaces it.
_REAL_READ_HOST = host_resources.read_host


def _reading(**changed) -> HostReading:
    """A reading inside every line, with ``changed`` applied."""
    roomy = HostReading(
        ram_available_bytes=16 * GB,
        save_folder='/data',
        disk_free_bytes=500 * GB,
        cpu_pct=20.0,
        programs=(),
        unread_processes=0,
        window_s=10.0,
        drawing_gpu=None,
        gpus=None,
    )
    return dataclasses.replace(roomy, **changed)


# ---------------------------------------------------------------- the judge


def test_the_demo_pcs_reading_names_memory_disk_and_the_program_each_with_its_consequence():
    reading = _reading(
        ram_available_bytes=round(1.5 * GB),
        disk_free_bytes=round(9.3 * GB),
        cpu_pct=58.0,
        programs=(ProgramCpu('Updater.exe', 4120, 45.0), ProgramCpu('explorer.exe', 88, 3.0)),
    )

    notice = judge(reading)

    assert (notice.ram_low, notice.disk_low) == (True, True)
    assert notice.busy == (ProgramCpu('Updater.exe', 4120, 45.0),)
    words = str(notice)
    assert (
        'At startup this PC had 1.5 GB of memory available, 9.3 GB free on the save drive' in words
    )
    assert (
        'With 1.5 GB of memory available, LumaViewPro may run slowly, drop frames or stall' in words
    )
    assert 'Updater.exe is using 45% of the CPU, so LumaViewPro may run slowly' in words
    assert (
        'With 9.3 GB free on the drive LumaViewPro saves to, long runs and recordings may not fit'
        in words
    )
    assert 'explorer.exe' not in words
    assert notice.title == TITLE


def test_a_pc_inside_every_line_is_told_nothing():
    assert judge(_reading(programs=(ProgramCpu('a', 1, 29.0), ProgramCpu('b', 2, 29.0)))) is None


def test_each_line_is_past_only_beyond_it():
    at_the_lines = _reading(
        ram_available_bytes=4 * GB,
        disk_free_bytes=50 * GB,
        programs=(ProgramCpu('a', 1, 30.0),),
    )
    one_past_each = _reading(
        ram_available_bytes=4 * GB - 1,
        disk_free_bytes=50 * GB - 1,
        programs=(ProgramCpu('a', 1, 30.001),),
    )

    notice = judge(one_past_each)

    assert judge(at_the_lines) is None
    assert (notice.ram_low, notice.disk_low, len(notice.busy)) == (True, True, 1)


def test_a_save_drive_that_could_not_be_read_is_not_judged_and_memory_still_is():
    notice = judge(
        _reading(ram_available_bytes=2 * GB, disk_free_bytes=None, save_folder='E:\\data')
    )

    assert (notice.ram_low, notice.disk_low) == (True, False)
    assert 'the free space of E:\\data could not be read' in str(notice)
    assert judge(_reading(disk_free_bytes=None)) is None


def test_a_popup_names_up_to_three_programs():
    # No more than three programs can each hold over 30% of the machine.
    busy = tuple(ProgramCpu(f'program{i}', i, 32.0) for i in range(3))

    words = str(judge(_reading(programs=busy)))

    assert all(f'program{i} is using 32% of the CPU' in words for i in range(3))


# ------------------------------------------------------------------ the GPU


def test_a_window_on_the_intel_beside_a_higher_ranked_nvidia_is_told_first_without_the_figures():
    notice = judge(_reading(drawing_gpu=ON_INTEL, gpus=(NVIDIA, INTEL)))

    assert notice.preferred_gpu == NVIDIA
    assert (notice.ram_low, notice.disk_low, notice.busy) == (False, False, ())
    assert str(notice).startswith(
        'LumaViewPro is drawing on the Intel(R) HD Graphics 530, not the NVIDIA GeForce GTX 970M '
        'Windows ranks for high performance'
    )
    assert 'Settings > System > Display > Graphics' in str(notice)
    assert 'At startup this PC had' not in str(notice)
    assert notice.title == TITLE


def test_the_gpu_and_low_memory_are_told_in_one_notice_the_gpu_first():
    notice = judge(_reading(drawing_gpu=ON_INTEL, gpus=(NVIDIA, INTEL), ram_available_bytes=GB))
    words = str(notice)

    assert notice.preferred_gpu == NVIDIA and notice.ram_low
    assert words.index('LumaViewPro is drawing on') < words.index('At startup this PC had 1.0 GB')


def test_a_window_on_the_higher_ranked_gpu_is_told_nothing():
    assert judge(_reading(drawing_gpu=ON_NVIDIA, gpus=(NVIDIA, INTEL))) is None


@pytest.mark.parametrize(
    'drawing_gpu, gpus, unjudged',
    [
        (None, (NVIDIA, INTEL), 'no window draws'),
        (ON_INTEL, None, 'no ranking of the GPUs on this system'),
        (ON_INTEL, (INTEL,), 'fewer than two makers among the GPUs'),
        (
            ON_INTEL,
            (GpuAdapter('Intel(R) Arc(TM) A770', 0x8086), INTEL),
            'fewer than two makers among the GPUs',
        ),
        (
            DrawingGpu('Google Inc.', 'ANGLE (Intel HD 530)'),
            (NVIDIA, INTEL),
            "no known maker in 'Google Inc.'",
        ),
    ],
    ids=['no_window', 'unranked', 'one_gpu', 'one_maker', 'unknown_maker'],
)
def test_a_gpu_that_cannot_be_told_apart_is_not_judged_and_the_log_says_why(
    drawing_gpu, gpus, unjudged
):
    reading = _reading(drawing_gpu=drawing_gpu, gpus=gpus)

    assert judge(reading) is None
    assert f'not judged: {unjudged}' in log_line(reading, 0.0)


def test_each_makers_vendor_string_is_matched_by_whole_word():
    amd = GpuAdapter('AMD Radeon RX 6600M', 0x1002)
    # "Corporation" holds "ati" as letters, not as a word: NVIDIA's string is NVIDIA's.
    vendors = {
        'ATI Technologies Inc.': amd,
        'AMD': amd,
        'NVIDIA Corporation': NVIDIA,
        'Intel': INTEL,
    }

    drawn_elsewhere = {
        vendor: judge(_reading(drawing_gpu=DrawingGpu(vendor, 'r'), gpus=(INTEL, NVIDIA, amd)))
        for vendor, own in vendors.items()
    }
    on_its_own = {
        vendor: judge(_reading(drawing_gpu=DrawingGpu(vendor, 'r'), gpus=(own, INTEL, NVIDIA, amd)))
        for vendor, own in vendors.items()
        if own != INTEL
    }

    assert drawn_elsewhere['Intel'] is None
    assert all(drawn_elsewhere[v].preferred_gpu == INTEL for v in vendors if vendors[v] != INTEL)
    assert all(notice is None for notice in on_its_own.values())


def test_the_log_line_names_the_gpu_drawing_and_every_gpu_in_windows_order():
    line = log_line(_reading(drawing_gpu=ON_INTEL, gpus=(NVIDIA, INTEL)), 0.0)

    assert (
        'drawing on: Intel(R) HD Graphics 530 (Intel) | GPUs, high performance first: '
        'NVIDIA GeForce GTX 970M (0x10DE), Intel(R) HD Graphics 530 (0x8086) | '
        'not on the high-performance GPU'
    ) in line


# ------------------------------------------------------- the shares by process


def test_each_other_programs_share_is_of_the_whole_machine_over_the_window():
    lvp, idle = 500, 0
    before = {
        (lvp, 1.0): ('LumaViewPro.exe', 10.0),
        (idle, 0.0): ('System Idle Process', 1000.0),
        (7, 2.0): ('Updater.exe', 5.0),
        (8, 3.0): ('ended.exe', 1.0),
        (9, 4.0): ('old holder of pid 9', 2.0),
    }
    after = {
        (lvp, 1.0): ('LumaViewPro.exe', 30.0),
        (idle, 0.0): ('System Idle Process', 1040.0),
        (7, 2.0): ('Updater.exe', 23.0),
        (10, 5.0): ('started.exe', 4.0),
        (9, 6.0): ('new holder of pid 9', 3.0),
    }

    shares = common_utils.program_cpu_shares(before, after, 10.0, 4, frozenset({lvp, idle}))

    # 18 CPU seconds over 10 s on 4 cores is 45% of the machine.
    assert shares == [('Updater.exe', 7, 45.0)]


def test_a_window_with_no_length_has_no_shares():
    table = {(7, 2.0): ('Updater.exe', 5.0)}
    assert common_utils.program_cpu_shares(table, table, 0.0, 4, frozenset()) == []


_times = namedtuple('pcputimes', 'user system')


class _Process:
    """One process as psutil gives it: each fact a getter, None where the OS denies it."""

    def __init__(self, pid, name, created, cpu):
        self.pid = pid
        self._facts = {'name': name, 'create_time': created, 'cpu_times': cpu}

    def __getattr__(self, fact):
        value = self.__dict__['_facts'][fact]

        def getter():
            if isinstance(value, BaseException):
                raise value
            return value

        return getter


def _host_running(monkeypatch, *processes):
    """psutil's process walk over ``processes``: each asked fact read as ``as_dict`` reads it.

    A fact the OS denies reads None, as ``as_dict`` keeps an ``AccessDenied``;
    any other error ends the walk, as ``process_iter`` lets it out.
    """

    def process_iter(attrs):
        for proc in processes:
            proc.info = {fact: getattr(proc, fact)() for fact in attrs}
            yield proc

    monkeypatch.setattr(common_utils.psutil, 'process_iter', process_iter)


def test_a_process_the_os_will_not_read_is_counted_not_read(monkeypatch):
    # a stand-in by design: the host's process table, with a process the OS denies.
    _host_running(
        monkeypatch,
        _Process(7, 'Updater.exe', 2.0, _times(3.0, 2.0)),
        _Process(88, 'WindowServer', 1.0, None),
    )

    assert common_utils.process_cpu_seconds() == ({(7, 2.0): ('Updater.exe', 5.0)}, 1)


def test_a_process_whose_name_psutil_cannot_give_is_counted_by_its_pid(monkeypatch):
    """On macOS psutil reads a long-named process's command line for its full
    name, and for a process the OS would not give it that read raised
    ``SystemError`` once, which ends psutil's walk. The walk goes on;
    that process is counted, named by its pid. Before, the error left the
    startup reading and ``ScopeSession.create`` with it."""
    # a stand-in by design: the host's process table, with the SystemError
    # psutil 7.2.2's macOS command-line read raised, which no simulator makes.
    cut = SystemError('<built-in function proc_cmdline> returned a result with an exception set')
    _host_running(
        monkeypatch,
        _Process(7, 'Updater.exe', 2.0, _times(3.0, 2.0)),
        _Process(41, cut, 3.0, _times(1.0, 0.5)),
        _Process(90, 'Backup Helper', 4.0, _times(2.0, 2.0)),
    )

    assert common_utils.process_cpu_seconds() == (
        {
            (7, 2.0): ('Updater.exe', 5.0),
            (41, 3.0): ('pid 41', 1.5),
            (90, 4.0): ('Backup Helper', 4.0),
        },
        0,
    )


# ------------------------------------------------------------ the save drive


def test_a_save_folder_that_does_not_exist_is_unread_never_the_drive_above_it(tmp_path):
    unplugged = _REAL_READ_HOST(str(tmp_path / 'unplugged' / 'data'), read_gpus=False)
    assert unplugged.disk_free_bytes is None
    assert _REAL_READ_HOST(str(tmp_path), read_gpus=False).disk_free_bytes > 0


# ------------------------------------------------------- the session's reading


class _Host:
    """The PC each end of the window reads: roomy at create, ``settled`` at the motion's end."""

    def __init__(self, monkeypatch, **settled):
        self.settled = settled
        monkeypatch.setattr(host_resources, 'read_host', self.read)

    def read(self, save_folder, *, read_gpus):
        roomy = a_roomy_host(save_folder, read_gpus=read_gpus)
        if save_folder is None:
            return roomy
        return dataclasses.replace(roomy, at_s=10.0, **self.settled)


@pytest.fixture
def notices() -> list:
    return []


def _session(notices) -> ScopeSession:
    def heard(notification):
        if notification.title == TITLE:
            notices.append(notification)

    return ScopeSession.create(
        complete_settings(microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
        outcome_listener=heard,
    )


def _settle(motion) -> None:
    """Wait until every done-callback handed before this one has run."""
    settled = threading.Event()
    motion.add_done_callback(lambda done: settled.set())
    assert settled.wait(SETTLED_S), 'the startup motion did not settle'


def _homed(*args, **kwargs) -> None:
    return None


def _stopped(slot) -> None:
    raise MoveStoppedError('T', 'stopped')


def _failed(target) -> None:
    raise HomingFaultError('ALL', 'error', ('Z',))


def _defect(target) -> None:
    raise ValueError('a defect')


def _closing(target) -> None:
    raise SessionClosingError('home')


@pytest.mark.parametrize(
    'motion',
    [
        {},
        {'home_fn': _failed},
        {'disable_homing': True},
        {'home_fn': _homed, 'turret_fn': _stopped},
    ],
    ids=['homed', 'failed_home', 'skipped', 'stopped'],
)
def test_each_end_of_the_startup_motion_reads_the_pc_once_and_tells_what_is_past(
    monkeypatch, notices, motion
):
    _Host(monkeypatch, ram_available_bytes=GB)
    session = _session(notices)
    try:
        _settle(session.start_application_session(**motion))

        assert session.host_reading.ram_available_bytes == GB
        assert session.status.host_reading == session.host_reading
        assert len(notices) == 1 and 'With 1.0 GB of memory available' in notices[0].message
    finally:
        session.shutdown()


def test_a_pc_inside_every_line_is_read_and_told_nothing(notices):
    session = _session(notices)
    try:
        _settle(session.start_application_session(disable_homing=True))

        assert session.host_reading.disk_free_bytes == 500 * GB
        assert notices == []
    finally:
        session.shutdown()


def test_a_second_startup_motion_reads_nothing_and_tells_nothing(monkeypatch, notices):
    host = _Host(monkeypatch, ram_available_bytes=GB)
    session = _session(notices)
    try:
        _settle(session.start_application_session(disable_homing=True))
        first = session.host_reading
        host.settled = {'ram_available_bytes': 2 * GB}

        _settle(session.start_application_session(disable_homing=True))

        assert session.host_reading is first
        assert len(notices) == 1
    finally:
        session.shutdown()


@pytest.mark.parametrize('home_fn', [_defect, _closing], ids=['a_defect', 'the_close'])
def test_a_motion_a_defect_or_the_close_ended_reads_nothing_and_leaves_the_reading_open(
    monkeypatch, notices, home_fn
):
    _Host(monkeypatch, ram_available_bytes=GB)
    session = _session(notices)
    try:
        _settle(session.start_application_session(home_fn=home_fn))
        unread = session.host_reading

        _settle(session.start_application_session(disable_homing=True))

        assert unread is None
        assert session.host_reading is not None and len(notices) == 1
    finally:
        session.shutdown()


def test_a_start_that_is_refused_reads_nothing_and_leaves_the_reading_open(monkeypatch, notices):
    _Host(monkeypatch, ram_available_bytes=GB)
    session = _session(notices)
    try:
        held = threading.Event()
        release = threading.Event()

        def holding_home(target):
            held.set()
            assert release.wait(SETTLED_S)
            session.scope.motion.home(target, put_back=False)

        first = session.start_application_session(home_fn=holding_home)
        assert held.wait(SETTLED_S)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.start_application_session()
        release.set()
        _settle(first)

        assert refused.value.reason == 'home_in_flight'
        assert session.host_reading is not None and len(notices) == 1
    finally:
        session.shutdown()


def test_the_gpu_the_host_hands_create_is_judged_in_the_reading(monkeypatch, notices):
    _Host(monkeypatch, gpus=(NVIDIA, INTEL))
    session = ScopeSession.create(
        complete_settings(microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
        outcome_listener=lambda n: notices.append(n) if n.title == TITLE else None,
        drawing_gpu=ON_INTEL,
    )
    try:
        _settle(session.start_application_session(disable_homing=True))

        assert session.host_reading.drawing_gpu == ON_INTEL
        assert session.host_reading.gpus == (NVIDIA, INTEL)
        assert len(notices) == 1 and notices[0].message.startswith('LumaViewPro is drawing on')
    finally:
        session.shutdown()
