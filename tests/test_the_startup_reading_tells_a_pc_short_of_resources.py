# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A PC short of resources at startup is told so, once, when the startup motion ends.

A distributor's demo PC with 1.5 GB of memory available, 9.3 GB free and
another program holding the CPU reported LumaViewPro as slow, freezing
and dropping USB, and was told to check the cable. The session now reads
the PC over a window from its ``create`` to its startup motion's end,
keeps the reading (``ScopeSession.host_reading``, and ``status``), and
tells ``LowSystemResourcesNotice`` naming each thing past its line: under
4 GB of memory, under 50 GB free on the save drive, another program over
30% of the machine, or a window drawn on the integrated GPU of a PC
Windows types as hybrid, beside its discrete one (a laptop's Intel beside
its NVIDIA showed the live view at 12 fps against 42). Inside every line,
nothing is told.
"""

from __future__ import annotations

import concurrent.futures
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
    GpuHybridType,
    HostReading,
    ProgramCpu,
    hybrid_type_of,
    judge,
    log_line,
)
from _lumascope.modules.notification_center import OutcomeKind
from _lumascope.modules.scope_session import ScopeSession
from tests.conftest import a_roomy_host
from tests.settings_fixtures import complete_settings

GB = 2**30
TITLE = 'This PC may slow LumaViewPro'
# The laptop's two GPUs as its dxdiag types them: the Intel integrated, the NVIDIA discrete.
INTEL = GpuAdapter('Intel(R) HD Graphics 530', 0x8086, GpuHybridType.INTEGRATED)
NVIDIA = GpuAdapter('NVIDIA GeForce GTX 970M', 0x10DE, GpuHybridType.DISCRETE)
ON_INTEL = DrawingGpu('Intel', 'Intel(R) HD Graphics 530')
ON_NVIDIA = DrawingGpu('NVIDIA Corporation', 'NVIDIA GeForce GTX 970M/PCIe/SSE2')
SETTLED_S = 30.0

# The real reader, bound before the suite's fixture replaces it.
_REAL_READ_HOST = host_resources.read_host


def _reading(**changed) -> HostReading:
    """A reading inside every line, with ``changed`` applied."""
    roomy = HostReading(
        ram_available_bytes=16 * GB,
        live_folder='/data',
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
        _reading(ram_available_bytes=2 * GB, disk_free_bytes=None, live_folder='E:\\data')
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


def test_a_window_on_the_integrated_gpu_listed_first_is_told_the_discrete_one_without_the_figures():
    # The bench's case: at Power saving, Windows offered LumaViewPro the Intel first.
    notice = judge(_reading(drawing_gpu=ON_INTEL, gpus=(INTEL, NVIDIA)))

    assert notice.preferred_gpu == NVIDIA
    assert (notice.ram_low, notice.disk_low, notice.busy) == (False, False, ())
    assert str(notice).startswith(
        'LumaViewPro is drawing on the Intel(R) HD Graphics 530, not the NVIDIA GeForce GTX 970M, '
        "this PC's high-performance GPU"
    )
    assert 'Settings > System > Display > Graphics' in str(notice)
    assert 'At startup this PC had' not in str(notice)
    assert notice.title == TITLE


def test_the_order_windows_lists_the_gpus_in_does_not_change_the_answer():
    listed_first = {
        gpus: judge(_reading(drawing_gpu=ON_INTEL, gpus=gpus))
        for gpus in ((INTEL, NVIDIA), (NVIDIA, INTEL))
    }
    on_the_nvidia = [
        judge(_reading(drawing_gpu=ON_NVIDIA, gpus=gpus))
        for gpus in ((INTEL, NVIDIA), (NVIDIA, INTEL))
    ]

    assert {notice.preferred_gpu for notice in listed_first.values()} == {NVIDIA}
    assert on_the_nvidia == [None, None]


def test_the_gpu_and_low_memory_are_told_in_one_notice_the_gpu_first():
    notice = judge(_reading(drawing_gpu=ON_INTEL, gpus=(INTEL, NVIDIA), ram_available_bytes=GB))
    words = str(notice)

    assert notice.preferred_gpu == NVIDIA and notice.ram_low
    assert words.index('LumaViewPro is drawing on') < words.index('At startup this PC had 1.0 GB')


_AMD_EGPU = GpuAdapter('AMD Radeon RX 7900 XT', 0x1002, GpuHybridType.NEITHER)


# parametrize-justified: each row is its own case of the judge, with its own reason.
@pytest.mark.parametrize(
    'drawing_gpu, gpus, unjudged',
    [
        (None, (INTEL, NVIDIA), 'no window draws'),
        (ON_INTEL, None, 'no list of the GPUs on this system'),
        (
            ON_INTEL,
            (
                GpuAdapter('NVIDIA GeForce RTX 3060', 0x10DE, GpuHybridType.NEITHER),
                GpuAdapter('Intel(R) UHD Graphics 770', 0x8086, GpuHybridType.NEITHER),
            ),
            'not a hybrid PC (Windows types 0 GPUs integrated and 0 discrete)',
        ),
        (
            ON_INTEL,
            (INTEL, GpuAdapter('Intel(R) Arc(TM) A370M', 0x8086, GpuHybridType.DISCRETE)),
            'the integrated and discrete GPUs are of one maker',
        ),
        (
            DrawingGpu('Google Inc.', 'ANGLE (Intel HD 530)'),
            (INTEL, NVIDIA),
            "no known maker in 'Google Inc.'",
        ),
        (
            DrawingGpu('ATI Technologies Inc.', 'AMD Radeon RX 7900 XT'),
            (INTEL, NVIDIA, _AMD_EGPU),
            'drawing on a GPU of neither the integrated nor the discrete maker',
        ),
        (
            ON_INTEL,
            (INTEL, NVIDIA, dataclasses.replace(_AMD_EGPU, hybrid=GpuHybridType.UNREAD)),
            'the type of AMD Radeon RX 7900 XT was not read',
        ),
        (
            ON_INTEL,
            (INTEL, dataclasses.replace(NVIDIA, hybrid=GpuHybridType.BOTH)),
            'NVIDIA GeForce GTX 970M is typed integrated and discrete',
        ),
    ],
    ids=[
        'no_window',
        'unlisted',
        'desktop_of_two_cards',
        'one_maker',
        'unknown_maker',
        'egpu',
        'type_unread',
        'typed_both',
    ],
)
def test_a_gpu_that_cannot_be_told_apart_is_not_judged_and_the_log_says_why(
    drawing_gpu, gpus, unjudged
):
    reading = _reading(drawing_gpu=drawing_gpu, gpus=gpus)

    assert judge(reading) is None
    assert f'not judged: {unjudged}' in log_line(reading, 0.0)


def test_each_makers_vendor_string_is_matched_by_whole_word():
    amd_integrated = GpuAdapter('AMD Radeon(TM) Graphics', 0x1002, GpuHybridType.INTEGRATED)
    nvidia_beside = GpuAdapter('NVIDIA GeForce RTX 4060', 0x10DE, GpuHybridType.DISCRETE)
    on_amd = (amd_integrated, nvidia_beside)

    told = {
        vendor: judge(_reading(drawing_gpu=DrawingGpu(vendor, 'r'), gpus=gpus))
        for vendor, gpus in {
            'ATI Technologies Inc.': on_amd,
            'AMD': on_amd,
            'Intel': (INTEL, NVIDIA),
            'Intel Corporation': (INTEL, NVIDIA),
        }.items()
    }
    # "Corporation" holds "ati" as letters, not as a word: no maker is named.
    no_maker = _reading(drawing_gpu=DrawingGpu('Microsoft Corporation', 'r'), gpus=on_amd)

    assert {vendor: notice.preferred_gpu for vendor, notice in told.items()} == {
        'ATI Technologies Inc.': nvidia_beside,
        'AMD': nvidia_beside,
        'Intel': NVIDIA,
        'Intel Corporation': NVIDIA,
    }
    assert judge(_reading(drawing_gpu=DrawingGpu('NVIDIA Corporation', 'r'), gpus=on_amd)) is None
    assert judge(no_maker) is None
    assert "not judged: no known maker in 'Microsoft Corporation'" in log_line(no_maker, 0.0)


def test_the_log_line_names_the_gpu_drawing_and_every_gpu_typed_in_windows_order():
    line = log_line(_reading(drawing_gpu=ON_INTEL, gpus=(INTEL, NVIDIA)), 0.0)

    assert (
        'drawing on: Intel(R) HD Graphics 530 (Intel) | GPUs, as Windows offers them to '
        'LumaViewPro: Intel(R) HD Graphics 530 (0x8086, integrated), '
        'NVIDIA GeForce GTX 970M (0x10DE, discrete) | not on the high-performance GPU'
    ) in line


def test_the_adapter_type_bits_name_its_hybrid_type():
    render, display = 1 << 0, 1 << 1
    decoded = {
        value: hybrid_type_of(value)
        for value in (
            0,
            render | display,
            render | 1 << 4,
            render | display | 1 << 5,
            1 << 4 | 1 << 5,
        )
    }

    assert decoded == {
        0: GpuHybridType.NEITHER,
        render | display: GpuHybridType.NEITHER,
        render | 1 << 4: GpuHybridType.DISCRETE,
        render | display | 1 << 5: GpuHybridType.INTEGRATED,
        1 << 4 | 1 << 5: GpuHybridType.BOTH,
    }


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
    """Wait for the startup motion's operation to be done, and for nothing after it."""
    done, _ = concurrent.futures.wait([motion], timeout=SETTLED_S)
    assert done, 'the startup motion did not settle'


class _WatchedHost(_Host):
    """``_Host``, whose motion's-end read waits for the motion in hand and notes whether it was done."""

    def __init__(self, monkeypatch, **settled):
        super().__init__(monkeypatch, **settled)
        self.motion = None
        self.in_hand = threading.Event()
        self.done_at_the_read: list[bool] = []

    def read(self, save_folder, *, read_gpus):
        if save_folder is not None:
            assert self.in_hand.wait(SETTLED_S)
            self.done_at_the_read.append(self.motion.done())
        return super().read(save_folder, read_gpus=read_gpus)


def _homed(*args, **kwargs) -> None:
    return None


def _stopped(slot) -> None:
    raise MoveStoppedError('T', 'stopped')


def _failed(target) -> None:
    raise HomingFaultError('ALL', 'driver_failed', ('Z',))


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


# The motion's task and a skip's are two bodies, each its own lane task.
@pytest.mark.parametrize(
    'motion',
    [{}, {'disable_homing': True}],
    ids=['homed', 'skipped'],
)
def test_a_caller_told_the_motion_has_ended_reads_the_pcs_reading(monkeypatch, notices, motion):
    """The startup motion's operation is done only once ``host_reading`` is kept.

    The reading was once taken in a done-callback, after a caller waiting on
    the motion had woken: ``host_reading`` was None on the line after
    ``run_application_session()``.
    """
    host = _WatchedHost(monkeypatch, ram_available_bytes=GB)
    session = _session(notices)
    try:
        host.motion = session.start_application_session(**motion)
        host.in_hand.set()
        host.motion.result(timeout=SETTLED_S)

        assert host.done_at_the_read == [False]
        assert session.host_reading.ram_available_bytes == GB
    finally:
        session.shutdown()


def test_a_read_windows_refuses_is_reported_and_the_motion_still_ends(monkeypatch):
    """A GPU list Windows refuses leaves no reading; the startup motion is not failed by it.

    The read raises ``OSError`` when DXGI fails; taken on the motion's lane,
    it once failed the motion, so ``run_application_session()`` raised it
    after a good home.
    """

    def refused(save_folder, *, read_gpus):
        if save_folder is None:
            return a_roomy_host(save_folder, read_gpus=read_gpus)
        raise OSError('CreateDXGIFactory1 failed: 0x887a0004')

    monkeypatch.setattr(host_resources, 'read_host', refused)
    heard = []
    session = ScopeSession.create(
        complete_settings(microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
        outcome_listener=heard.append,
    )
    try:
        session.run_application_session()

        assert session.host_reading is None
        assert [(n.category, n.kind) for n in heard if n.category == 'Host'] == [
            ('Host', OutcomeKind.FAULT)
        ]
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
