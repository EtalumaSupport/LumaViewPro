# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each metrics line is written on its own, and a probe that fails says so.

``config_helpers.log_system_metrics`` writes one line per group the host
has, in a fixed order and fixed words: support reads ``metrics.log`` by its
tags. A group that does not apply on the host is left out. A group whose
probe fails is written as ``[<TAG>] unavailable: <type>: <message>`` at
WARNING, so it reaches the errors log, and every other line of the tick is
still written. A field whose reading needs an earlier one reads ``NA`` until
it has one, never 0.0. The tick measures the live folder as stored and never
creates it.

The host here is stubbed below the lines, at psutil, the process handle and
the Windows helpers: a stand-in by design, since the subject is the words
each reading becomes, and the Windows counters and the failures are not ones
this host can be made to produce (the simulator has no host to fault).
"""

import logging
import types
from collections import namedtuple

import psutil
import pytest

from _lumascope.modules import common_utils, config_helpers
from _lumascope.modules.lumascope_api.imaging import DeliveredRate
from tests.log_capture import records_of

# The parent's 20 lines, captured at LVP eb9b879d8 through these same
# readings on the second of two ticks, before each line was its own function.
GOLDEN = [
    '[SYSTEM METRICS] CPU Usage: 12.3% | RAM Available: 7.6 GB | RAM Usage: 45.6% | uptime_hr=98.7',
    '[DISK METRICS] Disk Free: 321.0 GB | Disk Usage: 54.3%',
    '[PROCESS METRICS] Process CPU Usage: 23.4% | Process RAM Usage: 512.3 MB, 3.2% | Private: 678.9 MB',
    '[EXTRA DISKS] D: 111.1 GB free (22.2% used)',
    '[GDI METRICS] gdi=345 | user=210',
    '[HANDLE METRICS] handles=1234 | open_files=17',
    '[THREAD METRICS] count=9 | MainThread=1, ThreadPoolExecutor=2, scope_display=1',
    '[THREAD CPU] total=35.8% | MainThread=31.5% scope_display=4.2%',
    '[GPU METRICS] util=7.5% (3d=5.2%) | shared_mem=123 MB | dedicated_mem=46 MB',
    '[GC METRICS] blocks=456789 | gen0=101 gen1=12 gen2=3',
    '[SWAP METRICS] used=2.3 GB (11.1%)',
    '[PROCESS IO] read=1600.0 MB | write=3200.0 MB | read_rate=10.00 MB/s | write_rate=20.00 MB/s',
    '[PAGE FAULTS] total=53000 | rate=50.0/s',
    '[PDH METRICS] standby_total=1230 MB (normal=1000 reserve=200 core=30) | nonpaged_pool=400 MB'
    ' | paged_pool=500 MB | sys_cache=600 MB | modified=70 MB | free_zero=800 MB | available=900 MB'
    ' | commit=9000/16000 MB',
    '[BUFFER METRICS] capture_fps=25.0 | display_fps=29.5 | camera_data_rate=50.0 MB/s'
    ' | frame_size=2000 KB',
    '[FRAME INTERVAL] p50=33.3 ms | p95=41.2 ms | p99=55.5 ms | max=70.1 ms | n=300',
    '[DEFENDER METRICS] private=210 MB | rss=150 MB | io_read_total=999 MB | io_read_rate=1.25 MB/s',
    '[GC PRESSURE] gen0_depth=701 | gen1_depth=8 | gen2_depth=2',
    '[QUEUE METRICS] io=3 | protocol=0 | io_thread=busy',
    '[FUTURES METRICS] io: alloc=10 pop=9 live=1 | protocol: alloc=5 pop=5 live=0',
]

_MIB = 1024 * 1024
_Mem = namedtuple('_Mem', 'rss vms num_page_faults private')
_IO = namedtuple('_IO', 'read_bytes write_bytes')
_Disk = namedtuple('_Disk', 'total used free percent')
_VM = namedtuple('_VM', 'total available percent used')
_Swap = namedtuple('_Swap', 'total used free percent sin sout')
_CpuTimes = namedtuple('_CpuTimes', 'user nice system idle')


class _Host:
    """One host's readings on the first and the second of two ticks,
    sixty seconds apart."""

    def __init__(self):
        self.tick = 1
        # An attribute, so a test can take it away as macOS's psutil does.
        self.io_counters = lambda: _IO(self.on(1000e6, 1600e6), self.on(2000e6, 3200e6))

    def on(self, first, second):
        return first if self.tick == 1 else second

    # The process handle.
    def cpu_times(self):
        return types.SimpleNamespace(user=self.on(10.0, 10.0 + 14.04), system=0.0)

    def memory_percent(self):
        return 3.2

    def memory_info(self):
        return _Mem(512.3e6, 0, self.on(50_000, 53_000), 678.9e6)

    def num_handles(self):
        return 1234

    def open_files(self):
        return [object()] * 17


class _Executor:
    def __init__(self, stats):
        self.stats = stats

    def caller_futures_stats(self):
        return self.stats


@pytest.fixture
def host(monkeypatch, tmp_path, sim_scope):
    """Every reading the tick takes, stubbed to distinct values, so a
    swapped field or unit shows in the words."""
    host = _Host()
    monkeypatch.setattr(common_utils, '_rate_state', {})
    monkeypatch.setattr(common_utils, '_system_cpu_last', {})
    monkeypatch.setattr(common_utils, 'self_process', lambda: host)
    monkeypatch.setattr(
        common_utils.psutil,
        'cpu_times',
        lambda: host.on(_CpuTimes(100.0, 0.0, 50.0, 850.0), _CpuTimes(200.0, 0.0, 73.0, 1727.0)),
    )
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _VM(0, 7.61e9, 45.6, 0))
    monkeypatch.setattr(psutil, 'swap_memory', lambda: _Swap(0, 2.34e9, 0, 11.1, 0, 0))
    real_disk_usage = psutil.disk_usage

    def disk_usage(path):
        real_disk_usage(path)  # a folder that cannot be read raises as it does
        return _Disk(0, 0, 321 * 1024**3, 54.3)

    monkeypatch.setattr(psutil, 'disk_usage', disk_usage)
    monkeypatch.setattr(common_utils._time, 'monotonic', lambda: host.on(1000.0, 1060.0))
    monkeypatch.setattr(common_utils, 'host_boot', lambda: common_utils.HostBoot(0.0, 98.7))
    monkeypatch.setattr(
        common_utils, 'get_extra_disks_info', lambda exclude_path: 'D: 111.1 GB free (22.2% used)'
    )
    monkeypatch.setattr(common_utils, 'gui_resources', lambda: (345, 210))
    names = ['MainThread', 'ThreadPoolExecutor-0_0', 'ThreadPoolExecutor-0_1', 'scope_display']
    monkeypatch.setattr(
        config_helpers,
        'threading',
        types.SimpleNamespace(
            active_count=lambda: 9,
            enumerate=lambda: [types.SimpleNamespace(name=n) for n in names],
        ),
    )
    monkeypatch.setattr(
        config_helpers, 'sys', types.SimpleNamespace(getallocatedblocks=lambda: 456789)
    )
    monkeypatch.setattr(
        config_helpers,
        'gc',
        types.SimpleNamespace(
            get_stats=lambda: [{'collections': 101}, {'collections': 12}, {'collections': 3}],
            get_count=lambda: (701, 8, 2),
        ),
    )
    monkeypatch.setattr(
        common_utils,
        'thread_cpu_percentages',
        lambda: host.on(None, {'MainThread': 31.5, 'scope_display': 4.25, 'tid_77': 0.05}),
    )
    monkeypatch.setattr(
        common_utils, 'query_gpu_metrics', lambda: common_utils.GpuReading(7.5, 5.25, 123.4, 45.6)
    )
    pdh_mb = {
        'standby_normal_bytes': 1000,
        'standby_reserve_bytes': 200,
        'standby_core_bytes': 30,
        'pool_nonpaged_bytes': 400,
        'pool_paged_bytes': 500,
        'system_cache_bytes': 600,
        'modified_page_bytes': 70,
        'free_zero_bytes': 800,
        'available_bytes': 900,
        'commit_bytes': 9000,
        'commit_limit_bytes': 16000,
    }
    monkeypatch.setattr(
        common_utils,
        'query_windows_perf_counters',
        lambda: {k: float(v * _MIB) for k, v in pdh_mb.items()},
    )
    monkeypatch.setattr(
        common_utils,
        'query_defender_metrics',
        lambda: common_utils.DefenderReading(210.0, 150.0, 999.0, host.on(None, 1.25)),
    )
    # The simulated scope, its delivered rate fixed so the line's words are known.
    host.scope = sim_scope
    monkeypatch.setattr(
        type(sim_scope.imaging),
        'delivered_rate',
        property(lambda self: DeliveredRate(frame_rate_fps=25.0, bytes_per_s=5.0e7)),
    )
    host.executors = types.SimpleNamespace(
        snapshot=lambda: {'io': 3, 'protocol': 0},
        thread_states=lambda: {'io_thread': 'busy'},
        queued_executors=lambda: [
            ('io', _Executor((10, 9, 1))),
            ('protocol', _Executor((5, 5, 0))),
        ],
    )
    host.display = types.SimpleNamespace(
        display_fps=lambda: 29.5,
        frame_interval_percentiles_ms=lambda: {
            'p50': 33.3,
            'p95': 41.2,
            'p99': 55.5,
            'max': 70.1,
            'n': 300,
        },
    )
    host.settings = {'live_folder': str(tmp_path), 'profiling': {'open_files_enabled': True}}
    return host


def _tick(caplog, host) -> list[tuple[int, str]]:
    """One tick's metrics lines, as (level, text)."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=config_helpers.metrics_logger.name):
        config_helpers.log_system_metrics(
            host.settings, scope=host.scope, executors=host.executors, display=host.display
        )
    return [
        (r.levelno, r.getMessage())
        for r in records_of(caplog, config_helpers)
        if r.name == config_helpers.metrics_logger.name
    ]


def _two_ticks(caplog, host) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    first = _tick(caplog, host)
    host.tick = 2
    return first, _tick(caplog, host)


def _line(lines, tag) -> tuple[int, str]:
    (found,) = [line for line in lines if line[1].startswith(f'[{tag}]')]
    return found


def _tags(lines) -> list[str]:
    return [text.split(']', 1)[0] + ']' for _, text in lines]


def test_every_line_keeps_its_words_once_it_has_a_baseline(caplog, host):
    _, second = _two_ticks(caplog, host)
    assert [text for _, text in second] == GOLDEN
    assert {level for level, _ in second} == {logging.INFO}


def test_the_first_tick_reads_na_where_a_reading_needs_an_earlier_one(caplog, host):
    first, _ = _two_ticks(caplog, host)
    assert 'CPU Usage: NA% |' in _line(first, 'SYSTEM METRICS')[1]
    assert 'Process CPU Usage: NA% |' in _line(first, 'PROCESS METRICS')[1]
    assert _line(first, 'PROCESS IO')[1].endswith('read_rate=NA MB/s | write_rate=NA MB/s')
    assert _line(first, 'PAGE FAULTS')[1] == '[PAGE FAULTS] total=50000 | rate=NA/s'
    assert _line(first, 'DEFENDER METRICS')[1].endswith('io_read_rate=NA MB/s')
    assert '[THREAD CPU]' not in _tags(first)
    assert {level for level, _ in first} == {logging.INFO}


def test_a_host_where_groups_do_not_apply_writes_no_line_for_them(caplog, host, monkeypatch):
    for helper in ('gui_resources', 'query_windows_perf_counters', 'query_gpu_metrics'):
        monkeypatch.setattr(common_utils, helper, lambda: None)
    monkeypatch.setattr(common_utils, 'query_defender_metrics', lambda: None)
    monkeypatch.setattr(common_utils, 'get_extra_disks_info', lambda exclude_path: None)
    del host.io_counters
    host.display = None

    lines = _tick(caplog, host)

    absent = {
        '[EXTRA DISKS]',
        '[GDI METRICS]',
        '[PDH METRICS]',
        '[GPU METRICS]',
        '[DEFENDER METRICS]',
        '[PROCESS IO]',
        '[THREAD CPU]',
        '[FRAME INTERVAL]',
    }
    assert _tags(lines) == [tag for tag in _tags([(0, t) for t in GOLDEN]) if tag not in absent]
    assert not any('unavailable' in text for _, text in lines)
    assert {level for level, _ in lines} == {logging.INFO}


def _raising(error):
    def probe(*args, **kwargs):
        raise error

    return probe


def test_a_failed_probe_names_itself_at_warning_and_the_tick_goes_on(caplog, host, monkeypatch):
    monkeypatch.setattr(psutil, 'swap_memory', _raising(OSError('swap file unreadable')))
    lines = _tick(caplog, host)
    assert _line(lines, 'SWAP METRICS') == (
        logging.WARNING,
        '[SWAP METRICS] unavailable: OSError: swap file unreadable',
    )
    assert _tags(lines)[-2:] == ['[QUEUE METRICS]', '[FUTURES METRICS]']


def test_a_raise_in_any_line_no_longer_ends_the_tick(caplog, host, monkeypatch):
    raising = _raising(RuntimeError('camera gone'))
    monkeypatch.setattr(
        type(host.scope.imaging), 'delivered_rate', property(lambda self: raising())
    )
    lines = _tick(caplog, host)
    assert _line(lines, 'BUFFER METRICS') == (
        logging.WARNING,
        '[BUFFER METRICS] unavailable: RuntimeError: camera gone',
    )
    assert '[QUEUE METRICS]' in _tags(lines)


def test_an_unreadable_boot_time_is_its_lines_failure_and_the_tick_goes_on(
    caplog, host, monkeypatch
):
    monkeypatch.setattr(common_utils, 'host_boot', _raising(psutil.Error()))
    lines = _tick(caplog, host)
    # An empty message adds nothing after the type.
    assert _line(lines, 'SYSTEM METRICS') == (
        logging.WARNING,
        '[SYSTEM METRICS] unavailable: psutil.Error',
    )
    assert len(lines) == len(GOLDEN) - 1  # the first tick has no [THREAD CPU]


def test_a_live_folder_that_is_not_there_is_not_replaced_or_created(caplog, host, tmp_path):
    missing = tmp_path / 'unmounted' / 'capture'
    host.settings['live_folder'] = str(missing)
    lines = _tick(caplog, host)
    level, text = _line(lines, 'DISK METRICS')
    assert level == logging.WARNING
    assert text.startswith('[DISK METRICS] unavailable: FileNotFoundError'), text
    assert not missing.parent.exists()
    assert '[QUEUE METRICS]' in _tags(lines)


def test_a_failed_pdh_start_is_named_on_every_tick(caplog, host, monkeypatch):
    # The real query, whose start fails as it does where pdh.dll will not load.
    monkeypatch.setattr(
        common_utils.ctypes, 'WinDLL', _raising(OSError('pdh.dll not found')), raising=False
    )
    monkeypatch.setattr(common_utils, '_pdh_counters_singleton', common_utils._PdhCountersOnce())
    monkeypatch.setattr(
        common_utils, 'query_windows_perf_counters', common_utils._pdh_counters_singleton.query
    )
    first, second = _two_ticks(caplog, host)
    for lines in (first, second):
        assert _line(lines, 'PDH METRICS') == (
            logging.WARNING,
            '[PDH METRICS] unavailable: OSError: pdh.dll not found',
        )


def test_a_pdh_counter_that_fails_its_read_is_named_in_its_field(caplog, host, monkeypatch):
    readings = common_utils.query_windows_perf_counters()
    readings['standby_core_bytes'] = common_utils.CounterFailed(0xC0000BC6)
    del readings['pool_paged_bytes']  # a counter this Windows build lacks
    monkeypatch.setattr(common_utils, 'query_windows_perf_counters', lambda: readings)
    level, text = _line(_tick(caplog, host), 'PDH METRICS')
    assert level == logging.WARNING
    assert text.startswith(
        '[PDH METRICS] standby_total=unavailable: 0xc0000bc6 '
        '(normal=1000 reserve=200 core=unavailable: 0xc0000bc6) | nonpaged_pool=400 MB'
        ' | paged_pool=NA MB |'
    ), text


class _FailingArrayRead:
    """The GPU query started, with its array read failing as PDH does."""

    def __init__(self, status):
        self.status = status

    def __call__(self, *args):
        return self.status


def test_a_gpu_read_that_fails_is_not_an_idle_gpu(caplog, host, monkeypatch):
    gpu = common_utils._GpuPdhCountersOnce()
    gpu._initialized = True
    gpu._PdhCollectQueryData = lambda query: 0
    gpu._counters = {'util': object()}
    gpu._PdhGetFormattedCounterArrayW = _FailingArrayRead(0xC0000BBC)
    monkeypatch.setattr(common_utils, 'query_gpu_metrics', gpu.query)
    assert _line(_tick(caplog, host), 'GPU METRICS') == (
        logging.WARNING,
        '[GPU METRICS] unavailable: OSError: PdhGetFormattedCounterArrayW failed: 0xc0000bbc',
    )


def test_a_gpu_instance_that_fails_is_named_in_its_field(caplog, host, monkeypatch):
    monkeypatch.setattr(
        common_utils,
        'query_gpu_metrics',
        lambda: common_utils.GpuReading(common_utils.CounterFailed(0xC0000BBA), 5.25, 123.4, None),
    )
    assert _line(_tick(caplog, host), 'GPU METRICS') == (
        logging.WARNING,
        '[GPU METRICS] util=unavailable: 0xc0000bba (3d=5.2%) | shared_mem=123 MB'
        ' | dedicated_mem=NA MB',
    )


@pytest.mark.parametrize(
    ('failing', 'expected'),
    [
        (
            'open_files',
            '[HANDLE METRICS] handles=1234 | open_files=unavailable: psutil.AccessDenied',
        ),
        (
            'num_handles',
            '[HANDLE METRICS] handles=unavailable: psutil.AccessDenied | open_files=17',
        ),
    ],
)
def test_each_handle_field_outlives_the_others_failure(
    caplog, host, monkeypatch, failing, expected
):
    # parametrize-justified: the two directions are two contracts, neither
    # reading hiding the other.
    monkeypatch.setattr(host, failing, _raising(psutil.AccessDenied()), raising=False)
    assert _line(_tick(caplog, host), 'HANDLE METRICS') == (logging.WARNING, expected)


def test_a_refused_defender_io_read_keeps_its_memory(caplog, host, monkeypatch):
    refused = psutil.AccessDenied()
    monkeypatch.setattr(
        common_utils,
        'query_defender_metrics',
        lambda: common_utils.DefenderReading(210.0, 150.0, refused, refused),
    )
    assert _line(_tick(caplog, host), 'DEFENDER METRICS') == (
        logging.WARNING,
        '[DEFENDER METRICS] private=210 MB | rss=150 MB'
        ' | io_read_total=unavailable: psutil.AccessDenied'
        ' | io_read_rate=unavailable: psutil.AccessDenied',
    )


def test_the_gpu_reading_sums_this_processs_instances_and_names_a_failed_one(monkeypatch):
    # The real query, its PDH reads faked: the Windows counters are not on
    # this host (a stand-in by design, the simulator has none).
    gpu = common_utils._GpuPdhCountersOnce()
    gpu._initialized = True
    gpu._PdhCollectQueryData = lambda query: 0
    gpu._counters = {'util': 'util', 'shared_mem': 'shared'}
    mine = gpu._pid_tag
    arrays = {
        'util': [
            (f'{mine}eng_0_engtype_3D', 0, 5.0),
            (f'{mine}eng_1_engtype_Copy', 1, 2.5),
            ('pid_1_eng_0_engtype_3D', 0, 90.0),  # another process
        ],
        'shared': [(f'{mine}phys_0', 0, 3 * _MIB), (f'{mine}phys_1', 0xC0000BC6, 0.0)],
    }
    monkeypatch.setattr(gpu, '_read_array', lambda handle: arrays[handle])
    assert gpu.query() == common_utils.GpuReading(
        util_total_percent=7.5,
        util_3d_percent=5.0,
        shared_mem_mb=common_utils.CounterFailed(0xC0000BC6),
        dedicated_mem_mb=None,
    )


class _Defender:
    """MsMpEng as psutil answers it to a process Windows refuses its I/O."""

    def is_running(self):
        return True

    def memory_info(self):
        return types.SimpleNamespace(rss=150 * _MIB, private=210 * _MIB)

    def io_counters(self):
        raise psutil.AccessDenied()


def test_a_refused_defender_io_read_is_its_fields_refusal(monkeypatch):
    # a stand-in by design: MsMpEng runs only on Windows.
    monkeypatch.setattr(common_utils, '_IS_WINDOWS', True)
    monkeypatch.setattr(common_utils, '_defender_process_cache', {'process': _Defender()})
    reading = common_utils.query_defender_metrics()
    assert (reading.private_mb, reading.rss_mb) == (210.0, 150.0)
    assert isinstance(reading.io_read_total_mb, psutil.AccessDenied)
    assert isinstance(reading.io_read_mbps, psutil.AccessDenied)
