# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import ctypes
import enum
import gc
import json
import os
import pathlib
import platform
import re
import threading
import time as _time

import numpy as np
import psutil

from lvp_logger import logger


# ---------------------------------------------------------------------------
# Hardware defaults (fallbacks when motorconfig/scope not available)
# ---------------------------------------------------------------------------
# LS850 full travel range -- used as default stage limits when scope is not connected.
DEFAULT_STAGE_TRAVEL_UM = {'x': 120000.0, 'y': 80000.0}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@enum.unique
class ColorChannel(enum.Enum):
    Blue = 0
    Green = 1
    Red = 2
    BF = 3
    PC = 4
    DF = 5
    Lumi = 6


@enum.unique
class PostFunction(enum.Enum):
    COMPOSITE = 'Composite'
    STITCHED = 'Stitched'
    ZPROJECT = 'ZProject'
    VIDEO = 'Video'
    HYPERSTACK = 'Hyperstack'

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def generate_default_step_name(
    well_label,
    color=None,
    z_height_idx=None,
    scan_count=None,
    tile_label=None,
    objective_short_name=None,
    custom_name_prefix=None,
    stitched: bool = False,
    video: bool = False,
    zprojection: str | None = None,
    stack: bool = False,
    hyperstack: bool = False,
    turret_position: int | None = None,
):
    if custom_name_prefix not in (None, ''):
        name = f'{custom_name_prefix}'
    else:
        name = f'{well_label}'

    if color not in (None, '') and color not in name:
        name = f'{name}_{color}'

    if tile_label not in (None, '', -1):
        if f'_T{tile_label}' not in name:
            name = f'{name}_T{tile_label}'

    if objective_short_name not in (None, '', -1):
        name = f'{name}_{objective_short_name}'

    if turret_position is not None:
        name = f'{name}_Turret{turret_position}'

    if z_height_idx not in (None, '', -1):
        if f'_Z{z_height_idx}' not in name:
            name = f'{name}_Z{z_height_idx}'

    DESIRED_SCAN_COUNT_DIGITS = 4
    if scan_count not in (None, ''):
        name = f'{name}_{scan_count:0>{DESIRED_SCAN_COUNT_DIGITS}}'

    if stitched:
        name = f'{name}_stitched'

    if video:
        name = f'{name}_video'

    if zprojection is not None:
        name = f'{name}_zproj_{zprojection}'

    if stack:
        name = f'{name}_stack'

    if hyperstack:
        name = f'{name}_hyperstack'

    return name


def resolve_step_rename(raw_text: str, sanitize) -> str | None:
    """Resolve a step-name field value to the name to persist, or None.

    Auto-named custom steps blank the name field so the default name shows
    as a hint placeholder rather than editable text. A blank field
    therefore means "no rename intended": persisting the empty string
    would wipe the auto-assigned name, leaving added steps unnamed and
    colliding on the same default name. Returns None for a blank field so
    callers keep the existing name; a non-empty entry is a real rename and
    is returned sanitized.

    Args:
        raw_text: the raw text from the step-name input field.
        sanitize: callable that cleans a name (e.g. strips invalid chars).

    Returns:
        The sanitized name to persist, or None if the field is blank.
    """
    cleaned = sanitize(raw_text)
    return cleaned if cleaned else None


def get_tile_label_from_name(name: str) -> str | None:
    name = name.split('_')

    if len(name) <= 2:
        return None

    segment = name[2]
    if segment.startswith('T'):
        return segment[1:]

    return None


def get_first_section_from_name(name: str) -> str | None:

    # This will retrieve just the filename if the name has parent folders
    name = pathlib.Path(name).name

    name = name.split('_')
    return name[0]


def get_layer_from_name(name: str) -> str | None:
    name = name.split('_')

    return name[1]


def replace_layer_in_step_name(step_name: str, new_layer_name: str) -> str | None:

    # Extract basename in case we are handling protocol with separate folders per channel
    base_name = os.path.basename(step_name)
    # if is_custom_name(name=base_name):
    #     return None

    # This replaces the parent folder when using per-channel folders for protocol runs
    split_name = list(os.path.split(step_name))
    if len(split_name) == 2:
        using_per_channel_folders = True
    else:
        using_per_channel_folders = False

    if using_per_channel_folders:
        split_name[0] = new_layer_name
        step_name = str(pathlib.Path(split_name[0]) / split_name[1])

    step_name_segments = step_name.split('_')

    # Confirm it's actually a layer before replacing it
    if step_name_segments[1] in get_layers():
        step_name_segments[1] = new_layer_name

    return '_'.join(step_name_segments)


def is_custom_name(name: str) -> bool:

    # This will retrieve just the filename if name includes parent folders
    name = pathlib.Path(name).name

    name = name.split('_')

    # All generated names have at least one '_'
    if len(name) <= 1:
        return True

    well = name[0]
    well_pattern = r'^[A-Z]{1,2}[0-9]+$'
    if not re.match(pattern=well_pattern, string=well):
        return True

    color = name[1]
    if color not in get_layers():
        return True

    return False


def get_z_slice_from_name(name: str) -> int | None:
    name = name.split('_')

    # Z-slice info can either be at segment index 2 (if no tile label is present), or segment index 3 (if tile label is present)
    if len(name) <= 2:
        return None

    if name[2].startswith('Z'):
        return name[2][1:]

    if len(name) <= 3:
        return None

    if name[3].startswith('Z'):
        return name[3][1:]

    return None


def convert_zstack_reference_position_setting_to_config(text_label: str) -> str:
    LABEL_MAP = {
        'Current Position at Top': 'top',
        'Current Position at Center': 'center',
        'Current Position at Bottom': 'bottom',
    }

    if text_label in LABEL_MAP:
        return LABEL_MAP[text_label]

    raise Exception(f'Unknown Z-stack position reference: {text_label}')


def get_layers() -> list[str]:
    return ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi']


def get_transmitted_layers() -> list[str]:
    return ['BF', 'PC', 'DF']


def get_fluorescence_layers() -> list[str]:
    return ['Blue', 'Green', 'Red']


def get_bright_background_layers() -> list[str]:
    """Layers whose field of view is bright, so overlays drawn on them
    (scale bar, annotation text) must be dark to stay visible. Brightfield
    and phase contrast have a bright background; darkfield does not -- it
    shows bright subjects on a dark field -- so it is excluded here even
    though it is a transmitted-light mode.
    """
    return ['BF', 'PC']


def get_luminescence_layers() -> list[str]:
    return ['Lumi']


def get_image_layers() -> list[str]:
    """Fluorescence + luminescence layers (false-color, displayed as colored images)."""
    return get_fluorescence_layers() + get_luminescence_layers()


def get_layers_with_led() -> list[str]:
    return get_transmitted_layers() + get_fluorescence_layers()


def get_opened_layer(lumaview_imagesettings) -> str | None:
    for layer in get_layers():
        try:
            layer_accordion_obj = lumaview_imagesettings.accordion_item_lookup(layer=layer)
            if not layer_accordion_obj.collapse:
                return layer
        except Exception as e:
            logger.debug(
                '[common_utils] get_opened_layer: accordion lookup for '
                'layer=%s raised; skipping: %s: %s',
                layer,
                type(e).__name__,
                e,
            )
            continue

    return None


def get_opened_layer_obj(lumaview_imagesettings):
    return lumaview_imagesettings.layer_lookup(layer=get_opened_layer(lumaview_imagesettings))


def to_bool(val) -> bool:
    if isinstance(val, str):
        return True if val.lower() == 'true' else False
    elif val in ('', None):
        return False
    else:
        return bool(float(val))


def to_float(val) -> float:
    if 'numpy' in str(type(val)):
        return val.astype(float)
    else:
        return float(val)


def to_int(val) -> int | None:
    if 'numpy' in str(type(val)):
        return int(val.astype(float))
    elif val in ('', None):
        return -1
    else:
        return int(float(val))


def get_pixel_size(
    focal_length: float,
    binning_size: int,
):
    # Read tube focal length and pixel size from scope capabilities
    # (motorconfig-sourced; per-installation override).
    import modules.app_context as _app_ctx

    ctx = _app_ctx.ctx
    if ctx is not None and ctx.scope is not None:
        tube_focal_length = ctx.scope.capabilities.lens_focal_length_mm
        pixel_width = ctx.scope.capabilities.pixel_size_um
    else:
        tube_focal_length = 47.8  # Etaluma default [mm]
        pixel_width = 2.0  # Basler default [um/pixel]
    magnification = tube_focal_length / focal_length
    um_per_pixel = pixel_width / magnification

    um_per_pixel_w_binning = um_per_pixel * binning_size

    return um_per_pixel_w_binning


def get_field_of_view(
    focal_length: float,
    frame_size: dict,
    binning_size: int,
):
    um_per_pixel = get_pixel_size(
        focal_length=focal_length,
        binning_size=binning_size,
    )
    fov_x = um_per_pixel * frame_size['width']
    fov_y = um_per_pixel * frame_size['height']

    return {'width': fov_x, 'height': fov_y}


def max_decimal_precision(parameter: str) -> int:
    DEFAULT_PRECISION = 5
    PRECISION_MAP = {'x': 4, 'y': 4, 'z': 5}

    return PRECISION_MAP.get(parameter, DEFAULT_PRECISION)


_IS_WINDOWS = platform.system() == 'Windows'


# ---------------------------------------------------------------------------
# Windows perf-counter query (PDH) -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Added on branch `perf-instrumentation-4.0.0-beta` to capture standby cache
# growth, nonpaged pool, and system file cache as part of the buffer-churn
# investigation. These counters are NOT in psutil. The PDH layer ("\Memory\..."
# perf-counter paths) is the same data PowerShell `Get-Counter` returns.
#
# Lifetime: this entire `_PdhCountersOnce` helper plus the PDH fields injected
# into `system_metrics()` are temporary. Remove once the buffer-reuse fixes
# land and the standby-cache trend is verified flat.
#
# Performance: PdhCollectQueryData is one syscall per counter, ~1 ms total.
# Safe to call once per minute.
# ---------------------------------------------------------------------------


class _PdhCountersOnce:
    """Lazy-initialized PDH query for a fixed set of counters.

    Opens the PDH query on first call, caches the counter handles, and
    re-collects on each subsequent call. On any failure, marks itself
    disabled so subsequent calls return {} without retry overhead.
    """

    # Counter paths -- match `Get-Counter` PowerShell paths exactly.
    # `\Memory\Available Bytes` is what Windows considers "available" -- equals
    # standby + free + zero pages. Useful as a sanity check against the breakdown.
    _COUNTERS = {
        'standby_normal_bytes': r'\Memory\Standby Cache Normal Priority Bytes',
        'standby_reserve_bytes': r'\Memory\Standby Cache Reserve Bytes',
        'standby_core_bytes': r'\Memory\Standby Cache Core Bytes',
        'pool_nonpaged_bytes': r'\Memory\Pool Nonpaged Bytes',
        'pool_paged_bytes': r'\Memory\Pool Paged Bytes',
        'system_cache_bytes': r'\Memory\Cache Bytes',
        'modified_page_bytes': r'\Memory\Modified Page List Bytes',
        'free_zero_bytes': r'\Memory\Free & Zero Page List Bytes',
        'available_bytes': r'\Memory\Available Bytes',
        'commit_bytes': r'\Memory\Committed Bytes',
        'commit_limit_bytes': r'\Memory\Commit Limit',
    }

    # PDH return codes / format flags (winperf.h)
    _PDH_FMT_DOUBLE = 0x00000200
    _ERROR_SUCCESS = 0

    def __init__(self):
        self._initialized = False
        self._disabled = False
        self._query = None
        self._handles = {}  # name -> counter handle

    def _init(self):
        try:
            pdh = ctypes.WinDLL('pdh')
            self._pdh = pdh

            self._PdhOpenQueryW = pdh.PdhOpenQueryW
            self._PdhAddCounterW = pdh.PdhAddCounterW
            self._PdhCollectQueryData = pdh.PdhCollectQueryData
            self._PdhGetFormattedCounterValue = pdh.PdhGetFormattedCounterValue

            class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
                _fields_ = [
                    ('CStatus', ctypes.c_ulong),
                    ('doubleValue', ctypes.c_double),
                ]

            self._PDH_FMT_COUNTERVALUE = _PDH_FMT_COUNTERVALUE

            query = ctypes.c_void_p()
            ret = self._PdhOpenQueryW(None, 0, ctypes.byref(query))
            if ret != self._ERROR_SUCCESS:
                raise OSError(f'PdhOpenQueryW failed: 0x{ret:08x}')
            self._query = query

            for name, path in self._COUNTERS.items():
                handle = ctypes.c_void_p()
                ret = self._PdhAddCounterW(query, path, 0, ctypes.byref(handle))
                if ret != self._ERROR_SUCCESS:
                    # Some counters (e.g. Standby Cache Core) may be absent
                    # on older Windows builds -- mark this one missing and
                    # continue. PdhCollectQueryData still works on the rest.
                    continue
                self._handles[name] = handle

            # First collect "primes" the counters; second collect gives real
            # values. Some counters (rate-based) need 2 samples -- for the byte
            # counters we use, a single collect is enough.
            self._PdhCollectQueryData(query)

            self._initialized = True
        except Exception:
            self._disabled = True

    def query(self):
        """Return {field: bytes_value} for all counters that are working.

        Returns {} if PDH unavailable or initialization failed.
        """
        if self._disabled:
            return {}
        if not self._initialized:
            self._init()
            if self._disabled:
                return {}

        try:
            ret = self._PdhCollectQueryData(self._query)
            if ret != self._ERROR_SUCCESS:
                return {}

            out = {}
            for name, handle in self._handles.items():
                value = self._PDH_FMT_COUNTERVALUE()
                ret = self._PdhGetFormattedCounterValue(
                    handle, self._PDH_FMT_DOUBLE, None, ctypes.byref(value)
                )
                if ret == self._ERROR_SUCCESS:
                    out[name] = float(value.doubleValue)
            return out
        except Exception:
            return {}


_pdh_counters_singleton = _PdhCountersOnce() if _IS_WINDOWS else None


def query_windows_perf_counters():
    """One-shot snapshot of selected Windows memory perf counters.

    Returns dict mapping field name to bytes value. Returns {} on non-Windows
    or if PDH is unavailable. TEMPORARY -- see `_PdhCountersOnce` docstring.
    """
    if _pdh_counters_singleton is None:
        return {}
    return _pdh_counters_singleton.query()


# ---------------------------------------------------------------------------
# Rate-tracker for cumulative counters -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Several metrics we want (page faults/sec, IO write/sec, MsMpEng read/sec)
# come from cumulative counters. Cache the last value+timestamp and compute
# a delta-rate on each call. Per-key state.
# ---------------------------------------------------------------------------

_rate_state = {}  # {key: (last_value, last_ts)}


def _delta_rate(key, current_value, now=None):
    """Return per-second rate for a cumulative counter. Returns 0.0 on first
    call or if time delta is too small to be meaningful."""
    if now is None:
        now = _time.monotonic()
    prev = _rate_state.get(key)
    _rate_state[key] = (current_value, now)
    if prev is None:
        return 0.0
    last_value, last_ts = prev
    dt = now - last_ts
    if dt < 0.5:  # avoid divide-by-near-zero on rapid back-to-back calls
        return 0.0
    return (current_value - last_value) / dt


# ---------------------------------------------------------------------------
# Defender (MsMpEng.exe) metrics -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Direct signal on the "Defender memory-maps every TIFF write" hypothesis.
# If MsMpEng's IO read rate climbs proportional to our save rate, that's
# the smoking gun for Defender being on the slowdown's critical path.
#
# Cache the PID across calls (psutil.process_iter is expensive). Re-resolve
# only if the cached PID has died.
# ---------------------------------------------------------------------------

_defender_pid_cache = {'pid': None, 'process': None}


def query_defender_metrics():
    """Return MsMpEng.exe metrics dict, or {} if not running / not Windows.

    Cumulative IO read MB is included; the caller is expected to also call
    _delta_rate for the per-sec rate.
    """
    if not _IS_WINDOWS:
        return {}

    proc = _defender_pid_cache.get('process')
    try:
        if proc is None or not proc.is_running():
            for p in psutil.process_iter(['pid', 'name']):
                try:
                    if p.info['name'] and p.info['name'].lower() == 'msmpeng.exe':
                        proc = psutil.Process(p.info['pid'])
                        _defender_pid_cache['pid'] = proc.pid
                        _defender_pid_cache['process'] = proc
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            else:
                _defender_pid_cache['pid'] = None
                _defender_pid_cache['process'] = None
                return {}
    except Exception:
        return {}

    try:
        mem = proc.memory_info()
        out = {
            'defender_private_mb': getattr(mem, 'private', mem.rss) / (1024 * 1024),
            'defender_rss_mb': mem.rss / (1024 * 1024),
        }
        try:
            io = proc.io_counters()
            out['defender_io_read_mb_total'] = io.read_bytes / (1024 * 1024)
            out['defender_io_read_mbps'] = _delta_rate('defender_io_read', io.read_bytes) / (
                1024 * 1024
            )
        except (psutil.AccessDenied, AttributeError):
            pass
        return out
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        _defender_pid_cache['pid'] = None
        _defender_pid_cache['process'] = None
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# tracemalloc top-N allocators -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Off by default. Enable via tracemalloc_enabled: true in data/settings.json.
# When on, tracemalloc is started at first call and a snapshot is taken;
# top-N allocators are returned. Overhead: ~10-30% process memory. Use only
# when needed.
# ---------------------------------------------------------------------------

_tracemalloc_started = False


def _read_tracemalloc_gate():
    # Reuse lvp_logger.lvp_appdata so the production-installed path
    # (~/Documents/LumaViewPro <version>/data/) resolves the same way
    # the logger's debug-mode gate does. Fall back to the source root
    # when lvp_logger isn't importable (e.g. unit tests that exercise
    # this module in isolation).
    from modules.settings_init import load_tracemalloc_setting

    try:
        import lvp_logger

        base_dir = lvp_logger.lvp_appdata
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_tracemalloc_setting(base_dir)


_tracemalloc_enabled = _read_tracemalloc_gate()


def query_tracemalloc_top_n(n=5):
    """Return list of top-N allocators by current size, or [] if disabled.

    Enable via tracemalloc_enabled: true in data/settings.json.
    """
    if not _tracemalloc_enabled:
        return []
    global _tracemalloc_started
    try:
        import tracemalloc

        if not _tracemalloc_started:
            tracemalloc.start(20)  # 20 frames of context
            _tracemalloc_started = True
            return []  # first call has no baseline
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics('lineno')
        out = []
        for s in stats[:n]:
            frame = s.traceback[0]
            out.append(
                {
                    'file': frame.filename,
                    'line': frame.lineno,
                    'size_kb': s.size / 1024,
                    'count': s.count,
                }
            )
        return out
    except Exception:
        return []


def system_metrics(path='/'):
    """Return a one-shot snapshot of process and host resource state.

    Used by `log_system_metrics()` (called hourly from `lumaviewpro.py`).
    Failures on individual metrics return -1 / None / 0.0 so callers
    can log "this metric isn't available on this platform" without the
    whole snapshot blowing up.

    Long-run leak detection: GDI/handle/thread/GC counts should plateau
    in steady state. A steady upward trend across hourly snapshots
    indicates a leak. See `docs/LOG_ANALYSIS_GUIDE.md` "Resource Health"
    section for healthy/unhealthy patterns.
    """
    proc = psutil.Process(os.getpid())
    disk = psutil.disk_usage(path)
    vmem = psutil.virtual_memory()

    metrics = {
        # CPU
        'cpu_percent_total': psutil.cpu_percent(),
        'cpu_percent_python': proc.cpu_percent(),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        # RAM
        'ram_available_gb': vmem.available / 1e9,
        'ram_percent_total': vmem.percent,
        'ram_used_python_percent': proc.memory_percent(),
        'ram_used_python_mb': proc.memory_info().rss / 1e6,
        'ram_used_total_mb': vmem.used / 1e6,
        # Disk
        'disk_free_gb': disk.free / 1024**3,
        'disk_used_percent': disk.percent,
    }

    # --- Private memory bytes (Windows-specific; falls back to RSS) ---
    # Working Set / RSS underreports because Windows can trim it while the
    # process still holds committed virtual memory. Private bytes is what
    # Task Manager calls "Memory (private working set)" and is the actual
    # leak indicator.
    try:
        mem = proc.memory_info()
        private = getattr(mem, 'private', mem.rss)
        metrics['ram_private_mb'] = private / 1e6
    except Exception:
        metrics['ram_private_mb'] = -1

    # --- System swap (catches page-file pressure even when "RAM looks low") ---
    try:
        swap = psutil.swap_memory()
        metrics['swap_percent'] = swap.percent
        metrics['swap_used_gb'] = swap.used / 1e9
    except Exception:
        metrics['swap_percent'] = -1
        metrics['swap_used_gb'] = -1

    # --- OS handles (Windows) / file descriptors (POSIX) ---
    # Windows caps process handles around 16M but typical apps run
    # 500-2000. A steady climb of a few handles per minute is a leak
    # of file/socket/thread handles. POSIX fds equivalent.
    try:
        if _IS_WINDOWS:
            metrics['os_handles'] = proc.num_handles()
        else:
            metrics['os_handles'] = proc.num_fds()
    except Exception:
        metrics['os_handles'] = -1

    # --- Open files count ---
    # Most actionable diagnostic when handles climb: tells you exactly
    # which files are leaked. We log only the count here; if it crosses
    # a threshold, the operator can dump the list manually via
    # `psutil.Process().open_files()`.
    try:
        metrics['open_files_count'] = len(proc.open_files())
    except Exception:
        metrics['open_files_count'] = -1

    # --- Process I/O bytes (cumulative + rates, per-process) ---
    # Distinguishes "we wrote 50 GB this hour" from "Windows Defender did".
    # Both bytes counters reset only when the process restarts.
    # Rates added 2026-04-30 (TEMPORARY) -- at 60 s sampling interval, the
    # rate gives MB/sec of TIFF writes; cross-reference with Defender IO
    # read rate to confirm "Defender mmaps every TIFF" hypothesis.
    try:
        io = proc.io_counters()
        metrics['io_read_mb'] = io.read_bytes / 1e6
        metrics['io_write_mb'] = io.write_bytes / 1e6
        metrics['io_read_mbps'] = _delta_rate('proc_io_read', io.read_bytes) / 1e6
        metrics['io_write_mbps'] = _delta_rate('proc_io_write', io.write_bytes) / 1e6
    except Exception:
        metrics['io_read_mb'] = -1
        metrics['io_write_mb'] = -1
        metrics['io_read_mbps'] = -1
        metrics['io_write_mbps'] = -1

    # --- Page faults (rate) ---
    # Sustained > 1000 pf/sec on a desktop = real memory pressure (paging
    # working set in/out). Useful as a sanity signal -- if pf/sec stays low
    # while standby grows, the slowdown is allocator/standby-cache, not
    # real paging. If pf/sec spikes during slow state, real paging.
    try:
        mem = proc.memory_info()
        pf = getattr(mem, 'pfaults', None) or getattr(mem, 'num_page_faults', None)
        if pf is not None:
            metrics['page_faults_total'] = pf
            metrics['page_faults_per_sec'] = _delta_rate('page_faults', pf)
    except Exception:
        pass

    # --- GDI / USER objects (Windows only -- main long-run-stability concern) ---
    # GDI is what causes Windows-wide slowdown after 24h+ runs. Every
    # `Texture.create()` and unclosed matplotlib figure adds a GDI handle.
    # Process limit is 10,000; Windows desktop degrades around 5,000.
    if _IS_WINDOWS:
        try:
            GR_GDIOBJECTS = 0
            GR_USEROBJECTS = 1
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            # Declare argtypes / restype so the 64-bit Windows HANDLE
            # is passed without truncation. Default ctypes types are
            # c_int (4 bytes signed), which silently truncates a 64-bit
            # pseudo-handle and causes GetGuiResources to return 0 -- a
            # broken metric that masquerades as "no GDI usage" in
            # metrics.log on healthy systems. (Repeated assignments are
            # idempotent, so doing this each call is safe.)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            user32.GetGuiResources.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            user32.GetGuiResources.restype = ctypes.c_uint
            handle = kernel32.GetCurrentProcess()
            metrics['gdi_objects'] = user32.GetGuiResources(handle, GR_GDIOBJECTS)
            metrics['user_objects'] = user32.GetGuiResources(handle, GR_USEROBJECTS)
        except Exception:
            metrics['gdi_objects'] = -1
            metrics['user_objects'] = -1
    else:
        metrics['gdi_objects'] = -1
        metrics['user_objects'] = -1

    # --- Thread count + names ---
    # Should plateau within ~30s of startup at ~20-25 (8 executors * 2
    # threads + camera + main + a few Kivy). Steady growth means an
    # executor/handler is spawning without joining.
    try:
        metrics['thread_count'] = threading.active_count()
        metrics['thread_names'] = sorted(t.name for t in threading.enumerate())
    except Exception:
        metrics['thread_count'] = -1
        metrics['thread_names'] = []

    # --- Python GC (catches reference-cycle / closure-capture leaks) ---
    # `gc.get_objects()` is somewhat expensive (iterates all tracked
    # objects) -- fine at hourly cadence. Steady linear growth indicates
    # accumulation, typically from observers/callbacks holding refs.
    try:
        metrics['gc_objects'] = len(gc.get_objects())
        gc_stats = gc.get_stats()
        metrics['gc_gen0_collections'] = gc_stats[0]['collections']
        metrics['gc_gen1_collections'] = gc_stats[1]['collections']
        metrics['gc_gen2_collections'] = gc_stats[2]['collections']
    except Exception:
        metrics['gc_objects'] = -1
        metrics['gc_gen0_collections'] = -1
        metrics['gc_gen1_collections'] = -1
        metrics['gc_gen2_collections'] = -1

    # --- Windows PDH memory counters ---
    # Standby cache + nonpaged pool are the specific signals for
    # diagnosing buffer-churn / slow-onset memory growth on Windows.
    try:
        pdh = query_windows_perf_counters()
        for k, v in pdh.items():
            metrics[f'pdh_{k}'] = v
    except Exception:
        pass

    # --- Defender (MsMpEng.exe) metrics ---
    try:
        for k, v in query_defender_metrics().items():
            metrics[k] = v
    except Exception:
        pass

    # --- Live GC depth (uncollected objects per generation) ---
    # Existing gc_genN_collections counts collections-since-start (a counter).
    # gc.get_count() is the CURRENT depth -- pairs with the counter to show
    # both rate (collections/min) and steady-state pressure (depth growing
    # = generation 2 leaks).
    try:
        c = gc.get_count()
        if len(c) >= 3:
            metrics['gc_count_gen0'] = c[0]
            metrics['gc_count_gen1'] = c[1]
            metrics['gc_count_gen2'] = c[2]
    except Exception:
        pass

    return metrics


def check_disk_space(path='/') -> float:
    """
    Returns free disk space in MB
    """

    disk = psutil.disk_usage(path)
    free_space_mb = disk.free / (1024**2)
    return free_space_mb


def check_disk_space_ok(path, required_mb: float) -> tuple[bool, float]:
    """Probe free disk space and compare against a threshold.

    Single canonical disk probe shared by protocol_image_writer, the
    protocol scan loop, and the record pre-flight in main_display. Each
    caller keeps its own threshold + abort/notification policy; only
    the probe itself is shared so the three sites cannot drift on
    backend choice (shutil vs psutil disagree on Windows partition
    mounts) or unit conversion.

    Args:
        path: Filesystem path to probe (str or pathlib.Path).
        required_mb: Minimum free space the caller needs, in MB.

    Returns:
        (ok, free_mb) -- ok is True iff free_mb >= required_mb.

    Raises:
        OSError / PermissionError: propagated from psutil.disk_usage so
            callers can decide whether to swallow (best-effort probes)
            or abort (load-bearing probes).
    """
    disk = psutil.disk_usage(str(path))
    free_mb = disk.free / (1024**2)
    return (free_mb >= required_mb, free_mb)


def get_extra_disks_info(exclude_path: str = '/') -> str | None:
    """
    Returns formatted disk information for extra disks (excluding the disk containing exclude_path).
    Returns None if only the excluded disk exists or no extra disks are found.
    Returns formatted string like: "D: 250.5 GB free (15.2% used) | E: 100.3 GB free (8.5% used)"
    """
    try:
        import psutil

        disk_partitions = psutil.disk_partitions(all=False)

        # Find which partition contains the exclude_path
        excluded_device = None
        try:
            for partition in disk_partitions:
                if exclude_path.startswith(partition.mountpoint):
                    excluded_device = partition.device
                    break
        except Exception:
            pass

        disk_info_list = []
        for partition in disk_partitions:
            # Skip the excluded device/path
            if excluded_device and partition.device == excluded_device:
                continue

            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info_list.append(
                    f'{partition.device}: {usage.free / (1024**3):.1f} GB free ({usage.percent:.1f}% used)'
                )
            except (PermissionError, OSError):
                continue

        return ' | '.join(disk_info_list) if disk_info_list else None

    except Exception:
        return None
