# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import enum
import json
import os
import pathlib
import re

import numpy as np


# ---------------------------------------------------------------------------
# Hardware defaults (fallbacks when motorconfig/scope not available)
# ---------------------------------------------------------------------------
# LS850 full travel range — used as default stage limits when scope is not connected.
DEFAULT_STAGE_TRAVEL_UM = {"x": 120000.0, "y": 80000.0}

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
    COMPOSITE = "Composite"
    STITCHED = "Stitched"
    ZPROJECT = "ZProject"
    VIDEO = "Video"
    HYPERSTACK = "Hyperstack"

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
    color = None,
    z_height_idx = None,
    scan_count = None,
    tile_label = None,
    objective_short_name = None,
    custom_name_prefix = None,
    stitched: bool = False,
    video: bool = False,
    zprojection: str | None = None,
    stack: bool = False,
    hyperstack: bool = False,
    turret_position: int | None = None,
):
    if custom_name_prefix not in (None, ""):
        name = f"{custom_name_prefix}"
    else:
        name = f"{well_label}"

    if color not in (None, "") and color not in name:
        name = f"{name}_{color}"
    
    if tile_label not in (None, "", -1):
        if not f"_T{tile_label}" in name:
            name = f"{name}_T{tile_label}"

    if objective_short_name not in (None, "", -1):
        name = f"{name}_{objective_short_name}"

    if turret_position is not None:
        name = f"{name}_Turret{turret_position}"

    if z_height_idx not in (None, "", -1):
        if not f"_Z{z_height_idx}" in name:
            name = f"{name}_Z{z_height_idx}"

    DESIRED_SCAN_COUNT_DIGITS = 4
    if scan_count not in (None, ""):
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
    well_pattern = r"^[A-Z]{1,2}[0-9]+$"
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
        'Current Position at Bottom': 'bottom'
    }

    if text_label in LABEL_MAP:
          return LABEL_MAP[text_label]
    
    raise Exception(f"Unknown Z-stack position reference: {text_label}")

                    
def get_layers() -> list[str]:
    return ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi']


def get_transmitted_layers() -> list[str]:
    return ['BF', 'PC', 'DF']


def get_fluorescence_layers() -> list[str]:
    return ['Blue', 'Green', 'Red']


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
        except Exception:
            continue
        
    return None

def get_opened_layer_obj(lumaview_imagesettings):
    return lumaview_imagesettings.layer_lookup(layer=get_opened_layer(lumaview_imagesettings))


def to_bool(val) -> bool:
    if isinstance(val, str):
        return True if val.lower() == "true" else False
    elif val in ("", None):
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
    elif val in ("", None):
        return -1
    else:
        return int(float(val))


def get_pixel_size(
    focal_length: float,
    binning_size: int,
):
    # Read tube focal length and pixel size from motorconfig if available
    import modules.app_context as _app_ctx
    ctx = _app_ctx.ctx
    if ctx is not None and ctx.scope is not None:
        tube_focal_length = ctx.scope.lens_focal_length()
        pixel_width = ctx.scope.pixel_size()
    else:
        tube_focal_length = 47.8  # Etaluma default [mm]
        pixel_width = 2.0         # Basler default [um/pixel]
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
    
    return {
        'width': fov_x,
        'height': fov_y
    }


def max_decimal_precision(parameter: str) -> int:
    DEFAULT_PRECISION = 5
    PRECISION_MAP = {
        'x': 4,
        'y': 4,
        'z': 5
    }

    return PRECISION_MAP.get(parameter, DEFAULT_PRECISION)


import ctypes
import gc
import os
import platform
import threading

import psutil

_IS_WINDOWS = platform.system() == 'Windows'


def system_metrics(path="/"):
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
        "cpu_percent_total": psutil.cpu_percent(),
        "cpu_percent_python": proc.cpu_percent(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),

        # RAM
        "ram_available_gb": vmem.available / 1e9,
        "ram_percent_total": vmem.percent,
        "ram_used_python_percent": proc.memory_percent(),
        "ram_used_python_mb": proc.memory_info().rss / 1e6,
        "ram_used_total_mb": vmem.used / 1e6,

        # Disk
        "disk_free_gb": disk.free / 1024**3,
        "disk_used_percent": disk.percent,
    }

    # --- Private memory bytes (Windows-specific; falls back to RSS) ---
    # Working Set / RSS underreports because Windows can trim it while the
    # process still holds committed virtual memory. Private bytes is what
    # Task Manager calls "Memory (private working set)" and is the actual
    # leak indicator.
    try:
        mem = proc.memory_info()
        private = getattr(mem, 'private', mem.rss)
        metrics["ram_private_mb"] = private / 1e6
    except Exception:
        metrics["ram_private_mb"] = -1

    # --- System swap (catches page-file pressure even when "RAM looks low") ---
    try:
        swap = psutil.swap_memory()
        metrics["swap_percent"] = swap.percent
        metrics["swap_used_gb"] = swap.used / 1e9
    except Exception:
        metrics["swap_percent"] = -1
        metrics["swap_used_gb"] = -1

    # --- OS handles (Windows) / file descriptors (POSIX) ---
    # Windows caps process handles around 16M but typical apps run
    # 500-2000. A steady climb of a few handles per minute is a leak
    # of file/socket/thread handles. POSIX fds equivalent.
    try:
        if _IS_WINDOWS:
            metrics["os_handles"] = proc.num_handles()
        else:
            metrics["os_handles"] = proc.num_fds()
    except Exception:
        metrics["os_handles"] = -1

    # --- Open files count ---
    # Most actionable diagnostic when handles climb: tells you exactly
    # which files are leaked. We log only the count here; if it crosses
    # a threshold, the operator can dump the list manually via
    # `psutil.Process().open_files()`.
    try:
        metrics["open_files_count"] = len(proc.open_files())
    except Exception:
        metrics["open_files_count"] = -1

    # --- Process I/O bytes (cumulative, per-process) ---
    # Distinguishes "we wrote 50 GB this hour" from "Windows Defender did".
    # Both bytes counters reset only when the process restarts.
    try:
        io = proc.io_counters()
        metrics["io_read_mb"] = io.read_bytes / 1e6
        metrics["io_write_mb"] = io.write_bytes / 1e6
    except Exception:
        metrics["io_read_mb"] = -1
        metrics["io_write_mb"] = -1

    # --- GDI / USER objects (Windows only — main long-run-stability concern) ---
    # GDI is what causes Windows-wide slowdown after 24h+ runs. Every
    # `Texture.create()` and unclosed matplotlib figure adds a GDI handle.
    # Process limit is 10,000; Windows desktop degrades around 5,000.
    if _IS_WINDOWS:
        try:
            GR_GDIOBJECTS = 0
            GR_USEROBJECTS = 1
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            metrics["gdi_objects"] = ctypes.windll.user32.GetGuiResources(
                handle, GR_GDIOBJECTS
            )
            metrics["user_objects"] = ctypes.windll.user32.GetGuiResources(
                handle, GR_USEROBJECTS
            )
        except Exception:
            metrics["gdi_objects"] = -1
            metrics["user_objects"] = -1
    else:
        metrics["gdi_objects"] = -1
        metrics["user_objects"] = -1

    # --- Thread count + names ---
    # Should plateau within ~30s of startup at ~20-25 (8 executors * 2
    # threads + camera + main + a few Kivy). Steady growth means an
    # executor/handler is spawning without joining.
    try:
        metrics["thread_count"] = threading.active_count()
        metrics["thread_names"] = sorted(t.name for t in threading.enumerate())
    except Exception:
        metrics["thread_count"] = -1
        metrics["thread_names"] = []

    # --- Python GC (catches reference-cycle / closure-capture leaks) ---
    # `gc.get_objects()` is somewhat expensive (iterates all tracked
    # objects) — fine at hourly cadence. Steady linear growth indicates
    # accumulation, typically from observers/callbacks holding refs.
    try:
        metrics["gc_objects"] = len(gc.get_objects())
        gc_stats = gc.get_stats()
        metrics["gc_gen0_collections"] = gc_stats[0]['collections']
        metrics["gc_gen1_collections"] = gc_stats[1]['collections']
        metrics["gc_gen2_collections"] = gc_stats[2]['collections']
    except Exception:
        metrics["gc_objects"] = -1
        metrics["gc_gen0_collections"] = -1
        metrics["gc_gen1_collections"] = -1
        metrics["gc_gen2_collections"] = -1

    return metrics


def check_disk_space(path="/") -> float:
    """
    Returns free disk space in MB
    """ 
    
    disk = psutil.disk_usage(path)
    free_space_mb = disk.free / (1024**2)
    return free_space_mb


def get_extra_disks_info(exclude_path: str = "/") -> str | None:
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
                    f"{partition.device}: {usage.free / (1024**3):.1f} GB free ({usage.percent:.1f}% used)"
                )
            except (PermissionError, OSError):
                continue
        
        return ' | '.join(disk_info_list) if disk_info_list else None
    
    except Exception:
        return None