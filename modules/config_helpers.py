# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
GUI-independent configuration and state helper functions.

These functions extract configuration data from the settings dict
and scope objects without any Kivy/GUI dependencies. They can be
used by LumaViewPro, the REST API, or standalone scripts.
"""

import datetime
import os
import pathlib

import psutil

from lvp_logger import logger
import modules.common_utils as common_utils


# ---------------------------------------------------------------------------
# Protocol / Step helpers
# ---------------------------------------------------------------------------

def find_nearest_step(x: float, y: float, protocol) -> int:
    """Given a position, find the nearest step index in the protocol."""
    if protocol is None or protocol.num_steps() <= 0:
        return -1

    steps_df = protocol.steps()
    idx = (steps_df[['X', 'Y']].sub([x, y]).pow(2).sum(axis=1)).idxmin()
    return idx


# ---------------------------------------------------------------------------
# Layer / channel configuration
# ---------------------------------------------------------------------------

def get_layer_configs(settings: dict, specific_layers: list | None = None) -> dict:
    """Build config dicts for each layer from settings.

    Returns:
        dict of layer_name -> config dict with keys:
        acquire, video_config, stim_config, autofocus, false_color,
        illumination, gain, auto_gain, exposure, sum, focus
    """
    layer_configs = {}
    for layer in common_utils.get_layers():

        if (specific_layers is not None) and (layer not in specific_layers):
            continue

        layer_settings = settings[layer]

        acquire = layer_settings['acquire']
        video_config = layer_settings['video_config']

        if 'stim_config' in layer_settings:
            # Copy stim_config so we don't mutate the input settings dict
            stim_config = dict(layer_settings['stim_config'])
        else:
            stim_config = None

        autofocus = layer_settings['autofocus']
        false_color = layer_settings['false_color']
        illumination = round(layer_settings['ill'], common_utils.max_decimal_precision('illumination'))
        sum_count = layer_settings['sum']
        gain = round(layer_settings['gain'], common_utils.max_decimal_precision('gain'))
        auto_gain = common_utils.to_bool(layer_settings['auto_gain'])
        exposure = round(layer_settings['exp'], common_utils.max_decimal_precision('exposure'))
        focus = layer_settings['focus']

        layer_configs[layer] = {
            'acquire': acquire,
            'video_config': video_config,
            'stim_config': stim_config,
            'autofocus': autofocus,
            'false_color': false_color,
            'illumination': illumination,
            'gain': gain,
            'auto_gain': auto_gain,
            'exposure': exposure,
            'sum': sum_count,
            'focus': focus,
        }

    return layer_configs


def get_stim_configs(settings: dict) -> dict:
    """Build per-layer stim configs from settings."""
    stim_configs = {}
    layer_configs = get_layer_configs(settings)
    for layer in layer_configs:
        if layer_configs[layer]['stim_config'] is not None:
            stim_configs[layer] = layer_configs[layer]['stim_config']
    return stim_configs


def get_enabled_stim_configs(settings: dict) -> dict:
    """Return only stim configs where enabled is True."""
    stim_configs = get_stim_configs(settings)
    return {layer: cfg for layer, cfg in stim_configs.items() if cfg['enabled']}


def get_auto_gain_settings(settings: dict) -> dict:
    """Extract auto gain settings, converting max_duration_seconds to timedelta."""
    autogain_settings = settings['protocol']['autogain'].copy()
    autogain_settings['max_duration'] = datetime.timedelta(
        seconds=autogain_settings['max_duration_seconds']
    )
    del autogain_settings['max_duration_seconds']
    return autogain_settings


def get_current_objective_info(settings: dict, objective_helper) -> tuple[str, dict]:
    """Return (objective_id, objective_info_dict) from current settings."""
    objective_id = settings['objective_id']
    objective = objective_helper.get_objective_info(objective_id=objective_id)
    return objective_id, objective


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def get_current_plate_position(
    scope,
    settings: dict,
    coordinate_transformer,
    wellplate_loader,
) -> dict:
    """Get current plate position in plate coordinates.

    Returns:
        dict with keys 'x', 'y', 'z' in plate coordinates (um).
    """
    if not scope.motor_connected:
        logger.error("Cannot retrieve current plate position")
        return {'x': 0, 'y': 0, 'z': 0}

    pos = scope.get_current_position(axis=None)

    labware_id = settings.get('protocol', {}).get('labware', '')
    try:
        labware = wellplate_loader.get_plate(plate_key=labware_id)
    except Exception:
        logger.warning(f"Could not load labware '{labware_id}' for position conversion")
        return {
            'x': round(pos.get('X', 0), common_utils.max_decimal_precision('x')),
            'y': round(pos.get('Y', 0), common_utils.max_decimal_precision('y')),
            'z': round(pos.get('Z', 0), common_utils.max_decimal_precision('z')),
        }

    px, py = coordinate_transformer.stage_to_plate(
        labware=labware,
        stage_offset=settings['stage_offset'],
        sx=pos['X'],
        sy=pos['Y'],
    )

    return {
        'x': round(px, common_utils.max_decimal_precision('x')),
        'y': round(py, common_utils.max_decimal_precision('y')),
        'z': round(pos['Z'], common_utils.max_decimal_precision('z')),
    }


# ---------------------------------------------------------------------------
# System / logging helpers
# ---------------------------------------------------------------------------

def log_environment_once():
    """Log fixed environment fingerprint — system boot time, uptime, OS build,
    Pylon SDK version, Defender state. Once-per-startup pairs with the
    per-tick log_system_metrics surface so post-mortem can correlate a
    runtime metric trace against the host's exact version state.

    Call once at startup before the periodic log_system_metrics schedule.
    Long-uptime hosts can be in pathological states (e.g. memory
    exhaustion at 632h that contaminated one perf-investigation run); without recording boot time, that contamination is
    invisible in post-hoc log analysis.
    """
    import datetime as _dt
    import platform as _platform

    try:
        boot_ts = psutil.boot_time()
        boot_dt = _dt.datetime.fromtimestamp(boot_ts).isoformat(timespec='seconds')
        uptime_hr = (_dt.datetime.now().timestamp() - boot_ts) / 3600.0
    except Exception:
        boot_dt = 'NA'
        uptime_hr = -1

    try:
        os_release = _platform.platform()
    except Exception:
        os_release = 'NA'

    try:
        ncores = psutil.cpu_count(logical=True)
    except Exception:
        ncores = -1

    pylon_ver = 'NA'
    try:
        from pypylon import pylon as _pylon
        pylon_ver = getattr(_pylon, '__version__', 'NA')
    except Exception:
        pass

    defender_state = 'NA'
    if common_utils._IS_WINDOWS:
        try:
            import subprocess as _sub
            out = _sub.check_output(
                ['powershell', '-Command',
                 '(Get-MpComputerStatus | '
                 'Select-Object -Property RealTimeProtectionEnabled,'
                 'AntivirusSignatureLastUpdated,'
                 'QuickScanStartTime | ConvertTo-Json -Compress)'],
                stderr=_sub.DEVNULL,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
                timeout=15,
            ).decode('utf-8', errors='ignore').strip()
            defender_state = out or 'NA'
        except Exception:
            pass

    logger.info(
        f"[ENV METRICS] boot={boot_dt} | uptime_hr={uptime_hr:.1f} | "
        f"os={os_release} | cores={ncores} | pylon={pylon_ver} | "
        f"defender={defender_state}",
        extra={'force_error': True},
    )


def log_system_metrics(settings: dict):
    """Log CPU, RAM, and disk metrics."""
    path = settings.get('live_folder', '.')
    # Resolve relative paths and handle missing directories gracefully.
    # On installed apps, live_folder may still be './capture' before
    # microscope_settings resolves it to Documents.
    import pathlib
    resolved = pathlib.Path(path).resolve()
    if not resolved.exists():
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except Exception:
            resolved = pathlib.Path.home()  # Fallback to home dir for metrics
    path = str(resolved)
    metrics = common_utils.system_metrics(path=path)
    free_space = common_utils.check_disk_space(path=path)

    if free_space < 1024:  # Less than 1 GB
        logger.error(
            f"Low disk space: {free_space:.1f} MB remaining",
            extra={'force_error': True},
        )

    # System uptime per-tick — pairs with [ENV METRICS] boot timestamp
    # logged once at startup. Lets post-hoc log analysis tag each tick
    # with "system has been up for N hours" to filter out runs
    # contaminated by long-uptime memory exhaustion.
    try:
        import time as _time
        uptime_hr = (_time.time() - psutil.boot_time()) / 3600.0
        uptime_str = f" | uptime_hr={uptime_hr:.1f}"
    except Exception:
        uptime_str = ""

    logger.info(
        f"[SYSTEM METRICS] CPU Usage: {metrics['cpu_percent_total']:.1f}% | "
        f"RAM Available: {metrics['ram_available_gb']:.1f} GB | "
        f"RAM Usage: {metrics['ram_percent_total']:.1f}%{uptime_str}",
        extra={'force_error': True},
    )
    logger.info(
        f"[DISK METRICS] Disk Free: {metrics['disk_free_gb']:.1f} GB | "
        f"Disk Usage: {metrics['disk_used_percent']:.1f}%",
        extra={'force_error': True},
    )
    logger.info(
        f"[PROCESS METRICS] Process CPU Usage: {metrics['cpu_percent_python']:.1f}% | "
        f"Process RAM Usage: {metrics['ram_used_python_mb']:.1f} MB, "
        f"{metrics['ram_used_python_percent']:.1f}% | "
        f"Private: {metrics.get('ram_private_mb', -1):.1f} MB",
        extra={'force_error': True},
    )

    extra_disks = common_utils.get_extra_disks_info(exclude_path=path)
    if extra_disks:
        logger.info(f"[EXTRA DISKS] {extra_disks}", extra={'force_error': True})

    # --- Long-run stability metrics ---
    # Each block is grep-able and logged with force_error so it lands in
    # both lumaviewpro.log and lumaviewpro_errors.log. See
    # docs/LOG_ANALYSIS_GUIDE.md "Resource Health" section for healthy
    # vs unhealthy patterns.

    # GDI / USER objects (Windows only). The #1 cause of "Windows feels
    # slow after 24 hours" — process limit is 10k, desktop degrades at ~5k.
    gdi = metrics.get('gdi_objects', -1)
    if gdi >= 0:
        logger.info(
            f"[GDI METRICS] gdi={gdi} | user={metrics.get('user_objects', -1)}",
            extra={'force_error': True},
        )

    # OS handles + open files count. Watch for steady upward trend.
    handles = metrics.get('os_handles', -1)
    open_files = metrics.get('open_files_count', -1)
    if handles >= 0 or open_files >= 0:
        logger.info(
            f"[HANDLE METRICS] handles={handles} | open_files={open_files}",
            extra={'force_error': True},
        )

    # Thread count. Should plateau ~20-25; growth means executor leak.
    thread_count = metrics.get('thread_count', -1)
    if thread_count >= 0:
        thread_names = metrics.get('thread_names', [])
        # Compact name summary: dedupe by stem (e.g. "ThreadPoolExecutor-3_4")
        # so 8 executor pool threads collapse to one entry with count.
        from collections import Counter
        name_summary = Counter()
        for n in thread_names:
            stem = n.split('-')[0] if '-' in n else n
            name_summary[stem] += 1
        names_str = ', '.join(f"{k}={v}" for k, v in sorted(name_summary.items()))
        logger.info(
            f"[THREAD METRICS] count={thread_count} | {names_str}",
            extra={'force_error': True},
        )

    # Python GC objects. Steady growth = closures or observers holding refs.
    gc_objects = metrics.get('gc_objects', -1)
    if gc_objects >= 0:
        logger.info(
            f"[GC METRICS] objects={gc_objects} | "
            f"gen0={metrics.get('gc_gen0_collections', -1)} "
            f"gen1={metrics.get('gc_gen1_collections', -1)} "
            f"gen2={metrics.get('gc_gen2_collections', -1)}",
            extra={'force_error': True},
        )

    # Swap pressure. "RAM looks low" doesn't catch page-file thrashing.
    swap_pct = metrics.get('swap_percent', -1)
    if swap_pct >= 0:
        logger.info(
            f"[SWAP METRICS] used={metrics.get('swap_used_gb', -1):.1f} GB "
            f"({swap_pct:.1f}%)",
            extra={'force_error': True},
        )

    # Per-process I/O bytes (cumulative + per-second rates).
    # Cumulative distinguishes "we wrote 50 GB this hour" from "Windows Defender did".
    # Rates give the steady-state save rate to compare against
    # camera_data_rate in [BUFFER METRICS] and Defender's read rate.
    io_read = metrics.get('io_read_mb', -1)
    io_write = metrics.get('io_write_mb', -1)
    io_read_rate = metrics.get('io_read_mbps', -1)
    io_write_rate = metrics.get('io_write_mbps', -1)
    if io_read >= 0 or io_write >= 0:
        rate_str = ''
        if io_read_rate >= 0 or io_write_rate >= 0:
            rate_str = (f" | read_rate={io_read_rate:.2f} MB/s"
                        f" | write_rate={io_write_rate:.2f} MB/s")
        logger.info(
            f"[PROCESS IO] read={io_read:.1f} MB | write={io_write:.1f} MB{rate_str}",
            extra={'force_error': True},
        )

    # Page-fault rate.
    # Sustained > 1000 pf/sec on a desktop = real memory pressure (paging).
    # If pf/sec stays low while standby grows, slowdown is not real paging.
    pf_total = metrics.get('page_faults_total')
    pf_rate = metrics.get('page_faults_per_sec')
    if pf_total is not None or pf_rate is not None:
        logger.info(
            f"[PAGE FAULTS] total={pf_total} | rate={pf_rate:.1f}/s",
            extra={'force_error': True},
        )

    # --- Windows PDH counters ---
    # Standby cache + nonpaged pool are the specific signals for
    # diagnosing buffer-churn / slow-onset memory growth on Windows.
    #
    # Standby split: Normal + Reserve + Core = total standby cache.
    #   - Standby growing while RAM available stays high → mapped-file
    #     accumulation (the slowdown signal).
    #   - Nonpaged pool growing → kernel-side leak (Pylon DMA, drivers).
    #   - System cache (\Memory\Cache Bytes) is the file-system cache —
    #     overlaps with standby on Windows; track both for cross-check.
    pdh_keys = ['pdh_standby_normal_bytes', 'pdh_standby_reserve_bytes',
                'pdh_standby_core_bytes', 'pdh_pool_nonpaged_bytes',
                'pdh_pool_paged_bytes', 'pdh_system_cache_bytes',
                'pdh_modified_page_bytes', 'pdh_free_zero_bytes',
                'pdh_available_bytes', 'pdh_commit_bytes',
                'pdh_commit_limit_bytes']
    if any(k in metrics for k in pdh_keys):
        def _mb(key):
            v = metrics.get(key)
            return f"{v / (1024*1024):.0f}" if v is not None else 'NA'
        standby_total_mb = (
            metrics.get('pdh_standby_normal_bytes', 0)
            + metrics.get('pdh_standby_reserve_bytes', 0)
            + metrics.get('pdh_standby_core_bytes', 0)
        ) / (1024 * 1024)
        logger.info(
            f"[PDH METRICS] standby_total={standby_total_mb:.0f} MB "
            f"(normal={_mb('pdh_standby_normal_bytes')} "
            f"reserve={_mb('pdh_standby_reserve_bytes')} "
            f"core={_mb('pdh_standby_core_bytes')}) | "
            f"nonpaged_pool={_mb('pdh_pool_nonpaged_bytes')} MB | "
            f"paged_pool={_mb('pdh_pool_paged_bytes')} MB | "
            f"sys_cache={_mb('pdh_system_cache_bytes')} MB | "
            f"modified={_mb('pdh_modified_page_bytes')} MB | "
            f"free_zero={_mb('pdh_free_zero_bytes')} MB | "
            f"available={_mb('pdh_available_bytes')} MB | "
            f"commit={_mb('pdh_commit_bytes')}/{_mb('pdh_commit_limit_bytes')} MB",
            extra={'force_error': True},
        )

    # --- Buffer-churn signals from the live capture path ---
    # capture_fps × frame_nbytes = MB/sec the camera produces. Each frame
    # currently allocates ~3 fresh OS-level buffers (camera copy, 12→8 LUT,
    # tobytes()). The standby-cache growth in [PDH METRICS] should track
    # this product roughly.
    try:
        from modules import app_context as _app_ctx  # noqa: WPS433
        sd = _app_ctx.ctx.scope_display if _app_ctx.ctx is not None else None
    except Exception:
        sd = None
    if sd is not None:
        try:
            capture_fps = float(getattr(sd, '_capture_fps_value', 0.0) or 0.0)
            display_fps = float(getattr(sd, '_display_fps_value', 0.0) or 0.0)
            camera_mbps = float(getattr(sd, '_camera_mbps', 0.0) or 0.0)
            frame_nbytes = int(getattr(sd, '_last_frame_nbytes', 0) or 0)
            logger.info(
                f"[BUFFER METRICS] capture_fps={capture_fps:.1f} | "
                f"display_fps={display_fps:.1f} | "
                f"camera_data_rate={camera_mbps:.1f} MB/s | "
                f"frame_size={frame_nbytes / 1024:.0f} KB",
                extra={'force_error': True},
            )
        except Exception as e:
            logger.debug(f'[BUFFER METRICS] unavailable: {e}')

        # Frame-interval percentiles — consumer-stall detection.
        # Spikes in p99/max correlate with main-thread congestion
        # or worker-thread blocks; tracking these surfaces UI lock
        # contention and IO scheduling issues.
        try:
            if hasattr(sd, 'frame_interval_percentiles_ms'):
                pcts = sd.frame_interval_percentiles_ms()
                if pcts:
                    logger.info(
                        f"[FRAME INTERVAL] "
                        f"p50={pcts['p50']:.1f} ms | "
                        f"p95={pcts['p95']:.1f} ms | "
                        f"p99={pcts['p99']:.1f} ms | "
                        f"max={pcts['max']:.1f} ms | "
                        f"n={pcts['n']}",
                        extra={'force_error': True},
                    )
        except Exception as e:
            logger.debug(f'[FRAME INTERVAL] unavailable: {e}')

    # --- Defender (MsMpEng.exe) metrics ---
    # Direct signal for the "Defender memory-maps every TIFF write"
    # interaction. If defender_io_read_mbps tracks our io_write_mbps
    # × ~1, Defender is the slowdown source. defender_private_mb
    # growing alongside standby_total_mb is also implicating.
    defender_private = metrics.get('defender_private_mb')
    if defender_private is not None:
        defender_rss = metrics.get('defender_rss_mb', -1)
        defender_read = metrics.get('defender_io_read_mb_total', -1)
        defender_read_rate = metrics.get('defender_io_read_mbps', -1)
        logger.info(
            f"[DEFENDER METRICS] private={defender_private:.0f} MB | "
            f"rss={defender_rss:.0f} MB | "
            f"io_read_total={defender_read:.0f} MB | "
            f"io_read_rate={defender_read_rate:.2f} MB/s",
            extra={'force_error': True},
        )

    # --- GC pressure ---
    # gc_count = current uncollected objects per generation (depth).
    # gc_genN_collections (existing [GC METRICS] block) = collections-since-start
    # (rate). Both together separate "lots of churn but clean steady state"
    # from "real generation-2 leak."
    g0 = metrics.get('gc_count_gen0')
    g1 = metrics.get('gc_count_gen1')
    g2 = metrics.get('gc_count_gen2')
    if g0 is not None and g1 is not None and g2 is not None:
        logger.info(
            f"[GC PRESSURE] gen0_depth={g0} | gen1_depth={g1} | gen2_depth={g2}",
            extra={'force_error': True},
        )

    # --- Queue depth ---
    # protocol_queue is unbounded with advisory-only depth warning.
    # Monotonic queue growth = save can't keep up, frames pile up
    # retaining 16-48 MB each. Direct mechanism for slow-onset
    # multi-hour memory growth on the save path.
    try:
        ctx = _app_ctx.ctx if _app_ctx.ctx is not None else None
    except Exception:
        ctx = None
    if ctx is not None:
        queue_parts = []
        # Walk known executors. Each may use a queue.Queue (qsize) or a
        # ThreadPoolExecutor (_work_queue.qsize). Both expose qsize().
        for name in ('sequenced_capture_executor', 'autofocus_executor',
                     'protocol_executor', 'io_executor', 'camera_executor',
                     'file_io_executor', 'autofocus_thread_executor',
                     'scope_display_thread_executor', 'reset_executor'):
            try:
                exe = getattr(ctx, name, None)
                if exe is None:
                    continue
                wq = getattr(exe, '_work_queue', None) or getattr(exe, 'queue', None)
                if wq is None and hasattr(exe, 'qsize'):
                    wq = exe
                if wq is not None and hasattr(wq, 'qsize'):
                    queue_parts.append(f"{name}={wq.qsize()}")
            except Exception:
                continue
        if queue_parts:
            logger.info(
                f"[QUEUE METRICS] {' | '.join(queue_parts)}",
                extra={'force_error': True},
            )

    # --- tracemalloc top-N (env-flag gated) ---
    # Off by default. Enable with LVP_TRACEMALLOC=1 env var. Adds 10-30%
    # process memory overhead so reserved for targeted runs. When on,
    # logs top-5 allocators by current size — direct pre/post verification
    # that audited buffer-reuse sites no longer allocate on hot path.
    try:
        from modules import common_utils as _cu  # noqa: WPS433
        tm = _cu.query_tracemalloc_top_n(n=5)
        if tm:
            for i, entry in enumerate(tm, 1):
                logger.info(
                    f"[TRACEMALLOC] #{i} {entry['size_kb']:.0f} KB "
                    f"(count={entry['count']}) at "
                    f"{entry['file']}:{entry['line']}",
                    extra={'force_error': True},
                )
    except Exception:
        pass


def focus_log(positions, values, focus_round: int, source_path: str) -> int:
    """Log autofocus positions and scores to file. Returns incremented focus_round."""
    if False:  # disabled — kept for future use
        log_file = os.path.join(source_path, 'logs', 'focus_log.txt')
        try:
            file = open(log_file, 'a')
        except Exception:
            if not os.path.isdir(os.path.join(source_path, 'logs')):
                raise FileNotFoundError("Couldn't find 'logs' directory.")
            else:
                raise
        for i, p in enumerate(positions):
            mssg = str(focus_round) + '\t' + str(p) + '\t' + str(values[i]) + '\n'
            file.write(mssg)
        file.close()
    return focus_round + 1


def block_wait_for_threads(futures: list, log_loc: str = "LVP") -> None:
    """Block until all futures complete, logging any errors."""
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logger.error(f"{log_loc} ] Thread Error: {e}")


# ---------------------------------------------------------------------------
# Headless config getters — GUI-free equivalents of config_ui_getters.py
#
# These read from the settings dict (or scope object) instead of Kivy widgets.
# Used by the REST API and any non-GUI context.
# ---------------------------------------------------------------------------

# Fallback exposure slider upper bound used when no camera is connected.
# Lumascope.camera_max_exposure returns None in that case; callers pattern
# is `scope.camera_max_exposure or DEFAULT_MAX_EXPOSURE_MS`. See #616.
DEFAULT_MAX_EXPOSURE_MS = 1000.0

# Fallback gain slider upper bound used when no camera is connected.
# Matches the legacy kv default (48 dB); the actual per-camera cap is
# derived from profile.gain.total_max_db and flows through
# Lumascope.camera_max_gain at connect time.
DEFAULT_MAX_GAIN_DB = 48.0


def get_binning_from_settings(settings: dict) -> int:
    """Read binning size from settings dict (no UI needed)."""
    try:
        return int(settings.get('binning_size', 1))
    except (ValueError, TypeError):
        return 1


def get_frame_dimensions_from_settings(settings: dict) -> dict:
    """Read frame dimensions from settings dict (no UI needed)."""
    frame = settings.get('frame', {})
    return {
        'width': int(frame.get('width', 1900)),
        'height': int(frame.get('height', 1900)),
    }


def get_protocol_time_params_from_settings(settings: dict) -> dict:
    """Read protocol time params from settings dict (no UI needed).

    Returns dict with 'period' and 'duration' as timedelta objects.
    """
    protocol = settings.get('protocol', {})
    period_minutes = float(protocol.get('period', 1))
    duration_hours = float(protocol.get('duration', 1))
    return {
        'period': datetime.timedelta(minutes=period_minutes),
        'duration': datetime.timedelta(hours=duration_hours),
    }


def get_image_capture_config_from_settings(settings: dict) -> dict:
    """Read image capture config from settings dict (no UI needed)."""
    output_format = settings.get('image_output_format', {})
    return {
        'output_format': {
            'live': output_format.get('live', 'TIFF'),
            'sequenced': output_format.get('sequenced', 'TIFF'),
        },
        'use_full_pixel_depth': settings.get('use_full_pixel_depth', False),
        'false_color_16bit': settings.get('false_color_16bit', False),
    }


DEFAULT_LABWARE_ID = '96 well microplate'


def get_selected_labware_from_settings(
    settings: dict,
    wellplate_loader,
) -> tuple[str, object]:
    """Read selected labware from settings dict (no UI needed).

    Always returns a valid (labware_id, wellplate_object) tuple. Per
    Eric's 2026-04-25 directive: callers shouldn't have to deal with
    None. If settings has no labware, or the requested labware doesn't
    exist in the loader, fall back to the shipped default
    (DEFAULT_LABWARE_ID) and finally to the first available plate.
    Issue #634/#632 cluster: every site that consumed this return
    treated None as a crash, so removing None from the contract retires
    the cluster by construction.

    The only way this raises is if the wellplate loader is empty (broken
    install / labware.json missing) — that's a genuine fatal that the
    caller cannot reasonably recover from.
    """
    labware_id = settings.get('protocol', {}).get('labware', '') or DEFAULT_LABWARE_ID
    try:
        labware_obj = wellplate_loader.get_plate(plate_key=labware_id)
        return labware_id, labware_obj
    except Exception:
        logger.warning(
            f"Could not load labware '{labware_id}', falling back to "
            f"default '{DEFAULT_LABWARE_ID}'"
        )
    # First fallback: the shipped default.
    if labware_id != DEFAULT_LABWARE_ID:
        try:
            labware_obj = wellplate_loader.get_plate(plate_key=DEFAULT_LABWARE_ID)
            return DEFAULT_LABWARE_ID, labware_obj
        except Exception:
            logger.warning(
                f"Default labware '{DEFAULT_LABWARE_ID}' also missing; "
                f"falling back to first available plate"
            )
    # Second fallback: anything in the loader. If the loader is empty,
    # this raises ValueError — at which point the install is broken.
    available = wellplate_loader.get_plate_list()
    if not available:
        raise ValueError(
            "wellplate_loader has no plates registered — labware.json "
            "is missing or unreadable"
        )
    fallback_id = available[0]
    return fallback_id, wellplate_loader.get_plate(plate_key=fallback_id)


def get_zstack_params_from_settings(settings: dict) -> dict:
    """Read z-stack params from settings dict (no UI needed)."""
    zstack = settings.get('protocol', {}).get('zstack', {})
    return {
        'range': float(zstack.get('range', 0)),
        'step_size': float(zstack.get('step_size', 1)),
        'z_reference': zstack.get('z_reference', 'center'),
    }


def get_sequenced_capture_config_from_settings(
    settings: dict,
    objective_helper,
    wellplate_loader=None,
) -> dict:
    """Build sequenced capture config from settings dict (no UI needed).

    This is the headless equivalent of config_getters.get_sequenced_capture_config_from_ui().
    """
    objective_id, _ = get_current_objective_info(settings, objective_helper)
    time_params = get_protocol_time_params_from_settings(settings)
    labware_id = settings.get('protocol', {}).get('labware', '')
    tiling = settings.get('protocol', {}).get('tiling', '1x1')
    use_zstacking = settings.get('protocol', {}).get('use_zstacking', False)
    frame_dimensions = get_frame_dimensions_from_settings(settings)
    zstack_params = get_zstack_params_from_settings(settings)
    layer_configs = get_layer_configs(settings)

    return {
        'labware_id': labware_id,
        'objective_id': objective_id,
        'zstack_params': zstack_params,
        'use_zstacking': use_zstacking,
        'tiling': tiling,
        'layer_configs': layer_configs,
        'period': time_params['period'],
        'duration': time_params['duration'],
        'frame_dimensions': frame_dimensions,
        'binning_size': get_binning_from_settings(settings),
        'stim_config': get_stim_configs(settings),
    }
