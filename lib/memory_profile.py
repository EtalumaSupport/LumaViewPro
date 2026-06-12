# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Gated memory profiler for mapping LumaViewPro's resident footprint.

Off by default. Enable via ``memory_profile_enabled: true`` in the live
settings (current.json once it exists, else settings.json) -- the same
merged-settings gate as profile_trace and tracemalloc. When disabled every
entry point here is a cheap no-op.

When enabled, ``start()`` begins tracemalloc as early in process life as the
caller invokes it (call it at the top of __main__, BEFORE the heavy imports,
so the resident baseline is captured rather than missed), records a baseline,
and launches a lightweight daemon thread that samples RSS on a fixed cadence.
``snapshot(state)`` records RSS + the top tracemalloc allocators + the growth
diff since the previous snapshot, so a run produces a per-lifecycle-state map
of where memory went:

    memory_profile.start(source_path)          # __main__, before heavy imports
    ...
    memory_profile.snapshot('cold_start_done') # end of App.on_start
    memory_profile.snapshot('post_homing')     # after homing completes
    memory_profile.snapshot('post_protocol')   # after a protocol run

All output goes to the standard logger tagged ``[MEM PROFILE]``. tracemalloc
sees Python-level allocations only (not GL textures or C-level SDK buffers),
so RSS is the absolute number and the tracemalloc top-N is the attribution of
the Python-tracked portion -- read them together.
"""

import threading
import time

import psutil

from lvp_logger import logger

# Frames of traceback tracemalloc keeps per allocation. Deliberately 1 (the
# allocating line only): tracemalloc stores this many frame records for EVERY
# traced allocation, so a deep value both slows startup ~6x (the whole import
# phase runs instrumented) AND inflates RSS during a frame flood -- which would
# distort the very footprint baseline this tool exists to measure. Depth 1
# still identifies the top sites (e.g. the scope_display live-view blit). Raise
# temporarily only when one specific site needs its full call chain.
_TRACEMALLOC_FRAMES = 1
_TOP_N = 15

_lock = threading.Lock()
_started = False
_enabled = False
_interval_s = 5.0
_proc = None
_t0 = 0.0
_prev_snapshot = None  # tracemalloc.Snapshot from the previous snapshot() call


def is_enabled() -> bool:
    """True if the profiler is active this run."""
    return _enabled


def start(source_path: str) -> None:
    """Read the gate and, if enabled, begin tracemalloc + RSS sampling.

    Idempotent and best-effort: any failure logs a warning and leaves the
    profiler disabled rather than disrupting startup. Call as early as
    possible in __main__ so the tracemalloc baseline reflects the resident
    set, not just post-startup growth.

    Args:
        source_path: App source/data directory, used to resolve the live
            settings file for the gate.
    """
    global _started, _enabled, _interval_s, _proc, _t0
    with _lock:
        if _started:
            return
        _started = True
    try:
        from modules.settings_init import load_memory_profile_setting

        cfg = load_memory_profile_setting(source_path)
        _enabled = cfg['enabled']
        _interval_s = cfg['interval_s']
    except Exception as e:
        logger.warning(f'[MEM PROFILE] gate read failed ({type(e).__name__}: {e}); disabled')
        _enabled = False
        return
    if not _enabled:
        return
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start(_TRACEMALLOC_FRAMES)
        _proc = psutil.Process()
        _t0 = time.monotonic()
        logger.info(
            f'[MEM PROFILE] enabled: tracemalloc started, RSS sampled every '
            f'{_interval_s:.1f}s. rss={_rss_mb():.1f} MB at start.'
        )
        thread = threading.Thread(target=_sampler, name='mem-profile-sampler', daemon=True)
        thread.start()
    except Exception as e:
        logger.warning(f'[MEM PROFILE] start failed ({type(e).__name__}: {e}); disabled')
        _enabled = False


def snapshot(state: str) -> None:
    """Log RSS + top tracemalloc allocators + growth since the last snapshot.

    No-op when the profiler is disabled. ``state`` names the lifecycle point
    (e.g. ``cold_start_done``, ``idle_live_off``, ``post_homing``) so the log
    reads as a per-state footprint map.
    """
    if not _enabled:
        return
    try:
        import tracemalloc

        rss = _rss_mb()
        elapsed = time.monotonic() - _t0
        snap = tracemalloc.take_snapshot()
        logger.info(f'[MEM PROFILE] state={state} t={elapsed:.1f}s rss={rss:.1f} MB')
        _log_top(snap, f'state={state}')
        global _prev_snapshot
        if _prev_snapshot is not None:
            _log_growth(snap, _prev_snapshot, state)
        _prev_snapshot = snap
    except Exception as e:
        logger.warning(f'[MEM PROFILE] snapshot({state}) failed: {type(e).__name__}: {e}')


def _rss_mb() -> float:
    return _proc.memory_info().rss / (1024 * 1024)


def _log_top(snap, tag: str) -> None:
    stats = snap.statistics('lineno')[:_TOP_N]
    logger.info(f'[MEM PROFILE] top {_TOP_N} tracked allocators at {tag}:')
    for i, stat in enumerate(stats, 1):
        size_mb = stat.size / (1024 * 1024)
        frame = stat.traceback[0]
        logger.info(f'[MEM PROFILE]   #{i} {size_mb:6.1f} MB  {frame.filename}:{frame.lineno}')


def _log_growth(cur, prev, state: str) -> None:
    diff = cur.compare_to(prev, 'lineno')[:_TOP_N]
    logger.info(f'[MEM PROFILE] top {_TOP_N} GROWTH sites since previous snapshot (-> {state}):')
    for i, stat in enumerate(diff, 1):
        delta_mb = stat.size_diff / (1024 * 1024)
        frame = stat.traceback[0]
        logger.info(f'[MEM PROFILE]   +{i} {delta_mb:+7.1f} MB  {frame.filename}:{frame.lineno}')


def _sampler() -> None:
    while True:
        time.sleep(_interval_s)
        try:
            logger.info(
                f'[MEM PROFILE] sample t={time.monotonic() - _t0:6.1f}s rss={_rss_mb():.1f} MB'
            )
        except Exception as e:
            logger.warning(f'[MEM PROFILE] sampler stopped: {type(e).__name__}: {e}')
            return
