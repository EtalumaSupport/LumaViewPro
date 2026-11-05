# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Gated memory profiler for mapping LumaViewPro's resident footprint.

Off by default. ``ScopeSession.create`` hands it ``memory_profile_enabled``
and ``memory_profile_interval_s`` from the prepared settings, so every
program that brings a scope up profiles when its settings say so; this
module reads no setting, as src/_lumascope/lib/ sits below the layer that owns them. When
disabled every entry point here is a cheap no-op.

When enabled, ``start()`` begins tracemalloc, records a baseline, and
launches a lightweight daemon thread that samples RSS on a fixed cadence.
What a process imports before it brings a scope up is not traced.
``snapshot(state)`` records RSS + the top tracemalloc allocators + the growth
diff since the previous snapshot, so a run produces a per-lifecycle-state map
of where memory went:

    memory_profile.start(enabled=..., interval_s=...)  # ScopeSession.create
    ...
    memory_profile.snapshot('post_homing')     # after the startup home succeeds
    memory_profile.snapshot('cold_start_done') # 5 s after the session's metrics start
    memory_profile.snapshot('post_protocol')   # after a protocol run

The snapshots come from different threads, so each is taken and compared
against the previous one under the module's lock, and the growth a line
reports is always since the snapshot logged before it.

All output goes to the standard logger tagged ``[MEM PROFILE]``. tracemalloc
sees Python-level allocations only (not GL textures or C-level SDK buffers),
so RSS is the absolute number and the tracemalloc top-N is the attribution of
the Python-tracked portion -- read them together.
"""

import threading
import time

import psutil

from _lumascope.lib.log_helpers import logger

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


def start(*, enabled: bool, interval_s: float) -> None:
    """If ``enabled``, begin tracemalloc + RSS sampling every ``interval_s`` seconds.

    Only on, and only when it is off: a disabled call changes nothing, so a
    second session in one process (the GUI's fallback onto the template,
    where the setting is off) neither stops a running profiler's snapshots
    nor keeps a later enabled call from starting it. A second enabled call
    is ignored, as tracemalloc ignores a second start.
    """
    global _started, _enabled, _interval_s, _proc, _t0
    if not enabled:
        return
    with _lock:
        if _started:
            return
        _started = True
    _interval_s = interval_s
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start(_TRACEMALLOC_FRAMES)
        _proc = psutil.Process()
        _t0 = time.monotonic()
        _enabled = True
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

        global _prev_snapshot
        with _lock:
            rss = _rss_mb()
            elapsed = time.monotonic() - _t0
            snap = tracemalloc.take_snapshot()
            logger.info(f'[MEM PROFILE] state={state} t={elapsed:.1f}s rss={rss:.1f} MB')
            _log_top(snap, f'state={state}')
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
