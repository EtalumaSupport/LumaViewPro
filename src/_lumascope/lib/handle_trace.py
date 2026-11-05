# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Opt-in Windows handle-count tracking for leak hunting.

Diagnostic instrumentation for narrowing per-iteration handle leaks that
manifest only over multi-hour soaks. Use this when a long-soak run shows
linear handle growth and you need to identify which per-call site is the
source without rebuilding the exe between iterations.

Default OFF. Enable via settings.json (customer / bench operators):
set `settings['profiling']['handle_trace_enabled'] = true`.
`ScopeSession.create` calls `enable()` when the flag is true, so every
program that brings a scope up traces when its settings say so.

Usage at call sites (zero-overhead when disabled, see ENABLE gate below):

    from _lumascope.lib.handle_trace import tick
    # ... operation that might leak ...
    tick('save_image')

Each labeled tick samples `psutil.Process().num_handles()` and logs the
delta vs. the previous baseline every `every_n` calls (default 50).

Reading the output:
    [HANDLE TRACE] save_image: +12 handles over 50 calls (+0.24/call), total now=131092

A non-zero `+N/call` value pinpoints the leaking site. Zero-delta lines
rule out a path. Bisection narrows multi-step paths.

Considered alternatives:
    - tracemalloc (already in src/_lumascope/lib/memory_profile.py): tracks PYTHON
      allocations, not Windows kernel handles. Won't catch handle leaks
      from ctypes / native SDK calls.
    - per-call logging (no batching): too noisy at protocol scan rates
      (~6 captures/min * multiple labels = log spam).
    - a count of Python objects by type through gc.get_objects(): the list
      it builds hands this thread every object the collector tracks,
      including a tuple another thread is still building, and that
      thread's tuple() then raises SystemError. tick() runs on the grab and
      save paths, during runs.
"""

import threading

try:
    import psutil

    _proc = psutil.Process()
except ImportError:
    _proc = None

from _lumascope.lib.log_helpers import logger


# Module-level gate flipped by enable(). tick() short-circuits when False.
ENABLE = False

# Detect num_handles() availability (Windows only).
_HAS_NUM_HANDLES = False
if _proc is not None:
    try:
        _proc.num_handles()
        _HAS_NUM_HANDLES = True
    except (AttributeError, NotImplementedError):
        _HAS_NUM_HANDLES = False

_state: dict = {}  # label -> (count, baseline_handles)
_lock = threading.Lock()


def enable() -> None:
    """Activate handle-trace. Safe to call multiple times."""
    global ENABLE
    if ENABLE:
        return
    ENABLE = True
    if not _HAS_NUM_HANDLES:
        logger.info(
            '[HANDLE TRACE] activated -- num_handles() unavailable on this '
            'host; no handle deltas will be logged.'
        )
    else:
        logger.info('[HANDLE TRACE] activated -- per-call handle delta.')


def disable() -> None:
    """Deactivate handle-trace. Safe to call when already disabled."""
    global ENABLE
    ENABLE = False


def tick(label: str, every_n: int = 50) -> None:
    """Sample handle count and log per-call delta every `every_n` calls.

    Args:
        label: Identifier for this call site (appears in log).
        every_n: Sampling interval. Default 50.
    """
    if not ENABLE:
        return
    with _lock:
        count, baseline = _state.get(label, (0, None))
        count += 1

        # Handle-count read (Windows only; falls back to None elsewhere).
        h = _proc.num_handles() if _HAS_NUM_HANDLES else None

        if baseline is None:
            _state[label] = (count, h)
            return
        if count % every_n != 0:
            _state[label] = (count, baseline)
            return
        if h is not None:
            delta = h - baseline
            per_call = delta / every_n
            logger.info(
                f'[HANDLE TRACE] {label}: +{delta} handles over {every_n} calls '
                f'(+{per_call:.2f}/call), total now={h}'
            )
        _state[label] = (count, h)


def snapshot() -> int | None:
    """One-shot read of current process handle count. None if unavailable."""
    if _proc is None:
        return None
    try:
        return _proc.num_handles()
    except Exception:
        return None
