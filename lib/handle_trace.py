# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Opt-in Windows handle-count tracking for leak hunting.

Diagnostic instrumentation for narrowing per-iteration handle leaks that
manifest only over multi-hour soaks. Use this when a long-soak run shows
linear handle growth and you need to identify which per-call site is the
source without rebuilding the exe between iterations.

Default OFF. Set `LVP_HANDLE_TRACE=1` to enable. Zero overhead when disabled
(single env-var read at module-load).

Usage at call sites (zero-overhead when disabled, see ENABLE gate below):

    from lib.handle_trace import tick
    # ... operation that might leak ...
    tick('save_image')

Each labeled tick samples `psutil.Process().num_handles()` and logs the
delta vs. the previous baseline every `every_n` calls (default 50).

Reading the output:
    [HANDLE TRACE] save_image: +12 handles over 50 calls (+0.24/call), total now=131092

A non-zero `+N/call` value pinpoints the leaking site. Zero-delta lines
rule out a path. Bisection narrows multi-step paths.

Considered alternatives:
    - tracemalloc (already in modules/memory_profiler.py): tracks PYTHON
      allocations, not Windows kernel handles. Won't catch handle leaks
      from ctypes / native SDK calls.
    - per-call logging (no batching): too noisy at protocol scan rates
      (~6 captures/min × multiple labels = log spam).
"""
import os
import threading

try:
    import psutil
    _proc = psutil.Process()
except ImportError:
    _proc = None

try:
    from lvp_logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


ENABLE = os.environ.get('LVP_HANDLE_TRACE') == '1'
# Detect num_handles() availability (Windows only). Object-type sampler
# (gc.get_objects()) works cross-platform, so the diagnostic stays useful
# on macOS / Linux for reproducing the Python-side leak shape via simulator
# camera, even when the Windows handle counter is unavailable.
_HAS_NUM_HANDLES = False
if ENABLE and _proc is not None:
    try:
        _proc.num_handles()
        _HAS_NUM_HANDLES = True
    except (AttributeError, NotImplementedError):
        logger.info(
            '[HANDLE TRACE] num_handles() unavailable (non-Windows host); '
            'object-type sampler still active.'
        )
# Per-tick object-type sampler: every Nth tick (default every 1000 ticks
# of any label), Counter the gc.get_objects() type names and log the top 20.
# Heavy (~1-3 s on 10M-object processes), so heavily throttled. Same env gate.
OBJ_SAMPLE_EVERY = int(os.environ.get('LVP_OBJ_SAMPLE_EVERY', '1000'))

_state: dict = {}  # label -> (count, baseline_handles)
_lock = threading.Lock()
_obj_sample_counter = 0


def tick(label: str, every_n: int = 50) -> None:
    """Sample handle count and log per-call delta every `every_n` calls.

    Args:
        label: Identifier for this call site (appears in log).
        every_n: Sampling interval. Default 50.
    """
    if not ENABLE:
        return
    with _lock:
        global _obj_sample_counter
        count, baseline = _state.get(label, (0, None))
        count += 1
        _obj_sample_counter += 1

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

        # Object-type sampler -- heavily throttled. Identifies which
        # Python object types are accumulating (Future / Lock / Condition
        # = Semaphore leak; ndarray = image buffer leak; etc.).
        if _obj_sample_counter >= OBJ_SAMPLE_EVERY:
            _obj_sample_counter = 0
            try:
                import gc
                from collections import Counter
                top = Counter(
                    type(o).__name__ for o in gc.get_objects()
                ).most_common(20)
                logger.info(
                    f'[HANDLE TRACE] gc.get_objects() top-20: {top}'
                )
            except Exception as e:
                logger.debug(f'[HANDLE TRACE] obj sampler failed: {e}')


def snapshot() -> int | None:
    """One-shot read of current process handle count. None if unavailable."""
    if _proc is None:
        return None
    try:
        return _proc.num_handles()
    except Exception:
        return None
