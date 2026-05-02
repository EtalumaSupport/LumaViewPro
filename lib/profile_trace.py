# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Opt-in runtime tracing for profiling + debugging.

Default OFF. Zero overhead when disabled — every trace site is guarded
by a single module-level flag check.

Enable three ways:
  1. Set env `LVP_PROFILE_TRACE=1` before launching LVP.
  2. Call `profile_trace.enable()` programmatically.
  3. Set `LVP_PROFILE_TRACE_DIR=/some/path` to override the output dir.

Writes CSV files under `./logs/profile/<timestamp>/` by default:
  - serial_trace.csv        (SerialBoard.exchange_command timings)
  - motion_trace.csv        (motion-monitor poll durations + axis state transitions)
  - frame_validity_trace.csv (invalidate/count/settle events)

Columns are documented in the trace-site wrappers (see timer() and trace()
callers in drivers/serialboard.py, modules/lumascope_api.py,
modules/frame_validity.py).

CSVs auto-close on process exit via atexit. Thread-safe via a single
module-level lock. Writes are line-buffered — no tail-buffer loss on crash.
"""
import atexit
import os
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from lvp_logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


ENABLE_PROFILE_TRACE = False
_output_dir = None
_lock = threading.Lock()
_writers = {}


def enable(output_dir=None):
    """Start writing trace CSVs. Safe to call multiple times."""
    global ENABLE_PROFILE_TRACE, _output_dir
    if ENABLE_PROFILE_TRACE:
        return
    if output_dir is None:
        env = os.environ.get("LVP_PROFILE_TRACE_DIR")
        if env:
            output_dir = Path(env)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./logs/profile") / ts
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    ENABLE_PROFILE_TRACE = True
    atexit.register(disable)
    logger.info(f"[PROFILE   ] Trace enabled. Writing to {_output_dir}")


def disable():
    """Flush and close all trace files. Safe to call if already disabled."""
    global ENABLE_PROFILE_TRACE
    if not ENABLE_PROFILE_TRACE:
        return
    ENABLE_PROFILE_TRACE = False
    with _lock:
        for fh in _writers.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        _writers.clear()


def trace(filename, header, fields):
    """Append one row to the named CSV. No-op when disabled."""
    if not ENABLE_PROFILE_TRACE:
        return
    try:
        with _lock:
            fh = _writers.get(filename)
            if fh is None:
                path = _output_dir / filename
                need_header = not path.exists()
                fh = open(path, "a", buffering=1)
                if need_header:
                    fh.write(header + "\n")
                _writers[filename] = fh
            fh.write(",".join(str(x) for x in fields) + "\n")
    except Exception as e:
        logger.warning(f"[PROFILE   ] trace write failed ({filename}): {e}")


class timer:
    """Context manager: captures elapsed ms, writes one row on exit.

    Usage:
        with profile_trace.timer(
            "serial_trace.csv",
            "ts_ms,duration_ms,board,command",
            lambda: ["led", command[:40]]
        ):
            do_stuff()

    The extra-fields callable is only invoked when tracing is enabled,
    so it's safe to do non-trivial formatting inside it.
    """
    __slots__ = ("filename", "header", "extra_fn", "t0")

    def __init__(self, filename, header, extra_fn):
        self.filename = filename
        self.header = header
        self.extra_fn = extra_fn
        self.t0 = None

    def __enter__(self):
        if ENABLE_PROFILE_TRACE:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if ENABLE_PROFILE_TRACE and self.t0 is not None:
            dt_ms = (time.perf_counter() - self.t0) * 1000
            ts_ms = int(time.time() * 1000)
            try:
                extra = self.extra_fn()
            except Exception as e:
                logger.warning(f"[PROFILE   ] timer extra_fn failed: {e}")
                return
            trace(self.filename, self.header, [ts_ms, f"{dt_ms:.3f}", *extra])


class TimedLock:
    """Drop-in wrapper for threading.Lock / threading.RLock that records
    acquire-wait + hold time per acquire-release cycle to `lock_trace.csv`
    when LVP_PROFILE_TRACE=1 is set.

    Threading audit §10.2 — validates SerialBoard._lock hold-time claim
    (~32 ms per round-trip, documented at drivers/motorboard.py:79 from a
    2026-04-13 bench run) across more sessions, and surfaces outliers.
    Zero overhead when tracing is disabled — __enter__/__exit__ short-circuit
    before time.perf_counter().

    Thread-safe for RLock re-entry: uses a per-instance thread-local
    stack of (t_wait_start, t_held_start) tuples so nested
    `with self._rlock: ... with self._rlock: ...` correctly records
    outer and inner acquire times independently instead of clobbering.

    Usage (same as threading.Lock):
        self._led_lock = TimedLock(threading.RLock(), name="led_lock")
        with self._led_lock:
            ...

    Also supports acquire()/release() for code that uses them directly.
    """
    __slots__ = ("_lock", "_name", "_tls")

    def __init__(self, lock, name):
        self._lock = lock
        self._name = name
        self._tls = threading.local()

    def _stack(self):
        s = getattr(self._tls, "stack", None)
        if s is None:
            s = []
            self._tls.stack = s
        return s

    def __enter__(self):
        if ENABLE_PROFILE_TRACE:
            t0 = time.perf_counter()
            self._lock.acquire()
            t1 = time.perf_counter()
            self._stack().append((t0, t1))
        else:
            self._lock.acquire()
        return self

    def __exit__(self, *_):
        if ENABLE_PROFILE_TRACE:
            stack = self._stack()
            if stack:
                t0, t1 = stack.pop()
                t2 = time.perf_counter()
                acquire_wait_ms = (t1 - t0) * 1000.0
                hold_ms = (t2 - t1) * 1000.0
                ts_ms = int(time.time() * 1000)
                thread_name = threading.current_thread().name
                trace(
                    "lock_trace.csv",
                    "ts_ms,duration_ms,lock_name,thread,acquire_wait_ms,hold_ms",
                    [ts_ms, f"{(acquire_wait_ms + hold_ms):.3f}", self._name,
                     thread_name, f"{acquire_wait_ms:.3f}", f"{hold_ms:.3f}"],
                )
        self._lock.release()
        return False

    # Pass-through API for code that calls acquire()/release() directly.
    # NOTE: these paths do NOT emit trace rows — only `with` context records
    # (common case, keeps hot path simple). Code that needs tracing on
    # explicit acquire/release can wrap the operation in `with self.lock:`.
    def acquire(self, *a, **kw):
        return self._lock.acquire(*a, **kw)

    def release(self):
        return self._lock.release()

    @property
    def name(self):
        return self._name


# Production default: instrumentation OFF unless LVP_PROFILE_TRACE=1 is
# set explicitly in the environment. The perf-instrumentation-4.0.0-beta
# branch flipped this to default-ON for the STALL-1 diagnostic run; the
# merge into the layer-audit chain restores the explicit env-var gate so
# production carries the tracer infrastructure for opt-in use without
# the always-on file-write overhead. To enable: set LVP_PROFILE_TRACE=1
# (any non-empty non-"0" value works in shell-script practice — exact
# match required here for clarity).
if os.environ.get("LVP_PROFILE_TRACE") == "1":
    enable()
