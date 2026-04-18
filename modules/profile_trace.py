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


if os.environ.get("LVP_PROFILE_TRACE") == "1":
    enable()
