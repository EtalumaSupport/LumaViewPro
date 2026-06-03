# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Opt-in runtime tracing for profiling + debugging.

Default OFF. Zero overhead when disabled -- every trace site is guarded
by a single module-level flag check.

Enable two ways:
  1. Set ``profile_trace_enabled: true`` in data/settings.json (or
     data/current.json) before launching LVP. Optionally set
     ``profile_trace_output_dir`` to override the output directory.
  2. Call ``profile_trace.enable()`` programmatically (tests, ad-hoc
     experiments).

Writes CSV files under `./logs/profile/<timestamp>/` by default:
  - serial_trace.csv        (SerialBoard.exchange_command timings)
  - motion_trace.csv        (motion-monitor poll durations + axis state transitions)
  - frame_validity_trace.csv (invalidate/count/settle events)

Columns are documented in the trace-site wrappers (see timer() and trace()
callers in drivers/serialboard.py, modules/lumascope_api.py,
modules/frame_validity.py).

CSVs auto-close on process exit via atexit. Thread-safe via a single
module-level lock. Writes are line-buffered -- no tail-buffer loss on crash.
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


# Diagnostic trace gate, toggled explicitly via enable() / disable() (not
# mirrored from a setting). Single global owning its own on/off state -- not
# the divergent cached-copy shape; reads here always see the latest toggle.
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
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('./logs/profile') / ts
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    ENABLE_PROFILE_TRACE = True
    atexit.register(disable)
    logger.info(f'[PROFILE   ] Trace enabled. Writing to {_output_dir}')


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
                fh = open(path, 'a', buffering=1)
                if need_header:
                    fh.write(header + '\n')
                _writers[filename] = fh
            fh.write(','.join(str(x) for x in fields) + '\n')
    except Exception as e:
        logger.warning(f'[PROFILE   ] trace write failed ({filename}): {e}')


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

    __slots__ = ('extra_fn', 'filename', 'header', 't0')

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
                logger.warning(f'[PROFILE   ] timer extra_fn failed: {e}')
                return
            trace(self.filename, self.header, [ts_ms, f'{dt_ms:.3f}', *extra])


class TimedLock:
    """Drop-in wrapper for threading.Lock / threading.RLock that records
    acquire-wait + hold time per acquire-release cycle to `lock_trace.csv`
    when ``profile_trace_enabled`` is set in settings.json.

    Validates SerialBoard._lock hold-time claim (~32 ms per round-trip,
    documented at drivers/motorboard.py:79) across more sessions, and
    surfaces outliers. Zero overhead when tracing is disabled --
    __enter__/__exit__ short-circuit before time.perf_counter().

    Thread-safe for RLock re-entry: uses a per-instance thread-local
    stack of (t_wait_start, t_held_start) tuples so nested
    `with self._rlock: ... with self._rlock: ...` correctly records
    outer and inner acquire times independently instead of clobbering.

    Usage (same as threading.Lock):
        self._led_lock = TimedLock(threading.RLock(), name="led_lock")
        with self._led_lock:
            ...

    Also supports acquire()/release() for code that uses them directly.

    Optional hold-duration invariant via ``warn_hold_threshold_ms``: when
    set, the lock fires a logger.warning at __exit__ time if the hold
    duration exceeded the threshold. Active regardless of the trace-CSV
    feature flag -- it's a structural guard, not an instrumentation
    knob. Use for locks with a documented "never hold across X" rule
    (motion._axis_state_lock has a 1 ms invariant; LED owners lock
    has a similar guard for serial-call hosts).
    """

    __slots__ = ('_lock', '_name', '_tls', '_warn_hold_threshold_ms')

    def __init__(self, lock, name, warn_hold_threshold_ms=None):
        self._lock = lock
        self._name = name
        self._warn_hold_threshold_ms = warn_hold_threshold_ms
        self._tls = threading.local()

    def _stack(self):
        s = getattr(self._tls, 'stack', None)
        if s is None:
            s = []
            self._tls.stack = s
        return s

    def __enter__(self):
        # Time the acquire only when tracing OR a hold-threshold is set;
        # both consumers need the t0/t1 snapshot stored on the per-thread
        # stack so __exit__ can compute hold_ms.
        if ENABLE_PROFILE_TRACE or self._warn_hold_threshold_ms is not None:
            t0 = time.perf_counter()
            self._lock.acquire()
            t1 = time.perf_counter()
            self._stack().append((t0, t1))
        else:
            self._lock.acquire()
        return self

    def __exit__(self, *_):
        if ENABLE_PROFILE_TRACE or self._warn_hold_threshold_ms is not None:
            stack = self._stack()
            if stack:
                t0, t1 = stack.pop()
                t2 = time.perf_counter()
                acquire_wait_ms = (t1 - t0) * 1000.0
                hold_ms = (t2 - t1) * 1000.0

                # Structural invariant guard. Fires regardless of the
                # trace-CSV flag because the rule is "never hold this
                # lock for X ms" -- a real bug, not an instrumentation
                # signal. Uses the per-lock-instance threshold so other
                # locks pay zero cost.
                if (
                    self._warn_hold_threshold_ms is not None
                    and hold_ms > self._warn_hold_threshold_ms
                ):
                    from lvp_logger import logger as _lock_logger

                    _lock_logger.warning(
                        f'[LOCK] {self._name} held {hold_ms:.2f}ms by '
                        f'{threading.current_thread().name} -- '
                        f'invariant threshold {self._warn_hold_threshold_ms}ms exceeded'
                    )

                if ENABLE_PROFILE_TRACE:
                    ts_ms = int(time.time() * 1000)
                    thread_name = threading.current_thread().name
                    trace(
                        'lock_trace.csv',
                        'ts_ms,duration_ms,lock_name,thread,acquire_wait_ms,hold_ms',
                        [
                            ts_ms,
                            f'{(acquire_wait_ms + hold_ms):.3f}',
                            self._name,
                            thread_name,
                            f'{acquire_wait_ms:.3f}',
                            f'{hold_ms:.3f}',
                        ],
                    )
        self._lock.release()
        return False

    # Pass-through API for code that calls acquire()/release() directly.
    # NOTE: these paths do NOT emit trace rows -- only `with` context records
    # (common case, keeps hot path simple). Code that needs tracing on
    # explicit acquire/release can wrap the operation in `with self.lock:`.
    def acquire(self, *a, **kw):
        return self._lock.acquire(*a, **kw)

    def release(self):
        return self._lock.release()

    @property
    def name(self):
        return self._name


# Production default: instrumentation OFF unless profile_trace_enabled
# is true in data/settings.json (or data/current.json). Optional sibling
# key profile_trace_output_dir overrides the default ./logs/profile/<TS>/
# location. Read at module-import time -- the same timing as
# load_debug_setting() in lvp_logger.py -- so the gate is decided before
# any trace site fires. Defaults to OFF + None on any read failure so
# the tracer infrastructure remains shippable without runtime config.
def _read_settings_gate():
    from modules.settings_init import load_profile_trace_setting

    # Reuse lvp_logger.lvp_appdata so the production-installed path
    # (~/Documents/LumaViewPro <version>/data/) resolves the same way
    # the logger's debug-mode gate does. Fall back to the source root
    # when lvp_logger isn't importable (e.g. unit tests that exercise
    # this module in isolation).
    try:
        import lvp_logger

        base_dir = lvp_logger.lvp_appdata
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = load_profile_trace_setting(base_dir)
    # Tests that register a bare MagicMock as `modules.settings_init`
    # (without configuring `load_profile_trace_setting`) cause the call
    # above to return a MagicMock. The MagicMock is truthy under
    # `result['enabled']` and Path-stringifiable as `result['output_dir']`,
    # which produced a stray `LumaViewPro/MagicMock/` directory at the
    # repo root. Treat any non-dict return as the safe-OFF default.
    if not isinstance(result, dict):
        return {'enabled': False, 'output_dir': None}
    return result


_gate = _read_settings_gate()
if _gate['enabled']:
    enable(output_dir=Path(_gate['output_dir']) if _gate['output_dir'] else None)
