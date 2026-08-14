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
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import threading
import time
import weakref
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
_base_dir = None
_run_index = 0
_atexit_registered = False
_lock = threading.Lock()
_writers = {}
_batch_traces = []

# Identity for a row emitted outside any recording (serial, motion,
# frame-validity). Sites pass it explicitly rather than omitting the argument:
# a file whose rows cannot be partitioned by recording gets averaged across
# recordings, and that produces a plausible wrong rate instead of a visible
# failure.
NO_RECORDING = '-'


def enable(output_dir=None):
    """Open a new run directory and start writing trace CSVs.

    Args:
        output_dir: base directory the timestamped run directories go under.
            Defaults to ``./logs/profile``.

    Every call starts a RUN: an already-running one is sealed first (batches
    flushed, handles closed) and a fresh directory opened. This used to return
    early when tracing was already on, which made a second run within one
    launch impossible -- so a session that changed one axis between recordings
    appended both to the same CSVs with nothing in the rows saying which
    configuration produced them.
    """
    global ENABLE_PROFILE_TRACE, _output_dir, _base_dir, _run_index, _atexit_registered
    if ENABLE_PROFILE_TRACE:
        disable()
    if output_dir is not None:
        _base_dir = Path(output_dir)
    elif _base_dir is None:
        _base_dir = Path('./logs/profile')
    _run_index += 1
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # The run index is in the name, not just the timestamp: two enable() calls
    # inside one second would otherwise resolve to the same directory, which is
    # the merge this exists to prevent.
    _output_dir = _base_dir / f'{ts}_run{_run_index}'
    _output_dir.mkdir(parents=True, exist_ok=True)
    ENABLE_PROFILE_TRACE = True
    if not _atexit_registered:
        atexit.register(disable)
        _atexit_registered = True
    _write_run_info()
    logger.info(f'[PROFILE   ] Trace enabled. Writing to {_output_dir}')


def _write_run_info():
    """Record what produced this run's rows, beside the rows themselves.

    Numbers read months later are worth little without the host and build
    that made them, and neither is recoverable from the CSVs. Written here
    rather than asked of callers: these are process-wide facts, so a caller
    supplying them is one more thing to forget. Written once per run, and
    only when tracing is on, so a disabled build never pays for the
    subprocesses this fires.
    """
    if _output_dir is None:
        return
    try:
        info = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'defender': _defender_state(),
        }
        info.update(_build_identity(Path(__file__).resolve().parent.parent))
        (_output_dir / 'run_info.json').write_text(json.dumps(info, indent=2, default=str))
    except Exception as e:
        logger.warning(f'[PROFILE   ] run info write failed: {e}')


def _build_identity(repo_root):
    """Name the build that produced this run, and say where the name came from.

    `git rev-parse` is preferred and is the only source that cannot go stale.
    It is also unavailable exactly where these runs happen: a source tree
    downloaded as a zip has no `.git`, so git answers nothing on the bench
    machine while answering fine on a developer clone. version.txt is the
    fallback rather than the primary because it does not refresh under a
    source run -- it has carried a SHA naming a commit that exists in no
    repository.

    Every field is paired with `build_identity_source` so a reader never has
    to guess which one answered. A bare null would say "unknown" in the one
    place whose whole purpose is stating what produced the rows.
    """
    sha = _git(repo_root, 'rev-parse', 'HEAD')
    if sha:
        return {
            'build_identity_source': 'git',
            'git_sha': sha,
            'git_branch': _git(repo_root, 'rev-parse', '--abbrev-ref', 'HEAD'),
            'git_dirty': bool(_git(repo_root, 'status', '--porcelain')),
        }
    identity = {
        'build_identity_source': 'fallback -- no git metadata in this source tree',
        'git_sha': None,
        'git_branch': None,
        'git_dirty': None,
        'source_dir': repo_root.name,
    }
    try:
        identity['version_txt'] = (repo_root / 'version.txt').read_text().strip()[:200]
    except OSError as e:
        identity['version_txt'] = f'unreadable: {e}'
    return identity


def _git(repo_root, *args):
    """Run one read-only git command in the repo, or return None."""
    try:
        out = subprocess.run(
            ['git', *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _defender_state():
    """Real-time monitoring state and exclusion paths, on Windows.

    Per-frame write cost gets compared across hosts, and an antivirus
    exclusion on one of them explains a difference that would otherwise be
    attributed to the hardware. Asking the host beats asking the operator to
    remember.
    """
    if not platform.system().startswith('Win'):
        return 'not_windows'
    try:
        out = subprocess.run(
            [
                'powershell',
                '-NoProfile',
                '-Command',
                '$p = Get-MpPreference; '
                '"realtime_disabled=$($p.DisableRealtimeMonitoring);'
                "exclusions=$($p.ExclusionPath -join ';')\"",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else 'query_failed'
    except Exception:
        return 'query_failed'


def disable():
    """Flush and close all trace files. Safe to call if already disabled."""
    global ENABLE_PROFILE_TRACE
    if not ENABLE_PROFILE_TRACE:
        return
    ENABLE_PROFILE_TRACE = False
    # Drain pending batches BEFORE closing the handles they write through.
    # flush() gates on its own row list rather than the flag, so it still
    # works after the flag is cleared.
    with _lock:
        # Drop refs whose owner has already been collected while we hold the
        # lock anyway, so the registry cannot grow without bound across a long
        # session of short recordings.
        live = [bt for bt in (ref() for ref in _batch_traces) if bt is not None]
        _batch_traces[:] = [weakref.ref(bt) for bt in live]
    for bt in live:
        bt.flush()
    with _lock:
        for fh in _writers.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        _writers.clear()
    _write_csv_md5()


def _write_csv_md5():
    """Seal the run with an md5 of every CSV it emitted.

    Two bench archives were once byte-identical -- one session delivered
    under two names -- and it was caught by hand after both had already been
    counted as independent replications. Comparing this file across runs
    answers that in one diff. Best-effort: a run that cannot be sealed is
    still a run whose rows are worth having.
    """
    if _output_dir is None:
        return
    try:
        lines = []
        for path in sorted(_output_dir.glob('*.csv')):
            # md5 detects a duplicated archive; it guards nothing.
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            lines.append(f'{digest}  {path.name}')
        if lines:
            (_output_dir / 'csv_md5.txt').write_text('\n'.join(lines) + '\n')
    except Exception as e:
        logger.warning(f'[PROFILE   ] csv md5 seal failed: {e}')


def _check_arity(filename, header, fields):
    """Raise when a row would misalign against its header.

    A misaligned row is worse than a missing one: it still parses, so every
    column past the short one silently shifts and the damage surfaces as wrong
    numbers rather than as an error. A header-shape check cannot cover this --
    the sites that get it wrong are the ones a given run never exercises -- so
    the check has to be per row.
    """
    expected = header.count(',') + 1
    if len(fields) != expected:
        raise ValueError(f'{filename}: row has {len(fields)} fields, header declares {expected}')


def _write_rows_unlocked(filename, header, rows):
    """Append rows to one trace file, opening and heading it on first use.

    Caller holds ``_lock``. This is the ONLY place a trace file is opened and
    the only place a row is serialized, so the row-at-a-time and the batched
    writer cannot drift apart on quoting or on line endings. Splitting this is
    how one writer's escaping rules come to differ from the other's.
    """
    fh = _writers.get(filename)
    if fh is None:
        path = _output_dir / filename
        need_header = not path.exists()
        # newline='' is csv's requirement, not a preference: without it the
        # writer's terminator gets translated again on Windows and every row
        # is followed by a blank one. utf-8 is explicit because traced fields
        # carry user-supplied text -- a step name outside the platform's
        # default codepage would raise inside the callers' except and lose the
        # row silently.
        fh = open(path, 'a', newline='', encoding='utf-8', buffering=1)  # noqa: SIM115 -- long-lived handle stored in _writers, closed in disable() via atexit
        if need_header:
            csv.writer(fh, lineterminator='\n').writerow(['recording_id', *header.split(',')])
        _writers[filename] = fh
    csv.writer(fh, lineterminator='\n').writerows(rows)


def trace(filename, header, fields, *, recording_id):
    """Append one row to the named CSV. No-op when disabled.

    Args:
        filename: CSV basename inside the active output directory.
        header: comma-separated column names, excluding the identity column.
        fields: one value per header column.
        recording_id: the recording this row belongs to, or NO_RECORDING.
            Keyword-only and required so that no site can emit an
            unattributable row.

    A field's CONTENT can never change how many columns the row has: the row
    goes through csv.writer, which quotes commas, quotes and newlines. Call
    sites used to carry that duty themselves and it was honoured at some and
    missed at others -- the motion monitor's axis field is a ','.join of the
    moving axes, so every simultaneous XY poll wrote one column too many under
    a header that never changed shape. Rows like that still parse, so the
    damage arrives as quietly wrong numbers rather than as an error.
    """
    if not ENABLE_PROFILE_TRACE:
        return
    _check_arity(filename, header, fields)
    try:
        with _lock:
            _write_rows_unlocked(filename, header, [[recording_id, *(str(x) for x in fields)]])
    except Exception as e:
        logger.warning(f'[PROFILE   ] trace write failed ({filename}): {e}')


class BatchTrace:
    """Accumulate rows in memory, write them to CSV in batches.

    For trace sites whose per-row cost would perturb what they measure.
    ``trace()`` takes the module-wide lock and the CSV handles are
    line-buffered, so every row costs a syscall serialized against every other
    trace site. On a camera SDK grab thread delivering 40+ fps that cost lands
    inside the very inter-frame interval being timed, and it biases the answer
    in one direction: the sink looks slower than it is. A row here costs a list
    append; the syscall amortizes over ``batch_size`` frames.

    Single-writer per instance: ``add()`` takes no lock, so each instance must
    be appended to from exactly one thread. The driver fan-out and the record
    callback both satisfy this -- each SDK fire-site is single-threaded. A
    shared instance would need a lock, reintroducing the contention this
    exists to avoid.
    """

    __slots__ = ('__weakref__', '_batch_size', '_filename', '_header', '_recording_id', '_rows')

    def __init__(self, filename, header, recording_id, batch_size=200):
        """Open a batched writer bound to one recording.

        Args:
            filename: CSV basename inside the active output directory.
            header: comma-separated column names, excluding the identity column.
            recording_id: the recording every row from this instance belongs
                to, or NO_RECORDING. Required and positional: recordings
                overlap -- a step's write runs on the file lane while the next
                step captures -- so identity cannot come from a module-level
                "current recording" without attributing rows to the wrong one.
            batch_size: rows accumulated before a write.
        """
        self._filename = filename
        self._header = header
        self._recording_id = recording_id
        self._batch_size = batch_size
        self._rows = []
        # Registry append is gated: with tracing off this instance can never
        # accumulate a row, so registering it would grow a list that teardown
        # walks for nothing -- on a path that constructs one per camera handler
        # in every shipped build.
        if ENABLE_PROFILE_TRACE:
            with _lock:
                # Weak refs: an instance lives as long as its owner, and the
                # registry exists only to flush stragglers at teardown. A
                # strong ref would retain every writer, and its buffered rows,
                # for the life of the process.
                _batch_traces.append(weakref.ref(self))

    def add(self, fields):
        """Record one row. No-op when disabled."""
        if not ENABLE_PROFILE_TRACE:
            return
        _check_arity(self._filename, self._header, fields)
        self._rows.append(fields)
        if len(self._rows) >= self._batch_size:
            self.flush()

    def flush(self):
        """Write accumulated rows. Safe to call when empty or disabled."""
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        rid = self._recording_id
        try:
            with _lock:
                _write_rows_unlocked(
                    self._filename,
                    self._header,
                    [[rid, *(str(x) for x in r)] for r in rows],
                )
        except Exception as e:
            logger.warning(f'[PROFILE   ] batch trace write failed ({self._filename}): {e}')


class timer:  # noqa: N801 -- deliberate stdlib-style lowercase context-manager name, used as profile_trace.timer(...) across files
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
            # Timed operations (serial round-trips, motion polls) are not
            # scoped to a recording, so they declare that rather than
            # inheriting an ambient one.
            trace(
                self.filename,
                self.header,
                [ts_ms, f'{dt_ms:.3f}', *extra],
                recording_id=NO_RECORDING,
            )


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
