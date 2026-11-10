# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-12 -- single owner for periodic runtime-health logging.

Collects what was previously scattered across lumaviewpro.py:

- 60 s system metrics (CPU, memory, handles, GC, buffer churn) via
  ``config_helpers.log_system_metrics``.
- 60 s executor queue depth snapshot + auto-prune of SCOPEDISPLAY
  backlog (was the ``_executor_watchdog`` closure).
- 4 hr camera temperatures via ``Lumascope._log_camera_temps``.

One module, one ``MetricsLogger`` class, three ``tick_*`` methods.
Every entry point boots LVP through the same ``start(...)`` call so
REST API + headless test runner + future CLI tools get identical
runtime-health logs without copy-paste. Engineering plugin / REST
status endpoint can also call the ``snapshot_*`` building blocks
directly to dump current state on demand without waiting for the
next tick.

Stays GUI-agnostic -- takes a ``Scheduler`` instead of importing Kivy.
"""

from __future__ import annotations

import datetime
import sys
import threading
import time
import traceback
from collections.abc import Callable

from _lumascope.lvp_logger import logger
import _lumascope.modules.config_helpers as config_helpers
from _lumascope.modules import wall_time
from _lumascope.modules.ui_queue import schedule_ui, ui_thread
from _lumascope.modules.scheduler import Scheduler


# Default cadences. system_metrics is the verbose snapshot (CPU, RAM,
# GC, page-faults, Defender, buffer-churn, handle counts, queue depth,
# caller_futures alloc/pop drift, frame-interval percentiles).
#
# This default was 60 s for three months, on the strength of a comment
# claiming the snapshot cost ~10-50 ms per tick. Measurement later put
# the open-files probe ALONE at 134-306 ms on healthy Windows hosts --
# the estimate was wrong by 3-30x, and nothing re-derived the cadence
# when the real number arrived. That probe is now opt-in, and this
# returns to the hourly cadence production always meant to run.
#
# Sub-minute sampling still exists where it is needed: a bench soak
# wants it, because the leak-rate budgets in PERFORMANCE_BUDGETS.md are
# per-hour figures and an hourly tick yields one sample. So engineering
# mode selects 60 s automatically -- the alternative, asking every bench
# operator to remember a settings override, is exactly the arrangement
# whose being forgotten produced a 38-minute soak with a single snapshot
# and no measurable handle-growth rate.
#
# The trade production accepts: metrics derived through _delta_rate
# (io_read_mbps, io_write_mbps, page_faults_per_sec) become hour-averages
# off-bench. Engineering plugin / REST status endpoint can still call
# snapshot_* on demand, and any caller can override via
# settings.profiling.metrics_interval_s.
DEFAULT_SYSTEM_METRICS_INTERVAL_S = 3600.0
# What engineering mode selects instead: sub-minute, so a soak shorter than
# an hour still produces the several samples a per-hour leak rate needs.
ENGINEERING_METRICS_INTERVAL_S = 60.0
DEFAULT_EXECUTOR_WATCHDOG_INTERVAL_S = 60.0
DEFAULT_CAMERA_TEMP_INTERVAL_S = 14400.0

# The heartbeat's cadence, and the gap between two of its runs that says no
# thread of the process ran. The gap is under the GUI's one-second hold
# report, so every hold it reports is judged against it, and over the
# cadence by enough that a busy host's late wake is not a stop.
HEARTBEAT_INTERVAL_S = 0.25
STOP_GAP_S = 1.0

# Backlog thresholds used by the executor-watchdog tick. Match the
# pre-LVP-A-12 values so behavior is identical.
_EXECUTOR_BACKLOG_WARN_TOTAL = 10
_SCOPE_DISPLAY_PRUNE_THRESHOLD = 20


class MetricsLogger:
    """Owns the periodic runtime-health logging surface for LVP."""

    def __init__(self, scope, executor_bundle, settings_snapshot: Callable[[], dict]):
        """Hold references; call ``start()`` to begin the schedules.

        Args:
            scope: ``Lumascope`` API instance -- used by the camera-temp
                tick (delegates to ``scope.imaging._log_camera_temps``) and any
                future tick that needs hardware access.
            executor_bundle: ``_lumascope.modules.executor_registry.ExecutorBundle``
                -- the executor watchdog reads its ``snapshot()`` and
                ``thread_states()``.
            settings_snapshot: Returns a private copy of the session's
                settings; each system tick takes one and hands it to
                ``config_helpers.log_system_metrics``, since the tick runs on
                the session's timer, not the thread that writes the settings.
        """
        self._scope = scope
        self._bundle = executor_bundle
        self._settings_snapshot = settings_snapshot
        # The host's live display, handed in at start(); None on a host
        # with none.
        self._display: config_helpers.DisplayRates | None = None

        # Scheduler bound at start() -- None means not started.
        self._scheduler: Scheduler | None = None
        # Active schedule handles per tick (so each can be cancelled
        # independently). Keys: 'heartbeat', 'system_metrics',
        # 'executor_watchdog', 'ui_watch'.
        # Camera-temp lives on Lumascope already (LVP-A-2) -- not stored
        # here because Lumascope owns its own handle.
        self._handles: dict[str, object] = {}

        # The UI thread's watch: the probe posted to it and not yet run (its
        # token and when it was posted), and whether its hold was reported.
        # Written by the watchdog tick and by the probe on the UI thread.
        self._ui_watch_lock = threading.Lock()
        self._ui_probe: object | None = None
        self._ui_probe_posted_at = 0.0
        self._ui_hold_reported = False

        # The heartbeat: when it last ran, and the last stretch in which no
        # thread of the process ran, on time.monotonic; None while it is not
        # running. Written by its tick; read by every report of a held UI
        # thread.
        self._beat_lock = threading.Lock()
        self._last_beat: float | None = None
        self._last_stop: tuple[float, float] | None = None

    # ---- Tick implementations (also callable on-demand) ----

    def tick_system_metrics(self) -> None:
        """One snapshot of CPU/memory/handles/GC/buffer-churn metrics.

        Internal metrics scheduling -- driven by the host's periodic
        scheduler and not part of the L2 API surface; an L2 caller reads
        metrics from the log surface below rather than driving a tick.

        Delegates to ``config_helpers.log_system_metrics`` so the format
        + content match the existing log surface; engineering tools
        that grep ``[PDH METRICS]`` / ``[BUFFER METRICS]`` keep working.
        Its callers are this logger's own: its start and its scheduled
        cadence.
        """
        try:
            config_helpers.log_system_metrics(
                self._settings_snapshot(),
                scope=self._scope,
                executors=self._bundle,
                display=self._display,
            )
        except Exception as e:
            logger.warning(f'[MetricsLogger] tick_system_metrics failed: {type(e).__name__}: {e}')

    def tick_heartbeat(self) -> None:
        """Note that the process ran; say once when it finds that it did not.

        Internal metrics scheduling -- driven by the host's periodic
        scheduler and not part of the L2 API surface.

        The tick runs on its own timer thread a few times a second, so a gap
        between two of its runs past ``STOP_GAP_S`` is a stretch in which no
        thread of the process ran: the operating system stopped it (a
        standby suspends it), or one call held the interpreter, which stops
        every Python thread. The two cannot be told apart from inside, so
        the line names both. A field bundle gave a standby's hour to the UI
        thread; every report of a held UI thread takes such a stretch out of
        what it gives the thread (``stopped_within``).
        """
        now = time.monotonic()
        with self._beat_lock:
            previous, self._last_beat = self._last_beat, now
            if previous is None or now - previous <= STOP_GAP_S:
                return
            self._last_stop = (previous, now)
        gap = now - previous
        ended = wall_time.now()
        began = ended - datetime.timedelta(seconds=gap)
        logger.warning(
            f'[Watchdog  ] LVP ran no thread for {gap:.1f} s, '
            f'{began.time().isoformat(timespec="milliseconds")} - '
            f'{ended.time().isoformat(timespec="milliseconds")} '
            '(the process was stopped, as in a standby, or one call held the interpreter)'
        )

    def stopped_within(self, began: float, ended: float) -> float:
        """The seconds of ``began`` to ``ended`` (``time.monotonic``) in which no thread of the process ran.

        Internal: the reports of a held UI thread and the live view's frame
        judges ask it, so none gives a stretch the process did not run to
        the UI thread or the display. It counts the last
        recorded stop and a gap still open: after a resume a report can run
        before the heartbeat's first run does, and the heartbeat's silence
        then is the stop not yet recorded. Zero while the heartbeat is not
        running.
        """
        now = time.monotonic()
        with self._beat_lock:
            stops = [self._last_stop] if self._last_stop is not None else []
            if self._last_beat is not None and now - self._last_beat > STOP_GAP_S:
                stops.append((self._last_beat, now))
        return sum(max(0.0, min(ended, end) - max(began, start)) for start, end in stops)

    def tick_executor_watchdog(self) -> None:
        """Log the lanes' queue depths and the queueless threads' states.

        Internal metrics scheduling -- driven by the host's periodic
        scheduler and not part of the L2 API surface.

        WARNING-level log when total backlog exceeds 10; DEBUG otherwise.
        The threshold is carried unchanged from the inline watchdog this
        replaced, so log lines stay comparable across that change.

        A snapshot that fails raises to the scheduler, whose error hook
        reports it; the timer re-arms either way.
        """
        depths = self._bundle.snapshot()
        total_q = sum(depths.values())
        fmt = ' '.join(
            f'{name}:{value}' for name, value in {**depths, **self._bundle.thread_states()}.items()
        )
        if total_q > _EXECUTOR_BACKLOG_WARN_TOTAL:
            logger.warning(f'[Watchdog  ] Queue backlog ({total_q} total) -- {fmt}')
        else:
            logger.debug(f'[Watchdog  ] Queues -- {fmt}')

        # SCOPEDISPLAY has no queue to prune. Watchdog has no actionable
        # response if the thread itself is wedged -- that's a
        # daemon=True process-exit-reap scenario, not a queue prune.

    def tick_ui_watch(self) -> None:
        """Post a probe to the UI thread; report once when the last one has not run.

        Internal metrics scheduling -- driven by the host's periodic
        scheduler and not part of the L2 API surface.

        A field GUI froze for four days and no log said so: the executor
        watchdog ran on its own timer thread throughout but read only the
        lanes. A probe posted at one tick and still waiting at the next
        means the UI thread has run nothing for at least a tick, so a hold
        is reported one to two ticks after it starts, with the stack the
        thread is in. One probe is outstanding at a time. A host with no UI
        thread posts none. A freeze that holds the GIL stops this timer
        too, so it cannot be seen from here. A probe whose wait spans a
        stretch the process did not run (``stopped_within``) is replaced,
        not reported: the heartbeat says that stretch, and a hold after it
        is seen from the new probe.
        """
        thread = ui_thread()
        if thread is None:
            return
        now = time.monotonic()
        with self._ui_watch_lock:
            outlived_a_stop = (
                self._ui_probe is not None
                and not self._ui_hold_reported
                and self.stopped_within(self._ui_probe_posted_at, now) > 0
            )
            if self._ui_probe is None or outlived_a_stop:
                probe = self._ui_probe = object()
                self._ui_probe_posted_at = now
                self._ui_hold_reported = False
            elif self._ui_hold_reported:
                return
            else:
                self._ui_hold_reported = True
                probe = None
                waited_s = now - self._ui_probe_posted_at
        if probe is not None:
            schedule_ui(lambda dt: self._ui_probe_ran(probe))
            return
        frame = sys._current_frames().get(thread.ident)
        stack = ''.join(traceback.format_stack(frame)) if frame is not None else '(no frame)\n'
        logger.warning(
            f'[Watchdog  ] UI thread has run nothing for at least {waited_s:.0f} s; '
            f'it is in:\n{stack.rstrip()}'
        )

    def _ui_probe_ran(self, probe: object) -> None:
        """The probe, on the UI thread: the thread ran; say so if its hold was reported.

        The probe was posted at a tick inside the hold, not at its start, so
        its wait is the hold's lower bound, and the line says so. A stretch
        the process did not run after the hold was reported is not the
        thread's, and the line takes it out.
        """
        now = time.monotonic()
        with self._ui_watch_lock:
            if probe is not self._ui_probe:
                return
            self._ui_probe = None
            reported = self._ui_hold_reported
            posted_at = self._ui_probe_posted_at
        if not reported:
            return
        stopped_s = self.stopped_within(posted_at, now)
        stopped = f'; for {stopped_s:.0f} s of it no LVP thread ran' if stopped_s else ''
        logger.info(
            f'[Watchdog  ] UI thread ran again; it was held at least {now - posted_at - stopped_s:.0f} s'
            f'{stopped}'
        )

    # ---- Lifecycle ----

    def start(
        self,
        scheduler: Scheduler,
        *,
        system_metrics_interval_s: float = DEFAULT_SYSTEM_METRICS_INTERVAL_S,
        executor_watchdog_interval_s: float = DEFAULT_EXECUTOR_WATCHDOG_INTERVAL_S,
        camera_temp_interval_s: float = DEFAULT_CAMERA_TEMP_INTERVAL_S,
        start_camera_temp: bool | None = None,
        display: config_helpers.DisplayRates | None = None,
    ) -> None:
        """Schedule all periodic ticks.

        Args:
            scheduler: The session's scheduler, which fires every tick.
            system_metrics_interval_s: How often to emit the
                CPU/memory/handles snapshot.
            executor_watchdog_interval_s: How often to emit the executor
                queue-depth snapshot + run the prune guard.
            camera_temp_interval_s: Forwarded to
                ``Lumascope.start_camera_temp_logging`` so the API still
                owns the camera-temp event handle (LVP-A-2).
            start_camera_temp: If None (default), starts the camera-temp
                logger only when ``scope.camera_connected``. Pass False to
                skip even when connected (rare; mostly tests).
            display: The host's live display, whose rates the system tick
                logs beside the camera's; None on a host with none.
        """
        self._display = display
        self._scheduler = scheduler

        # First, so every report of a held UI thread has it to ask.
        self._handles['heartbeat'] = self._scheduler.schedule_interval(
            self.tick_heartbeat, HEARTBEAT_INTERVAL_S
        )

        # Initial snapshot -- match the pre-LVP-A-12 behavior of logging
        # once on startup so the very first log line carries fingerprint
        # values rather than empty cells.
        self.tick_system_metrics()

        self._handles['system_metrics'] = self._scheduler.schedule_interval(
            self.tick_system_metrics, system_metrics_interval_s
        )
        self._handles['executor_watchdog'] = self._scheduler.schedule_interval(
            self.tick_executor_watchdog, executor_watchdog_interval_s
        )
        # Its own handle at the watchdog's cadence, so a snapshot that
        # raises every tick cannot also stop the UI thread's watch.
        self._handles['ui_watch'] = self._scheduler.schedule_interval(
            self.tick_ui_watch, executor_watchdog_interval_s
        )

        if start_camera_temp is False:
            return
        if start_camera_temp is None and not self._scope.camera_connected:
            return

        # Camera-temp scheduling stays inside Lumascope so the API keeps
        # full ownership of the event handle; the tick skips samples while
        # the camera is disconnected but never self-cancels (a transient
        # connectivity False must not end logging), and the schedule is
        # torn down by this logger's stop() and by Lumascope.disconnect().
        # Hand it adapter callables matching the Scheduler so Lumascope
        # doesn't need to learn about Scheduler.
        self._scope.imaging.start_camera_temp_logging(
            self._scheduler.schedule_interval,
            self._scheduler.unschedule,
            schedule_once_fn=self._scheduler.schedule_once,
            interval_s=camera_temp_interval_s,
        )

        logger.info(
            f'[MetricsLogger] started: system_metrics={system_metrics_interval_s}s, '
            f'executor_watchdog={executor_watchdog_interval_s}s, '
            f'camera_temp={camera_temp_interval_s}s'
        )

    def stop(self) -> None:
        """Cancel every scheduled tick. Idempotent."""
        # A probe still queued on the UI thread runs after this; it finds
        # none outstanding and logs nothing.
        with self._ui_watch_lock:
            self._ui_probe = None
        if self._scheduler is None:
            return
        for name, handle in list(self._handles.items()):
            try:
                self._scheduler.unschedule(handle)
            except Exception as e:
                logger.warning(f'[MetricsLogger] unschedule {name} failed: {e}')
        self._handles.clear()
        # Once its timer is gone, a heartbeat no longer running reads as no
        # stop at all, not as one growing until the next start.
        with self._beat_lock:
            self._last_beat = self._last_stop = None
        try:
            self._scope.imaging.stop_camera_temp_logging(self._scheduler.unschedule)
        except Exception as e:
            logger.warning(f'[MetricsLogger] stop_camera_temp_logging failed: {e}')
