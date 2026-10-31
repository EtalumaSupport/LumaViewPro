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

Stays GUI-agnostic -- takes Clock-style scheduler / unscheduler
callables instead of importing Kivy.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections.abc import Callable

from lvp_logger import logger
import modules.config_helpers as config_helpers
from modules.kivy_utils import schedule_ui, ui_thread
from modules.scheduler import Scheduler, _CallablePairScheduler


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

# Backlog thresholds used by the executor-watchdog tick. Match the
# pre-LVP-A-12 values so behavior is identical.
_EXECUTOR_BACKLOG_WARN_TOTAL = 10
_SCOPE_DISPLAY_PRUNE_THRESHOLD = 20


class MetricsLogger:
    """Owns the periodic runtime-health logging surface for LVP."""

    def __init__(self, scope, executor_bundle, settings):
        """Hold references; call ``start()`` to begin the schedules.

        Args:
            scope: ``Lumascope`` API instance -- used by the camera-temp
                tick (delegates to ``scope.imaging._log_camera_temps``) and any
                future tick that needs hardware access.
            executor_bundle: ``modules.executor_registry.ExecutorBundle``
                -- the executor watchdog reads its ``snapshot()`` and
                ``thread_states()``.
            settings: LVP settings dict, passed verbatim to
                ``config_helpers.log_system_metrics``.
        """
        self._scope = scope
        self._bundle = executor_bundle
        self._settings = settings
        # The host's live display, handed in at start(); None on a host
        # with none.
        self._display: config_helpers.DisplayRates | None = None

        # Scheduler bound at start() -- None means not started.
        self._scheduler: Scheduler | None = None
        # Active schedule handles per tick (so each can be cancelled
        # independently). Keys: 'system_metrics', 'executor_watchdog',
        # 'ui_watch'.
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

    # ---- Tick implementations (also callable on-demand) ----

    def tick_system_metrics(self) -> None:
        """One snapshot of CPU/memory/handles/GC/buffer-churn metrics.

        Internal metrics scheduling -- driven by the host's periodic
        scheduler and not part of the L2 API surface; an L2 caller reads
        metrics from the log surface below rather than driving a tick.

        Delegates to ``config_helpers.log_system_metrics`` so the format
        + content match the existing log surface; engineering tools
        that grep ``[PDH METRICS]`` / ``[BUFFER METRICS]`` keep working.
        Safe to call on demand from a status endpoint.
        """
        try:
            config_helpers.log_system_metrics(
                self._settings, scope=self._scope, executors=self._bundle, display=self._display
            )
        except Exception as e:
            logger.warning(f'[MetricsLogger] tick_system_metrics failed: {type(e).__name__}: {e}')

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
        too, so it cannot be seen from here.
        """
        thread = ui_thread()
        if thread is None:
            return
        now = time.monotonic()
        with self._ui_watch_lock:
            if self._ui_probe is None:
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
        its wait is the hold's lower bound, and the line says so.
        """
        now = time.monotonic()
        with self._ui_watch_lock:
            if probe is not self._ui_probe:
                return
            self._ui_probe = None
            reported = self._ui_hold_reported
            waited_s = now - self._ui_probe_posted_at
        if reported:
            logger.info(f'[Watchdog  ] UI thread ran again; it was held at least {waited_s:.0f} s')

    # ---- Lifecycle ----

    def start(
        self,
        scheduler: Scheduler | Callable[[Callable[..., object], float], object] | None = None,
        unschedule_fn: Callable[[object], object] | None = None,
        *,
        system_metrics_interval_s: float = DEFAULT_SYSTEM_METRICS_INTERVAL_S,
        executor_watchdog_interval_s: float = DEFAULT_EXECUTOR_WATCHDOG_INTERVAL_S,
        camera_temp_interval_s: float = DEFAULT_CAMERA_TEMP_INTERVAL_S,
        start_camera_temp: bool | None = None,
        display: config_helpers.DisplayRates | None = None,
    ) -> None:
        """Schedule all periodic ticks.

        LVP-A-13: ``scheduler`` is now a :class:`modules.scheduler.Scheduler`
        instance. The legacy two-callable form (``schedule_interval_fn,
        unschedule_fn``) is auto-wrapped via ``_CallablePairScheduler``
        for backwards compatibility, so any caller that hasn't migrated
        yet keeps working unchanged.

        Args:
            scheduler: A ``Scheduler`` instance OR (legacy) a callable
                matching ``Clock.schedule_interval(func, interval_s)``.
            unschedule_fn: Legacy. Required when ``scheduler`` is a
                callable; ignored when ``scheduler`` is a ``Scheduler``.
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
        # Normalize scheduler argument: Scheduler instance, callable
        # pair (legacy), or None (rejected).
        if scheduler is None:
            raise ValueError(
                'MetricsLogger.start: scheduler is required (a Scheduler '
                'instance or, legacy, a schedule_interval callable plus '
                'unschedule_fn)'
            )
        if isinstance(scheduler, Scheduler):
            self._scheduler = scheduler
        elif callable(scheduler):
            if unschedule_fn is None or not callable(unschedule_fn):
                raise ValueError(
                    'MetricsLogger.start: legacy callable form requires '
                    'unschedule_fn (a callable matching Clock.unschedule)'
                )
            self._scheduler = _CallablePairScheduler(scheduler, unschedule_fn)
        else:
            raise TypeError(
                f'MetricsLogger.start: scheduler must be a Scheduler or '
                f'callable; got {type(scheduler).__name__}'
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
        try:
            self._scope.imaging.stop_camera_temp_logging(self._scheduler.unschedule)
        except Exception as e:
            logger.warning(f'[MetricsLogger] stop_camera_temp_logging failed: {e}')
