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

from lvp_logger import logger
import modules.config_helpers as config_helpers
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
                -- the executor watchdog reads ``.snapshot()`` and prunes
                the SCOPEDISPLAY queue when it exceeds the threshold.
            settings: LVP settings dict, passed verbatim to
                ``config_helpers.log_system_metrics``.
        """
        self._scope = scope
        self._bundle = executor_bundle
        self._settings = settings

        # Scheduler bound at start() -- None means not started.
        self._scheduler: Scheduler | None = None
        # Active schedule handles per tick (so each can be cancelled
        # independently). Keys: 'system_metrics', 'executor_watchdog'.
        # Camera-temp lives on Lumascope already (LVP-A-2) -- not stored
        # here because Lumascope owns its own handle.
        self._handles: dict[str, object] = {}

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
            config_helpers.log_system_metrics(self._settings)
        except Exception as e:
            logger.warning(f'[MetricsLogger] tick_system_metrics failed: {type(e).__name__}: {e}')

    def tick_executor_watchdog(self) -> None:
        """Snapshot executor queue depths + auto-prune SCOPEDISPLAY backlog.

        Internal metrics scheduling -- driven by the host's periodic
        scheduler and not part of the L2 API surface.

        WARNING-level log when total backlog exceeds 10; DEBUG otherwise.
        SCOPEDISPLAY queue >20 is pruned (UI-responsiveness guard). The
        thresholds are carried unchanged from the inline watchdog this
        replaced, so log lines stay comparable across that change.
        """
        try:
            snap = self._bundle.snapshot()
            total_q = sum(q for q in snap.values() if q > 0)
            fmt = ' '.join(f'{name}:{q}' for name, q in snap.items())
            if total_q > _EXECUTOR_BACKLOG_WARN_TOTAL:
                logger.warning(f'[Watchdog  ] Queue backlog ({total_q} total) -- {fmt}')
            else:
                logger.debug(f'[Watchdog  ] Queues -- {fmt}')

            # Stage B1: SCOPEDISPLAY is a bare Thread with no queue;
            # the prune-on-backlog branch retires. The snapshot reports
            # 0 (running) / -1 (stopped). Watchdog has no actionable
            # response if the thread itself is wedged -- that's a
            # daemon=True process-exit-reap scenario, not a queue
            # prune.
        except Exception:
            # Best-effort -- a watchdog that crashes silently mid-tick
            # is preferable to one that takes the app down with it.
            pass

    def snapshot_executors(self) -> dict[str, int]:
        """On-demand executor depth snapshot for status endpoints.

        Internal queue introspection -- not part of the L2 API surface;
        the depths are host telemetry, not a contract a client reads.

        Returns ``{logical_name: queue_depth}`` from the bundle without
        emitting any log lines.
        """
        try:
            return self._bundle.snapshot()
        except Exception:
            return {}

    # ---- Lifecycle ----

    def start(
        self,
        scheduler=None,
        unschedule_fn=None,
        *,
        system_metrics_interval_s: float = DEFAULT_SYSTEM_METRICS_INTERVAL_S,
        executor_watchdog_interval_s: float = DEFAULT_EXECUTOR_WATCHDOG_INTERVAL_S,
        camera_temp_interval_s: float = DEFAULT_CAMERA_TEMP_INTERVAL_S,
        start_camera_temp: bool | None = None,
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
        """
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
