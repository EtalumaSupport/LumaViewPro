# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-12 -- single owner for periodic runtime-health logging.

Collects what was previously scattered across lumaviewpro.py:

- 60 s system metrics (CPU, memory, handles, GC, buffer churn) via
  ``config_helpers.log_system_metrics``.
- 60 s executor queue depth snapshot + auto-prune of SCOPEDISPLAY
  backlog (was the ``_executor_watchdog`` closure).
- 4 hr camera temperatures via ``Lumascope.log_camera_temps``.

One module, one ``MetricsLogger`` class, three ``tick_*`` methods.
Every entry point boots LVP through the same ``start(...)`` call so
REST API + headless test runner + future CLI tools get identical
runtime-health logs without copy-paste. Engineering plugin / REST
status endpoint can also call the ``snapshot_*`` building blocks
directly to dump current state on demand without waiting for the
next tick.

Stays GUI-agnostic per Rule 15 -- takes Clock-style scheduler/
unscheduler callables instead of importing Kivy.
"""

from __future__ import annotations

from typing import Optional

from lvp_logger import logger
import modules.config_helpers as config_helpers
from modules.scheduler import Scheduler, _CallablePairScheduler


# Default cadences. system_metrics is the verbose snapshot (CPU, RAM,
# GC, page-faults, Defender, buffer-churn, frame-interval percentiles)
# costing ~10-50 ms per tick; 1-hr cadence keeps post-mortem coverage
# without measurable steady-state cost. Engineering plugin / REST
# status endpoint can call snapshot_* on demand for finer detail.
# Bench / perf-investigation runs override via start(...) kwargs.
DEFAULT_SYSTEM_METRICS_INTERVAL_S = 3600.0
DEFAULT_EXECUTOR_WATCHDOG_INTERVAL_S = 60.0
DEFAULT_CAMERA_TEMP_INTERVAL_S = 14400.0

# Backlog thresholds used by the executor-watchdog tick. Match the
# pre-LVP-A-12 values so behavior is identical.
_EXECUTOR_BACKLOG_WARN_TOTAL = 10
_SCOPE_DISPLAY_PRUNE_THRESHOLD = 20

# Frame-flow heartbeat: piggybacks on tick_system_metrics' production
# 1-hr cadence to detect silent grab failures (camera reports
# active=True + is_grabbing=True but no frames are flowing). Catches
# scenarios like Pylon SDK grab thread dead, USB transport stalled
# without formal camera removal, or buffer queue jammed. Threshold
# is set well below even 0.5 fps so legitimate slow grabs don't trip
# it; consecutive-tick guard avoids alarms during the second between
# a fresh grab-start and the first frame arriving. Alarm latency at
# the 1-hr cadence is ~2 hours -- acceptable for sustained-soak
# detection; for sub-hour interactive responsiveness, callers can
# override system_metrics_interval_s via start(...) kwargs.
_FRAME_FLOW_STALL_FPS = 0.1
_FRAME_FLOW_STALL_TICK_THRESHOLD = 2


class MetricsLogger:
    """Owns the periodic runtime-health logging surface for LVP."""

    def __init__(self, scope, executor_bundle, settings):
        """Hold references; call ``start()`` to begin the schedules.

        Args:
            scope: ``Lumascope`` API instance -- used by the camera-temp
                tick (delegates to ``scope.log_camera_temps``) and any
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

        # Scheduler bound at start() — None means not started.
        self._scheduler: Optional[Scheduler] = None
        # Active schedule handles per tick (so each can be cancelled
        # independently). Keys: 'system_metrics', 'executor_watchdog'.
        # Camera-temp lives on Lumascope already (LVP-A-2) — not stored
        # here because Lumascope owns its own handle.
        self._handles: dict[str, object] = {}

        # Consecutive ticks where the camera was reported active +
        # grabbing yet capture_fps was below _FRAME_FLOW_STALL_FPS.
        # Reset whenever fps recovers OR camera is no longer grabbing.
        # The frame-flow heartbeat fires WARNING when this exceeds
        # _FRAME_FLOW_STALL_TICK_THRESHOLD, surfacing silent grab
        # failures that don't raise an exception or trigger a timeout.
        self._frame_flow_stalled_ticks = 0
        # Sticky flag so the user-facing stall notification fires once
        # per stall episode -- set when the warning is first surfaced;
        # cleared when fps recovers above _FRAME_FLOW_STALL_FPS. Re-stall
        # after recovery re-fires the notification (persistent faults
        # must resurface; dedup-suppressed notifications hide real bugs).
        self._frame_flow_stall_notified = False

    # ---- Tick implementations (also callable on-demand) ----

    def tick_system_metrics(self) -> None:
        """One snapshot of CPU/memory/handles/GC/buffer-churn metrics.

        Delegates to ``config_helpers.log_system_metrics`` so the format
        + content match the existing log surface; engineering tools
        that grep ``[PDH METRICS]`` / ``[BUFFER METRICS]`` keep working.
        Also runs the frame-flow heartbeat on the same cadence; see
        ``_check_frame_flow_heartbeat``. Safe to call on demand from a
        status endpoint.
        """
        try:
            config_helpers.log_system_metrics(self._settings)
        except Exception as e:
            logger.warning(
                f'[MetricsLogger] tick_system_metrics failed: '
                f'{type(e).__name__}: {e}')
        # Heartbeat is best-effort and never propagates exceptions out
        # of the metrics tick (would lose all subsequent ticks).
        try:
            self._check_frame_flow_heartbeat()
        except Exception as e:
            logger.debug(
                f'[MetricsLogger] frame-flow heartbeat failed: '
                f'{type(e).__name__}: {e}')

    def _check_frame_flow_heartbeat(self) -> None:
        """Detect silent grab failure: camera active + is_grabbing()
        reports True, but capture_fps is essentially zero for multiple
        consecutive ticks. Catches scenarios where the SDK grab thread
        is alive but no frames are flowing (USB transport stalled
        without formal removal, Pylon-side grab loop hung, buffer
        queue jammed). All-zero FRAME CONTENT is detected separately
        in the char tool's data-validity guard; this catches the
        zero-frame-RATE case at the API layer.

        Resets the consecutive-stalled-ticks counter whenever the
        camera is not grabbing (so a paused live view doesn't trip
        the alarm) or fps recovers above _FRAME_FLOW_STALL_FPS.
        """
        try:
            cam = getattr(self._scope, 'camera', None)
            if cam is None or not getattr(cam, 'active', False):
                self._frame_flow_stalled_ticks = 0
                return
            if not cam.is_grabbing():
                self._frame_flow_stalled_ticks = 0
                return
        except Exception:
            self._frame_flow_stalled_ticks = 0
            return

        capture_fps = 0.0
        try:
            from modules import app_context as _app_ctx  # noqa: WPS433
            sd = _app_ctx.ctx.scope_display if _app_ctx.ctx is not None else None
            if sd is not None:
                capture_fps = float(getattr(sd, '_capture_fps_value', 0.0) or 0.0)
        except Exception:
            return

        if capture_fps >= _FRAME_FLOW_STALL_FPS:
            if self._frame_flow_stall_notified:
                logger.info(
                    f'[FRAME FLOW] capture_fps recovered to '
                    f'{capture_fps:.2f} after silent-grab stall')
            self._frame_flow_stalled_ticks = 0
            self._frame_flow_stall_notified = False
            return

        self._frame_flow_stalled_ticks += 1
        if self._frame_flow_stalled_ticks >= _FRAME_FLOW_STALL_TICK_THRESHOLD:
            logger.warning(
                f'[FRAME FLOW] capture_fps={capture_fps:.2f} below '
                f'{_FRAME_FLOW_STALL_FPS} for '
                f'{self._frame_flow_stalled_ticks} consecutive ticks while '
                f'camera reports active=True + is_grabbing=True -- possible '
                f'silent grab failure. Check camera.log for last successful '
                f'grab; investigate USB transport / Pylon SDK state.')
            # Fire user-facing notification once per stall episode so the
            # silent-stuck state is visible at the GUI, not just buried in
            # the log. Re-stall after fps recovery re-fires (persistent
            # faults must resurface).
            if not self._frame_flow_stall_notified:
                self._frame_flow_stall_notified = True
                try:
                    from modules.notification_center import notifications
                    notifications.warning(
                        "Camera",
                        "Camera frame flow stalled",
                        "Captures have not arrived for several seconds. "
                        "The camera reports active but frames are not flowing. "
                        "The protocol will continue retrying; if this persists, "
                        "restart the program."
                    )
                except Exception as _e:
                    logger.debug(
                        f'[FRAME FLOW] notification suppressed: {_e}')

    def tick_executor_watchdog(self) -> None:
        """Snapshot executor queue depths + auto-prune SCOPEDISPLAY backlog.

        WARNING-level log when total backlog exceeds 10; DEBUG otherwise.
        SCOPEDISPLAY queue >20 is pruned (UI-responsiveness guard).
        Same thresholds as the pre-LVP-A-12 inline watchdog.
        """
        try:
            snap = self._bundle.snapshot()
            total_q = sum(q for q in snap.values() if q > 0)
            fmt = ' '.join(f'{name}:{q}' for name, q in snap.items())
            if total_q > _EXECUTOR_BACKLOG_WARN_TOTAL:
                logger.warning(
                    f"[Watchdog  ] Queue backlog ({total_q} total) -- {fmt}")
            else:
                logger.debug(f"[Watchdog  ] Queues -- {fmt}")

            if snap.get('SCOPEDISPLAY', 0) > _SCOPE_DISPLAY_PRUNE_THRESHOLD:
                try:
                    self._bundle.scope_display_thread_executor.clear_pending()
                    logger.warning(
                        "[Watchdog  ] Cleared ScopeDisplay pending "
                        "queue to prevent backlog")
                except Exception:
                    pass
        except Exception:
            # Best-effort — a watchdog that crashes silently mid-tick
            # is preferable to one that takes the app down with it.
            pass

    def snapshot_executors(self) -> dict[str, int]:
        """On-demand executor depth snapshot for status endpoints.

        Returns ``{logical_name: queue_depth}`` from the bundle without
        emitting any log lines.
        """
        try:
            return self._bundle.snapshot()
        except Exception:
            return {}

    # ---- Lifecycle ----

    def start(self,
              scheduler=None,
              unschedule_fn=None,
              *,
              system_metrics_interval_s: float = DEFAULT_SYSTEM_METRICS_INTERVAL_S,
              executor_watchdog_interval_s: float = DEFAULT_EXECUTOR_WATCHDOG_INTERVAL_S,
              camera_temp_interval_s: float = DEFAULT_CAMERA_TEMP_INTERVAL_S,
              start_camera_temp: Optional[bool] = None) -> None:
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
                logger only when ``scope.camera_is_connected()``. Pass
                False to skip even when connected (rare; mostly tests).
        """
        # Normalize scheduler argument: Scheduler instance, callable
        # pair (legacy), or None (rejected).
        if scheduler is None:
            raise ValueError(
                'MetricsLogger.start: scheduler is required (a Scheduler '
                'instance or, legacy, a schedule_interval callable plus '
                'unschedule_fn)')
        if isinstance(scheduler, Scheduler):
            self._scheduler = scheduler
        elif callable(scheduler):
            if unschedule_fn is None or not callable(unschedule_fn):
                raise ValueError(
                    'MetricsLogger.start: legacy callable form requires '
                    'unschedule_fn (a callable matching Clock.unschedule)')
            self._scheduler = _CallablePairScheduler(scheduler, unschedule_fn)
        else:
            raise TypeError(
                f'MetricsLogger.start: scheduler must be a Scheduler or '
                f'callable; got {type(scheduler).__name__}')

        # Initial snapshot — match the pre-LVP-A-12 behavior of logging
        # once on startup so the very first log line carries fingerprint
        # values rather than empty cells.
        self.tick_system_metrics()

        self._handles['system_metrics'] = self._scheduler.schedule_interval(
            self.tick_system_metrics, system_metrics_interval_s)
        self._handles['executor_watchdog'] = self._scheduler.schedule_interval(
            self.tick_executor_watchdog, executor_watchdog_interval_s)

        if start_camera_temp is False:
            return
        if start_camera_temp is None and not self._scope.camera_is_connected():
            return

        # Camera-temp scheduling stays inside Lumascope (LVP-A-2) so the
        # API keeps full ownership of the event handle and self-
        # unschedules cleanly when the camera disconnects mid-run.
        # Hand it adapter callables matching the Scheduler so Lumascope
        # doesn't need to learn about Scheduler.
        self._scope.start_camera_temp_logging(
            self._scheduler.schedule_interval,
            self._scheduler.unschedule,
            interval_s=camera_temp_interval_s)

        logger.info(
            f'[MetricsLogger] started: system_metrics={system_metrics_interval_s}s, '
            f'executor_watchdog={executor_watchdog_interval_s}s, '
            f'camera_temp={camera_temp_interval_s}s')

    def stop(self) -> None:
        """Cancel every scheduled tick. Idempotent."""
        if self._scheduler is None:
            return
        for name, handle in list(self._handles.items()):
            try:
                self._scheduler.unschedule(handle)
            except Exception as e:
                logger.warning(
                    f'[MetricsLogger] unschedule {name} failed: {e}')
        self._handles.clear()
        try:
            self._scope.stop_camera_temp_logging(self._scheduler.unschedule)
        except Exception as e:
            logger.warning(
                f'[MetricsLogger] stop_camera_temp_logging failed: {e}')
