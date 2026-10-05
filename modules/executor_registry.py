# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-10 / LVP-A-8 -- central construction point for the LVP executor topology.

Every entry point that boots LVP (Kivy app, REST API, headless test
runner, future CLI tools) needs the same topology of SequentialIOExecutor
instances. Two of them are the scope's own, built and stopped by the
``Lumascope`` itself so a bare scope in a script serializes its commands
too; the bundle holds them beside the ones it builds:

    IO          -- the scope's: generic motor/serial work (also aliased as
                  stage, turret because all motor serial I/O goes through
                  one executor to prevent concurrent motor-board access)
    CAMERA      -- the scope's: camera-config / settings writes
                  (CAMERA_WORKER thread)
    FILE        -- file IO; a run's writes are paced by the run's own batch
    POSTPROC    -- post-processing builds (stitch, z-projection, composite,
                  video, Quick Enhance, cell count). Its own lane so a build
                  of a large folder never sits in front of a run's writes
    SCOPEDISPLAY-- display pull loop dispatcher (bare Thread, no queue)
    PROTOCOL    -- protocol orchestration (bare Thread, no queue)
    WORKER_POOL -- priority-aware executor (not a device lane: it may wait
                  on the lanes) for short-lived work that needs
                  to jump ahead of MED (abort cleanup at PRIORITY_HIGH,
                  diagnostics at PRIORITY_LOW). HIGH/MED/LOW ordering;
                  FIFO tie-break within priority. protocol_queue stays
                  FIFO regardless.

The AF lane is intentionally absent from the registry; AutofocusThread
is constructed by ScopeSession.create once the Lumascope + the
AutofocusRunner it drives are available, and lives on the session.

Until LVP-A-10 every entry point open-coded ~45 lines of construct +
start + register, with the failure mode that adding (e.g.) a new REST
shell silently forgot one executor and surfaced as a deep deferred
RuntimeError. ``create_default(io, camera, ui_dispatcher)`` returns a
single ``ExecutorBundle`` that holds every executor with the aliases
already wired and ``start()`` already called. Callers unpack the bundle
into their context object.

LVP-A-8 -- ``ExecutorBundle.snapshot()`` returns ``{name: queue_depth}``
so the App's executor watchdog (and engineering-plugin, REST status
endpoint, future health-check) can read the same view through the same
lens instead of hardcoding executor handle names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lvp_logger import logger
from modules.protocol_thread import ProtocolThread
from modules.scope_display_thread import ScopeDisplayThread
from modules.sequential_io_executor import SequentialIOExecutor


@dataclass
class ExecutorBundle:
    """Holds the executors + long-lived threads LVP needs at runtime."""

    io_executor: SequentialIOExecutor
    camera_executor: SequentialIOExecutor
    protocol_thread: ProtocolThread
    file_io_executor: SequentialIOExecutor
    post_processing_executor: SequentialIOExecutor
    scope_display_thread: ScopeDisplayThread
    worker_pool: SequentialIOExecutor
    diagnostics_executor: SequentialIOExecutor

    def snapshot(self) -> dict[str, int]:
        """Return ``{logical_name: queue_size}`` for every executor.

        Aliased executors (stage, turret) are omitted to avoid double-
        counting their queue depth. Engineering plugin / REST status
        endpoint / app watchdog all consume the same view.

        SCOPEDISPLAY and PROTOCOL are bare Threads -- no queue, so their
        slots report 0 (running) or -1 (stopped) instead of a queue depth.
        AUTOFOCUS is similarly a bare Thread and reported via
        AppContext.autofocus_thread (not in this bundle).
        WORKER_POOL is priority-aware; queue_size aggregates all
        priorities (HIGH + MED + LOW).
        """
        executors = [
            ('IO', self.io_executor),
            ('CAMERA', self.camera_executor),
            ('FILE', self.file_io_executor),
            ('POSTPROC', self.post_processing_executor),
            ('WORKER_POOL', self.worker_pool),
            ('DIAGNOSTICS', self.diagnostics_executor),
        ]
        out = {}
        for name, ex in executors:
            try:
                out[name] = ex.queue_size()
            except Exception:
                out[name] = -1
        try:
            out['SCOPEDISPLAY'] = 0 if self.scope_display_thread.is_running else -1
        except Exception:
            out['SCOPEDISPLAY'] = -1
        try:
            out['PROTOCOL'] = 0 if self.protocol_thread.is_running else -1
        except Exception:
            out['PROTOCOL'] = -1
        return out

    def shutdown(self) -> None:
        """Stop every thread this bundle started, without waiting for queued work.

        The IO and CAMERA lanes are the scope's and stop when the scope
        disconnects; the display and protocol threads that consume them stop
        here first either way.
        """
        self.scope_display_thread.stop()
        self.protocol_thread.stop(timeout=2.0)
        self.file_io_executor.shutdown(wait=False)
        self.post_processing_executor.shutdown(wait=False)
        self.worker_pool.shutdown(wait=False)
        self.diagnostics_executor.shutdown(wait=False)


def create_default(
    io_executor: SequentialIOExecutor,
    camera_executor: SequentialIOExecutor,
    ui_dispatcher: Callable[[Callable, float], Any] | None,
    ctx_provider: Callable[[], Any] | None = None,
) -> ExecutorBundle:
    """Construct + start the standard LVP executor topology around a scope's lanes.

    Args:
        io_executor: The scope's IO lane (``scope.io_lane()``), already
            started by the scope.
        camera_executor: The scope's CAMERA lane (``scope.camera_lane()``),
            already started by the scope.
        ui_dispatcher: Callable matching ``Clock.schedule_once(func, dt)``
            so executors can hand callbacks back to the GUI thread without
            importing Kivy (executors stay GUI-agnostic). Headless callers
            pass None and callbacks run inline on the worker.
        ctx_provider: The display thread's context provider -- a callable
            returning the object that carries the ``scope_display`` widget
            and the ``scope`` handle, or None while the host has neither.
            The GUI hands its app context in; a headless host has no
            display and passes nothing.

    Returns:
        ExecutorBundle holding the scope's two lanes and every executor and
        thread it built, those started. The session that holds the bundle
        tears it down in ``ScopeSession.shutdown()``.
    """
    # No run mode and no bound: a run's writes are counted and paced by the
    # run's own write batch, on this lane's one ordinary queue.
    file_io_executor = SequentialIOExecutor(name='FILE', ui_dispatcher=ui_dispatcher)
    post_processing_executor = SequentialIOExecutor(name='POSTPROC', ui_dispatcher=ui_dispatcher)
    # Thread is constructed here but NOT started. The host starts it once
    # its display widget and this thread are both reachable through the
    # provider; starting earlier races that wiring and silently no-ops.
    scope_display_thread = ScopeDisplayThread(ctx_provider=ctx_provider)
    # Protocol scan-loop driver. Generic callable runner; SCE.run()
    # submits self._run_loop_executor.run_loop and receives a Future.
    protocol_thread = ProtocolThread()
    # Not a lane: no device sits behind it. A Stop's inline cleanup and the
    # GUI's jobs run here and wait on the device lanes through the blocking
    # public members, which a lane worker is forbidden to do.
    worker_pool = SequentialIOExecutor(
        name='WORKER_POOL', ui_dispatcher=ui_dispatcher, priority_aware=True, lane=False
    )
    # Not a lane either, for a diagnostic that runs for minutes across both
    # device lanes (the support report). On the worker pool it would hold
    # the one worker, and a Stop would wait behind it; its own worker keeps
    # the pool free, and a second request queues behind the first.
    diagnostics_executor = SequentialIOExecutor(
        name='DIAGNOSTICS', ui_dispatcher=ui_dispatcher, lane=False
    )

    bundle = ExecutorBundle(
        io_executor=io_executor,
        camera_executor=camera_executor,
        protocol_thread=protocol_thread,
        file_io_executor=file_io_executor,
        post_processing_executor=post_processing_executor,
        scope_display_thread=scope_display_thread,
        worker_pool=worker_pool,
        diagnostics_executor=diagnostics_executor,
    )

    file_io_executor.start()
    post_processing_executor.start()
    worker_pool.start()
    diagnostics_executor.start()
    protocol_thread.start()

    logger.info(
        '[LVP Main  ] ExecutorRegistry: created + started FILE, POSTPROC, '
        "WORKER_POOL and DIAGNOSTICS + protocol_thread around the scope's IO and CAMERA "
        'lanes; scope_display_thread constructed (started separately from '
        'lumaviewpro.build); stage/turret aliased to IO'
    )
    return bundle
