# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-10 / LVP-A-8 -- central construction point for the LVP executor topology.

Every entry point that boots LVP (Kivy app, REST API, headless test
runner, future CLI tools) needs the same topology of SequentialIOExecutor
instances:

    IO          -- generic motor/serial work (also aliased as stage,
                  turret because all motor serial I/O goes through one
                  executor to prevent concurrent motor-board access)
    CAMERA      -- camera-config / settings writes (CAMERA_WORKER thread)
    FILE        -- file IO; protocol_queue bounded at 32 (F-2)
    SCOPEDISPLAY-- display pull loop dispatcher (bare Thread, no queue)
    PROTOCOL    -- protocol orchestration (bare Thread, no queue)
    WORKER_POOL -- priority-aware lane for short-lived work that needs
                  to jump ahead of MED (abort cleanup at PRIORITY_HIGH,
                  diagnostics at PRIORITY_LOW). HIGH/MED/LOW ordering;
                  FIFO tie-break within priority. protocol_queue stays
                  FIFO regardless.

The AF lane is intentionally absent from the registry; AutofocusThread
is constructed in lumaviewpro.py:build() once the Lumascope + the
AutofocusRunner it drives are available, and lives directly on
AppContext.

Until LVP-A-10 every entry point open-coded ~45 lines of construct +
start + register, with the failure mode that adding (e.g.) a new REST
shell silently forgot one executor and surfaced as a deep deferred
RuntimeError. ``ExecutorRegistry.create_default(ui_dispatcher)`` returns
a single ``ExecutorBundle`` that holds every executor with the aliases
already wired and ``start()`` already called. Callers unpack the bundle
into their context object.

LVP-A-8 -- ``ExecutorBundle.snapshot()`` returns ``{name: queue_depth}``
so the App's executor watchdog (and engineering-plugin, REST status
endpoint, future health-check) can read the same view through the same
lens instead of hardcoding executor handle names.
"""

from __future__ import annotations

from dataclasses import dataclass

from lvp_logger import logger
from modules.protocol_thread import ProtocolThread
from modules.scope_display_thread import ScopeDisplayThread
from modules.sequential_io_executor import SequentialIOExecutor


# F-2: file_io_executor's protocol_queue is bounded at 32. See
# lumaviewpro.py F-2 commit message + LVP_PERF_FINDINGS_INDEX_2026-04-30.
_FILE_IO_PROTOCOL_QUEUE_MAXSIZE = 32


@dataclass
class ExecutorBundle:
    """Holds the executors + long-lived threads LVP needs at runtime."""

    io_executor: SequentialIOExecutor
    camera_executor: SequentialIOExecutor
    protocol_thread: ProtocolThread
    file_io_executor: SequentialIOExecutor
    scope_display_thread: ScopeDisplayThread
    worker_pool: SequentialIOExecutor

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
            ('IO',          self.io_executor),
            ('CAMERA',      self.camera_executor),
            ('FILE',        self.file_io_executor),
            ('WORKER_POOL', self.worker_pool),
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


def create_default(ui_dispatcher) -> ExecutorBundle:
    """Construct + start the standard LVP executor topology.

    Args:
        ui_dispatcher: Callable matching ``Clock.schedule_once(func, dt)``
            so executors can hand callbacks back to the GUI thread without
            importing Kivy (Rule 15). Headless callers pass a direct
            dispatcher (the executor's default is fine for tests; this
            method requires an explicit dispatcher to make the lifecycle
            obvious).

    Returns:
        ExecutorBundle with every executor constructed, named, aliased,
        and started. Caller is responsible for calling ``shutdown()`` /
        ``shutdown_threads()`` at app teardown.
    """
    import modules.app_context as _app_ctx

    io_executor = SequentialIOExecutor(
        name="IO", ui_dispatcher=ui_dispatcher)
    camera_executor = SequentialIOExecutor(
        name="CAMERA", ui_dispatcher=ui_dispatcher)
    # F-2: bounded protocol_queue prevents a save thread that falls
    # behind from letting the queue grow without bound.
    file_io_executor = SequentialIOExecutor(
        name="FILE", ui_dispatcher=ui_dispatcher,
        protocol_queue_maxsize=_FILE_IO_PROTOCOL_QUEUE_MAXSIZE)
    # Thread is constructed here but NOT started. Start happens in
    # lumaviewpro.py:build() after ctx.scope_display (widget) and
    # ctx.scope_display_thread (this) are both wired into ctx;
    # starting earlier races the ctx wiring and silently no-ops.
    scope_display_thread = ScopeDisplayThread(
        ui_dispatcher=ui_dispatcher,
        ctx_provider=lambda: _app_ctx.ctx,
    )
    # Protocol scan-loop driver. Generic callable runner; SCE.run()
    # submits self._run_loop_executor.run_loop and receives a Future.
    protocol_thread = ProtocolThread(ui_dispatcher=ui_dispatcher)
    worker_pool = SequentialIOExecutor(
        name="WORKER_POOL", ui_dispatcher=ui_dispatcher,
        priority_aware=True)

    bundle = ExecutorBundle(
        io_executor=io_executor,
        camera_executor=camera_executor,
        protocol_thread=protocol_thread,
        file_io_executor=file_io_executor,
        scope_display_thread=scope_display_thread,
        worker_pool=worker_pool,
    )

    for ex in (
        io_executor, camera_executor,
        file_io_executor, worker_pool,
    ):
        ex.start()
    protocol_thread.start()

    logger.info(
        '[LVP Main  ] ExecutorRegistry: created + started '
        '4 SequentialIOExecutor instances (IO, CAMERA, FILE, '
        'WORKER_POOL) + protocol_thread + scope_display_thread '
        '(started separately from lumaviewpro.build); stage/turret '
        'aliased to IO; AutofocusThread constructed in lumaviewpro.build '
        'with the AFE handle')
    return bundle
