# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-10 / LVP-A-8 -- central construction point for the LVP executor topology.

Every entry point that boots LVP (Kivy app, REST API, headless test
runner, future CLI tools) needs the same topology of SequentialIOExecutor
instances:

    IO          -- generic motor/serial work (also aliased as stage,
                  turret because all motor serial I/O goes through one
                  executor to prevent concurrent motor-board access)
    CAMERA      -- camera-config / settings writes (CAMERA_WORKER thread)
    PROTOCOL    -- protocol orchestration (long-running runs)
    FILE        -- file IO; protocol_queue bounded at 32 (F-2)
    AUTOFOCUS   -- autofocus measurement loop
    SCOPEDISPLAY-- display pull loop dispatcher
    RESET       -- emergency stop / reconnect operations

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
from modules.sequential_io_executor import SequentialIOExecutor


# F-2: file_io_executor's protocol_queue is bounded at 32. See
# lumaviewpro.py F-2 commit message + LVP_PERF_FINDINGS_INDEX_2026-04-30.
_FILE_IO_PROTOCOL_QUEUE_MAXSIZE = 32


@dataclass
class ExecutorBundle:
    """Holds every SequentialIOExecutor LVP needs at runtime.

    ``stage_executor`` and ``turret_executor`` are the same object as
    ``io_executor`` (all motor serial I/O serializes through one
    executor). They are kept as named fields so callers don't have to
    know about the aliasing.
    """

    io_executor: SequentialIOExecutor
    camera_executor: SequentialIOExecutor
    protocol_executor: SequentialIOExecutor
    file_io_executor: SequentialIOExecutor
    autofocus_thread_executor: SequentialIOExecutor
    scope_display_thread_executor: SequentialIOExecutor
    reset_executor: SequentialIOExecutor
    stage_executor: SequentialIOExecutor    # alias for io_executor
    turret_executor: SequentialIOExecutor   # alias for io_executor

    def snapshot(self) -> dict[str, int]:
        """LVP-A-8: return ``{logical_name: queue_size}`` for every executor.

        Aliased executors (stage, turret) are omitted to avoid double-
        counting their queue depth. Engineering plugin / REST status
        endpoint / app watchdog all consume the same view.
        """
        executors = [
            ('IO',          self.io_executor),
            ('CAMERA',      self.camera_executor),
            ('PROTOCOL',    self.protocol_executor),
            ('FILE',        self.file_io_executor),
            ('AUTOFOCUS',   self.autofocus_thread_executor),
            ('SCOPEDISPLAY', self.scope_display_thread_executor),
            ('RESET',       self.reset_executor),
        ]
        out = {}
        for name, ex in executors:
            try:
                out[name] = ex.queue_size()
            except Exception:
                out[name] = -1
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
    io_executor = SequentialIOExecutor(
        name="IO", ui_dispatcher=ui_dispatcher)
    camera_executor = SequentialIOExecutor(
        name="CAMERA", ui_dispatcher=ui_dispatcher)
    protocol_executor = SequentialIOExecutor(
        name="PROTOCOL", ui_dispatcher=ui_dispatcher)
    # F-2: bounded protocol_queue prevents a save thread that falls
    # behind from letting the queue grow without bound.
    file_io_executor = SequentialIOExecutor(
        name="FILE", ui_dispatcher=ui_dispatcher,
        protocol_queue_maxsize=_FILE_IO_PROTOCOL_QUEUE_MAXSIZE)
    autofocus_thread_executor = SequentialIOExecutor(
        name="AUTOFOCUS", ui_dispatcher=ui_dispatcher)
    scope_display_thread_executor = SequentialIOExecutor(
        name="SCOPEDISPLAY", ui_dispatcher=ui_dispatcher)
    reset_executor = SequentialIOExecutor(
        name="RESET", ui_dispatcher=ui_dispatcher)

    bundle = ExecutorBundle(
        io_executor=io_executor,
        camera_executor=camera_executor,
        protocol_executor=protocol_executor,
        file_io_executor=file_io_executor,
        autofocus_thread_executor=autofocus_thread_executor,
        scope_display_thread_executor=scope_display_thread_executor,
        reset_executor=reset_executor,
        # Aliases — same object, named so callers don't have to know.
        stage_executor=io_executor,
        turret_executor=io_executor,
    )

    for ex in (
        io_executor, camera_executor, protocol_executor,
        file_io_executor, autofocus_thread_executor,
        scope_display_thread_executor, reset_executor,
    ):
        ex.start()

    logger.info(
        '[LVP Main  ] ExecutorRegistry: created + started '
        '7 executors (IO, CAMERA, PROTOCOL, FILE, AUTOFOCUS, '
        'SCOPEDISPLAY, RESET); stage/turret aliased to IO')
    return bundle
