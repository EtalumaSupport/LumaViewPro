# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Central construction point for the LVP executor topology.

Every entry point that boots LVP (Kivy app, REST API, headless test
runner, future CLI tools) needs the same topology of SequentialIOExecutor
instances. Two of them are the scope's own, built and stopped by the
``Lumascope`` itself so a bare scope in a script serializes its commands
too; the bundle holds them beside the ones it builds:

    IO          -- the scope's: generic motor/serial work; all motor serial
                  I/O goes through this one executor to prevent concurrent
                  motor-board access
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
    DIAGNOSTICS -- the support report's own worker (not a device lane)

Each executor is declared once, as its ``ExecutorBundle`` field: its
logical name, and for one the bundle builds, the options it is built
with. ``create_default`` builds and starts, ``queued_executors`` lists and
``shutdown`` stops what those declarations name, so a new executor is one
field.

The AF lane is intentionally absent from the registry; AutofocusThread
is constructed by ScopeSession.create once the Lumascope + the
AutofocusRunner it drives are available, and lives on the session.

Before the registry every entry point open-coded ~45 lines of construct +
start + register, with the failure mode that adding (e.g.) a new REST
shell silently forgot one executor and surfaced as a deep deferred
RuntimeError. ``create_default(io, camera)`` returns a
single ``ExecutorBundle`` that holds every executor with ``start()``
already called on the ones it built. Callers unpack the bundle
into their context object.

``ExecutorBundle.snapshot()`` returns ``{name: queue_depth}`` and
``thread_states()`` the queueless threads' states, so the executor
watchdog and the system-metrics line read one view instead of
hardcoding executor handle names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from _lumascope.lvp_logger import logger
from _lumascope.modules.protocol_thread import ProtocolThread
from _lumascope.modules.scope_display_thread import ScopeDisplayThread
from _lumascope.modules.exceptions import SessionClosingError
from _lumascope.modules.sequential_io_executor import SequentialIOExecutor

# An executor field's metadata: its logical name, and for one the bundle
# builds, the options it is built with. A field without a name is not an
# executor.
_NAME = 'executor_name'
_BUILT_WITH = 'built_with'


def _the_scopes(name: str) -> dict:
    """An executor the scope built and stops; the bundle lists it and never starts or stops it."""
    return {_NAME: name}


def _built(name: str, **options) -> dict:
    """An executor the bundle builds with ``options``, starts and stops."""
    return {_NAME: name, _BUILT_WITH: options}


@dataclass
class ExecutorBundle:
    """Holds the executors + long-lived threads LVP needs at runtime."""

    io_executor: SequentialIOExecutor = field(metadata=_the_scopes('IO'))
    camera_executor: SequentialIOExecutor = field(metadata=_the_scopes('CAMERA'))
    protocol_thread: ProtocolThread
    # No run mode and no bound: a run's writes are counted and paced by the
    # run's own write batch, on this lane's one ordinary queue.
    file_io_executor: SequentialIOExecutor = field(metadata=_built('FILE'))
    post_processing_executor: SequentialIOExecutor = field(metadata=_built('POSTPROC'))
    scope_display_thread: ScopeDisplayThread
    # Not a lane: no device sits behind it. A Stop's inline cleanup and the
    # GUI's jobs run here and wait on the device lanes through the blocking
    # public members, which a lane worker is forbidden to do.
    worker_pool: SequentialIOExecutor = field(
        metadata=_built('WORKER_POOL', priority_aware=True, lane=False)
    )
    # Not a lane either, for a diagnostic that runs for minutes across both
    # device lanes (the support report). On the worker pool it would hold
    # the one worker, and a Stop would wait behind it; its own worker keeps
    # the pool free, and a second request queues behind the first.
    diagnostics_executor: SequentialIOExecutor = field(metadata=_built('DIAGNOSTICS', lane=False))

    def queued_executors(self) -> list[tuple[str, SequentialIOExecutor]]:
        """Every executor with a queue, by its logical name."""
        return [(f.metadata[_NAME], getattr(self, f.name)) for f in _executor_fields()]

    def snapshot(self) -> dict[str, int]:
        """Return ``{logical_name: queue_size}`` for every executor with a queue.

        Every value is a depth: the threads with no queue are
        ``thread_states``'s, and a depth that cannot be read raises rather
        than standing in as a number. WORKER_POOL is priority-aware;
        queue_size aggregates all priorities (HIGH + MED + LOW).
        """
        return {name: ex.queue_size() for name, ex in self.queued_executors()}

    def thread_states(self) -> dict[str, str]:
        """The bundle's threads with no queue, each's state in a word.

        PROTOCOL is ``running`` while a run is in flight, else ``idle``;
        SCOPEDISPLAY is ``running`` or ``off`` (never started, as on a host
        with no display, or stopped). AUTOFOCUS is a bare Thread too, held
        on the session, not in this bundle.
        """
        return {
            'SCOPEDISPLAY': 'running' if self.scope_display_thread.is_running else 'off',
            'PROTOCOL': 'running' if self.protocol_thread.is_running else 'idle',
        }

    def shutdown(self) -> None:
        """Stop every thread this bundle started, without waiting for queued work.

        The IO and CAMERA lanes are the scope's and stop when the scope
        disconnects; the display and protocol threads that consume them stop
        here first either way.
        """
        self.scope_display_thread.stop()
        self.protocol_thread.stop(timeout=2.0)
        for _name, executor in _built_executors(self):
            executor.shutdown(wait=False)


def _executor_fields():
    """The bundle's executor fields, in declaration order, chosen by their metadata."""
    return [f for f in fields(ExecutorBundle) if _NAME in f.metadata]


def _built_executors(bundle: ExecutorBundle) -> list[tuple[str, SequentialIOExecutor]]:
    """The executors ``bundle`` built, by logical name, in declaration order."""
    return [
        (f.metadata[_NAME], getattr(bundle, f.name))
        for f in _executor_fields()
        if _BUILT_WITH in f.metadata
    ]


def session_lane_closed(member: str) -> SessionClosingError:
    """What a call the session's own executors will not run is told once they have closed.

    They close only as the session closes: the work was not done because
    LumaViewPro is closing.
    """
    return SessionClosingError(None, work=member)


def create_default(
    io_executor: SequentialIOExecutor,
    camera_executor: SequentialIOExecutor,
) -> ExecutorBundle:
    """Construct + start the standard LVP executor topology around a scope's lanes.

    Args:
        io_executor: The scope's IO lane (``scope.io_lane()``), already
            started by the scope.
        camera_executor: The scope's CAMERA lane (``scope.camera_lane()``),
            already started by the scope.

    Returns:
        ExecutorBundle holding the scope's two lanes and every executor and
        thread it built, those started. The session that holds the bundle
        tears it down in ``ScopeSession.shutdown()``.
    """
    built = {
        f.name: SequentialIOExecutor(
            name=f.metadata[_NAME], closed_answer=session_lane_closed, **f.metadata[_BUILT_WITH]
        )
        for f in _executor_fields()
        if _BUILT_WITH in f.metadata
    }
    bundle = ExecutorBundle(
        io_executor=io_executor,
        camera_executor=camera_executor,
        # Protocol scan-loop driver. Generic callable runner; SCE.run()
        # submits self._run_loop_executor.run_loop and receives a Future.
        protocol_thread=ProtocolThread(),
        # Thread is constructed here but NOT started: a host with a display
        # starts it from its display widget, which it renders through.
        scope_display_thread=ScopeDisplayThread(),
        **built,
    )

    # A start that raises leaves no caller holding the bundle, so the
    # starts before it are stopped here before the raise goes on.
    try:
        for _name, executor in _built_executors(bundle):
            executor.start()
        bundle.protocol_thread.start()
    except BaseException:
        bundle.shutdown()
        raise

    built_names = [name for name, _executor in _built_executors(bundle)]
    scope_names = [f.metadata[_NAME] for f in _executor_fields() if _BUILT_WITH not in f.metadata]
    logger.info(
        f'[LVP Main  ] ExecutorRegistry: created + started {", ".join(built_names)} '
        f"+ protocol_thread around the scope's {' and '.join(scope_names)} lanes; "
        "scope_display_thread constructed (started by the host's display)"
    )
    return bundle
