# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope bring-up is session-owned: construction services the scope.

A new Lumascope needs three service registrations (executors, executor
bundle, protocol source path) before it behaves like the application's
scope: without executors every *_async dispatch falls back to INLINE
execution on the calling thread, losing per-lane serialization and the
protocol fence. Bring-up used to be open-coded at three sites (GUI
startup, both session factories) and absent at the fourth (reconnect);
now ScopeSession registers the services in __init__, so a scope the
session drives can never be left service-less.

Two layers of pins, deliberately redundant:

- CONTRACT-SEAM pins (spec fakes) assert the registration calls with
  the session's resolved handles -- they catch a dropped or mis-wired
  call cheaply.
- INVARIANT pins (real sim scope + recording executors) assert the
  harm itself cannot recur: after construction, dispatch
  lands ON the session's executor instead of running inline. A
  regression that keeps the call but breaks the wiring ships green
  through the seam pins and red here.

Also pinned: shutdown()
ownership (owns_executors, not bundle-presence, decides teardown
scope).
"""

from unittest.mock import MagicMock

import pytest

from tests.settings_fixtures import complete_settings

from modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope, spec_scope
from tests.test_scope_api import _RecordingExecutor


def _make_spec_session(**kwargs):
    defaults = {
        'settings': {},
        'scope': spec_scope(),
        'io_executor': MagicMock(),
        'camera_executor': MagicMock(),
    }
    defaults.update(kwargs)
    return ScopeSession(**defaults)


# Real executors run real worker threads and real sim scopes hold
# simulated drivers; every rig built below is torn down here or the
# threads/drivers outlive the test that made them.
_LIVE = []


@pytest.fixture(autouse=True)
def _teardown_live_rigs():
    yield
    while _LIVE:
        kind, obj = _LIVE.pop()
        try:
            if kind == 'executor':
                obj.shutdown()
            else:
                obj.disconnect()
        except Exception:
            pass


def _real_scope():
    scope = build_scope(simulate=True)
    _LIVE.append(('scope', scope))
    return scope


def _real_executor(name):
    ex = _RecordingExecutor(name=name)
    ex.start()
    _LIVE.append(('executor', ex))
    return ex


# ===========================================================================
# Invariant pins: dispatch lands on the executor, never inline
# ===========================================================================


class TestDispatchInvariant:
    def test_construction_services_the_scope_for_dispatch(self):
        io_ex = _real_executor('BRINGUP_IO2')
        cam_ex = _real_executor('BRINGUP_CAM2')
        scope = _real_scope()  # bare: nothing pre-registered
        ScopeSession(
            settings={},
            scope=scope,
            io_executor=io_ex,
            camera_executor=cam_ex,
        )

        scope.motion.home('Z')
        assert io_ex.submitted, (
            'a session-composed scope must dispatch through the session '
            'executors from construction on; inline execution here means '
            '__init__ did not service the scope'
        )


# ===========================================================================
# Contract-seam pins: the registration calls, with resolved handles
# ===========================================================================


class TestConstructionServicesContract:
    def test_direct_construction_registers_trio(self):
        # DIRECT construction, never a factory: the factories were
        # green-before (they open-coded the trio) and would void this
        # pin's fail-before.
        io, cam = MagicMock(), MagicMock()
        scope = spec_scope()
        ScopeSession(
            settings={},
            scope=scope,
            io_executor=io,
            camera_executor=cam,
            source_path='/somewhere',
        )
        scope.register_executors.assert_called_once_with(
            camera_executor=cam,
            io_executor=io,
            file_io_executor=None,
            camera_override_key=cam.ask_claim.return_value,
        )
        scope.protocols.register_source_path.assert_called_once_with('/somewhere')
        scope.register_executor_bundle.assert_not_called()

    def test_construction_registers_resolved_file_io_from_bundle(self):
        # The trio must register the RESOLVED file-io handle (derived
        # from the bundle when no explicit one is passed), or the
        # bundle-building factory path silently degrades protocol
        # file-IO to inline execution.
        bundle = MagicMock()
        io, cam = MagicMock(), MagicMock()
        scope = spec_scope()
        ScopeSession(
            settings={},
            scope=scope,
            io_executor=io,
            camera_executor=cam,
            executor_bundle=bundle,
        )
        scope.register_executors.assert_called_once_with(
            camera_executor=cam,
            io_executor=io,
            file_io_executor=bundle.file_io_executor,
            camera_override_key=cam.ask_claim.return_value,
        )
        scope.register_executor_bundle.assert_called_once_with(bundle, settings={})


# ===========================================================================
# shutdown(): ownership is the explicit fact, not bundle-presence
# ===========================================================================


class TestShutdownOwnership:
    def test_host_injected_session_shutdown_leaves_the_bundle_alone(self):
        # A host (the GUI) passes its bundle so the session can service
        # scopes -- that must NOT hand the session teardown rights over
        # the host's executor topology. The False path keeps today's
        # documented contract: it still stops the handles the caller
        # passed in (io, camera, AF thread) and nothing else.
        bundle = MagicMock()
        af = MagicMock()
        session = _make_spec_session(
            executor_bundle=bundle,
            autofocus_thread=af,
        )
        session.shutdown()

        bundle.scope_display_thread.stop.assert_not_called()
        bundle.protocol_thread.stop.assert_not_called()
        bundle.io_executor.shutdown.assert_not_called()
        bundle.camera_executor.shutdown.assert_not_called()
        bundle.file_io_executor.shutdown.assert_not_called()
        bundle.worker_pool.shutdown.assert_not_called()

        session.io_executor.shutdown.assert_called_once()
        session.camera_executor.shutdown.assert_called_once()
        af.stop.assert_called_once()

    def test_factory_session_shutdown_still_tears_down_its_bundle(self):
        # Green before the owns_executors change BY DESIGN: this pins
        # the True side -- a factory that forgot owns_executors=True
        # (or a flipped default) would leak the bundle's threads on
        # every headless session, and nothing else in the suite
        # asserts this teardown.
        session = ScopeSession.create(complete_settings(), simulate=True)
        bundle = session.executor_bundle
        session.shutdown()

        # The scope is the factory's too: the same call disconnects it.
        assert session.scope.imaging.is_streaming() is False
        assert session.scope.motor_connected is False
        for wrapper in (bundle.protocol_thread, bundle.scope_display_thread):
            thread = wrapper._thread
            assert thread is None or not thread.is_alive(), (
                f'{type(wrapper).__name__} still running after shutdown()'
            )
        for executor in (
            bundle.io_executor,
            bundle.camera_executor,
            bundle.file_io_executor,
        ):
            worker = executor._worker_thread
            if worker is not None:
                worker.join(timeout=5.0)
                assert not worker.is_alive(), (
                    f'{executor.executor_name} worker still running after shutdown()'
                )
        session.scope.disconnect()
