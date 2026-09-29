# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope bring-up is session-owned: construction services the scope.

A new Lumascope needs two service registrations (executors, protocol
source path) before it behaves like the application's
scope: without executors every *_async dispatch falls back to INLINE
execution on the calling thread, losing per-lane serialization and the
protocol fence. Bring-up used to be open-coded at three sites (GUI
startup, both session factories) and absent at the fourth (reconnect);
now ScopeSession registers the services in __init__, so a scope the
session drives can never be left service-less.

The INVARIANT pins (real sim scope + recording executors) assert the
harm itself cannot recur: after construction, dispatch lands ON the
session's executor instead of running inline.

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
