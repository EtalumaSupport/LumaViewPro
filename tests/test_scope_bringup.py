# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope bring-up is session-owned: construction services the scope.

A scope builds its own IO and CAMERA lanes, so it dispatches on them
whoever holds it; the session registers the rest of its services (the
camera override key, the protocol source path) in __init__, so a scope
the session drives can never be left service-less.

The INVARIANT pins (real sim scope + recording lanes) assert that a
session composed around a scope leaves its dispatch on the scope's lane.

Also pinned: shutdown() ownership (owns_scope decides the hardware
teardown; the bundle the session holds always stops).
"""

from unittest.mock import MagicMock

import pytest

from tests.settings_fixtures import complete_settings

from modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope, spec_scope, swap_lanes
from tests.test_scope_api import _RecordingExecutor


def _make_spec_session(**kwargs):
    defaults = {
        'settings': {},
        'scope': spec_scope(),
        'executor_bundle': MagicMock(),
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
        scope = swap_lanes(_real_scope(), io=io_ex, camera=cam_ex)
        ScopeSession(settings={}, scope=scope, executor_bundle=MagicMock())

        scope.motion.home('Z')
        assert io_ex.submitted, (
            "a session-composed scope must dispatch through the scope's own "
            'lane; the session composing around it must not reroute it'
        )


# ===========================================================================
# shutdown(): ownership is the explicit fact, not bundle-presence
# ===========================================================================


class TestShutdownOwnership:
    def test_a_session_stops_the_bundle_it_holds_and_leaves_a_callers_scope(self):
        # The bundle is the session's whoever built it: a caller that hands
        # one to the constructor hands it over. The scope's lanes are the
        # scope's, and a scope passed in is the caller's, so they run on.
        bundle = MagicMock()
        af = MagicMock()
        session = _make_spec_session(
            executor_bundle=bundle,
            autofocus_thread=af,
        )
        session.shutdown()

        bundle.shutdown.assert_called_once()
        af.stop.assert_called_once()
        session.scope.io_lane.return_value.shutdown.assert_not_called()
        session.scope.camera_lane.return_value.shutdown.assert_not_called()
        session.scope.disconnect.assert_not_called()

    def test_factory_session_shutdown_still_tears_down_its_bundle(self):
        # A factory-built session stops everything it built: the bundle's
        # threads and, through the scope's disconnect, the scope's lanes.
        # Nothing else in the suite asserts this teardown.
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
