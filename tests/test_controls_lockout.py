# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Full-lockout contract: one derived lock, bound at the containers.

While an exclusive activity runs (protocol run OR live manual
recording), the whole control surface locks except the record/stop
toggle. The design is ONE derived App property (``controls_locked``)
that kv bindings and the gesture-motion funnel read -- never a second
per-site flag. kv is declarative source with no headless seam, so the
binding topology is pinned on the source text (established precedent);
the gesture funnel is pinned behaviorally.
"""

import pathlib
import sys
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parent.parent

APP_SRC = (REPO / 'lumaviewpro.py').read_text()
KV_SRC = (REPO / 'ui' / 'lumaviewpro.kv').read_text()


class TestDerivedLockProperty:
    def test_recording_mirror_property_exists(self):
        assert 'recording_active = BooleanProperty(False)' in APP_SRC

    def test_controls_locked_derives_from_both_activities(self):
        # The one derived lock: protocol OR recording, bound to both so
        # kv re-evaluates when either flips.
        assert 'controls_locked = AliasProperty(' in APP_SRC
        assert "bind=['protocol_running', 'recording_active']" in APP_SRC

    def test_main_display_writes_the_mirror(self):
        src = (REPO / 'ui' / 'main_display.py').read_text()
        assert 'app.recording_active = value' in src
        assert '_set_recording_active(True)' in src
        assert '_set_recording_active(False)' in src


class TestKvBindingTopology:
    def test_image_sidebar_locks_at_the_accordion(self):
        # Container-level binding: Kivy's disabled refcount dominates the
        # whole child tree. Only the IMAGE sidebar keeps the accordion-level
        # bind -- it holds no stop-capable toggle.
        image_idx = KV_SRC.find('id: accordion_id')
        assert image_idx > 0
        assert 'disabled: app.controls_locked' in KV_SRC[image_idx : image_idx + 120]

    def test_motion_accordion_carries_no_lock(self):
        # The motion sidebar holds the run/stop toggles; an accordion-level
        # bind swallows their abort clicks mid-run (a disabled ancestor eats
        # touches before children see them). The lock lives on interior
        # containers instead; the toggles escape it.
        motion_idx = KV_SRC.find('id: motionsettings_accordion_id')
        assert motion_idx > 0
        snippet = KV_SRC[motion_idx : motion_idx + 120]
        assert 'controls_locked' not in snippet, (
            'motion accordion must not lock: it strands every stop toggle'
        )

    def test_lock_moved_to_region_roots(self):
        # The non-exempt motion regions re-acquire the lock at their rule
        # or content roots so the locked surface matches the old accordion
        # bind minus exactly the stop toggles.
        for marker in ('<MicroscopeSettings>:', '<PostProcessingAccordion>:'):
            idx = KV_SRC.find(marker)
            assert idx > 0, marker
            assert 'disabled: app.controls_locked' in KV_SRC[idx : idx + 200], (
                f'{marker} must lock at its root during any exclusive activity'
            )

    def test_camera_bar_buttons_take_the_derived_lock(self):
        for btn in ('live_folder_btn', 'live_btn', 'capture_btn', 'composite_btn'):
            idx = KV_SRC.find(f'id: {btn}')
            assert idx > 0, btn
            snippet = KV_SRC[idx : idx + 120]
            assert 'disabled: app.controls_locked' in snippet, (
                f'{btn} must lock during any exclusive activity'
            )

    def test_record_button_stays_actionable_during_recording(self):
        # record_btn doubles as Stop: it locks for protocol runs only,
        # never on the derived lock (which includes recording).
        idx = KV_SRC.find('id: record_btn')
        assert idx > 0
        snippet = KV_SRC[idx : idx + 120]
        assert 'disabled: app.protocol_running' in snippet
        assert 'controls_locked' not in snippet


class TestCommittedStartRestore:
    """start() can refuse AFTER the UI commit (a rival claim held during
    a recording's file drain, an already-running race). The boundary must
    restore the pre-commit state or the lockout strands with no run live
    -- the defect class confirmed live in the sim 2026-08-10."""

    def _wire_ctx(self, event_preset=False, motion=True):
        import threading
        import types

        import modules.app_context as _app_ctx

        self._made_ctx = _app_ctx.ctx is None
        if self._made_ctx:
            _app_ctx.ctx = types.SimpleNamespace()
        ctx = _app_ctx.ctx
        self._saved = (getattr(ctx, 'protocol_running', None), getattr(ctx, 'stage', None))
        ctx.protocol_running = threading.Event()
        if event_preset:
            ctx.protocol_running.set()

        class _Stage:
            def __init__(self, enabled):
                self._enabled = enabled

            def motion_capability(self):
                return self._enabled

            def set_motion_capability(self, enabled):
                self._enabled = enabled

        ctx.stage = _Stage(motion)
        return ctx

    def _unwire(self):
        import modules.app_context as _app_ctx

        if self._made_ctx:
            _app_ctx.ctx = None
            return
        ctx = _app_ctx.ctx
        ctx.protocol_running, ctx.stage = self._saved

    def test_post_commit_refusal_restores_and_reraises(self):
        import pytest

        from modules.exceptions import ProtocolRunRefusedError
        from ui.ui_helpers import run_committed_start

        ctx = self._wire_ctx()
        try:

            def commit():
                ctx.protocol_running.set()
                ctx.stage.set_motion_capability(False)

            def start():
                raise ProtocolRunRefusedError('already_running', 't', 'm')

            with pytest.raises(ProtocolRunRefusedError):
                run_committed_start(commit, start)

            assert not ctx.protocol_running.is_set(), 'stranded Event = stranded lockout'
            assert ctx.stage.motion_capability() is True
        finally:
            self._unwire()

    def test_rival_held_event_stays_held(self):
        import pytest

        from modules.exceptions import ProtocolRunRefusedError
        from ui.ui_helpers import run_committed_start

        ctx = self._wire_ctx(event_preset=True, motion=False)
        try:

            def start():
                raise ProtocolRunRefusedError('exclusive_activity_running', 't', 'm')

            with pytest.raises(ProtocolRunRefusedError):
                run_committed_start(lambda: None, start)

            assert ctx.protocol_running.is_set(), 'restore must not clear a rival-held Event'
            assert ctx.stage.motion_capability() is False
        finally:
            self._unwire()

    def test_successful_start_leaves_commit_in_place(self):
        from ui.ui_helpers import run_committed_start

        ctx = self._wire_ctx()
        try:
            run_committed_start(lambda: ctx.protocol_running.set(), lambda: None)
            assert ctx.protocol_running.is_set()
        finally:
            self._unwire()

    def test_both_start_sites_route_through_the_boundary(self):
        src = (REPO / 'ui' / 'protocol_settings.py').read_text()
        assert src.count('run_committed_start(') >= 2, (
            'both commit sites (run_sequenced_capture + the AF-scan '
            'closure) must commit through the restoring boundary'
        )


class TestGestureMotionFunnel:
    """Bound input observers (viewer right-click, scroll-to-focus) fire
    before any widget's disabled state is consulted, so the shared
    ui_helpers movers enforce the lock themselves."""

    def _locked_app(self, locked):
        app_cls = sys.modules['kivy.app'].App
        stub = mock.Mock()
        stub.controls_locked = locked
        app_cls.get_running_app = mock.Mock(return_value=stub)
        return app_cls

    def test_locked_app_blocks_relative_move(self):
        from ui import ui_helpers

        self._locked_app(True)
        # No ctx is wired in this test process: reaching past the guard
        # would raise on ctx access, so returning cleanly IS the proof.
        ui_helpers.move_relative_position('X', 5.0)

    def test_unlocked_app_reaches_for_the_scope(self):
        from ui import ui_helpers

        self._locked_app(False)
        assert ui_helpers._user_motion_locked('X') is False

    def test_mocked_nonbool_lock_reads_unlocked(self):
        # Headless / mocked hosts (no real BooleanProperty) must never
        # read as locked: only an explicit True engages the lock.
        from ui import ui_helpers

        self._locked_app(mock.MagicMock())
        assert ui_helpers._user_motion_locked('X') is False

    def test_all_three_movers_guard(self):
        src = (REPO / 'ui' / 'ui_helpers.py').read_text()
        for mover in ('move_relative_position', 'move_absolute_position', 'move_home'):
            idx = src.find(f'def {mover}(')
            nxt = src.find('\ndef ', idx + 1)
            body = src[idx:nxt]
            assert '_user_motion_locked(' in body, f'{mover} must enforce the lock'
        # The protocol leg stays open: the protocol's own moves are not
        # user gestures.
        idx = src.find('def move_absolute_position(')
        body = src[idx : src.find('\ndef ', idx + 1)]
        assert 'if not protocol and _user_motion_locked(' in body
