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
    def test_both_sidebars_lock_at_the_accordion(self):
        # Container-level binding: Kivy's disabled refcount dominates the
        # whole child tree, so two bindings cover both sidebars without
        # fighting the imperative per-widget writers.
        motion_idx = KV_SRC.find('id: motionsettings_accordion_id')
        assert motion_idx > 0
        assert 'disabled: app.controls_locked' in KV_SRC[motion_idx : motion_idx + 120]
        image_idx = KV_SRC.find('id: accordion_id')
        assert image_idx > 0
        assert 'disabled: app.controls_locked' in KV_SRC[image_idx : image_idx + 120]

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
