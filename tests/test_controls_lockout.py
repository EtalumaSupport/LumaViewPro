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
from types import ModuleType
from unittest import mock
from unittest.mock import MagicMock

# ui.vertical_control / ui.protocol_settings are Kivy widget modules;
# conftest mocks `kivy` but not the uix submodules, and the widget
# classes subclass BoxLayout/FloatLayout (a bare MagicMock can't be
# subclassed). Real minimal bases for those; permissive MagicMocks for
# the rest -- the same seam test_led_button_run_end_reconcile uses.


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    if name in sys.modules and not isinstance(sys.modules[name], MagicMock):
        return
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())

_real_base_module('kivy.uix.floatlayout', FloatLayout=_StubWidget)
_real_base_module('kivy.uix.boxlayout', BoxLayout=_StubWidget)
_real_base_module('kivy.uix.scrollview', ScrollView=_StubWidget)
_real_base_module('kivy.uix.widget', Widget=_StubWidget)

REPO = pathlib.Path(__file__).resolve().parent.parent

APP_SRC = (REPO / 'lumaviewpro.py').read_text()
KV_SRC = (REPO / 'ui' / 'lumaviewpro.kv').read_text()


class TestDerivedLockProperty:
    def test_recording_mirror_property_exists(self):
        assert 'recording_active = BooleanProperty(False)' in APP_SRC

    def test_controls_locked_is_listener_published(self):
        # The derivation lives on the session; the App property is a
        # plain mirror the one run-state listener writes. An
        # AliasProperty here would re-derive from the OTHER mirrors and
        # drift from session truth in the drain windows.
        assert 'controls_locked = BooleanProperty(False)' in APP_SRC
        assert 'AliasProperty' not in APP_SRC
        assert 'def publish_run_state' in APP_SRC

    def test_publish_order_is_fail_safe(self):
        # Kivy dispatches bindings synchronously inside each setattr; a
        # torn observer must see OVER-locked, never under-locked: the
        # tightening property (controls_locked) writes first on lock
        # and last on unlock.
        body = APP_SRC[APP_SRC.index('def publish_run_state') :]
        body = body[: body.index('def on_start')]
        lock_branch = body[body.index('if locked:') : body.index('else:')]
        unlock_branch = body[body.index('else:') :]
        assert lock_branch.index('controls_locked') < lock_branch.index('run_lockout'), (
            'locking must write controls_locked first'
        )
        assert unlock_branch.index('run_lockout') < unlock_branch.index(
            'controls_locked = False'
        ), 'unlocking must write controls_locked last'

    def test_main_display_republishes_the_mirror(self):
        # The live->drain flip has no claim transition of its own, so
        # the recording UI paths trigger the session republish; the
        # listener derives recording_active from the engine phase.
        src = (REPO / 'ui' / 'main_display.py').read_text()
        assert src.count('session.notify_run_state()') >= 3
        assert 'app.recording_active = value' not in src


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
        assert 'disabled: app.run_lockout' in snippet
        assert 'controls_locked' not in snippet


class TestNoCallerSideRunStateCommit:
    """Run-state truth is the session claim, committed inside start()
    and mirrored by the session run-state listener. No UI code may
    write the retired caller-side stores -- each reappearance is a
    second store whose strand / mis-restore family this migration
    retired (the refused-run wedge, the blanket motion re-enable on
    stage-less scopes, the generation-token AF lockout)."""

    FORBIDDEN = (
        'protocol_running.set()',
        'protocol_running.clear()',
        'publish_protocol_running(',
        'run_committed_start(',
        'set_motion_capability(',
    )

    def test_no_ui_file_writes_run_state(self):
        for path in [*sorted((REPO / 'ui').glob('*.py')), REPO / 'lumaviewpro.py']:
            text = path.read_text()
            for marker in self.FORBIDDEN:
                assert marker not in text, (
                    f'{path.name} contains "{marker}" -- run-state truth '
                    'lives on the session; UI code reads the derivations '
                    'and owns cosmetics only'
                )

    def test_xystage_fact_is_derived_not_written(self):
        # The UI used to push the XY fact onto the session, which made the
        # session's copy only as fresh as the last apply. motion_enabled now
        # reads the driver, so there is no write for the UI to get wrong.
        ui_src = (REPO / 'ui' / 'microscope_settings.py').read_text()
        assert 'xystage_configured' not in ui_src, (
            'the UI must not write an XY capability fact onto the session; '
            'motion_enabled derives it from the drivers'
        )
        session_src = (REPO / 'modules' / 'scope_session.py').read_text()
        assert 'capabilities.has_xy_stage' in session_src, (
            'motion_enabled must derive the XY fact from the live scope'
        )


class TestStandaloneAfLockout:
    """The standalone Autofocus button is a RUN through the sequenced-
    capture engine: the claim inside start() commits run state, the
    session listener mirrors it to kv, and the run lifecycle releases
    it. The old wrapper machinery (generation-token lockout, direct
    AF-thread dispatch, caller-side Event commit) must stay dead."""

    @staticmethod
    def _starter_def():
        import ast

        import tests.ast_seams as ast_seams

        node = ast_seams.find_def(
            'ui/vertical_control.py', 'run_autofocus_from_ui', class_name='VerticalControl'
        )
        assert node is not None, 'run_autofocus_from_ui not found'
        return ast, node

    def test_standalone_af_runs_through_the_engine(self):
        ast, node = self._starter_def()
        uses_run_mode = any(
            isinstance(sub, ast.Attribute) and sub.attr == 'SINGLE_AUTOFOCUS_SCAN'
            for sub in ast.walk(node)
        )
        assert uses_run_mode, (
            'the standalone button must start a sequenced-capture run; '
            'a direct AutofocusThread dispatch bypasses the claim, the '
            'executor fencing, and the file queue the engineering '
            'AF-data save rides on'
        )
        direct_dispatch = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == 'run_autofocus'
            for sub in ast.walk(node)
        )
        assert not direct_dispatch, (
            'no direct AutofocusThread.run_autofocus dispatch from the standalone starter'
        )

    def test_start_ordered_after_every_refusal_gate(self):
        ast, node = self._starter_def()
        first_line: dict[str, int] = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name = (
                    sub.func.attr
                    if isinstance(sub.func, ast.Attribute)
                    else getattr(sub.func, 'id', '')
                )
                if name in ('run_in_progress', 'require_file_writes_idle', 'prepare'):
                    first_line.setdefault(name, sub.lineno)
        assert (
            0
            < first_line.get('run_in_progress', 0)
            < first_line.get('require_file_writes_idle', 0)
            < first_line.get('prepare', 0)
        ), (
            'the rival-run and files-idle gates must run before the '
            'engine prepare; a cosmetics commit before a refusal shows '
            'a mid-run button for a run that never started'
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
        ui_helpers.move_relative('X', 5.0)

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
        for mover in ('move_relative', 'move_absolute', 'move_home'):
            idx = src.find(f'def {mover}(')
            nxt = src.find('\ndef ', idx + 1)
            body = src[idx:nxt]
            assert '_user_motion_locked(' in body, f'{mover} must enforce the lock'
        # The protocol leg stays open: the protocol's own moves are not
        # user gestures.
        idx = src.find('def move_absolute(')
        body = src[idx : src.find('\ndef ', idx + 1)]
        assert 'if not protocol and _user_motion_locked(' in body
