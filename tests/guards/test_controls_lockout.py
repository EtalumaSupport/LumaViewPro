# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Full-lockout contract: one lock record, bound at the containers.

While an exclusive activity runs (protocol run OR live manual
recording), the whole control surface locks except the record/stop
toggle. The Session answers the lock as one record
(``ScopeSession.controls_lock``, a ``ControlsLock``); the App holds the
last edge's copy (``app.lock``) that kv bindings and the gesture-motion
funnel read -- never a second per-site flag. kv is declarative source
with no headless seam, so the binding topology is pinned on the source
text (established precedent); the gesture funnel is pinned behaviorally.
"""

import re
import sys
from types import ModuleType
from unittest import mock
from unittest.mock import MagicMock

from tests.ast_seams import REPO_ROOT

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


APP_SRC = (REPO_ROOT / 'lumaviewpro.py').read_text()
KV_SRC = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()


class TestTheLockRecord:
    def test_the_app_publishes_no_recording_mirror(self):
        # Each run control greys through its own held flag, which the
        # Session answers for a recording live or draining; an App-level
        # recording mirror would be a second answer with no reader.
        assert 'recording_active = BooleanProperty' not in APP_SRC
        assert 'self.recording_active' not in APP_SRC, 'publish_run_state writes a dropped mirror'

    def test_the_app_holds_the_two_records_and_no_lock_flag(self):
        # The lock is the Session's one record, copied whole on each edge:
        # its two surfaces change in one write, so no binding sees one
        # without the other, and no per-flag mirror can drift from it.
        # rebind, so a binding through a field of the record re-evaluates
        # when the record is replaced (the kv test below runs it).
        assert 'lock = ObjectProperty(None, rebind=True)' in APP_SRC
        assert 'live_work = ObjectProperty(None, rebind=True)' in APP_SRC
        for flag in ('run_lockout', 'controls_locked', 'homing'):
            assert f'{flag} = BooleanProperty' not in APP_SRC, (
                f'{flag} is a mirror of the lock record'
            )
        assert 'AliasProperty' not in APP_SRC
        body = APP_SRC[APP_SRC.index('def publish_run_state') :]
        body = body[: body.index('def _draw_run_controls')]
        assert 'self.lock = session.controls_lock' in body
        assert 'self.live_work = session.live_work' in body

    def test_every_lock_binding_reads_the_record(self):
        # kv in the main file and in the advanced settings' inline rules:
        # a binding to a bare App flag would read a property nothing writes.
        kv_sources = {
            'ui/lumaviewpro.kv': KV_SRC,
            # pin-justified: the advanced settings' kv rules are a string in
            # the .py, read as the kv file is; kv has no headless seam.
            'ui/advanced_settings.py': (REPO_ROOT / 'ui' / 'advanced_settings.py').read_text(),
        }
        bare = re.compile(r'app\.(run_lockout|controls_locked|homing)\b')
        offenders = [
            f'{name}:{number}: {line.strip()}'
            for name, text in kv_sources.items()
            for number, line in enumerate(text.splitlines(), 1)
            if bare.search(line)
        ]
        assert not offenders, offenders
        bound = sum(
            len(re.findall(r'app\.lock\.(run_lockout|controls_locked)\b', text))
            for text in kv_sources.values()
        )
        assert bound == 100, f'{bound} lock bindings; each greyed surface binds the record'

    def test_main_display_supplies_no_run_state_edge(self):
        # A recording announces its own edges (live, draining, finish
        # ended) through the claim, so a headless listener hears them
        # too; a republish from the GUI would be an edge only the GUI
        # can see.
        src = (REPO_ROOT / 'ui' / 'main_display.py').read_text()
        assert 'notify_run_state' not in src
        assert 'app.recording_active = value' not in src


class TestKvBindingTopology:
    def test_image_sidebar_locks_at_the_accordion(self):
        # Container-level binding: Kivy's disabled refcount dominates the
        # whole child tree. Only the IMAGE sidebar keeps the accordion-level
        # bind -- it holds no stop-capable toggle.
        image_idx = KV_SRC.find('id: accordion_id')
        assert image_idx > 0
        assert 'disabled: app.lock.controls_locked' in KV_SRC[image_idx : image_idx + 120]

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
            assert 'disabled: app.lock.controls_locked' in KV_SRC[idx : idx + 200], (
                f'{marker} must lock at its root during any exclusive activity'
            )

    def test_camera_bar_buttons_take_the_derived_lock(self):
        # composite_btn is deliberately absent: it starts a run of its own,
        # so it follows the starter idiom (clickable during its own run, the
        # second click aborting) and is pinned with the other stop toggles in
        # test_issue_166_kv_lock_bindings.py. Re-adding it here would assert
        # the binding that strands its abort. live_btn is absent too: a
        # freeze is the display's alone, and it stays live during a run
        # (pinned in test_the_freeze_has_one_store.py).
        for btn in ('live_folder_btn', 'capture_btn'):
            idx = KV_SRC.find(f'id: {btn}')
            assert idx > 0, btn
            snippet = KV_SRC[idx : idx + 120]
            assert 'disabled: app.lock.controls_locked' in snippet, (
                f'{btn} must lock during any exclusive activity'
            )

    def test_record_button_stays_actionable_during_recording(self):
        # record_btn doubles as Stop: it locks for protocol runs only,
        # never on the derived lock (which includes recording).
        idx = KV_SRC.find('id: record_btn')
        assert idx > 0
        snippet = KV_SRC[idx : idx + 120]
        assert 'disabled: app.lock.run_lockout' in snippet
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
        for path in [*sorted((REPO_ROOT / 'ui').glob('*.py')), REPO_ROOT / 'lumaviewpro.py']:
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
        ui_src = (REPO_ROOT / 'ui' / 'microscope_settings.py').read_text()
        assert 'xystage_configured' not in ui_src, (
            'the UI must not write an XY capability fact onto the session; '
            'motion_enabled derives it from the drivers'
        )
        session_src = (REPO_ROOT / 'modules' / 'scope_session.py').read_text()
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

    def test_standalone_af_runs_through_the_member(self):
        ast, node = self._starter_def()
        called = {
            sub.func.attr
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }
        assert {'create_protocol_runner', 'start_autofocus'} <= called, (
            'the standalone button must run ProtocolRunner.start_autofocus, the '
            'one autofocus a script and REST also run; a second assembly in '
            'the widget drifts from it'
        )
        assert not called & {'prepare', 'start'}, (
            "the button builds no run of its own: prepare() and start() are the member's"
        )
        reaches_the_thread = any(
            isinstance(sub, ast.Attribute) and sub.attr == 'autofocus_thread'
            for sub in ast.walk(node)
        )
        assert not reaches_the_thread, (
            'no direct AutofocusThread dispatch from the standalone starter; it '
            'bypasses the claim, the executor fencing, and the file queue the '
            'engineering AF-data save rides on'
        )

    def test_start_ordered_after_its_own_stop(self):
        ast, node = self._starter_def()
        first_line: dict[str, int] = {}
        for sub in ast.walk(node):
            # The own-run question is the handle's `is_live`; the start is
            # the member's start_autofocus call.
            if isinstance(sub, ast.Attribute) and sub.attr == 'is_live':
                first_line['is_live'] = min(first_line.get('is_live', sub.lineno), sub.lineno)
            elif isinstance(sub, ast.Call):
                name = (
                    sub.func.attr
                    if isinstance(sub.func, ast.Attribute)
                    else getattr(sub.func, 'id', '')
                )
                if name == 'start_autofocus':
                    first_line['start_autofocus'] = min(
                        first_line.get('start_autofocus', sub.lineno), sub.lineno
                    )
        assert 0 < first_line.get('is_live', 0) < first_line.get('start_autofocus', 0), (
            'the own-run stop must be decided before the member starts the run; '
            'a start ahead of it runs an autofocus the press should have stopped. '
            f'Found: {first_line}'
        )


class TestGestureMotionFunnel:
    """Bound input observers (viewer right-click, scroll-to-focus) fire
    before any widget's disabled state is consulted, so the shared
    ui_helpers movers obey the App's lock record themselves."""

    def _locked_app(self, recording):
        from modules.scope_session import ControlsLock

        app_cls = sys.modules['kivy.app'].App
        # a stand-in by design: the App exists only when lumaviewpro.py runs
        # as the program; the funnel reads one attribute of it, the record.
        stub = mock.Mock()
        stub.lock = ControlsLock(run_named=None, recording=recording)
        app_cls.get_running_app = mock.Mock(return_value=stub)
        return app_cls

    def test_locked_app_blocks_relative_move(self):
        from ui import ui_helpers

        self._locked_app(True)
        assert ui_helpers._user_motion_locked('X') is True
        # No ctx is wired in this test process: reaching past the guard
        # would raise on ctx access, so returning cleanly IS the proof.
        ui_helpers.move_relative('X', 5.0)

    def test_unlocked_app_reaches_for_the_scope(self):
        from ui import ui_helpers

        self._locked_app(False)
        assert ui_helpers._user_motion_locked('X') is False

    def test_a_headless_host_is_never_locked(self):
        from ui import ui_helpers

        sys.modules['kivy.app'].App.get_running_app = mock.Mock(return_value=None)
        assert ui_helpers._user_motion_locked('X') is False

    def test_all_three_movers_guard(self):
        src = (REPO_ROOT / 'ui' / 'ui_helpers.py').read_text()
        for mover in ('move_relative', 'move_absolute', 'move_home', 'submit_move'):
            idx = src.find(f'def {mover}(')
            nxt = src.find('\ndef ', idx + 1)
            body = src[idx:nxt]
            assert '_user_motion_locked(' in body, f'{mover} must enforce the lock'
        # No protocol leg to leave open: a run moves the scope through its
        # own step runner, never through these gesture movers.
        idx = src.find('def move_absolute(')
        signature = src[idx : src.find(':\n', idx)]
        assert 'protocol' not in signature
