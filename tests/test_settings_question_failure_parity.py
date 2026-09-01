# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The unanswerable settings question, and failure parity for the write.

An unreadable current.json used to be raised with the user inside
``build()`` -- before Kivy attaches the root widget, so the dialog was
created, opened, logged, and then painted UNDER the app root with every
touch routed to the root. Never rendered, never answerable. The user
therefore never resolved the provisional state, so ``save_settings``
refused for the rest of the session -- silently, returning ``None`` --
and the whole session's changes were lost at exit with nothing said.

Two arms fix that, and each covers the other:

* the question is asked from ``on_start`` and Clock-deferred inside the
  helper, so it is posed at the one moment an answer can be given, and
  its failure-to-present stays fatal at the new call site;
* the two refusals in ``save_settings`` RAISE
  ``SettingsSaveRefusedError`` with a machine-readable reason, so a
  session that somehow still never sees the dialog fails loudly on every
  save instead of losing the user's work quietly -- and a REST caller
  gets the FAIL rather than a success on a write that never happened.

Around them sit the observability facts this defect's own evidence chain
needed: a popup opened before the mainloop is marked and logged at
ERROR, popup/notification records stay one physical line each, and the
debug banner stops naming a settings file it could not read.

Two seams are worth naming, because a reader will otherwise assume more
than these tests deliver:

RENDERING is not tested. Whether a popup is visible is a canvas/z-order
fact of a real Window and is not reachable headlessly. What IS tested is
CALL ORDER and deferral shape -- which is where the defect actually
lived.

``lumaviewpro.py`` is the Kivy entry point and is NOT importable under
the test harness (its module globals -- ``ctx``, ``logger``, ``Clock``
-- are bound inside ``if __name__ == '__main__'``). The alternative to
compiling a method node out of its AST is a source-substring scan, which
cannot run the code at all. ``_app_method`` below compiles the REAL
FunctionDef from the file and supplies only the module globals it closes
over, so the production body runs verbatim.
"""

import ast
import importlib
import json
import logging
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import modules.notification_center as notification_center
import modules.settings_init as settings_init
import ui.notification_popup as notification_popup
from modules import gui_logger
from modules.exceptions import SettingsSaveRefusedError
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.ast_seams import REPO_ROOT, parse_module
from tests.test_objective_prompt_single_flight import _direct_call_names
from ui.vertical_control import VerticalControl

SHIPPED_TEMPLATE = REPO_ROOT / 'data' / 'settings.json'

CATALOGUE = ['4x Oly', '10x Oly', '20x Oly', '40x Oly']

# The dialog whose own message is the worst offender in the multiline
# cluster: seven physical lines, written into a line-oriented record.
SEVEN_LINE_MESSAGE = (
    'C:/Users/op/Documents/LumaViewPro/data/current.json\n\n'
    'not valid JSON (Expecting value: line 1 column 1)\n\n'
    'LumaViewPro is running on default settings. Your file has not been '
    'changed, and nothing will be saved until you choose.\n\n'
    'Start over with defaults, and keep the old file alongside it for '
    'support? Or quit now so the file can be repaired?'
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _app_class() -> ast.ClassDef:
    tree = parse_module('lumaviewpro.py')
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'LumaViewProApp':
            return node
    raise AssertionError('lumaviewpro.py: class LumaViewProApp is gone')


def _app_methods() -> dict:
    """The class's OWN methods, by name (nested closures excluded)."""
    return {n.name: n for n in _app_class().body if isinstance(n, ast.FunctionDef)}


def _mixin_methods() -> dict:
    """TooltipMixin's methods -- build() calls into the mixin, not only self."""
    tree = parse_module('ui/tooltip.py')
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'TooltipMixin':
            return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
    return {}


def _self_call_names(fn) -> list[str]:
    """``self.<name>()`` called in this function's OWN body.

    Modelled on _direct_call_names, but attribute-qualified: the plain
    walker cannot tell ``self.stop()`` from ``popup.stop()``, and a
    reachability walk that followed every attribute name would resolve
    unrelated objects' methods to LumaViewProApp's.
    """
    names = []
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'self'
        ):
            names.append(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _calls_self(node, method: str) -> bool:
    """Does anything under ``node`` call ``self.<method>()``?"""
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == method
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == 'self'
        for n in ast.walk(node)
    )


def _app_method(name: str, **module_globals):
    """Compile one REAL method out of lumaviewpro.py and return it.

    See the module docstring for why this exists rather than an import.
    Only the module globals the body closes over are supplied by the
    caller; the body itself is the file's own bytes.
    """
    node = _app_methods().get(name)
    assert node is not None, f'lumaviewpro.py: LumaViewProApp.{name} is gone'
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(module_globals)
    exec(compile(module, str(REPO_ROOT / 'lumaviewpro.py'), 'exec'), namespace)
    return namespace[name]


# ---------------------------------------------------------------------------
# Runtime stands
# ---------------------------------------------------------------------------


class _FakeClock:
    """Records deferrals instead of running them, so a test owns the frame."""

    def __init__(self):
        self.queue = []

    def schedule_once(self, callback, timeout=0):
        self.queue.append(callback)
        return object()

    def run_all(self, order=None):
        callbacks = list(self.queue)
        self.queue.clear()
        if order is not None:
            callbacks = [callbacks[i] for i in order]
        for callback in callbacks:
            callback(0)


class _VerticalControlStand:
    """The real prompt methods; only the widget tree is stood in for."""

    prompt_objective_selection = VerticalControl.prompt_objective_selection
    maybe_prompt_objective_selection = VerticalControl.maybe_prompt_objective_selection

    def __init__(self):
        self.ids = {'objective_spinner2': SimpleNamespace(values=list(CATALOGUE), text='')}

    def load_objectives(self):
        pass

    def update_all_turret_btn_states(self, position):
        pass

    def set_turret_objective(self):
        pass


class _AppStand:
    """The two collaborators the settings question calls back into."""

    def __init__(self):
        self.re_asked = 0
        self.objective_prompts = 0
        self.stopped = 0

    def _ask_about_rejected_settings(self):
        self.re_asked += 1

    def _prompt_objective_if_needed(self):
        self.objective_prompts += 1

    def stop(self):
        self.stopped += 1


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured_logs():
    """Attach a handler directly to named loggers and hand back the records.

    Not caplog: these loggers are configured with propagate=False in a
    real run, so a root-level capture would see nothing there even
    though it happens to work under the harness.
    """
    attached = []

    def _capture(name):
        log = logging.getLogger(name)
        handler = _CaptureHandler()
        attached.append((log, handler, log.level))
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        return handler.records

    yield _capture

    for log, handler, level in attached:
        log.removeHandler(handler)
        log.setLevel(level)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A headless session over a private data directory."""
    data = tmp_path / 'data'
    data.mkdir()
    shutil.copy(SHIPPED_TEMPLATE, data / 'settings.json')
    shutil.copy(SHIPPED_TEMPLATE, data / 'current.json')
    monkeypatch.setattr(settings_init, 'settings', None)
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    return ScopeSession.create_headless(source_path=str(tmp_path))


def _current_json(tmp_path) -> str:
    """The ONE place this file reads the user's configuration off disk.

    pin-justified: every refusal test makes the same claim -- "the user's
    only copy was left exactly as it was" -- which is a fact about a file
    on disk, so there is no AST seam to assert instead. (The ratchet
    already records that rationale for tests/test_session_save_settings.py,
    whose assertions these mirror.) Funnelled through one helper so the
    whole file carries a single read site rather than a pair per test.
    """
    return (tmp_path / 'data' / 'current.json').read_text()


@pytest.fixture
def untouched(session, tmp_path):
    """Snapshot current.json; the returned check asserts it never moved.

    Depends on ``session`` so the snapshot is taken after that fixture
    has laid the file down, whatever order a test names them in.
    """
    before = _current_json(tmp_path)

    def _check():
        assert _current_json(tmp_path) == before, (
            "the refused save must leave the user's only copy exactly as it was"
        )

    return _check


def _with_hardware(session, monkeypatch):
    monkeypatch.setattr(
        session,
        'scope',
        SimpleNamespace(camera_connected=True, motor_connected=True, led_connected=True),
    )


def _without_hardware(session, monkeypatch):
    monkeypatch.setattr(
        session,
        'scope',
        SimpleNamespace(camera_connected=False, motor_connected=False, led_connected=False),
    )


def _make_provisional(monkeypatch, tmp_path):
    monkeypatch.setattr(
        settings_init,
        'rejected_current_json',
        (str(tmp_path / 'data' / 'current.json'), 'not valid JSON'),
    )


# ---------------------------------------------------------------------------
# E1 -- the question is posed where an answer can be given
# ---------------------------------------------------------------------------


class TestTheQuestionIsAskable:
    def test_the_settings_question_is_not_asked_from_build(self):
        """build() returns BEFORE Kivy attaches the root widget, so no
        position inside it is late enough for a dialog to be seen. The
        ask belongs to on_start, and the deferral to the helper."""
        build = _app_methods().get('build')
        assert build is not None, 'lumaviewpro.py: LumaViewProApp.build is gone'

        assert '_ask_about_rejected_settings' not in _direct_call_names(build)

    def test_the_question_is_deferred_and_its_failure_stays_fatal(self):
        """Deferral alone would drop the fatal property: after the move,
        an exception opening the dialog escapes into Kivy's exception
        manager instead of stopping the app. The try/except has to
        travel WITH the open, into the Clock callback."""
        fn = _app_methods()['_ask_about_rejected_settings']

        nested = {
            node.name: node
            for node in ast.walk(fn)
            if isinstance(node, ast.FunctionDef) and node is not fn
        }
        deferred = [
            nested[node.args[0].id]
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'schedule_once'
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'Clock'
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in nested
        ]
        assert deferred, (
            '_ask_about_rejected_settings must Clock.schedule_once a callback '
            'defined in its own body -- an undeferred open is painted under the root'
        )

        guarded = [
            node
            for callback in deferred
            for node in ast.walk(callback)
            if isinstance(node, ast.Try) and any(_calls_self(h, 'stop') for h in node.handlers)
        ]
        assert guarded, (
            'the deferred callback must keep the try/except whose handler calls '
            'self.stop() -- failing to present the question is fatal'
        )

        opens = [
            node
            for try_node in guarded
            for statement in try_node.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'show_confirmation_popup'
        ]
        assert opens, 'the confirm dialog must open INSIDE the fatal-guarded try'

    def test_no_popup_is_reachable_from_build(self):
        """The class-level arm: not just this one call, but any popup
        opened from build()'s own body or from a method build() calls
        directly. A deferred open inside a closure is fine and is
        excluded -- that is the fix, not the defect."""
        methods = _app_methods()
        resolvable = dict(_mixin_methods())
        resolvable.update(methods)

        build = methods['build']
        walked = {'build': build}
        for name in _self_call_names(build):
            if name in resolvable:
                walked[name] = resolvable[name]

        # Non-vacuity: build() really does call into a resolvable method,
        # so a refactor that renamed everything would not silently pass.
        assert len(walked) > 1, 'the transitive walk resolved nothing -- the pin is vacuous'

        offenders = [
            (owner, call)
            for owner, fn in walked.items()
            for call in _direct_call_names(fn)
            if call.startswith('show_') and call.endswith('_popup')
        ]
        assert offenders == [], f'popup(s) opened before the root attaches: {offenders}'

    def test_the_answer_path_survives_a_locked_file(self, monkeypatch, tmp_path):
        """The retire is a file rename. Under a Windows AV/indexer lock
        it raises -- from a button callback, where an escape kills the
        process with no teardown. It must say so, re-present the
        question, and NOT go on to the objective prompt: settings are
        still provisional, so that answer still could not be kept."""
        _make_provisional(monkeypatch, tmp_path)

        def _locked():
            raise PermissionError('current.json is in use by another program')

        ctx = SimpleNamespace(
            session=SimpleNamespace(
                settings_are_provisional=lambda: True,
                retire_rejected_settings=_locked,
            )
        )
        clock = _FakeClock()
        log = MagicMock()
        stand = _AppStand()

        errors = []
        monkeypatch.setattr(
            notification_center.notifications,
            'error',
            lambda category, title, message, **kw: errors.append((category, title, message)),
        )

        shown = {}
        monkeypatch.setattr(
            notification_popup,
            'show_confirmation_popup',
            lambda **kwargs: shown.update(kwargs),
        )

        ask = _app_method('_ask_about_rejected_settings', ctx=ctx, logger=log, Clock=clock)
        ask(stand)

        # Nothing opened yet -- the whole point of the deferral.
        assert shown == {}
        clock.run_all()
        assert shown['confirm_text'] == 'Use defaults'

        # The user answers "Use defaults" and the rename fails.
        shown['on_confirm']()

        assert stand.stopped == 0, 'a locked file is recoverable; it must not stop the app'
        assert log.error.called
        assert errors and errors[0][1] == 'Settings file could not be replaced'
        assert stand.re_asked == 1, 'the unresolved question must be put back to the user'
        assert stand.objective_prompts == 0, (
            'settings are still provisional, so the objective answer still could not be kept'
        )


# ---------------------------------------------------------------------------
# E2 -- failure parity at the API boundary
# ---------------------------------------------------------------------------


class TestTheSaveRefusalIsAudible:
    def test_session_exposes_the_provisional_query_and_remedy(self, session, monkeypatch, tmp_path):
        """The GUI must not reach around the Session into settings_init:
        an L2 caller needs the same query and the same remedy."""
        assert session.settings_are_provisional() is False

        _make_provisional(monkeypatch, tmp_path)
        assert session.settings_are_provisional() is True

        calls = []

        def _retire():
            calls.append(True)
            return '/data/current.json.rejected-20260831-120000'

        monkeypatch.setattr(settings_init, 'retire_rejected_current_json', _retire)

        assert session.retire_rejected_settings() == '/data/current.json.rejected-20260831-120000'
        assert calls == [True]

    def test_a_provisional_save_raises(self, session, monkeypatch, tmp_path, untouched):
        """Returning None here is how a whole session's changes were lost
        with nothing said -- and how a REST PUT reported success on a
        write that never happened."""
        _with_hardware(session, monkeypatch)
        _make_provisional(monkeypatch, tmp_path)

        session.settings['live_folder'] = '/data/template_values'
        with pytest.raises(SettingsSaveRefusedError) as excinfo:
            session.save_settings()

        assert excinfo.value.reason == 'settings_provisional'
        assert excinfo.value.file.endswith('current.json')
        untouched()

    def test_force_does_not_override_the_provisional_refusal(
        self, session, monkeypatch, tmp_path, untouched
    ):
        """force overrides "the sliders are at defaults", not "this file
        is the user's only copy and we could not read it"."""
        _with_hardware(session, monkeypatch)
        _make_provisional(monkeypatch, tmp_path)

        with pytest.raises(SettingsSaveRefusedError) as excinfo:
            session.save_settings(force=True)

        assert excinfo.value.reason == 'settings_provisional'
        untouched()

    def test_a_provisional_save_to_another_destination_still_writes(
        self, session, monkeypatch, tmp_path
    ):
        """The refusal is conjoined with the destination. A caller
        exporting a configuration elsewhere is not writing the user's
        live file and must not be blocked."""
        _with_hardware(session, monkeypatch)
        _make_provisional(monkeypatch, tmp_path)
        elsewhere = tmp_path / 'exported.json'

        session.settings['live_folder'] = '/data/exported'
        session.save_settings(str(elsewhere))

        with open(elsewhere) as f:
            assert json.load(f)['live_folder'] == '/data/exported'

    def test_a_hardware_less_save_raises(self, session, monkeypatch, tmp_path, untouched):
        """The more common silent no-op for an API caller: with nothing
        attached the per-channel values in memory are slider defaults."""
        _without_hardware(session, monkeypatch)

        session.settings['live_folder'] = '/data/should_not_persist'
        with pytest.raises(SettingsSaveRefusedError) as excinfo:
            session.save_settings()

        assert excinfo.value.reason == 'no_hardware'
        untouched()

        # force is the deliberate-write escape hatch, and still works.
        session.save_settings(force=True)
        with open(tmp_path / 'data' / 'current.json') as f:
            assert json.load(f)['live_folder'] == '/data/should_not_persist'

    def test_shutdown_survives_a_refused_save(self, session, monkeypatch, tmp_path):
        """Two halves, because on_stop cannot be driven headlessly: it
        tears down executors, threads, notification listeners and real
        hardware. The BEHAVIOUR half proves the refusal is an exception
        that would abort an unguarded caller; the STRUCTURAL half proves
        on_stop catches exactly that type and still reaches the hardware
        teardown after it."""
        _without_hardware(session, monkeypatch)
        _make_provisional(monkeypatch, tmp_path)

        # Provisional is checked first, so it is what shutdown meets on a
        # hardware-less session over an unreadable file -- the bench case.
        with pytest.raises(SettingsSaveRefusedError) as excinfo:
            session.save_settings('./data/current.json')
        assert excinfo.value.reason == 'settings_provisional'

        on_stop = _app_methods()['on_stop']
        guards = [
            node
            for node in ast.walk(on_stop)
            if isinstance(node, ast.Try)
            and any(
                h.type is not None and ast.unparse(h.type) == 'SettingsSaveRefusedError'
                for h in node.handlers
            )
        ]
        assert len(guards) == 1, 'on_stop must catch the refusal around its save'
        guard = guards[0]
        assert 'save_settings' in [
            n.func.attr
            for n in ast.walk(guard)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]

        teardown = [
            node.lineno
            for node in ast.walk(on_stop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'disconnect'
        ]
        assert teardown, 'on_stop must still tear the hardware down'
        assert max(teardown) > guard.end_lineno, (
            'the hardware teardown must sit AFTER the guarded save, not inside it'
        )

    def test_the_periodic_flush_survives_a_refused_save(self, monkeypatch):
        """A 300 s timer must not turn an expected condition into a
        traceback every five minutes -- and must not, in buying that
        quiet, swallow a real write failure."""

        def _refused(file):
            raise SettingsSaveRefusedError(reason='no_hardware', file=file)

        ctx = SimpleNamespace(session=SimpleNamespace(save_settings=_refused))
        log = MagicMock()

        flush = _app_method('_flush_current_json', ctx=ctx, logger=log)
        flush(object(), 0.0)

        assert not log.exception.called, 'an expected refusal must not log a traceback'
        assert log.debug.called
        assert 'no_hardware' in log.debug.call_args[0][0]

        def _broken(file):
            raise OSError('disk full')

        ctx.session.save_settings = _broken
        log.reset_mock()
        flush(object(), 0.0)

        assert log.exception.called, 'a real write failure must still surface'


# ---------------------------------------------------------------------------
# E4a -- one popup, one log line
# ---------------------------------------------------------------------------


class TestOnePopupOneLine:
    def test_a_multiline_message_logs_as_one_line(self, captured_logs):
        """A continuation line carries no level or timestamp prefix, so
        every line-oriented consumer of the log miscounts -- including
        the post-mortem this defect's own evidence came from. Fixed at
        the record, never per caller."""
        gui = captured_logs('LVP.gui_interactions')
        notify = captured_logs('LVP.notifications')

        gui_logger.notification('WARNING', 'Settings file could not be read', SEVEN_LINE_MESSAGE)
        assert len(gui) == 1
        assert '\n' not in gui[0].getMessage()
        assert '\\n' in gui[0].getMessage(), 'the breaks must be escaped, not deleted'

        gui.clear()
        notification_center.notifications.notify(
            Severity.WARNING,
            'Settings',
            'Settings file could not be read (one line pin)',
            SEVEN_LINE_MESSAGE,
        )
        assert len(notify) == 1
        assert '\n' not in notify[0].getMessage()
        # The bus writes the forensic record too; both surfaces stay flat.
        assert gui and all('\n' not in record.getMessage() for record in gui)

        gui.clear()
        gui_logger.popup_response('Objective\nconfirm', 'Use\ndefaults')
        assert len(gui) == 1
        assert '\n' not in gui[0].getMessage()

    def test_one_line_is_reversible(self):
        """Escaping rather than indenting: an indented continuation is
        still multiple physical lines, and the original text has to be
        recoverable from the record."""
        assert gui_logger.one_line('a\nb\r\nc\rd') == 'a\\nb\\nc\\nd'


# ---------------------------------------------------------------------------
# E3 -- an unrenderable popup is a visible defect
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop_status(monkeypatch):
    """Set the loop status the detector will actually read.

    BOTH cases arrange it; neither trusts the ambient conftest stub.
    Two facts make ambient unknowable from here: a dev/CI machine may
    have real Kivy installed, and another test file (test_audit_fixes)
    purges and restores the kivy stubs, so which object
    ``from kivy.base import EventLoop`` resolves to depends on suite
    ORDER. A negative test that trusted the stub was green alone and red
    after that file -- green for the wrong reason either way, since the
    real loop is 'idle' under pytest and would have marked every popup.
    """

    def _set(status):
        module = importlib.import_module('kivy.base')
        monkeypatch.setattr(module, 'EventLoop', SimpleNamespace(status=status), raising=False)

    return _set


class TestThePreMainloopDetector:
    def test_a_pre_mainloop_popup_is_marked(self, event_loop_status, captured_logs):
        """ "Opened but unrenderable" is a legal state with no detector,
        so make it loud on both surfaces instead of silent on neither."""
        event_loop_status('idle')
        popup_log = captured_logs('LVP.ui.notification_popup')
        gui = captured_logs('LVP.gui_interactions')

        notification_popup._log_show('confirm', 'WARNING', 'Settings', 'nobody can see this')

        assert len(popup_log) == 1
        assert popup_log[0].levelno == logging.ERROR
        assert '(pre-mainloop)' in popup_log[0].getMessage()
        assert gui and '(pre-mainloop)' in gui[0].getMessage()

    def test_no_marker_under_a_running_loop(self, event_loop_status, captured_logs):
        """The everyday case must stay quiet, or the marker means
        nothing -- every popup in a normal session would carry it."""
        event_loop_status('started')
        popup_log = captured_logs('LVP.ui.notification_popup')
        gui = captured_logs('LVP.gui_interactions')

        notification_popup._log_show('confirm', 'WARNING', 'Settings', 'the user can see this')

        assert len(popup_log) == 1
        assert popup_log[0].levelno == logging.INFO
        assert '(pre-mainloop)' not in popup_log[0].getMessage()
        assert gui and '(pre-mainloop)' not in gui[0].getMessage()


# ---------------------------------------------------------------------------
# E4b -- the banner tells the truth
# ---------------------------------------------------------------------------


class TestTheDebugBannerTellsTheTruth:
    def test_a_rejected_current_json_is_not_named_as_the_debug_source(self, tmp_path, monkeypatch):
        """The startup banner reported "(from current.json)" for a file
        the reader had just refused -- the one line that would have told
        the bench operator the truth 19 s into the log."""
        data = tmp_path / 'data'
        data.mkdir()
        (data / 'current.json').write_text('{"debug_mode": true,,,')
        monkeypatch.setattr(settings_init, 'debug_setting_source', None)

        with pytest.raises(settings_init.SettingsFileError):
            settings_init.load_debug_setting(str(tmp_path))

        assert settings_init.debug_setting_source is None

    def test_a_readable_file_is_still_named(self, tmp_path, monkeypatch):
        """Control: the banner must still name the file it did read, or
        the fix has simply blinded it."""
        data = tmp_path / 'data'
        data.mkdir()
        (data / 'current.json').write_text(json.dumps({'debug_mode': True}))
        monkeypatch.setattr(settings_init, 'debug_setting_source', None)

        assert settings_init.load_debug_setting(str(tmp_path)) is True
        assert settings_init.debug_setting_source == 'current.json'


# ---------------------------------------------------------------------------
# The startup frame: two deferred questions, one order-independent answer
# ---------------------------------------------------------------------------


class TestTheDoubleDeferralFrame:
    @pytest.mark.parametrize('no_hardware', [False, True])
    @pytest.mark.parametrize('order', [(0, 1), (1, 0)])
    def test_the_double_deferral_frame(self, monkeypatch, tmp_path, no_hardware, order):
        """on_start queues both questions into the same frame. Clock FIFO
        puts the settings one first, but the objective prompt must not
        DEPEND on that: whichever callback runs first, an answer to the
        objective question cannot be kept while settings are provisional,
        and the cancel-less objective modal must not cover the dialog
        whose resolution is what makes answers saveable."""
        _make_provisional(monkeypatch, tmp_path)

        vertical_control = _VerticalControlStand()
        settings = {
            'microscope': 'LS850',
            'objective_confirmed': False,
            'turret_position': 1,
            'turret_objectives': {1: None},
            'objective_id': '20x Oly',
        }
        ctx = SimpleNamespace(
            settings=settings,
            lumaview=SimpleNamespace(scope=SimpleNamespace(no_hardware=no_hardware)),
            session=SimpleNamespace(
                settings_are_provisional=settings_init.settings_are_provisional,
                retire_rejected_settings=settings_init.retire_rejected_current_json,
                update_settings=lambda key, value: settings.__setitem__(key, value),
            ),
            motion_settings=SimpleNamespace(
                ids={
                    'microscope_settings_id': SimpleNamespace(
                        scopes={'LS850': {'Turret': True}},
                    ),
                    'verticalcontrol_id': vertical_control,
                }
            ),
        )
        monkeypatch.setattr(_app_ctx, 'ctx', ctx)

        confirms = []
        objectives = []
        monkeypatch.setattr(
            notification_popup, 'show_confirmation_popup', lambda **kw: confirms.append(kw)
        )
        monkeypatch.setattr(
            notification_popup,
            'show_objective_selection_popup',
            lambda **kw: objectives.append(kw),
        )

        clock = _FakeClock()
        log = MagicMock()
        stand = _AppStand()
        ask = _app_method('_ask_about_rejected_settings', ctx=ctx, logger=log, Clock=clock)
        prompt = _app_method('_prompt_objective_if_needed', ctx=ctx, logger=log, Clock=clock)

        # on_start's order: the settings question is queued first.
        ask(stand)
        prompt(stand)
        assert len(clock.queue) == 2, 'both questions must be deferred, not opened inline'

        clock.run_all(order=order)

        assert len(confirms) == 1, 'the settings question must be asked exactly once'
        assert objectives == [], (
            'the objective prompt must stay suppressed while settings are provisional'
        )
        assert settings['objective_confirmed'] is False, (
            'unanswered, not answered-by-default -- the next usable session must ask'
        )
