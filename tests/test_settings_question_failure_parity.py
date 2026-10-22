# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The unanswerable settings question, and failure parity for the write.

An unreadable current.json used to be raised with the user inside
``build()`` -- before Kivy attaches the root widget, so the dialog was
created, opened, logged, and then painted UNDER the app root with every
touch routed to the root. Never rendered, never answerable. The user
therefore never resolved the provisional state, so nothing was saved
for the rest of the session -- silently -- and the whole session's
changes were lost at exit with nothing said.

Two arms fix that, and each covers the other:

* the question is asked from ``on_start`` and Clock-deferred inside the
  helper, so it is posed at the one moment an answer can be given, and
  its failure-to-present stays fatal at the new call site;
* a provisional session keeps its writes in memory and says so once
  (``SettingsNotSavedNotice``, pinned in
  ``tests/test_every_accepted_settings_write_is_saved.py``), so a session
  that somehow still never sees the dialog says it is not saving
  instead of losing the user's work quietly.

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

import pathlib
import ast
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
from modules.exceptions import SettingsFileNotReplacedError
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.ast_seams import REPO_ROOT, direct_call_names, parse_module
from tests.settings_fixtures import complete_settings
from tests.installation_fixtures import copy_installation_files
from ui.ui_helpers import run_reported
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

    Modelled on direct_call_names, but attribute-qualified: the plain
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
    """The real renderer methods; only the widget tree is stood in for."""

    prompt_if_objective_unknown = VerticalControl.prompt_if_objective_unknown
    _render_objective_question = VerticalControl._render_objective_question
    _apply_objective_answer = VerticalControl._apply_objective_answer
    # Borrowed, not stubbed: it decides whether a startup step waiting
    # on the objective runs at all.
    _resolve_objective = VerticalControl._resolve_objective

    def __init__(self):
        self.ids = {}
        for position in range(1, 5):
            self.ids[f'turret_pos_{position}_btn'] = SimpleNamespace(text=str(position))

    def show_turret_state(self, prompt=True):
        pass


class _AppStand:
    """The two collaborators the settings question calls back into."""

    def __init__(self):
        self.re_asked = 0
        self.objective_prompts = 0
        self.stopped = 0
        self.protocol_loads = 0
        self._persisted_protocol_loaded = False

    def _load_persisted_protocol_once(self):
        # The real one is latched for the same reason: the objective
        # question is asked from several places and only the first may
        # load the saved protocol.
        if self._persisted_protocol_loaded:
            return
        self._persisted_protocol_loaded = True
        self.protocol_loads += 1

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
    # The factory builds the session's helpers from this root and refuses
    # to configure the scope without them.
    copy_installation_files(data)
    monkeypatch.setattr(settings_init, 'settings', None)
    monkeypatch.setattr(settings_init, 'rejected_current_json', None)
    return ScopeSession.create(
        ScopeSession.load_user_settings(str(tmp_path)), source_path=str(tmp_path), simulate=True
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

        assert '_ask_about_rejected_settings' not in direct_call_names(build)

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
            for call in direct_call_names(fn)
            if call.startswith('show_') and call.endswith('_popup')
        ]
        assert offenders == [], f'popup(s) opened before the root attaches: {offenders}'

    def test_the_answer_path_survives_a_locked_file(self, monkeypatch, tmp_path, centre_posts):
        """The retire is a file rename. Under a Windows AV/indexer lock
        it raises -- from a button callback, where an escape kills the
        process with no teardown. It must say so, re-present the
        question, and NOT go on to the objective prompt: settings are
        still provisional, so that answer still could not be kept.

        What it says is the retire's own typed fault, shown once by the
        one reporter: the handler writes no popup and no log line of its
        own, and names the OS's reason rather than guessing one."""
        _make_provisional(monkeypatch, tmp_path)

        def _locked(src, dst):
            raise PermissionError(13, 'The process cannot access the file', src)

        monkeypatch.setattr(settings_init.os, 'replace', _locked)
        from modules.lumascope_api.bring_up import BringUpRecord, SettingsSetAside

        ctx = SimpleNamespace(
            session=SimpleNamespace(
                settings_are_provisional=settings_init.settings_are_provisional,
                retire_rejected_settings=settings_init.retire_rejected_current_json,
                bring_up_record=lambda: BringUpRecord(
                    parts=(),
                    settings_set_aside=SettingsSetAside(*settings_init.rejected_current_json),
                ),
            )
        )
        clock = _FakeClock()
        log = MagicMock()
        stand = _AppStand()

        shown = {}
        monkeypatch.setattr(
            notification_popup,
            'show_confirmation_popup',
            lambda **kwargs: shown.update(kwargs),
        )

        ask = _app_method(
            '_ask_about_rejected_settings',
            ctx=ctx,
            logger=log,
            Clock=clock,
            run_reported=run_reported,
        )
        ask(stand)

        # Nothing opened yet -- the whole point of the deferral.
        assert shown == {}
        clock.run_all()
        assert shown['confirm_text'] == 'Use defaults'

        # The user answers "Use defaults" and the rename fails.
        shown['on_confirm']()

        assert stand.stopped == 0, 'a locked file is recoverable; it must not stop the app'
        assert not log.error.called, 'the handler logs nothing of its own beside the report'
        assert [(n.title, n.severity) for n in centre_posts] == [
            ('Settings File Not Replaced', notification_center.Severity.ERROR)
        ]
        assert 'The process cannot access the file' in centre_posts[0].message
        assert 'in use by another program' not in centre_posts[0].message
        assert stand.re_asked == 1, 'the unresolved question must be put back to the user'
        assert stand.objective_prompts == 0, (
            'settings are still provisional, so the objective answer still could not be kept'
        )

    def test_a_failed_retire_raises_its_own_fault_and_keeps_the_settings_provisional(
        self, monkeypatch, tmp_path
    ):
        """The retire is the API an L2 caller reaches too, so the failure
        is raised there, typed and chained from the OS's error, and the
        state it would have changed is left as it was."""
        _make_provisional(monkeypatch, tmp_path)
        locked = PermissionError(13, 'Permission denied', 'current.json')

        def _locked(src, dst):
            raise locked

        monkeypatch.setattr(settings_init.os, 'replace', _locked)

        with pytest.raises(SettingsFileNotReplacedError) as raised:
            settings_init.retire_rejected_current_json()

        assert raised.value.__cause__ is locked
        assert 'Permission denied' in str(raised.value)
        assert settings_init.settings_are_provisional()


# ---------------------------------------------------------------------------
# E2 -- failure parity at the API boundary
# ---------------------------------------------------------------------------


class TestTheProvisionalStateIsTheSessions:
    def test_session_exposes_the_provisional_query_and_remedy(self, session, monkeypatch, tmp_path):
        """The GUI must not reach around the Session into settings_init:
        an L2 caller needs the same query and the same remedy."""
        assert session.settings_are_provisional() is False

        _make_provisional(monkeypatch, tmp_path)
        assert session.settings_are_provisional() is True

        calls = []

        def _retire():
            calls.append(True)
            return pathlib.Path('/data/current.json.rejected-20260831-120000')

        monkeypatch.setattr(settings_init, 'retire_rejected_current_json', _retire)

        assert session.retire_rejected_settings() == pathlib.Path(
            '/data/current.json.rejected-20260831-120000'
        )
        assert calls == [True]

    def test_the_apps_close_leaves_saving_and_teardown_to_the_session(self):
        """Structural, because the app's close cannot be driven headlessly: it
        tears down executors, threads, notification listeners and real
        hardware. Both ways the app closes -- the window's close and on_stop
        -- run the host's part before they reach the session's close, and
        the host's part neither saves the settings nor tears anything down:
        the session's close saves once and tears down, alone."""
        methods = _app_methods()

        def _chain(node):
            out = []
            value = node
            while isinstance(value, ast.Attribute):
                out.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                out.append(value.id)
            return out

        def _calls(method):
            return [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]

        teardown = methods['_shut_the_session_down']
        assert any(
            node.func.attr == 'shutdown' and _chain(node.func.value) == ['session', 'ctx']
            for node in _calls(teardown)
        ), 'the close must still tear the session (and its hardware) down'

        # Each way the app closes runs the host's part first, then reaches the
        # session's close through _end_the_session (directly, or inside a
        # served REST server's close): on_stop calls it; the window's close
        # starts the thread that runs it.
        assert any(
            isinstance(n, ast.Attribute) and n.attr == '_shut_the_session_down'
            for n in ast.walk(methods['_end_the_session'])
        ), "_end_the_session must reach the session's close"
        for name in ('on_stop', '_close_the_session'):
            method = methods[name]
            prepared = [n.lineno for n in _calls(method) if n.func.attr == '_prepare_the_close']
            torn = [
                n.lineno
                for n in ast.walk(method)
                if isinstance(n, ast.Attribute) and n.attr == '_end_the_session'
            ]
            assert prepared and torn, f"{name} must prepare, then reach the session's close"
            assert min(prepared) < min(torn), f"{name} must prepare BEFORE the session's close"

        # The teardown is the session's, whole: no part of the app's close
        # stops a thread, lane or hardware itself.
        for name in ('on_stop', '_prepare_the_close', '_end_the_session', '_shut_the_session_down'):
            stray = [
                (node.lineno, ast.unparse(node.func))
                for node in _calls(methods[name])
                if node.func.attr
                in ('disconnect', 'shutdown_threads', 'stop_motion', 'disable_metrics')
                or (node.func.attr == 'shutdown' and _chain(node.func.value) != ['session', 'ctx'])
                or (node.func.attr == 'stop' and _chain(node.func.value) != ['profiling_helper'])
            ]
            assert stray == [], f'{name} performs teardown steps itself: {stray}'

        # The settings are the session's to save, at every accepted write
        # and at its close; the host's close saves nothing itself.
        for name in ('on_stop', '_close_the_session', '_prepare_the_close'):
            saves = [ast.unparse(n.func) for n in _calls(methods[name]) if 'save' in n.func.attr]
            assert saves == [], f'{name} saves the settings itself: {saves}'


# ---------------------------------------------------------------------------
# E4a -- one popup, one log line
# ---------------------------------------------------------------------------


class TestOnePopupOneLine:
    def test_a_multiline_message_logs_as_one_line(self, captured_logs):
        """A continuation line carries no level or timestamp prefix, so
        every line-oriented consumer of the log miscounts -- including
        the post-mortem this defect's own evidence came from. Fixed at
        the record, never per caller.

        Each count is of this test's own record, found by its words: the
        loggers are process-wide, and a record another thread writes
        while the test runs is not one of these."""
        gui = captured_logs('LVP.gui_interactions')
        notify = captured_logs('LVP.notifications')

        def own(records, words):
            return [record for record in records if words in record.getMessage()]

        gui_logger.notification('WARNING', 'Settings file could not be read', SEVEN_LINE_MESSAGE)
        written = own(gui, 'Settings file could not be read')
        assert len(written) == 1
        assert '\n' not in written[0].getMessage()
        assert '\\n' in written[0].getMessage(), 'the breaks must be escaped, not deleted'

        gui.clear()
        notification_center.notifications.notify(
            Severity.WARNING,
            'Settings',
            'Settings file could not be read (one line pin)',
            SEVEN_LINE_MESSAGE,
        )
        posted = own(notify, 'Settings file could not be read (one line pin)')
        assert len(posted) == 1
        assert '\n' not in posted[0].getMessage()
        # The bus writes the forensic record too; both surfaces stay flat.
        forensic = own(gui, 'Settings file could not be read (one line pin)')
        assert len(forensic) == 1
        assert '\n' not in forensic[0].getMessage()

        gui.clear()
        gui_logger.popup_response('Objective\nconfirm', 'Use\ndefaults')
        answered = own(gui, 'Objective')
        assert len(answered) == 1
        assert '\n' not in answered[0].getMessage()

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
    purges and restores the kivy stubs, so which object ``kivy.base``
    names depends on suite ORDER. A negative test that trusted the stub
    was green alone and red after that file -- green for the wrong reason
    either way, since the real loop is 'idle' under pytest and would have
    marked every popup. The detector reads the name its own module bound
    at import, so that name is the one set here.
    """

    def _set(status):
        monkeypatch.setattr(notification_popup, 'EventLoop', SimpleNamespace(status=status))

    return _set


class TestThePreMainloopDetector:
    def test_a_pre_mainloop_popup_is_marked(self, event_loop_status, captured_logs):
        """ "Opened but unrenderable" is a legal state with no detector,
        so make it loud on both surfaces instead of silent on neither."""
        event_loop_status('idle')
        popup_log = captured_logs('LVP.ui.notification_popup')
        gui = captured_logs('LVP.gui_interactions')

        notification_popup._log_show('confirm', 'Settings', 'nobody can see this')

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

        notification_popup._log_show('confirm', 'Settings', 'the user can see this')

        assert len(popup_log) == 1
        assert popup_log[0].levelno == logging.INFO
        assert '(pre-mainloop)' not in popup_log[0].getMessage()
        assert gui and '(pre-mainloop)' not in gui[0].getMessage()


class TestADialogRecordNamesNoLevel:
    def test_the_gui_record_of_a_dialog_is_a_dialog_record(self, event_loop_status, captured_logs):
        """A dialog has no severity: a question asks, and a notice's level is
        its notification's, which the notification center records once. The
        dialog's record read 'NOTIFICATION INFO' for every dialog, so a
        refusal the main log had at WARNING read INFO beside it."""
        event_loop_status('started')
        gui = captured_logs('LVP.gui_interactions')

        notification_popup._log_show('dialog', 'Protocol Refused', 'x.tsv was not loaded')

        assert [r.getMessage() for r in gui] == ['DIALOG | Protocol Refused | x.tsv was not loaded']


# ---------------------------------------------------------------------------
# E4b -- the banner tells the truth
# ---------------------------------------------------------------------------


class TestTheDebugBannerTellsTheTruth:
    def test_a_rejected_current_json_is_not_named_as_the_debug_source(self, tmp_path, monkeypatch):
        """The startup banner reported "(from current.json)" for a file
        the reader had just refused -- the one line that would have told
        the bench operator the truth 19 s into the log. The shipped file
        answers instead, as preparation's fallback does, and is named."""
        data = tmp_path / 'data'
        data.mkdir()
        (data / 'current.json').write_text('{"debug_mode": true,,,')
        (data / 'settings.json').write_text(json.dumps({'debug_mode': False}))
        monkeypatch.setattr(settings_init, 'debug_setting_source', None)

        assert settings_init.load_debug_setting(str(tmp_path)) is False
        assert settings_init.debug_setting_source == 'settings.json'

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
        # The decision to withhold the question is the Session's, so the
        # case runs a real one: a fresh install on a turret model, whose
        # question is owed and must still not be asked while provisional.
        session = ScopeSession.create(
            complete_settings(
                microscope='LS850T',
                objective_confirmed=None,
                turret_position=1,
                turret_objectives={'1': None, '2': None, '3': None, '4': None},
                objective_id='20x Oly',
            ),
            simulate=True,
        )
        request_shutdown = session.shutdown
        try:
            self._run_the_double_deferral_frame(
                monkeypatch, session, vertical_control, no_hardware, order
            )
        finally:
            request_shutdown()

    def _run_the_double_deferral_frame(
        self, monkeypatch, session, vertical_control, no_hardware, order
    ):
        session.scope._no_hardware = no_hardware
        settings = session.settings
        ctx = SimpleNamespace(
            settings=settings,
            session=session,
            motion_settings=SimpleNamespace(ids={'verticalcontrol_id': vertical_control}),
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
        assert settings['objective_confirmed'] is None, (
            'unanswered, not answered-by-default -- the next usable session must ask'
        )
