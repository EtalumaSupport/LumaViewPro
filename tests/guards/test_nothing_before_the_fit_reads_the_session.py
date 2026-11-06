# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Until the fit step, the window reads no session: it has none.

The GUI brings the scope up on a thread of its own (``ScopeSession.create``
off the Kivy thread), so the window draws and stays live while the scope
comes up. Until the session is handed over, the window's context (``ctx``) is
None, and the window shows the bring-up view: the splash, a progress bar and
the step, written from bring-up's own progress report. A read of ``ctx`` on
any path that can run before the fit is a crash on that path; a control on
the bring-up view is a control with no session behind it.

What can run before the fit: ``build()``, ``on_start``, the bring-up thread,
its step's display, the hand-over, a close requested during bring-up (and
the close's popup), ``on_stop`` after a platform quit, and a bring-up popup's
remedy. Each is held here by its source, since no test builds the App.
"""

from __future__ import annotations

import ast

from tests.ast_seams import MemberNames, find_def, parse_module

_APP = 'LumaViewProApp'
# Every App method that can run while no session exists.
_BEFORE_THE_FIT = MemberNames(
    'lumaviewpro.py',
    (
        'build',
        'on_start',
        '_bring_up_on_its_thread',
        '_bring_up_the_session',
        '_show_bring_up_step',
        '_closing_once_up',
        '_hand_over',
        'with_the_session',
        '_close_during_bring_up',
        'on_stop',
    ),
)


def _app_method(name):
    node = find_def('lumaviewpro.py', name, class_name=_APP)
    assert node is not None, f'{_APP}.{name} is gone'
    return node


def _ctx_reads(node):
    return sorted({n.lineno for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == 'ctx'})


def test_no_path_before_the_fit_reads_the_windows_context():
    reads = {name: _ctx_reads(_app_method(name)) for name in sorted(_BEFORE_THE_FIT)}
    assert {name: lines for name, lines in reads.items() if lines} == {}, (
        'read ctx before the fit, where it is None'
    )


def test_the_close_and_the_remedy_read_the_context_only_once_the_fit_has_run():
    # Each reads the session through ctx only past its own question whether
    # the fit has run: on_request_close and _prepare_the_close ask the phase
    # first, and return before any read when it is not FITTED.
    for name in ('on_request_close', '_prepare_the_close'):
        method = _app_method(name)
        (asked,) = [
            n.lineno
            for n in ast.walk(method)
            if isinstance(n, ast.Compare) and 'self._phase is not _Phase.FITTED' in ast.unparse(n)
        ]
        assert all(line > asked for line in _ctx_reads(method)), name

    remedy = find_def('ui/notification_popup.py', '_take_remedy')
    assert remedy is not None
    source = ast.unparse(remedy)
    assert 'with_the_session' in source and '.ctx' not in source, (
        "a bring-up popup's remedy takes its session from bring-up, not the window's context"
    )


def test_the_scope_comes_up_on_a_daemon_thread_stamped_as_the_gui():
    on_start = _app_method('on_start')
    (thread,) = [
        n
        for n in ast.walk(on_start)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == 'threading.Thread'
    ]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in thread.keywords}
    assert keywords == {
        'target': 'self._bring_up_on_its_thread',
        'name': "'bring-up'",
        'daemon': 'True',
    }, keywords
    # The hand-over is posted to the Kivy thread, never run on the bring-up thread.
    assert 'schedule_ui(lambda dt: self._hand_over(done))' in ast.unparse(on_start)

    bring_up = ast.unparse(_app_method('_bring_up_on_its_thread'))
    assert 'with calling(Caller.GUI):' in bring_up
    assert 'self._bring_up_the_session()' in bring_up


def test_build_makes_the_bring_up_view_and_brings_nothing_up():
    build = _app_method('build')
    called = {ast.unparse(n.func) for n in ast.walk(build) if isinstance(n, ast.Call)}
    assert 'BringUpView' in called
    assert not {'ScopeSession.create', 'MainDisplay', 'AppContext.over_session'} & called, called


def test_a_close_during_bring_up_runs_the_windows_own_close():
    # One close: the hand-over gives the session to the same machinery the
    # window's close runs, never a second one.
    hand_over = ast.unparse(_app_method('_hand_over'))
    assert 'self._run_the_close(session,' in hand_over
    assert 'self._fit(session)' in hand_over
    close = ast.unparse(_app_method('_close_the_session'))
    assert 'self._run_the_close(session,' in close
    # A platform quit during bring-up closes what bring-up brings up.
    assert 'session = self._bring_up.result()' in ast.unparse(_app_method('on_stop'))


def test_the_bring_up_view_holds_no_session():
    tree = parse_module('ui/bring_up_view.py')
    imports = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert [m for m in imports if not m.startswith('kivy')] == [], imports
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not {'ctx', 'session', 'scope', 'settings'} & names
