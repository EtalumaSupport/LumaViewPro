# Copyright Etaluma, Inc.
"""Regression for native file dialogs freezing the app (#750).

All three dialog button classes used to run their Windows/Linux tkinter
picker SYNCHRONOUSLY on the Kivy main thread -- the app froze for as long
as the panel was open, and because the blocked thread was the one that
reports health, the failure was self-censoring. macOS had an async runner
since the beachball fix, but it was wired per-platform inside each choose()
method, leaving three synchronous tkinter branches behind.

The contract now: ONE canonical runner (_run_native_dialog_async) through
which every dialog open flows on every platform, three blocking platform
primitives confined to the worker thread, an app-wide single-flight guard
(module-level -- a fresh per-call button instance defeats a per-button
flag), a worker exception policy that can never leave the guard latched,
and a guard expiry so one wedged panel cannot lock out every dialog
context until restart.

Folds in the still-valid cases from the retired test_macos_dialog_async.py
(delivery, cancel, in-flight guard, flag clear, thread spawn) -- the macOS
beachball intent is preserved and broadened to all platforms.

file_dialogs.py imports kivy at module top (mocked in the suite), so the
runner is carved out of source via AST and exec'd with fake threading +
Clock -- exercising the real function body, not a copy. Seam tests are
AST-based (a raw text scan for `Tk(` trips on prose comments).
"""

import ast
import logging
import pathlib
import time
import typing

_SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'file_dialogs.py'
_SRC = _SRC_PATH.read_text()
_TREE = ast.parse(_SRC)

# The only functions allowed to touch tkinter: the three platform
# primitives plus the shared foregrounded-root builder they use.
_PRIMITIVE_NAMES = {
    '_foregrounded_tk_root',
    '_platform_native_choose_folder',
    '_platform_native_open_file',
    '_platform_native_save_file',
}
_CHOOSE_CLASSES = ('FileChooseBTN', 'FolderChooseBTN', 'FileSaveBTN')


def _function_defs():
    return [n for n in ast.walk(_TREE) if isinstance(n, ast.FunctionDef)]


def _primitive_line_spans():
    return [(n.lineno, n.end_lineno) for n in _function_defs() if n.name in _PRIMITIVE_NAMES]


def _choose_method(class_name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'choose':
                    return item
    raise AssertionError(f'{class_name}.choose not found')


# ---------------------------------------------------------------------------
# Seam tests: the cluster stays closed (no fourth synchronous branch).
# ---------------------------------------------------------------------------


def test_tkinter_confined_to_platform_primitives():
    """No tkinter usage (Tk construction, filedialog.* call, tkinter import)
    anywhere in file_dialogs.py outside the platform primitives -- the guard
    that keeps a synchronous main-thread picker from quietly returning."""
    spans = _primitive_line_spans()
    assert len(spans) == len(_PRIMITIVE_NAMES), 'a platform primitive is missing'

    def _in_primitive(lineno):
        return any(lo <= lineno <= hi for lo, hi in spans)

    offenders = []
    for node in ast.walk(_TREE):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = getattr(node, 'module', None) or ''
            is_tk_import = 'tkinter' in names or any('tkinter' in a.name for a in node.names)
            if is_tk_import and not _in_primitive(node.lineno):
                offenders.append(f'tkinter import at line {node.lineno}')
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == 'Tk' and not _in_primitive(node.lineno):
                offenders.append(f'Tk() at line {node.lineno}')
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == 'filedialog'
                and not _in_primitive(node.lineno)
            ):
                offenders.append(f'filedialog.{fn.attr} at line {node.lineno}')
    assert offenders == [], f'tkinter usage escaped the primitives: {offenders}'


def test_choose_methods_route_through_the_one_runner():
    """Each choose() body calls _run_native_dialog_async and carries no
    sys.platform branch -- platform dispatch lives in the primitives only."""
    for class_name in _CHOOSE_CLASSES:
        method = _choose_method(class_name)
        body_src = ast.get_source_segment(_SRC, method)
        assert '_run_native_dialog_async(' in body_src, (
            f'{class_name}.choose does not use the canonical async runner'
        )
        assert 'sys.platform' not in body_src, (
            f'{class_name}.choose grew its own platform branch back'
        )


def test_tkinter_root_is_foregrounded_and_always_destroyed():
    """The shared root builder lifts + focuses (pickers must not open
    buried), and every tkinter primitive destroys its root in a finally so
    an exception cannot leak a root across threads."""
    builder_src = ast.get_source_segment(
        _SRC, next(n for n in _function_defs() if n.name == '_foregrounded_tk_root')
    )
    assert 'lift()' in builder_src and 'focus_force()' in builder_src

    for name in _PRIMITIVE_NAMES - {'_foregrounded_tk_root'}:
        node = next(n for n in _function_defs() if n.name == name)
        src = ast.get_source_segment(_SRC, node)
        assert 'finally:' in src and 'root.destroy()' in src, (
            f'{name} does not guarantee root destruction on the exception path'
        )


def test_dialog_runs_on_a_background_thread():
    """The runner must spawn a daemon worker -- guards against a refactor
    dropping the thread and reintroducing the main-thread block."""
    runner_src = ast.get_source_segment(
        _SRC, next(n for n in _function_defs() if n.name == '_run_native_dialog_async')
    )
    assert 'threading.Thread(' in runner_src
    assert 'daemon=True' in runner_src


def test_macos_primitives_use_the_zombie_backstop_timeout():
    """The osascript timeout is the shared hour-long backstop, not a short
    user-facing limit that silently cancels a legitimately-open panel."""
    assert 'timeout=120' not in _SRC
    assert _SRC.count('timeout=_MACOS_DIALOG_TIMEOUT_S') == 3


# ---------------------------------------------------------------------------
# Behavior tests: the REAL runner body, exec'd with synchronous fakes.
# ---------------------------------------------------------------------------


class _SyncThread:
    """threading.Thread stand-in that runs the target synchronously on start."""

    def __init__(self, target=None, daemon=None, **_):
        self._target = target
        self.daemon = daemon

    def start(self):
        self._target()


class _DeferredThread:
    """threading.Thread stand-in that records the target for a later manual
    run -- lets a test hold a dialog 'open' across further runner calls."""

    pending: typing.ClassVar[list] = []

    def __init__(self, target=None, daemon=None, **_):
        self._target = target
        self.daemon = daemon

    def start(self):
        _DeferredThread.pending.append(self._target)


class _SyncClock:
    """Clock stand-in: schedule_once runs the callback immediately."""

    @staticmethod
    def schedule_once(cb, _timeout):
        cb(0)


class _Button:
    def __init__(self, context='test_context'):
        self.context = context


def _load_runner(thread_cls=_SyncThread):
    """Exec the real _run_native_dialog_async body with fakes; returns
    (runner, guard_dict, log_records list)."""
    node = next(n for n in _function_defs() if n.name == '_run_native_dialog_async')
    guard = {'active': False, 'context': '', 'since': 0.0, 'token': 0}
    test_logger = logging.getLogger('test_750_runner')
    namespace = {
        'threading': type('T', (), {'Thread': thread_cls}),
        'Clock': _SyncClock,
        'time': time,
        'logger': test_logger,
        '_dialog_in_flight': guard,
        '_DIALOG_STUCK_NOTIFY_S': 60.0,
        '_DIALOG_GUARD_EXPIRY_S': 3600.0,
    }
    exec(ast.get_source_segment(_SRC, node), namespace)
    return namespace['_run_native_dialog_async'], guard


def test_delivers_selected_path_to_callback():
    runner, _guard = _load_runner()
    delivered = []
    runner(_Button(), lambda: '/some/path', lambda p: delivered.append(p))
    assert delivered == ['/some/path']


def test_cancel_does_not_invoke_callback():
    runner, guard = _load_runner()
    delivered = []
    runner(_Button(), lambda: None, lambda p: delivered.append(p))
    assert delivered == []
    assert guard['active'] is False


def test_in_flight_guard_blocks_second_dialog_even_on_a_fresh_instance():
    """The guard is app-wide: a SECOND button instance (the programmatic
    fresh-FileSaveBTN shape) must be rejected while a dialog is open.
    Pre-fix the flag was per-instance, so a fresh instance sailed past."""
    runner, _guard = _load_runner(thread_cls=_DeferredThread)
    _DeferredThread.pending.clear()
    calls = []
    runner(_Button('first'), lambda: calls.append('one') or '/p', lambda p: None)
    runner(_Button('second'), lambda: calls.append('two') or '/q', lambda p: None)
    assert len(_DeferredThread.pending) == 1, 'second dialog stacked behind the first'
    _DeferredThread.pending.pop()()
    assert calls == ['one']


def test_in_flight_flag_clears_after_delivery():
    runner, guard = _load_runner()
    runner(_Button(), lambda: '/p', lambda p: None)
    assert guard['active'] is False


def test_rejected_reclick_logs_context_and_elapsed(caplog):
    runner, guard = _load_runner(thread_cls=_DeferredThread)
    _DeferredThread.pending.clear()
    runner(_Button('load_protocol'), lambda: '/p', lambda p: None)
    with caplog.at_level(logging.WARNING, logger='test_750_runner'):
        runner(_Button('save_graph'), lambda: '/q', lambda p: None)
    assert any(
        'save_graph' in r.message and 'load_protocol' in r.message for r in caplog.records
    ), f'rejection log must name both contexts; got {[r.message for r in caplog.records]}'
    _DeferredThread.pending.pop()()
    assert guard['active'] is False


def test_stuck_dialog_reclick_notifies_user(monkeypatch):
    from modules.notification_center import notifications

    fired = []
    monkeypatch.setattr(notifications, 'warning', lambda *a, **k: fired.append(a))
    runner, guard = _load_runner(thread_cls=_DeferredThread)
    _DeferredThread.pending.clear()
    runner(_Button('first'), lambda: '/p', lambda p: None)
    guard['since'] -= 120.0
    runner(_Button('second'), lambda: '/q', lambda p: None)
    assert any(a[1] == 'A File Dialog May Already Be Open' for a in fired), (
        f'a re-click on a long-stuck dialog must notify; got {fired}'
    )
    _DeferredThread.pending.pop()()


def test_raising_dialog_clears_guard_and_notifies(monkeypatch):
    """A primitive that raises (missing python3-tk, TclError) must clear the
    guard and tell the user -- a latched guard would silently lock out every
    dialog context until restart."""
    from modules.notification_center import notifications

    fired = []
    monkeypatch.setattr(notifications, 'error', lambda *a, **k: fired.append(a))
    runner, guard = _load_runner()
    delivered = []

    def _boom():
        raise RuntimeError('no display')

    runner(_Button(), _boom, lambda p: delivered.append(p))
    assert guard['active'] is False, 'a raising primitive latched the guard'
    assert delivered == []
    assert any(a[1] == 'File Dialog Failed' for a in fired), (
        f'the user must hear about a failed picker; got {fired}'
    )


def test_expired_guard_rearms_and_drops_the_stale_result():
    """One wedged panel must not lock dialogs out forever: past the expiry
    the guard re-arms for the next request, and the stale panel's eventual
    result is dropped instead of firing its callback into the newer flow."""
    runner, guard = _load_runner(thread_cls=_DeferredThread)
    _DeferredThread.pending.clear()
    stale_delivered = []
    fresh_delivered = []

    runner(_Button('stale'), lambda: '/stale', lambda p: stale_delivered.append(p))
    stale_worker = _DeferredThread.pending.pop()
    guard['since'] -= 4000.0

    runner(_Button('fresh'), lambda: '/fresh', lambda p: fresh_delivered.append(p))
    fresh_worker = _DeferredThread.pending.pop()

    stale_worker()
    assert stale_delivered == [], 'a stale dialog result leaked past the token check'
    assert guard['active'] is True, 'the stale delivery must not clear the fresh guard'

    fresh_worker()
    assert fresh_delivered == ['/fresh']
    assert guard['active'] is False
