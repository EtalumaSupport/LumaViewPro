# Copyright Etaluma, Inc.
"""Regression for the macOS file-dialog beachball.

The macOS open/save/choose-folder pickers shell out to osascript via
subprocess.run. Run inline on the Kivy main thread, that subprocess froze
the event loop for the entire time the native panel was open -- the app
beachballed ("Application Not Responding"). The fix runs the osascript call
on a daemon thread and marshals the chosen path back to the main thread via
Clock (_run_macos_dialog_async in ui/file_dialogs.py).

file_dialogs.py imports kivy at module top (mocked in the suite), so the
pure helper is loaded from source and exec'd with fake threading + Clock --
exercising the real function body, not a copy. The fakes run synchronously so
the threading contract is asserted deterministically.
"""

import ast
import pathlib

_SRC = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'file_dialogs.py').read_text()


class _SyncThread:
    """threading.Thread stand-in that runs the target synchronously on start."""

    def __init__(self, target=None, daemon=None, **_):
        self._target = target
        self.daemon = daemon

    def start(self):
        self._target()


class _SyncClock:
    """Clock stand-in: schedule_once runs the callback immediately."""

    @staticmethod
    def schedule_once(cb, _timeout):
        cb(0)


def _load_helper():
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_run_macos_dialog_async':
            namespace = {
                'threading': type('T', (), {'Thread': _SyncThread}),
                'Clock': _SyncClock,
            }
            exec(ast.get_source_segment(_SRC, node), namespace)
            return namespace['_run_macos_dialog_async']
    raise AssertionError('_run_macos_dialog_async not found in file_dialogs.py')


_run_async = _load_helper()


class _Button:
    """Minimal stand-in for the Kivy button that holds the in-flight flag."""


def test_delivers_selected_path_to_callback():
    button = _Button()
    delivered = []
    _run_async(button, lambda: '/some/path', lambda p: delivered.append(p))
    assert delivered == ['/some/path']


def test_cancel_does_not_invoke_callback():
    # osascript returns None when the user cancels the panel.
    button = _Button()
    delivered = []
    _run_async(button, lambda: None, lambda p: delivered.append(p))
    assert delivered == []


def test_in_flight_guard_blocks_second_dialog():
    # While a panel is open the Kivy button stays clickable; a second click
    # must not stack a second panel.
    button = _Button()
    button._dialog_in_flight = True
    calls = []
    _run_async(button, lambda: calls.append('opened') or '/p', lambda p: None)
    assert calls == []  # dialog_fn never ran


def test_in_flight_flag_clears_after_delivery():
    button = _Button()
    _run_async(button, lambda: '/p', lambda p: None)
    # Flag reset so the next click is accepted.
    assert button._dialog_in_flight is False


def test_dialog_runs_on_a_background_thread():
    # The dialog_fn must be invoked from the worker the Thread runs, not
    # inline on the caller -- guards against a future refactor dropping the
    # thread and reintroducing the main-thread block.
    assert 'threading.Thread(' in _SRC
    assert 'daemon=True' in _SRC


def test_darwin_callers_use_async_runner():
    # All three macOS dialog branches (open / save / choose-folder) must
    # route through the async runner, not call the blocking _macos_* helper
    # inline on the main thread. The runner is the helper def plus 3 calls.
    assert _SRC.count('_run_macos_dialog_async(') >= 4
    for primitive in ('_macos_open_file', '_macos_choose_folder', '_macos_save_file'):
        assert primitive in _SRC
