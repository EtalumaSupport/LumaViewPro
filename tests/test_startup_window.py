# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The one window for a refusal before the GUI exists (``ui/startup_window.py``).

It shows the words unwrapped, as wide as the longest line or the screen's
cap with a scroll bar past it; offers to start the install script where the
platform can open it in a terminal, and to copy the pip line; brings itself
in front; returns its own failure rather than writing it; and imports under
the oldest Python the version refusal reaches, with nothing but the
standard library and no tkinter until it shows.

The built-window tests use a real Tk root, withdrawn and never mapped, so no
window opens during the suite; where Tk cannot start (CI has no display)
they are skipped, and the layout and launch tests carry the contract.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ui import startup_window
from ui.startup_window import COPIED, TextLayout, build, layout, start_install_script

REPO = pathlib.Path(__file__).resolve().parents[1]
WORDS = 'LumaViewPro cannot start.\n\n    /a/long/path/to/scripts/install_mac.sh\n    /a/run.sh'


def test_the_text_is_as_wide_as_its_longest_line_and_scrolls_past_the_cap():
    assert layout(WORDS, max_columns=100, max_lines=50) == TextLayout(42, 4, False)
    assert layout(WORDS, max_columns=30, max_lines=3) == TextLayout(30, 3, True)


def test_windows_starts_the_script_with_its_arguments(monkeypatch):
    started = []
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(
        startup_window.os,
        'startfile',
        lambda path, arguments: started.append((path, arguments)),
        raising=False,
    )

    assert start_install_script(r'C:\lvp (13)\scripts\install_windows.bat', ('--venv',)) is None
    assert started == [(r'C:\lvp (13)\scripts\install_windows.bat', '--venv')]


def test_the_mac_hands_terminal_the_path_as_an_argument_and_reports_a_refusal(monkeypatch):
    calls = []

    def osascript(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stderr='Not authorized to send Apple events.\n')

    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(startup_window.subprocess, 'run', osascript)
    script = '/Users/a/Downloads/LumaViewPro "x"/scripts/install_mac.sh'

    assert start_install_script(script, ()) == 'Not authorized to send Apple events.'
    (command,) = calls
    assert command[0] == 'osascript'
    assert command[-1] == script
    # Never spliced into the AppleScript's source.
    sources = command[1:-1]
    assert all(script not in part for part in sources)
    assert 'do script command' in sources


def test_a_start_that_raises_is_returned_as_its_error(monkeypatch):
    def startfile(path, arguments):
        raise FileNotFoundError(2, 'The system cannot find the file specified', path)

    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(startup_window.os, 'startfile', startfile, raising=False)

    assert 'The system cannot find the file specified' in start_install_script(r'C:\x.bat', ())


def test_the_window_returns_its_own_failure_and_writes_nothing(monkeypatch, capsys):
    # A stand-in by design: tkinter made unimportable, as on a Python built without it.
    monkeypatch.setitem(sys.modules, 'tkinter', None)

    failure = startup_window.show_refusal('Title', WORDS)

    assert failure.startswith('The window could not be shown: this Python has no tkinter')
    assert capsys.readouterr() == ('', '')


def test_the_module_imports_and_lays_out_under_the_oldest_python_without_tkinter():
    oldest = pathlib.Path('/usr/bin/python3')
    if not oldest.is_file():
        pytest.skip('no system Python here')
    code = (
        'import sys\n'
        'from ui.startup_window import layout\n'
        "assert layout('ab\\nabcd', 3, 5) == (3, 2, True)\n"
        "assert not any(name.split('.')[0] in ('tkinter', '_lumascope', 'kivy') for name in sys.modules)\n"
    )
    done = subprocess.run(
        [str(oldest), '-I', '-c', f'import sys; sys.path.insert(0, {str(REPO)!r})\n{code}'],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr


def _is_open(window):
    """Whether ``window`` still exists; a destroyed root refuses every query instead."""
    import tkinter

    try:
        return bool(window.winfo_exists())
    except tkinter.TclError:
        return False


@pytest.fixture
def root():
    tkinter = pytest.importorskip('tkinter')
    try:
        window = tkinter.Tk()
    except tkinter.TclError as e:
        pytest.skip(f'Tk cannot start here: {e}')
    window.withdraw()
    yield window
    if _is_open(window):
        window.destroy()


def test_the_built_window_shows_every_word_unwrapped_with_the_buttons_given(root):
    built = build(root, 'Title', WORDS, install_script='/a/install.sh', pip_line='pip line')

    assert built.text.get('1.0', 'end-1c') == WORDS
    assert built.text.cget('wrap') == 'none'
    assert built.text.cget('state') == 'disabled'
    offered = list(built.buttons)
    if sys.platform in ('win32', 'darwin'):
        assert offered == ['Run the install script', 'Copy pip line', 'Close']
    else:
        assert offered == ['Copy pip line', 'Close']
    assert root.bind('<Return>') and root.bind('<Escape>')


def test_a_window_with_nothing_to_act_on_offers_close_alone(root):
    assert list(build(root, 'Title', WORDS).buttons) == ['Close']


def test_a_script_that_will_not_start_leaves_the_window_open_saying_why(root, monkeypatch):
    built = build(root, 'Title', WORDS, install_script='/a/install.sh')
    if 'Run the install script' not in built.buttons:
        pytest.skip('no terminal to start a script in on this platform')
    monkeypatch.setattr(startup_window, 'start_install_script', lambda *_: 'refused')

    built.buttons['Run the install script'].invoke()

    assert _is_open(root)
    assert built.status.cget('text') == 'The install script could not be started: refused'


def test_a_script_that_starts_closes_the_window(root, monkeypatch):
    built = build(root, 'Title', WORDS, install_script='/a/install.sh')
    if 'Run the install script' not in built.buttons:
        pytest.skip('no terminal to start a script in on this platform')
    monkeypatch.setattr(startup_window, 'start_install_script', lambda *_: None)

    built.buttons['Run the install script'].invoke()

    assert not _is_open(root)


def test_copy_puts_the_pip_line_on_the_clipboard_and_says_to_paste_it(root):
    import tkinter

    try:
        before = root.clipboard_get()
    except tkinter.TclError:  # an empty clipboard
        before = None
    built = build(root, 'Title', WORDS, pip_line='"/py" -m pip install -r "/r.txt"')
    try:
        built.buttons['Copy pip line'].invoke()

        assert root.clipboard_get() == '"/py" -m pip install -r "/r.txt"'
        assert built.status.cget('text') == COPIED
        assert _is_open(root)
    finally:
        # The suite's host keeps its own clipboard.
        root.clipboard_clear()
        if before is not None:
            root.clipboard_append(before)


def test_bring_to_front_sets_the_window_topmost(root):
    startup_window.bring_to_front(root)

    assert root.attributes('-topmost') == 1
