# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The one window for a refusal that stops LumaViewPro before its GUI exists.

The Python-version refusal, the packages refusal and a second copy already
running are each shown here, before Kivy or the API can load, so the module
is the standard library only and ``tkinter`` is imported where it is used.
It also runs under the oldest Python the version refusal reaches (Apple's
3.9), so it uses nothing newer.

A platform's native message box sets its own width and breaks a long path
mid-word wherever it runs out; a refusal is read for the paths it names. So
the words are shown unwrapped, in a fixed-width text area as wide as the
longest line, with a scroll bar past the screen's width.

The window writes nothing of its own to stderr and logs nothing: its own
failure is returned to its caller as one line, and each caller reports it
on its own channel.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import NamedTuple

# Columns and lines kept free around the text, for the window's frame and
# its buttons.
_MARGIN_COLUMNS = 8
_MARGIN_LINES = 10

COPIED = 'Copied. Paste it before you close this window.'


class TextLayout(NamedTuple):
    """The text area's size, in characters and lines.

    Attributes:
        columns: Its width: the longest line, or the screen's cap.
        lines: Its height: the line count, or the screen's cap.
        scrolls: Whether a line is longer than the cap, so the area scrolls
            sideways rather than break it.
    """

    columns: int
    lines: int
    scrolls: bool


def layout(words: str, max_columns: int, max_lines: int) -> TextLayout:
    """The size of the text area that shows ``words`` without breaking a line."""
    rows = words.split('\n')
    longest = max(len(row) for row in rows)
    return TextLayout(
        columns=min(longest, max_columns),
        lines=min(len(rows), max_lines),
        scrolls=longest > max_columns,
    )


def bring_to_front(window) -> None:
    """Raise a Tk window above every other and give it the keyboard.

    ``-topmost`` alone lets a window open buried behind another program's on
    Windows, and behind the running LumaViewPro's; ``lift`` and
    ``focus_force`` after it bring it up.
    """
    window.attributes('-topmost', True)
    window.lift()
    window.focus_force()


def _start_on_windows(script: str, args: Sequence[str]) -> str | None:
    # The .bat opens in its own console exactly as a double click opens it,
    # and finds its folder from its own path, whatever the working folder.
    os.startfile(script, arguments=' '.join(args))
    return None


# A downloaded script carries the quarantine and may lack the exec bit, and
# `open -a Terminal` refuses the first and silently does nothing for the
# second; `bash <script>` in a Terminal window runs it either way. The paths
# reach AppleScript as arguments, never spliced into its source.
_TERMINAL_RUNS_ITS_ARGUMENTS = (
    'on run argv',
    'set command to "bash"',
    'repeat with part in argv',
    'set command to command & " " & quoted form of part',
    'end repeat',
    'tell application "Terminal"',
    'activate',
    'do script command',
    'end tell',
    'end run',
)


def _start_on_mac(script: str, args: Sequence[str]) -> str | None:
    lines = [arg for line in _TERMINAL_RUNS_ITS_ARGUMENTS for arg in ('-e', line)]
    # Waited for, so a refusal (a declined automation prompt among them) is
    # shown rather than lost.
    done = subprocess.run(
        ['osascript', *lines, script, *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        return done.stderr.strip() or f'osascript exited {done.returncode}'
    return None


# Linux has no standard terminal to open the script in; there the words name it.
_STARTERS: dict[str, Callable[[str, Sequence[str]], str | None]] = {
    'win32': _start_on_windows,
    'darwin': _start_on_mac,
}


def start_install_script(script: str, args: Sequence[str]) -> str | None:
    """Start the install script where the user can watch it, without waiting for it.

    Returns:
        None once it has started; else why it could not be.
    """
    try:
        return _STARTERS[sys.platform](script, args)
    except OSError as e:
        return str(e)


class _Window(NamedTuple):
    """A built refusal window's parts that act."""

    text: object
    buttons: dict
    status: object


def build(
    root,
    title: str,
    words: str,
    *,
    install_script: str | None = None,
    install_args: Sequence[str] = (),
    pip_line: str | None = None,
) -> _Window:
    """Lay the refusal out in ``root``, a Tk root, without showing it."""
    import tkinter
    from tkinter import font as tkfont

    root.title(title)
    root.resizable(False, False)
    fixed = tkfont.nametofont('TkFixedFont')
    size = layout(
        words,
        max_columns=root.winfo_screenwidth() // fixed.measure('0') - _MARGIN_COLUMNS,
        max_lines=root.winfo_screenheight() // fixed.metrics('linespace') - _MARGIN_LINES,
    )

    frame = tkinter.Frame(root, padx=16, pady=12)
    frame.pack(fill='both', expand=True)
    text = tkinter.Text(
        frame,
        font=fixed,
        wrap='none',
        width=size.columns,
        height=size.lines,
        borderwidth=0,
        highlightthickness=0,
    )
    text.insert('1.0', words)
    # Read-only, still selectable and copyable.
    text.configure(state='disabled')
    text.grid(row=0, column=0, sticky='nsew')
    if size.scrolls:
        bar = tkinter.Scrollbar(frame, orient='horizontal', command=text.xview)
        text.configure(xscrollcommand=bar.set)
        bar.grid(row=1, column=0, sticky='ew')

    status = tkinter.Label(
        frame, text='', justify='left', anchor='w', wraplength=size.columns * fixed.measure('0')
    )
    status.grid(row=2, column=0, sticky='ew', pady=(8, 0))

    def run_script():
        failure = start_install_script(install_script, install_args)
        if failure is None:
            root.destroy()
        else:
            status.configure(text=f'The install script could not be started: {failure}')

    def copy_pip_line():
        root.clipboard_clear()
        root.clipboard_append(pip_line)
        status.configure(text=COPIED)

    actions = []
    if install_script is not None and sys.platform in _STARTERS:
        actions.append(('Run the install script', run_script))
    if pip_line is not None:
        actions.append(('Copy pip line', copy_pip_line))
    actions.append(('Close', root.destroy))

    row = tkinter.Frame(frame)
    row.grid(row=3, column=0, sticky='e', pady=(12, 0))
    buttons = {}
    for label, act in actions:
        button = tkinter.Button(row, text=label, command=act)
        button.pack(side='left', padx=(8, 0))
        buttons[label] = button
    first = buttons[actions[0][0]]
    first.configure(default='active')
    root.bind('<Return>', lambda _event: first.invoke())
    root.bind('<Escape>', lambda _event: root.destroy())
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    return _Window(text, buttons, status)


def show_refusal(
    title: str,
    words: str,
    *,
    install_script: str | None = None,
    install_args: Sequence[str] = (),
    pip_line: str | None = None,
) -> str | None:
    """Show a refusal in front of every window until the user closes it.

    Args:
        title: The window's title.
        words: The refusal, shown unwrapped.
        install_script: The install script a button starts, where the
            platform can open it in a terminal; no button when None.
        install_args: The arguments it is started with.
        pip_line: The command a button copies; no button when None.

    Returns:
        None once the window has been shown and closed; else why it could
        not be shown, in one line.
    """
    try:
        import tkinter
    except ImportError as e:
        return f'The window could not be shown: this Python has no tkinter ({e}).'
    try:
        root = tkinter.Tk()
    except tkinter.TclError as e:
        return f'The window could not be shown: {e}.'
    root.withdraw()
    build(
        root,
        title,
        words,
        install_script=install_script,
        install_args=install_args,
        pip_line=pip_line,
    )
    # Sized without update(), which never returns once a window is mapped
    # under Apple's Tk 8.5.
    root.update_idletasks()
    width, height = root.winfo_reqwidth(), root.winfo_reqheight()
    left = max(0, (root.winfo_screenwidth() - width) // 2)
    top = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f'+{left}+{top}')
    root.deiconify()
    bring_to_front(root)
    root.mainloop()
    return None
