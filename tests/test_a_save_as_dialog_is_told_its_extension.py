# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every Save As dialog is told the extension of the file it saves.

The writer writes exactly the name the dialog returns, so the dialog is
the one that adds the extension to a name typed without it -- and asks
about replacing a file of that full name. The macOS dialog was AppleScript's
``choose file name``, which takes no file type: typing ``plate1`` returned
``plate1``, the dialog asked about no file, and ``plate1.tsv`` was written
over. The macOS panel is told the type; Tk is given ``defaultextension``.
"""

from __future__ import annotations

from types import SimpleNamespace

import ui.file_dialogs as file_dialogs

_CONTEXTS = {
    'saveas_protocol': 'tsv',
    'saveas_cell_count_method': 'json',
    'save_graph': 'png',
}


def _open_each(monkeypatch, tmp_path):
    """Open each context's Save As, running its dialog in place of the worker."""
    monkeypatch.setattr(
        file_dialogs._app_ctx, 'ctx', SimpleNamespace(settings={'live_folder': str(tmp_path)})
    )
    monkeypatch.setattr(
        file_dialogs,
        '_run_native_dialog_async',
        lambda btn, open_, deliver, *, on_cancel: open_(),
    )
    for context in _CONTEXTS:
        file_dialogs.FileSaveBTN.choose(SimpleNamespace(), context)


def test_the_macos_panel_is_told_the_type(monkeypatch, tmp_path):
    scripts = []
    monkeypatch.setattr(file_dialogs, 'sys', SimpleNamespace(platform='darwin'))
    # a stand-in by design: the panel is a native window only a person answers.
    monkeypatch.setattr(file_dialogs, '_osascript_choice', lambda script: scripts.append(script))

    _open_each(monkeypatch, tmp_path)

    told = [
        f'typeWithFilenameExtension:"{extension}"' in script
        for script, extension in zip(scripts, _CONTEXTS.values(), strict=True)
    ]
    assert told == [True] * len(_CONTEXTS), scripts


def test_the_tk_dialog_is_given_the_default_extension(monkeypatch, tmp_path):
    given = []
    monkeypatch.setattr(file_dialogs, 'sys', SimpleNamespace(platform='win32'))
    # a stand-in by design: the dialog is a native window only a person answers.
    monkeypatch.setattr(
        file_dialogs, '_foregrounded_tk_root', lambda: SimpleNamespace(destroy=lambda: None)
    )
    from tkinter import filedialog

    monkeypatch.setattr(
        filedialog, 'asksaveasfilename', lambda **kw: given.append(kw['defaultextension'])
    )

    _open_each(monkeypatch, tmp_path)

    assert given == [f'.{extension}' for extension in _CONTEXTS.values()]
