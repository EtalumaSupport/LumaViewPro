# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every open-file picker starts in the live folder.

Only Load Protocol gave its picker a starting folder. Load Image, Load Method,
Load Object-Analysis Data and Quick Enhance's load passed none, so macOS
opened wherever its last dialog was, often Documents, while the folder, save
and file-or-folder pickers all started in the live folder (Eric's batch 3
Part B sim walk, 2026-10-02).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ui.file_dialogs as file_dialogs


@pytest.mark.parametrize(
    'context',
    [
        'load_protocol',
        'load_cell_count_input_image',
        'load_quick_enhance_input_image',
        'load_cell_count_method',
        'load_graphing_data',
    ],
)
def test_the_picker_opens_in_the_live_folder(monkeypatch, tmp_path, context):
    opened = []
    monkeypatch.setattr(
        file_dialogs._app_ctx, 'ctx', SimpleNamespace(settings={'live_folder': str(tmp_path)})
    )
    monkeypatch.setattr(
        file_dialogs,
        '_platform_native_open_file',
        lambda initial_dir, filetypes: opened.append(initial_dir),
    )
    monkeypatch.setattr(
        file_dialogs, '_run_native_dialog_async', lambda btn, open_, deliver: open_()
    )
    file_dialogs.FileChooseBTN.choose(SimpleNamespace(), context)
    assert opened == [str(tmp_path)]
