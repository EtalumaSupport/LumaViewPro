# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every image picker offers every image the readers take.

The cell-count preview's picker offered TIFF alone while the count reads
PNG, JPEG and BMP as well, so a PNG the count accepts could not be picked
(Eric's walk, 2026-10-04: ``no_scale.png``). The pickers now offer
``image_utils.IMAGE_SUFFIXES``, the one list the readers take.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ui.file_dialogs as file_dialogs
from modules import image_utils


def _offered(filetypes) -> set[str]:
    return {suffix for _label, suffixes in filetypes for suffix in suffixes.split()}


@pytest.fixture
def live_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(
        file_dialogs._app_ctx, 'ctx', SimpleNamespace(settings={'live_folder': str(tmp_path)})
    )
    monkeypatch.setattr(
        file_dialogs, '_run_native_dialog_async', lambda btn, open_, deliver, on_cancel: open_()
    )


@pytest.mark.parametrize(
    'context', ['load_cell_count_input_image', 'load_quick_enhance_input_image']
)
def test_an_image_file_picker_offers_every_image(monkeypatch, live_folder, context):
    offered = []
    monkeypatch.setattr(
        file_dialogs,
        '_platform_native_open_file',
        lambda initial_dir, filetypes: offered.append(_offered(filetypes)),
    )

    file_dialogs.FileChooseBTN.choose(SimpleNamespace(), context)

    assert offered == [set(image_utils.IMAGE_SUFFIXES)]


def test_the_quick_enhance_target_picker_offers_every_image(monkeypatch, live_folder):
    offered = []
    monkeypatch.setattr(
        file_dialogs,
        '_platform_native_choose_file_or_folder',
        lambda initial_dir, filetypes: offered.append(_offered(filetypes)),
    )

    file_dialogs.FileOrFolderChooseBTN.choose(SimpleNamespace(), 'choose_quick_enhance_target')

    assert offered == [set(image_utils.IMAGE_SUFFIXES)]
