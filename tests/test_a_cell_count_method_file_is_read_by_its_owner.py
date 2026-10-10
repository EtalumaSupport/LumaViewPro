# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell-count method file is read, checked and written by the method's owner.

The panel read the file itself, checked only that a metadata entry named a
type and a version (raising a bare Exception, with nothing around it to
report it), and stored the dict before reading its fields -- so a file with
pixels-per-micron 0 loaded and counted nothing, and a file missing a field
left the panel holding a method it could not use. The owner in
modules/post_processing.py now reads, checks and writes the file, and the
panel stores a method only from the owner's answer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ui.post_processing as panel_module
from modules.exceptions import PostProcessingRefusedError
from modules.post_processing import (
    default_cell_count_method,
    load_cell_count_method,
    save_cell_count_method,
)


def _write(path, saved) -> str:
    path.write_text(json.dumps(saved) if not isinstance(saved, str) else saved)
    return str(path)


def _saved(**context) -> dict:
    method = default_cell_count_method()
    method['context'].update(context)
    method['metadata'] = {'type': 'cell_count_method', 'version': '1'}
    return method


def test_a_saved_method_loads_back_as_it_was(tmp_path):
    method = default_cell_count_method()
    method['context']['pixels_per_um'] = 2.5
    path = tmp_path / 'method.json'
    save_cell_count_method(method, path)
    loaded = load_cell_count_method(path)
    assert loaded['context']['pixels_per_um'] == 2.5
    assert loaded['metadata'] == {'type': 'cell_count_method', 'version': '2'}
    assert 'metadata' not in method


def test_a_method_the_count_cannot_use_is_not_saved(tmp_path):
    method = default_cell_count_method()
    method['context']['pixels_per_um'] = 0
    path = tmp_path / 'method.json'
    with pytest.raises(PostProcessingRefusedError):
        save_cell_count_method(method, path)
    assert not path.exists()


@pytest.mark.parametrize(
    ('saved', 'reason', 'words'),
    [
        (_saved(pixels_per_um=0), 'method_invalid', 'context.pixels_per_um'),
        (
            {k: v for k, v in _saved().items() if k != 'segmentation'},
            'method_invalid',
            'segmentation',
        ),
        ({k: v for k, v in _saved().items() if k != 'metadata'}, 'method_unreadable', 'metadata'),
        ('{"context": ', 'method_unreadable', 'not JSON'),
        ('[1, 2]', 'method_unreadable', 'metadata'),
    ],
)
def test_a_file_that_is_not_a_usable_method_is_refused_naming_it(tmp_path, saved, reason, words):
    path = _write(tmp_path / 'method.json', saved)
    with pytest.raises(PostProcessingRefusedError) as refused:
        load_cell_count_method(path)
    assert refused.value.reason == reason
    assert path in str(refused.value)
    assert words in str(refused.value)


def test_a_missing_file_is_refused_naming_it(tmp_path):
    path = str(tmp_path / 'nowhere.json')
    with pytest.raises(PostProcessingRefusedError) as refused:
        load_cell_count_method(path)
    assert refused.value.reason == 'method_unreadable'
    assert path in str(refused.value)


@pytest.fixture
def boundary(monkeypatch):
    """The GUI boundary, inline: run the call, keep what it raised, then redraw."""
    outcomes = []

    def reported(call, redraw, label):
        try:
            call()
        except Exception as e:
            outcomes.append((label, e))
        if redraw is not None:
            redraw()

    monkeypatch.setattr(panel_module, 'run_reported', reported)
    return outcomes


def _panel() -> SimpleNamespace:
    shown = []
    return SimpleNamespace(
        _settings=default_cell_count_method(),
        _set_ui_to_settings=shown.append,
        shown=shown,
    )


def test_the_panel_holds_a_loaded_method_and_shows_it(tmp_path, boundary):
    panel = _panel()
    path = _write(tmp_path / 'method.json', _saved(pixels_per_um=3.0))
    panel_module.CellCountControls.load_method_from_file(panel, path)
    assert boundary == []
    assert panel._settings['context']['pixels_per_um'] == 3.0
    assert panel.shown == [panel._settings]


def test_a_refused_file_leaves_the_panels_method_as_it_was(tmp_path, boundary):
    panel = _panel()
    before = panel._settings
    path = _write(tmp_path / 'method.json', _saved(pixels_per_um=0))
    panel_module.CellCountControls.load_method_from_file(panel, path)
    [(label, refused)] = boundary
    assert label == 'LOAD_CELL_COUNT_METHOD'
    assert isinstance(refused, PostProcessingRefusedError)
    assert panel._settings is before
    assert panel.shown == [before]


def test_the_panel_saves_through_the_owner(tmp_path, boundary):
    panel = _panel()
    panel._settings['context']['pixels_per_um'] = 4.0
    path = str(tmp_path / 'method.json')
    panel_module.CellCountControls.save_method_as(panel, path)
    assert boundary == []
    assert load_cell_count_method(path)['context']['pixels_per_um'] == 4.0
    assert 'metadata' not in panel._settings
