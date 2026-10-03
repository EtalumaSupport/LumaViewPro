# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A typed pixels-per-micron reaches the cell-count method, through its owner.

The box's handler ran on every keystroke and stored a value only when a
preview image was loaded, so with none loaded a typed 2.5 never reached the
method: a folder count ran at 1.0 and its areas were 6.25x off while the box
and the interaction log said 2.5. It also checked the value itself. The box
now hands its text to the method's owner when it is committed, preview or
not; a refused value is reported and the box shows the method's value again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ui.post_processing as panel_module
from modules.exceptions import PostProcessingRefusedError
from modules.post_processing import default_cell_count_method, with_pixels_per_um


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


@pytest.fixture
def records(monkeypatch):
    seen = []
    monkeypatch.setattr(
        panel_module.gui_logger, 'text_input', lambda name, value: seen.append((name, value))
    )
    return seen


def _panel(typed: str, preview=None) -> SimpleNamespace:
    panel = SimpleNamespace(
        ids={'text_cell_count_pixels_per_um_id': SimpleNamespace(text=typed)},
        _settings=default_cell_count_method(),
        _preview_image=preview,
        filter_max_for=[],
    )
    panel.update_filter_max = lambda image: panel.filter_max_for.append(image)
    panel._show_pixels_per_um = lambda: panel_module.CellCountControls._show_pixels_per_um(panel)
    return panel


def _commit(panel) -> None:
    panel_module.CellCountControls.commit_pixels_per_um(panel)


def test_a_typed_scale_reaches_the_method_with_no_preview_loaded(boundary, records):
    panel = _panel('2.5')
    _commit(panel)
    assert boundary == []
    assert panel._settings['context']['pixels_per_um'] == 2.5
    assert panel.ids['text_cell_count_pixels_per_um_id'].text == '2.5'
    assert records == [('CELL_COUNT_PIXELS_PER_UM', '2.5')]
    assert panel.filter_max_for == []


def test_a_whole_number_is_not_recorded_as_a_correction(boundary, records):
    panel = _panel('5')
    _commit(panel)
    assert panel._settings['context']['pixels_per_um'] == 5
    assert records == [('CELL_COUNT_PIXELS_PER_UM', '5')]


@pytest.mark.parametrize('typed', ['0', '-1', 'abc', '', 'nan'])
def test_a_refused_scale_is_reported_and_the_box_shows_the_methods(boundary, records, typed):
    panel = _panel(typed)
    before = panel._settings
    _commit(panel)
    [(label, refused)] = boundary
    assert label == 'CELL_COUNT_PIXELS_PER_UM'
    assert isinstance(refused, PostProcessingRefusedError)
    assert 'pixels_per_um' in str(refused)
    assert panel._settings is before
    assert panel.ids['text_cell_count_pixels_per_um_id'].text == '1.0'
    assert records == [
        ('CELL_COUNT_PIXELS_PER_UM', typed),
        ('CELL_COUNT_PIXELS_PER_UM_APPLIED', '1.0'),
    ]


def test_with_a_preview_loaded_the_filter_maxima_follow_the_scale(boundary, records):
    preview = object()
    panel = _panel('2.0', preview=preview)
    _commit(panel)
    assert panel._settings['context']['pixels_per_um'] == 2.0
    assert panel.filter_max_for == [preview]


def test_the_owner_returns_a_copy_and_leaves_the_method_alone():
    method = default_cell_count_method()
    changed = with_pixels_per_um(method, '3.5')
    assert changed['context']['pixels_per_um'] == 3.5
    assert method['context']['pixels_per_um'] == 1.0
    changed['filters']['area']['max'] = 7
    assert method['filters']['area']['max'] == 100
