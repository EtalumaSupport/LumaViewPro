# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cell-count method file is read, checked and written by the method's owner.

The panel read the file itself, checked only that a metadata entry named a
type and a version (raising a bare Exception, with nothing around it to
report it), and stored the dict before reading its fields -- so a file with
pixels-per-micron 0 loaded and counted nothing, and a file missing a field
left the panel holding a method it could not use. The Session's
``load_cell_count_method`` and ``save_cell_count_method`` now read and
write the file, the load answering a ``CellCountMethod`` the count can use,
and the panel stores a method only from that answer.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

import ui.post_processing as panel_module
from modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds, saved_form
from modules.exceptions import LiveFolderPathRefusedError, PostProcessingRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def post(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session.post_processing
    session.shutdown()


def _write(path, saved) -> str:
    path.write_text(json.dumps(saved) if not isinstance(saved, str) else saved)
    return str(path)


def _saved(**context) -> dict:
    saved = saved_form(CellCountMethod())
    saved['context'].update(context)
    saved['metadata'] = {'type': 'cell_count_method', 'version': '2'}
    return saved


def test_a_saved_method_loads_back_as_it_was(post, tmp_path):
    method = CellCountMethod(
        pixels_per_um=2.5,
        fluorescent_mode=False,
        threshold_pct=35,
        filters=CellCountFilters(
            area_um2=FilterBounds(10, 500), intensity_mean_pct=FilterBounds(5, 90)
        ),
    )
    path = tmp_path / 'method.json'
    post.save_cell_count_method(method, path)
    assert post.load_cell_count_method(path) == method
    assert json.loads(path.read_text())['metadata'] == {'type': 'cell_count_method', 'version': '2'}


def test_the_file_keeps_the_form_earlier_releases_read(post, tmp_path):
    path = tmp_path / 'method.json'
    post.save_cell_count_method(CellCountMethod(pixels_per_um=2.5), path)
    saved = json.loads(path.read_text())
    assert saved['context'] == {'pixels_per_um': 2.5, 'fluorescent_mode': True}
    assert saved['segmentation'] == {'algorithm': 'initial', 'parameters': {'threshold': 20}}
    assert saved['filters']['intensity']['mean'] == {'min': 0, 'max': 100}


def test_a_folder_at_the_path_is_refused_and_left_as_it_was(post, tmp_path):
    folder = tmp_path / 'method.json'
    folder.mkdir()
    (folder / 'kept.txt').write_text('kept')
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        post.save_cell_count_method(CellCountMethod(), folder)
    assert refused.value.reason == 'not_a_file'
    assert [p.name for p in folder.iterdir()] == ['kept.txt']


def _without(saved: dict, *path: str) -> dict:
    holder = saved
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    return saved


# parametrize-justified: each case writes its own file and is refused for its own reason
@pytest.mark.parametrize(
    ('saved', 'reason', 'words'),
    [
        (_saved(pixels_per_um=0), 'method_invalid', 'pixels_per_um'),
        (_saved(fluorescent_mode='yes'), 'method_invalid', 'fluorescent_mode'),
        (_without(_saved(), 'segmentation'), 'method_invalid', 'segmentation'),
        (
            _without(_saved(), 'segmentation', 'algorithm'),
            'method_invalid',
            'segmentation.algorithm',
        ),
        (
            _without(_saved(), 'filters', 'intensity', 'mean'),
            'method_invalid',
            'filters.intensity.mean',
        ),
        (_without(_saved(), 'filters', 'area', 'max'), 'method_invalid', 'filters.area.max'),
        ({k: v for k, v in _saved().items() if k != 'metadata'}, 'method_unreadable', 'metadata'),
        ('{"context": ', 'method_unreadable', 'not JSON'),
        ('[1, 2]', 'method_unreadable', 'metadata'),
    ],
)
def test_a_file_that_is_not_a_usable_method_is_refused_naming_it(
    post, tmp_path, saved, reason, words
):
    path = _write(tmp_path / 'method.json', saved)
    with pytest.raises(PostProcessingRefusedError) as refused:
        post.load_cell_count_method(path)
    assert refused.value.reason == reason
    assert path in str(refused.value)
    assert words in str(refused.value)


def test_a_missing_file_is_refused_naming_it(post, tmp_path):
    path = str(tmp_path / 'nowhere.json')
    with pytest.raises(PostProcessingRefusedError) as refused:
        post.load_cell_count_method(path)
    assert refused.value.reason == 'method_unreadable'
    assert path in str(refused.value)


@pytest.fixture
def boundary(monkeypatch, post):
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
    monkeypatch.setattr(
        panel_module._app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(post_processing=post))
    )
    return outcomes


def _panel(method: CellCountMethod | None = None) -> SimpleNamespace:
    shown = []
    panel = SimpleNamespace(
        ids={
            'slider_cell_count_area_id': SimpleNamespace(max=100),
            'slider_cell_count_perimeter_id': SimpleNamespace(max=100),
        },
        _method=method or CellCountMethod(),
        _preview_source_image=None,
        _preview_pixel_size_um=None,
        _size_scale=None,
        _set_ui_to_method=shown.append,
        shown=shown,
    )
    panel._take_size_scale = panel_module.CellCountControls._take_size_scale.__get__(panel)
    return panel


def test_the_panel_holds_a_loaded_method_and_shows_it_at_its_scale(tmp_path, boundary):
    panel = _panel()
    path = _write(tmp_path / 'method.json', _saved(pixels_per_um=3.0))
    panel_module.CellCountControls.load_method_from_file(panel, path)
    assert boundary == []
    assert panel._method == CellCountMethod(pixels_per_um=3.0)
    assert panel.shown == [panel._method]
    assert panel._size_scale.pixels_per_um == 3.0


def test_a_refused_file_leaves_the_panels_method_as_it_was(tmp_path, boundary):
    panel = _panel()
    before = panel._method
    path = _write(tmp_path / 'method.json', _saved(pixels_per_um=0))
    panel_module.CellCountControls.load_method_from_file(panel, path)
    [(label, refused)] = boundary
    assert label == 'LOAD_CELL_COUNT_METHOD'
    assert refused.reason == 'method_invalid'
    assert panel._method is before
    assert panel.shown == [before]


def test_the_panel_saves_through_the_session(post, tmp_path, boundary):
    method = dataclasses.replace(CellCountMethod(), pixels_per_um=4.0)
    path = str(tmp_path / 'method.json')
    panel_module.CellCountControls.save_method_as(_panel(method), path)
    assert boundary == []
    assert post.load_cell_count_method(path) == method
