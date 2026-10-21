# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The cell-count panel writes its method only from a person's release of a slider.

Two ways the panel wrote bounds nobody set: showing a method ran the area and
perimeter handlers, which wrote the sliders' positions back as bounds; and the
six range sliders were bound to on_touch_up, which Kivy delivers for a touch
anywhere in the window, so every button click recorded six slider moves and
rewrote six bounds. A slider at its stop is an open bound. The kv binding
to on_release is pinned in
tests/guards/test_the_range_sliders_commit_on_their_own_release.py.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

import ui.post_processing as panel_module
from modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds

REPO = pathlib.Path(__file__).resolve().parent.parent
_RANGE_SLIDERS = (
    'area',
    'perimeter',
    'sphericity',
    'min_intensity',
    'mean_intensity',
    'max_intensity',
)


def _slider(lo=0, hi=100):
    return SimpleNamespace(min=lo, max=hi, value=(lo, hi))


@pytest.fixture
def size_scale(tmp_path):
    """The session's size scale with no preview image, as the panel starts with it."""
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session.post_processing.cell_count_size_scale(CellCountMethod())
    session.shutdown()


def _panel(method, size_scale):
    ids = {f'slider_cell_count_{name}_id': _slider() for name in _RANGE_SLIDERS}
    ids['slider_cell_count_sphericity_id'] = _slider(0, 1.0)
    ids.update(
        slider_cell_count_threshold_id=SimpleNamespace(value=0),
        text_cell_count_pixels_per_um_id=SimpleNamespace(text='x'),
        cell_count_fluorescent_mode_id=SimpleNamespace(active=None),
        label_cell_count_area_id=SimpleNamespace(text=''),
        label_cell_count_perimeter_id=SimpleNamespace(text=''),
    )
    panel = SimpleNamespace(
        ids=SimpleNamespace(**ids),
        _method=method,
        _size_scale=size_scale,
        _regenerate_image_preview=lambda: None,
        ENABLE_PREVIEW_AUTO_REFRESH=False,
    )
    cls = panel_module.CellCountControls
    for name in ('_show_size_filters', '_take_filter'):
        setattr(panel, name, getattr(cls, name).__get__(panel))
    return panel


class _Ids(dict):
    __getattr__ = dict.__getitem__


def test_showing_a_method_writes_nothing_and_shows_open_bounds_at_the_stops(
    monkeypatch, size_scale
):
    records = []
    monkeypatch.setattr(panel_module.gui_logger, 'slider', lambda *a: records.append(a))
    method = CellCountMethod(filters=CellCountFilters(perimeter=FilterBounds(20, None)))
    panel = _panel(method, size_scale)
    panel.ids = _Ids(vars(panel.ids))

    panel_module.CellCountControls._set_ui_to_method(panel, method)

    assert panel._method is method, 'showing the method changed it'
    assert records == [], 'showing the method recorded a slider move'
    assert panel.ids['slider_cell_count_area_id'].value == (0, 100)
    assert panel.ids['label_cell_count_area_id'].text == 'any \u03bcm\u00b2'
    assert panel.ids['slider_cell_count_perimeter_id'].value[1] == 100
    assert panel.ids['label_cell_count_perimeter_id'].text == '\u2265 20 \u03bcm'
    assert panel.ids['text_cell_count_pixels_per_um_id'].text == ''


_RELEASE_PROBE = r"""
import os, sys
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
sys.path.insert(0, sys.argv[1])
from types import SimpleNamespace
from ui.range_slider import RangeSlider
released = []
slider = RangeSlider()
slider.bind(on_release=lambda *_: released.append(True))
slider.on_touch_up(SimpleNamespace(grab_current=None, ungrab=lambda w: None))
print(len(released))
slider.on_touch_up(SimpleNamespace(grab_current=slider, ungrab=lambda w: None))
print(len(released))
"""


@pytest.mark.needs_display
def test_a_range_slider_releases_only_the_touch_it_grabbed(tmp_path):
    # Real Kivy in a child process: the test environment stands its widgets in.
    # Kivy writes its config and icons into KIVY_HOME, so the child gets one
    # of its own rather than the shared ~/.kivy.
    done = subprocess.run(
        [sys.executable, '-c', _RELEASE_PROBE, str(REPO)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, 'KIVY_HOME': str(tmp_path)},
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    elsewhere, grabbed = done.stdout.split()[-2:]
    assert elsewhere == '0', 'a touch elsewhere released the slider'
    assert grabbed == '1'


def test_a_released_size_slider_writes_open_bounds_at_its_stops_and_records_once(
    monkeypatch, size_scale
):
    records = []
    monkeypatch.setattr(panel_module.gui_logger, 'slider', lambda *a: records.append(a))
    panel = _panel(CellCountMethod(), size_scale)
    panel.ids = _Ids(vars(panel.ids))
    panel._size_bounds_from_slider = (
        panel_module.CellCountControls._size_bounds_from_slider.__get__(panel)
    )

    panel.ids['slider_cell_count_area_id'].value = (10, 100)
    panel_module.CellCountControls.slider_adjustment_area(panel)

    assert panel._method.filters.area == FilterBounds(10, None)
    assert records == [('CELL_COUNT_AREA_RANGE', '\u2265 10')]
    assert panel.ids['label_cell_count_area_id'].text == '\u2265 10 \u03bcm\u00b2'

    panel.ids['slider_cell_count_area_id'].value = (0, 100)
    panel_module.CellCountControls.slider_adjustment_area(panel)
    assert panel._method.filters.area == FilterBounds(None, None)
