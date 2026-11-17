# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A setting toggle or picker is one row of one table: one handler pair writes it, one show draws it.

Each setting control was written three times -- a kv row, a handler, a line
in its panel's show -- and the handlers disagreed: four toggles wrote bare,
so a refusal escaped the press unreported and the toggle kept showing what
the store had refused; the tiling overlap's show wrote a stored 12.5 back as
12, read from the label it had rounded. The rows now live in
``ui/setting_controls.py`` and ``SettingControls`` (``ui/ui_helpers.py``) is
their one write and one show.

Kivy is stubbed in the test process: the real mixin runs over stand-in
widgets shaped like the kv's, against a simulated session.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.settings_paths import PREFERENCES
from tests import gui_logging_census
from tests.settings_fixtures import complete_settings
from ui.advanced_settings import AdvancedSettings
from ui.listener_bridge import _routed
from ui.microscope_settings import MicroscopeSettings
from ui.post_processing import QuickEnhanceControls
from ui.setting_controls import Picker, Toggle
from ui.ui_helpers import SettingControls
from ui.zstack import ZStack

REPO = pathlib.Path(__file__).resolve().parents[1]

# Each panel's kv rule, by its class's name, and the rows the class names.
PANELS = {
    cls.__name__: cls.setting_controls
    for cls in (AdvancedSettings, MicroscopeSettings, ZStack, QuickEnhanceControls)
}


class _Picker:
    """A PickSpinner's text: a write of a different text runs its ``on_text``, as the kv binds it."""

    def __init__(self, panel, widget_id):
        self._panel = panel
        self._id = widget_id
        self._text = 'Select'
        self.values = []
        self.log_name = widget_id.upper()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        if value != self._text:
            self._text = value
            self._panel.pick_control(self._id)


class _Panel(SettingControls):
    """The mixin over one panel's rows, each widget a stand-in of its kind."""

    def __init__(self, rows):
        # a stand-in by design: Kivy is stubbed in the test process; the subject is the mixin
        self.setting_controls = rows
        self.ids = {
            row.widget: _Picker(self, row.widget)
            if isinstance(row, Picker)
            else SimpleNamespace(state='normal')
            for row in rows
        }


@pytest.fixture
def session(tmp_path, monkeypatch):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(session=s, settings=s.settings, update_settings=s.update_settings),
    )
    stored = []
    s.add_settings_listener(stored.append)
    s.stored = stored
    yield s
    s.remove_settings_listener(stored.append)
    s.shutdown()


def test_a_refused_toggle_is_told_and_shows_the_stored_value(session, centre_posts):
    """Each toggle a held control session refuses is reported, and shows what is stored.

    The preferences pass a held session (``PREFERENCES``), so they have no
    refusal to show.
    """
    session.take_control('client')
    refusable = [
        row
        for rows in PANELS.values()
        for row in rows
        if isinstance(row, Toggle) and row.path not in PREFERENCES
    ]
    panel = _Panel(tuple(refusable))
    panel.show_controls()
    before = {row.widget: panel.ids[row.widget].state for row in refusable}

    for row in refusable:
        widget = panel.ids[row.widget]
        widget.state = 'normal' if widget.state == 'down' else 'down'
        panel.toggle_control(row.widget)

    assert session.stored == []
    assert {row.widget: panel.ids[row.widget].state for row in refusable} == before
    assert len(centre_posts) == len(refusable), [p.title for p in centre_posts]


def test_the_table_imports_no_kivy():
    """Another toolkit's panel reads the same rows: importing the table loads no Kivy."""
    probe = (
        'import sys\n'
        f'sys.path[:0] = [{str(REPO)!r}, {str(REPO / "src")!r}]\n'
        'import ui.setting_controls\n'
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'kivy')\n"
        'assert not loaded, loaded\n'
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_every_row_is_bound_in_its_panels_kv_and_every_binding_is_a_row():
    """A toggle's release and a picker's text call its handler with the row's own id."""
    handler = {Toggle: ('toggle_control', 'on_release'), Picker: ('pick_control', 'on_text')}
    expected = {
        (root, row.widget, *handler[type(row)]) for root, rows in PANELS.items() for row in rows
    }
    bound = {
        (block['root'], block['id'], method, block['event'])
        for label, text in gui_logging_census.kv_sources()
        for block in gui_logging_census._blocks(label, text)
        for method in block['root_methods']
        if method in ('toggle_control', 'pick_control') and block['args'] == [block['id']]
    }
    every_binding = {
        (block['root'], block['handler'])
        for label, text in gui_logging_census.kv_sources()
        for block in gui_logging_census._blocks(label, text)
        if {'toggle_control', 'pick_control'} & set(block['root_methods'])
    }
    assert bound == expected
    assert len(every_binding) == len(expected), sorted(every_binding)


def test_every_row_is_redrawn_when_its_setting_changes():
    """A row's setting written by anyone -- a REST client among them -- reaches the panel's show."""
    unrouted = [row.path for rows in PANELS.values() for row in rows if not _routed(row.path)]
    assert unrouted == []


def test_a_tiling_overlap_off_the_choices_is_shown_as_itself_and_kept(session):
    """A REST client's 12.5 survives Advanced Settings opening: shown 12.5%, still stored 12.5."""
    session.update_settings('tiling_overlap_percent', 12.5)
    session.stored.clear()
    panel = _Panel(AdvancedSettings.setting_controls)

    panel.show_controls()

    assert panel.ids['tiling_overlap_spinner'].text == '12.5%'
    assert session.setting('tiling_overlap_percent') == 12.5
    assert session.stored == []


def test_a_show_from_a_complete_store_writes_nothing(session):
    """Every panel's show sets each widget from the store, and its handlers take that as no pick."""
    panels = {name: _Panel(rows) for name, rows in PANELS.items()}

    for panel in panels.values():
        panel.show_controls()

    assert session.stored == []
    shown = {
        name: {wid: getattr(w, 'text', None) or w.state for wid, w in panel.ids.items()}
        for name, panel in panels.items()
    }
    assert 'Select' not in str(shown), shown


def test_a_toggle_released_and_a_choice_picked_are_stored_as_their_values(session, caplog):
    """Every toggle stores its new state, and every choice of every picker its value, not its label.

    A pick is recorded by its PickSpinner, never by the handler, which runs
    on a value change as well; a toggle is recorded by its handler, on the
    release only a person makes.
    """
    import logging

    wrong = []
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        for rows in PANELS.values():
            panel = _Panel(rows)
            panel.show_controls()
            for row in rows:
                widget = panel.ids[row.widget]
                if isinstance(row, Toggle):
                    wanted = widget.state != 'down'
                    widget.state = 'down' if wanted else 'normal'
                    panel.toggle_control(row.widget)
                    if session.setting(row.path) != wanted:
                        wrong.append((row.path, wanted, session.setting(row.path)))
                    continue
                for choice in row.choices:
                    widget.text = row.label(choice)
                    if session.setting(row.path) != choice:
                        wrong.append((row.path, choice, session.setting(row.path)))

    toggles = [row.record for rows in PANELS.values() for row in rows if isinstance(row, Toggle)]
    recorded = [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']
    assert wrong == []
    assert [line.split()[1] for line in recorded] == toggles, recorded


def test_a_pickers_show_offers_every_choice_and_shows_the_stored_ones_label(session):
    panels = {name: _Panel(rows) for name, rows in PANELS.items()}
    shown, wanted = {}, {}
    for panel in panels.values():
        panel.show_controls()
        for row in panel.setting_controls:
            if isinstance(row, Picker):
                widget = panel.ids[row.widget]
                shown[row.widget] = (widget.values, widget.text)
                wanted[row.widget] = (
                    [row.label(choice) for choice in row.choices],
                    row.label(session.setting(row.path)),
                )
    assert shown == wanted
