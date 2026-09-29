# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step-list edit made while a step move is still in flight wins.

A step click moves the pointer only once its IO-lane task has moved the
stage. Delete and Add do not wait for that: each rewrites the step list,
places the pointer at once and sends its own move. Clicked faster than
the stage travels, the earlier move's completion used to land after the
edit and write back an index read against the old list.

Two results were seen in the simulator. Twenty rapid Deletes emptied a
protocol and a late completion left the pointer at 0 on no steps; Add then
put it past the end, go_to_step turned that into -1 beside a real step,
and the step panel's next redraw asked the protocol for step -1 and took
the app down. Short of emptying the list, a late completion leaves the
pointer on a step the panel is not showing, so the next Delete removes a
step the user did not pick.

These tests drive the real panel handlers, the real go_to_step and a real
Protocol, with the lane held so each move's completion lands when the
test says.
"""

import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())

for _name, _attr in (
    ('kivy.uix.floatlayout', 'FloatLayout'),
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.scrollview', 'ScrollView'),
    ('kivy.uix.widget', 'Widget'),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, _StubWidget)
        sys.modules[_name] = _mod

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
import ui.step_navigation as nav
import ui.ui_helpers as ui_helpers
from modules.protocol import Protocol


REPO = pathlib.Path(__file__).resolve().parent.parent

LAYER_CONFIG = {
    'autofocus': False,
    'false_color': False,
    'illumination_ma': 100.0,
    'gain_db': 0.0,
    'auto_gain': False,
    'exposure_ms': 10.0,
    'sum': 1,
    'acquire': 'image',
    'video_config': {},
    'focus': None,
}


def _insert(protocol, *, x, before_step=None, after_step=None):
    return protocol.insert_step(
        step_name=None,
        layer='BF',
        layer_config=LAYER_CONFIG,
        plate_position={'x': x, 'y': 0.0, 'z': 0.0},
        objective_id='10x Oly',
        stim_configs={},
        before_step=before_step,
        after_step=after_step,
    )


def _protocol(num_steps):
    """A real protocol whose step i stands at X = i, so a step is known by its X."""
    protocol = Protocol(
        tiling_configs_file_loc=REPO / 'data' / 'tiling.json',
        config={'steps': pd.DataFrame(), 'custom_step_count': 0},
    )
    for i in range(num_steps):
        _insert(protocol, x=float(i), after_step=protocol.num_steps() - 1)
    return protocol


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree and the redraw trigger stubbed."""

    def __init__(self, protocol, curr_step):
        self.ids = {
            'step_number_input': SimpleNamespace(text=''),
            'step_total_input': SimpleNamespace(text=''),
            'step_name_input': SimpleNamespace(text='', hint_text=''),
        }
        self._protocol = protocol
        self.curr_step = curr_step

    def update_step_ui(self):
        # Kivy's debounced trigger: the test runs the frame itself.
        pass


@pytest.fixture
def env(monkeypatch):
    held = []

    def submit_gesture(label, *, axes, then, moves, on_moved=None):
        held.append(on_moved)

    loaded = []
    monkeypatch.setattr(ui_helpers, 'submit_gesture', submit_gesture)
    monkeypatch.setattr(nav, '_schedule_ui', lambda fn, timeout: None)
    monkeypatch.setattr(nav, '_load_step_into_layer', lambda **kw: loaded.append(kw['step']['X']))
    monkeypatch.setattr(nav, 'go_to_step_update_ui', lambda step: None)
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)

    def add_step(protocol, *, before_step=None, after_step=None):
        return [_insert(protocol, x=99.0, before_step=before_step, after_step=after_step)]

    ctx = SimpleNamespace(
        settings={'protocol_led_on': False, 'protocol': {'filepath': 'plate.tsv'}},
        motion_settings=SimpleNamespace(ids={}),
        scope=SimpleNamespace(
            protocols=SimpleNamespace(refuse_unaddressable_objectives=lambda objectives: None),
            capabilities=SimpleNamespace(has_turret=False, axes=('X', 'Y', 'Z')),
            imaging=SimpleNamespace(active_cached=False),
            motor_connected=True,
            motion=MagicMock(),
            illumination=MagicMock(),
        ),
        image_settings=SimpleNamespace(layer_lookup=lambda layer: MagicMock()),
        session=SimpleNamespace(
            is_protocol_running=False,
            run_lockout=False,
            add_step=add_step,
        ),
        stage=MagicMock(),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)

    def panel(num_steps, curr_step):
        p = _Panel(_protocol(num_steps), curr_step)
        ctx.motion_settings.ids['protocol_settings_id'] = p
        return p

    def land(count=None):
        """Let the held moves complete, oldest first."""
        while held and (count is None or count > 0):
            on_moved = held.pop(0)
            on_moved()
            if count is not None:
                count -= 1

    return SimpleNamespace(panel=panel, land=land, loaded=loaded)


def _xs(protocol):
    return list(protocol.steps()['X'])


def test_deleting_every_step_then_adding_one_points_at_the_new_step(env):
    panel = env.panel(num_steps=3, curr_step=2)

    for _ in range(3):
        panel.delete_step()
    env.land()
    panel.update_step_ui_immediate()

    assert panel._protocol.num_steps() == 0
    assert panel.curr_step == -1, 'a move that landed after the list emptied moved the pointer'

    panel.insert_step_ex(after_current_step=True)
    env.land()
    panel.update_step_ui_immediate()

    assert _xs(panel._protocol) == [99.0]
    assert panel.curr_step == 0


def test_delete_removes_the_step_the_last_edit_selected(env):
    panel = env.panel(num_steps=5, curr_step=2)

    panel.delete_step()  # removes X=2; the pointer goes to X=1
    panel.delete_step()  # removes X=1; the pointer goes to X=0
    env.land(count=1)  # the first delete's move, to old index 1, completes now

    assert panel.curr_step == 0
    panel.delete_step()
    assert _xs(panel._protocol) == [3.0, 4.0], 'the delete removed a step the user did not pick'


def test_a_superseded_move_does_not_load_its_step_into_the_layer(env):
    panel = env.panel(num_steps=3, curr_step=2)

    panel.delete_step()  # move to index 1 (X=1) held
    panel.delete_step()  # move to index 0 (X=0) held
    env.land()

    assert env.loaded == [0.0]
    assert panel.curr_step == 0


def test_an_add_while_a_move_is_in_flight_keeps_the_added_step(env):
    panel = env.panel(num_steps=3, curr_step=0)

    panel.go_to_step(step_idx=2)  # a click on X=2, held
    panel.insert_step_ex(after_current_step=True)  # the new step sits at index 1
    env.land(count=1)  # the click's move completes after the Add

    assert _xs(panel._protocol) == [0.0, 99.0, 1.0, 2.0]
    assert panel.curr_step == 1, 'the late click moved the pointer off the added step'


def test_a_move_for_a_replaced_protocol_does_not_land_on_the_new_one(env):
    panel = env.panel(num_steps=3, curr_step=0)
    # Built the same way, so its step list is at the same revision: only
    # being a different protocol tells the two apart.
    replacement = _protocol(3)
    assert replacement.step_list_revision == panel._protocol.step_list_revision

    panel.go_to_step(step_idx=2)  # a click on the old protocol's X=2, held
    panel.new_protocol_ex(replacement)
    env.land(count=1)

    assert panel._protocol is replacement
    assert panel.curr_step == 0, 'a move sent for the old protocol moved the new one'


def test_a_move_the_list_did_not_change_under_still_lands(env):
    panel = env.panel(num_steps=3, curr_step=0)

    panel.go_to_step(step_idx=2)
    assert panel.curr_step == 0, 'the pointer moves only once the stage has'
    env.land()

    assert panel.curr_step == 2
    assert env.loaded == [2.0]
