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
from modules.objectives_loader import ObjectiveLoader
from modules.exceptions import ProtocolError
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
        config={'steps': pd.DataFrame(), 'custom_step_count': 0, 'period': None, 'duration': None},
    )
    for i in range(num_steps):
        _insert(protocol, x=float(i), after_step=protocol.num_steps() - 1)
    return protocol


def _drawers(opened):
    """The layer drawers' lookup, with only ``opened`` open."""
    return lambda layer: SimpleNamespace(collapse=layer != opened)


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree and the redraw trigger stubbed."""

    def __init__(self, protocol, curr_step):
        self.ids = {
            'step_number_input': SimpleNamespace(text=''),
            'step_total_input': SimpleNamespace(text=''),
            'step_name_input': SimpleNamespace(text='', hint_text=''),
            'capture_period': SimpleNamespace(text=''),
            'capture_dur': SimpleNamespace(text=''),
        }
        self._protocol = protocol
        self.curr_step = curr_step

    def update_step_ui(self):
        # Kivy's debounced trigger: the test runs the frame itself.
        pass


@pytest.fixture
def env(monkeypatch):
    held = []

    def submit_move(label, *, axes, call, on_moved=None):
        held.append((call, on_moved))

    # The X of each step the Session was asked to go to, in the order the
    # lane ran them: a superseded click never reaches it.
    loaded = []
    monkeypatch.setattr(ui_helpers, 'submit_move', submit_move)
    monkeypatch.setattr(nav, '_schedule_ui', lambda fn, timeout: None)
    monkeypatch.setattr(nav, 'go_to_step_update_ui', lambda step: None)
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'text_input', lambda *a, **kw: None)

    def add_step(protocol, *, before_step=None, after_step=None):
        return [_insert(protocol, x=99.0, before_step=before_step, after_step=after_step)]

    def update_step(protocol, step_idx, *, layer, label=None):
        protocol.modify_step(
            step_idx=step_idx,
            layer=layer,
            layer_config=LAYER_CONFIG,
            plate_position={'x': 99.0, 'y': 0.0, 'z': 0.0},
            objective_id='10x Oly',
            stim_configs={},
            label=label,
        )
        return protocol.step(idx=step_idx)['Name']

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
            objective_helper=ObjectiveLoader(),
        ),
        image_settings=SimpleNamespace(
            layer_lookup=lambda layer: MagicMock(),
            accordion_item_lookup=_drawers(opened=None),
        ),
        session=SimpleNamespace(
            is_protocol_running=False,
            run_lockout=False,
            go_to_step=lambda protocol, step_idx: loaded.append(protocol.step(idx=step_idx)['X']),
            add_step=add_step,
            delete_step=lambda protocol, step_idx: protocol.delete_step(step_idx=step_idx),
            rename_step=lambda protocol, step_idx, name: protocol.modify_name(
                step_idx=step_idx, step_name=name
            ),
            update_step=update_step,
        ),
        stage=MagicMock(),
    )
    # The GUI reads the scope through the main widget as well as directly.
    ctx.lumaview = SimpleNamespace(scope=ctx.scope)
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)

    def panel(num_steps, curr_step):
        p = _Panel(_protocol(num_steps), curr_step)
        ctx.motion_settings.ids['protocol_settings_id'] = p
        return p

    def land(count=None):
        """Let the held moves run and complete, oldest first."""
        while held and (count is None or count > 0):
            call, on_moved = held.pop(0)
            call()
            on_moved()
            if count is not None:
                count -= 1

    return SimpleNamespace(panel=panel, land=land, loaded=loaded, ctx=ctx)


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


def test_a_delete_the_protocol_refuses_is_reported_and_changes_nothing(env, monkeypatch):
    """A pointer past the end asks the protocol for a step it has not got.

    The protocol's refusal is the answer; the panel hands it to the one
    reporter under the gesture's label and writes no popup of its own.
    """
    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda exc, **kw: reported.append((type(exc).__name__, kw['category'])),
    )
    panel = env.panel(num_steps=3, curr_step=7)

    panel.delete_step()

    assert reported == [('StepNotFoundError', 'UI:DELETE_STEP')]
    assert _xs(panel._protocol) == [0.0, 1.0, 2.0]
    assert panel.curr_step == 7


@pytest.fixture
def refusals(monkeypatch):
    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda exc, **kw: reported.append((type(exc).__name__, kw['category'])),
    )
    return reported


def test_a_delete_on_an_empty_protocol_is_the_protocols_refusal(env, refusals):
    """The panel does not decide that there is nothing to delete; the API says so."""
    panel = env.panel(num_steps=0, curr_step=-1)

    panel.delete_step()

    assert refusals == [('StepNotFoundError', 'UI:DELETE_STEP')]
    assert panel.curr_step == -1


def test_a_name_with_nothing_to_keep_is_refused_and_the_step_keeps_its_name(env, refusals):
    panel = env.panel(num_steps=2, curr_step=1)
    panel._protocol.modify_name(step_idx=1, step_name='center')
    before = panel._protocol.step(1)['Name']
    panel.ids['step_name_input'].text = '!!!'  # what the person typed

    panel.step_name_validation('!!!')
    panel.update_step_ui_immediate()

    assert refusals == [('StepEditRefusedError', 'UI:RENAME_STEP')]
    assert panel._protocol.step(1)['Name'] == before
    assert panel.ids['step_name_input'].text == 'center', 'the field still shows the refused text'


def test_a_blank_name_field_keeps_the_name_without_asking_the_protocol(env, refusals):
    panel = env.panel(num_steps=2, curr_step=1)
    before = panel._protocol.step(1)['Name']

    panel.step_name_validation('  ')

    assert refusals == []
    assert panel._protocol.step(1)['Name'] == before
    assert panel.ids['step_name_input'].text == ''


def test_a_blank_field_on_a_named_step_shows_its_name_again(env, refusals):
    """Blank keeps the name, so the field shows it, not the auto name's hint."""
    panel = env.panel(num_steps=2, curr_step=1)
    panel._protocol.modify_name(step_idx=1, step_name='center')
    panel.ids['step_name_input'].text = ''

    panel.step_name_validation('')

    assert refusals == []
    assert panel._protocol.step(1)['Label'] == 'center'
    assert panel.ids['step_name_input'].text == 'center'


def test_a_rename_shows_the_label_the_protocol_kept(env, refusals):
    panel = env.panel(num_steps=2, curr_step=1)

    panel.ids['step_name_input'].text = 'my step!'  # what the person typed
    panel.step_name_validation('my step!')
    panel.update_step_ui_immediate()

    assert refusals == []
    assert panel._protocol.step(1)['Label'] == 'mystep'
    assert panel.ids['step_name_input'].text == 'mystep'


def test_update_with_a_name_of_nothing_to_keep_is_refused(env):
    """The name field reaches Update as typed; the protocol refuses it, not the panel."""
    panel = env.panel(num_steps=2, curr_step=1)
    before = panel._protocol.step(1)['Name']

    with pytest.raises(ProtocolError, match='at least one letter'):
        panel.modify_step_ex('BF', '!!!')

    assert panel._protocol.step(1)['Name'] == before


def test_update_with_a_blank_name_field_keeps_the_label(env):
    panel = env.panel(num_steps=2, curr_step=1)
    panel._protocol.modify_name(step_idx=1, step_name='mine')

    panel.modify_step_ex('BF', '')

    assert panel._protocol.step(1)['Label'] == 'mine'


def test_an_edit_that_leaves_a_step_invalid_gets_no_popup_from_the_panel(env, monkeypatch):
    """The API notices an invalid step; the panel composes nothing of its own."""
    import ui.notification_popup as popup

    composed = []
    monkeypatch.setattr(popup, 'show_notification_popup', lambda **kw: composed.append(kw))
    panel = env.panel(num_steps=2, curr_step=0)
    panel._protocol.steps().at[0, 'Exposure'] = 0.0

    panel.insert_step_ex(after_current_step=True)

    assert composed == []


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


def _add_bf_and_blue(protocol, *, before_step=None, after_step=None):
    names = []
    for layer in ('BF', 'Blue'):
        names.append(
            protocol.insert_step(
                step_name=None,
                layer=layer,
                layer_config=LAYER_CONFIG,
                plate_position={'x': 99.0, 'y': 0.0, 'z': 0.0},
                objective_id='10x Oly',
                stim_configs={},
                before_step=before_step,
                after_step=after_step,
            )
        )
        inserted_at = before_step if before_step is not None else after_step + 1
        before_step, after_step = None, inserted_at
    return names


@pytest.mark.parametrize(
    ('opened', 'selected_color'),
    [('BF', 'BF'), ('Blue', 'Blue'), ('Red', 'BF'), (None, 'BF')],
)
def test_add_goes_to_the_new_step_of_the_channel_being_viewed(env, opened, selected_color):
    """Adding BF and Blue while viewing BF stays on BF; nothing on screen changes.

    Going to the last channel added lit Blue and moved to its focus on the
    bench (LS850T, 2026-10-02). A viewed channel that acquires nothing gets
    the first step added.
    """
    env.ctx.session.add_step = _add_bf_and_blue
    env.ctx.image_settings.accordion_item_lookup = _drawers(opened)
    panel = env.panel(num_steps=2, curr_step=0)

    panel.insert_step_ex(after_current_step=True)

    assert list(panel._protocol.steps()['Color']) == ['BF', 'BF', 'Blue', 'BF']
    assert panel.curr_step in (1, 2)
    assert panel._protocol.step(panel.curr_step)['Color'] == selected_color
