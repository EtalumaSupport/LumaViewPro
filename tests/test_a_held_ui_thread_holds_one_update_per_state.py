# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A burst of state changes leaves one UI update waiting per state, the newest.

The listener bridge posted a callback per position poll, and the progress
proxies one per write, so a UI thread held up during a run held one per
event, without bound. Each now posts the state it shows
(``schedule_ui_latest``), keyed by its owner: a held-up UI thread holds one
entry per position axis, turret, LED address, shown channel, settings
redraw and popup attribute, however many changes came, and runs the newest.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.kivy_utils import UiDispatcher, ui_backlog
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0, LED1
from ui.listener_bridge import UIListenerBridge
from ui.progress_popup import _PopupProxy

_BURST = 500


class _Held:
    """A UI thread that is held up: what the dispatcher is asked to run waits until the test runs it."""

    def __init__(self):
        self.queue = []

    def schedule(self, func, timeout):
        self.queue.append(func)

    def run(self):
        queue, self.queue = self.queue, []
        for func in queue:
            func(0)


@pytest.fixture
def held():
    ui = _Held()
    ScopeSession.set_ui_dispatcher(UiDispatcher(schedule=ui.schedule, thread=None))
    yield ui
    ScopeSession.set_ui_dispatcher(None)


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(microscope='LS850'), simulate=True)
    yield built
    built.shutdown()


def _bridge(session, drawn):
    # a stand-in by design: Kivy is stubbed in the test process; the widgets
    # are the bridge's interchangeable collaborators, recording each draw.
    def layer(name):
        return SimpleNamespace(
            ids={'enable_led_btn': SimpleNamespace(state='normal')},
            sync_widgets_from_settings=lambda: drawn.append(('settings', name)),
        )

    z_ctrl = SimpleNamespace(
        show_z=lambda um: drawn.append(('z', um)),
        show_turret_state=lambda prompt: drawn.append(('turret',)),
    )
    ctx = SimpleNamespace(
        ready=True,
        session=session,
        image_settings=SimpleNamespace(
            draw_shown_channel=lambda: drawn.append(('shown_channel',)),
            layer_lookup=lambda layer: layer_of(layer),
        ),
        motion_settings=SimpleNamespace(
            ids={'verticalcontrol_id': z_ctrl},
            update_xy_stage_control_gui=lambda: drawn.append(('xy',)),
        ),
    )
    layers = {}

    def layer_of(name):
        return layers.setdefault(name, layer(name))

    stage = SimpleNamespace(draw_labware=lambda: drawn.append(('labware',)))
    return UIListenerBridge(scope=session.scope, ctx=ctx, stage=stage)


def test_a_burst_of_each_state_leaves_one_update_waiting_per_key(session, held):
    drawn = []
    bridge = _bridge(session, drawn)
    first, second = (r.address for r in session.scope.illumination.get_led_states()[:2])
    for i in range(_BURST):
        bridge._on_position_change('X', SimpleNamespace(position_um=i))
        bridge._on_position_change('Y', SimpleNamespace(position_um=i))
        bridge._on_position_change('Z', SimpleNamespace(position_um=i))
        bridge._on_turret_change(None)
        bridge._on_led_state_changed(first, None, None, 0)
        bridge._on_led_state_changed(second, None, None, 0)
        bridge._on_shown_channel_changed(LED0, False)
        bridge._on_settings_changed((f'{LED0}.gain_db',))
    # XY, Z, the turret, two LED addresses, the shown channel, the settings.
    assert ui_backlog() == 7
    held.run()
    assert sorted(drawn) == sorted(
        [
            ('xy',),
            ('labware',),
            ('z', _BURST - 1),
            ('turret',),
            ('shown_channel',),
            ('settings', LED0),
        ]
    )


def test_a_setting_changed_while_its_redraw_takes_the_last_is_redrawn(session, held):
    # The second change's paths arrive while the UI thread takes the pending
    # set for the first redraw: a path added to the set already taken is
    # never drawn.
    drawn = []
    bridge = _bridge(session, drawn)
    bridge._on_settings_changed((f'{LED0}.gain_db',))

    def arriving_as_the_redraw_runs():
        held.run()
        yield f'{LED1}.gain_db'

    bridge._on_settings_changed(arriving_as_the_redraw_runs())
    held.run()
    assert drawn == [('settings', LED0), ('settings', LED1)]


def test_a_popups_writes_leave_one_per_attribute_and_a_write_after_dismiss_runs_after_it(held):
    shown = []

    class _Popup:
        # a stand-in by design: the popup is a Kivy widget; its writes and dismiss are recorded.
        def __setattr__(self, name, value):
            shown.append((name, value))

        def dismiss(self):
            shown.append('dismissed')

    proxy = _PopupProxy(_Popup())
    for i in range(_BURST):
        proxy.progress = i
        proxy.text = f'step {i}'
    assert ui_backlog() == 2
    proxy.dismiss()
    proxy.text = 'after'
    held.run()
    assert shown == [('progress', _BURST - 1), 'dismissed', ('text', 'after')]
