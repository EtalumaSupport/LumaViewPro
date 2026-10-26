# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The layer drawers draw the Session's shown channel, whoever changed it.

The open drawer was the GUI's answer to "which channel is current", and a
drawer opening turned the other LEDs off and pushed its layer's settings
itself. The Session now holds the shown channel; the listener bridge hears
each change -- a person's show, a step gone to, a run's step -- and the
drawers draw it. A run is not special: the bridge draws during one as at any
other time.
"""

from types import SimpleNamespace

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from ui.listener_bridge import UIListenerBridge
from tests.stock_channels import LED0, LED1


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(microscope='LS850'), simulate=True)
    yield built
    built.shutdown()


def _bridge(session, *, ready=True):
    drawn = []
    # a stand-in by design: Kivy is stubbed in the test process; the drawers'
    # draw is the subject's interchangeable collaborator
    led_button = SimpleNamespace(ids={'enable_led_btn': SimpleNamespace(state='normal')})
    image_settings = SimpleNamespace(
        draw_shown_channel=lambda: drawn.append(session.shown_channel.channel),
        layer_lookup=lambda layer: led_button,
    )
    ctx = SimpleNamespace(ready=ready, session=session, image_settings=image_settings)
    bridge = UIListenerBridge(scope=session.scope, ctx=ctx, stage=SimpleNamespace())
    bridge.register_all()
    return drawn


def test_each_change_of_the_shown_channel_draws_the_drawers(session):
    drawn = _bridge(session)

    session.show_channel(LED1)
    session.light_shown_channel(True)
    session.show_channel(LED0)

    assert drawn == [LED1, LED1, LED0]


def test_a_show_that_changes_nothing_draws_nothing(session):
    drawn = _bridge(session)

    session.show_channel('BF')

    assert drawn == []


def test_before_the_window_is_ready_nothing_is_drawn(session):
    drawn = _bridge(session, ready=False)

    session.show_channel(LED1)

    assert drawn == []
