# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer slider writes what the API accepts, run or no run; a display of a step writes nothing.

A write the API refuses stores nothing and tells no settings listener, so
the slider and its box are drawn from the store once the write has
answered: a drag refused because a control session was taken after it
began snaps back.

Before, the illumination, gain and exposure slider handlers returned
without writing whenever ``session.run_lockout`` held, a decision the GUI
made where the API accepts the write: ``update_settings`` takes a layer's
gain while a run holds the scope. The GUI now decides nothing there. What
keeps a run's per-step display from writing is the guard every
programmatic slider write sets, ``_initializing`` (``set_step_state``), and
it still holds with the returns gone.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules.exceptions import UNDER_CONTROL_TITLE, ControlSessionRefusedError
from tests.gesture_fakes import inline_submit_reported
from ui.channel_control import ChannelControl
from tests.stock_channels import LED0

# Each handler, the widgets it reads, and the setting it writes.
_HANDLERS = (
    ('ill_slider', 'ill_text', 'illumination_ma', 200.0),
    ('gain_slider', 'gain_text', 'gain_db', 6.0),
    ('exp_slider', 'exp_text', 'exposure_ms', 40.0),
)


# The layer's stored values, apart from every value a handler writes.
_STORED = {'illumination_ma': 50.0, 'gain_db': 1.0, 'exposure_ms': 10.0}


def _panel(monkeypatch, *, initializing, update_settings=None):
    writes = []

    def _stored(path, value):
        writes.append((path, value))
        _layer, key = path.split('.')
        settings[LED0][key] = value

    settings = {'fx2_debug_wire_enabled': False, LED0: dict(_STORED)}
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings=settings,
            settings_lock=threading.Lock(),
            session=SimpleNamespace(run_lockout=True, update_settings=update_settings or _stored),
            camera_executor=object(),
            io_executor=object(),
        ),
    )
    import ui.channel_control as channel_control

    monkeypatch.setattr(channel_control, 'submit_reported', inline_submit_reported)
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's decision
    panel = ChannelControl.__new__(ChannelControl)
    panel.channel = LED0
    panel._initializing = initializing
    panel.ids = {}
    for slider, text, _key, value in _HANDLERS:
        panel.ids[slider] = SimpleNamespace(value=value, disabled=False, min=0.0, max=1000.0)
        panel.ids[text] = SimpleNamespace(text='')
    return panel, writes


def test_a_slider_moved_while_a_run_holds_the_scope_writes_its_setting(monkeypatch):
    panel, writes = _panel(monkeypatch, initializing=False)

    for handler, *_ in _HANDLERS:
        getattr(ChannelControl, handler)(panel)

    assert writes == [(f'{LED0}.{key}', value) for *_, key, value in _HANDLERS]


def test_a_display_of_a_step_during_a_run_writes_no_setting(monkeypatch):
    panel, writes = _panel(monkeypatch, initializing=True)

    for handler, *_ in _HANDLERS:
        getattr(ChannelControl, handler)(panel)

    assert writes == []


def test_a_refused_slider_write_shows_the_stored_value_again(monkeypatch, centre_posts):
    def _refuse(path, value):
        raise ControlSessionRefusedError('control_session_held', control_session='a client')

    panel, _writes = _panel(monkeypatch, initializing=False, update_settings=_refuse)
    import ui.channel_control as channel_control
    from ui.ui_helpers import run_reported

    # The write and its redraw run once the handler has returned, as on the
    # lane and then the GUI thread; inline, the handler's own text update
    # would land after the redraw.
    submitted = []
    monkeypatch.setattr(
        channel_control,
        'submit_reported',
        lambda call, redraw, label, **_: submitted.append((call, redraw, label)),
    )

    for handler, *_ in _HANDLERS:
        getattr(ChannelControl, handler)(panel)
    for call, redraw, label in submitted:
        run_reported(call, redraw, label)

    shown = {
        key: (panel.ids[slider].value, panel.ids[text].text) for slider, text, key, _ in _HANDLERS
    }
    assert shown == {key: (value, str(value)) for key, value in _STORED.items()}
    assert [post.title for post in centre_posts] == [UNDER_CONTROL_TITLE] * len(_HANDLERS)
