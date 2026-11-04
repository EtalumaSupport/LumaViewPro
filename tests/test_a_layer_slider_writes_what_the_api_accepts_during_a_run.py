# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer slider writes what the API accepts, run or no run; a display of a step writes nothing.

Before, the illumination, gain and exposure slider handlers returned
without writing whenever ``session.run_lockout`` held, a decision the GUI
made where the API accepts the write: ``update_settings`` takes a layer's
gain while a run holds the scope. The GUI now decides nothing there. What
keeps a run's per-step display from writing is the guard every
programmatic slider write sets, ``_initializing`` (``set_step_state``), and
it still holds with the returns gone.
"""

from __future__ import annotations

from types import SimpleNamespace

import modules.app_context as _app_ctx
from tests.gesture_fakes import inline_submit_reported
from ui.layer_control import LayerControl
from tests.stock_channels import LED0

# Each handler, the widgets it reads, and the setting it writes.
_HANDLERS = (
    ('ill_slider', 'ill_text', 'illumination_ma', 200.0),
    ('gain_slider', 'gain_text', 'gain_db', 6.0),
    ('exp_slider', 'exp_text', 'exposure_ms', 40.0),
)


def _panel(monkeypatch, *, initializing):
    writes = []
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings={'fx2_debug_wire_enabled': False},
            session=SimpleNamespace(
                run_lockout=True, update_settings=lambda path, value: writes.append((path, value))
            ),
            camera_executor=object(),
            io_executor=object(),
        ),
    )
    import ui.layer_control as layer_control

    monkeypatch.setattr(layer_control, 'submit_reported', inline_submit_reported)
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's decision
    panel = LayerControl.__new__(LayerControl)
    panel.layer = LED0
    panel._initializing = initializing
    panel.ids = {}
    for slider, text, _key, value in _HANDLERS:
        panel.ids[slider] = SimpleNamespace(value=value, disabled=False)
        panel.ids[text] = SimpleNamespace(text='')
    return panel, writes


def test_a_slider_moved_while_a_run_holds_the_scope_writes_its_setting(monkeypatch):
    panel, writes = _panel(monkeypatch, initializing=False)

    for handler, *_ in _HANDLERS:
        getattr(LayerControl, handler)(panel)

    assert writes == [(f'{LED0}.{key}', value) for *_, key, value in _HANDLERS]


def test_a_display_of_a_step_during_a_run_writes_no_setting(monkeypatch):
    panel, writes = _panel(monkeypatch, initializing=True)

    for handler, *_ in _HANDLERS:
        getattr(LayerControl, handler)(panel)

    assert writes == []
