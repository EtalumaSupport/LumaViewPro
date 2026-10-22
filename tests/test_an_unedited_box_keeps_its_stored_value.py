# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer's box commits what was typed, and nothing when nothing was typed.

The kv fires every layer text handler on focus LOSS, not on edit, so a
click in and out arrives with the box's text untouched. The shared text
handler returns without committing when that text equals the stored
value. The exposure and illumination boxes had their own handlers
without that return, so a stray click clipped the stored value to the
box's bound: a layer stored at 2000 ms, kept on purpose above the FX2's
1000 ms cap, became 1000 ms, and the periodic flush persisted the loss.

An edit is written as typed: the writer decides what is stored (it
refuses a value outside the setting's range, and the box then shows what
is stored), and the slider's own range narrows nothing. The handler used
to clip the typed value to the slider's range, or to a widget ceiling,
before the write, so the API's refusal was unreachable from the GUI and
the store held a number the person did not type (Eric, 2026-10-08).
"""

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
from modules.notification_center import notifications
from ui.layer_control import LayerControl
from tests.gesture_fakes import inline_submit_reported
from tests.settings_fixtures import settings_writer


class _Slider:
    def __init__(self, low, high, value):
        self.min = low
        self.max = high
        self.value = value


def _blue_layer(monkeypatch, *, exposure_ms, illumination_ma):
    settings = {
        'Blue': {'exposure_ms': exposure_ms, 'illumination_ma': illumination_ma},
        'fx2_debug_wire_enabled': False,
    }
    writer = settings_writer(settings)
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings=settings,
            update_settings=writer,
            session=SimpleNamespace(update_settings=writer),
            camera_executor=object(),
            io_executor=object(),
        ),
    )
    import ui.layer_control as layer_control

    monkeypatch.setattr(layer_control, 'submit_reported', inline_submit_reported)
    layer = LayerControl.__new__(LayerControl)
    layer.layer = 'Blue'
    layer._initializing = False
    layer.ids = {
        'exp_slider': _Slider(0.01, 1000.0, 1000.0),
        'exp_text': SimpleNamespace(focus=False, text=str(exposure_ms)),
        'ill_slider': _Slider(0.0, 150.0, 150.0),
        'ill_text': SimpleNamespace(focus=False, text=str(illumination_ma)),
    }
    # The Session puts the write on the hardware; the box makes no apply of its own.
    applied = []
    layer.apply_settings = lambda: applied.append('show')
    return layer, settings, applied


@pytest.mark.parametrize(
    ('handler', 'key', 'stored'),
    [('exp_text', 'exposure_ms', 2000.0), ('ill_text', 'illumination_ma', 300.0)],
)
def test_a_stored_value_above_the_bound_survives_a_click_in_and_out(
    monkeypatch, handler, key, stored
):
    kwargs = {'exposure_ms': 1000.0, 'illumination_ma': 100.0, key: stored}
    layer, settings, applied = _blue_layer(monkeypatch, **kwargs)

    getattr(layer, handler)()

    assert settings['Blue'][key] == stored
    assert applied == []


@pytest.mark.parametrize(
    ('handler', 'text_id', 'key', 'typed', 'committed'),
    [
        # Above the slider's 1000: stored as typed, the slider pins at its end.
        ('exp_text', 'exp_text', 'exposure_ms', '1500', 1500.0),
        ('ill_text', 'ill_text', 'illumination_ma', '42.5', 42.5),
    ],
)
def test_an_edit_is_committed_as_typed_with_no_apply_of_the_boxs_own(
    monkeypatch, handler, text_id, key, typed, committed
):
    layer, settings, applied = _blue_layer(monkeypatch, exposure_ms=500.0, illumination_ma=100.0)
    layer.ids[text_id].text = typed

    getattr(layer, handler)()

    assert settings['Blue'][key] == committed
    assert layer.ids[text_id].text == str(committed)
    assert applied == []


def test_an_edit_the_writer_refuses_is_not_stored_and_the_box_goes_back(monkeypatch):
    reported = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda exc, **kw: reported.append(type(exc).__name__)
    )
    layer, settings, applied = _blue_layer(monkeypatch, exposure_ms=500.0, illumination_ma=100.0)
    layer.ids['exp_text'].text = '-5'

    layer.exp_text()

    assert settings['Blue']['exposure_ms'] == 500.0
    assert layer.ids['exp_text'].text == '500.0'
    assert reported == ['SettingRefusedError']
    assert applied == []
