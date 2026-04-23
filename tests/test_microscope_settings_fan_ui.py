# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stage 4 tests — MicroscopeSettings fan UI handlers.

These tests don't render Kivy widgets; they instantiate
``MicroscopeSettings.__new__`` with mock ``ids`` dicts and call the
handler methods directly. The goal is to cover the status→widget
mapping, the user-drag suppression, and the fan_ui_kind visibility gate
without standing up a full Kivy window.

See ``ui/microscope_settings.py::_update_fan_ui_visibility`` and the
handler methods below it.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Monkeypatched kivy/ui mocks — autouse so every test in this file gets
# the real-class BoxLayout + HoverBehavior needed for MicroscopeSettings
# to import cleanly. Scoped via monkeypatch so sys.modules is restored
# between tests; avoids leaking fakes into adjacent test files.
# ---------------------------------------------------------------------------

class _FakeKivyWidget:
    def __init__(self, **kwargs):
        pass


class _FakeHoverBehavior:
    pass


def _install_mocks(monkeypatch):
    boxlayout_mod = types.ModuleType('kivy.uix.boxlayout')
    boxlayout_mod.BoxLayout = _FakeKivyWidget
    monkeypatch.setitem(sys.modules, 'kivy.uix.boxlayout', boxlayout_mod)

    props = types.ModuleType('kivy.properties')
    for _name in ('ListProperty', 'StringProperty', 'NumericProperty',
                  'BooleanProperty', 'ObjectProperty'):
        setattr(props, _name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'kivy.properties', props)

    hb = types.ModuleType('ui.hover_behavior')
    hb.HoverBehavior = _FakeHoverBehavior
    monkeypatch.setitem(sys.modules, 'ui.hover_behavior', hb)

    hv_mod = types.ModuleType('kivy.uix.behaviors.hover')
    hv_mod.HoverBehavior = _FakeHoverBehavior
    monkeypatch.setitem(sys.modules, 'kivy.uix.behaviors.hover', hv_mod)

    button_mod = types.ModuleType('kivy.uix.button')
    button_mod.Button = _FakeKivyWidget
    monkeypatch.setitem(sys.modules, 'kivy.uix.button', button_mod)

    for name in [
        'kivy.app', 'kivy.core', 'kivy.core.window', 'kivy.factory',
        'kivy.graphics', 'kivy.graphics.texture',
        'kivy.graphics.instructions', 'kivy.graphics.vertex_instructions',
        'kivy.lang', 'kivy.metrics', 'kivy.uix',
        'kivy.uix.floatlayout', 'kivy.uix.gridlayout', 'kivy.uix.image',
        'kivy.uix.label', 'kivy.uix.popup', 'kivy.uix.scrollview',
        'kivy.uix.slider', 'kivy.uix.spinner', 'kivy.uix.textinput',
        'kivy.uix.togglebutton', 'kivy.uix.widget',
        'kivy.uix.behaviors',
    ]:
        monkeypatch.setitem(sys.modules, name, MagicMock())

    # Drop any cached ui.* modules so the next import picks up the fakes.
    for mod_name in [
        'ui.microscope_settings', 'ui.motion_settings',
        'ui.image_settings', 'ui.ui_helpers', 'ui.file_dialogs',
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)


@pytest.fixture(autouse=True)
def _kivy_mocks(monkeypatch):
    _install_mocks(monkeypatch)
    yield


@pytest.fixture
def patched_ctx():
    ctx = MagicMock()
    ctx.lumaview = MagicMock()
    ctx.lumaview.scope = MagicMock()
    with patch('modules.app_context.ctx', ctx):
        yield ctx


def _make_widget(kind='generic'):
    w = MagicMock()
    w.height = '0dp'
    w.opacity = 0
    if kind == 'slider':
        w.value = 50.0
    if kind == 'toggle':
        w.state = 'normal'
    if kind == 'label':
        w.text = ''
    return w


def _make_ids_full():
    """Return an ids dict populated with the full fan section."""
    return {
        'fan_section_id': _make_widget(),
        'fan_section_heading_id': _make_widget(),
        'fan_hilo_row_id': _make_widget(),
        'fan_pwm_row_id': _make_widget(),
        'fan_rpm_row_id': _make_widget(),
        'fan_hi_btn_id': _make_widget('toggle'),
        'fan_lo_btn_id': _make_widget('toggle'),
        'fan_off_btn_id': _make_widget('toggle'),
        'fan_pwm_slider_id': _make_widget('slider'),
        'fan_pwm_text_id': _make_widget('label'),
        'fan_rpm_label_id': _make_widget('label'),
    }


def _make_microscope_settings(ids):
    from ui.microscope_settings import MicroscopeSettings
    ms = MicroscopeSettings.__new__(MicroscopeSettings)
    ms.ids = ids
    ms._fan_ui_kind = None
    ms._fan_pwm_user_active = False
    ms._fan_listener_registered = False
    return ms


# ---------------------------------------------------------------------------
# _update_fan_ui_visibility — fan_ui_kind gating
# ---------------------------------------------------------------------------

class TestUpdateFanUiVisibility:

    def test_none_hides_entire_section(self, patched_ctx):
        patched_ctx.lumaview.scope.fan_ui_kind = MagicMock(return_value=None)
        ms = _make_microscope_settings(_make_ids_full())
        ms._update_fan_ui_visibility()

        assert ms.ids['fan_section_id'].opacity == 0
        assert ms.ids['fan_hilo_row_id'].opacity == 0
        assert ms.ids['fan_pwm_row_id'].opacity == 0
        assert ms.ids['fan_rpm_row_id'].opacity == 0

    def test_hilo_shows_radio_only(self, patched_ctx):
        patched_ctx.lumaview.scope.fan_ui_kind = MagicMock(return_value='HILO')
        patched_ctx.lumaview.scope.get_fan_status = MagicMock(return_value={
            'mode': 'HILO', 'state': 'HI', 'fan_pct': None, 'tach_rpm': None,
        })
        patched_ctx.lumaview.scope.add_fan_listener = MagicMock()
        ms = _make_microscope_settings(_make_ids_full())
        ms._update_fan_ui_visibility()

        assert ms.ids['fan_section_id'].opacity == 1
        assert ms.ids['fan_hilo_row_id'].opacity == 1
        # PWM + RPM rows stay hidden (HILO fan has no tach).
        assert ms.ids['fan_pwm_row_id'].opacity == 0
        assert ms.ids['fan_rpm_row_id'].opacity == 0

    def test_pwm_shows_slider_and_rpm(self, patched_ctx):
        patched_ctx.lumaview.scope.fan_ui_kind = MagicMock(return_value='PWM')
        patched_ctx.lumaview.scope.get_fan_status = MagicMock(return_value={
            'mode': 'PWM', 'state': None, 'fan_pct': 40, 'tach_rpm': 2400,
        })
        patched_ctx.lumaview.scope.add_fan_listener = MagicMock()
        ms = _make_microscope_settings(_make_ids_full())
        ms._update_fan_ui_visibility()

        assert ms.ids['fan_section_id'].opacity == 1
        assert ms.ids['fan_hilo_row_id'].opacity == 0
        assert ms.ids['fan_pwm_row_id'].opacity == 1
        assert ms.ids['fan_rpm_row_id'].opacity == 1

    def test_listener_registered_exactly_once(self, patched_ctx):
        patched_ctx.lumaview.scope.fan_ui_kind = MagicMock(return_value='PWM')
        patched_ctx.lumaview.scope.get_fan_status = MagicMock(return_value=None)
        patched_ctx.lumaview.scope.add_fan_listener = MagicMock()
        ms = _make_microscope_settings(_make_ids_full())
        ms._update_fan_ui_visibility()
        ms._update_fan_ui_visibility()
        ms._update_fan_ui_visibility()
        assert patched_ctx.lumaview.scope.add_fan_listener.call_count == 1

    def test_fan_ui_kind_missing_is_silent(self, patched_ctx):
        del patched_ctx.lumaview.scope.fan_ui_kind
        ms = _make_microscope_settings(_make_ids_full())
        ms._update_fan_ui_visibility()
        assert ms.ids['fan_section_id'].opacity == 0


# ---------------------------------------------------------------------------
# _apply_fan_status_to_widgets — status dict → widget state
# ---------------------------------------------------------------------------

class TestApplyFanStatusToWidgets:

    def test_hilo_hi_sets_hi_button_down(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'HILO', 'state': 'HI', 'fan_pct': None, 'tach_rpm': None,
        })
        assert ms.ids['fan_hi_btn_id'].state == 'down'
        assert ms.ids['fan_lo_btn_id'].state == 'normal'
        assert ms.ids['fan_off_btn_id'].state == 'normal'

    def test_hilo_lo_sets_lo_button_down(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'HILO', 'state': 'LO', 'fan_pct': None, 'tach_rpm': None,
        })
        assert ms.ids['fan_lo_btn_id'].state == 'down'
        assert ms.ids['fan_hi_btn_id'].state == 'normal'

    def test_pwm_pct_updates_slider(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'PWM', 'state': None, 'fan_pct': 75, 'tach_rpm': 2000,
        })
        assert ms.ids['fan_pwm_slider_id'].value == 75.0

    def test_pwm_pct_skipped_while_user_drags(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._fan_pwm_user_active = True
        ms.ids['fan_pwm_slider_id'].value = 50.0
        ms._apply_fan_status_to_widgets({
            'mode': 'PWM', 'state': None, 'fan_pct': 75, 'tach_rpm': 2000,
        })
        # Slider left untouched mid-drag.
        assert ms.ids['fan_pwm_slider_id'].value == 50.0

    def test_rpm_label_prefers_avg(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'PWM', 'state': None, 'fan_pct': 40,
            'tach_rpm': 2400, 'tach_rpm_avg': 2380.4,
        })
        # int(round(2380.4)) = 2380
        assert ms.ids['fan_rpm_label_id'].text == '2380'

    def test_rpm_label_falls_back_to_raw(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'PWM', 'state': None, 'fan_pct': 40, 'tach_rpm': 2400,
        })
        assert ms.ids['fan_rpm_label_id'].text == '2400'

    def test_rpm_label_placeholder_when_no_data(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'HILO', 'state': 'HI',
            'fan_pct': None, 'tach_rpm': None,
        })
        assert ms.ids['fan_rpm_label_id'].text == '—'

    def test_rpm_label_placeholder_when_negative(self, patched_ctx):
        # Firmware returns -1 on tach Timer callback crash.
        ms = _make_microscope_settings(_make_ids_full())
        ms._apply_fan_status_to_widgets({
            'mode': 'PWM', 'state': None,
            'fan_pct': 40, 'tach_rpm': -1,
        })
        assert ms.ids['fan_rpm_label_id'].text == '—'

    def test_empty_status_is_noop(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms.ids['fan_rpm_label_id'].text = 'preserved'
        ms._apply_fan_status_to_widgets(None)
        assert ms.ids['fan_rpm_label_id'].text == 'preserved'


# ---------------------------------------------------------------------------
# set_fan_hilo_from_ui / set_fan_pwm_* — write-through to scope API
# ---------------------------------------------------------------------------

class TestHandlers:

    def test_hilo_button_writes_through(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms.set_fan_hilo_from_ui('LO')
        patched_ctx.lumaview.scope.set_fan_hilo.assert_called_once_with('LO')

    def test_pwm_drag_flag(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms.set_fan_pwm_drag(True)
        assert ms._fan_pwm_user_active is True
        ms.set_fan_pwm_drag(False)
        assert ms._fan_pwm_user_active is False

    def test_pwm_release_writes_through_and_clears_drag(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms._fan_pwm_user_active = True
        ms.set_fan_pwm_from_ui(42)
        assert ms._fan_pwm_user_active is False
        patched_ctx.lumaview.scope.set_fan_pwm.assert_called_once_with(42)

    def test_pwm_release_clamps_out_of_range(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        ms.set_fan_pwm_from_ui(150)
        patched_ctx.lumaview.scope.set_fan_pwm.assert_called_once_with(100)
        patched_ctx.lumaview.scope.set_fan_pwm.reset_mock()
        ms.set_fan_pwm_from_ui(-5)
        patched_ctx.lumaview.scope.set_fan_pwm.assert_called_once_with(0)

    def test_handler_swallows_exceptions(self, patched_ctx):
        ms = _make_microscope_settings(_make_ids_full())
        patched_ctx.lumaview.scope.set_fan_hilo.side_effect = RuntimeError('nope')
        # Must not raise.
        ms.set_fan_hilo_from_ui('HI')
