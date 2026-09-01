# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The composite-capture worker guarantees LED extinguish on failure.

The worker's channel loop turns an LED on, captures, then turns it off --
the off is reachable only on the non-raising path, and nothing below the
host extinguishes on its own (no firmware watchdog, no driver auto-off).
So the worker itself must darken every channel when its inner worker
raises, and the LED enable button must then be shown OFF rather than
restored from the pre-capture snapshot, which would re-command the LED on
over dark hardware. On success the worker must NOT add an extra off: the
success path extinguishes per-channel inside the loop and later restores
the operator's previous LED state on the main thread, and an extra off
here would race that restore.

The lit-before-fault assertions are load-bearing: an extinguish test that
never proves the LED was lit can pass over a fix that silently refuses
the on-command (dark captures) just as green as over a working one.

save_live_image's off-after-capture flag carries the same contract one
layer down: a caller that asked for the off is relying on it to end
illumination, so it must hold when the capture raises.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules import image_save
from modules.lumascope_api import Lumascope


# ui.composite_capture is a Kivy widget module; conftest mocks `kivy` but not
# the uix submodules, and CompositeCapture subclasses FloatLayout (a bare
# MagicMock cannot be subclassed).
class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)

import modules.app_context as _app_ctx
import ui.composite_capture as cc


LAYER = 'Blue'
ILLUMINATION_MA = 10.0


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    return s


def _lit(scope):
    return scope.illumination.get_led_state(LAYER)['enabled']


class _RaisingSelf:
    """Stands in for the CompositeCapture instance; its inner worker raises."""

    def _composite_capture_worker_inner(self, **kwargs):
        raise RuntimeError('injected capture failure')


class _SucceedingSelf:
    def _composite_capture_worker_inner(self, **kwargs):
        pass


@pytest.fixture
def app_ctx(scope):
    saved = getattr(_app_ctx, 'ctx', None)
    _app_ctx.ctx = SimpleNamespace(
        scope=scope,
        lumaview=MagicMock(),
        image_settings=MagicMock(),
    )
    yield _app_ctx.ctx
    _app_ctx.ctx = saved


def _run_worker(worker_self):
    cc.CompositeCapture._composite_capture_worker(
        worker_self,
        z_stage_present=False,
        initial_layer=LAYER,
        led_restore_state=True,
        capture_depth=8,
        save_encoding='raw',
        saved_video_false_color=None,
    )


def test_inner_raise_extinguishes_all_leds(scope, app_ctx):
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: the LED must be lit before the fault'

    _run_worker(_RaisingSelf())

    assert not _lit(scope), (
        'a raise out of the inner worker must leave every LED dark; '
        'the per-channel off calls are unreachable on this path'
    )


def test_inner_raise_forces_led_button_off(scope, app_ctx, monkeypatch):
    opened_layer_obj = MagicMock()
    monkeypatch.setattr(cc.common_utils, 'get_opened_layer_obj', lambda _settings: opened_layer_obj)
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: the LED must be lit before the fault'

    cc.Clock.schedule_once.reset_mock()
    _run_worker(_RaisingSelf())

    assert cc.Clock.schedule_once.called, 'the UI restore must be scheduled'
    restore_cb = cc.Clock.schedule_once.call_args[0][0]
    restore_cb(0)

    assert opened_layer_obj.ids['enable_led_btn'].state == 'normal', (
        'after a failed capture the LED button must show OFF (hardware is '
        'dark); a snapshot restore would re-command the LED on'
    )
    opened_layer_obj.apply_settings.assert_not_called()


def test_success_path_does_not_add_an_extra_off(scope, app_ctx):
    # Behavior-preservation guard: passes before and after the fix. The
    # success path's LED end-state belongs to the loop's own off calls and
    # the scheduled restore, not to the worker's finally.
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope)

    _run_worker(_SucceedingSelf())

    assert _lit(scope), (
        'a clean worker pass must not extinguish behind the scheduled LED-state restore'
    )


def test_inactive_camera_refusal_leaves_capture_usable(scope, app_ctx, monkeypatch):
    """A composite click with the camera inactive must refuse loudly and
    leave the app usable: `_capturing` clear (both capture buttons work
    again), no display state half-toggled, nothing dispatched, and a
    notification telling the operator why nothing happened."""
    app_ctx.disable_homing = True
    app_ctx.settings = {}
    app_ctx.worker_pool = MagicMock()
    monkeypatch.setattr(cc.common_utils, 'get_opened_layer', lambda _settings: LAYER)
    histo_off = MagicMock()
    monkeypatch.setattr(cc, 'live_histo_off', histo_off)
    notifications = MagicMock()
    monkeypatch.setattr('modules.notification_center.notifications', notifications)

    with scope.imaging._camera_cache_lock:
        scope.imaging._camera_cache['active'] = False
    assert not scope.imaging.active_cached, 'precondition: camera not streaming'
    cc.CompositeCapture._capturing.clear()

    try:
        cc.CompositeCapture.composite_capture(_SucceedingSelf())

        assert not cc.CompositeCapture._capturing.is_set(), (
            'an inactive-camera refusal must clear the capture guard; a set '
            'guard with no worker to clear it wedges BOTH capture buttons '
            'for the process lifetime'
        )
        assert not histo_off.called, (
            'the refusal must precede display-state changes; a refused click '
            'must not leave live-histogram equalization off'
        )
        assert not app_ctx.worker_pool.put.called, 'nothing to dispatch on a refusal'
        assert notifications.warning.called or notifications.error.called, (
            'the operator must be told why nothing happened'
        )
    finally:
        cc.CompositeCapture._capturing.clear()


def test_save_live_image_off_flag_holds_on_capture_raise(scope, tmp_path):
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: the LED must be lit before the fault'

    with (
        patch.object(scope.imaging, '_capture_and_wait_impl', side_effect=RuntimeError('boom')),
        pytest.raises(RuntimeError),
    ):
        image_save.save_live_image(
            scope,
            save_folder=tmp_path,
            file_root='ext_',
            turn_off_all_leds_after=True,
            channel=LAYER,
            false_color_on=False,
            save_encoding='raw',
        )

    assert not _lit(scope), (
        'turn_off_all_leds_after promises the off; it must hold when the capture raises'
    )
