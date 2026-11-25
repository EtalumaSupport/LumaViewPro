# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What every run shares is drawn once, from what holds the scope.

Live histogram equalization and the LED enable toggles belong to no one
run control. Every control redraws on
every run-state edge -- the edge where another run takes the scope
included -- so one writer draws these from the edge's lock record and live
work, and a control drawing itself idle cannot undo the live run's display.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import _lumascope.modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
from _lumascope.modules import live_work
from _lumascope.modules.live_work import LiveWork, SessionLifecycle, WorkItem
from _lumascope.modules.scope_session import ControlsLock


@pytest.fixture
def scene(monkeypatch):
    saved = getattr(_app_ctx, 'ctx', None)
    display = SimpleNamespace(use_live_image_histogram_equalization=False)
    _app_ctx.ctx = SimpleNamespace(
        scope_display=display,
        live_histo_setting=True,
        ui_listener_bridge=MagicMock(),
    )
    yield SimpleNamespace(
        lock=ControlsLock(run_named=None, control_named=None, recording=False),
        work=(),
        display=display,
        ctx=_app_ctx.ctx,
    )
    _app_ctx.ctx = saved


def _draw(scene):
    ui_helpers.draw_shared_run_displays(
        scene.lock, LiveWork(work=tuple(scene.work), lifecycle=SessionLifecycle.OPEN)
    )


def test_a_live_run_keeps_equalization_off_and_its_leds(scene):
    scene.lock = ControlsLock(
        run_named='A protocol run is in progress.', control_named=None, recording=False
    )
    scene.work = [WorkItem(live_work.WorkKind.PROTOCOL, 'A protocol run')]
    scene.display.use_live_image_histogram_equalization = True

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_not_called()


def test_a_drain_with_nothing_holding_the_scope_hands_back_the_leds(scene):
    scene.lock = ControlsLock(
        run_named="A protocol's files are still being written.", control_named=None, recording=False
    )
    scene.work = [WorkItem(live_work.WorkKind.RUN_FILES, "A run's images", left=3)]
    scene.display.use_live_image_histogram_equalization = True

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_called_once()


def test_idle_hands_back_equalization_and_the_led_toggles(scene):
    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is True
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_called_once()


def test_an_idle_control_session_keeps_the_live_equalization(scene):
    # The controls lock for a client's session, but nothing runs: the live
    # picture is still the scope's, and the person keeps seeing it.
    scene.lock = ControlsLock(
        run_named=None, control_named='plate reader has remote control.', recording=False
    )

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is True


def test_the_holders_run_turns_equalization_off_as_any_run_does(scene):
    scene.lock = ControlsLock(
        run_named='A protocol run is in progress.',
        control_named='plate reader has remote control.',
        recording=False,
    )
    scene.work = [WorkItem(live_work.WorkKind.PROTOCOL, 'A protocol run')]
    scene.display.use_live_image_histogram_equalization = True

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False


def test_equalization_comes_back_only_as_the_setting_says(scene):
    scene.ctx.live_histo_setting = False

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False


@pytest.mark.parametrize('holder', ['recording', 'diagnostic'])
def test_another_holder_keeps_its_leds(scene, holder):
    # A diagnostic may be lighting LEDs the toggles must not be reconciled
    # over, and a recording holds the scope as a run does.
    scene.lock = ControlsLock(
        run_named='A diagnostic is in progress.' if holder == 'diagnostic' else None,
        control_named=None,
        recording=holder == 'recording',
    )
    scene.work = [WorkItem(holder, 'the holder')]

    _draw(scene)

    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_not_called()
