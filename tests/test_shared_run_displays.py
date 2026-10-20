# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What every run shares is drawn once, from what holds the scope.

Live histogram equalization, the window title's run suffix and the LED
enable toggles belong to no one run control. Every control redraws on
every run-state edge -- the edge where another run takes the scope
included -- so one writer draws these from the edge's lock record and live
work, and a control drawing itself idle cannot undo the live run's display.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
from modules import live_work
from modules.live_work import LiveWork, WorkItem
from modules.scope_session import ControlsLock


@pytest.fixture
def scene(monkeypatch):
    saved = getattr(_app_ctx, 'ctx', None)
    display = SimpleNamespace(use_live_image_histogram_equalization=False)
    _app_ctx.ctx = SimpleNamespace(
        scope_display=display,
        live_histo_setting=True,
        ui_listener_bridge=MagicMock(),
    )
    ui_helpers.set_title_event_text('left by someone else')
    yield SimpleNamespace(
        lock=ControlsLock(run_named=None, recording=False),
        work=(),
        display=display,
        ctx=_app_ctx.ctx,
    )
    _app_ctx.ctx = saved
    ui_helpers.set_title_event_text(None)


def _draw(scene):
    ui_helpers.draw_shared_run_displays(
        scene.lock, LiveWork(work=tuple(scene.work), closing=False, closed=False)
    )


def test_a_live_run_keeps_equalization_off_and_its_own_title(scene):
    scene.lock = ControlsLock(run_named='A protocol run is in progress.', recording=False)
    scene.work = [WorkItem(live_work.PROTOCOL, 'A protocol run')]
    scene.display.use_live_image_histogram_equalization = True

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False
    assert ui_helpers.get_title_event_text() == 'left by someone else', (
        "a live run's title is its own writers'"
    )
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_not_called()


def test_a_drain_with_nothing_holding_the_scope_says_files_are_writing(scene):
    scene.lock = ControlsLock(
        run_named="A protocol's files are still being written.", recording=False
    )
    scene.work = [WorkItem(live_work.RUN_FILES, "A run's images", left=3)]
    scene.display.use_live_image_histogram_equalization = True

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False
    assert ui_helpers.get_title_event_text() == 'Writing protocol scan files to disk...'
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_called_once()


def test_idle_hands_back_equalization_the_title_and_the_led_toggles(scene):
    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is True
    assert ui_helpers.get_title_event_text() is None
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_called_once()


def test_equalization_comes_back_only_as_the_setting_says(scene):
    scene.ctx.live_histo_setting = False

    _draw(scene)

    assert scene.display.use_live_image_histogram_equalization is False


@pytest.mark.parametrize('holder', ['recording', 'diagnostic'])
def test_another_holder_keeps_its_title_and_its_leds(scene, holder):
    # A recording and a diagnostic title the window themselves, and a
    # diagnostic may be lighting LEDs the toggles must not be reconciled over.
    scene.lock = ControlsLock(
        run_named='A diagnostic is in progress.' if holder == 'diagnostic' else None,
        recording=holder == 'recording',
    )
    scene.work = [WorkItem(holder, 'the holder')]

    _draw(scene)

    assert ui_helpers.get_title_event_text() == 'left by someone else'
    scene.ctx.ui_listener_bridge.reconcile_led_buttons.assert_not_called()
