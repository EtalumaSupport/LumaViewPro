# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The two rate-cap sliders in Advanced Settings store where they are released.

Recording max FPS (``video.max_fps``) and Display max FPS (``live_view_fps``)
are one control: a slider of whole frames per second from 1 to the setting's
highest cap, and one position past it that means no cap, stored as 0. A
stored cap is shown at its position, and no cap at the top.

Kivy is stubbed in the test process: the real handlers are driven on a
stand-in shaped like the panel, against a simulated session's settings.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
from modules import gui_logger
from modules.scope_session import ScopeSession
from modules.settings_paths import LIVE_VIEW_FPS_LIMIT, VIDEO_MAX_FPS_LIMIT
from tests.settings_fixtures import complete_settings
from ui.advanced_settings import AdvancedSettings

_SLIDERS = (
    ('recording_max_fps_slider', 'video.max_fps', VIDEO_MAX_FPS_LIMIT, 'RECORDING_MAX_FPS'),
    ('display_max_fps_slider', 'live_view_fps', LIVE_VIEW_FPS_LIMIT, 'DISPLAY_MAX_FPS'),
)


@pytest.fixture
def session(monkeypatch):
    s = ScopeSession.create(complete_settings(), simulate=True)
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=s, update_settings=s.update_settings)
    )
    yield s
    s.shutdown()


@pytest.fixture
def recorded(monkeypatch):
    lines = []
    monkeypatch.setattr(gui_logger, 'slider', lambda name, value: lines.append((name, value)))
    return lines


class _Panel:
    """The real handlers, with only the widget tree stubbed."""

    _FPS_LIMITS = AdvancedSettings._FPS_LIMITS
    _fps_limit_released = AdvancedSettings._fps_limit_released
    _store_fps_limit = AdvancedSettings._store_fps_limit
    _show_fps_limit = AdvancedSettings._show_fps_limit
    recording_max_fps_slider = AdvancedSettings.recording_max_fps_slider
    display_max_fps_slider = AdvancedSettings.display_max_fps_slider

    def __init__(self):
        self.ids = {slider_id: SimpleNamespace(value=0) for slider_id, *_ in _SLIDERS}


def test_a_released_position_is_stored_as_that_cap_and_recorded(session, recorded):
    panel = _Panel()
    for slider_id, path, _limit, name in _SLIDERS:
        panel.ids[slider_id].value = 25
        getattr(panel, slider_id)()
        assert session.get_setting(path) == 25, slider_id
        assert (name, 25) in recorded, slider_id


def test_the_top_position_is_no_cap_and_no_cap_is_shown_there(session, recorded):
    panel = _Panel()
    for slider_id, path, limit, name in _SLIDERS:
        session.update_settings(path, 10)
        panel.ids[slider_id].value = limit + 1
        getattr(panel, slider_id)()
        assert session.get_setting(path) == 0, slider_id
        assert (name, 0) in recorded, slider_id

        panel.ids[slider_id].value = 1
        panel._show_fps_limit(slider_id)
        assert panel.ids[slider_id].value == limit + 1, slider_id


def test_a_stored_cap_is_shown_at_its_position(session):
    panel = _Panel()
    for slider_id, path, limit, _name in _SLIDERS:
        session.update_settings(path, limit)
        panel._show_fps_limit(slider_id)
        assert panel.ids[slider_id].value == limit, slider_id
