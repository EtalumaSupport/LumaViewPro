# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A format spinner set from the store is not a pick: nothing is logged or written.

The settings load sets both image-format spinners from the store, and the kv's
``on_text`` fires their pick handlers for it. Each start-up then recorded two
picks nobody made and wrote the stored formats back through the writer.

Handlers are called unbound on a stand-in panel rather than instantiating Kivy
widgets.
"""

from types import SimpleNamespace

import pytest

from tests.settings_fixtures import settings_writer

_HANDLERS = [
    ('select_live_image_output_format', 'live_image_output_format_spinner', 'live'),
    ('select_sequenced_image_output_format', 'sequenced_image_output_format_spinner', 'sequenced'),
]


@pytest.fixture
def recorded(monkeypatch):
    import modules.app_context as _app_ctx
    import ui.microscope_settings as microscope_settings

    picks = []
    settings = {'image_output_format': {'live': 'TIFF', 'sequenced': 'TIFF'}}
    monkeypatch.setattr(
        microscope_settings.gui_logger, 'select', lambda name, value: picks.append(value)
    )
    monkeypatch.setattr(microscope_settings, 'run_reported', lambda call, redraw, label: call())
    # The writer's own check stands in for the Session's, so a handler that
    # wrote a format the writer refuses would fail here too.
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(settings=settings, update_settings=settings_writer(settings)),
    )
    return picks, settings['image_output_format']


@pytest.mark.parametrize('handler, spinner, key', _HANDLERS)
def test_the_stored_format_is_not_logged_or_written(recorded, handler, spinner, key):
    from ui.microscope_settings import MicroscopeSettings

    panel = SimpleNamespace(ids={spinner: SimpleNamespace(text='TIFF')})
    getattr(MicroscopeSettings, handler)(panel)

    assert recorded == ([], {'live': 'TIFF', 'sequenced': 'TIFF'})


@pytest.mark.parametrize('handler, spinner, key', _HANDLERS)
def test_a_picked_format_is_logged_and_written(recorded, handler, spinner, key):
    from ui.microscope_settings import MicroscopeSettings

    # A pick also refreshes the JPG depth hint, which this test does not observe.
    panel = SimpleNamespace(
        ids={spinner: SimpleNamespace(text='JPG')}, _refresh_jpg_depth_hint=lambda: None
    )
    getattr(MicroscopeSettings, handler)(panel)

    picks, stored = recorded
    assert picks == ['JPG']
    assert stored[key] == 'JPG'
    assert stored[{'live': 'sequenced', 'sequenced': 'live'}[key]] == 'TIFF'
