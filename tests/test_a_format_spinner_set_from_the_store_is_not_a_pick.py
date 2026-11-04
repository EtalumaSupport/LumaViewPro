# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A format spinner set from the store is not a pick: nothing is written, and the handler records nothing.

The settings load sets both image-format spinners from the store, and the kv's
``on_text`` fires their handlers for it. Each start-up then recorded two picks
nobody made and wrote the stored formats back through the writer. A person's
pick is recorded by the spinner itself (``PickSpinner``), so the handler never
records, whoever set the text.

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
def test_a_picked_format_is_written_and_left_to_the_spinner_to_record(
    recorded, handler, spinner, key
):
    from ui.microscope_settings import MicroscopeSettings

    # A pick also refreshes the JPG depth hint and hands the panel's redraw to
    # the reporter, neither of which this test observes.
    panel = SimpleNamespace(
        ids={spinner: SimpleNamespace(text='JPG')},
        _refresh_jpg_depth_hint=lambda: None,
        show_capture_settings=lambda: None,
    )
    getattr(MicroscopeSettings, handler)(panel)

    picks, stored = recorded
    assert picks == []
    assert stored[key] == 'JPG'
    assert stored[{'live': 'sequenced', 'sequenced': 'live'}[key]] == 'TIFF'


def _hyperstack_panel():
    return SimpleNamespace(
        ids={'sequenced_image_output_format_spinner': SimpleNamespace(text='Hyperstack')},
        _refresh_jpg_depth_hint=lambda: None,
        show_capture_settings=lambda: None,
    )


def test_a_pick_of_the_hyperstack_label_stores_its_format(recorded):
    """The selector reads 'Hyperstack'; the store, the run and REST keep 'OME-TIFF Hyperstack'."""
    from modules.image_mode import OutputFormat
    from ui.microscope_settings import MicroscopeSettings

    MicroscopeSettings.select_sequenced_image_output_format(_hyperstack_panel())

    _picks, stored = recorded
    assert stored == {'live': 'TIFF', 'sequenced': OutputFormat.HYPERSTACK}


def test_the_hyperstack_label_restored_from_the_store_writes_nothing(recorded, monkeypatch):
    """The load sets the label from a stored 'OME-TIFF Hyperstack': not a pick."""
    import modules.app_context as _app_ctx
    from modules.image_mode import OutputFormat
    from ui.microscope_settings import MicroscopeSettings

    picks, stored = recorded
    stored['sequenced'] = OutputFormat.HYPERSTACK
    writes = []
    monkeypatch.setattr(_app_ctx.ctx, 'update_settings', lambda *call: writes.append(call))

    MicroscopeSettings.select_sequenced_image_output_format(_hyperstack_panel())

    assert (picks, writes) == ([], [])
