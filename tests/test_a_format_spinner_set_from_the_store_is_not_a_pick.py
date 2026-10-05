# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A format spinner set from the store is not a pick: nothing is logged or written.

The settings load sets both image-format spinners from the store, and the kv's
``on_text`` fires their pick handlers for it. Each start-up then recorded two
picks nobody made and wrote the stored formats back through the writer.

Handlers are called unbound on a stand-in panel rather than instantiating Kivy
widgets.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_HANDLERS = [
    ('select_live_image_output_format', 'live_image_output_format_spinner', 'live'),
    ('select_sequenced_image_output_format', 'sequenced_image_output_format_spinner', 'sequenced'),
]


@pytest.fixture
def recorded(monkeypatch):
    import modules.app_context as _app_ctx
    import ui.microscope_settings as microscope_settings

    picks, writes = [], []
    monkeypatch.setattr(
        microscope_settings.gui_logger, 'select', lambda name, value: picks.append(value)
    )
    monkeypatch.setattr(microscope_settings, 'run_reported', lambda call, redraw, label: call())
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings={'image_output_format': {'live': 'TIFF', 'sequenced': 'TIFF'}},
            update_settings=MagicMock(side_effect=lambda path, value: writes.append(path)),
        ),
    )
    return picks, writes


@pytest.mark.parametrize('handler, spinner, key', _HANDLERS)
def test_the_stored_format_is_not_logged_or_written(recorded, handler, spinner, key):
    from ui.microscope_settings import MicroscopeSettings

    panel = SimpleNamespace(ids={spinner: SimpleNamespace(text='TIFF')})
    getattr(MicroscopeSettings, handler)(panel)

    assert recorded == ([], [])


@pytest.mark.parametrize('handler, spinner, key', _HANDLERS)
def test_a_picked_format_is_logged_and_written(recorded, handler, spinner, key):
    from ui.microscope_settings import MicroscopeSettings

    panel = SimpleNamespace(ids={spinner: SimpleNamespace(text='JPG')})
    getattr(MicroscopeSettings, handler)(panel)

    assert recorded == (['JPG'], [f'image_output_format.{key}'])
