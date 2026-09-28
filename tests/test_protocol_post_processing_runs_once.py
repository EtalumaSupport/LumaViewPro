# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A finished Full Protocol hands its folder to the opted-in plugins once.

The engine ends every run with run_complete and then files_complete --
the second deferred until the file writer drains, or fired at once when
nothing is left to write. The auto-run post-processing belongs to the
one that always comes after the files are on disk, so a run with no
pending writes does not dispatch from both and run each plugin twice.
"""

import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())

for _name, _attr in (
    ('kivy.uix.floatlayout', 'FloatLayout'),
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.scrollview', 'ScrollView'),
    ('kivy.uix.widget', 'Widget'),
):
    if _name not in sys.modules:
        _mod = ModuleType(_name)
        setattr(_mod, _attr, _StubWidget)
        sys.modules[_name] = _mod

import modules.app_context as _app_ctx
from ui.protocol_settings import ProtocolSettings


class _Stand:
    """The panel's completion handlers over hand-built widget state."""

    _protocol_run_complete = ProtocolSettings._protocol_run_complete
    _protocol_files_complete = ProtocolSettings._protocol_files_complete
    _update_protocol_write_status = ProtocolSettings._update_protocol_write_status

    def __init__(self):
        self._scan_files_completed_event = threading.Event()
        self._file_write_status_event = None
        self._pending_run_dir = None
        self.dispatched = []

    def _panel_run_ended(self):
        pass

    def _dispatch_post_processing_auto_run(self, ctx, **kwargs):
        self.dispatched.append(kwargs.get('run_dir'))


@pytest.fixture
def stand(monkeypatch):
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(protocol_files_draining=False))
    )
    return _Stand()


def _end_run(stand, files_pending):
    """The engine's order: run_complete, then files_complete."""
    _app_ctx.ctx.session.protocol_files_draining = files_pending
    protocol = MagicMock()
    stand._protocol_run_complete(protocol=protocol, status='completed', run_dir='/run/1')
    _app_ctx.ctx.session.protocol_files_draining = False
    stand._protocol_files_complete(protocol=protocol, run_dir='/run/1')


@pytest.mark.parametrize('files_pending', [False, True], ids=['nothing-to-write', 'files-draining'])
def test_the_finished_run_is_post_processed_once(stand, files_pending):
    _end_run(stand, files_pending)

    assert stand.dispatched == ['/run/1'], (
        f'the run folder must reach the opted-in plugins exactly once; got {stand.dispatched}'
    )
