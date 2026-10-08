# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A finished Full Protocol hands its folder to the opted-in plugins once.

The engine ends every run with run_ended and then files_written -- the
second sent once by the run's write batch when it completes, after the
run's files are written or abandoned. The auto-run post-processing
belongs to files_written, the one that always comes after the files, so
run_ended never dispatches and each run's plugins run once.
"""

import sys
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
import ui.protocol_settings as ps
from ui.protocol_settings import ProtocolSettings


class _Stand:
    """The panel's real Run starter, its events and dispatcher over hand-built widget state."""

    _protocol_start = ProtocolSettings._protocol_start
    _sequenced_capture_start = ProtocolSettings._sequenced_capture_start
    _dispatch_post_processing_auto_run = ProtocolSettings._dispatch_post_processing_auto_run

    def __init__(self):
        self._protocol = MagicMock()
        self.ids = {'protocol_filename': SimpleNamespace(text='seq')}
        self._runs_started_here = {}

    def reset_autofocus_ui(self):
        pass

    def draw_protocol_buttons(self):
        pass

    def _run_scan_pre_callback(self):
        pass


@pytest.fixture
def dispatched(monkeypatch):
    """Each run folder the panel hands to the plugin dispatcher, with its files outcome."""
    import modules.plugins as plugins

    calls = []
    monkeypatch.setattr(
        plugins,
        'run_protocol_complete_processors',
        lambda ctx, *, input_dir, manifest, output_dir, files: calls.append((input_dir, files)),
    )
    for name in ('restore_display_after_run', 'set_last_save_folder'):
        monkeypatch.setattr(ps, name, lambda *a, **k: None)
    monkeypatch.setattr(ps, 'is_image_saving_enabled', lambda: True)
    return calls


def _start_run(monkeypatch, stand, run_dir):
    """Press Run on *stand*; the events its run was handed."""
    started = {}

    def _run_protocol(protocol, **kw):
        started['events'] = kw['events']
        return SimpleNamespace(run_dir=run_dir)

    session = SimpleNamespace(
        create_protocol_runner=lambda: SimpleNamespace(run_protocol=_run_protocol)
    )
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session, engineering_mode=False))
    stand._protocol_start()()
    return started['events']


def _end_run(monkeypatch, stand, run_dir, files='written'):
    """The engine's order: run_ended, then files_written."""
    events = _start_run(monkeypatch, stand, run_dir)
    events.run_ended(MagicMock(status='completed'), run_dir, MagicMock())
    events.files_written(run_dir, files)


def test_the_finished_run_is_post_processed_once(dispatched, monkeypatch):
    _end_run(monkeypatch, _Stand(), '/run/1')

    assert dispatched == [('/run/1', 'written')], (
        f'the run folder must reach the opted-in plugins exactly once; got {dispatched}'
    )


def test_back_to_back_runs_each_dispatch_their_own_auto_run(dispatched, monkeypatch):
    """Each run's files_written reaches the plugins with its own folder, the
    second run's no less than the first's: the panel holds no per-run state
    that one run's completion could consume for the next."""
    stand = _Stand()
    _end_run(monkeypatch, stand, '/run/1')
    _end_run(monkeypatch, stand, '/run/2')

    assert dispatched == [('/run/1', 'written'), ('/run/2', 'written')], (
        f'every run folder must reach the opted-in plugins exactly once; got {dispatched}'
    )


@pytest.mark.parametrize('files', ['written', 'incomplete'])
def test_the_runs_files_outcome_reaches_the_plugins_as_the_batch_gave_it(
    dispatched, monkeypatch, files
):
    """The plugins decide from the outcome whether the folder is whole; the
    panel passes on what the run's write batch reported, never a value of
    its own, so a folder missing images is never built from."""
    _end_run(monkeypatch, _Stand(), '/run/1', files=files)

    assert dispatched == [('/run/1', files)]
