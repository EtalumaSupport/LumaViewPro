# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A finished Full Protocol hands its folder to the opted-in plugins once.

The engine ends every run with run_complete and then files_complete --
the second sent once by the run's write batch when it completes, after
the run's files are written or abandoned. The auto-run post-processing
belongs to files_complete, the one that always comes after the files,
so run_complete never dispatches and each run's plugins run once.
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
from ui.protocol_settings import ProtocolSettings


class _Stand:
    """The panel's real completion handlers and dispatcher over hand-built widget state."""

    _protocol_run_complete = ProtocolSettings._protocol_run_complete
    _protocol_files_complete = ProtocolSettings._protocol_files_complete
    _dispatch_post_processing_auto_run = ProtocolSettings._dispatch_post_processing_auto_run

    def reset_autofocus_ui(self):
        pass


@pytest.fixture
def dispatched(monkeypatch):
    """Each run folder the panel hands to the plugin dispatcher, with its files outcome."""
    import modules.plugins as plugins

    calls = []
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace())
    monkeypatch.setattr(
        plugins,
        'run_protocol_complete_processors',
        lambda ctx, *, input_dir, manifest, output_dir, files: calls.append((input_dir, files)),
    )
    return calls


def _end_run(stand, run_dir):
    """The engine's order: run_complete, then files_complete."""
    protocol = MagicMock()
    stand._protocol_run_complete(protocol=protocol, status='completed', run_dir=run_dir)
    stand._protocol_files_complete(protocol=protocol, run_dir=run_dir, files='written')


def test_the_finished_run_is_post_processed_once(dispatched):
    _end_run(_Stand(), '/run/1')

    assert dispatched == [('/run/1', 'written')], (
        f'the run folder must reach the opted-in plugins exactly once; got {dispatched}'
    )


def test_back_to_back_runs_each_dispatch_their_own_auto_run(dispatched):
    """Each run's files_complete reaches the plugins with its own folder, the
    second run's no less than the first's: the panel holds no per-run state
    that one run's completion could consume for the next."""
    stand = _Stand()
    _end_run(stand, '/run/1')
    _end_run(stand, '/run/2')

    assert dispatched == [('/run/1', 'written'), ('/run/2', 'written')], (
        f'every run folder must reach the opted-in plugins exactly once; got {dispatched}'
    )


@pytest.mark.parametrize('files', ['written', 'abandoned'])
def test_the_runs_files_outcome_reaches_the_plugins_as_the_batch_gave_it(dispatched, files):
    """The plugins decide from the outcome whether the folder is whole; the
    panel passes on what the run's write batch reported, never a value of
    its own, so a folder missing images is never built from."""
    _Stand()._protocol_files_complete(protocol=MagicMock(), run_dir='/run/1', files=files)

    assert dispatched == [('/run/1', files)]
