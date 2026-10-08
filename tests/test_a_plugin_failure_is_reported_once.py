# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every way a plugin fails is reported once, by the plugin's name.

A plugin is separately versioned and may not be ours, so the host catches
its failure and carries on. Whether it failed to load or failed after
loading, the failure goes through one of the registry's two recorders,
which keep it for diagnostics and hand it to the one reporter: one log
record carrying the plugin's own traceback, and one notice whose title
names the plugin, so two plugins failing together are two notices.
"""

from __future__ import annotations

import ast
import logging
import sys
import types
from unittest.mock import patch

import pytest

import modules.notification_center as notification_center
import modules.plugins as plugins
from modules.exceptions import PluginFailedError, PluginNotLoadedError
from modules.notification_center import NotificationCenter, Severity
from modules.plugins import PluginSpec, ProcessorResult
from tests.ast_seams import REPO_ROOT, find_def
from tests.plugin_test_harness import harness_ctx  # noqa: F401 -- pytest fixture


@pytest.fixture
def shown(monkeypatch):
    # A centre of its own: the shared one's dedup window remembers what
    # earlier tests posted, and would swallow these notices. Installed on
    # the centre's module as well as the plugin module's binding, so the
    # host's crash guard and any late import see the same one.
    centre = NotificationCenter()
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.INFO)
    monkeypatch.setattr(notification_center, 'notifications', centre)
    monkeypatch.setattr(plugins, 'notifications', centre, raising=False)
    return seen


@pytest.fixture
def outcome_records(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


def _faults(caplog):
    return [r for r in caplog.records if r.name == 'LVP.outcomes' and r.levelno == logging.ERROR]


def _plugin_logger_errors(caplog):
    return [r for r in caplog.records if r.name == 'lvp_logger' and r.levelno >= logging.WARNING]


class _EntryPoint:
    def __init__(self, name, module=None, raises=None):
        self.name = name
        self._module = module
        self._raises = raises

    def load(self):
        if self._raises is not None:
            raise self._raises
        return self._module


def _module(name, *, requires='>=4.0.0', register=None, spec=True, **spec_kw):
    mod = types.ModuleType(f'fake_plugin_{name}')
    mod.__version__ = '0.1.0'
    if spec:
        mod.spec = PluginSpec(
            name=name,
            version='0.1.0',
            requires_lvp_version=requires,
            description=f'fake {name}',
            **spec_kw,
        )
    if register is not None:
        mod.register = register
    return mod


def _load(ctx, *eps):
    with patch('importlib.metadata.entry_points', return_value=list(eps)):
        ctx.plugins.load(ctx, '4.0.0')


def _assert_one_report(shown, caplog, kind, title, *, cause=None):
    assert [(n.severity, n.title) for n in shown] == [(Severity.ERROR, title)]
    faults = _faults(caplog)
    assert len(faults) == 1, [r.getMessage() for r in faults]
    reported = faults[0].exc_info[1]
    assert isinstance(reported, kind)
    assert reported.title == title
    if cause is not None:
        assert reported.__cause__ is cause
        assert faults[0].exc_info[1].__cause__.__traceback__ is not None
    assert _plugin_logger_errors(caplog) == [], 'a second log line beside the report'


# ---------------------------------------------------------------------------
# At load
# ---------------------------------------------------------------------------


def test_a_plugin_that_cannot_be_imported_is_reported(harness_ctx, shown, outcome_records):
    boom = ImportError('no module named numpy2')
    try:
        raise boom
    except ImportError:
        pass
    _load(harness_ctx, _EntryPoint('broken_import', raises=boom))

    _assert_one_report(
        shown, outcome_records, PluginNotLoadedError, 'Plugin Not Loaded: broken_import', cause=boom
    )
    assert 'broken_import' in [s.name for s in harness_ctx.plugins.not_loaded()]


def test_a_package_that_is_not_a_plugin_is_reported(harness_ctx, shown, outcome_records):
    _load(harness_ctx, _EntryPoint('not_a_plugin', _module('not_a_plugin', spec=False)))

    _assert_one_report(
        shown, outcome_records, PluginNotLoadedError, 'Plugin Not Loaded: not_a_plugin'
    )
    assert 'PluginSpec' in shown[0].message


def test_a_plugin_for_another_version_is_reported(harness_ctx, shown, outcome_records):
    mod = _module('too_new', requires='>=5.0.0', register=lambda ctx: None)
    _load(harness_ctx, _EntryPoint('too_new', mod))

    _assert_one_report(shown, outcome_records, PluginNotLoadedError, 'Plugin Not Loaded: too_new')
    assert '>=5.0.0' in shown[0].message and '4.0.0' in shown[0].message
    assert 'too_new' in [s.name for s in harness_ctx.plugins.not_loaded()]


def test_a_plugin_with_no_register_is_reported(harness_ctx, shown, outcome_records):
    _load(harness_ctx, _EntryPoint('no_register', _module('no_register')))

    _assert_one_report(
        shown, outcome_records, PluginNotLoadedError, 'Plugin Not Loaded: no_register'
    )


def test_a_plugin_whose_register_raises_is_reported(harness_ctx, shown, outcome_records):
    boom = RuntimeError('register exploded')

    def register(ctx):
        raise boom

    _load(harness_ctx, _EntryPoint('bad_register', _module('bad_register', register=register)))

    _assert_one_report(
        shown, outcome_records, PluginNotLoadedError, 'Plugin Not Loaded: bad_register', cause=boom
    )
    assert 'register exploded' in shown[0].message


def test_two_plugins_failing_together_are_both_shown(harness_ctx, shown):
    _load(
        harness_ctx,
        _EntryPoint('alpha', raises=ImportError('a')),
        _EntryPoint('beta', raises=ImportError('b')),
    )

    assert [n.title for n in shown] == ['Plugin Not Loaded: alpha', 'Plugin Not Loaded: beta']


# ---------------------------------------------------------------------------
# After load
# ---------------------------------------------------------------------------


def _loaded_post_processor(ctx, name, processor, **spec_kw):
    """A plugin registered into the post_processing namespace and tracked as loaded."""
    mod = _module(name, **spec_kw)
    ctx.plugins.post_processing.register(mod.spec, processor)
    ctx.plugins._track(name, mod)
    return mod


def _raised(exc):
    try:
        raise exc
    except type(exc) as caught:
        return caught


@pytest.mark.parametrize('hook', ['ui_event', 'mount'])
def test_a_ui_crash_or_a_failed_mount_is_reported(harness_ctx, shown, outcome_records, hook):
    _loaded_post_processor(harness_ctx, 'crashy', lambda *a: None)
    boom = _raised(ValueError('handler bug'))

    harness_ctx.plugins.record_runtime_error('crashy', hook, boom)

    _assert_one_report(
        shown, outcome_records, PluginFailedError, 'Plugin Error: crashy', cause=boom
    )
    errors = harness_ctx.plugins.post_processing.health().last_runtime_errors
    assert [(e.plugin_name, e.hook, e.exc_type) for e in errors] == [('crashy', hook, 'ValueError')]


def test_a_settings_handler_that_raises_is_reported(harness_ctx, shown, outcome_records):
    mod = _loaded_post_processor(harness_ctx, 'listener', lambda *a: None, subscribes_to=('a',))
    boom = RuntimeError('settings handler bug')

    def on_settings_changed(ctx, settings):
        raise boom

    mod.on_settings_changed = on_settings_changed

    harness_ctx.plugins.notify_settings_changed(harness_ctx, {'a': 1}, ['a'])

    _assert_one_report(
        shown, outcome_records, PluginFailedError, 'Plugin Error: listener', cause=boom
    )
    errors = harness_ctx.plugins.post_processing.health().last_runtime_errors
    assert [e.hook for e in errors] == ['on_settings_changed']


def test_a_plugin_in_no_namespace_is_still_reported(harness_ctx, shown, outcome_records):
    mod = _module('homeless', subscribes_to=('a',))

    def on_settings_changed(ctx, settings):
        raise RuntimeError('no namespace')

    mod.on_settings_changed = on_settings_changed
    harness_ctx.plugins._track('homeless', mod)

    harness_ctx.plugins.notify_settings_changed(harness_ctx, {'a': 1}, ['a'])

    assert [n.title for n in shown] == ['Plugin Error: homeless']


def _run_complete(ctx, tmp_path):
    ctx.plugins.run_protocol_complete_processors(str(tmp_path), {}, str(tmp_path), 'written')


def test_a_post_run_processor_that_raises_is_reported(
    harness_ctx, shown, outcome_records, tmp_path
):
    boom = RuntimeError('processor bug')

    def processor(*_):
        raise boom

    _loaded_post_processor(harness_ctx, 'auto_bad', processor, auto_run_on_protocol_complete=True)
    _run_complete(harness_ctx, tmp_path)

    _assert_one_report(
        shown, outcome_records, PluginFailedError, 'Plugin Error: auto_bad', cause=boom
    )


def test_a_post_run_processor_returning_the_wrong_type_is_reported(
    harness_ctx, shown, outcome_records, tmp_path
):
    _loaded_post_processor(
        harness_ctx, 'auto_wrong', lambda *_: 'done', auto_run_on_protocol_complete=True
    )
    _run_complete(harness_ctx, tmp_path)

    _assert_one_report(shown, outcome_records, PluginFailedError, 'Plugin Error: auto_wrong')
    assert 'str' in shown[0].message


def test_a_post_run_processor_reporting_failure_is_shown_in_its_words(
    harness_ctx, shown, outcome_records, tmp_path
):
    _loaded_post_processor(
        harness_ctx,
        'auto_failed',
        lambda *_: ProcessorResult(success=False, message='no tiles found in the run folder'),
        auto_run_on_protocol_complete=True,
    )
    _run_complete(harness_ctx, tmp_path)

    _assert_one_report(shown, outcome_records, PluginFailedError, 'Plugin Error: auto_failed')
    assert 'no tiles found in the run folder' in shown[0].message


def test_a_post_run_processor_that_succeeds_is_not_reported(harness_ctx, shown, tmp_path):
    _loaded_post_processor(
        harness_ctx,
        'auto_ok',
        lambda *_: ProcessorResult(success=True, message='stitched'),
        auto_run_on_protocol_complete=True,
    )
    _run_complete(harness_ctx, tmp_path)

    assert shown == []


# ---------------------------------------------------------------------------
# The host's two catch sites hand the failure to the recorder
# ---------------------------------------------------------------------------


class _ExceptionManager:
    RAISE = 'raise'
    PASS = 'pass'


def _crash_guard(session):
    """The REAL handle_exception body, compiled out of lumaviewpro.py.

    The guard is a class nested in build(), and lumaviewpro.py is not
    importable under the harness, so its FunctionDef is compiled with only
    the module names it closes over supplied: the GUI's context, which
    reaches the plugins through its session.
    """
    node = find_def('lumaviewpro.py', 'handle_exception', class_name='_PluginCrashGuard')
    assert node is not None, 'lumaviewpro.py: _PluginCrashGuard.handle_exception is gone'
    namespace = {
        'ctx': types.SimpleNamespace(session=session),
        'sys': sys,
        'ExceptionManager': _ExceptionManager,
        'logger': logging.getLogger('lvp_logger'),
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(REPO_ROOT / 'lumaviewpro.py'), 'exec'), namespace)
    return namespace['handle_exception']


def test_the_crash_guard_reports_a_plugin_crash_through_the_recorder(
    harness_ctx, shown, outcome_records, monkeypatch
):
    _loaded_post_processor(harness_ctx, 'crashy', lambda *a: None)
    monkeypatch.setattr(harness_ctx.plugins, 'attribute_exception', lambda tb: 'crashy')
    handle = _crash_guard(harness_ctx)

    try:
        raise ValueError('button handler bug')
    except ValueError as crash:
        caught = crash
        answer = handle(None, crash)

    assert answer == _ExceptionManager.PASS
    _assert_one_report(
        shown, outcome_records, PluginFailedError, 'Plugin Error: crashy', cause=caught
    )


def test_the_crash_guard_leaves_our_own_crashes_to_raise(harness_ctx, shown, monkeypatch):
    monkeypatch.setattr(harness_ctx.plugins, 'attribute_exception', lambda tb: None)
    handle = _crash_guard(harness_ctx)

    try:
        raise ValueError('core bug')
    except ValueError as crash:
        answer = handle(None, crash)

    assert answer == _ExceptionManager.RAISE
    assert shown == []


def test_a_failed_mount_goes_to_the_recorder():
    """The mount loop runs inside build(), with the widget tree; what can be
    held is its shape: the handler around a plugin's builder() hands the
    failure to the recorder and does nothing else with it."""
    build = find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')
    assert build is not None
    handlers = [
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'builder'
            for stmt in node.body
            for call in ast.walk(stmt)
        )
    ]
    assert len(handlers) == 1, 'expected one try around a plugin builder() call'
    (handler,) = handlers[0].handlers
    calls = [
        call.func.attr
        for call in ast.walk(handler)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]
    assert calls == ['record_runtime_error'], calls


def test_a_plugin_that_never_loaded_is_in_no_namespace(harness_ctx, shown, outcome_records):
    # A plugin is filed under a namespace by registering, which this one
    # never did: its record is the registry's own, not the ui namespace's.
    _load(harness_ctx, _EntryPoint('no_register', _module('no_register')))

    not_loaded = harness_ctx.plugins.not_loaded()
    assert [entry.name for entry in not_loaded] == ['no_register']
    assert not_loaded[0].reason
    for health in harness_ctx.plugins.all_health():
        assert 'no_register' not in [status.name for status in health.loaded]
