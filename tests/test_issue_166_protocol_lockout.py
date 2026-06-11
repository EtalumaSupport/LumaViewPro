"""Regression tests for the protocol-running UI lockout (issue #166, #704).

A `protocol_running` BooleanProperty on the App mirrors the
ctx.protocol_running Event so kv `disabled:` bindings grey out interactive
controls during a scan. The property must be published True at every run
start and False at every run-reset (the abort-safe convergence point), or
controls would either never lock or stay stuck disabled after a scan. These
assert on source structure (the suite mocks Kivy; widgets are not built),
ruff-format-agnostic.
"""

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (REPO / rel).read_text()


def _def_source(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def test_app_defines_protocol_running_boolean_property():
    src = _read('lumaviewpro.py')
    tree = ast.parse(src)
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == 'LumaViewProApp'
    )
    found = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'protocol_running' for t in node.targets)
        and 'BooleanProperty' in ast.unparse(node.value)
        for node in cls.body
    )
    assert found, 'LumaViewProApp must define protocol_running = BooleanProperty(...)'


def test_publish_helper_schedules_flip_on_main_thread():
    body = _def_source(_read('ui/protocol_settings.py'), '_publish_protocol_running')
    assert body is not None, '_publish_protocol_running helper missing'
    assert 'Clock.schedule_once' in body, 'the property flip must run on the Kivy main thread'
    assert 'protocol_running' in body


def test_every_run_start_publishes_true():
    src = _read('ui/protocol_settings.py')
    for method in (
        '_run_scan_from_ui_inner',
        '_run_protocol_from_ui_inner',
        'run_autofocus_scan_from_ui',
    ):
        body = _def_source(src, method)
        assert body is not None, f'{method} missing'
        assert re.search(r'_publish_protocol_running\(\s*True\s*\)', body), (
            f'{method} must publish protocol_running True at run start'
        )


def test_every_run_reset_publishes_false():
    src = _read('ui/protocol_settings.py')
    for method in (
        '_reset_run_scan_button',
        '_reset_run_protocol_button',
        '_reset_run_autofocus_scan_button',
    ):
        body = _def_source(src, method)
        assert body is not None, f'{method} missing'
        assert re.search(r'_publish_protocol_running\(\s*False\s*\)', body), (
            f'{method} must publish protocol_running False on reset (abort-safe clear)'
        )


def test_session_exposes_is_protocol_running_accessor():
    body = _def_source(_read('modules/scope_session.py'), 'is_protocol_running')
    assert body is not None, 'ScopeSession.is_protocol_running accessor missing'
    assert 'protocol_running.is_set' in body.replace(' ', '')
