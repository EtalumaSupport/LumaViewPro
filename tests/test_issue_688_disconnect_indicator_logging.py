"""Regression: the on-screen no-camera indicator is the only signal a user
gets when the camera drops (there is no popup), yet the show/restore
transitions were not logged -- a support bundle could not show the moment
the user saw the indicator.

ScopeDisplay subclasses kivy.uix.image.Image, which the test harness stubs
with a MagicMock (tests/conftest.py), so the widget cannot be imported or
constructed headless. Following the source-assertion pattern used for other
un-constructable widget/driver internals (test_audit_fixes.py), these tests
read the two method sources and assert each still emits its INFO record. They
fail before the logging was added and pass after; they fail again if a future
edit drops either log line.
"""

import ast
from pathlib import Path

_SCOPE_DISPLAY = Path(__file__).resolve().parents[1] / 'ui' / 'scope_display.py'


def _method_source(method_name):
    tree = ast.parse(_SCOPE_DISPLAY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(_SCOPE_DISPLAY.read_text(), node)
    raise AssertionError(f'{method_name} not found in {_SCOPE_DISPLAY}')


def _info_log_messages(method_name):
    """Return the format-string of every logger.info(...) call in the method."""
    src = _method_source(method_name)
    messages = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == 'info'
            and isinstance(func.value, ast.Name)
            and func.value.id == 'logger'
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            messages.append(node.args[0].value)
    return messages


def test_showing_no_camera_indicator_is_logged():
    messages = _info_log_messages('set_camera_disconnected_display')
    assert any('no-camera indicator' in m for m in messages), messages


def test_live_view_restored_is_logged():
    messages = _info_log_messages('source_clear')
    assert any('live view restored' in m for m in messages), messages
