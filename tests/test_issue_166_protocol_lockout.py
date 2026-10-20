"""Regression tests for the protocol-running UI lockout (issue #166, #704).

The App holds the Session's lock record (``app.lock``) so kv ``disabled:``
bindings grey out interactive controls during a scan. It must be written at
every run-state edge, from the Session's listener, or controls would either
never lock or stay stuck disabled after a scan. These assert on source
structure (the suite mocks Kivy; widgets are not built), ruff-format-agnostic.
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (REPO / rel).read_text()


def _def_source(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def test_mirror_is_listener_published_not_caller_pushed():
    """The App's lock record is written by its run-state publisher, fed
    by the session's transition listener -- no caller push helper
    remains. A reappearing per-caller publisher is a second writer for
    the record and the drift it caused."""
    helpers = _read('ui/ui_helpers.py')
    assert 'def publish_protocol_running' not in helpers
    app_src = _read('lumaviewpro.py')
    assert 'def publish_run_state' in app_src
    assert 'add_run_state_listener' in app_src


def test_session_exposes_is_protocol_running_accessor():
    body = _def_source(_read('modules/scope_session.py'), 'is_protocol_running')
    assert body is not None, 'ScopeSession.is_protocol_running accessor missing'
    # The claim's run: its holder when that is a run, or a run lent a
    # diagnostic's claim -- a lent run is still a run in progress.
    assert 'activity_claim.run_holder is not None' in body
