# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: UI call sites must gate started-run follow-ups on run().

Bug
---
SequencedCaptureRunner.run() refuses to start for several reasons
(hardware disconnected, files still writing, empty or invalid
protocol) and used to return None either way. Both UI call sites
treated the call as fire-and-forget and continued into their
started-run follow-ups:

- ProtocolSettings.run_sequenced_capture called run_dir() and
  protocol_interval() on a runner that never loaded a protocol ->
  AttributeError inside a UI handler (the bench protocol-start crash).
- ZStack.run_zstack_acquire_from_ui pointed set_last_save_folder at
  the PREVIOUS run's directory, silently landing captures in stale
  data.

Fix
---
run() returns bool on every exit path; both UI call sites capture the
result and gate their follow-ups on it.

Test approach
-------------
The Kivy UI classes cannot be instantiated headlessly (ids, _app_ctx,
worker pool), so this locks the call-site structure via AST, in the
same style as the #680 empty-protocol guard tests: the run() result
must be captured and the follow-up must be gated behind it, ordered
before/around set_last_save_folder. The behavioral half of the
contract (what run() returns and what the getters answer) lives in
test_protocol_execution.py::TestRunReturnValueContract.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent


def _method_node(source_file: pathlib.Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {source_file}')


def _assigns_run_result(node) -> bool:
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == 'run'
        and any(isinstance(t, ast.Name) and t.id == 'started' for t in node.targets)
    )


def test_run_sequenced_capture_gates_followups_on_started():
    """A refused run must return before run_dir()/protocol_interval()."""
    method = _method_node(
        REPO / 'ui' / 'protocol_settings.py', 'ProtocolSettings', 'run_sequenced_capture'
    )
    statements = list(ast.walk(method))

    assert any(_assigns_run_result(s) for s in statements), (
        'run_sequenced_capture must capture the runner.run() result '
        '(started = ...run(...)) instead of fire-and-forget'
    )

    guards = [
        s
        for s in statements
        if isinstance(s, ast.If)
        and isinstance(s.test, ast.UnaryOp)
        and isinstance(s.test.op, ast.Not)
        and isinstance(s.test.operand, ast.Name)
        and s.test.operand.id == 'started'
        and any(isinstance(inner, ast.Return) for inner in ast.walk(s))
    ]
    assert guards, (
        'run_sequenced_capture must return early on a refused run '
        '(if not started: return) before calling run_dir() or '
        'protocol_interval() on a runner that never loaded a protocol'
    )

    src = ast.unparse(method)
    guard_pos = src.index('if not started')
    assert 'set_last_save_folder' in src
    assert guard_pos < src.index('set_last_save_folder'), (
        'The refusal gate must run BEFORE set_last_save_folder, or a '
        'refused run points the save folder at the previous run'
    )


def test_run_zstack_acquire_gates_save_folder_on_started():
    """Z-stack must only record a save folder for a run that started."""
    method = _method_node(REPO / 'ui' / 'zstack.py', 'ZStack', 'run_zstack_acquire_from_ui')
    statements = list(ast.walk(method))

    assert any(_assigns_run_result(s) for s in statements), (
        'run_zstack_acquire_from_ui must capture the runner.run() result '
        '(started = ...run(...)) instead of fire-and-forget'
    )

    gated = [
        s
        for s in statements
        if isinstance(s, ast.If)
        and isinstance(s.test, ast.Name)
        and s.test.id == 'started'
        and 'set_last_save_folder' in ast.unparse(s)
    ]
    assert gated, (
        'set_last_save_folder must be gated behind the started result; '
        'on a refused run run_dir() answers for the PREVIOUS run and '
        'silently points the save folder at stale data'
    )
