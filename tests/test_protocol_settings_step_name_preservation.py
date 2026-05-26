# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for empty-step-name preservation in modify_step_ex.

Bug
---
Repro: blank labware -> add 2x Blue video step -> add another in a new
location -> change Step 2 to BF. The step name "custom0001" gets
replaced with an empty string.

Root cause: generate_step_name_input intentionally blanks
step_name_input.text for auto-named custom#### steps so the default
name shows in the hint instead. modify_step_ex then reads
step_name_input.text (empty) and passes it through to
Protocol.modify_step, which writes the empty string into the Name
column.

Fix
---
In modify_step_ex, treat an empty input as "no rename intended" and
preserve the existing step name from the protocol DataFrame. The guard
runs before the stim-was-active preservation path and before the
Protocol.modify_step call so any subsequent path sees the preserved
name, not the empty string.

Test approach
-------------
Source-level structural lock via AST: extract modify_step_ex's body and
assert the empty-name guard exists, reads from self._protocol.step,
and runs before the modify_step call. Behavioral exec is impractical
here -- modify_step_ex pulls in Kivy ids, _app_ctx, show_notification_popup,
ctx.stage, and several module-scope helpers; the mocking surface
overwhelms the signal.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    """Return the AST node for ClassName.method_name."""
    source = PROTOCOL_SETTINGS_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in source')


class TestModifyStepExStepNamePreservation:
    """Source-level lock on the empty-step-name guard in modify_step_ex."""

    def test_step_name_read_from_input_field(self):
        """Sanity-check the pre-existing read of step_name_input.text.
        If this disappears the guard's premise is gone and the bug
        cannot occur this way -- but neither can callers rename via
        the input field. Fail loud so the guard test below can be
        re-evaluated."""
        body_src = ast.unparse(_method_node('ProtocolSettings', 'modify_step_ex'))
        assert "self.ids['step_name_input'].text" in body_src, (
            'modify_step_ex must read step_name from '
            "self.ids['step_name_input'].text. If the input source moved, "
            'update this test and revisit the empty-name guard.'
        )

    def test_empty_step_name_guard_present(self):
        """modify_step_ex must contain an `if not step_name:` guard that
        falls back to the existing protocol step's Name. Without this,
        custom#### auto-named steps get clobbered to '' on Modify."""
        method = _method_node('ProtocolSettings', 'modify_step_ex')
        found_guard = False
        for node in ast.walk(method):
            if not isinstance(node, ast.If):
                continue
            # Match `if not step_name:`
            test = node.test
            if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
                continue
            if not (isinstance(test.operand, ast.Name) and test.operand.id == 'step_name'):
                continue
            # Body must assign to step_name from self._protocol.step(...)
            body_src = '\n'.join(ast.unparse(s) for s in node.body)
            assert 'step_name' in body_src and 'self._protocol.step' in body_src, (
                '`if not step_name:` guard must assign step_name from '
                "self._protocol.step(idx=self.curr_step)['Name']. Found "
                f'body: {body_src!r}'
            )
            found_guard = True
            break
        assert found_guard, (
            'modify_step_ex must contain `if not step_name:` guard that '
            'preserves the existing protocol step name. See class '
            'docstring for the custom#### clobber bug this prevents.'
        )

    def test_guard_runs_before_modify_step_call(self):
        """The guard must run before Protocol.modify_step(...) is called,
        otherwise modify_step still sees the empty string."""
        method = _method_node('ProtocolSettings', 'modify_step_ex')
        guard_lineno = None
        modify_call_lineno = None
        for node in ast.walk(method):
            if (
                guard_lineno is None
                and isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == 'step_name'
            ):
                guard_lineno = node.lineno
            if (
                modify_call_lineno is None
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'modify_step'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == '_protocol'
            ):
                modify_call_lineno = node.lineno
        assert guard_lineno is not None, 'step_name guard not found'
        assert modify_call_lineno is not None, (
            'self._protocol.modify_step(...) call not found in modify_step_ex'
        )
        assert guard_lineno < modify_call_lineno, (
            f'step_name guard at line {guard_lineno} must run before '
            f'self._protocol.modify_step call at line {modify_call_lineno}; '
            'otherwise modify_step still receives the empty step_name.'
        )
