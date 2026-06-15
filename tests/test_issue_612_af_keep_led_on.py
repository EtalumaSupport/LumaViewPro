# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#612 regression: AF in protocol must not redundantly cycle LED.

Bug
---
In a protocol with AF every Interval, the AF runner turned LED off
in its finally clause; then capture() immediately turned LED back on
at the SAME channel + illumination as the AF pass. ~50-200 ms +
extra mechanical cycle wasted per AF-every-N step.

Fix
---
AutofocusRunner.run gains a `keep_led_on: bool = False` parameter.
When True, the finally clause skips _led_off() + restore_led_state.
ProtocolStepRunner.scan_iterate sets keep_led_on=True when invoking
AF -- AF + capture in protocol context always share color +
illumination (the BF-AF-for-fluor branch retires AF entirely earlier
so the same-color invariant holds).

Interactive AF runs (non-protocol triggers) default to False, so
pre-AF state still restores as before.

Test approach
-------------
1. AST guard on AutofocusRunner.run signature: keep_led_on present
   with default False.
2. AST guard on the finally clause: a conditional path on
   self._keep_led_on that skips _led_off + restore_led_state.
3. AST guard on protocol_step_runner.run_autofocus invocation:
   keep_led_on=True is passed.

Direct exec is impractical (AutofocusRunner needs a Lumascope +
multiple executors).
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
RUNNER_SRC = REPO / 'modules' / 'autofocus_runner.py'
PSR_SRC = REPO / 'modules' / 'protocol_step_runner.py'


def _module_tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _method_node(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found')


def test_runner_run_accepts_keep_led_on_with_false_default():
    method = _method_node(_module_tree(RUNNER_SRC), 'AutofocusRunner', 'run')
    kwonly_names = [a.arg for a in method.args.kwonlyargs]
    pos_names = [a.arg for a in method.args.args]
    all_params = pos_names + kwonly_names
    assert 'keep_led_on' in all_params, (
        'AutofocusRunner.run must accept keep_led_on parameter. (#612)'
    )

    # Walk the entire signature paired with defaults to find keep_led_on.
    defaults_map = {}
    args = method.args
    # Pos args + their defaults (right-aligned).
    n_pos = len(args.args)
    n_defaults = len(args.defaults)
    for i, arg in enumerate(args.args):
        if i >= n_pos - n_defaults:
            defaults_map[arg.arg] = args.defaults[i - (n_pos - n_defaults)]
    # Kwonly args + kw_defaults are 1:1.
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        if default is not None:
            defaults_map[arg.arg] = default

    keep_default = defaults_map.get('keep_led_on')
    assert keep_default is not None, 'keep_led_on must have a default value. (#612)'
    assert isinstance(keep_default, ast.Constant) and keep_default.value is False, (
        f'keep_led_on default must be False (interactive AF preserves '
        f'pre-AF state); got {ast.unparse(keep_default)}. (#612)'
    )


def test_runner_stores_keep_led_on_on_self():
    method = _method_node(_module_tree(RUNNER_SRC), 'AutofocusRunner', 'run')
    src = ast.unparse(method)
    assert 'self._keep_led_on' in src and 'keep_led_on' in src, (
        'AutofocusRunner.run must store keep_led_on on self for the '
        'finally clause to consult. (#612)'
    )


def test_runner_finally_skips_led_off_when_flag_set():
    method = _method_node(_module_tree(RUNNER_SRC), 'AutofocusRunner', 'run')
    src = ast.unparse(method)  # noqa: F841 -- deferred
    # Find an If gating on self._keep_led_on; in its else branch the
    # _led_off + restore_led_state calls are present.
    has_keep_guard = False
    for node in ast.walk(method):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if 'self._keep_led_on' not in test_src:
                continue
            # The else branch must contain self._led_off().
            else_src = '\n'.join(ast.unparse(s) for s in node.orelse)
            if 'self._led_off()' in else_src and 'restore_led_state' in else_src:
                has_keep_guard = True
                break
    assert has_keep_guard, (
        'AutofocusRunner.run finally must gate the _led_off + '
        'restore_led_state pair on `if self._keep_led_on: ... else: ...` '
        'so the protocol-AF path can inherit the LED state. (#612)'
    )


def test_protocol_step_runner_passes_keep_led_on_true():
    psr_tree = _module_tree(PSR_SRC)
    # Find any Call to run_autofocus(...) and check its kwargs.
    matches = []
    for node in ast.walk(psr_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'run_autofocus'
        ):
            kw_map = {k.arg: k.value for k in node.keywords}
            matches.append(kw_map)
    assert matches, 'protocol_step_runner must invoke run_autofocus somewhere. (#612)'
    for kw in matches:
        assert 'keep_led_on' in kw, (
            'protocol_step_runner.run_autofocus call must include '
            'keep_led_on= so AF skips its off + restore cycle. (#612)'
        )
        val = kw['keep_led_on']
        assert isinstance(val, ast.Constant) and val.value is True, (
            f'keep_led_on must be True in protocol_step_runner; got {ast.unparse(val)}. (#612)'
        )
