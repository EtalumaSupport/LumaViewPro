"""#696 regression: the same-channel keep-LED-on optimization is opt-in.

In a single-channel protocol the LED stayed lit across every step because
the runner unconditionally held the LED on between consecutive same-color
steps (a speed optimization). The behavior is now gated on a
keep_led_between_steps flag that defaults False, so the LED extinguishes
between steps unless the optimization is explicitly enabled.

The hold policy now lives in the LED authority's STEP_BOUNDARY decision
(hold within a z-stack group, or across a same-color move only when the
opt-in is on); that pure function is tested directly in
test_led_authority_skeleton. What stays AST-pinned here is the flag's
plumbing and the step runner's WIRING into that decision -- the step-capture
method that builds the ctx needs a full Lumascope + executors to exec:
1. SequencedCaptureRunner.run accepts keep_led_between_steps, default False.
2. run stores it on self for the step runner to read.
3. protocol_step_runner feeds the flag into the STEP_BOUNDARY decision as
   keep_led_across_moves, and consults Z-Stack Group ID for same_zstack_group.
4. protocol_runner passes the value from settings, defaulting False.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNNER_SRC = REPO / 'modules' / 'sequenced_capture_runner.py'
PSR_SRC = REPO / 'modules' / 'protocol_step_runner.py'
PROTO_RUNNER_SRC = REPO / 'modules' / 'protocol_runner.py'


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _method_node(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found')


def _default_for(method: ast.FunctionDef, arg_name: str):
    args = method.args
    n_pos = len(args.args)
    n_defaults = len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg == arg_name and i >= n_pos - n_defaults:
            return args.defaults[i - (n_pos - n_defaults)]
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        if arg.arg == arg_name:
            return default
    return None


def test_run_accepts_keep_led_between_steps_default_false():
    method = _method_node(_tree(RUNNER_SRC), 'SequencedCaptureRunner', 'run')
    names = [a.arg for a in method.args.args] + [a.arg for a in method.args.kwonlyargs]
    assert 'keep_led_between_steps' in names, (
        'SequencedCaptureRunner.run must accept keep_led_between_steps'
    )
    default = _default_for(method, 'keep_led_between_steps')
    assert isinstance(default, ast.Constant) and default.value is False, (
        'keep_led_between_steps must default to False so the LED turns off '
        'between steps unless the optimization is opted in'
    )


def test_run_stores_flag_on_self():
    method = _method_node(_tree(RUNNER_SRC), 'SequencedCaptureRunner', 'run')
    assert 'self._keep_led_between_steps = keep_led_between_steps' in ast.unparse(method)


def test_step_runner_gates_same_color_hold_on_flag():
    # The same-color hold must be gated on BOTH an actual color comparison and
    # the opt-in flag, so it cannot fire across a color-switching move. Pin both
    # inside the step method (not anywhere in the file): same_color derives from
    # the next vs current Color, and the flag flows in as keep_led_across_moves.
    method = _method_node(_tree(PSR_SRC), 'ProtocolStepRunner', 'scan_iterate')
    src = ast.unparse(method)
    assert (
        "next_step['Color'] == step['Color']" in src or "step['Color'] == next_step['Color']" in src
    ), 'step runner must derive same_color from the next vs current step Color'
    flag_wired = any(
        isinstance(node, ast.keyword)
        and node.arg == 'keep_led_across_moves'
        and '_keep_led_between_steps' in ast.unparse(node.value)
        for node in ast.walk(method)
    )
    assert flag_wired, (
        'step runner must pass p._keep_led_between_steps as keep_led_across_moves '
        'into the STEP_BOUNDARY decision, so the same-color hold honors the flag'
    )


def test_step_runner_consults_zstack_group_for_boundary_decision():
    # Slices of one z-stack (same Z-Stack Group ID) are a single acquisition, so
    # the LED holds across the Z moves even with the opt-in off -- otherwise the
    # sample blinks on every slice. The hold policy lives in the authority's
    # STEP_BOUNDARY predicate (z-stack OR same-color opt-in, tested directly in
    # test_led_authority_skeleton); pin here that the step method consults the
    # Z-Stack Group ID in code and feeds same_zstack_group into the decision.
    method = _method_node(_tree(PSR_SRC), 'ProtocolStepRunner', 'scan_iterate')
    src = ast.unparse(method)
    assert 'Z-Stack Group ID' in src, (
        'step runner must consult Z-Stack Group ID to hold the LED across z-stack slices'
    )
    fed = any(
        isinstance(node, ast.keyword) and node.arg == 'same_zstack_group'
        for node in ast.walk(method)
    )
    assert fed, (
        'protocol_step_runner must feed same_zstack_group into the STEP_BOUNDARY '
        'decision so z-stack slices hold the LED regardless of the opt-in flag'
    )


def test_protocol_runner_passes_flag_from_settings_default_false():
    for node in ast.walk(_tree(PROTO_RUNNER_SRC)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'run'
        ):
            kw = {k.arg: k.value for k in node.keywords}
            if 'keep_led_between_steps' in kw:
                src = ast.unparse(kw['keep_led_between_steps'])
                assert "'keep_led_between_steps'" in src and 'False' in src, (
                    'protocol_runner must read keep_led_between_steps from '
                    'settings with a False default'
                )
                return
    raise AssertionError('protocol_runner must pass keep_led_between_steps to the runner')
