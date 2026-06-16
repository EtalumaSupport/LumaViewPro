"""#696 regression: the same-channel keep-LED-on optimization is opt-in.

In a single-channel protocol the LED stayed lit across every step because
the runner unconditionally held the LED on between consecutive same-color
steps (a speed optimization). The behavior is now gated on a
keep_led_between_steps flag that defaults False, so the LED extinguishes
between steps unless the optimization is explicitly enabled.

Structural guards (the decision lives inline in a step-capture method that
needs a full Lumascope + executors to exec, so it is pinned by AST, the
same approach as the #612 / #524 step-runner guards):
1. SequencedCaptureRunner.run accepts keep_led_between_steps, default False.
2. run stores it on self for the step runner to read.
3. protocol_step_runner gates the same-color hold on _keep_led_between_steps.
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
    # An If on _keep_led_between_steps must wrap the same-color _keep_led=True
    # assignment, so the hold cannot fire when the flag is False.
    found = False
    for node in ast.walk(_tree(PSR_SRC)):
        if (
            isinstance(node, ast.If)
            and '_keep_led_between_steps' in ast.unparse(node.test)
            and '_keep_led = True' in '\n'.join(ast.unparse(s) for s in node.body)
        ):
            found = True
            break
    assert found, (
        'protocol_step_runner must gate the same-color _keep_led=True hold on '
        'p._keep_led_between_steps'
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
