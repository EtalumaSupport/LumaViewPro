"""#696 regression: the same-channel keep-LED-on optimization is opt-in.

In a single-channel protocol the LED stayed lit across every step because
the runner unconditionally held the LED on between consecutive same-color
steps (a speed optimization). The behavior is now gated on a
keep_led_between_steps flag that defaults False, so the LED extinguishes
between steps unless the optimization is explicitly enabled.

The hold policy now lives in the LED authority's STEP_BOUNDARY decision
(hold within a z-stack group, or across a same-color move only when the
opt-in is on); that pure function is tested directly in
test_led_authority_skeleton, and the step runner's use of it through real
runs on the simulated scope in test_led_lifecycle_sequence (s1-s4: dark
between same-colour steps by default, held with the opt-in, held across a
z-stack's slices whatever the flag, dark across a colour switch). What stays
AST-pinned here is the flag's plumbing:
1. SequencedCaptureRunner.prepare accepts keep_led_between_steps, default
   False.
2. start stores the plan's value on self for the step runner to read.
3. protocol_runner passes the value from settings, defaulting False.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNNER_SRC = REPO / 'src/_lumascope/modules' / 'sequenced_capture_runner.py'
PROTO_RUNNER_SRC = REPO / 'src/_lumascope/modules' / 'protocol_runner.py'


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'))


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
    method = _method_node(_tree(RUNNER_SRC), 'SequencedCaptureRunner', 'prepare')
    names = [a.arg for a in method.args.args] + [a.arg for a in method.args.kwonlyargs]
    assert 'keep_led_between_steps' in names, (
        'SequencedCaptureRunner.prepare must accept keep_led_between_steps'
    )
    default = _default_for(method, 'keep_led_between_steps')
    assert isinstance(default, ast.Constant) and default.value is False, (
        'keep_led_between_steps must default to False so the LED turns off '
        'between steps unless the optimization is opted in'
    )


def test_run_stores_flag_on_self():
    method = _method_node(_tree(RUNNER_SRC), 'SequencedCaptureRunner', 'start')
    assert 'self._keep_led_between_steps = plan.keep_led_between_steps' in ast.unparse(method)


def test_protocol_runner_forwards_run_params_via_helper():
    # protocol_runner forwards the settings-derived run params (including
    # keep_led_between_steps) into prepare() through the single-source helper,
    # not a hand-passed kwarg, so the GUI and API paths cannot drift over which
    # settings they forward. The helper's False default is pinned in
    # test_sequenced_run_settings_single_source.py.
    for node in ast.walk(_tree(PROTO_RUNNER_SRC)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'prepare'
        ):
            for kw in node.keywords:
                if kw.arg is None and 'get_sequenced_run_settings' in ast.unparse(kw.value):
                    return
    raise AssertionError(
        'protocol_runner must spread get_sequenced_run_settings(settings) into '
        'the runner.prepare() call (the single source for keep_led_between_steps)'
    )
