"""Regression test: do not wrap self-dispatching motion wrappers in IOTask.

The UI helpers ``ui_helpers.move_absolute_position``, ``move_relative_position``,
and ``move_home`` already submit their hardware call to the io_executor via the
``*_async`` API (``move_absolute_async`` -> ``ex.put(IOTask(...))``). Wrapping
them in an outer ``IOTask(action=move_absolute_position, ...)`` causes TWO
trips through the executor for one hardware move -- redundant queue puts,
context switches, and callback dispatches.

The correct pattern is to call the wrapper directly from any non-io_executor
thread (UI thread, session init, REST handler):

    move_absolute_position('X', stage_x)   # wrapper internally dispatches

Not:

    io_executor.put(IOTask(action=move_absolute_position, args=('X', stage_x)))
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SELF_DISPATCHING_WRAPPERS = frozenset(
    {
        'move_absolute_position',
        'move_relative_position',
        'move_home',
    }
)

SCAN_DIRS = ('ui', 'modules')


def _iter_production_python_files() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for sub in SCAN_DIRS:
        root = REPO_ROOT / sub
        if not root.exists():
            continue
        for p in root.rglob('*.py'):
            if '__pycache__' in p.parts:
                continue
            paths.append(p)
    return paths


def _iotask_action_argument(call: ast.Call) -> ast.expr | None:
    """Return the AST node passed as the IOTask action (first positional or
    ``action=`` kwarg), or None if the call isn't an IOTask invocation."""
    func = call.func
    if (isinstance(func, ast.Name) and func.id == 'IOTask') or (
        isinstance(func, ast.Attribute) and func.attr == 'IOTask'
    ):
        pass
    else:
        return None

    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == 'action':
            return kw.value
    return None


def _violations_in_file(path: pathlib.Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding='utf-8')
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        action = _iotask_action_argument(node)
        if action is None:
            continue
        if isinstance(action, ast.Name) and action.id in SELF_DISPATCHING_WRAPPERS:
            out.append((node.lineno, action.id))
    return out


def test_no_iotask_wrap_of_motion_wrappers():
    findings: list[str] = []
    for path in _iter_production_python_files():
        for lineno, name in _violations_in_file(path):
            rel = path.relative_to(REPO_ROOT)
            findings.append(f'{rel}:{lineno} wraps {name} in IOTask')

    assert not findings, (
        'Self-dispatching motion wrappers must not be wrapped in IOTask. '
        'Call them directly; the wrapper submits the hardware IOTask itself. '
        'Offending sites:\n  ' + '\n  '.join(findings)
    )
