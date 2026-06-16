"""Regression: the failed-capture log must not test a pandas row for truthiness.

In the captured_image-is-None branch, _write_capture built its log message
with `step.get("Name") if step else "?"`. When step is a pandas Series (the
normal case), `if step` raises `ValueError: The truth value of a Series is
ambiguous`, so the intended "image is None" diagnostic was replaced by a
stack trace on every failed capture during a hardware disconnect. The guard
must be `step is not None`.

This is pinned at the source level: the branch lives deep in _write_capture,
which needs a full writer + executors to exec, so an AST check is the cheapest
faithful guard (same approach as the step-runner / post-processor guards).
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / 'modules' / 'protocol_image_writer.py'


def test_write_capture_guards_step_with_is_not_none():
    src = SRC.read_text()

    # A pandas Series in a boolean context raises; the failed-capture log must
    # use an identity check, not bare truthiness.
    assert 'if step else' not in src, (
        'protocol_image_writer must not test a pandas row with bare `if step` '
        '(raises ValueError on a Series); use `if step is not None`'
    )
    assert 'if step is not None else' in src, (
        'the failed-capture log message must look up the step name via '
        '`step.get(...) if step is not None else "?"`'
    )

    # Belt-and-suspenders: no bare `if step:` / `if step else` truthiness on the
    # row variable anywhere in the module.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        test = node.test if isinstance(node, (ast.IfExp, ast.If)) else None
        if isinstance(test, ast.Name) and test.id == 'step':
            raise AssertionError(
                f'bare `if step` truthiness on a pandas row at line {node.lineno} '
                '-- use `step is not None`'
            )
