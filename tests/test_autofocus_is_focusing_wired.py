# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: AutofocusRunner mirrors AF lifecycle to scope.imaging.is_focusing.

Bug
---
ImagingAPI exposes is_focusing (property + setter at imaging.py:1593-1608)
that reads/writes self._focusing_event, but NO caller ever set or read
it. AutofocusRunner kept the real "AF in flight" state on its private
_af_in_progress (cleared LAST in the finally block, after camera/LED/Z
restore). External callers asking scope.imaging.is_focusing got False
during a live autofocus run. Rule-35 semantic-duplicate audit
2026-05-19, finding 4.

Note the audit's recommendation said to wire via _is_focusing_event,
but autofocus_runner.py:626-632 documents that _is_focusing_event
clears mid-flight in _iterate (before camera/LED/Z restore), while
_af_in_progress clears at the END of the finally block. The fix
mirrors _af_in_progress, not _is_focusing_event, so the public surface
stays True until the scope is genuinely safe to use again.

Fix
---
autofocus_runner.run() sets self._scope.imaging.is_focusing = True
immediately after self._af_in_progress.set(); the finally block sets
it back to False immediately after self._af_in_progress.clear().

Test approach
-------------
AST source scan -- behavioral exec of AutofocusRunner.run() requires a
real scope + camera_executor + io_executor + objective config and is
out of scope for a regression test. The structural test catches a
re-removal of either assignment.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
AF_RUNNER_SRC = REPO / "modules" / "autofocus_runner.py"


def _run_method() -> ast.FunctionDef:
    tree = ast.parse(AF_RUNNER_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AutofocusRunner":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "run":
                    return child
    raise AssertionError("AutofocusRunner.run not found")


def _is_imaging_is_focusing_assign(node: ast.AST, value: bool) -> bool:
    """Return True if `node` is `self._scope.imaging.is_focusing = <value>`."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Attribute) or target.attr != "is_focusing":
        return False
    if not (isinstance(target.value, ast.Attribute) and target.value.attr == "imaging"):
        return False
    inner = target.value.value
    if not (isinstance(inner, ast.Attribute) and inner.attr == "_scope"):
        return False
    if not (isinstance(inner.value, ast.Name) and inner.value.id == "self"):
        return False
    if not isinstance(node.value, ast.Constant):
        return False
    return node.value.value is value


def test_af_start_mirrors_is_focusing_true():
    """run() sets scope.imaging.is_focusing = True at AF start."""
    run = _run_method()
    found = any(
        _is_imaging_is_focusing_assign(n, True) for n in ast.walk(run)
    )
    assert found, (
        "AutofocusRunner.run() must set self._scope.imaging.is_focusing = True "
        "at AF start so external callers (scope.imaging.is_focusing) see the "
        "right answer during a live run"
    )


def test_af_finally_mirrors_is_focusing_false():
    """run() sets scope.imaging.is_focusing = False at end of finally block."""
    run = _run_method()
    found = any(
        _is_imaging_is_focusing_assign(n, False) for n in ast.walk(run)
    )
    assert found, (
        "AutofocusRunner.run()'s finally block must set "
        "self._scope.imaging.is_focusing = False after camera/LED/Z restore "
        "so callers don't see a stuck-True public surface"
    )


def test_clear_is_paired_with_af_in_progress_clear():
    """The False-assign sits next to _af_in_progress.clear() (lifecycle pairing)."""
    src = AF_RUNNER_SRC.read_text()
    needle = (
        "self._af_in_progress.clear()\n"
        "            # Clear the public ImagingAPI mirror"
    )
    assert needle in src, (
        "The is_focusing=False assignment must immediately follow "
        "self._af_in_progress.clear() so the two flip together when "
        "restoration completes"
    )
