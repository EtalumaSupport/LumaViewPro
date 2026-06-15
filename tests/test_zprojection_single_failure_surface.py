# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: one post-processing failure produces ONE user-facing popup.

Bug
---
The z-projection callback surfaced a bad-folder failure twice: it set the
failure message as the progress popup's text (a popup the user still saw
for 5 more seconds) AND fired a notification popup with the same message.
The user got two stacked popups for one wrong folder pick.

Fix
---
On failure the progress popup dismisses immediately and the notification
is the single surface. This AST lock fails if any branch in
ui/post_processing.py regains the double-surface shape (assigning
popup.text in the same branch that fires a notifications warning/error).
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
POST_PROCESSING_SRC = REPO / 'ui' / 'post_processing.py'


def _is_popup_text_assign(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Assign):
        return False
    return any(
        isinstance(t, ast.Attribute)
        and t.attr == 'text'
        and isinstance(t.value, ast.Name)
        and t.value.id == 'popup'
        for t in stmt.targets
    )


def _calls_notification(stmt: ast.stmt) -> bool:
    for node in ast.walk(stmt):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ('warning', 'error', 'critical')
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'notifications'
        ):
            return True
    return False


def test_no_branch_sets_popup_text_and_notifies():
    """No branch may show the same failure on the progress popup AND a
    notification -- the notification is the single failure surface."""
    tree = ast.parse(POST_PROCESSING_SRC.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for body in (node.body, node.orelse):
            has_text = any(_is_popup_text_assign(s) for s in body)
            has_notification = any(_calls_notification(s) for s in body)
            if has_text and has_notification:
                offenders.append(node.lineno)
    assert offenders == [], (
        'These branch(es) surface one failure through BOTH the progress '
        'popup text and a notification popup, stacking two popups for one '
        f'failure -- pick the notification only: lines {offenders}'
    )
