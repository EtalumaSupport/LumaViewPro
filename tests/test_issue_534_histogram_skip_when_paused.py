# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#534 regression: histogram must skip when preview paused / protocol running.

Bug
---
ui/image_settings.py schedules a 0.5 s Clock.schedule_interval that
calls Histogram.histogram(). The interval is unscheduled only on
accordion collapse or layer switch. The play/pause button pauses the
display thread + sets scope_display.play=False, but the histogram
Clock keeps ticking. Likewise during a protocol acquisition the
histogram fires every 0.5 s. Each tick pulls a frame from the camera
buffer, downsamples + bins it, and rebuilds a 128-bin GPU mesh -- a
contended read against the capture pipeline + wasted CPU/GPU work.

Fix
---
Two early-return guards inside Histogram.histogram():
- Skip if scope_display.play is False (preview paused via cam_toggle)
- Skip if ctx.protocol_running.is_set() (protocol acquisition)

Behavior-preserving for the happy path (preview running, no protocol).

Test approach
-------------
AST-based structural lock on the method's body: assert both guards
exist, both early-return, and both run before any work
(get_image_from_buffer, np.histogram, mesh build).
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
HISTOGRAM_SRC = REPO / 'ui' / 'histogram.py'


def _histogram_method() -> ast.FunctionDef:
    source = HISTOGRAM_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Histogram':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'histogram':
                    return child
    raise AssertionError('Histogram.histogram not found in ui/histogram.py')


def test_histogram_skips_when_preview_paused():
    method = _histogram_method()
    src = ast.unparse(method)
    assert 'scope_display' in src and '.play' in src, (
        'Histogram.histogram must check scope_display.play to skip the '
        'histogram work when live preview is paused. (#534)'
    )


def test_histogram_skips_during_protocol():
    method = _histogram_method()
    src = ast.unparse(method)
    assert 'protocol_running' in src and 'is_set' in src, (
        'Histogram.histogram must check ctx.protocol_running.is_set() '
        'to skip the histogram work during a protocol acquisition. (#534)'
    )


def test_guards_run_before_camera_read():
    """Both guards must early-return before any get_image_from_buffer."""
    method = _histogram_method()

    def first_index_where(predicate) -> int:
        for i, stmt in enumerate(ast.walk(method)):
            if predicate(stmt):
                return i
        return -1

    # Find the first statement that calls get_image_from_buffer.
    body_indices = []
    for i, stmt in enumerate(method.body):
        unparsed = ast.unparse(stmt)
        body_indices.append((i, unparsed))

    play_guard_idx = -1
    protocol_guard_idx = -1
    camera_read_idx = -1

    for i, unparsed in body_indices:
        if play_guard_idx == -1 and 'scope_display' in unparsed and '.play' in unparsed:
            play_guard_idx = i
        if protocol_guard_idx == -1 and 'protocol_running' in unparsed and 'is_set' in unparsed:
            protocol_guard_idx = i
        if camera_read_idx == -1 and 'get_image_from_buffer' in unparsed:
            camera_read_idx = i

    assert play_guard_idx >= 0, 'scope_display.play guard not found (#534)'
    assert protocol_guard_idx >= 0, 'protocol_running guard not found (#534)'
    assert camera_read_idx >= 0, (
        'get_image_from_buffer call not found in Histogram.histogram; '
        'test needs updating for the new shape.'
    )
    assert play_guard_idx < camera_read_idx, (
        f'play guard at statement {play_guard_idx} must run BEFORE '
        f'camera read at statement {camera_read_idx}; otherwise the '
        f'guard does not prevent the contended frame fetch. (#534)'
    )
    assert protocol_guard_idx < camera_read_idx, (
        f'protocol guard at statement {protocol_guard_idx} must run '
        f'BEFORE camera read at statement {camera_read_idx}. (#534)'
    )


def test_guards_are_early_returns():
    """Each guard must produce a bare `return`, not a fallthrough."""
    method = _histogram_method()

    def is_if_with_bare_return(stmt: ast.stmt, must_contain: tuple[str, ...]) -> bool:
        if not isinstance(stmt, ast.If):
            return False
        test_src = ast.unparse(stmt.test)
        if not all(token in test_src for token in must_contain):
            return False
        for inner in stmt.body:
            if isinstance(inner, ast.Return) and inner.value is None:
                return True
        return False

    play_ok = any(
        is_if_with_bare_return(s, ('scope_display', '.play')) for s in method.body
    )
    proto_ok = any(
        is_if_with_bare_return(s, ('protocol_running', 'is_set')) for s in method.body
    )

    assert play_ok, (
        'scope_display.play guard must be `if ...: return` (early-return '
        'shape), not a conditional that falls through. (#534)'
    )
    assert proto_ok, (
        'protocol_running guard must be `if ...: return` (early-return '
        'shape). (#534)'
    )
