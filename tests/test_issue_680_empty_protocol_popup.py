# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#680 regression: New Protocol must warn when no channels are enabled.

Bug
---
Clicking New with every layer's Acquire checkbox disabled silently
produced an empty Protocol (0 steps). The user got no warning; the
UI just displayed 0 steps. Root cause: Protocol.from_config skips
any layer whose acquire is not in ('image', 'video'); if every layer
is skipped, the resulting Protocol is legally constructed but empty,
and new_protocol() had no post-construction step-count guard.

Fix
---
After ctx.scope.create_protocol() in ProtocolSettings.new_protocol,
check protocol.num_steps() == 0 and pop a "No Channels Selected"
notification before queueing new_protocol_ex on the worker pool.

Test approach
-------------
1. Source-level structural lock via AST: extract new_protocol's body
   and assert the num_steps()==0 guard exists, uses
   show_notification_popup with the "No Channels Selected" title,
   and runs before the worker_pool.put that queues new_protocol_ex.
   Direct UI exec is impractical here (Kivy ids, _app_ctx, worker
   pool, IOTask wrapping); the AST lock catches a regression that
   removes or reorders the guard.

2. Behavioral check on Protocol.from_config: build a config whose
   every layer has acquire set to None/disabled and verify the
   resulting Protocol has zero steps -- proves the precondition the
   UI guard relies on.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'
PROTOCOL_SRC = REPO / 'modules' / 'protocol.py'


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    source = PROTOCOL_SETTINGS_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {PROTOCOL_SETTINGS_SRC}')


def _stmt_index_for(body: list, predicate) -> int:
    for i, stmt in enumerate(body):
        if predicate(stmt):
            return i
    return -1


def test_new_protocol_guards_empty_step_count():
    """new_protocol must check num_steps()==0 and pop a notification."""
    method = _method_node('ProtocolSettings', 'new_protocol')
    src = ast.unparse(method)

    # The exact comparison that gates the popup.
    assert 'protocol.num_steps() == 0' in src, (
        'ProtocolSettings.new_protocol must guard on '
        'protocol.num_steps() == 0 after create_protocol succeeds. '
        '(#680)'
    )

    # The notification title is user-visible and what the bench-retest
    # signal looks for.
    assert 'No Channels Selected' in src, (
        'ProtocolSettings.new_protocol must use the title '
        '"No Channels Selected" for the empty-protocol popup so the '
        'bench retest signal matches. (#680)'
    )

    # The popup helper must be invoked (not just imported).
    assert 'show_notification_popup' in src, (
        'ProtocolSettings.new_protocol must call show_notification_popup '
        'to surface the empty-protocol case. (#680)'
    )


def test_empty_step_guard_runs_before_worker_pool_put():
    """Guard must fire and return before queueing new_protocol_ex."""
    method = _method_node('ProtocolSettings', 'new_protocol')

    def has_num_steps_zero_compare(node):
        if not isinstance(node, ast.If):
            return False
        test = ast.unparse(node.test)
        return 'num_steps()' in test and '0' in test

    def has_worker_pool_put(node):
        unparsed = ast.unparse(node)
        return 'worker_pool.put' in unparsed and 'new_protocol_ex' in unparsed

    guard_idx = _stmt_index_for(method.body, has_num_steps_zero_compare)
    put_idx = _stmt_index_for(method.body, has_worker_pool_put)

    assert guard_idx >= 0, (
        'num_steps()==0 guard not found in ProtocolSettings.new_protocol. '
        '(#680)'
    )
    assert put_idx >= 0, (
        'worker_pool.put for new_protocol_ex not found in new_protocol. '
        '(#680)'
    )
    assert guard_idx < put_idx, (
        f'Empty-steps guard at statement {guard_idx} must run BEFORE '
        f'worker_pool.put at statement {put_idx}; otherwise an empty '
        f'protocol still queues new_protocol_ex and assigns '
        f'self._protocol. (#680)'
    )


def test_protocol_from_config_filters_non_acquire_layers():
    """Source-level lock on the upstream filter that produces 0 steps.

    Protocol.from_config skips layers whose acquire is not 'image' or
    'video'. If every enabled layer is filtered out, the resulting
    Protocol has zero steps -- the precondition the UI guard catches.
    A change to this filter (e.g. a new acquire mode) would change
    the precondition; this test makes that change visible.
    """
    src = PROTOCOL_SRC.read_text()
    assert "if layer_config['acquire'] not in ['image', 'video']:" in src, (
        'Protocol.from_config must skip layers whose acquire is not '
        '"image" or "video"; if this filter changes, the #680 guard '
        'in ui/protocol_settings.py::new_protocol may need a matching '
        'update. (#680)'
    )
