# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: shutdown leds_off timeout logging is truthful and distinct.

Bug
---
The app-exit leds_off wait caught every exception with one message,
"leds_off via io_executor timed out / failed" -- firing the identical
warning whether the task genuinely failed or merely sat queued behind
protocol-abort cleanup that had ALREADY turned the LEDs off. A bench
log showed the warning two seconds after serial confirmed LEDS_OFF
completed, reading as if LEDs were left on.

Fix
---
The wait distinguishes TimeoutError from real failure, and the timeout
branch reports the LED state cache so the log answers "were LEDs left
on?" directly. This AST lock fails if the handler collapses back to a
single generic except around the shutdown leds_off wait.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LVP_MAIN_SRC = REPO / 'lumaviewpro.py'


def _result_try_nodes(tree: ast.AST):
    """Yield Try nodes whose body awaits a future .result(timeout=...)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == 'result'
                and any(kw.arg == 'timeout' for kw in sub.keywords)
            ):
                yield node
                break


def test_shutdown_ledsoff_wait_distinguishes_timeout():
    """The shutdown leds_off wait must catch TimeoutError separately and
    consult the LED state cache in that branch."""
    tree = ast.parse(LVP_MAIN_SRC.read_text())
    for try_node in _result_try_nodes(tree):
        handler_names = [ast.unparse(h.type) for h in try_node.handlers if h.type is not None]
        if 'TimeoutError' not in handler_names:
            continue
        timeout_handler = next(
            h
            for h in try_node.handlers
            if h.type is not None and ast.unparse(h.type) == 'TimeoutError'
        )
        consults_cache = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'get_led_states'
            for n in ast.walk(timeout_handler)
        )
        assert consults_cache, (
            'The TimeoutError branch of the shutdown leds_off wait must '
            'report the LED state cache; a bare timeout warning reads as '
            '"LEDs left on" even when cleanup already turned them off'
        )
        return
    raise AssertionError(
        'lumaviewpro.py must catch TimeoutError separately on the shutdown '
        'leds_off future wait; a single generic except conflates "queued '
        'behind cleanup at exit" with a real failure'
    )
