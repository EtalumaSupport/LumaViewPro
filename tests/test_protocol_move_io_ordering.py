"""Regression: protocol-context moves/turret ops route through the io
executor instead of calling the scope primitive directly.

During a protocol run the scan loop calls go_to_step on protocol_thread,
which is a DIFFERENT thread from the io executor's single worker. If a
move (or turret thome/tmove) is issued by calling the scope primitive
directly on protocol_thread, it races the leds_off/led_on that the
capture path queues on the io worker -- when the move wins the race, the
previous step's LED stays lit through the well-to-well move (the
red-LED-stuck-on report).

The fix routes the protocol-context move/turret ops through
io_executor.protocol_put so they land on the single io worker in FIFO
order behind the step's leds_off and ahead of the next led_on.

These tests fail before the fix (direct call) and pass after (queued).
"""

from __future__ import annotations

import ast
import pathlib
import sys
from unittest.mock import MagicMock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# conftest mocks `kivy`, `kivy.clock`, `kivy.base` but not the submodules
# ui_helpers imports directly. Register them so the behavioral test can
# import ui.ui_helpers (the AST tests below don't need this).
for _kivy_submod in ('kivy.core', 'kivy.core.window', 'kivy.uix', 'kivy.uix.scrollview'):
    sys.modules.setdefault(_kivy_submod, MagicMock())


# ---------------------------------------------------------------------------
# B-1: ui_helpers.move_absolute_position(protocol=True) -- behavioral
# ---------------------------------------------------------------------------


def test_protocol_move_routes_through_io_executor(monkeypatch):
    """A protocol-context X/Y/Z move must be enqueued on
    io_executor.protocol_put, NOT called directly on the calling thread.
    """
    import modules.app_context as app_context
    import ui.ui_helpers as ui_helpers

    ctx = MagicMock()
    # protocol_put returns a future-like object the wrapper waits on.
    fut = MagicMock()
    ctx.io_executor.protocol_put.return_value = fut
    monkeypatch.setattr(app_context, 'ctx', ctx)
    # The trailing UI refresh is scheduled, not run, in production; make
    # it a no-op so the test doesn't reach Kivy's Clock.
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda *a, **k: None)

    ui_helpers.move_absolute_position('X', 1234.0, protocol=True)

    # The move was queued exactly once on the protocol queue.
    ctx.io_executor.protocol_put.assert_called_once()
    task = ctx.io_executor.protocol_put.call_args.args[0]
    assert task.action is ctx.scope.motion._move_absolute_position_impl, (
        'protocol-context move must enqueue the non-dispatching move body as '
        'the IOTask action so it serializes on the io worker'
    )
    # The wrapper waits for completion (preserves the prior synchronous
    # semantics on protocol_thread).
    fut.result.assert_called_once()
    # It must NOT have been called directly on the calling thread -- that is
    # the bypass that races leds_off.
    ctx.scope.motion._move_absolute_position_impl.assert_not_called()


# ---------------------------------------------------------------------------
# B-2 / B-3: vertical_control.turret_select(protocol=True) -- source guard
#
# turret_select is a method on a Kivy widget (debounced, schedules Clock
# callbacks), so instantiating it for a behavioral assert is impractical.
# Guard the structure instead: in the protocol branches thome/tmove must
# appear only as IOTask action references, never as direct calls.
# ---------------------------------------------------------------------------


def _turret_select_node() -> ast.FunctionDef:
    src = (REPO_ROOT / 'ui' / 'vertical_control.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'turret_select':
            return node
    raise AssertionError('turret_select not found in ui/vertical_control.py')


def _direct_calls_to(node: ast.AST, names: set[str]) -> list[str]:
    """Attribute calls like `scope.motion.thome()` -- the direct-call
    bypass. A reference passed to IOTask (`IOTask(scope.motion.thome)`)
    is an ast.Attribute, not an ast.Call, so it is not flagged.
    """
    out: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in names:
            out.append(n.func.attr)
    return out


def test_protocol_turret_ops_not_called_directly():
    """thome/tmove must be routed through io_executor (referenced inside
    IOTask), never invoked directly on protocol_thread.
    """
    direct = _direct_calls_to(_turret_select_node(), {'thome', 'tmove'})
    assert not direct, (
        'turret_select calls these motion primitives directly -- they must '
        'be routed through io_executor.protocol_put(IOTask(...)) so the '
        f'protocol branch serializes on the io worker. Direct calls: {direct}'
    )


def test_protocol_turret_branch_uses_protocol_put():
    """The protocol branches enqueue via protocol_put (paired with the
    direct-call guard above)."""
    node = _turret_select_node()
    put_calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'protocol_put'
    ]
    assert len(put_calls) >= 2, (
        'turret_select must route both thome and tmove through '
        f'io_executor.protocol_put in the protocol branch; found {len(put_calls)}'
    )
