# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refused pick or click is still a thing the user did.

Three handlers judged the action before recording it, so the one case a
forensic reader most wants to see -- the user asked for something and did not
get it -- was the one case that left no line at all. The bundle showed a
warning with nothing saying which control provoked it, or a stage that never
moved with nothing saying anyone had asked it to.

Attribution is the point, and it is why recording at the top beats recording
at each refusal. The notification text does not identify the control: the
file-writes gate says the same words for five different buttons, and the
protocol builder's refusal is shared by nine callers. The record names the
control; the notification that follows says why it was refused.

The ordering pins read the AST because the invariant is an ORDERING -- the
record comes before the branch that returns -- and because the suite mocks
Kivy rather than instantiating widgets. The stage boxes are driven through
their real handlers.
"""

from __future__ import annotations

import ast

import pytest

from tests.ast_seams import find_def

# handler -> the emitter and record that must precede its first refusal.
_RECORD_BEFORE_REFUSAL = (
    (
        'ui/microscope_settings.py',
        'MicroscopeSettings',
        'select_binning_size',
        'select',
        'BINNING',
    ),
    (
        'ui/protocol_settings.py',
        'ProtocolSettings',
        'new_protocol',
        'button',
        'NEW_PROTOCOL',
    ),
)


def _emitter_calls(fn, attr, record=None):
    found = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == attr
            and isinstance(func.value, ast.Name)
            and func.value.id == 'gui_logger'
        ):
            continue
        if record is not None and not (
            node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == record
        ):
            continue
        found.append(node)
    return found


@pytest.mark.parametrize(('module', 'cls', 'handler', 'attr', 'record'), _RECORD_BEFORE_REFUSAL)
def test_the_action_is_recorded_before_the_branch_that_refuses_it(
    module, cls, handler, attr, record
):
    fn = find_def(module, handler, class_name=cls)
    assert fn is not None, f'{cls}.{handler} moved or was renamed'

    emits = _emitter_calls(fn, attr, record)
    assert emits, f'{cls}.{handler} no longer records {record} at all'

    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, (
        f'{cls}.{handler} has no early return; this pin assumes the refusal '
        f'leaves through one, so it has gone stale rather than passed'
    )
    assert min(e.lineno for e in emits) < min(r.lineno for r in returns), (
        f'{cls}.{handler} records {record} only after the branch that refuses '
        f'the action, so a refused one leaves no line naming the control'
    )


@pytest.mark.parametrize('typed', ['', '-', '.', '-.'])
@pytest.mark.parametrize(
    ('handler', 'box', 'record'),
    (
        ('set_xposition', 'x_pos_id', 'SET_X_POSITION'),
        ('set_yposition', 'y_pos_id', 'SET_Y_POSITION'),
    ),
)
def test_an_unparseable_stage_entry_is_recorded_as_a_refusal_and_the_box_goes_back(
    monkeypatch, handler, box, record, typed
):
    """What the kv float filter lets through but is not a number moves
    nothing. The refusal leaves a line saying so -- the name alone would read
    as a successful move -- and then the box shows the target again, and
    that is recorded too."""
    from types import SimpleNamespace

    import modules.app_context as _app_ctx
    import ui.motion_settings as ms
    from modules import gui_logger

    lines = []
    monkeypatch.setattr(gui_logger, 'button', lambda name, detail='': lines.append((name, detail)))
    monkeypatch.setattr(gui_logger, 'text_input', lambda name, value: lines.append((name, value)))
    moves = []
    monkeypatch.setattr(ms, 'move_absolute', lambda *a, **k: moves.append((a, k)))
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(controls_locked=False))
    )
    stand = SimpleNamespace(ids={box: SimpleNamespace(text=typed)})
    stand.update_gui = lambda: setattr(stand.ids[box], 'text', '12.50')

    getattr(ms.XYStageControl, handler)(stand, typed)

    assert moves == []
    assert stand.ids[box].text == '12.50'
    assert lines == [(record, f'refused: {typed!r}'), (f'{record}_APPLIED', '12.50')]
