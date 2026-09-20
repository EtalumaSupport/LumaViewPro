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

These read the AST because the invariant is an ORDERING -- the record comes
before the branch that returns -- and because the suite mocks Kivy rather
than instantiating widgets.
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


@pytest.mark.parametrize(
    ('handler', 'record'),
    (('set_xposition', 'SET_X_POSITION'), ('set_yposition', 'SET_Y_POSITION')),
)
def test_an_unparseable_stage_entry_is_recorded_as_a_refusal(handler, record):
    """The except branch is the refusal; it has to leave a line behind."""
    fn = find_def('ui/motion_settings.py', handler, class_name='XYStageControl')
    assert fn is not None, f'XYStageControl.{handler} moved or was renamed'

    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    assert handlers, f'{handler} no longer has a parse-failure branch'

    recorded = [
        call
        for h in handlers
        for call in _emitter_calls(ast.Module(body=h.body, type_ignores=[]), 'button', record)
    ]
    assert recorded, (
        f'{handler} returns silently on an entry it cannot parse, so a stage '
        f'that did not move looks like a stage nobody asked to move'
    )


def test_the_refusal_record_says_it_was_refused():
    """The name alone would read as a successful move to the same box."""
    for handler in ('set_xposition', 'set_yposition'):
        fn = find_def('ui/motion_settings.py', handler, class_name='XYStageControl')
        details = [
            ast.unparse(call.args[1])
            for h in (n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler))
            for call in _emitter_calls(ast.Module(body=h.body, type_ignores=[]), 'button')
            if len(call.args) > 1
        ]
        assert details, f'{handler} records the refusal with no detail at all'
        assert any('refused' in d for d in details), (
            f'{handler} records a refused entry in the same shape as a '
            f'successful move, so the two are indistinguishable: {details}'
        )
