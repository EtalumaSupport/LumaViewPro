"""A typed z-stack step or range is recorded as typed, before the handler that rewrites it.

The z-stack step-size and range boxes each bind two handlers to one commit in
``ui/lumaviewpro.kv``: ``log_step_field``, which records what the box holds,
and ``set_steps``, which puts an entry that is not a number back to the stored
value by writing into the box. Kivy runs a widget's handlers in declaration
order, so the record carries the typed text only while ``log_step_field`` is
bound first.

The test reads the kv as TEXT. The suite mocks Kivy -- ``kivy.lang.parser`` is
not importable here -- so the parse tree and the dispatch order are out of
reach at test time, and the order of the two lines in each id's block is the
fact that would regress. The behavioural form waits on the record moving into
``ZStack.set_steps`` itself, ahead of its read of the box.

The layer panel's typed records, once pinned here through the AST, are pinned
by driving the helper in ``tests/test_layer_text_input_logging.py``.
"""

import re

from tests.ast_seams import REPO_ROOT

_KV = 'ui/lumaviewpro.kv'


def _indent_width(line):
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _block_for(control_id):
    """The kv lines belonging to one widget: its id line and deeper siblings."""
    lines = (REPO_ROOT / _KV).read_text().split('\n')
    id_pat = re.compile(r'^[ \t]*id:\s*' + re.escape(control_id) + r'\s*$')
    hits = [i for i, ln in enumerate(lines) if id_pat.match(ln)]
    assert len(hits) == 1, f'{control_id}: expected exactly one id line, found {len(hits)}'
    start = hits[0]
    depth = _indent_width(lines[start])
    block = [lines[start]]
    for ln in lines[start + 1 :]:
        if not ln.strip():
            continue
        if _indent_width(ln) < depth:
            break
        block.append(ln)
    return block


def test_the_zstack_log_binding_runs_before_the_handler_that_coerces():
    """Ordering is load-bearing: set_steps rewrites the box it is reading.

    ``set_steps`` puts an entry that is not a number back to the stored value,
    writing it into the widget. ``log_step_field`` reads the widget to record
    what was typed, so it must be bound FIRST -- kv appends handlers in
    declaration order. Reversed, the "typed" record would carry the put-back
    value and the raw entry would be lost.
    """
    for control_id in ('zstack_stepsize_id', 'zstack_range_id'):
        block = _block_for(control_id)
        log_at = next(i for i, ln in enumerate(block) if 'log_step_field' in ln)
        set_at = next(i for i, ln in enumerate(block) if 'set_steps' in ln)
        assert log_at < set_at, (
            f'{control_id}: set_steps is bound before log_step_field, so the typed '
            f'value is overwritten before it is read'
        )
