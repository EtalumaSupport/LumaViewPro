# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI asks the Session what the run is doing, not the engine beneath it.

The Session answers "is a run in progress" (``run_in_progress``) and "is a
close still waiting on work" (``live_work``) once, for
every client. A GUI read of the engine's own method, or an OR of the two
drains assembled in the close handler, is a second copy of that answer:
it drifts the day the Session's changes, and REST never sees it.
"""

import ast

from tests.ast_seams import parse_module

GUI_MODULES = ('lumaviewpro.py', 'ui/step_navigation.py', 'ui/vertical_control.py')

# What the Session already answers, read below it.
BELOW_THE_SESSION = {'run_in_progress', 'video_drain_busy', 'is_busy'}


def _reads_below_the_session(rel_path: str) -> list[str]:
    found = []
    for node in ast.walk(parse_module(rel_path)):
        if not isinstance(node, ast.Attribute) or node.attr not in BELOW_THE_SESSION:
            continue
        owner = node.value
        # session.run_in_progress is the Session's own answer.
        if isinstance(owner, ast.Attribute) and owner.attr == 'session':
            continue
        found.append(f'{rel_path}:{node.lineno} .{node.attr}')
    return found


def test_the_gui_reads_run_state_from_the_session():
    below = [hit for path in GUI_MODULES for hit in _reads_below_the_session(path)]
    assert not below, 'the GUI reads run state below the Session: ' + ', '.join(below)


def test_the_guard_sees_a_read_below_the_session(monkeypatch):
    """The instrument's own positive: an engine read is reported, the
    Session's is not."""
    import tests.guards.test_the_gui_reads_run_state_from_the_session as guard

    source = (
        'if ctx.sequenced_capture_runner.run_in_progress():\n'
        '    pass\n'
        'if ctx.session.run_in_progress:\n'
        '    pass\n'
    )
    monkeypatch.setattr(guard, 'parse_module', lambda _path: ast.parse(source))
    assert guard._reads_below_the_session('x.py') == ['x.py:1 .run_in_progress']
