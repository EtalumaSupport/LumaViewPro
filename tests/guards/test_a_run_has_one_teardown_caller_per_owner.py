# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's teardown has exactly three callers, one per owner of the run.

The run loop's finally unwinds a dispatched run; start() unwinds a run
whose loop was never dispatched (its setup failed, or a Stop arrived
during it); force_reset unwinds, at shutdown, a run whose loop ended
without unwinding it. No two can reach the same run, which is why
_cleanup carries no lock and no "is this still my run" check.

A fourth caller would bring both back: the run loop's ending sites each
called _cleanup themselves, so a normal run was torn down three times,
and a Stop that tore the run down on its own thread left both lanes in
run mode under an idle runner. A new way for a run to end returns its
ending to the loop's finally instead of calling _cleanup.
"""

import ast

from tests.ast_seams import production_modules, walk_defs

THE_OWNERS = {
    ('modules/protocol_run_loop.py', 'ProtocolRunLoop.run_loop'),
    ('modules/sequenced_capture_runner.py', 'SequencedCaptureRunner._unwind_undispatched_run'),
    ('modules/sequenced_capture_runner.py', 'SequencedCaptureRunner.force_reset'),
}


def _teardown_callers():
    callers = []
    for rel_path, tree in production_modules():
        for qualname, fn in walk_defs(tree.body):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == '_cleanup'
                ):
                    callers.append((rel_path, qualname))
    return callers


def test_the_teardown_is_called_once_from_each_owner_and_nowhere_else():
    callers = _teardown_callers()
    # walk_defs yields a closure inside its enclosing def as well, so a
    # call is counted once per def that lexically contains it; the owners
    # have no closures around their call.
    assert sorted(callers) == sorted(THE_OWNERS), (
        f'_cleanup callers: {sorted(callers)}; expected exactly one call from '
        f'each of {sorted(THE_OWNERS)}'
    )
