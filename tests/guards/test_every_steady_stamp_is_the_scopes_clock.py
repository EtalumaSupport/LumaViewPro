# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every host steady stamp is read from the scope's one clock, ``drivers/host_clock.py``.

A frame's arrival, a change's write and in-force instant and the clock
latch's bracket are set against one another, so they must be one clock at
one resolution. Chosen at each site, the choice was ``time.monotonic_ns()``,
which on Windows under Python 3.12 is ``GetTickCount64()``, a 15.6 ms step:
the latch bracket read 0 wide on the bench and a frame and a change within
a step read as one instant. ``host_clock.steady_ns()`` is the one reader,
and the scope publishes it as ``Lumascope.host_steady_ns()``.

Outside ``drivers/host_clock.py`` and this guard, every ``.py`` in the
checkout -- drivers, modules, ui, rest, tools, tests and the root -- is
refused any reference to a nanosecond clock of the time module, called or
not, any import of one (aliased or not), and any string that names one (a
``getattr``'s attribute, a patch target); prose that mentions one is not a
read. A seconds clock scaled to nanoseconds, ``int(time.monotonic() *
1e9)``, is the rule's stated limit: ``time.monotonic()`` itself is the
timeouts' clock, where 15.6 ms is below a Windows wait's own grain.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import REPO_ROOT, Words, tracked_python_files

_THE_CLOCK = 'drivers/host_clock.py'
# This guard names the clocks it refuses.
_THIS_GUARD = 'tests/guards/test_every_steady_stamp_is_the_scopes_clock.py'
_NANOSECOND_CLOCKS = Words({'monotonic_ns', 'perf_counter_ns', 'time_ns'})
# A string that is a name -- a getattr's attribute, a patch target -- not prose.
_A_NAME = re.compile(r'[\w.]+')


def _offences(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in _NANOSECOND_CLOCKS) or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name in _NANOSECOND_CLOCKS for alias in node.names)
        ):
            found.append((node.lineno, ast.unparse(node)))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _A_NAME.fullmatch(node.value)
            and node.value.rsplit('.', 1)[-1] in _NANOSECOND_CLOCKS
        ):
            found.append((node.lineno, f'the string {node.value!r}'))
    return found


def _source_files() -> list[str]:
    return tracked_python_files()


def test_no_steady_stamp_is_read_beside_the_scopes_clock():
    files = _source_files()
    assert _THE_CLOCK in files and any(f.startswith('tests/') for f in files), (
        'the walk misses the clock or the tests: it reads too little to judge'
    )
    offenders = [
        f'{rel}:{line} {what}'
        for rel in files
        if rel not in (_THE_CLOCK, _THIS_GUARD)
        for line, what in _offences(ast.parse((REPO_ROOT / rel).read_text(encoding='utf-8')))
    ]
    assert not offenders, (
        "a nanosecond clock read beside the scope's; read host_clock.steady_ns() "
        '(from drivers import host_clock), or session.scope.host_steady_ns() above the '
        'scope:\n' + '\n'.join(offenders)
    )


def test_every_spelling_of_a_nanosecond_clock_is_named():
    planted = ast.parse(
        'import time\n'
        'import time as t\n'
        'from time import perf_counter_ns as pc\n'
        'a = time.monotonic_ns()\n'
        'b = t.time_ns\n'
        "c = getattr(time, 'perf_counter_ns')\n"
        "d = mock.patch('modules.motion.time.monotonic_ns')\n"
        'e = exposure_time_ns\n'
        'f = time.monotonic()\n'
        "g = 'stamped on time.monotonic_ns at arrival'\n"
    )
    assert sorted(line for line, _ in _offences(planted)) == [3, 4, 5, 6, 7]
    assert _offences(ast.parse((REPO_ROOT / _THE_CLOCK).read_text(encoding='utf-8')))
