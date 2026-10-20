# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A control bound to the lock record greys and frees as each edge replaces the record.

The App holds the Session's ``ControlsLock`` as ``app.lock`` and a greyed
control binds a field of it (``disabled: app.lock.controls_locked``). The
record is frozen, so each run-state edge replaces it whole; a kv binding
through it re-evaluates only if the App's property rebinds. Without
``rebind=True`` the control keeps the value it had when it was built,
whatever the lock says (the W3d pass measured it).

It runs real Kivy in a child process, because the test environment stands
Kivy in. The App class exists only when lumaviewpro.py runs as the
program, so the child declares the App's ``lock`` property from its own
line in lumaviewpro.py, and the rule from a ``disabled:`` line of
ui/lumaviewpro.kv.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

from tests.ast_seams import parse_module

REPO = pathlib.Path(__file__).resolve().parent.parent

_GREYED_AS_THE_RECORD_CHANGES = r"""
import os, sys
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from modules.scope_session import ControlsLock

declaration, binding = sys.argv[1], sys.argv[2]
namespace = {'ObjectProperty': ObjectProperty}
exec(declaration, namespace)

class ProbeApp(App):
    lock = namespace['lock']

class Probe(Widget):
    pass

app = ProbeApp()
App._running_app = app
app.lock = ControlsLock(run_named=None, recording=False)
Builder.load_string('<Probe>:\n    ' + binding + '\n')
probe = Probe()
seen = [probe.disabled]
for lock in (
    ControlsLock(run_named=None, recording=True),
    ControlsLock(run_named=None, recording=False),
    ControlsLock(run_named='A protocol run is in progress.', recording=False),
):
    app.lock = lock
    seen.append(probe.disabled)
print(' '.join(str(value) for value in seen))
"""


def _the_apps_lock_declaration() -> str:
    tree = parse_module('lumaviewpro.py')
    app = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'LumaViewProApp'
    )
    return next(
        ast.unparse(node)
        for node in app.body
        if isinstance(node, ast.Assign)
        and [getattr(target, 'id', None) for target in node.targets] == ['lock']
    )


@pytest.mark.needs_display
def test_a_control_bound_to_the_lock_follows_each_record(tmp_path):
    binding = 'disabled: app.lock.controls_locked'
    assert binding in (REPO / 'ui' / 'lumaviewpro.kv').read_text()
    # Kivy reads its config from and writes its logs and icons into KIVY_HOME,
    # so the child gets one of its own rather than the shared ~/.kivy.
    done = subprocess.run(
        [
            sys.executable,
            '-c',
            _GREYED_AS_THE_RECORD_CHANGES,
            _the_apps_lock_declaration(),
            binding,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, 'KIVY_HOME': str(tmp_path)},
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    # Free, greyed by a recording, freed, greyed by a run.
    assert done.stdout.split()[-4:] == ['False', 'True', 'False', 'True']
