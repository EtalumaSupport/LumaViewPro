# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capability probe asks the connected scope only through `hardware_session()`.

The probes ask their questions of the simulator; with `--hardware` they ask the
scope on the desk. A probe with no hardware path must not answer a hardware
question with a simulator's verdict, so its simulated constructor refuses under
the flag. And the door itself configures the session from the installation's
own `current.json`: without one the session would come up on the shipped
template and describe a scope that is not the one connected, so it refuses
before any session is built.

Both run the probe in a subprocess, as the suite runs the probes, and neither
can reach hardware: each is refused before a session exists.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

CAPABILITY = pathlib.Path(__file__).resolve().parent / 'capability'
REPO = CAPABILITY.parents[1]


def _run(args, scratch, code=None):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(CAPABILITY), str(REPO), env.get('PYTHONPATH', '')])
    env['LVP_CAPABILITY_SCRATCH'] = str(scratch)
    command = [sys.executable, *(['-c', code] if code else []), *args]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=str(REPO), env=env
    )


def test_a_simulated_probe_refuses_the_hardware_flag_by_name(tmp_path):
    done = _run([str(CAPABILITY / 'motion_p00_smoke.py'), '--hardware'], tmp_path)

    assert done.returncode == 2, done.stdout + done.stderr
    assert 'REFUSED: motion_p00_smoke has no hardware target' in done.stdout


def test_the_hardware_door_refuses_an_installation_with_no_current_json(tmp_path):
    root = tmp_path / 'install'
    (root / 'data').mkdir(parents=True)
    # The door reads the root the app reads; here it is pointed at an empty
    # installation, so the refusal is reached before any session is built.
    code = (
        'import sys\n'
        "sys.argv = ['door_probe', '--hardware']\n"
        'import pathlib, harness\n'
        'import modules.path_utils as path_utils\n'
        f'path_utils.get_source_root = lambda source_path=None: pathlib.Path({str(root)!r})\n'
        'import modules.scope_session as scope_session\n'
        "scope_session.ScopeSession.create = lambda *a, **k: sys.exit('CREATE REACHED')\n"
        'with harness.hardware_session():\n'
        '    pass\n'
    )
    done = _run([], tmp_path, code=code)

    assert done.returncode == 2, done.stdout + done.stderr
    assert f'REFUSED: no {root / "data" / "current.json"}' in done.stdout
    assert 'CREATE REACHED' not in done.stdout + done.stderr
