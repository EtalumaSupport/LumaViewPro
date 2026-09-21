# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Run every capability probe: can a SCRIPT do what the GUI does?

Each probe drives `ScopeSession` / `modules.lumascope_api` only and reports
what it observed. The probes were written for
`CENSUS_API_CAPABILITY_SORT_2026-09-20.md`, which sorted 106 capabilities by
RUNNING them; this module is that census's instrument, kept runnable so the
answer stays a measurement rather than a snapshot.

Two kinds of probe live here, and the difference matters when reading a green
run:

  * **19 probes assert** (the motion and settings slices, plus the two
    post-processing probes that declare a void). They call `harness.check` /
    `harness.void` and fail this module when a check breaks or a void gets
    filled.
  * **27 probes are smoke** (the rest of the layer, post-processing and
    protocol slices). They print a narrative and record nothing, so they
    prove only that the capability still runs without crashing -- which is
    real, because `lvp_logger`'s excepthook turns an uncaught raise into a
    silent exit 1. Converting their prose verdicts into `check` / `void`
    calls is follow-up work; until then a green run here is NOT 46
    capabilities verified.

Why subprocesses: `tests/conftest.py` installs Kivy mocks at import time, and
a probe's load-bearing assertion is that Kivy is ABSENT from `sys.modules`.
In-process collection would destroy the contract being probed. It also keeps
these clear of the module-level mock leakage that makes multi-file targeted
runs report failures neither file has alone.

THE RATCHET. `harness.void(...)` marks a capability a script cannot perform.
A void is enforced by the probe that declares it: if the capability starts
working, `harness.report()` fails that probe and names the void to retire, so
the enforcement is per-probe and needs no global pin. Two counts were tried
and rejected: grepping the source for the verb (one line inside a loop
declares eight voids, a try/except pair declares one), and registering a
ratchet here (these probes are `slow`, so a run that skipped them would
announce zero voids, which reads as a finished migration).
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The harness and its helpers are not probes.
_NOT_A_PROBE = {'harness.py', 'runfolder.py', '__init__.py', pathlib.Path(__file__).name}

# Void names reported by the probes THIS RUN. Populated as probes execute, so
# the announced number describes what actually ran and nothing more.
_OBSERVED_VOIDS: set[str] = set()

_VOID_LINE = re.compile(r'^VOID (?!FILLED)(.+?)\s{3}\[', re.M)


def probe_files():
    return sorted(p for p in HERE.glob('*.py') if p.name not in _NOT_A_PROBE)


def run_probe(path, scratch):
    """Run one probe in a clean interpreter. Returns (exit code, output)."""
    env = dict(os.environ)
    # The probes import `harness` by name, the way they did as standalone
    # scripts; this directory is what makes that resolve.
    env['PYTHONPATH'] = os.pathsep.join([str(HERE), str(REPO), env.get('PYTHONPATH', '')])
    env['LVP_CAPABILITY_SCRATCH'] = str(scratch)
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
        env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.slow
@pytest.mark.parametrize('probe', probe_files(), ids=lambda p: p.stem)
def test_probe(probe, tmp_path):
    """Green when every `check` passed AND every `void` is still a void.

    A filled void fails here on purpose: it is the ratchet asking to be
    lowered in the same commit that filled it.
    """
    rc, out = run_probe(probe, tmp_path)
    for name in _VOID_LINE.findall(out):
        _OBSERVED_VOIDS.add(f'{probe.stem}: {name.strip()}')
    assert rc == 0, f'{probe.name} reported a failure:\n{out}'


def test_the_void_count_is_reported():
    """Print the voids this run observed, for the runs that ran the probes.

    Deliberately NOT a registered ratchet. A ratchet is announced by every
    run, and these probes are `slow`: on a run that skipped them the measure
    would announce zero voids, which reads as a finished migration. The
    enforcement does not need a global count anyway -- a filled void fails
    the probe that declares it.
    """
    if not _OBSERVED_VOIDS:
        pytest.skip('probes did not run in this selection')
    print(f'\ncapabilities a script cannot perform, observed: {len(_OBSERVED_VOIDS)}')
    for name in sorted(_OBSERVED_VOIDS):
        print(f'  VOID {name}')
