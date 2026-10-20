# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A bench run's results land in the one log folder; a simulator suite run leaves none there.

A bench run (a pytest run with a hardware flag) and every capability probe
run, on either target, write their results under
``<log folder>/bench_results/``, which the support zip carries with the
logs; a probe leaves its whole console beside its verdict. The simulator
suite's writes stay in temporary folders, so a suite run puts nothing there.

Each run is a subprocess in a scratch folder of links into the checkout:
the log folder follows the folder a run is launched from, so the runs write
into the scratch folder and never into the checkout's own logs.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.bench_results import linked_checkout


@pytest.fixture
def folder(tmp_path):
    return linked_checkout(tmp_path)


def _pytest(folder, *flags):
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            'tests/test_tmcm6110_hardware.py',
            '-p',
            'no:randomly',
            '-p',
            'no:cacheprovider',
            *flags,
        ],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_a_hardware_flagged_pytest_run_leaves_its_results_file_in_the_log_folder(folder):
    done = _pytest(folder, '--run-hardware')

    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    log_folder = folder / 'logs' / 'LVP_Log'
    (path,) = (log_folder / 'bench_results').glob('pytest_*_results.json')
    results = json.loads(path.read_text())
    assert results['log_folder'] == str(log_folder)
    assert '--run-hardware' in results['command']
    assert {r['outcome'] for r in results['results']} == {'skipped'}
    assert len(results['results']) == 10
    # The real logger ran, so the run's logs are in the same folder.
    assert (log_folder / 'lumaviewpro.log').is_file()


def test_a_simulator_run_leaves_nothing_in_the_log_folder(folder):
    done = _pytest(folder)

    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert not (folder / 'logs').exists()


def test_a_probe_under_hardware_leaves_its_verdict_in_the_log_folder(folder):
    # Launched from the folder's root: a process started inside a linked
    # directory is handed the clone's path as its working directory.
    code = (
        'import sys\n'
        "sys.path.insert(0, 'tests/capability')\n"
        "sys.argv = ['verdict_probe', '--hardware']\n"
        'import harness\n'
        "harness.figure('a figure', 1)\n"
        'harness._write_verdict(False)\n'
    )
    done = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    (path,) = (folder / 'logs' / 'LVP_Log' / 'bench_results').glob('verdict_probe_*_verdict.json')
    verdict = json.loads(path.read_text())
    assert verdict['figures'] == {'a figure': 1}
    assert verdict['target'] == 'hardware'


def test_a_probe_run_leaves_its_console_and_its_verdict_in_the_log_folder(folder):
    # A simulator run, ended by a raise the way a crashing probe ends: the
    # console file holds the traceback and every line before it.
    code = (
        'import sys\n'
        "sys.path.insert(0, 'tests/capability')\n"
        "sys.argv = ['console_probe']\n"
        'import harness\n'
        "harness.check('a check', True)\n"
        'harness.report()\n'
        "raise RuntimeError('the probe crashed')\n"
    )
    done = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 1, done.stdout[-3000:] + done.stderr[-3000:]
    results = folder / 'logs' / 'LVP_Log' / 'bench_results'
    (console,) = results.glob('console_probe_*_console.log')
    (verdict_path,) = results.glob('console_probe_*_verdict.json')
    assert console.name.removesuffix('_console.log') == verdict_path.name.removesuffix(
        '_verdict.json'
    )
    text = console.read_text()
    assert 'PASS a check' in text
    assert 'RuntimeError: the probe crashed' in text
    verdict = json.loads(verdict_path.read_text())
    assert verdict['target'] == 'simulator'
    assert verdict['checks'] == [{'name': 'a check', 'passed': True, 'kind': 'check'}]


def test_a_probe_ended_before_its_report_still_leaves_what_it_measured(folder):
    # A Ctrl-C mid-run: the figure recorded before it is in the verdict.
    code = (
        'import sys\n'
        "sys.path.insert(0, 'tests/capability')\n"
        "sys.argv = ['interrupted_probe']\n"
        'import harness\n'
        "harness.figure('measured before the interrupt', 7)\n"
        'raise KeyboardInterrupt\n'
    )
    done = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode != 0, done.stdout[-3000:] + done.stderr[-3000:]
    results = folder / 'logs' / 'LVP_Log' / 'bench_results'
    (verdict_path,) = results.glob('interrupted_probe_*_verdict.json')
    verdict = json.loads(verdict_path.read_text())
    assert verdict['figures'] == {'measured before the interrupt': 7}
    assert verdict['passed'] is False
    assert verdict['ended'].startswith('before report()')
