# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scripts/install_mac.sh``'s FX2 step runs the driver's own gate.

The step's two functions are cut from the script and run in bash, with a
stub ``brew`` first on PATH so nothing is ever installed. The real-driver
case pins that the script calls names the driver actually has; the stub
interpreter cases pin the three branches the gate's answer selects.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='bash installer')

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / 'scripts' / 'install_mac.sh'
EXTRACT = 'sed -n \'/^fx2_driver_py()/,/^}/p;/^setup_fx2_usb()/,/^}/p\' "$SCRIPT"'


def _stub(path: Path, body: str) -> None:
    path.write_text('#!/bin/bash\n' + body)
    path.chmod(0o755)


def _run_step(tmp_path: Path, python: Path) -> subprocess.CompletedProcess:
    venv_bin = tmp_path / 'venv' / 'bin'
    venv_bin.mkdir(parents=True)
    # A wrapper, not a symlink: a symlinked venv interpreter would look for
    # pyvenv.cfg beside the link and lose the running venv's packages.
    _stub(venv_bin / 'python', f'exec "{python}" "$@"\n')
    stub_bin = tmp_path / 'stub_bin'
    stub_bin.mkdir()
    _stub(stub_bin / 'brew', f'echo "brew $*" >> "{tmp_path}/brew_calls"\n')
    env = {
        **os.environ,
        'PATH': f'{stub_bin}:/usr/bin:/bin',
        'SCRIPT': str(SCRIPT),
        'PROJECT_DIR': str(REPO),
        'VENV_DIR': str(tmp_path / 'venv'),
    }
    return subprocess.run(
        # eval, not `source <(...)`: macOS's bash 3.2 sources nothing from a
        # process substitution, which would leave every case untested.
        [
            'bash',
            '-c',
            f'set -e; eval "$({EXTRACT})"; type setup_fx2_usb >/dev/null; setup_fx2_usb',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _brew_calls(tmp_path: Path) -> list[str]:
    calls = tmp_path / 'brew_calls'
    return calls.read_text().splitlines() if calls.exists() else []


def test_the_step_prints_the_real_drivers_readiness_line(tmp_path):
    expected = (
        subprocess.run(
            [
                sys.executable,
                '-c',
                'from drivers.fx2driver import fx2_readiness_line; print(fx2_readiness_line())',
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[-1]
    )

    result = _run_step(tmp_path, Path(sys.executable))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


def test_a_missing_libusb_is_installed_through_homebrew(tmp_path):
    python = tmp_path / 'fake_python'
    _stub(
        python,
        'case "$2" in *fx2_readiness_line*) echo "FX2 line";; *) echo False;; esac\n',
    )

    result = _run_step(tmp_path, python)

    assert result.returncode == 0, result.stderr
    assert _brew_calls(tmp_path) == ['brew install libusb']
    assert result.stdout.strip().splitlines()[-1] == 'FX2 line'


def test_a_present_libusb_installs_nothing(tmp_path):
    python = tmp_path / 'fake_python'
    _stub(
        python,
        'case "$2" in *fx2_readiness_line*) echo "FX2 line";; *) echo True;; esac\n',
    )

    result = _run_step(tmp_path, python)

    assert result.returncode == 0, result.stderr
    assert _brew_calls(tmp_path) == []


def test_a_driver_that_fails_to_import_aborts_without_installing(tmp_path):
    python = tmp_path / 'fake_python'
    _stub(python, 'echo "ImportError: boom" >&2; exit 1\n')

    result = _run_step(tmp_path, python)

    assert result.returncode != 0
    assert 'ImportError: boom' in result.stderr
    assert _brew_calls(tmp_path) == []
    assert 'FX2 (LS560/LS620/LS720) support' not in result.stdout
