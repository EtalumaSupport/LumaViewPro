# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scripts/install_mac.sh``'s FX2 step runs the driver's own gate.

The step's two functions are cut from the script and run in bash, with a
stub ``brew`` first on PATH that records any call: the library comes from
pip, so no case may reach Homebrew. The real-driver case pins that the
script calls names the driver actually has; the stub interpreter cases pin
that a not-ready gate is reported, not repaired, and an import failure aborts.
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
    assert _brew_calls(tmp_path) == []


def test_a_not_ready_gate_is_reported_and_homebrew_is_never_called(tmp_path):
    python = tmp_path / 'fake_python'
    _stub(python, 'echo "FX2 (LS560/LS620/LS720) support: NOT ready -- libusb-package missing"\n')

    result = _run_step(tmp_path, python)

    assert result.returncode == 0, result.stderr
    assert _brew_calls(tmp_path) == []
    assert result.stdout.strip().splitlines()[-1].endswith('libusb-package missing')


def test_a_driver_that_fails_to_import_aborts_without_installing(tmp_path):
    python = tmp_path / 'fake_python'
    _stub(python, 'echo "ImportError: boom" >&2; exit 1\n')

    result = _run_step(tmp_path, python)

    assert result.returncode != 0
    assert 'ImportError: boom' in result.stderr
    assert _brew_calls(tmp_path) == []
    assert 'FX2 (LS560/LS620/LS720) support' not in result.stdout
