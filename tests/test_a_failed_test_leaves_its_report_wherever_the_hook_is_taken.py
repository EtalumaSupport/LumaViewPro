"""A failed test leaves its report file whichever conftest takes the hook.

The Firmware repo's conftests take ``pytest_runtest_makereport`` from this
suite's conftest by name, without its ``pytest_configure``; the hook must
bring everything it needs with it, so a red there is an ordinary failure
with its file written, never an internal error that ends the run and hides
which test failed.
"""

import os
import pathlib
import subprocess
import sys

import pytest

_LVP_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_a_run_that_takes_only_the_report_hook_writes_the_failure_file(tmp_path):
    (tmp_path / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'conftest.py').write_text(
        'import tests.conftest  # noqa: F401\n'
        'from tests.conftest import pytest_runtest_makereport  # noqa: F401\n',
        encoding='utf-8',
    )
    (tmp_path / 'test_red.py').write_text('def test_red():\n    assert 1 == 2\n', encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(_LVP_ROOT)}

    run = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-p', 'no:xdist', 'test_red.py'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert run.returncode == pytest.ExitCode.TESTS_FAILED, run.stdout + run.stderr
    written = list((tmp_path / 'build' / 'test_failures').glob('*/test_red.py_test_red__call.txt'))
    assert len(written) == 1, run.stdout + run.stderr
    assert 'assert 1 == 2' in written[0].read_text(encoding='utf-8')
