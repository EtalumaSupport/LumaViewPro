"""A failed test leaves its report file whichever conftest takes the hook, on any host.

The Firmware repo's conftests take ``pytest_runtest_makereport`` from this
suite's conftest by name, without its ``pytest_configure``; the hook must
bring everything it needs with it, so a red there is an ordinary failure
with its file written, never an internal error that ends the run and hides
which test failed. A bench host can run a copy of the tree with no git
installed; a red there is the same ordinary failure, its file saying the
commit was unread.
"""

import os
import pathlib
import subprocess
import sys

import pytest

_LVP_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_one_red_test(tmp_path, **env_overrides):
    """One failing test run in its own folder on the report hook alone; its file's text."""
    (tmp_path / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'conftest.py').write_text(
        'import tests.conftest  # noqa: F401\n'
        'from tests.conftest import pytest_runtest_makereport  # noqa: F401\n',
        encoding='utf-8',
    )
    (tmp_path / 'test_red.py').write_text('def test_red():\n    assert 1 == 2\n', encoding='utf-8')
    env = {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(_LVP_ROOT / 'src'), str(_LVP_ROOT)]),
        **env_overrides,
    }

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
    text = written[0].read_text(encoding='utf-8')
    assert 'assert 1 == 2' in text
    return text


def test_a_run_that_takes_only_the_report_hook_writes_the_failure_file(tmp_path):
    _run_one_red_test(tmp_path)


def test_a_red_on_a_host_with_no_git_is_an_ordinary_failure_with_its_file(tmp_path):
    # PATH names only an empty folder, so the hook's git call finds no program,
    # as on a bench PC running a downloaded copy of the tree.
    no_programs = tmp_path / 'no_programs'
    no_programs.mkdir()

    text = _run_one_red_test(tmp_path, PATH=str(no_programs))

    assert 'unread (no git on this host)' in text
