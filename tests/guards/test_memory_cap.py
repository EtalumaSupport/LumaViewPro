"""The memory cap fires, names the test, and the run cannot read as success.

Three xdist workers grew to about 114 GB on a 48 GB machine while their
state stayed R and their output stayed quiet; a person killed them from
Activity Monitor and the run reported a tainted result. The cap in
tests/conftest.py is a per-process watchdog; these tests run a victim
under a small cap and read the exit status, the banner and the report
file, and run the same victim under the default cap as the control.
"""

import subprocess
import sys

import pytest

from tests import conftest
from tests.ast_seams import REPO_ROOT

# The tests share build/memcap_*.txt and each clears it, so two running at
# once on different xdist workers delete each other's report: one group
# keeps them on one worker (`--dist loadgroup`, in the pytest addopts).
pytestmark = pytest.mark.xdist_group('memcap_reports')

VICTIM = 'tests/guards/memcap_victim.py'
SMALL_CAP = 256 * 1024 * 1024


def _run_victim(*extra):
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-o',
            'addopts=',
            '-p',
            'no:cacheprovider',
            VICTIM,
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _reports():
    return sorted((REPO_ROOT / 'build').glob('memcap_*.txt'))


@pytest.fixture
def clean_reports():
    for path in _reports():
        path.unlink()
    yield
    for path in _reports():
        path.unlink()


def test_serial_process_over_the_cap_exits_with_the_banner(clean_reports):
    done = _run_victim('--memory-cap-bytes', str(SMALL_CAP))
    assert done.returncode == conftest.MEMORY_CAP_EXIT_STATUS, done.stdout + done.stderr
    assert conftest.MEMORY_CAP_BANNER in done.stderr
    assert 'memcap_victim.py::test_allocates_and_holds' in done.stderr
    reports = _reports()
    assert len(reports) == 1
    text = reports[0].read_text()
    assert text.startswith(conftest.MEMORY_CAP_BANNER)
    assert 'Thread' in text and 'test_allocates_and_holds' in text


def test_xdist_worker_over_the_cap_fails_the_run_and_the_summary_names_it(clean_reports):
    done = _run_victim('-n', '1', '--memory-cap-bytes', str(SMALL_CAP))
    assert done.returncode != 0, done.stdout + done.stderr
    assert conftest.MEMORY_CAP_BANNER in done.stdout
    assert 'memcap_victim.py::test_allocates_and_holds' in done.stdout
    assert len(_reports()) == 1


def test_the_same_victim_under_the_default_cap_passes(clean_reports):
    done = _run_victim()
    assert done.returncode == 0, done.stdout + done.stderr
    assert conftest.MEMORY_CAP_BANNER not in done.stdout + done.stderr
    assert _reports() == []


def test_the_default_cap_is_five_gib_and_bound_to_the_real_psutil():
    assert conftest.MEMORY_CAP_BYTES == 5 * 1024**3
    assert type(conftest._MEMCAP_PROCESS.memory_info().rss) is int
