"""A failed test's report lists the full garbage collections that ran during it.

A test that waits on the simulator under a wall-time bound can go red when
a generation-2 collection pauses the process inside the wait. The report
lists each collection's start and length on the scope's steady clock, the
clock a capture's give-up line states its wait on, so the two can be set
side by side; a passed test's report is unchanged. tests/conftest.py's
report hook adds the section, and the engineering plugin's tests import
that hook by name.

Each child run starts in a scratch folder of links into the checkout, as
tests/guards/test_serial_ports_are_refused.py's do.
"""

import re
import subprocess
import sys

from tests import conftest
from tests.bench_results import linked_checkout

VICTIM = 'tests/guards/gc_report_victim.py'

_COLLECTION = re.compile(r'^at steady \d+\.\d{3}s, \d+\.\d ms$', re.MULTILINE)


def _run(tmp_path, test, *extra):
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-o',
            'addopts=',
            '-p',
            'no:cacheprovider',
            f'{VICTIM}::{test}',
            *extra,
        ],
        cwd=linked_checkout(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_red_lists_the_full_collection_it_ran(tmp_path):
    done = _run(tmp_path, 'test_a_red_after_one_full_collection')

    assert done.returncode == 1, done.stdout + done.stderr
    _, _, section = done.stdout.partition(conftest.GC_SECTION)
    assert section, f'no {conftest.GC_SECTION!r} section in the red report:\n{done.stdout}'
    section = section.split('=====')[0]
    # At least the one the test forced; the interpreter may run its own.
    assert _COLLECTION.findall(section), section


def test_a_green_report_carries_no_section(tmp_path):
    done = _run(tmp_path, 'test_a_green_after_one_full_collection', '-rP')

    assert done.returncode == 0, done.stdout + done.stderr
    assert conftest.GC_SECTION not in done.stdout, done.stdout
