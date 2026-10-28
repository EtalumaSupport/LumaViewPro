"""No test reaches a real serial port, and one that tries fails by name.

Two concurrent full-suite runs held both boards' ports on the bench and
interrupted the motor firmware; the same tests pass quietly on a machine
with no scope. tests/conftest.py refuses enumeration and open unless
--run-hardware is set, and fails the test from the recorded touch, because
the registry's auto mode swallows the refusal into a null board.

Each child run starts in a scratch folder of links into the checkout: a run
with --run-hardware is a bench run and writes its logs and results file to
the log folder of the folder it starts in, which must not be the
checkout's.
"""

import subprocess
import sys
from unittest.mock import Mock

import serial

from tests import conftest
from tests.ast_seams import REPO_ROOT
from tests.bench_results import linked_checkout

VICTIM = 'tests/guards/serial_refusal_victim.py'


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


def test_a_touch_the_diagnostic_path_swallowed_still_fails_the_test_by_name(tmp_path):
    done = _run(tmp_path, 'test_the_diagnostic_path_swallows_the_refusal_into_null_boards')
    assert done.returncode != 0, done.stdout + done.stderr
    assert conftest.SERIAL_REFUSAL_BANNER in done.stdout
    assert (
        'serial_refusal_victim.py::test_the_diagnostic_path_swallows_the_refusal_into_null_boards'
        in done.stdout
    )
    assert 'first: enumerate' in done.stdout


def test_a_bare_serialboard_given_a_port_passes(tmp_path):
    done = _run(tmp_path, 'test_a_bare_serialboard_given_a_port_touches_nothing')
    assert done.returncode == 0, done.stdout + done.stderr


def test_run_hardware_leaves_pyserial_real_and_the_default_does_not(tmp_path):
    with_flag = _run(tmp_path / 'with', 'test_pyserial_is_the_real_one', '--run-hardware')
    assert with_flag.returncode == 0, with_flag.stdout + with_flag.stderr
    without = _run(tmp_path / 'without', 'test_pyserial_is_the_real_one')
    assert without.returncode != 0, without.stdout + without.stderr


REFUSED_OPEN = """
import gc
import tests.conftest
import serial
try:
    serial.Serial('/dev/cu.usbmodem-not-a-port')
except serial.SerialException:
    pass
gc.collect()
"""


def test_a_refused_open_leaves_no_instance_for_pyserial_to_close_later():
    # A refused open that left a half-built instance had pyserial's __del__
    # call close() on it, raising inside whichever later test collected it.
    # A child under -X dev, since below 3.13 the finalizer drops that error
    # unless dev mode is on.
    done = subprocess.run(
        [sys.executable, '-X', 'dev', '-c', REFUSED_OPEN],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert 'Exception ignored' not in done.stderr, done.stderr


def test_the_refuser_is_a_subclass_so_specced_mocks_keep_their_methods():
    assert serial.Serial.__qualname__ != 'Serial'
    assert issubclass(serial.Serial, serial.Serial.__mro__[1])
    assert Mock(spec=serial.Serial).readline is not None
