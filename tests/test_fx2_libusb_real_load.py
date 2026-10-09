# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 driver's libusb load, run against the real bundled library.

The fake-module tests in ``test_audit_fixes.py`` prove the load's decisions;
they cannot see what ctypes does with the handles. pyusb and python-libusb1
each declare argument types on the functions of the library handle they
hold, so one handle shared between them leaves pyusb calling with
python-libusb1's structure types, and every descriptor read raises
ArgumentError -- no FX2 connects, and no hardware is needed to see it. A
fresh interpreter runs the driver's import, then reads the declarations
each binding now has.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The probe prints its answers and the parent asserts them, so a failure
# names every value the child read.
_PROBE = """
import usb.backend.libusb1 as ub
from drivers import fx2driver
backend = ub._lib_object
argtypes = backend.lib.libusb_get_device_descriptor.argtypes
print('refusal', fx2driver._LIBUSB_REFUSAL)
print('own-declarations', argtypes[1]._type_ is ub._libusb_device_descriptor)
print('file', backend.lib._name)
"""


def _answers() -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, '-c', _PROBE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    return dict(line.split(' ', 1) for line in result.stdout.splitlines() if ' ' in line)


def test_pyusb_keeps_its_own_declarations_after_the_driver_loads_libusb():
    answers = _answers()
    assert answers['refusal'] == 'None'
    assert answers['own-declarations'] == 'True', answers
    assert answers['file'].endswith('libusb_package/libusb-1.0' + _suffix()), answers


def _suffix() -> str:
    return {'darwin': '.dylib', 'win32': '.dll'}.get(sys.platform, '.so')
