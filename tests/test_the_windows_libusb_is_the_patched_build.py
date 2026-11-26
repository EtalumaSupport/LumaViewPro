# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Windows runs the Classic on libusb built with libusb PR #1967, and on no other copy.

What the C change does is the bench's: no runner has a Classic, and the
simulated scope does not reach libusb. These read the committed files, so
a stock DLL committed in the folder, a patch rewritten without the
change, or a DLL other than the one the README names fails on any host.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from _lumascope.drivers import libusb_windows

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / 'src' / '_lumascope' / 'drivers' / 'libusb_windows'
TAG = '-etaluma.1'


def _added_lines(patch: Path) -> str:
    return '\n'.join(
        line[1:]
        for line in patch.read_text(encoding='utf-8').splitlines()
        if line.startswith('+') and not line.startswith('+++')
    )


def test_the_dll_names_itself_the_etaluma_build():
    # libusb compiles LIBUSB_RC into the library its version call returns.
    assert TAG.encode() in libusb_windows.library_path().read_bytes()


def test_the_dll_is_the_one_the_readme_names():
    readme = (FOLDER / 'README.md').read_text(encoding='utf-8')
    name = libusb_windows.LIBRARY_NAME
    (stated,) = re.findall(rf'\| `{re.escape(name)}` \| `([0-9a-f]{{64}})` \|', readme)
    assert hashlib.sha256(libusb_windows.library_path().read_bytes()).hexdigest() == stated


def test_the_first_patch_is_pr_1967s_change_to_the_iso_callback():
    (patch,) = FOLDER.glob('0001-*.patch')
    added = _added_lines(patch)
    # Terminal statuses reach the caller's callback unretried ...
    for status in ('CANCELLED', 'TIMED_OUT', 'NO_DEVICE'):
        assert f'transfer->status == LIBUSB_TRANSFER_{status}' in added, status
    # ... and a retry the backend cannot submit is reported to it.
    assert 'r = libusb_submit_transfer(transfer);' in added
    assert 'if (r != LIBUSB_SUCCESS) {' in added


def test_the_second_patch_tags_the_build():
    (patch,) = FOLDER.glob('0002-*.patch')
    assert f'#define LIBUSB_RC "{TAG}"' in _added_lines(patch)
