# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The libusb-1.0 the Lumascope Classic runs on under Windows x64.

libusb 1.0.30 built with libusb PR #1967, which libusb-package's wheel
does not carry yet; this folder's README says why, what is in it, and
the release that retires it. Standard library only, so the release spec
can load this file by path without importing LumaViewPro.
"""

from __future__ import annotations

from pathlib import Path

LIBRARY_NAME = 'libusb-1.0.dll'


def library_path() -> Path:
    """This folder's DLL, beside this file in a source run and in the frozen build alike."""
    return Path(__file__).resolve().parent / LIBRARY_NAME
