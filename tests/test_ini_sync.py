# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests that motor-config INI files in data/firmware/ stay in sync with
the authoritative copies in the Firmware repo.

Skips gracefully when the Firmware repo is not available (e.g. CI).
"""

import pytest
from pathlib import Path

# Authoritative source lives in the Firmware repo, which is expected to be
# a sibling checkout (../../Firmware or ../Firmware relative to LVP root).
LVP_ROOT = Path(__file__).resolve().parent.parent
LVP_FIRMWARE_DIR = LVP_ROOT / "data" / "firmware"

_CANDIDATE_PATHS = [
    LVP_ROOT / ".." / "Firmware",
    LVP_ROOT / ".." / ".." / "Firmware",
]

FIRMWARE_REPO = None
for _p in _CANDIDATE_PATHS:
    _resolved = _p.resolve()
    if (_resolved / "Motor Controller" / "Firmware").is_dir():
        FIRMWARE_REPO = _resolved
        break

FIRMWARE_SRC_DIR = (
    FIRMWARE_REPO / "Motor Controller" / "Firmware" if FIRMWARE_REPO else None
)

INI_FILES = [
    "xymotorconfig.ini",
    "ztmotorconfig.ini",
    "ztmotorconfig2.ini",
    "ztmotorconfig3.ini",
]


@pytest.fixture(autouse=True)
def _require_firmware_repo():
    """Skip all tests in this module if the Firmware repo is not found."""
    if FIRMWARE_REPO is None:
        pytest.skip(
            "Firmware repo not found at any expected location; "
            "skipping INI sync checks"
        )


@pytest.mark.parametrize("filename", INI_FILES)
def test_ini_file_matches_firmware_repo(filename):
    """INI file in data/firmware/ must be byte-identical to Firmware repo copy."""
    src = FIRMWARE_SRC_DIR / filename
    dst = LVP_FIRMWARE_DIR / filename

    assert src.exists(), f"Authoritative file missing: {src}"
    assert dst.exists(), f"LVP copy missing: {dst}"

    src_bytes = src.read_bytes()
    dst_bytes = dst.read_bytes()

    assert src_bytes == dst_bytes, (
        f"{filename} differs between Firmware repo and LVP data/firmware/. "
        f"Run: cp '{src}' '{dst}'"
    )
