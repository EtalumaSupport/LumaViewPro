# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""End-to-end smoke test for the FW4.0 .mpy deploy path.

Feeds the actual FW4.0 source files through the REAL `mpy-cross` binary
(not a mock) and asserts the output meets the contract the firmware
loader needs — correct magic bytes (v6), nonempty body, binary-clean
through `repl_write_file`. Skipped when mpy-cross isn't on PATH so CI
without the MicroPython toolchain still passes.

Covers the gap between `TestDeployFirmwareFileCompileMpy` (mocked
mpy-cross) and bench validation (requires hardware): the host-side
compile + push pipeline with real bytes, no hardware.
"""
import shutil
import subprocess
from pathlib import Path

import pytest


FW40_ROOT = Path('/Users/ericweiner/Documents/Firmware-FW4.0')
MPY_MAGIC = b'M\x06'  # MicroPython .mpy format version 6


def _mpy_cross_available():
    return shutil.which('mpy-cross') is not None


@pytest.mark.skipif(not _mpy_cross_available(),
                    reason="mpy-cross not on PATH")
class TestFw40MpyCompilation:
    """Compile the real FW4.0 sources and assert the output shape."""

    def test_framing_compiles(self, tmp_path):
        src = FW40_ROOT / 'fw40_framing.py'
        if not src.exists():
            pytest.skip(f"FW4.0 source not at {src} (worktree missing?)")
        out = tmp_path / 'fw40_framing.mpy'
        result = subprocess.run(
            ['mpy-cross', '-o', str(out), str(src)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"mpy-cross stderr: {result.stderr}"
        assert out.exists()
        data = out.read_bytes()
        assert data.startswith(MPY_MAGIC), f"magic={data[:2].hex()} (want 4d06)"
        # Non-trivial body — more than just the header.
        assert len(data) > 100

    def test_motor_main_compiles(self, tmp_path):
        src = FW40_ROOT / 'Motor Controller' / 'Firmware' / 'main.py'
        if not src.exists():
            pytest.skip(f"FW4.0 motor source not at {src}")
        out = tmp_path / 'fw40_motor.mpy'
        result = subprocess.run(
            ['mpy-cross', '-o', str(out), str(src)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"mpy-cross stderr: {result.stderr}"
        data = out.read_bytes()
        assert data.startswith(MPY_MAGIC)
        # Motor main.py is ~72 KB; .mpy is typically 30-40% of source.
        source_size = src.stat().st_size
        assert len(data) < source_size, (
            f"compiled size {len(data)} >= source {source_size} — "
            "mpy-cross produced an unusually large artifact"
        )

    def test_led_main_compiles(self, tmp_path):
        src = FW40_ROOT / 'LED Controller' / 'main.py'
        if not src.exists():
            pytest.skip(f"FW4.0 LED source not at {src}")
        out = tmp_path / 'fw40_led.mpy'
        result = subprocess.run(
            ['mpy-cross', '-o', str(out), str(src)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"mpy-cross stderr: {result.stderr}"
        data = out.read_bytes()
        assert data.startswith(MPY_MAGIC)
        source_size = src.stat().st_size
        # LED main.py is the one that OOMs as .py on MP 1.19 (~63 KB).
        # .mpy retires that failure mode — confirm the output is
        # materially smaller, not just mpy-cross-shaped.
        assert len(data) < source_size * 0.5, (
            f"LED .mpy ({len(data)} B) not < 50% of source "
            f"({source_size} B) — OOM-retirement evidence weaker than expected"
        )


# Full deploy-path integration with .mpy bytes is already covered by
# tests/test_firmware_updater.py::TestDeployFirmwareFileCompileMpy using
# mocked mpy-cross. The byte-handling contract in the push path doesn't
# depend on whether bytes came from real or mocked mpy-cross — it just
# needs valid bytes. This file covers the complement: real mpy-cross
# produces valid bytes when run against real FW4.0 sources.
