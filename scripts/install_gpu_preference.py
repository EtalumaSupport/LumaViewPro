#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Set the Python that runs this script to Windows' high-performance GPU.

LumaViewPro run from source is that Python's process, so the GPU Windows
gives it is the one Windows gives the interpreter. On a laptop with an
Intel GPU beside an NVIDIA one, Windows gave it the Intel and the live view
drew at 12 fps; set to High performance, 42 fps. This writes the same
setting Windows' Settings > System > Display > Graphics writes, for the
interpreter's own executable (``sys._base_executable``: a virtual
environment's ``python.exe`` hands off to it, so that is the process).

- Only the ``GpuPreference`` key of the setting is written; any other key
  Windows keeps in the same value is kept.
- A different choice already there is replaced, and named.
- Every program run with this Python is set too, and a new Python (a minor
  upgrade is a new folder) needs the script run again; both are said.

The installed LumaViewPro is set by its installer; this is the source
install's. ``install_windows.bat`` runs it with the Python it installed into.

Usage:
    <python> scripts/install_gpu_preference.py
"""

from __future__ import annotations

import sys

KEY = r'Software\Microsoft\DirectX\UserGpuPreferences'
PREFERENCE = 'GpuPreference'
HIGH_PERFORMANCE = '2'
CHOICES = {'0': 'Let Windows decide', '1': 'Power saving', '2': 'High performance'}


def with_high_performance(value: str | None) -> tuple[str, str | None]:
    """``value`` with its ``GpuPreference`` set to High performance, every other key kept.

    Args:
        value: The setting as Windows stores it (``key=value;`` pairs), or
            None when the executable has none.

    Returns:
        The setting to store, and the choice it replaced when that was
        another one; else None.
    """
    pairs = [pair.split('=', 1) for pair in (value or '').split(';') if '=' in pair]
    previous = next((v for k, v in pairs if k == PREFERENCE), None)
    kept = [f'{k}={v}' for k, v in pairs if k != PREFERENCE]
    stored = ';'.join([*kept, f'{PREFERENCE}={HIGH_PERFORMANCE}']) + ';'
    return stored, (previous if previous not in (None, HIGH_PERFORMANCE) else None)


def main() -> int:
    import winreg

    python = sys._base_executable
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, KEY, 0, winreg.KEY_READ | winreg.KEY_WRITE
    ) as key:
        try:
            value, _ = winreg.QueryValueEx(key, python)
        except FileNotFoundError:
            value = None
        stored, replaced = with_high_performance(value)
        winreg.SetValueEx(key, python, 0, winreg.REG_SZ, stored)
    if replaced is not None:
        print(
            f'{python} was set to {CHOICES.get(replaced, replaced)}; it is set to High performance now.'
        )
    print(
        f"{python} is set to Windows' High performance GPU, for LumaViewPro and every other "
        'program run with this Python. A new Python needs this install run again.'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
