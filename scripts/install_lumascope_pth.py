#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Make this checkout's API importable by the Python that runs this script.

Writes ``lumascope.pth``, naming this checkout's ``src/``, into the running
Python's own site-packages (``sysconfig``'s ``purelib``), so a program run by
that Python imports the API from any folder, with no ``sys.path`` line. Only
``src/`` is named: the checkout's ``tests``, ``tools`` and ``ui`` never enter
the user's namespace.

- A site-packages this user cannot write is refused, naming it. The user site
  is never used instead: a virtual environment, ``python -s`` and
  ``python -I`` do not read it. The install scripts' virtual environment is
  the way round it.
- A ``lumascope.pth`` naming another checkout is replaced, and both are
  named: the last install is the user's install.
- The result is checked in a fresh interpreter started outside the checkout:
  the package it imports must be this checkout's.

The install scripts run it with the Python they installed the requirements
into. First-party code (the GUI, the tools, the suite) never needs it: each
puts its own checkout's ``src/`` first.

Usage:
    <python> scripts/install_lumascope_pth.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import sysconfig
import tempfile

PTH = 'lumascope.pth'
SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
PACKAGE = 'lumascope'


class RefusedError(Exception):
    """The ``.pth`` could not be installed, or the interpreter does not import this checkout."""


def install(site: pathlib.Path, src: pathlib.Path) -> str | None:
    """Write ``lumascope.pth`` naming ``src`` into ``site``.

    Returns:
        The folder the file named before, when it named another one; else None.

    Raises:
        RefusedError: ``site`` cannot be written.
    """
    pth = site / PTH
    previous = pth.read_text(encoding='utf-8').strip() if pth.is_file() else None
    try:
        pth.write_text(f'{src}\n', encoding='utf-8')
    except PermissionError as e:
        raise RefusedError(
            f'{site} cannot be written by this user, so {PTH} cannot be installed there. '
            'Install into a virtual environment instead: install_mac.sh and install_linux.sh '
            'always make one, and install_windows.bat --venv does.'
        ) from e
    return previous if previous not in (None, str(src)) else None


def imported_from(python: str, outside: pathlib.Path) -> pathlib.Path:
    """The folder ``python``, started in ``outside``, imports the package from.

    Raises:
        RefusedError: The package does not import.
    """
    done = subprocess.run(
        [python, '-c', f'import {PACKAGE}; print({PACKAGE}.__file__)'],
        cwd=str(outside),
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        last = (done.stderr.strip().splitlines() or [f'exit {done.returncode}'])[-1]
        raise RefusedError(f'{python} does not import {PACKAGE} after the install: {last}')
    return pathlib.Path(done.stdout.strip()).resolve().parent


def main() -> int:
    site = pathlib.Path(sysconfig.get_paths()['purelib'])
    own = (SRC / PACKAGE).resolve()
    try:
        previous = install(site, SRC)
        with tempfile.TemporaryDirectory() as outside:
            found = imported_from(sys.executable, pathlib.Path(outside))
        if found != own:
            raise RefusedError(
                f'{sys.executable} imports {PACKAGE} from {found}, not from this checkout '
                f'({own}): something earlier on its path holds another copy'
            )
    except RefusedError as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 1
    if previous is not None:
        print(f'Replaced {site / PTH}: it named {previous}; it names {SRC} now.')
    print(f'{sys.executable} imports {PACKAGE} from {SRC}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
