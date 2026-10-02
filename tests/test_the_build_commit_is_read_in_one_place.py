# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The commit a build came from is read in one place.

The startup banner's ``Git:`` line and the bench harness's verdict file both
name the build; they read ``lvp_logger.git_revision`` so the two cannot
disagree. A GitHub ZIP carries the commit in ``.git_archival.txt``, which is
read first.

Run in a subprocess: the suite replaces ``lvp_logger`` with a mock.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_archival_file_names_the_commit(tmp_path):
    (tmp_path / '.git_archival.txt').write_text('node: 0123456789abcdef0123456789abcdef01234567\n')
    code = f'import lvp_logger; print(lvp_logger.git_revision({str(tmp_path)!r}))'

    done = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=60, cwd=str(REPO)
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == '0123456789ab'
