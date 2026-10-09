# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The commit a build came from is read in one place, where the code lives.

The startup banner's ``Git:`` line, the bench harness's verdict file, a
profile trace and every plugin record name the build through
``lvp_logger.git_revision``, so they cannot disagree. A GitHub ZIP carries
the commit in ``.git_archival.txt``, which is read first. The simulator is
launched from a folder of links into a clone; the commit is the clone's,
where the folder has no ``.git``.

Run in a subprocess: the suite replaces ``lvp_logger`` with a mock.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _revision(code: str, cwd: pathlib.Path) -> str:
    done = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=60, cwd=str(cwd)
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


def test_the_archival_file_names_the_commit(tmp_path):
    (tmp_path / '.git_archival.txt').write_text('node: 0123456789abcdef0123456789abcdef01234567\n')
    code = (
        'import pathlib, lvp_logger; '
        f'lvp_logger.get_script_root = lambda: pathlib.Path({str(tmp_path)!r}); '
        'print(lvp_logger.git_revision())'
    )

    assert _revision(code, REPO) == '0123456789ab'


def test_a_launch_from_a_folder_of_links_names_the_clones_commit(tmp_path):
    # The simulator recipe's shape: the launch folder holds links into the
    # clone and no .git of its own.
    for entry in ('lvp_logger.py', 'version.txt', 'modules', 'lib', 'drivers', 'ui', 'data'):
        os.symlink(REPO / entry, tmp_path / entry)
    head = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert _revision('import lvp_logger; print(lvp_logger.git_revision())', tmp_path) == head


def test_a_frozen_build_asks_git_nothing():
    # A clone's archival file holds the unsubstituted placeholder, so only git
    # could answer here; a frozen build must not ask it.
    code = 'import sys, lvp_logger; sys.frozen = True; print(lvp_logger.git_revision())'

    assert _revision(code, REPO) == 'None'


def test_a_profile_trace_names_the_one_readers_commit_and_whether_the_tree_was_clean():
    code = (
        'import json, pathlib, lvp_logger; from lib import profile_trace; '
        f'identity = profile_trace._build_identity(pathlib.Path({str(REPO)!r})); '
        'print(json.dumps([identity["git_sha"], lvp_logger.git_revision(), identity["git_dirty"]]))'
    )

    sha, revision, dirty = json.loads(_revision(code, REPO))

    assert sha == revision and sha
    assert isinstance(dirty, bool)


def test_the_banner_reads_the_commit_guid_from_line_3_and_names_no_branch(tmp_path):
    (tmp_path / 'version.txt').write_text('4.0.0-x\n2026-10-09 00:00\nabcd1234\n')
    code = (
        'import json, logging, lvp_logger; lines = []; '
        'handler = logging.Handler(); handler.emit = lambda r: lines.append(r.getMessage()); '
        'lvp_logger.logger.addHandler(handler); '
        f'lvp_logger.log_environment_banner({str(tmp_path)!r}, "4.0.0-x", []); '
        'print(json.dumps(lines))'
    )

    lines = json.loads(_revision(code, REPO))

    assert '[LVP Main  ] CommitGUID: abcd1234' in lines
    assert '[LVP Main  ] Built:     2026-10-09 00:00' in lines
    assert not [line for line in lines if 'Committed on' in line]
