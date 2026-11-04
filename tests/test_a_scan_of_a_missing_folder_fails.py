# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scan of a source folder fails, naming it, when git tracks nothing there.

``tests/ast_seams.py`` ``tracked_files`` is the one lister of the
repository's files. A guard asserting "no file under ``drivers/`` does X"
must not pass over no files when ``drivers/`` has moved. The lister is
driven over a ``git init``-ed tree of its own, so no file is planted in
the checkout the other workers scan.
"""

from __future__ import annotations

import subprocess

import pytest

from tests.ast_seams import iter_package_modules, tracked_files


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    """A git tree: tracked ``a/x.py``, ``a/sub/y.py``, ``a/data.txt``,
    ``b/readme.md``, ``top.py``; untracked ``a/new.py`` and ``c/only.py``."""
    root = tmp_path_factory.mktemp('tree')
    for rel in ('a/x.py', 'a/sub/y.py', 'a/data.txt', 'b/readme.md', 'top.py'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x = 1\n')
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    for rel in ('a/new.py', 'c/only.py'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x = 1\n')
    return root


def test_a_folder_git_tracks_nothing_under_is_refused_by_name(tree):
    with pytest.raises(FileNotFoundError, match=r"'gone'"):
        tracked_files(('a', 'gone'), root=tree)
    # On disk but untracked is not there either.
    with pytest.raises(FileNotFoundError, match=r"'c'"):
        tracked_files(('c',), root=tree)


def test_a_folder_holding_files_none_matching_answers_none(tree):
    assert tracked_files(('b',), root=tree) == []


def test_the_pattern_is_read_as_pathlib_reads_a_glob(tree):
    assert tracked_files(('a',), '*.py', root=tree) == ['a/x.py']
    assert tracked_files(('a',), root=tree) == ['a/sub/y.py', 'a/x.py']
    assert tracked_files(pattern='*.py', root=tree) == ['top.py']


def test_an_untracked_file_is_not_listed(tree):
    assert 'a/new.py' not in tracked_files(root=tree)
    assert tracked_files(root=tree) == ['a/sub/y.py', 'a/x.py', 'top.py']


def test_the_package_walker_refuses_a_package_that_is_not_there():
    with pytest.raises(FileNotFoundError, match=r"'no_such_package'"):
        list(iter_package_modules(('modules', 'no_such_package')))
