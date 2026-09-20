"""version.txt has one writer, and a detached checkout keeps its branch line.

Two hooks stamp the file. Until 2026-09-20 they were two hand-kept copies:
the pre-commit's asked git for the branch name with a command that prints
the literal ``HEAD`` in a detached checkout, so the triage worktree stamped
``HEAD`` onto the trunk; and the post-merge, unmanaged, wrote three lines
into a four-line format and so dropped the commit id on every merge, then
fired on every peer's pull because line 3 no longer matched a branch. Both
scripts now embed one block, pinned here through the installer's symbols.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tools.install_hooks import _HOOK_SCRIPT, _POST_MERGE_SCRIPT, _STAMP_BLOCK

_WRITE = '> "$VERSION_FILE"'

# The guard gate runs this module inside the pre-commit hook, where git has
# put GIT_INDEX_FILE and GIT_DIR into the environment so the hook judges the
# index being committed. Inherited by a git command in a temporary repo,
# they redirect it at the repository being committed: the first run of this
# test re-inited the shared repository as bare and rewrote its user. The hook
# now drops them before running the guards; this scrub stays because a clone
# runs the hook it has installed, not the one in the tree.
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}


def test_both_hooks_embed_the_one_stamp_block():
    assert _STAMP_BLOCK in _HOOK_SCRIPT
    assert _STAMP_BLOCK in _POST_MERGE_SCRIPT


def test_nothing_writes_the_file_outside_the_block():
    assert _STAMP_BLOCK.count(_WRITE) == 1
    for script in (_HOOK_SCRIPT, _POST_MERGE_SCRIPT):
        assert script.count(_WRITE) == 1, 'a second writer of version.txt has appeared in a hook'


def test_the_block_does_not_ask_git_for_a_branch_it_cannot_name():
    # `rev-parse --abbrev-ref HEAD` succeeds in a detached checkout and prints
    # the literal HEAD; the fallback after it never fires.
    assert 'rev-parse --abbrev-ref' not in _STAMP_BLOCK
    assert 'rev-parse --abbrev-ref' not in _POST_MERGE_SCRIPT


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ['git', '-C', str(repo), *args], text=True, env=_CLEAN_ENV
    ).strip()


def _repo_with_version_file(tmp_path: Path, branch_line: str) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'dev/x')
    _git(repo, 'config', 'user.email', 't@t')
    _git(repo, 'config', 'user.name', 't')
    (repo / 'version.txt').write_text(f'1.0\n2000-01-01 00:00\n{branch_line}\n00000000\n')
    _git(repo, 'add', 'version.txt')
    _git(repo, 'commit', '-q', '-m', 'seed')
    return repo


def _run_stamp(repo: Path) -> list[str]:
    script = f'set -e\ncd "{repo}"\nVERSION_FILE="{repo}/version.txt"\n{_STAMP_BLOCK}'
    subprocess.run(['bash', '-c', script], check=True, env=_CLEAN_ENV)
    return (repo / 'version.txt').read_text().splitlines()


def test_a_detached_checkout_keeps_line_3_and_all_four_lines(tmp_path):
    repo = _repo_with_version_file(tmp_path, 'dev/x')
    _git(repo, 'checkout', '-q', '--detach')
    lines = _run_stamp(repo)
    assert len(lines) == 4
    assert lines[2] == 'dev/x'
    assert lines[3] != '00000000'


def test_a_named_branch_stamps_its_own_name(tmp_path):
    repo = _repo_with_version_file(tmp_path, 'dev/x')
    _git(repo, 'checkout', '-q', '-b', 'feature/y')
    lines = _run_stamp(repo)
    assert len(lines) == 4
    assert lines[2] == 'feature/y'
