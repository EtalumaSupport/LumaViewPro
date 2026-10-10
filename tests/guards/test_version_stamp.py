"""version.txt has one writer, three lines, and names no branch.

The pre-commit hook stamps the file: the release, the commit timestamp and
a random commit GUID. It named a branch on line 3 until 2026-10-08, and a
commit made from a detached worktree, as every track's are, kept whatever
name the worktree inherited, so `fx2/stage4` rode 30 of 40 trunk commits
into every banner. A post-merge hook existed only to restamp that line and
went with it.

A clone runs the hook it has installed, not the one in the tree, so an old
hook would go on writing four lines; the shape test below stops such a
clone at its next commit and says how to update the hook.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from tests.ast_seams import REPO_ROOT
from tools.install_hooks import _HOOK_SCRIPT, _STAMP_BLOCK

_WRITE = '> "$VERSION_FILE"'

# The guard gate runs this module inside the pre-commit hook, where git has
# put GIT_INDEX_FILE and GIT_DIR into the environment so the hook judges the
# index being committed. Inherited by a git command in a temporary repo,
# they redirect it at the repository being committed: the first run of this
# test re-inited the shared repository as bare and rewrote its user. The hook
# now drops them before running the guards; this scrub stays because a clone
# runs the hook it has installed, not the one in the tree.
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}

# A GUID the stamp makes: eight hex digits, or its fallback when neither
# python3 nor openssl can make one.
_GUID = re.compile(r'[0-9a-f]{8}|nogenuid')


def test_the_hook_embeds_the_one_stamp_block():
    assert _STAMP_BLOCK in _HOOK_SCRIPT


def test_nothing_writes_the_file_outside_the_block():
    assert _STAMP_BLOCK.count(_WRITE) == 1
    assert _HOOK_SCRIPT.count(_WRITE) == 1, (
        'a second writer of version.txt has appeared in the hook'
    )


def test_the_block_asks_git_for_no_branch():
    for asks in ('symbolic-ref', '@{u}', '--abbrev-ref'):
        assert asks not in _STAMP_BLOCK


def test_version_txt_has_the_three_line_shape():
    lines = (REPO_ROOT / 'version.txt').read_text(encoding='utf-8-sig').splitlines()
    assert len(lines) == 3 and re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d', lines[1]), (
        f'version.txt is {lines!r}, not release / timestamp / GUID. A four-line file '
        f'was written by an out-of-date stamp hook: run python3 tools/install_hooks.py '
        f'--install, then restage version.txt'
    )
    assert _GUID.fullmatch(lines[2])


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ['git', '-C', str(repo), *args], text=True, env=_CLEAN_ENV
    ).strip()


def _repo_with_version_file(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'dev/x')
    _git(repo, 'config', 'user.email', 't@t')
    _git(repo, 'config', 'user.name', 't')
    (repo / 'version.txt').write_text('1.0\n2000-01-01 00:00\n00000000\n')
    _git(repo, 'add', 'version.txt')
    _git(repo, 'commit', '-q', '-m', 'seed')
    return repo


def _run_stamp(repo: Path) -> list[str]:
    script = f'set -e\ncd "{repo}"\nVERSION_FILE="{repo}/version.txt"\n{_STAMP_BLOCK}'
    subprocess.run(['bash', '-c', script], check=True, env=_CLEAN_ENV)
    return (repo / 'version.txt').read_text().splitlines()


def _assert_stamped(lines: list[str]) -> None:
    assert len(lines) == 3
    assert lines[0] == '1.0'
    assert re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d', lines[1]) and lines[1] != '2000-01-01 00:00'
    assert _GUID.fullmatch(lines[2]) and lines[2] != '00000000'


def test_a_named_branch_stamps_three_lines(tmp_path):
    repo = _repo_with_version_file(tmp_path)
    _git(repo, 'checkout', '-q', '-b', 'feature/y')
    _assert_stamped(_run_stamp(repo))


def test_a_detached_checkout_stamps_three_lines(tmp_path):
    repo = _repo_with_version_file(tmp_path)
    _git(repo, 'checkout', '-q', '--detach')
    _assert_stamped(_run_stamp(repo))


def test_a_branch_tracking_the_trunk_stamps_three_lines(tmp_path):
    repo = _repo_with_version_file(tmp_path)
    origin = repo.parent / 'origin.git'
    _git(repo, 'init', '-q', '--bare', str(origin))
    _git(repo, 'remote', 'add', 'origin', str(origin))
    _git(repo, 'push', '-q', 'origin', 'dev/x')
    _git(repo, 'checkout', '-q', '-b', 'triage/shape-a-5.3')
    _git(repo, 'branch', '-q', '-u', 'origin/dev/x')
    _assert_stamped(_run_stamp(repo))
