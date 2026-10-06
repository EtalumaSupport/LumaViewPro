"""The pre-commit hook judges the staged production hunks, in every commit form.

The no-quick-fix judge used to sit on the Edit and Write tools alone, so an
edit written through Bash landed unjudged. The hook the installer writes
now pipes the staged .py diff into the judge a Claude Code session names in
``COMMIT_JUDGE``; a terminal, which has no such variable, commits as
before, and a merge commit is not judged (Eric, 2026-10-03). These tests
install the real template into a throwaway repository and commit through it
with a stub judge that records the diff it was handed and refuses a
swallowed ValueError. Every test sets or drops the variable itself, so none
inherits the session's real judge and pays for an LLM call.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.install_hooks import _HOOK_SCRIPT, _JUDGE_STAGE

# Inherited from a hook, GIT_INDEX_FILE and GIT_DIR point a git command in
# a temporary repository at the repository being committed (one run of a
# sibling test re-inited the shared checkout as bare). The session's own
# judge is dropped too: a test that forgot to set the variable would
# otherwise make paid calls from inside pytest.
_CLEAN_ENV = {
    k: v for k, v in os.environ.items() if not k.startswith('GIT_') and k != 'COMMIT_JUDGE'
}

_CLEAN = 'def read(path):\n    with open(path) as f:\n        return f.read()\n'
_CLEANER = 'def read(path):\n    with open(path) as f:\n        return f.read().strip()\n'
_BANDAID = (
    'def read(path):\n'
    '    try:\n'
    '        with open(path) as f:\n'
    '            return f.read()\n'
    '    except ValueError:\n'
    '        return None\n'
)
_MERGE_BANNER = 'pre-commit: merge commit, not judged by the no-quick-fix judge (Eric, 2026-10-03)'


def test_the_hook_carries_the_judge_stage_before_the_version_stamp():
    """The stage is read through the installer's symbol: what it writes is the gate.
    A deny before ``VERSION_FILE=`` leaves version.txt untouched by the refused commit."""
    assert _JUDGE_STAGE in _HOOK_SCRIPT
    assert _HOOK_SCRIPT.index(_JUDGE_STAGE) < _HOOK_SCRIPT.index(
        'VERSION_FILE="$REPO_ROOT/version.txt"'
    )
    assert _HOOK_SCRIPT.index('tests/guards') < _HOOK_SCRIPT.index(_JUDGE_STAGE), (
        'the paid judge runs after the free gates'
    )


def test_the_stage_pins_the_diff_it_reads():
    for flag in (
        '--text',
        '--no-textconv',
        '-M',
        '--src-prefix=a/',
        '--dst-prefix=b/',
        '-U3',
        '--no-ext-diff',
        '--no-color',
    ):
        assert flag in _JUDGE_STAGE, (
            f'the stage must pin {flag}; a session git setting could change the diff otherwise'
        )
    assert 'unset GIT_DIFF_OPTS' in _JUDGE_STAGE
    assert 'set -o pipefail' in _JUDGE_STAGE
    assert 'MERGE_HEAD' in _JUDGE_STAGE
    assert _MERGE_BANNER in _JUDGE_STAGE


class _Repo:
    def __init__(self, root: Path):
        self.root = root
        self.log = root.parent / 'judge.log'
        self.stub = root.parent / 'judge_stub.sh'
        self.stub.write_text(
            '#!/bin/sh\n'
            'diff=$(cat)\n'
            f'printf \'%s\' "$diff" > "{self.log}"\n'
            'case "$diff" in *"except ValueError"*) echo "stub: refused" >&2; exit 1;; esac\n'
            'exit 0\n'
        )
        self.stub.chmod(0o755)

    def git(
        self, *args: str, judged: bool = True, cwd: Path | None = None
    ) -> subprocess.CompletedProcess:
        env = dict(_CLEAN_ENV)
        if judged:
            env['COMMIT_JUDGE'] = str(self.stub)
        return subprocess.run(
            ['git', *args], cwd=cwd or self.root, env=env, capture_output=True, text=True
        )

    def ok(self, *args: str, **kw) -> str:
        r = self.git(*args, **kw)
        assert r.returncode == 0, f'git {args} failed:\n{r.stdout}\n{r.stderr}'
        return r.stdout

    def write(self, rel: str, text: str, root: Path | None = None) -> None:
        p = (root or self.root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def version(self) -> str:
        return (self.root / 'version.txt').read_text()

    def judged_diff(self) -> str | None:
        return self.log.read_text() if self.log.exists() else None


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    root = tmp_path / 'repo'
    root.mkdir()
    r = _Repo(root)
    r.ok('init', '-q', '-b', 'main')
    r.ok('config', 'user.email', 't@t')
    r.ok('config', 'user.name', 't')
    r.write('version.txt', '1.0\n2000-01-01 00:00\nmain\n00000000\n')
    r.write('modules/x.py', _CLEAN)
    r.write('docs/a.md', 'a\n')
    r.ok('add', 'version.txt', 'modules/x.py', 'docs/a.md')
    r.ok('commit', '-q', '-m', 'seed', judged=False)
    hook = root / '.git' / 'hooks' / 'pre-commit'
    hook.write_text(_HOOK_SCRIPT)
    hook.chmod(0o755)
    return r


def test_ih1_a_pathspec_commit_of_an_unstaged_bandaid_is_refused_and_version_txt_is_clean(repo):
    before = repo.version()
    repo.write('modules/x.py', _BANDAID)
    r = repo.git('commit', '-q', '-m', 'x', '--', 'modules/x.py')
    assert r.returncode != 0
    assert 'modules/x.py' in (repo.judged_diff() or '')
    assert repo.version() == before


def test_ih2_a_staged_bandaid_is_refused(repo):
    repo.write('modules/x.py', _BANDAID)
    repo.ok('add', 'modules/x.py')
    assert repo.git('commit', '-q', '-m', 'x').returncode != 0


def test_ih3_an_amend_with_a_staged_bandaid_is_refused(repo):
    repo.write('modules/x.py', _CLEANER)
    repo.ok('commit', '-q', '-m', 'clean', '--', 'modules/x.py')
    repo.write('modules/x.py', _BANDAID)
    repo.ok('add', 'modules/x.py')
    assert repo.git('commit', '-q', '--amend', '--no-edit').returncode != 0


def test_ih4_a_pathspec_commit_beside_a_peers_staged_bandaid_lands_and_the_judge_saw_only_its_file(
    repo,
):
    repo.write('modules/y.py', _BANDAID)
    repo.ok('add', 'modules/y.py')
    repo.write('modules/x.py', _CLEANER)
    repo.ok('commit', '-q', '-m', 'x only', '--', 'modules/x.py')
    seen = repo.judged_diff() or ''
    assert 'modules/x.py' in seen
    assert 'modules/y.py' not in seen


def test_ih5_without_the_variable_the_commit_lands_and_the_judge_is_not_called(repo):
    repo.write('modules/x.py', _BANDAID)
    repo.ok('commit', '-q', '-m', 'terminal', '--', 'modules/x.py', judged=False)
    assert repo.judged_diff() is None


def test_ih6_a_docs_only_commit_hands_the_judge_no_hunk(repo):
    repo.write('docs/a.md', 'b\n')
    repo.ok('commit', '-q', '-m', 'docs', '--', 'docs/a.md')
    assert repo.judged_diff() == ''


def test_ih7_a_worktree_commit_runs_the_shared_hook(repo, tmp_path):
    wt = tmp_path / 'wt'
    repo.ok('worktree', 'add', '-q', str(wt), '-b', 'w')
    repo.write('modules/x.py', _BANDAID, root=wt)
    r = repo.git('commit', '-q', '-m', 'x', '--', 'modules/x.py', cwd=wt)
    assert r.returncode != 0
    assert 'modules/x.py' in (repo.judged_diff() or '')


def test_ih8_a_diff_attribute_cannot_hide_the_hunk(repo, tmp_path):
    attrs = tmp_path / 'attributes'
    attrs.write_text('modules/x.py -diff\n')
    repo.write('modules/x.py', _BANDAID)
    r = repo.git(
        '-c', f'core.attributesFile={attrs}', 'commit', '-q', '-m', 'x', '--', 'modules/x.py'
    )
    assert r.returncode != 0
    assert 'except ValueError' in (repo.judged_diff() or ''), (
        'the stage must read the text hunk, not "Binary files differ"'
    )


def test_ih9_a_merge_concluded_by_hand_is_not_judged_and_says_so(repo):
    repo.ok('checkout', '-q', '-b', 'side')
    repo.write('modules/x.py', _CLEANER)
    repo.ok('commit', '-q', '-m', 'side', '--', 'modules/x.py', judged=False)
    repo.ok('checkout', '-q', 'main')
    repo.write('modules/x.py', _CLEAN.replace('read', 'load'))
    repo.ok('commit', '-q', '-m', 'main', '--', 'modules/x.py', judged=False)
    # A pathspec commit stamps version.txt into the commit's temporary index
    # and leaves the real index behind; that dirt would make the merge refuse
    # before it conflicts, which is not the shape under test.
    repo.ok('reset', '-q', '--hard')
    assert repo.git('merge', 'side').returncode != 0
    assert (repo.root / '.git' / 'MERGE_HEAD').exists(), (
        'the merge must conflict for the hand-concluded shape'
    )
    repo.write('modules/x.py', _BANDAID)
    repo.ok('checkout', '--ours', '--', 'version.txt')
    repo.ok('add', 'modules/x.py', 'version.txt')
    r = repo.git('commit', '-q', '-m', 'merge')
    assert r.returncode == 0, r.stderr
    assert repo.judged_diff() is None
    assert _MERGE_BANNER in r.stderr
