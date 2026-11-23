"""A file renamed and edited in one commit is checked for its edits.

The pre-commit (`tools/check_rules.py --staged`) reads a renamed file as if
it had been edited in place: the lines the edit adds are checked, the lines
carried over from the old path are not, and a pure rename is checked as an
in-place commit that changes nothing. Each case runs under every
`diff.renames` setting, since the user's git config decides how a rename is
shown unless the checker names `-M` itself.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import main

# Rule 24 is line-scoped, so it shows which lines the checker called added.
NON_ASCII = chr(0xB0)
CARRIED = f'# carried {NON_ASCII}\n'
BODY = ''.join(f'value_{n} = {n}\n' for n in range(30))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path, name: str, repo_key: str, renames: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'test')
    _git(repo, 'config', 'diff.renames', renames)
    (repo / 'pyproject.toml').write_text(f'[tool.etaluma]\nrepo = "{repo_key}"\n')
    return repo


def _commit(repo: Path, path: str, text: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text, encoding='utf-8')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'seed')


@pytest.mark.parametrize('renames', ['true', 'false', 'copies'])
def test_a_renamed_file_is_checked_for_the_lines_its_edit_adds(
    tmp_path, monkeypatch, capsys, renames
):
    repo = _repo(tmp_path, 'LumaViewPro', 'lumaviewpro', renames)
    _commit(repo, 'old_name.py', CARRIED + BODY)
    _git(repo, 'mv', 'old_name.py', 'new_name.py')
    (repo / 'new_name.py').write_text(CARRIED + BODY + f'# added {NON_ASCII}\n', encoding='utf-8')
    _git(repo, 'add', 'new_name.py')
    monkeypatch.chdir(repo)

    assert main(['--staged']) == 1
    flagged = [line for line in capsys.readouterr().err.splitlines() if 'rule_24' in line]
    assert [line.split(' ', 1)[0] for line in flagged] == ['new_name.py:32:8']


@pytest.mark.parametrize('renames', ['true', 'false', 'copies'])
def test_a_pure_rename_is_checked_as_a_commit_that_changes_nothing(
    tmp_path, monkeypatch, capsys, renames
):
    repo = _repo(tmp_path, 'LumaViewPro', 'lumaviewpro', renames)
    _commit(repo, 'old_name.py', CARRIED + BODY)
    _git(repo, 'mv', 'old_name.py', 'new_name.py')
    monkeypatch.chdir(repo)

    assert main(['--staged']) == 0, capsys.readouterr().err


def test_a_misordered_daily_log_moved_with_no_edit_is_not_refused(tmp_path, monkeypatch, capsys):
    """The whole-file doc checks read a shard's content; a pure move changes
    none of it, so it is not the commit that put the dates out of order."""
    repo = _repo(tmp_path, 'Firmware', 'etaluma-firmware', 'true')
    misordered = '# Log\n\n## 2026-10-01 (t): older\n\nA.\n\n## 2026-10-02 (t): newer\n\nB.\n'
    _commit(repo, 'docs/notes.md', misordered)
    _git(repo, 'mv', 'docs/notes.md', 'docs/DAILY_LOG_2026-10.md')
    monkeypatch.chdir(repo)

    assert main(['--staged']) == 0, capsys.readouterr().err
