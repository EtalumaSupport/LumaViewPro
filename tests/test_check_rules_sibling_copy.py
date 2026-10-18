"""tools/check_rules.py is one file in both repos, kept so by copy.

A commit that touches the checker is refused while the staged copy differs
from the sibling checkout's copy, and refused when that copy is absent. The
sibling is the other repo's checkout beside this one carrying this
checkout's own track suffix: `../Firmware-<track>` beside
`LumaViewPro-<track>`, `../Firmware` beside the bare `LumaViewPro`. Every
track commits in its own worktree and the shared checkouts move only on a
pull, so a witness in the shared checkout could not hold two concerns as
two commits in both repos (2026-10-16: LVP 1f73a417 and 10707964 landed in
Firmware as the one commit 98050e16).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import _check_sibling_copy, _sibling_checker, main

CHECKER = ROOT / 'tools' / 'check_rules.py'


def _sibling_with(tmp_path: Path, checkout: str, text: str) -> Path:
    sibling = tmp_path / checkout / 'tools' / 'check_rules.py'
    sibling.parent.mkdir(parents=True)
    sibling.write_text(text, encoding='utf-8')
    return sibling


def test_a_staged_copy_that_differs_from_the_sibling_is_blocked(tmp_path):
    sibling = _sibling_with(tmp_path, 'Firmware-track', 'a = 1\n')
    violations = _check_sibling_copy('a = 2\n', sibling)
    assert [v.rule for v in violations] == ['sibling_copy']
    assert violations[0].severity == 'block'
    assert str(sibling) in violations[0].message


def test_a_staged_copy_identical_to_the_sibling_passes(tmp_path):
    sibling = _sibling_with(tmp_path, 'Firmware-track', 'a = 1\n')
    assert _check_sibling_copy('a = 1\n', sibling) == []


def test_a_missing_sibling_copy_is_blocked_and_the_expected_path_named(tmp_path):
    """A skipped check is how the two copies drifted before the check existed;
    a worktree whose sibling is not where the rule puts it is told where."""
    expected = tmp_path / 'Firmware-track' / 'tools' / 'check_rules.py'
    violations = _check_sibling_copy('anything\n', expected)
    assert [v.rule for v in violations] == ['sibling_copy']
    assert violations[0].severity == 'block'
    assert str(expected) in violations[0].message


def test_the_sibling_copy_carries_this_checkouts_track_suffix(tmp_path, monkeypatch):
    """One rule for both shapes: `<other>-<track>` beside `<repo>-<track>`,
    the bare `<other>` beside the bare `<repo>`; a path is returned whether or
    not the copy exists, so a missing one is refused, never skipped."""
    for checkout, sibling in (
        ('LumaViewPro-track', 'Firmware-track'),
        ('LumaViewPro', 'Firmware'),
    ):
        repo = tmp_path / checkout
        repo.mkdir()
        subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
        monkeypatch.chdir(repo)
        assert _sibling_checker('Firmware') == tmp_path / sibling / 'tools' / 'check_rules.py', (
            checkout
        )


def test_main_refuses_the_staged_checker_only_while_the_sibling_differs(
    tmp_path, monkeypatch, capsys
):
    repo = tmp_path / 'LumaViewPro-track'
    (repo / 'tools').mkdir(parents=True)
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    (repo / 'pyproject.toml').write_text('[tool.etaluma]\nrepo = "lumaviewpro"\n')
    current = CHECKER.read_text(encoding='utf-8')
    (repo / 'tools' / 'check_rules.py').write_text(current, encoding='utf-8')
    subprocess.run(['git', 'add', 'tools/check_rules.py'], cwd=repo, check=True)
    sibling = _sibling_with(tmp_path, 'Firmware-track', 'drifted\n')
    monkeypatch.chdir(repo)

    assert main(['--staged']) == 1
    assert 'sibling_copy' in capsys.readouterr().err

    sibling.write_text(current, encoding='utf-8')
    main(['--staged'])
    assert 'sibling_copy' not in capsys.readouterr().err
