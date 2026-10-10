"""tools/check_rules.py is one file in both repos, kept so by copy.

A commit that touches the checker is refused while the staged copy differs
from the sibling checkout's copy beside this repo; with no sibling checkout
beside it the check is skipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import _check_sibling_copy, _sibling_checker, main

CHECKER = ROOT / 'tools' / 'check_rules.py'


def _sibling_with(tmp_path: Path, text: str) -> Path:
    sibling = tmp_path / 'Firmware' / 'tools' / 'check_rules.py'
    sibling.parent.mkdir(parents=True)
    sibling.write_text(text, encoding='utf-8')
    return sibling


def test_a_staged_copy_that_differs_from_the_sibling_is_blocked(tmp_path):
    sibling = _sibling_with(tmp_path, 'a = 1\n')
    violations = _check_sibling_copy('a = 2\n', sibling)
    assert [v.rule for v in violations] == ['sibling_copy']
    assert violations[0].severity == 'block'
    assert str(sibling) in violations[0].message


def test_a_staged_copy_identical_to_the_sibling_passes(tmp_path):
    sibling = _sibling_with(tmp_path, 'a = 1\n')
    assert _check_sibling_copy('a = 1\n', sibling) == []


def test_no_sibling_checkout_skips_the_check():
    assert _check_sibling_copy('anything\n', None) == []


def test_the_sibling_copy_is_found_beside_the_git_toplevel(tmp_path, monkeypatch):
    repo = tmp_path / 'LumaViewPro-track'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    sibling = _sibling_with(tmp_path, '')
    monkeypatch.chdir(repo)
    assert _sibling_checker('Firmware') == sibling
    assert _sibling_checker('NoSuchCheckout') is None


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
    sibling = _sibling_with(tmp_path, 'drifted\n')
    monkeypatch.chdir(repo)

    assert main(['--staged']) == 1
    assert 'sibling_copy' in capsys.readouterr().err

    sibling.write_text(current, encoding='utf-8')
    main(['--staged'])
    assert 'sibling_copy' not in capsys.readouterr().err
