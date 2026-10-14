# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``tools/tests_reaching.py`` names every test file that reaches a changed module.

The contract is the tool's docstring: a test file is in the set when it
imports the changed module at any depth, imports a tests helper that does,
requests a conftest fixture whose body reaches it, or is a guard; a module
an autouse fixture imports makes the set the whole suite; a name in a
string reaches nothing. One throwaway tree holds one file per clause, so
each assertion names the clause it pins.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.tests_reaching import main, reach_of

_TREE = {
    'modules/owner.py': 'def member():\n    return 1\n',
    'modules/other.py': 'X = 1\n',
    'modules/autoused.py': 'Y = 2\n',
    'tests/__init__.py': '',
    'tests/conftest.py': (
        'import pytest\n'
        'from modules.owner import member\n'
        '\n'
        '@pytest.fixture\n'
        'def owner_fixture():\n'
        '    return member()\n'
        '\n'
        '@pytest.fixture\n'
        'def other_fixture():\n'
        '    import modules.other\n'
        '    return modules.other.X\n'
    ),
    'tests/helper.py': 'from modules import owner\n',
    'tests/test_top_level.py': 'import modules.owner\n\ndef test_a():\n    pass\n',
    'tests/test_in_function.py': 'def test_b():\n    from modules.owner import member\n    assert member()\n',
    'tests/test_via_helper.py': 'from tests import helper\n\ndef test_c():\n    pass\n',
    'tests/test_via_fixture.py': 'def test_d(owner_fixture):\n    pass\n',
    'tests/test_usefixtures.py': (
        'import pytest\n\n@pytest.mark.usefixtures("other_fixture")\ndef test_e():\n    pass\n'
    ),
    'tests/test_string_only.py': 'NAME = "modules.owner.member"\n\ndef test_f():\n    pass\n',
    'tests/test_unrelated.py': 'import modules.other\n\ndef test_g():\n    pass\n',
    'tests/guards/test_a_guard.py': 'def test_h():\n    pass\n',
    'modules/pkg/__init__.py': 'from ._inner import thing\n',
    'modules/pkg/_inner.py': 'thing = 1\n',
    'tests/test_via_package.py': 'from modules.pkg import thing\n\ndef test_i():\n    pass\n',
    'tests/test_relative.py': 'from .helper import owner\n\ndef test_j():\n    pass\n',
}


def _tree(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    for rel, text in {**_TREE, **(extra or {})}.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_a_change_is_reached_by_every_import_shape_and_by_no_string(tmp_path):
    reach = reach_of(_tree(tmp_path), ['modules/owner.py'])
    assert reach.files == [
        'tests/guards/test_a_guard.py',
        'tests/test_in_function.py',
        'tests/test_relative.py',
        'tests/test_top_level.py',
        'tests/test_via_fixture.py',
        'tests/test_via_helper.py',
    ], reach.reasons
    assert reach.reasons == [
        "6 test files for ['modules.owner']: 4 by import (module level or inside a function, "
        'or through a tests helper), 1 through a conftest fixture, 1 guards (always)'
    ]


def test_a_module_its_package_exports_is_reached_through_the_package(tmp_path):
    reach = reach_of(_tree(tmp_path), ['modules/pkg/_inner.py'])
    assert reach.files == ['tests/guards/test_a_guard.py', 'tests/test_via_package.py'], (
        reach.reasons
    )


def test_a_fixture_named_by_usefixtures_reaches_what_its_body_imports(tmp_path):
    reach = reach_of(_tree(tmp_path), ['modules/other.py'])
    assert reach.files == [
        'tests/guards/test_a_guard.py',
        'tests/test_unrelated.py',
        'tests/test_usefixtures.py',
    ], reach.reasons


def test_a_module_an_autouse_fixture_imports_is_the_whole_suite_with_its_reason(tmp_path):
    conftest = _TREE['tests/conftest.py'] + (
        '\n@pytest.fixture(autouse=True)\ndef _always():\n    import modules.autoused\n'
    )
    root = _tree(tmp_path, {'tests/conftest.py': conftest})
    reach = reach_of(root, ['modules/autoused.py'])
    assert reach.files == sorted(rel for rel in _TREE if Path(rel).name.startswith('test_'))
    assert reach.reasons == [
        'the whole suite: modules.autoused is imported by the autouse fixture _always'
    ]


def test_the_command_prints_one_path_per_line_and_refuses_no_production_path(tmp_path, capsys):
    root = _tree(tmp_path)
    assert main(['--root', str(root), 'modules/pkg/_inner.py', 'docs/x.md']) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ['tests/guards/test_a_guard.py', 'tests/test_via_package.py']
    assert err.splitlines()[0] == 'reached through the package import too: modules.pkg'
    assert main(['--root', str(root), 'docs/x.md']) == 2
    assert capsys.readouterr().err.strip() == 'no changed production .py path given'


def test_a_rest_path_is_production_and_is_reached_by_the_tests_that_import_it(tmp_path, capsys):
    """``rest/`` is the shipped REST host, an L2 caller in the repository, so a
    change under it is a changed production path like one under ``modules/``.
    Before, the tool answered "no changed production .py path given" for a
    REST-only change and the session built its list by hand (primary,
    2026-10-14, at 97abcaca)."""
    root = _tree(
        tmp_path,
        {
            'rest/__init__.py': '',
            'rest/app.py': 'def app():\n    return 1\n',
            'tests/test_rest_app.py': 'from rest.app import app\n\ndef test_k():\n    assert app()\n',
        },
    )
    assert main(['--root', str(root), 'rest/app.py']) == 0
    out, _err = capsys.readouterr()
    assert out.splitlines() == ['tests/guards/test_a_guard.py', 'tests/test_rest_app.py']


def test_the_changed_paths_come_from_a_commit_or_the_working_tree(tmp_path, capsys):
    root = _tree(tmp_path)

    def git(*args):
        subprocess.run(
            ['git', '-C', str(root), '-c', 'user.email=t@x', '-c', 'user.name=t', *args],
            check=True,
        )

    git('init', '-q')
    git('add', '.')
    git('commit', '-qm', 'base')
    (root / 'modules/other.py').write_text('X = 2\n')
    git('commit', '-qam', 'other')
    assert main(['--root', str(root), '--commit', 'HEAD']) == 0
    assert capsys.readouterr().out.splitlines() == [
        'tests/guards/test_a_guard.py',
        'tests/test_unrelated.py',
        'tests/test_usefixtures.py',
    ]
    (root / 'modules/owner.py').write_text('def member():\n    return 2\n')
    assert main(['--root', str(root), '--diff']) == 0
    assert 'tests/test_top_level.py' in capsys.readouterr().out.splitlines()
