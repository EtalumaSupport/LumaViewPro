# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``tools/tests_reaching.py`` names every test file that reaches a changed module.

The contract is the tool's docstring: a test file is in the set when it
imports the changed module at any depth, imports a tests helper that does,
requests a conftest fixture whose body reaches it, imports a production
module that imports the changed module at module level (closed over the
production graph), or is a guard; a module an autouse fixture imports
itself makes the set the whole suite; a name in a string reaches nothing.
One throwaway tree holds one file per clause, so each assertion names the
clause it pins.
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

# The production importers of ``modules.owner``, one per import shape, each
# with a test file that imports only the importer. Kept out of ``_TREE`` so
# the clauses above keep their sets.
_CALLERS = {
    'modules/caller.py': 'import modules.owner\n',
    'modules/caller_of_caller.py': 'from modules import caller\n',
    'modules/guarded_caller.py': 'try:\n    import modules.owner\nexcept ImportError:\n    pass\n',
    'modules/lazy_caller.py': 'def f():\n    from modules.owner import member\n    return member()\n',
    'modules/typed_caller.py': (
        'from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import modules.owner\n'
    ),
    'tests/test_via_caller.py': 'import modules.caller\n\ndef test_l():\n    pass\n',
    'tests/test_via_chain.py': 'import modules.caller_of_caller\n\ndef test_m():\n    pass\n',
    'tests/test_via_guarded_caller.py': 'import modules.guarded_caller\n\ndef test_n():\n    pass\n',
    'tests/test_via_lazy_caller.py': 'import modules.lazy_caller\n\ndef test_o():\n    pass\n',
    'tests/test_via_typed_caller.py': 'import modules.typed_caller\n\ndef test_p():\n    pass\n',
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
        'through a tests helper, or through a production module importing it), '
        '1 through a conftest fixture, 1 guards (always)'
    ]


def test_a_module_its_package_exports_is_reached_through_the_package(tmp_path):
    reach = reach_of(_tree(tmp_path), ['modules/pkg/_inner.py'])
    assert reach.files == ['tests/guards/test_a_guard.py', 'tests/test_via_package.py'], (
        reach.reasons
    )


def test_a_test_importing_a_production_importer_of_the_change_is_reached_through_it(tmp_path):
    """A production module that imports the changed module at module level is
    reached, and so is one that imports that module (the closure); a test file
    importing only the importer is in the set. An import that runs when the
    module is imported counts, under a top-level ``try`` too; one inside a
    function does not (49 of 60 commits at the pin land on one 895-file set
    when function-level production imports are edges), and one under
    ``if TYPE_CHECKING:`` never runs. LVP ce7c3748 changed the post-processor
    and the stitcher's tests, which import the stitcher and not the processor,
    were left out of the targeted run and failed at the push."""
    reach = reach_of(_tree(tmp_path, _CALLERS), ['modules/owner.py'])
    assert 'tests/test_via_caller.py' in reach.files
    assert 'tests/test_via_chain.py' in reach.files
    assert 'tests/test_via_guarded_caller.py' in reach.files
    assert 'tests/test_via_lazy_caller.py' not in reach.files
    assert 'tests/test_via_typed_caller.py' not in reach.files
    assert reach.reasons[0] == (
        'reached through 3 production modules importing it at module level: '
        'modules.caller, modules.caller_of_caller, modules.guarded_caller'
    )


def test_a_module_reached_only_through_an_importer_the_autouse_fixture_imports_is_not_the_whole_suite(
    tmp_path,
):
    """The whole-suite rule is keyed on the edited modules: a change to a module
    the autouse fixture reaches only through a production importer is matched
    like any other import, not widened to every test. Nearly every production
    module is in the session module's closure, so the other reading makes 44
    of 60 commits at the pin the whole suite."""
    conftest = _TREE['tests/conftest.py'] + (
        '\n@pytest.fixture(autouse=True)\ndef _always():\n    import modules.caller\n'
    )
    root = _tree(tmp_path, {**_CALLERS, 'tests/conftest.py': conftest})
    reach = reach_of(root, ['modules/owner.py'])
    assert not any(reason.startswith('the whole suite') for reason in reach.reasons)
    assert 'tests/test_string_only.py' not in reach.files
    whole = reach_of(root, ['modules/caller.py'])
    assert whole.reasons == [
        'the whole suite: modules.caller is imported by the autouse fixture _always'
    ]


def test_a_package_exporting_the_change_inside_a_function_is_reached_and_seeds_no_closure(tmp_path):
    """A package is what a test names, and its lazy import runs under the test
    that calls it (``modules/plugins/__init__.py`` imports ``builtin`` inside
    ``load``), so the package rule reads the ``__init__`` at any depth. The
    package is not a seed of the closure: a module importing the package at
    module level is not reached through its lazy export, or a ``builtin``
    change pulls 623 files through the session's import of the package."""
    root = _tree(
        tmp_path,
        {
            'modules/lazypkg/__init__.py': (
                'def load():\n    from modules.lazypkg import _lazy\n    return _lazy\n'
            ),
            'modules/lazypkg/_lazy.py': 'LAZY = 1\n',
            'modules/lazypkg_user.py': 'import modules.lazypkg\n',
            'tests/test_via_lazypkg.py': 'import modules.lazypkg\n\ndef test_q():\n    pass\n',
            'tests/test_via_lazypkg_user.py': (
                'import modules.lazypkg_user\n\ndef test_r():\n    pass\n'
            ),
        },
    )
    reach = reach_of(root, ['modules/lazypkg/_lazy.py'])
    assert reach.files == ['tests/guards/test_a_guard.py', 'tests/test_via_lazypkg.py'], (
        reach.reasons
    )
    assert reach.reasons[0] == 'reached through the package import too: modules.lazypkg'


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
    assert err.splitlines()[0] == (
        'reached through 1 production module importing it at module level: modules.pkg'
    )
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
