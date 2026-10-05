"""tools/confirm_pins.py proves every production hunk of a range is pinned.

The pure stages (the hunk cutter, the scope filter, the neutrality decision,
the junit mapping, the observing set, the tip baseline, the verdict) are
tested on recorded inputs; the whole tool runs end to end on a temporary
repository through the real git and the real pytest, and the restore is
asserted with the real ``git status``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import confirm_pins as cp

# A diff cut at -U1: one blank context line (a single space) ends the first hunk.
RECORDED_DIFF = '\n'.join(
    [
        'diff --git a/modules/calc.py b/modules/calc.py',
        'index 1111111..2222222 100644',
        '--- a/modules/calc.py',
        '+++ b/modules/calc.py',
        '@@ -2,3 +2,3 @@ def add(a, b):',
        ' def add(a, b):',
        '-    return a + b + 1',
        '+    return a + b',
        ' ',
        '@@ -8,2 +8,2 @@ def mul(a, b):',
        ' def mul(a, b):',
        '-    return a * b + 1',
        '+    return a * b',
        'diff --git a/modules/new.py b/modules/new.py',
        'new file mode 100644',
        'index 0000000..3333333',
        '--- /dev/null',
        '+++ b/modules/new.py',
        '@@ -0,0 +1,2 @@',
        '+def fresh():',
        '+    return 1',
        'diff --git a/modules/old.py b/modules/old.py',
        'deleted file mode 100644',
        'index 4444444..0000000',
        '--- a/modules/old.py',
        '+++ /dev/null',
        '@@ -1,2 +0,0 @@',
        '-def stale():',
        '-    return 0',
        '',
    ]
)


class TestTheHunkCutter:
    def test_each_hunk_is_cut_with_its_file_and_header(self):
        hunks = cp.cut_hunks(RECORDED_DIFF)
        assert [(h.path, h.header) for h in hunks] == [
            ('modules/calc.py', '@@ -2,3 +2,3 @@ def add(a, b):'),
            ('modules/calc.py', '@@ -8,2 +8,2 @@ def mul(a, b):'),
            ('modules/new.py', '@@ -0,0 +1,2 @@'),
            ('modules/old.py', '@@ -1,2 +0,0 @@'),
        ]

    def test_a_hunk_patch_is_its_file_header_then_the_one_hunk(self):
        second = cp.cut_hunks(RECORDED_DIFF)[1].patch
        assert second.startswith('diff --git a/modules/calc.py b/modules/calc.py\n')
        assert '--- a/modules/calc.py\n+++ b/modules/calc.py\n@@ -8,2 +8,2 @@' in second
        assert '@@ -2,3' not in second

    def test_a_new_and_a_deleted_file_carry_their_mode_lines(self):
        new, old = cp.cut_hunks(RECORDED_DIFF)[2:]
        assert 'new file mode 100644\n' in new.patch and '--- /dev/null\n' in new.patch
        assert 'deleted file mode 100644\n' in old.patch and '+++ /dev/null\n' in old.patch

    def test_an_empty_diff_has_no_hunks(self):
        assert cp.cut_hunks('') == []

    def test_a_move_names_both_hunks(self):
        hunks = cp.cut_hunks(
            textwrap.dedent("""\
            diff --git a/modules/m.py b/modules/m.py
            index 1..2 100644
            --- a/modules/m.py
            +++ b/modules/m.py
            @@ -1,3 +1,1 @@
            -def f():
            -    return 1
             x = 1
            @@ -9,1 +7,3 @@
             y = 2
            +def f():
            +    return 1
            """)
        )
        assert cp.moved_pairs(hunks) == {0: 1, 1: 0}


class TestTheScopeFilter:
    @pytest.mark.parametrize(
        'path, production',
        [
            ('modules/protocol.py', True),
            ('drivers/camera.py', True),
            ('ui/panel.py', True),
            ('tools/confirm_pins.py', True),
            ('lumaviewpro.py', True),
            ('ui/panel.kv', False),
            ('data/image.png', False),
            ('tests/test_protocol.py', False),
            ('plugins/engineering/plugin.py', False),
            ('scripts/build.py', False),
        ],
    )
    def test_only_a_py_at_the_root_or_under_a_production_package(self, path, production):
        assert cp.is_production_path(path) is production


class TestTheNeutralityDecision:
    BEFORE = 'def f(x):\n    """Old words."""\n    # old note\n    return x + 1\n'

    def test_a_docstring_edit_is_neutral(self):
        assert cp.is_neutral(self.BEFORE, self.BEFORE.replace('Old words', 'New words'))

    def test_a_comment_edit_is_neutral(self):
        assert cp.is_neutral(self.BEFORE, self.BEFORE.replace('old note', 'new note'))

    def test_a_one_token_code_edit_is_behavioural(self):
        assert not cp.is_neutral(self.BEFORE, self.BEFORE.replace('x + 1', 'x + 2'))

    def test_a_file_absent_on_one_side_is_behavioural(self):
        assert not cp.is_neutral(None, self.BEFORE)
        assert not cp.is_neutral(self.BEFORE, None)

    def test_a_parse_failure_is_behavioural(self):
        assert not cp.is_neutral(self.BEFORE, self.BEFORE + 'def (:\n')


class TestTheJunitMapping:
    FILES = frozenset({'tests/test_a.py', 'tests/sub/test_b.py'})

    def test_a_plain_test(self):
        assert cp.node_id('tests.test_a', 'test_x', self.FILES) == 'tests/test_a.py::test_x'

    def test_a_class_method_in_a_subdirectory(self):
        assert (
            cp.node_id('tests.sub.test_b.TestK', 'test_m', self.FILES)
            == 'tests/sub/test_b.py::TestK::test_m'
        )

    def test_a_parametrised_id_keeps_the_space_in_its_bracket(self):
        assert (
            cp.node_id('tests.test_a', 'test_p[a b]', self.FILES) == 'tests/test_a.py::test_p[a b]'
        )

    def test_an_xdist_group_suffix_is_stripped_but_not_an_at_inside_the_bracket(self):
        assert cp.node_id('tests.test_a', 'test_g@grp one', self.FILES) == 'tests/test_a.py::test_g'
        assert (
            cp.node_id('tests.test_a', 'test_p[c@d]@grp', self.FILES)
            == 'tests/test_a.py::test_p[c@d]'
        )

    def test_a_collection_error_maps_to_its_file(self):
        xml = (
            '<testsuites><testsuite><testcase classname="" name="tests.sub.test_b">'
            '<error message="collection failure">x</error></testcase></testsuite></testsuites>'
        )
        assert cp.read_outcomes(xml, self.FILES) == [
            cp.Outcome('tests/sub/test_b.py', 'collection-error')
        ]


RECORDED_JUNIT = """<testsuites><testsuite>
<testcase classname="tests.test_a" name="test_pass"/>
<testcase classname="tests.test_a" name="test_fail"><failure message="m">x</failure></testcase>
<testcase classname="tests.test_a" name="test_err"><error message="m">x</error></testcase>
<testcase classname="tests.test_a" name="test_skip"><skipped type="pytest.skip" message="s"/></testcase>
<testcase classname="tests.test_a" name="test_xf"><skipped type="pytest.xfail" message="x"/></testcase>
<testcase classname="" name="tests.sub.test_b"><error message="collection failure">x</error></testcase>
</testsuite></testsuites>"""

COLLECTION = {
    'tests/test_a.py::test_pass',
    'tests/test_a.py::test_fail',
    'tests/test_a.py::test_err',
    'tests/test_a.py::test_skip',
    'tests/test_a.py::test_xf',
    'tests/sub/test_b.py::test_one',
    'tests/sub/test_b.py::test_two',
}


def _outcomes():
    return cp.read_outcomes(RECORDED_JUNIT, {'tests/test_a.py', 'tests/sub/test_b.py'})


class TestTheObservingSet:
    def test_failed_errored_skipped_and_uncollectable_are_taken_passed_and_xfail_are_not(self):
        assert cp.observing_tests(_outcomes(), COLLECTION) == {
            'tests/test_a.py::test_fail',
            'tests/test_a.py::test_err',
            'tests/test_a.py::test_skip',
            'tests/sub/test_b.py::test_one',
            'tests/sub/test_b.py::test_two',
        }

    def test_a_suite_that_passes_everything_observes_nothing(self):
        passing = [cp.Outcome(t, 'passed') for t in COLLECTION]
        assert cp.observing_tests(passing, COLLECTION) == set()

    def test_a_run_reports_on_a_file_that_failed_to_collect(self):
        only_error = [cp.Outcome('tests/sub/test_b.py', 'collection-error')]
        assert cp.reported_tests(only_error, COLLECTION) == {
            'tests/sub/test_b.py::test_one',
            'tests/sub/test_b.py::test_two',
        }

    def test_a_run_that_names_nothing_collected_reports_on_nothing(self):
        stray = [cp.Outcome('tests/elsewhere.py::test_x', 'failed')]
        assert cp.reported_tests(stray, COLLECTION) == set()

    def test_an_observing_id_the_collection_lacks_is_refused_by_name(self):
        with pytest.raises(cp.RefusedError, match=r'1 observing id.*tests/gone\.py::test_x'):
            cp.validate_observing(
                {'tests/gone.py::test_x', 'tests/test_a.py::test_pass'}, COLLECTION
            )
        cp.validate_observing({'tests/test_a.py::test_pass'}, COLLECTION)


class TestTheTipBaseline:
    def test_the_baseline_is_the_passes_and_the_skips_are_counted(self):
        outcomes = [
            cp.Outcome('tests/test_a.py::test_pass', 'passed'),
            cp.Outcome('tests/test_a.py::test_skip', 'skipped'),
        ]
        expected = {'tests/test_a.py::test_pass', 'tests/test_a.py::test_skip'}
        assert cp.tip_baseline(outcomes, expected) == ({'tests/test_a.py::test_pass'}, 1)

    def test_a_test_failing_at_the_tip_refuses_by_name(self):
        outcomes = [cp.Outcome('tests/test_a.py::test_fail', 'failed')]
        with pytest.raises(cp.RefusedError, match=r'tests/test_a.py::test_fail is failed'):
            cp.tip_baseline(outcomes, {'tests/test_a.py::test_fail'})

    def test_an_expected_test_absent_from_the_outcomes_is_the_instrument(self):
        outcomes = [cp.Outcome('tests/test_a.py::test_pass', 'passed')]
        with pytest.raises(cp.RefusedError, match=r'INSTRUMENT: tests/test_a.py::test_gone'):
            cp.tip_baseline(outcomes, {'tests/test_a.py::test_pass', 'tests/test_a.py::test_gone'})

    def test_a_file_that_does_not_collect_at_the_tip_refuses(self):
        outcomes = [cp.Outcome('tests/sub/test_b.py', 'collection-error')]
        with pytest.raises(cp.RefusedError, match=r'tests/sub/test_b.py does not collect'):
            cp.tip_baseline(outcomes, set())


class TestTheVerdict:
    BASELINE = frozenset({'tests/test_a.py::test_one', 'tests/test_a.py::test_two'})

    def _run(self, one='passed', two='passed', extra=()):
        outcomes = [
            cp.Outcome('tests/test_a.py::test_one', one),
            cp.Outcome('tests/test_a.py::test_two', two),
            *extra,
        ]
        return cp.hunk_verdict(outcomes, self.BASELINE)

    def test_a_failure_is_red_and_named(self):
        assert self._run(two='failed') == ('RED', 'tests/test_a.py::test_two failed')

    def test_a_skip_after_the_revert_is_red(self):
        assert self._run(one='skipped') == ('RED', 'tests/test_a.py::test_one skipped')

    def test_a_collection_error_is_red_and_names_the_file(self):
        extra = [cp.Outcome('tests/test_c.py', 'collection-error')]
        assert self._run(extra=extra) == ('RED', 'fails to collect: tests/test_c.py')

    def test_every_baseline_test_present_and_passed_is_green(self):
        assert self._run() == ('GREEN', '2 tests passed')

    def test_a_baseline_test_absent_from_the_outcomes_is_the_instrument(self):
        outcomes = [cp.Outcome('tests/test_a.py::test_one', 'passed')]
        assert cp.hunk_verdict(outcomes, self.BASELINE) == (
            'INSTRUMENT',
            'tests/test_a.py::test_two is absent from the outcomes',
        )

    def test_an_xfail_flip_is_not_a_change(self):
        assert self._run(one='xfail') == ('GREEN', '2 tests passed')

    def test_a_nameless_collection_error_is_never_named_as_the_red_test(self):
        extra = [cp.Outcome('', 'collection-error')]
        assert self._run(extra=extra) == ('RED', 'fails to collect: (no file named)')


# ------------------------------------------------------------ end to end


def _git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True)


def _commit(repo, message):
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', message)


def _write(repo, rel, text):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def _repo(tmp_path):
    """A repository whose tip fixes two functions, adds, deletes and renames a file.

    The test at the tip pins ``add`` and imports the new module, so the
    ``add`` hunk and the added file are observed; ``mul``, the deleted file
    and both halves of the rename are not. The docstring edit is neutral.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.invalid')
    _git(repo, 'config', 'user.name', 'confirm_pins test')
    _write(repo, 'pytest.ini', '[pytest]\naddopts = -q --dist loadgroup\npythonpath = .\n')
    _write(repo, '.gitignore', '__pycache__/\n.pytest_cache/\n')
    _write(repo, 'modules/__init__.py', '')
    _write(
        repo,
        'modules/calc.py',
        '''\
        """Old words."""


        def add(a, b):
            return a + b + 1


        def mul(a, b):
            return a * b + 1
        ''',
    )
    _write(repo, 'modules/old.py', 'def stale():\n    return 0\n')
    _write(repo, 'modules/moved_from.py', 'def carried():\n    return 2\n')
    _write(
        repo,
        'tests/test_calc.py',
        'from modules.calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 5\n',
    )
    _commit(repo, 'base')
    _write(
        repo,
        'modules/calc.py',
        '''\
        """New words."""


        def add(a, b):
            return a + b


        def mul(a, b):
            return a * b
        ''',
    )
    (repo / 'modules/old.py').unlink()
    (repo / 'modules/moved_from.py').rename(repo / 'modules/moved_to.py')
    _write(repo, 'modules/new.py', 'def fresh():\n    return 1\n')
    _write(
        repo,
        'tests/test_calc.py',
        """\
        from modules.calc import add
        from modules.new import fresh


        def test_add():
            assert add(2, 2) == 4


        def test_fresh():
            assert fresh() == 1
        """,
    )
    _commit(repo, 'tip')
    return repo


def _run_tool(repo, tmp_path, *extra):
    scratch = tmp_path / 'scratch'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GIT_', 'PYTEST_'))}
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'tools' / 'confirm_pins.py'),
            '--checkout',
            str(repo),
            '--base',
            'HEAD~1',
            '--scratch',
            str(scratch),
            '--workers',
            '2',
            *extra,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    report = (scratch / 'report.txt').read_text()
    return done.returncode, report, done


def _hunk_lines(report):
    """The report's per-hunk lines keyed by (file, hunk header)."""
    lines = {}
    for line in report.splitlines():
        if '@@' in line:
            _, path, header, rest = line.strip().split('  ', 3)
            lines[(path, header)] = rest
    return lines


def test_end_to_end_one_hunk_per_kind_through_git_and_pytest(tmp_path):
    repo = _repo(tmp_path)
    code, report, done = _run_tool(repo, tmp_path)
    lines = report.splitlines()
    assert code == cp.EXIT_UNPINNED, report + done.stderr
    assert 'production files: 5  hunks: 7' in lines
    hunks = _hunk_lines(report)
    assert len(hunks) == 7, report
    by_file = {path: rest for (path, _), rest in hunks.items()}
    calc = [rest for (path, _), rest in hunks.items() if path == 'modules/calc.py']
    assert any(rest.startswith('NEUTRAL  docstrings or comments only') for rest in calc)
    assert any(rest.startswith('RED  tests/test_calc.py::test_add failed') for rest in calc)
    assert any(rest.startswith('GREEN  2 tests passed') for rest in calc)
    assert by_file['modules/new.py'].startswith('RED  fails to collect: tests/test_calc.py')
    assert by_file['modules/old.py'].startswith('GREEN  2 tests passed')
    # A rename under --no-renames is a deleted file and an added file, one hunk each.
    assert by_file['modules/moved_from.py'].startswith('GREEN  2 tests passed')
    assert by_file['modules/moved_to.py'].startswith('GREEN  2 tests passed')
    assert 'RED 2  GREEN 4  NEUTRAL 1  (behavioural 6)' in lines
    assert lines[-1] == 'result: UNPINNED (exit 1)'
    # The checkout was never touched, and the scratch worktree is gone.
    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=repo, capture_output=True, text=True, check=True
    )
    assert status.stdout == ''
    assert not (tmp_path / 'scratch' / 'worktree').exists()


def test_end_to_end_a_pre_existing_red_refuses_with_no_hunk_lines(tmp_path):
    repo = _repo(tmp_path)
    _write(
        repo,
        'modules/calc.py',
        '''\
        """New words."""


        def add(a, b):
            return a + b + 2


        def mul(a, b):
            return a * b
        ''',
    )
    _commit(repo, 'break add at the tip')
    code, report, _ = _run_tool(repo, tmp_path, '--base', 'HEAD~2')
    assert code == cp.EXIT_REFUSED, report
    assert 'refused: tests/test_calc.py::test_add is failed at the tip' in report
    assert not any('@@' in line for line in report.splitlines())


def test_end_to_end_a_range_no_test_observes_is_a_pass(tmp_path):
    repo = _repo(tmp_path)
    _write(
        repo,
        'modules/calc.py',
        '''\
        """New words."""


        def add(a, b):
            total = a + b
            return total


        def mul(a, b):
            return a * b
        ''',
    )
    _commit(repo, 'refactor add')
    code, report, _ = _run_tool(repo, tmp_path, '--base', 'HEAD~1')
    lines = report.splitlines()
    assert code == cp.EXIT_PINNED, report
    assert 'observing tests: 0' in report
    assert any('NOT-OBSERVED  no test observes the range' in line for line in lines)
    assert lines[-1] == 'result: PASS, no test observes the range (exit 0)'
