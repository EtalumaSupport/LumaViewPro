# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The api_examples do what they say AND they go through the Session.

Each example is a task a reader copies: its test runs it and asserts what
the task leaves -- the file, the record, the outcome -- and runs its REST
twin, ``<name>.sh``, line by line on a simulated server and asserts the same.
An example that only had to run raised nothing while it saved a saturated
frame, so ran-without-raising is not a test of one.

And it goes through the Session. Every example ran green here for months
while three of the four built a
bare ``Lumascope(simulate=True)`` and never touched ``ScopeSession`` --
the layer an L2 caller is required to come through. A smoke test that
only asks "did it run" cannot see that, so the second property asks what
the example DEMONSTRATES, by resolving the call against the live class
rather than matching text. A ban on a spelling would be fooled by an
alias import and would mis-fire on the first real-hardware example.

Each example is the SAME code path both ways. In-suite, the conftest's
heavy-dep mocks serve the imports and the settings the process prepared
are the fast tier's; standalone, the real installed deps and the user's
settings do. The subprocess witness proves the standalone form still works
outside the suite's mock environment; one witness suffices because the
import environment is the only thing that differs between the two forms.

Per-example wall time is budgeted (each builds a full simulate-mode
scope; see PERFORMANCE_BUDGETS.md `api_example_runtime_s`) -- the
budget is enforced by review of suite timings, not a per-test timer,
so a slow CI box cannot flake the suite on wall clock.
"""

import ast
import inspect
import pathlib
import runpy
import subprocess
import sys

import pytest
from tests.case_walk import failing_cases
from tests.guards.test_api_doc_guard import ShellRun, curl_failures
from tests.settings_fixtures import complete_settings

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'api_examples'
# Derived from the directory, never hand-listed: a roster has to be
# remembered, and the example someone forgets to add is exactly the one
# nobody checked. A new file under api_examples/ is covered the day it
# lands.
EXAMPLES = sorted(path.stem for path in EXAMPLES_DIR.glob('*.py'))

# The examples written before each had a task, a twin and an outcome test:
# they are only run. Each leaves this set when the example for its task
# replaces it, and the set and test_example_runs_in_suite go with the last.
_NOT_YET_REWRITTEN = frozenset({'protocol_execution', 'z_stack'})

_SENTINEL = object()


@pytest.fixture(autouse=True)
def prepared_settings(tmp_path):
    """The settings this process prepared, as an example reads them, for the test's length.

    ``load_user_settings()`` answers the settings a process has prepared, as
    the GUI's own scripts read them, and only an unprepared process reads
    ``data/current.json`` -- gitignored and per-machine. So the test prepares
    them: the shipped template on the fast tier, the live folder in the
    test's own directory. A shipped script carries no test-only parameter,
    and what it saves is the test's to read. The suite's later simulated
    scopes read the same globals for their model, so they are restored.
    """
    from modules import settings_init

    saved = {name: getattr(settings_init, name) for name in ('settings', 'rejected_current_json')}
    settings_init.settings = complete_settings(live_folder=str(tmp_path / 'live'))
    yield settings_init.settings
    for name, value in saved.items():
        setattr(settings_init, name, value)


def _run_example(name):
    """What the example's ``main()`` returns, run as its ``__main__`` would run it."""
    return runpy.run_path(str(EXAMPLES_DIR / f'{name}.py'))['main']()


@pytest.fixture
def served(prepared_settings):
    """Runs an example's REST twin on a simulated server; answers its client and its run.

    The server is the one ``python -m rest --simulate`` serves: a session
    brought up from the prepared settings, its startup motion run.
    """
    from fastapi.testclient import TestClient

    from modules.scope_session import ScopeSession
    from rest.app import build_app

    session = ScopeSession.create(prepared_settings.copy(), simulate=True)
    session.run_application_session()
    with TestClient(build_app(session)) as client:

        def run_twin(name):
            path = EXAMPLES_DIR / f'{name}.sh'
            run = ShellRun()
            lines = enumerate(path.read_text().splitlines(), 1)
            failures = curl_failures(lines, path.name, client, run)
            assert not failures, (
                f'{path.name} has a curl line that does not answer as its comment states. '
                'Correct the line or its comment.\n' + '\n'.join(failures)
            )
            return client, run

        yield run_twin
    session.shutdown()


@pytest.mark.slow
# parametrize-justified: each example runs its own __main__ in-process
@pytest.mark.parametrize('name', sorted(_NOT_YET_REWRITTEN))
def test_example_runs_in_suite(name):
    """Run the example's __main__ path in-process under the suite mocks."""
    runpy.run_path(str(EXAMPLES_DIR / f'{name}.py'), run_name='__main__')


def test_every_example_has_its_twin_and_its_outcome_test():
    """An example is run by a test of what it leaves, and has its REST twin."""
    tested = {name for name in EXAMPLES if any(t.startswith(f'test_{name}_') for t in globals())}
    for name in sorted(set(EXAMPLES) - _NOT_YET_REWRITTEN):
        assert (EXAMPLES_DIR / f'{name}.sh').is_file(), f'{name}.py has no REST twin {name}.sh'
        assert name in tested, f'{name}.py has no test_{name}_... asserting what it leaves'


def _full_scale(bits):
    return (1 << bits) - 1


def test_first_still_saves_a_lit_unsaturated_still_and_its_record():
    from modules.image_utils import read_file_record

    still = _run_example('first_still')
    assert (still.channel, still.well_label) == ('BF', 'B2')
    assert still.frame.held_channel == 'BF'
    assert still.frame.dark is False
    assert still.statistics.maximum < _full_scale(still.significant_bits)
    record = read_file_record(still.file)
    assert (record['channel'], record['illumination_ma'], record['dark']) == ('BF', 5.0, False)


def test_first_still_over_rest_saves_the_same_still(served, tmp_path):
    from modules.image_utils import read_file_record

    client, run = served('first_still')
    still = client.get(f'/api/v1/jobs/{run.variables["JOB"]}').json()['result'][0]
    assert (still['channel'], still['well_label']) == ('BF', 'B2')
    assert still['frame']['dark'] is False
    assert still['statistics']['maximum'] < _full_scale(still['significant_bits'])
    downloaded = tmp_path / 'still.tiff'
    downloaded.write_bytes(run.written['still.tiff'])
    record = read_file_record(downloaded)
    assert (record['channel'], record['illumination_ma'], record['dark']) == ('BF', 5.0, False)


def test_first_still_standalone_subprocess():
    """The one standalone witness: the example runs in its own interpreter,
    outside the suite's mock environment, against the real installed deps."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / 'first_still.py')],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f'stderr tail: {proc.stderr[-2000:]}'


# What channel_settings asks of each still, in both its forms: the channel
# it is named for, the channel the camera held, the current of each LED lit,
# the exposure and the gain.
_CHANNEL_STILLS = (
    ('BF', 'BF', [5.0], 2.0, 1.0),
    ('Ex488', 'Ex488', [300.0], 100.0, 10.0),
    ('BF', None, [5.0], 1.5, 3.0),
)


def _lit_and_unsaturated(dark, maximum, bits):
    return dark is False and maximum < _full_scale(bits)


def test_channel_settings_takes_each_still_with_the_settings_it_wrote():
    stored, raw = _run_example('channel_settings')
    stills = [stored['BF'], stored['Ex488'], raw]
    assert [
        (
            s.channel,
            s.frame.held_channel,
            list(s.frame.illumination_ma.values()),
            s.frame.exposure_ms,
            s.frame.gain_db,
        )
        for s in stills
    ] == list(_CHANNEL_STILLS)
    for s in stills:
        assert _lit_and_unsaturated(s.frame.dark, s.statistics.maximum, s.significant_bits), s


def test_channel_settings_over_rest_takes_the_same_stills(served):
    client, run = served('channel_settings')
    stills = [
        client.get(f'/api/v1/jobs/{run.variables[job]}').json()['result'][0]
        for job in ('BF_JOB', 'EX488_JOB', 'RAW_JOB')
    ]
    assert [
        (
            s['channel'],
            s['frame']['held_channel'],
            list(s['frame']['illumination_ma'].values()),
            s['frame']['exposure_ms'],
            s['frame']['gain_db'],
        )
        for s in stills
    ] == list(_CHANNEL_STILLS)
    for s in stills:
        assert _lit_and_unsaturated(
            s['frame']['dark'], s['statistics']['maximum'], s['significant_bits']
        ), s


def test_stop_a_run_ends_it_aborted_after_its_first_scan_with_its_images_written():
    from modules.run_outcome import FilesOutcome, RunStatus

    outcome, files = _run_example('stop_a_run')
    assert (outcome.status, outcome.reason) == (RunStatus.ABORTED, 'stopped')
    assert outcome.captures.captured == 3 < outcome.captures.asked
    assert (files.outcome, files.written) == (FilesOutcome.WRITTEN, 3)


def test_stop_a_run_over_rest_ends_its_job_aborted(served):
    client, run = served('stop_a_run')
    job = client.get(f'/api/v1/jobs/{run.variables["JOB"]}').json()
    assert job['status'] == 'completed', job
    outcome = job['result']
    assert (outcome['status'], outcome['reason']) == ('aborted', 'stopped')
    assert outcome['captures']['captured'] < outcome['captures']['asked']


def test_refusals_and_faults_meets_one_refusal_of_each_cause():
    from modules.exceptions import RefusalCause

    refusals = _run_example('refusals_and_faults')
    assert [(r.reason, r.cause) for r in refusals] == [
        ('position_unknown', RefusalCause.STATE),
        ('position_out_of_range', RefusalCause.REQUEST),
        ('not_connected', RefusalCause.RESTART),
    ]


def test_refusals_and_faults_over_rest_answers_each_problem(served):
    client, run = served('refusals_and_faults')
    job = client.get(f'/api/v1/jobs/{run.variables["JOB"]}').json()
    assert job['status'] == 'failed', job
    assert (job['error']['reason'], job['error']['status']) == ('not_connected', 503)


def _scope_session_calls(path):
    """Every ``ScopeSession.<member>(...)`` call an example makes, as
    (member, positional count, keyword names, line)."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != 'ScopeSession':
            continue
        kwnames = [kw.arg for kw in node.keywords if kw.arg is not None]
        found.append((func.attr, len(node.args), kwnames, node.lineno))
    return found


def test_the_derived_corpus_is_not_empty():
    """A roster derived from a glob that matches nothing reports every
    property green over zero files, which is the failure mode a derived
    roster is supposed to remove rather than hide."""
    assert EXAMPLES, f'no examples found under {EXAMPLES_DIR}'


def test_every_example_obtains_a_scope_session():
    """The layer rule: an L2 caller reaches the instrument through the
    Session. An example that never names ScopeSession is teaching a
    reader to skip it."""

    def check(name):
        calls = _scope_session_calls(EXAMPLES_DIR / f'{name}.py')
        assert calls, (
            f'{name}.py never calls a ScopeSession member, so it demonstrates '
            'reaching the scope without the Session layer an L2 caller is '
            'required to come through. Build the session from the supported '
            'factory and take the composition root from session.scope.'
        )

    failed = failing_cases(EXAMPLES, check)
    assert not failed, failed


def test_every_session_call_resolves_and_binds():
    """The member an example names exists on the live class and accepts the
    arguments the example passes it -- resolution, not spelling, so a
    renamed factory is caught here instead of in a customer's terminal."""

    def check(name):
        from modules.scope_session import ScopeSession

        failures = []
        for member, nargs, kwnames, lineno in _scope_session_calls(EXAMPLES_DIR / f'{name}.py'):
            target = getattr(ScopeSession, member, None)
            if target is None or not callable(target):
                failures.append(f'  {name}.py:{lineno}  ScopeSession.{member} does not resolve')
                continue
            try:
                inspect.signature(target).bind(
                    *(_SENTINEL,) * nargs, **dict.fromkeys(kwnames, _SENTINEL)
                )
            except TypeError as exc:
                failures.append(f'  {name}.py:{lineno}  ScopeSession.{member}(...) -> {exc}')

        assert not failures, (
            'A shipped example calls a ScopeSession member the live class will not '
            'accept. Correct the EXAMPLE to match the signature.\n' + '\n'.join(failures)
        )

    failed = failing_cases(EXAMPLES, check)
    assert not failed, failed


def _session_and_scope_chains(source):
    """``(receiver, chain, line)`` for each attribute chain read off ``session`` or ``scope``."""
    tree = ast.parse(source)
    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in inner:
            continue
        chain, value = [], node
        while isinstance(value, ast.Attribute):
            chain.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name) and value.id in ('session', 'scope'):
            yield value.id, tuple(reversed(chain)), node.lineno


@pytest.fixture(scope='module')
def live_roots(tmp_path_factory):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    yield {'session': session, 'scope': session.scope}
    session.shutdown()


def _unmarked_calls(label, source, roots):
    from tests.guards.test_api_doc_guard import _first_unmarked_step

    return [
        f'  {label}:{line}  {receiver}.{".".join(chain)}  -> {step}'
        for receiver, chain, line in _session_and_scope_chains(source)
        if (step := _first_unmarked_step(roots[receiver], receiver, chain))
    ]


def test_every_example_reads_only_api_off_the_session_and_scope(live_roots):
    """What an example reaches through the session or the scope is a marked member
    at every step, so an example cannot teach an internal reader to build on."""
    failures = [
        failure
        for name in EXAMPLES
        for failure in _unmarked_calls(
            f'{name}.py', (EXAMPLES_DIR / f'{name}.py').read_text(), live_roots
        )
    ]
    assert not failures, (
        'A shipped example reaches a member that is not API. Rewrite the EXAMPLE to the '
        'marked member.\n' + '\n'.join(failures)
    )


def test_every_project_name_an_example_imports_is_api():
    """A name an example imports from the project is API, and its calls bind."""
    from tests.guards.test_api_doc_guard import project_name_failures

    failures = [
        failure
        for name in EXAMPLES
        for failure in project_name_failures(
            f'{name}.py', (EXAMPLES_DIR / f'{name}.py').read_text()
        )
    ]
    assert not failures, (
        'A shipped example imports or calls a project name that is not API. Rewrite the '
        'EXAMPLE to the member that does the job.\n' + '\n'.join(failures)
    )


def test_a_planted_unmarked_read_and_import_are_reported(live_roots):
    from tests.guards.test_api_doc_guard import project_name_failures

    planted = (
        'from modules.image_save import save_image\n'
        "z = scope.motion.get_current_position('Z')\n"
        'scope.wellplate_loader.get_plate_list()\n'
    )
    reported = '\n'.join(
        _unmarked_calls('planted', planted, live_roots) + project_name_failures('planted', planted)
    )
    for expected in (
        'planted:1  modules.image_save.save_image  -> an unmarked function',
        'planted:2  scope.motion.get_current_position  -> MotionAPI.get_current_position',
        'planted:3  scope.wellplate_loader.get_plate_list  -> Lumascope.wellplate_loader',
    ):
        assert expected in reported, reported
