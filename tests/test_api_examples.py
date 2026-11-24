# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The api_examples do what they say AND they go through the Session.

Each example is a task a reader copies: its test runs it and asserts what
the task leaves -- the file, the record, the outcome -- and runs its REST
twin, ``<name>.sh``, under bash against a simulated server and asserts the
same.
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
heavy-dep mocks serve the imports and the installation each test runs from
holds the fast tier's settings; standalone, the real installed deps and the
user's settings do. The subprocess witness proves the standalone form still works
outside the suite's mock environment; one witness suffices because the
import environment is the only thing that differs between the two forms.

Per-example wall time is budgeted (each builds a full simulate-mode
scope; see PERFORMANCE_BUDGETS.md `api_example_runtime_s`) -- the
budget is enforced by review of suite timings, not a per-test timer,
so a slow CI box cannot flake the suite on wall clock.
"""

import ast
import inspect
import itertools
import json
import os
import pathlib
import runpy
import shutil
import socket
import subprocess
import sys
import threading

import pytest
from tests.ast_seams import REPO_ROOT, tracked_files
from tests.case_walk import failing_cases
from tests.installation_fixtures import an_installation_holding
from tests.settings_fixtures import complete_settings

EXAMPLES_DIR = REPO_ROOT / 'docs' / 'api_examples'
# Derived from the directory, never hand-listed: a roster has to be
# remembered, and the example someone forgets to add is exactly the one
# nobody checked. A new file under api_examples/ is covered the day it
# lands.
EXAMPLES = sorted(
    pathlib.PurePath(rel).stem for rel in tracked_files(('docs/api_examples',), '*.py')
)

_SENTINEL = object()


@pytest.fixture(autouse=True)
def prepared_settings(tmp_path, monkeypatch):
    """The settings an example reads, for the test's length.

    An example reads ``load_user_settings('.')``, the installation it is run
    from, and the checkout's own ``data/current.json`` is gitignored and
    per-machine. So each test runs from an installation of its own holding
    them: the shipped template on the fast tier, the live folder in the
    test's own directory. A shipped script carries no test-only parameter,
    and what it saves is the test's to read.

    They are a new install's, as the reader's first run is: nobody has told
    the scope which objective is mounted, so each example, and each twin
    through ``served``, answers that itself or is refused.
    """
    settings = complete_settings(live_folder=str(tmp_path / 'live'), objective_confirmed=None)
    monkeypatch.chdir(an_installation_holding(tmp_path / 'run_from', settings))
    return settings


def _run_example(name):
    """What the example's ``main()`` returns, run as its ``__main__`` would run it."""
    return runpy.run_path(str(EXAMPLES_DIR / f'{name}.py'))['main']()


@pytest.fixture
def served(prepared_settings, tmp_path):
    """Runs an example's REST twin under bash on a simulated server; answers its client and its folder.

    The server is the one the README starts, ``python -m lumascope.rest
    --simulate --source-path <a copy of data/>``: a session brought up as that
    command brings it up, on the test's installation -- the shipped data and a
    new install's settings -- its startup motion run, served by the one
    builder on a port the system found free. The twin runs as a reader runs
    it, ``bash <twin>`` in a folder of its own with ``LVP_PORT`` naming the
    port, and exits 0: each of its curls ends the script on an answer it did
    not expect, and each refusal it shows on purpose is checked to be that
    refusal. What it downloads is in the folder answered.
    """
    import httpx

    from _lumascope.modules.scope_session import ScopeSession
    from _lumascope.rest.server import build_server

    missing = [tool for tool in ('bash', 'curl', 'jq') if shutil.which(tool) is None]
    if missing:
        pytest.skip(f'the REST twins run under bash with curl and jq; not installed: {missing}')
    # The installation prepared_settings made and runs the test from.
    installation = pathlib.Path.cwd()
    with socket.create_server(('127.0.0.1', 0)) as probe:
        port = probe.getsockname()[1]
    settings_file = ScopeSession.user_settings_file(installation)
    current = json.loads(pathlib.Path(settings_file).read_text(encoding='utf-8'))
    current['rest_api'] = {
        'enabled': True,
        'address': '127.0.0.1',
        'port': port,
        'allowed_clients': [],
    }
    pathlib.Path(settings_file).write_text(json.dumps(current), encoding='utf-8')
    session = ScopeSession.create(
        ScopeSession.load_user_settings(str(installation)),
        source_path=str(installation),
        simulate=True,
        warn_pre_release=False,
        settings_file=settings_file,
    )
    server = build_server(session, shut_down=session.shutdown)
    thread = threading.Thread(target=server.run)
    try:
        # The startup motion first, as the command runs it before it serves.
        session.run_application_session()
        thread.start()
        while not server.serving:
            assert thread.is_alive(), 'the server ended before it served'
            thread.join(0.01)
        folder = tmp_path / 'twin'
        folder.mkdir()

        def run_twin(name):
            path = EXAMPLES_DIR / f'{name}.sh'
            ran = subprocess.run(
                ['bash', str(path)],
                cwd=folder,
                env={**os.environ, 'LVP_PORT': str(port)},
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert ran.returncode == 0, (
                f'{path.name} exited {ran.returncode}: an answer it did not expect.\n'
                f'stdout tail:\n{ran.stdout[-2000:]}\nstderr tail:\n{ran.stderr[-2000:]}'
            )
            return client, folder

        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=60) as client:
            yield run_twin
    finally:
        if not server.close():
            # It stopped before it served, so the session is the test's to close.
            session.shutdown()
        if thread.ident is not None:
            thread.join(60)
            assert not thread.is_alive()


def _jobs_of(client, member):
    """The served session's jobs of the member named ``member``, oldest first."""
    return [
        job
        for job in reversed(client.get('/api/v1/jobs').json())
        if job['member'].rsplit('/', 1)[-1] == member
    ]


def _files_job(client, run_job):
    """The files job of ``run_job``, read once it has ended."""
    return client.get(f'/api/v1/jobs/{run_job["files"]["id"]}').json()


def test_every_example_has_its_twin_and_its_outcome_tests():
    """An example and its REST twin are each run by a test of what it leaves.

    A twin's test runs it under bash through ``served``, which fails it on
    any exit but 0, so every twin is run.
    """
    tests = [t for t in globals() if t.startswith('test_')]
    for name in EXAMPLES:
        assert (EXAMPLES_DIR / f'{name}.sh').is_file(), f'{name}.py has no REST twin {name}.sh'
        over_rest = f'test_{name}_over_rest_'
        assert any(t.startswith(f'test_{name}_') and not t.startswith(over_rest) for t in tests), (
            f'{name}.py has no test_{name}_... asserting what it leaves'
        )
        assert any(t.startswith(over_rest) for t in tests), (
            f'{name}.sh has no {over_rest}... running it and asserting what it leaves'
        )


def _full_scale(bits):
    return (1 << bits) - 1


def test_first_still_saves_a_lit_unsaturated_still_and_its_record():
    from _lumascope.modules.image_utils import read_file_record

    still = _run_example('first_still')
    assert (still.channel, still.well_label) == ('BF', 'B2')
    assert still.record.held_channel == 'BF'
    assert still.record.dark is False
    assert still.statistics.maximum < _full_scale(still.significant_bits)
    record = read_file_record(still.file)
    assert (record['channel'], record['illumination_ma'], record['dark']) == ('BF', 5.0, False)


def test_first_still_over_rest_saves_the_same_still(served):
    from _lumascope.modules.image_utils import read_file_record

    client, folder = served('first_still')
    [job] = _jobs_of(client, 'start_save_still')
    still = job['result'][0]
    assert (still['channel'], still['well_label']) == ('BF', 'B2')
    assert still['record']['dark'] is False
    assert still['statistics']['maximum'] < _full_scale(still['significant_bits'])
    record = read_file_record(folder / 'still.tiff')
    assert (record['channel'], record['illumination_ma'], record['dark']) == ('BF', 5.0, False)


@pytest.mark.slow
def test_first_still_standalone_subprocess(tmp_path):
    """The one standalone witness: the example runs in its own interpreter,
    outside the suite's mock environment, against the real installed deps.

    It runs from a fresh installation of its own, the shipped ``data/`` and
    no user ``current.json``, as a reader runs it from theirs: the still it
    saves lands in that installation's live folder, never in the one a
    developer's ``current.json`` names.
    """
    installation = tmp_path / 'installation'
    shutil.copytree(
        EXAMPLES_DIR.parents[1] / 'data',
        installation / 'data',
        ignore=shutil.ignore_patterns('current.json'),
    )
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / 'first_still.py')],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=installation,
    )
    assert proc.returncode == 0, f'stderr tail: {proc.stderr[-2000:]}'
    assert len(list((installation / 'capture').rglob('*.tiff'))) == 1


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
            s.record.held_channel,
            list(s.record.illumination_ma.values()),
            s.record.exposure_ms,
            s.record.gain_db,
        )
        for s in stills
    ] == list(_CHANNEL_STILLS)
    for s in stills:
        assert _lit_and_unsaturated(s.record.dark, s.statistics.maximum, s.significant_bits), s


def test_channel_settings_over_rest_takes_the_same_stills(served):
    client, _ = served('channel_settings')
    stills = [job['result'][0] for job in _jobs_of(client, 'start_save_still')]
    assert [
        (
            s['channel'],
            s['record']['held_channel'],
            list(s['record']['illumination_ma'].values()),
            s['record']['exposure_ms'],
            s['record']['gain_db'],
        )
        for s in stills
    ] == list(_CHANNEL_STILLS)
    for s in stills:
        assert _lit_and_unsaturated(
            s['record']['dark'], s['statistics']['maximum'], s['significant_bits']
        ), s


def test_scan_completes_and_keeps_an_image_of_each_well():
    from _lumascope.modules.image_utils import read_file_record
    from _lumascope.modules.run_outcome import FilesOutcome, RunStatus

    outcome, files, run_dir = _run_example('scan')
    assert outcome.status is RunStatus.COMPLETED, outcome.message
    assert (outcome.captures.asked, outcome.captures.captured) == (2, 2)
    assert (files.outcome, files.written) == (FilesOutcome.WRITTEN, 2)
    records = [read_file_record(p) for p in sorted(run_dir.glob('*.tiff'))]
    assert [(r['channel'], r['plate']['well_label'], r['dark']) for r in records] == [
        ('BF', 'B2', False),
        ('BF', 'B3', False),
    ]


def test_scan_over_rest_completes_and_keeps_its_images(served):
    from _lumascope.modules.image_utils import read_file_record

    client, folder = served('scan')
    [job] = _jobs_of(client, 'start_single_scan')
    assert job['result']['status'] == 'completed', job
    assert (job['result']['captures']['asked'], job['result']['captures']['captured']) == (2, 2)
    files = _files_job(client, job)
    assert files['status'] == 'completed', files
    assert (files['result']['outcome'], files['result']['written']) == ('written', 2)
    record = read_file_record(folder / 'first.tiff')
    assert (record['channel'], record['plate']['well_label'], record['dark']) == (
        'BF',
        'B2',
        False,
    )


def _slice_steps_um(paths):
    from _lumascope.modules.image_utils import read_file_record

    z = sorted(read_file_record(p)['z_pos_um'] for p in paths)
    return [round(b - a, 1) for a, b in itertools.pairwise(z)]


def test_z_stack_saved_writes_a_slice_per_step_of_its_range():
    from _lumascope.modules.run_outcome import FilesOutcome, RunStatus

    outcome, files, slices = _run_example('z_stack_saved')
    assert outcome.status is RunStatus.COMPLETED, outcome.message
    assert (files.outcome, files.written, len(slices)) == (FilesOutcome.WRITTEN, 5, 5)
    assert _slice_steps_um(slices) == [5.0] * 4


def test_z_stack_saved_over_rest_writes_the_same_slices(served, prepared_settings):
    client, _ = served('z_stack_saved')
    [job] = _jobs_of(client, 'start_zstack')
    assert job['result']['status'] == 'completed', job
    files = _files_job(client, job)
    assert files['status'] == 'completed', files
    assert (files['result']['outcome'], files['result']['written']) == ('written', 5)
    run_dir = pathlib.Path(prepared_settings['live_folder']) / job['progress']['run_dir']['name']
    assert _slice_steps_um(sorted(run_dir.glob('*.tiff'))) == [5.0] * 4


def test_stop_a_run_ends_it_aborted_after_its_first_scan_with_its_images_written():
    from _lumascope.modules.run_outcome import FilesOutcome, RunStatus

    outcome, files = _run_example('stop_a_run')
    assert (outcome.status, outcome.reason) == (RunStatus.ABORTED, 'stopped')
    assert outcome.captures.captured == 3 < outcome.captures.asked
    assert (files.outcome, files.written) == (FilesOutcome.WRITTEN, 3)


def test_stop_a_run_over_rest_ends_it_aborted_after_its_first_scan_with_its_images_written(
    served,
):
    client, _ = served('stop_a_run')
    [job] = _jobs_of(client, 'start_protocol')
    assert job['status'] == 'completed', job
    outcome = job['result']
    assert (outcome['status'], outcome['reason']) == ('aborted', 'stopped')
    assert outcome['captures']['captured'] == 3 < outcome['captures']['asked']
    files = _files_job(client, job)
    assert (files['result']['outcome'], files['result']['written']) == ('written', 3)


def test_refusals_and_faults_meets_one_refusal_of_each_cause():
    from _lumascope.modules.exceptions import RefusalCause

    refusals = _run_example('refusals_and_faults')
    assert [(r.reason, r.cause) for r in refusals] == [
        ('position_unknown', RefusalCause.STATE),
        ('position_out_of_range', RefusalCause.REQUEST),
        ('not_connected', RefusalCause.RESTART),
    ]


def test_refusals_and_faults_over_rest_answers_each_problem(served):
    client, _ = served('refusals_and_faults')
    assert _jobs_of(client, 'start_move_z') == []
    [job] = _jobs_of(client, 'start_save_still')
    assert job['status'] == 'failed', job
    assert (job['error']['reason'], job['error']['status']) == ('not_connected', 503)


def test_control_session_acts_in_it_and_refuses_outside_it_and_after_it():
    still, held, ended = _run_example('control_session')
    assert still.file.is_file() and still.channel == 'BF'
    assert (held.reason, ended.reason) == ('control_session_held', 'control_session_ended')


def test_control_session_over_rest_acts_in_it_and_refuses_outside_it_and_after_it(served):
    client, _ = served('control_session')
    assert _jobs_of(client, 'start_move_z') == []
    [job] = _jobs_of(client, 'start_save_still')
    assert job['status'] == 'completed', job
    assert job['result'][0]['channel'] == 'BF'
    assert client.get('/api/v1/control').json() is None


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
        from _lumascope.modules.scope_session import ScopeSession

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
    from _lumascope.modules.scope_session import ScopeSession
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
        'from _lumascope.modules.image_save import save_image\n'
        "z = scope.motion.get_current_position('Z')\n"
        'scope.wellplate_loader.get_plate_list()\n'
    )
    reported = '\n'.join(
        _unmarked_calls('planted', planted, live_roots) + project_name_failures('planted', planted)
    )
    for expected in (
        'planted:1  _lumascope.modules.image_save.save_image  -> an unmarked function',
        'planted:2  scope.motion.get_current_position  -> MotionAPI.get_current_position',
        'planted:3  scope.wellplate_loader.get_plate_list  -> Lumascope.wellplate_loader',
    ):
        assert expected in reported, reported
