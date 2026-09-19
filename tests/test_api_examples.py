# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The api_examples are executable AND they go through the Session.

Two properties, because executability alone is what let the corpus drift.
Every example ran green here for months while three of the four built a
bare ``Lumascope(simulate=True)`` and never touched ``ScopeSession`` --
the layer an L2 caller is required to come through. A smoke test that
only asks "did it run" cannot see that, so the second property asks what
the example DEMONSTRATES, by resolving the call against the live class
rather than matching text. A ban on a spelling would be fooled by an
alias import and would mis-fire on the first real-hardware example.

Each example is the SAME code path both ways -- the examples ARE the
smoke corpus (a hand-rolled per-file import environment was the defect
this replaces). In-suite, the conftest's heavy-dep mocks serve the
imports; standalone, the real installed deps do. The subprocess witness
proves the standalone form still works outside the suite's mock
environment; one witness suffices because the import environment is the
only thing that differs between the two forms.

Per-example wall time is budgeted (each builds a full simulate-mode
scope; see PERFORMANCE_BUDGETS.md `api_example_runtime_s`) -- the
budget is enforced by review of suite timings, not a per-test timer,
so a slow CI box cannot flake the suite on wall clock.
"""

import ast
import inspect
import json
import pathlib
import runpy
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'api_examples'
# Derived from the directory, never hand-listed: a roster has to be
# remembered, and the example someone forgets to add is exactly the one
# nobody checked. A new file under api_examples/ is covered the day it
# lands.
EXAMPLES = sorted(path.stem for path in EXAMPLES_DIR.glob('*.py'))

_SENTINEL = object()


@pytest.fixture(autouse=True)
def _restore_settings_globals():
    """An example may load settings the documented way, which publishes
    them as process globals; the suite's later simulated scopes read the
    same globals for their model, so restore them after each run.

    The globals are also PINNED to the shipped template for the duration.
    ``create_headless()`` resolves settings from the process globals when
    it is passed none, and the unpinned source underneath them is
    ``data/current.json`` -- gitignored and per-machine. Without this pin
    an example's printed frame geometry, and whether it comes up at all,
    varies by whose checkout runs it. A customer SHOULD see their own
    configuration; the smoke corpus should not.

    ``protocol_execution.py`` reads the settings file itself rather than
    taking the globals, so the pin does not reach it -- pre-existing, and
    the reason that example is not disk-independent here.
    """
    module = sys.modules.get('modules.settings_init')
    saved = {
        name: getattr(module, name, None)
        for name in ('settings', 'rejected_current_json')
        if module is not None
    }
    if module is not None:
        with (EXAMPLES_DIR.parents[1] / 'data' / 'settings.json').open() as handle:
            module.settings = json.load(handle)
    yield
    for name, value in saved.items():
        setattr(module, name, value)


@pytest.mark.parametrize('name', EXAMPLES)
def test_example_runs_in_suite(name):
    """Run the example's __main__ path in-process under the suite mocks."""
    runpy.run_path(str(EXAMPLES_DIR / f'{name}.py'), run_name='__main__')


def test_basic_capture_standalone_subprocess():
    """The one standalone witness: the example runs in its own interpreter,
    outside the suite's mock environment, against the real installed deps."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / 'basic_capture.py')],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f'stderr tail: {proc.stderr[-2000:]}'


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


@pytest.mark.parametrize('name', EXAMPLES)
def test_every_example_obtains_a_scope_session(name):
    """The layer rule: an L2 caller reaches the instrument through the
    Session. An example that never names ScopeSession is teaching a
    reader to skip it."""
    calls = _scope_session_calls(EXAMPLES_DIR / f'{name}.py')
    assert calls, (
        f'{name}.py never calls a ScopeSession member, so it demonstrates '
        'reaching the scope without the Session layer an L2 caller is '
        'required to come through. Build the session from the supported '
        'factory and take the composition root from session.scope.'
    )


@pytest.mark.parametrize('name', EXAMPLES)
def test_every_session_call_resolves_and_binds(name):
    """The member an example names exists on the live class and accepts the
    arguments the example passes it -- resolution, not spelling, so a
    renamed factory is caught here instead of in a customer's terminal."""
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
