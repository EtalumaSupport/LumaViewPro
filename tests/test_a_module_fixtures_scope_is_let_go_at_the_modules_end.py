# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope a module-scoped fixture built is let go of when its module ends.

Every scope, session and simulated camera a test process builds is queued
for its stop in `tests.scope_fakes`, and each queued stop holds what it
stops. A test's teardown runs the stops queued after the test began; a
module fixture's scope is queued before that, and before the owner
(`_stop_what_a_module_built` in `tests/conftest.py`) nothing ran or dropped
it, so every module fixture's session stayed alive for the rest of the
worker's run. The claim is about two modules in order, which no test inside
one module can observe, so it runs a pytest of two modules of its own.
"""

import subprocess
import sys

from tests.ast_seams import REPO_ROOT

_BUILDS = """
import weakref

import pytest

from tests.scope_fakes import build_scope

BUILT = []


@pytest.fixture(scope='module')
def scope():
    # Left for the module's end to stop, as a fixture that forgets to is.
    built = build_scope(simulate=True)
    BUILT.append(weakref.ref(built))
    return built


def test_uses_the_module_scope(scope):
    assert scope is not None
"""

_LATER = """
import gc

import test_a_module_builds_a_scope as earlier

from tests import scope_fakes


def test_nothing_the_earlier_module_built_is_still_held():
    gc.collect()
    assert scope_fakes._TEARDOWNS == [], scope_fakes._TEARDOWNS
    assert earlier.BUILT[0]() is None, 'the earlier module fixture\\'s scope is still alive'
"""


def test_a_later_module_finds_nothing_an_earlier_module_fixture_built(tmp_path):
    (tmp_path / 'test_a_module_builds_a_scope.py').write_text(_BUILDS)
    (tmp_path / 'test_b_a_later_module.py').write_text(_LATER)
    run = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-n',
            '0',
            '-p',
            'no:cacheprovider',
            '-p',
            'tests.conftest',
            '--rootdir',
            str(tmp_path),
            str(tmp_path / 'test_a_module_builds_a_scope.py'),
            str(tmp_path / 'test_b_a_later_module.py'),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert '2 passed' in run.stdout, run.stdout
