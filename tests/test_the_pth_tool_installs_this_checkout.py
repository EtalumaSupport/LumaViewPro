# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scripts/install_lumascope_pth.py`` is the user's way in: one ``.pth`` naming this checkout's ``src/``.

Run with a fresh virtual environment's Python, the API imports from a user's
own folder, with no ``sys.path`` line, and only the API: the user's own
``tests`` and ``lib`` stay theirs, and the checkout's ``ui``, ``tools`` and
the old top-level folders are not importable. A site-packages the user
cannot write is refused, never swapped for the user site; a ``.pth`` naming
another checkout is replaced, and both are named.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import venv

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / 'scripts' / 'install_lumascope_pth.py'


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    """A fresh virtual environment: its Python and its site-packages."""
    root = tmp_path_factory.mktemp('venv')
    venv.create(root, with_pip=False)
    python = str(root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
    site = subprocess.run(
        [python, '-c', "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return python, pathlib.Path(site)


def _run(python, *args, cwd=None):
    # The suite exports this checkout's src/ in PYTHONPATH for its children;
    # a user's interpreter has no such line.
    child_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    return subprocess.run(
        [python, *args], capture_output=True, text=True, cwd=cwd, env=child_env, timeout=60
    )


@pytest.mark.slow
def test_a_user_folder_imports_the_api_and_keeps_its_own_packages(env, tmp_path):
    python, site = env
    (site / 'lumascope.pth').unlink(missing_ok=True)
    installed = _run(python, str(TOOL))
    assert installed.returncode == 0, installed.stdout + installed.stderr
    assert (site / 'lumascope.pth').read_text(encoding='utf-8').strip() == str(REPO / 'src')

    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / '__init__.py').write_text('')
    (tmp_path / 'tests' / 'helpers.py').write_text("WHOSE = 'theirs'\n")
    (tmp_path / 'lib').mkdir()
    (tmp_path / 'lib' / '__init__.py').write_text("WHOSE = 'theirs'\n")
    probe = (
        'import importlib.util, json, _lumascope, lib, tests.helpers\n'
        'print(json.dumps([_lumascope.__file__, lib.WHOSE, tests.helpers.WHOSE,\n'
        '    {n: importlib.util.find_spec(n) is not None\n'
        "     for n in ('ui', 'tools', 'modules', 'drivers', 'rest', 'lvp_logger')}]))\n"
    )
    done = _run(python, '-c', probe, cwd=str(tmp_path))

    assert done.returncode == 0, done.stderr
    api, lib, helpers, found = json.loads(done.stdout)
    assert pathlib.Path(api).resolve().parent == (REPO / 'src' / '_lumascope').resolve()
    assert (lib, helpers) == ('theirs', 'theirs')
    assert found == dict.fromkeys(
        ('ui', 'tools', 'modules', 'drivers', 'rest', 'lvp_logger'), False
    )


@pytest.mark.slow
def test_a_pth_naming_another_checkout_is_replaced_and_both_are_named(env):
    python, site = env
    elsewhere = '/elsewhere/LumaViewPro/src'
    (site / 'lumascope.pth').write_text(f'{elsewhere}\n', encoding='utf-8')

    done = _run(python, str(TOOL))

    assert done.returncode == 0, done.stdout + done.stderr
    assert f'it named {elsewhere}; it names {REPO / "src"} now' in done.stdout
    assert (site / 'lumascope.pth').read_text(encoding='utf-8').strip() == str(REPO / 'src')


@pytest.mark.slow
@pytest.mark.skipif(
    os.name == 'nt' or os.geteuid() == 0, reason='a mode bit a user cannot write past'
)
def test_a_site_packages_the_user_cannot_write_is_refused_and_named(env):
    python, site = env
    (site / 'lumascope.pth').unlink(missing_ok=True)
    mode = site.stat().st_mode
    site.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        done = _run(python, str(TOOL))
    finally:
        site.chmod(mode)

    assert done.returncode == 1
    assert f'REFUSED: {site} cannot be written by this user' in done.stderr
    assert not (site / 'lumascope.pth').exists()
    user_site = _run(python, '-c', 'import site; print(site.getusersitepackages())').stdout.strip()
    assert not (pathlib.Path(user_site) / 'lumascope.pth').exists()
