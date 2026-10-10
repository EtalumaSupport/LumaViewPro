"""The Windows build's venv is the app's requirements plus the packer, nothing else.

scripts/appBuild/build.ps1 installs requirements-build.txt into the venv
PyInstaller packs from. Anything installed there can be collected into the
installer through a library's optional import (scipy and scikit-image import
pytest inside a function), so the build file adds exactly PyInstaller and its
hooks to requirements.txt, each pinned, and is the one file that names them.
"""

from packaging.requirements import Requirement

from tests.ast_seams import REPO_ROOT

PACKER = {'pyinstaller', 'pyinstaller-hooks-contrib'}


def _requirements(name):
    """Every requirement a file installs, following its -r includes."""
    found = []
    for raw in REPO_ROOT.joinpath(name).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('-r '):
            found += _requirements(line[3:].strip())
            continue
        assert not line.startswith('-'), f'{name}: an option this reader does not follow: {line}'
        found.append(Requirement(line))
    return found


def test_the_build_file_adds_exactly_the_pinned_packer_to_the_app():
    app = {r.name.lower() for r in _requirements('requirements.txt')}
    added = [r for r in _requirements('requirements-build.txt') if r.name.lower() not in app]
    assert {r.name.lower() for r in added} == PACKER
    for requirement in added:
        assert len(requirement.specifier) == 1, requirement
        assert next(iter(requirement.specifier)).operator == '==', requirement


def test_no_other_requirements_file_names_the_packer():
    for path in sorted(REPO_ROOT.glob('requirements*.txt')):
        if path.name == 'requirements-build.txt':
            continue
        own = {
            Requirement(line).name.lower()
            for raw in path.read_text().splitlines()
            if (line := raw.split('#', 1)[0].strip()) and not line.startswith('-')
        }
        assert not own & PACKER, path.name
