# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""How the process was launched, and where its data lives, have one answer.

``path_utils.app_runtime`` names the run: source, a PyInstaller bundle the
installer did not install, or an installed build. ``get_source_root`` places
the data root from it, and every reader takes that root: the logger, the
diagnostics, the GUI's environment and the Session. A bundle run from
``dist`` was logged as a source run, and in a folder of links into a clone
(the simulator's scratch layout) the Session read the clone's data while the
GUI and the logger read the scratch folder's.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from modules import path_utils
from modules.exceptions import InstallationFileError
from modules.path_utils import INSTALLED_MARKER, AppRuntime

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def exe_folder(tmp_path, monkeypatch):
    """This process as a PyInstaller build whose exe is in a temp folder."""
    folder = tmp_path / 'LumaViewPro'
    folder.mkdir()
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(folder / 'LumaViewPro.exe'))
    return folder


def test_a_checkout_run_by_python_is_source():
    assert path_utils.app_runtime() is AppRuntime.SOURCE
    assert not path_utils.app_runtime().frozen
    assert path_utils.launch_root() == REPO
    assert path_utils.get_source_root() == REPO


def test_a_frozen_build_without_the_marker_is_a_bundle(exe_folder):
    assert path_utils.app_runtime() is AppRuntime.BUNDLE
    assert path_utils.app_runtime().frozen
    assert path_utils.launch_root() == exe_folder
    # Not installed: its data stays beside it, as a source run's does.
    assert path_utils.get_source_root() == exe_folder


def test_a_frozen_build_beside_the_marker_is_installed_and_keeps_its_data_in_documents(
    exe_folder, tmp_path, monkeypatch
):
    (exe_folder / INSTALLED_MARKER).write_text('')
    documents = tmp_path / 'Documents'
    # platformdirs is a conftest stand-in; the Documents folder is the user's.
    monkeypatch.setattr(sys.modules['platformdirs'], 'user_documents_dir', lambda: str(documents))
    version, _ = path_utils.read_version()

    assert path_utils.app_runtime() is AppRuntime.INSTALLED
    assert path_utils.get_source_root() == documents / f'LumaViewPro {version}'


def test_an_installed_build_whose_version_txt_names_no_version_is_refused(exe_folder, monkeypatch):
    # Its data folder is named for its version; with none, there is no folder
    # to answer, and the install folder it once fell back to cannot be written.
    (exe_folder / INSTALLED_MARKER).write_text('')
    monkeypatch.setattr(path_utils, 'read_version', lambda: ('', ''))

    with pytest.raises(InstallationFileError) as refused:
        path_utils.get_source_root()

    assert refused.value.file_path == path_utils.build_root() / 'version.txt'


def test_the_marker_does_not_make_a_source_run_installed(tmp_path, monkeypatch):
    # Only the installer writes the marker, and only beside a frozen exe.
    (tmp_path / INSTALLED_MARKER).write_text('')
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))

    assert path_utils.app_runtime() is AppRuntime.SOURCE


_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
from modules import path_utils
import lvp_logger
from modules.app_environment import init_environment
from drivers import camera_profiles
from drivers.fx2driver import _FX2Connection
env = init_environment()
print(json.dumps({
    'data_roots': {
        'launch_root': str(path_utils.launch_root()),
        'get_source_root': str(path_utils.get_source_root()),
        'lvp_appdata': lvp_logger.lvp_appdata,
        'init_environment': env.source_path,
    },
    'versions': {
        'init_environment': env.version,
        'read_version': path_utils.read_version()[0],
        'lvp_logger': lvp_logger.version,
    },
    'build_files': {
        'timing_record': str(
            camera_profiles.timing_record_path(camera_profiles.simulated_profile()).parent
        ),
        'fx2_hex': _FX2Connection.find_firmware_path(),
    },
}))
"""


@pytest.fixture
def farm(tmp_path):
    """A folder of links into this checkout, with its own copy of version.txt
    naming another version, and a working folder elsewhere holding decoys of
    the build's files."""
    folder = tmp_path / 'launch'
    folder.mkdir()
    for item in ('lvp_logger.py', 'modules', 'drivers', 'lib', 'data'):
        (folder / item).symlink_to(REPO / item)
    (folder / 'version.txt').write_text('9.9.9-farm\n2026-11-04 00:00\n', encoding='utf-8')
    decoy = tmp_path / 'decoy'
    for item in ('ui/lumaviewpro.kv', 'data/new_default_protocol.tsv', 'version.txt'):
        (decoy / item).parent.mkdir(parents=True, exist_ok=True)
        (decoy / item).write_text('decoy\n', encoding='utf-8')
    return folder, decoy


def test_a_folder_of_links_keeps_its_data_and_runs_the_checkouts_build(farm):
    folder, decoy = farm
    checkout_version = (REPO / 'version.txt').read_text(encoding='utf-8-sig').splitlines()[0]

    child = subprocess.run(
        [sys.executable, '-c', _CHILD, str(folder)],
        cwd=str(decoy),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    answers = json.loads(child.stdout.splitlines()[-1])

    assert answers['data_roots'] == dict.fromkeys(answers['data_roots'], str(folder))
    assert answers['versions'] == dict.fromkeys(answers['versions'], checkout_version)
    assert answers['build_files'] == {
        'timing_record': str(REPO / 'data' / 'camera_timing'),
        'fx2_hex': str(REPO / 'firmware' / 'LumascopeClassic.hex'),
    }
