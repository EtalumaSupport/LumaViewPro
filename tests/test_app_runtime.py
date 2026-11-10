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
import os
import pathlib
import subprocess
import sys

import pytest

from _lumascope.modules import path_utils
from _lumascope.modules.exceptions import InstallationFileError
from _lumascope.modules.path_utils import INSTALLED_MARKER, AppRuntime

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def exe_folder(tmp_path, monkeypatch):
    """This process as a PyInstaller build whose exe is in a temp folder."""
    folder = tmp_path / 'LumaViewPro'
    folder.mkdir()
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    # The bundle's folder, which a frozen process has beside sys.frozen: this
    # checkout's holds the build's files.
    monkeypatch.setattr(sys, '_MEIPASS', str(REPO), raising=False)
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
    # A file of the build's own: a reinstall puts it back.
    assert 'reinstall LumaViewPro' in str(refused.value)


def _unusable_labware(data_root: pathlib.Path) -> pathlib.Path:
    path = data_root / 'data' / 'labware.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"Wellplate": ')
    return path


def test_an_unusable_file_in_an_installed_builds_data_folder_is_mended_or_removed_never_reinstalled(
    exe_folder, tmp_path, monkeypatch
):
    """The installer writes nothing in Documents and each launch puts back a missing shipped
    file, so the refusal advises mending the file or removing it and launching again."""
    (exe_folder / INSTALLED_MARKER).write_text('')
    documents = tmp_path / 'Documents'
    # platformdirs is a conftest stand-in; the Documents folder is the user's.
    monkeypatch.setattr(sys.modules['platformdirs'], 'user_documents_dir', lambda: str(documents))
    path = _unusable_labware(path_utils.get_source_root())

    with pytest.raises(InstallationFileError) as refused:
        path_utils.read_installation_file(path)

    words = str(refused.value)
    assert 'remove it and start LumaViewPro again' in words, words
    assert 'reinstall' not in words, words


def test_an_unusable_data_file_on_a_build_the_installer_did_not_install_is_reinstalled(
    exe_folder,
):
    """A bundle keeps its data beside it and no launch puts a file back: today's advice stands."""
    path = _unusable_labware(path_utils.get_source_root())

    with pytest.raises(InstallationFileError) as refused:
        path_utils.read_installation_file(path)

    words = str(refused.value)
    assert 'reinstall LumaViewPro' in words, words
    assert 'start LumaViewPro again' not in words, words


def test_the_marker_does_not_make_a_source_run_installed(tmp_path, monkeypatch):
    # Only the installer writes the marker, and only beside a frozen exe.
    (tmp_path / INSTALLED_MARKER).write_text('')
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python'))

    assert path_utils.app_runtime() is AppRuntime.SOURCE


_CHILD = """
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], 'src'))
from _lumascope.modules import path_utils
import _lumascope.lvp_logger
from _lumascope.modules.app_environment import init_environment
from _lumascope.drivers import camera_profiles
from _lumascope.drivers.fx2driver import _FX2Connection
from _lumascope.modules.settings_init import prepare_settings
env = init_environment()
settings, _rejected = prepare_settings(
    _lumascope.lvp_logger.logger, str(path_utils.get_source_root()), fall_back_to_template=False
)
print(json.dumps({
    'data_roots': {
        'launch_root': str(path_utils.launch_root()),
        'get_source_root': str(path_utils.get_source_root()),
        'lvp_appdata': _lumascope.lvp_logger.lvp_appdata,
        'init_environment': env.source_path,
    },
    'versions': {
        'init_environment': env.version,
        'read_version': path_utils.read_version()[0],
        'lvp_logger': _lumascope.lvp_logger.version,
    },
    'build_files': {
        'timing_record': str(
            camera_profiles.timing_record_path(camera_profiles.simulated_profile()).parent
        ),
        'fx2_hex': _FX2Connection.find_firmware_path(),
    },
    'remembered_protocol': settings['protocol']['filepath'],
}))
"""


@pytest.fixture
def farm(tmp_path):
    """A folder of links into this checkout, with its own copy of version.txt
    naming another version, and a working folder elsewhere holding decoys of
    the build's files.

    Its data folder links the shipped files one by one, leaving out a
    checkout's own current.json, so its settings load from the template."""
    folder = tmp_path / 'launch'
    folder.mkdir()
    for item in ('src',):
        (folder / item).symlink_to(REPO / item)
    (folder / 'data').mkdir()
    for item in subprocess.run(
        ['git', 'ls-tree', '--name-only', 'HEAD', 'data/'],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split():
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
    # The shipped default protocol, a relative path, is the farm's data's.
    assert answers['remembered_protocol'] == str(folder / 'data' / 'new_default_protocol.tsv')


_RESOLVE_IN_CHILD = """
import sys
from kivy.resources import resource_find
print(resource_find(sys.argv[1]))
"""


def test_the_gui_runs_the_checkouts_kv_file_whatever_folder_it_starts_in(farm):
    # The kv file as the app names it, resolved as Kivy resolves it, from a
    # working folder holding a decoy at the same relative path. The entry is
    # read here, where the conftest's Kivy stand-ins let it import: a child
    # importing it would open a window.
    _folder, decoy = farm
    from lumaviewpro import LumaViewProApp

    child = subprocess.run(
        [sys.executable, '-c', _RESOLVE_IN_CHILD, LumaViewProApp.kv_file],
        cwd=str(decoy),
        env={**os.environ, 'KIVY_NO_ARGS': '1', 'KIVY_NO_CONSOLELOG': '1'},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert child.stdout.splitlines()[-1] == str(REPO / 'ui' / 'lumaviewpro.kv')


_INSTALLED_START_IN_CHILD = """
import json, os, shutil, sys
repo, bundle, exe_folder, documents, cut = sys.argv[1:6]
sys.path.insert(0, os.path.join(repo, 'src'))
# The process as an installed build, before the logger's import fixes its
# data root: frozen, its bundle the build's files, the marker beside its exe.
sys.frozen = True
sys._MEIPASS = bundle
sys.executable = os.path.join(exe_folder, 'LumaViewPro.exe')
import platformdirs
platformdirs.user_documents_dir = lambda: documents
if cut:
    # A stand-in by design: a disk that fills partway through one file is a
    # fault no simulator produces. The copy of `cut` writes half its bytes
    # in place, as a copy stopped mid-file leaves them, then fails.
    _copy = shutil.copyfile

    def _cut_short(src, dst, *args, **kwargs):
        if os.path.basename(src) != cut:
            return _copy(src, dst, *args, **kwargs)
        with open(src, 'rb') as whole, open(dst, 'wb') as part:
            part.write(whole.read()[: os.path.getsize(src) // 2])
        raise OSError(28, 'No space left on device')

    shutil.copyfile = _cut_short
from _lumascope.lvp_logger import start_program_logging
try:
    start_program_logging()
except Exception as e:
    print(json.dumps({'started': False, 'error': f'{type(e).__name__}: {e}'}))
else:
    print(json.dumps({'started': True, 'error': None}))
"""


@pytest.fixture
def installed(tmp_path):
    """An installed build's folders: its exe's, with the marker; its bundle,
    holding the shipped data and a version; and an empty Documents.

    The bundle's data is the checkout's tracked files, as the build ships
    them from a fresh clone, so no checkout's own current.json is in it."""
    exe_folder = tmp_path / 'LumaViewPro'
    exe_folder.mkdir()
    (exe_folder / INSTALLED_MARKER).write_text('')
    bundle = tmp_path / 'bundle'
    for item in subprocess.run(
        ['git', 'ls-files', 'data/'],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines():
        (bundle / item).parent.mkdir(parents=True, exist_ok=True)
        (bundle / item).write_bytes((REPO / item).read_bytes())
    (bundle / 'version.txt').write_text('9.9.9-installed\n2026-11-09 00:00\n', encoding='utf-8')
    documents = tmp_path / 'Documents'
    documents.mkdir()
    return exe_folder, bundle, documents


def _start_installed(installed, *, cut=''):
    exe_folder, bundle, documents = installed
    child = subprocess.run(
        [
            sys.executable,
            '-c',
            _INSTALLED_START_IN_CHILD,
            *(str(REPO), str(bundle), str(exe_folder), str(documents), cut),
        ],
        cwd=str(exe_folder),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(child.stdout.splitlines()[-1])


def _files(folder):
    return {
        path.relative_to(folder): path.read_bytes() for path in folder.rglob('*') if path.is_file()
    }


def test_an_installed_builds_first_launch_seeds_its_data_before_the_logging_start_reads_it(
    installed,
):
    # The start reads debug_mode from the data folder's settings.json, so a
    # data folder that does not exist yet is filled first, every shipped file.
    _exe_folder, bundle, documents = installed

    answer = _start_installed(installed)

    assert answer == {'started': True, 'error': None}
    data = documents / 'LumaViewPro 9.9.9-installed' / 'data'
    assert _files(data) == _files(bundle / 'data')


def test_an_installed_launch_puts_back_a_missing_shipped_file_and_keeps_the_users(installed):
    # A data folder missing a shipped file gets it back at the next launch;
    # a file already there, here the user's own plates, is never overwritten.
    _exe_folder, bundle, documents = installed
    data = documents / 'LumaViewPro 9.9.9-installed' / 'data'
    data.mkdir(parents=True)
    (data / 'labware.json').write_text('{"my plate": {}}', encoding='utf-8')

    answer = _start_installed(installed)

    assert answer == {'started': True, 'error': None}
    assert (data / 'settings.json').read_bytes() == (bundle / 'data' / 'settings.json').read_bytes()
    assert (data / 'labware.json').read_text(encoding='utf-8') == '{"my plate": {}}'


def test_a_seeding_cut_short_leaves_no_part_of_a_file_and_the_next_launch_completes_it(
    installed,
):
    # A copy stopped partway through settings.json fails that launch with its
    # reason and leaves nothing at the file's name, so the next launch copies
    # it whole instead of keeping a truncated file it would refuse forever.
    _exe_folder, bundle, documents = installed
    data = documents / 'LumaViewPro 9.9.9-installed' / 'data'

    first = _start_installed(installed, cut='settings.json')

    assert first['started'] is False
    assert 'No space left on device' in first['error']
    assert not (data / 'settings.json').exists()

    second = _start_installed(installed)

    assert second == {'started': True, 'error': None}
    assert (data / 'settings.json').read_bytes() == (bundle / 'data' / 'settings.json').read_bytes()
