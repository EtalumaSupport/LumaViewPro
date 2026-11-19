# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A source run starts only on the packages ``requirements.txt`` pins.

The judge (``packages_unlike_the_pins``) compares each pin that applies to
this host with the installed metadata; the package's import refuses a row
(``PackagesNotAsPinnedError``) on every host, before anything third-party
loads; an installed build is not judged; the GUI entry shows the refusal in
its startup window and exits 1, and lets any other failure of its first
import go up unchanged. The refusal's words name each row and the fix, each
half for the launch it fixes, the pip half only where the Python that ran
can install into itself, each path and command on a line of its own; the
refusal carries the fix as data for the window to act on.

The installed versions are real metadata: a stand-in ``.dist-info`` earlier
on the path is what ``importlib.metadata`` reads first, so no test branch
sits in the judge.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from _lumascope.lib.build_files import build_root
from _lumascope.lib.installed_packages import (
    PackagesNotAsPinnedError,
    PinRow,
    RunningPython,
    packages_unlike_the_pins,
)

REPO = build_root()
PROBE = 'lvp-pin-probe'


def _installed(folder, name, version):
    """A stand-in installed distribution: the metadata ``importlib.metadata`` reads."""
    info = folder / f'{name.replace("-", "_")}-{version}.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n', encoding='utf-8'
    )
    return folder


def _pins(tmp_path, *lines):
    path = tmp_path / 'requirements.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def test_a_package_installed_as_pinned_is_no_row(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_installed(tmp_path / 'site', PROBE, '1.2')))

    assert packages_unlike_the_pins(_pins(tmp_path, '# a comment', '', f'{PROBE}==1.2')) == []


def test_a_package_below_or_above_its_pin_or_missing_is_one_row_each(tmp_path, monkeypatch):
    site = tmp_path / 'site'
    _installed(site, 'lvp-pin-below', '1.1')
    _installed(site, 'lvp-pin-above', '1.3')
    monkeypatch.syspath_prepend(str(site))
    pins = _pins(tmp_path, 'lvp-pin-below==1.2', 'lvp-pin-above==1.2', 'lvp-pin-absent==1.2')

    assert packages_unlike_the_pins(pins) == [
        PinRow('lvp-pin-below', '1.1', '==1.2'),
        PinRow('lvp-pin-above', '1.3', '==1.2'),
        PinRow('lvp-pin-absent', None, '==1.2'),
    ]


def test_a_pin_whose_marker_does_not_hold_here_is_not_judged(tmp_path):
    pins = _pins(tmp_path, 'lvp-pin-absent==1.2; sys_platform == "no-such-platform"')

    assert packages_unlike_the_pins(pins) == []


def test_without_packaging_the_one_row_is_packaging_s(tmp_path, monkeypatch):
    # None makes the import raise ImportError, as a Python without packaging does.
    monkeypatch.setitem(sys.modules, 'packaging.requirements', None)

    assert packages_unlike_the_pins(_pins(tmp_path, 'numpy==2.4.6', 'packaging==26.3')) == [
        PinRow('packaging', None, '==26.3')
    ]


def test_an_unreadable_or_unparsable_requirements_file_is_refused_as_unread(tmp_path):
    with pytest.raises(PackagesNotAsPinnedError) as missing:
        packages_unlike_the_pins(tmp_path / 'requirements.txt')
    with pytest.raises(PackagesNotAsPinnedError) as unparsable:
        packages_unlike_the_pins(_pins(tmp_path, 'not a requirement ==='))

    assert (missing.value.reason, unparsable.value.reason) == ('pins_unread', 'pins_unread')
    assert f'\n    {tmp_path / "requirements.txt"}\n' in str(missing.value)
    assert 'Get the LumaViewPro source again' in str(missing.value)
    # Its fix is the source got again: no script to start, no pip line to copy.
    assert (missing.value.install_script, missing.value.pip_line) == (None, None)


_ROWS = [PinRow('pypylon', '26.4.1', '==26.10'), PinRow('matplotlib', None, '==3.10.9')]


def _refusal(monkeypatch, platform, **facts):
    monkeypatch.setattr(sys, 'platform', platform)
    python = RunningPython(**{'executable': '/py/bin/python3', **facts})
    return PackagesNotAsPinnedError.not_as_pinned(_ROWS, python)


def test_the_words_name_each_row_and_both_halves_of_the_fix_each_path_on_its_own_line(
    monkeypatch,
):
    words = str(
        _refusal(monkeypatch, 'darwin', in_venv=False, has_pip=True, externally_managed=False)
    )
    pip_line = f'"/py/bin/python3" -m pip install -r "{REPO / "requirements.txt"}"'

    assert words.startswith(
        'LumaViewPro cannot start: 2 Python packages are not the versions they need.\n'
    )
    assert '\n    pypylon       26.4.1 installed, 26.10 needed\n' in words
    assert '\n    matplotlib    not installed, 3.10.9 needed\n' in words
    assert (
        'Run the install script, then start LumaViewPro with run.sh:\n'
        f'    {REPO / "scripts" / "install_mac.sh"}\n'
        f'    {REPO / "run.sh"}\n'
    ) in words
    assert words.endswith(
        f'To run your own program or the REST server with this Python instead:\n    {pip_line}'
    )


def test_the_refusal_carries_its_fix_as_the_words_state_it(monkeypatch):
    refusal = _refusal(monkeypatch, 'darwin', in_venv=False, has_pip=True, externally_managed=False)

    assert refusal.install_script == str(REPO / 'scripts' / 'install_mac.sh')
    assert refusal.install_args == ()
    assert refusal.pip_line == f'"/py/bin/python3" -m pip install -r "{REPO / "requirements.txt"}"'
    assert f'    {refusal.pip_line}' in str(refusal)


def test_a_python_that_cannot_install_into_itself_is_given_no_pip_line(monkeypatch):
    managed = _refusal(monkeypatch, 'linux', in_venv=False, has_pip=True, externally_managed=True)
    pipless = _refusal(monkeypatch, 'linux', in_venv=True, has_pip=False, externally_managed=False)

    for refusal, why in ((managed, 'it is externally managed'), (pipless, 'it has no pip')):
        words = str(refusal)
        assert refusal.pip_line is None
        assert '-m pip install' not in words
        assert f'cannot install packages into itself ({why}):\n    /py/bin/python3\n' in words
        assert words.endswith(f'installs into:\n    {REPO / "venv" / "bin" / "python"}')
        assert f'\n    {REPO / "scripts" / "install_linux.sh"}\n' in words


def test_a_windows_venv_s_python_is_told_to_update_its_venv(monkeypatch):
    venv = _refusal(monkeypatch, 'win32', in_venv=True, has_pip=True, externally_managed=False)
    system = _refusal(monkeypatch, 'win32', in_venv=False, has_pip=True, externally_managed=False)
    script = str(REPO / 'scripts' / 'install_windows.bat')

    assert (venv.install_script, venv.install_args) == (script, ('--venv',))
    assert (system.install_script, system.install_args) == (script, ())
    assert f'\n    {script} --venv\n    {REPO / "run.bat"}\n' in str(venv)
    assert f'\n    {script}\n    {REPO / "run.bat"}\n' in str(system)


def _child(tmp_path, code, *, stand_in=True):
    """Run ``code`` in a fresh Python from outside the checkout, pypylon stood in at 26.4.1."""
    site = _installed(tmp_path / 'site', 'pypylon', '26.4.1') if stand_in else tmp_path
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(site), str(REPO / 'src')])}
    return subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=60,
    )


def test_the_package_s_import_refuses_a_run_whose_packages_are_not_as_pinned(tmp_path):
    done = _child(tmp_path, 'import _lumascope')

    assert done.returncode == 1
    assert 'PackagesNotAsPinnedError' in done.stderr
    assert '    pypylon    26.4.1 installed, 26.10 needed' in done.stderr


def test_the_package_s_import_passes_on_the_pins(tmp_path):
    """The suite's own environment is the pins, so the same import with nothing stood in passes."""
    done = _child(tmp_path, 'import _lumascope', stand_in=False)

    assert done.returncode == 0, done.stderr


def test_an_installed_build_is_not_judged(tmp_path):
    done = _child(
        tmp_path,
        f'import sys; sys.frozen = True; sys._MEIPASS = {str(tmp_path)!r}; import _lumascope',
    )

    assert done.returncode == 0, done.stderr


# A stand-in by design: tkinter is made unimportable in the child, as on a
# Python built without it, so no window opens on the host running the suite.
# The entry's folder goes first on the path, as `python lumaviewpro.py` puts it.
_GUI_ENTRY = (
    'import runpy, sys\n'
    "sys.modules['tkinter'] = None\n"
    f'sys.path.insert(0, {str(REPO)!r})\n'
    '{before}'
    f'sys.argv = [{str(REPO / "lumaviewpro.py")!r}]\n'
    f'runpy.run_path({str(REPO / "lumaviewpro.py")!r}, run_name="__main__")\n'
)


def test_the_gui_entry_shows_the_refusal_and_exits_1(tmp_path):
    done = _child(tmp_path, _GUI_ENTRY.format(before=''))

    assert done.returncode == 1
    assert done.stderr.startswith('ERROR: LumaViewPro cannot start'), done.stderr
    assert done.stderr.endswith(
        '\nThe window could not be shown: this Python has no tkinter '
        '(import of tkinter halted; None in sys.modules).\n'
    ), done.stderr
    assert 'Traceback' not in done.stderr


def test_the_gui_entry_lets_any_other_first_import_failure_go_up_unchanged(tmp_path):
    # None makes the API's first module fail to import after the judge passes.
    before = "sys.modules['_lumascope.modules.wall_time'] = None\n"
    done = _child(tmp_path, _GUI_ENTRY.format(before=before), stand_in=False)

    assert done.returncode == 1
    assert done.stderr.startswith('Traceback'), done.stderr
    assert 'ModuleNotFoundError: import of _lumascope.modules.wall_time halted' in done.stderr
    assert 'ERROR:' not in done.stderr
