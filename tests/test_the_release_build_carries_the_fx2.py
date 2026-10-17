# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Windows release build carries what the FX2 needs, where the frozen app looks.

The release spec shipped no firmware hex, so an installed LumaViewPro could
not upload the FX2's image ("Camera not found"), and bundled libusb at the
app root from a DLL that build.ps1 handed it through an environment
variable, where ``libusb_package.get_library_path()`` -- the one library the
driver runs on -- never looks. Both were silent: the build succeeded.

The spec now takes both through ``fx2_bundle``: the hex files into
``firmware/`` and libusb-package's own library into ``libusb_package/``,
refusing the build before PyInstaller runs and again on the built tree when
either is missing. PyInstaller does not run here, so the gate is run on
temporary trees.

The installer's driver package was an INF staged by hand from the Firmware
repo, dated 2024-01-01, so any Zadig binding made since outranked it (Windows
ranks equally signed packages for one device by the INF's date). The INF is
now in the repo, binding both of the FX2's identities, and every build
stamps its ``DriverVer`` through ``fx2_driver_package``.
"""

from __future__ import annotations

import configparser
import datetime
import importlib.util
import pathlib
import sys

import pytest

from drivers import fx2driver

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / 'scripts' / 'appBuild' / 'config'
BUILD_EXE = REPO_ROOT / 'scripts' / 'appBuild' / 'build_exe'
SHIPPED_INF = BUILD_EXE / 'fx2_driver' / 'Lumascope_Classic.inf'


def _load_config_module(name):
    # The spec and build.ps1 run these from their folder by path; the test does the same.
    spec = importlib.util.spec_from_file_location(name, CONFIG / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx2_bundle = _load_config_module('fx2_bundle')
fx2_driver_package = _load_config_module('fx2_driver_package')


def _library(tmp_path) -> pathlib.Path:
    library = tmp_path / 'wheel' / 'libusb-1.0.dll'
    library.parent.mkdir()
    library.write_bytes(b'dll')
    return library


def _source_tree(tmp_path, hex_names) -> pathlib.Path:
    root = tmp_path / 'clone'
    (root / 'firmware').mkdir(parents=True)
    for name in hex_names:
        (root / 'firmware' / name).write_text(':00000001FF\n')
    return root


def _dist(tmp_path, datas, binaries) -> pathlib.Path:
    """A dist tree laid out as PyInstaller lays out the entries and the
    spec's two executables."""
    dist = tmp_path / 'dist'
    for source, dest in [*datas, *binaries]:
        target = dist / dest / pathlib.Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pathlib.Path(source).read_bytes())
    (dist / fx2_bundle.DRIVER_STEP_EXE).write_bytes(b'exe')
    return dist


def test_the_entries_put_both_hex_files_in_firmware_and_the_library_in_its_package(tmp_path):
    library = _library(tmp_path)
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)

    datas, binaries = fx2_bundle.fx2_bundle_entries(str(root), library)

    assert sorted(pathlib.Path(source).name for source, _ in datas) == sorted(
        fx2_bundle.FX2_HEX_FILES
    )
    assert {dest for _, dest in datas} == {'firmware'}
    assert binaries == [(str(library), 'libusb_package')]


def test_the_shipped_clone_has_both_hex_files(tmp_path):
    datas, _ = fx2_bundle.fx2_bundle_entries(str(REPO_ROOT), _library(tmp_path))

    assert len(datas) == len(fx2_bundle.FX2_HEX_FILES)


def test_a_missing_hex_refuses_the_build_naming_it(tmp_path):
    root = _source_tree(tmp_path, ['LumascopeClassic.hex'])

    with pytest.raises(SystemExit, match=r'Lumascope600\.hex'):
        fx2_bundle.fx2_bundle_entries(str(root), _library(tmp_path))


def test_a_wheel_with_no_library_refuses_the_build(tmp_path):
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)

    with pytest.raises(SystemExit, match='holds no library'):
        fx2_bundle.fx2_bundle_entries(str(root), None)


def test_a_built_tree_with_the_library_at_its_root_is_refused(tmp_path):
    library = _library(tmp_path)
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)
    datas, _ = fx2_bundle.fx2_bundle_entries(str(root), library)
    # Where the old route put it: the app root, not the package's folder.
    dist = _dist(tmp_path, datas, [(str(library), '.')])

    with pytest.raises(SystemExit, match='libusb_package'):
        fx2_bundle.check_fx2_dist(str(dist), library.name)


def test_a_built_tree_laid_out_from_the_entries_passes(tmp_path):
    library = _library(tmp_path)
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)
    datas, binaries = fx2_bundle.fx2_bundle_entries(str(root), library)

    fx2_bundle.check_fx2_dist(str(_dist(tmp_path, datas, binaries)), library.name)


def test_a_built_tree_without_the_driver_step_is_refused(tmp_path):
    library = _library(tmp_path)
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)
    datas, binaries = fx2_bundle.fx2_bundle_entries(str(root), library)
    dist = _dist(tmp_path, datas, binaries)
    (dist / fx2_bundle.DRIVER_STEP_EXE).unlink()

    with pytest.raises(SystemExit, match=r'lvp_fx2_driver_step\.exe'):
        fx2_bundle.check_fx2_dist(str(dist), library.name)


def test_repair_runs_the_driver_step_the_build_ships():
    from modules import fx2_driver_step

    assert fx2_driver_step.DRIVER_STEP_EXE == fx2_bundle.DRIVER_STEP_EXE


def test_the_frozen_driver_finds_the_hex_where_the_build_puts_it(tmp_path, monkeypatch):
    library = _library(tmp_path)
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)
    datas, binaries = fx2_bundle.fx2_bundle_entries(str(root), library)
    dist = _dist(tmp_path, datas, binaries)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(dist), raising=False)

    found = fx2driver._FX2Connection.find_firmware_path()

    assert pathlib.Path(found) == dist / 'firmware' / 'LumascopeClassic.hex'


def test_the_spec_runs_the_gate_and_reads_no_dll_from_the_environment():
    # pin-justified: the spec is executed by PyInstaller, not imported.
    spec = (CONFIG / 'lumaviewpro_win_release.spec').read_text(encoding='utf-8')

    assert 'fx2_bundle_entries(SPECPATH, _libusb_path)' in spec
    assert 'check_fx2_dist(_dist_root,' in spec
    assert 'FX2_LIBUSB_DLL' not in spec
    assert "['lvp_fx2_driver_step.py']" in spec
    assert "name='lvp_fx2_driver_step'" in spec


def _read_inf(path) -> configparser.ConfigParser:
    inf = configparser.ConfigParser(
        delimiters=('=',),
        comment_prefixes=(';',),
        interpolation=None,
        strict=False,
        allow_no_value=True,
    )
    inf.optionxform = str
    inf.read(path, encoding='ascii')
    return inf


def test_the_stamped_package_binds_both_identities_dated_and_versioned_by_the_build(tmp_path):
    stamped = fx2_driver_package.stamp_driver_package(
        str(SHIPPED_INF), str(tmp_path), '4.0.0.38', datetime.date(2026, 10, 17)
    )

    inf = _read_inf(stamped)
    assert inf['Version']['DriverVer'].strip() == '10/17/2026,4.0.0.38'
    strings = {key: value.strip().strip('"') for key, value in inf['Strings'].items()}
    bound = {
        model.split(',')[1].strip(): strings[name.strip('%')]
        for name, model in inf['Standard.NTamd64'].items()
    }
    assert bound == {
        'USB\\VID_04B4&PID_EA17': 'Lumascope Classic',
        'USB\\VID_04B4&PID_8613': 'Lumascope Classic (Bootloader)',
    }
    assert strings['ManufacturerName'] == 'Etaluma'


def test_a_version_that_is_not_four_numbers_refuses_the_stamp(tmp_path):
    with pytest.raises(SystemExit, match='four-part numeric version'):
        fx2_driver_package.stamp_driver_package(
            str(SHIPPED_INF), str(tmp_path), '4.0.0-beta38', datetime.date(2026, 10, 17)
        )


def test_an_inf_without_its_placeholder_refuses_the_stamp(tmp_path):
    already = tmp_path / 'source' / 'Lumascope_Classic.inf'
    already.parent.mkdir()
    already.write_text(
        SHIPPED_INF.read_text(encoding='ascii').replace(
            fx2_driver_package.PLACEHOLDER, '01/01/2024,1.0.0.0'
        ),
        encoding='ascii',
    )

    with pytest.raises(SystemExit, match='0 times, not once'):
        fx2_driver_package.stamp_driver_package(
            str(already), str(tmp_path / 'out'), '4.0.0.38', datetime.date(2026, 10, 17)
        )


def test_every_build_stamps_the_repos_inf_and_the_installer_runs_the_driver_step():
    # pin-justified: build.ps1 runs on the Windows build box, not here.
    build = (REPO_ROOT / 'scripts' / 'appBuild' / 'build.ps1').read_text(encoding='utf-8')
    package = (BUILD_EXE / 'wix' / 'Package.wxs').read_text(encoding='utf-8-sig')

    stamp = build.index('fx2_driver_package.py')
    assert 'build_exe\\fx2_driver\\Lumascope_Classic.inf' in build[stamp : stamp + 300]
    assert build.index('"Fx2InfPath=$fx2_inf"') > stamp
    assert 'if ($fx2_inf)' not in build
    assert '<?ifdef Fx2InfPath ?>' not in package
    assert '[INSTALLFOLDER]lvp_fx2_driver_step.exe&quot; install' in package
    assert '[INSTALLFOLDER]lvp_fx2_driver_step.exe&quot; uninstall' in package
    assert 'NOT UPGRADINGPRODUCTCODE' in package
