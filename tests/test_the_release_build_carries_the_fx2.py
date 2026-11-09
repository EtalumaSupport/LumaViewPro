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
import re

import pytest

from _lumascope.drivers import fx2driver

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
    from _lumascope.modules import fx2_driver_step

    assert fx2_driver_step.DRIVER_STEP_EXE == fx2_bundle.DRIVER_STEP_EXE


def test_the_frozen_driver_finds_the_hex_where_the_build_puts_it(tmp_path, monkeypatch):
    library = _library(tmp_path)
    root = _source_tree(tmp_path, fx2_bundle.FX2_HEX_FILES)
    datas, binaries = fx2_bundle.fx2_bundle_entries(str(root), library)
    dist = _dist(tmp_path, datas, binaries)
    # Frozen, the build root is the bundle's folder the spec collects into.
    monkeypatch.setattr(fx2driver, 'build_root', lambda: dist)

    found = fx2driver._FX2Connection.find_firmware_path()

    assert pathlib.Path(found) == dist / 'firmware' / 'LumascopeClassic.hex'


def test_the_spec_runs_the_gate_and_reads_no_dll_from_the_environment():
    # pin-justified: the spec is executed by PyInstaller, not imported.
    spec = (CONFIG / 'lumaviewpro_win_release.spec').read_text(encoding='utf-8')

    assert 'fx2_bundle_entries(SPECPATH, _libusb_path)' in spec
    assert 'check_fx2_dist(_dist_root,' in spec
    assert 'FX2_LIBUSB_DLL' not in spec
    assert "['scripts\\\\appBuild\\\\build_exe\\\\fx2_driver\\\\lvp_fx2_driver_step.py']" in spec
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


def _repo_inf_copy(tmp_path) -> pathlib.Path:
    inf = tmp_path / 'checkout' / 'Lumascope_Classic.inf'
    inf.parent.mkdir()
    inf.write_bytes(SHIPPED_INF.read_bytes())
    return inf


def test_signing_writes_the_signing_days_driver_ver_into_the_repos_inf(tmp_path):
    # The catalog is signed over the INF's bytes as committed, so the
    # writer changes the DriverVer value and nothing else.
    inf = _repo_inf_copy(tmp_path)
    before = inf.read_bytes()

    written = fx2_driver_package.write_driver_ver(str(inf), '4.0.0.44', datetime.date(2026, 11, 9))

    assert written == '11/09/2026,4.0.0.44'
    assert _read_inf(inf)['Version']['DriverVer'].strip() == written
    assert inf.read_bytes() == before.replace(
        fx2_driver_package.PLACEHOLDER.encode('ascii'), written.encode('ascii')
    )


def test_signing_again_writes_over_the_last_signings_driver_ver(tmp_path):
    inf = _repo_inf_copy(tmp_path)
    fx2_driver_package.write_driver_ver(str(inf), '4.0.0.44', datetime.date(2026, 11, 9))

    written = fx2_driver_package.write_driver_ver(str(inf), '4.0.0.45', datetime.date(2026, 12, 1))

    assert _read_inf(inf)['Version']['DriverVer'].strip() == written == '12/01/2026,4.0.0.45'


def test_signing_refuses_a_driver_ver_the_inf_already_has(tmp_path):
    # The driver step finds our current package by its DriverVer, so two
    # signed packages with one DriverVer would let it keep a stale one and
    # delete the shipped one as older.
    inf = _repo_inf_copy(tmp_path)
    fx2_driver_package.write_driver_ver(str(inf), '4.0.0.44', datetime.date(2026, 11, 9))
    signed = inf.read_bytes()

    with pytest.raises(SystemExit, match=r'already 11/09/2026,4\.0\.0\.44'):
        fx2_driver_package.write_driver_ver(str(inf), '4.0.0.44', datetime.date(2026, 11, 9))
    assert inf.read_bytes() == signed


def test_sign_classic_driver_signs_the_named_checkouts_package_and_builds_nothing():
    # pin-justified: build.ps1 runs on the Windows build box, not here. The
    # mode signs the package in a checkout for commit (a fresh clone has no
    # .git to commit from), through the build's one certificate lookup and
    # one signer, and the four-part version the installer uses.
    build = (REPO_ROOT / 'scripts' / 'appBuild' / 'build.ps1').read_text(encoding='utf-8')

    assert '[string]$SignClassicDriver = ""' in build
    mode = build[build.index('if ($SignClassicDriver) {\n') :]
    mode = mode[: mode.index('\n}\n')]
    assert build.index('function Invoke-Sign(') < build.index('if ($SignClassicDriver) {\n')
    assert build.index('if ($SignClassicDriver) {\n') < build.index(
        '# Clean previous temp, clone fresh'
    )
    assert 'config\\fx2_driver_package.py' in mode and '"sign"' in mode
    assert 'Get-InstallerVersion' in mode
    assert '& $inf2cat "/driver:$fx2_package_dir" "/os:10_X64"' in mode
    assert 'Invoke-Sign (Join-Path $fx2_package_dir "Lumascope_Classic.cat")' in mode
    assert mode.rstrip().endswith('Exit 0')
    assert build.count('$wix_ver = Get-InstallerVersion $version') == 1
    assert build.count("-match '^(\\d+\\.\\d+\\.\\d+)-beta(\\d+)$'") == 1


def test_a_signed_build_alone_carries_the_classic_driver_and_the_installer_only_uninstalls_it():
    # pin-justified: build.ps1 and the WiX sources run on the Windows build
    # box, not here. Windows accepts a driver package only with a signed
    # catalog, so an unsigned build leaves the package out; and the driver is
    # bound where Windows can ask the person (the setup bundle's step, and the
    # app's Repair), never by an MSI action running as SYSTEM.
    build = (REPO_ROOT / 'scripts' / 'appBuild' / 'build.ps1').read_text(encoding='utf-8')
    package = (BUILD_EXE / 'wix' / 'Package.wxs').read_text(encoding='utf-8-sig')

    signed = build.index('if ($signing_thumbprint) {\n    $fx2_dir_out')
    stamp = build.index('fx2_driver_package.py', signed)
    assert signed < stamp < build.index('"Fx2CatPath=$fx2_cat"')
    assert 'build_exe\\fx2_driver\\Lumascope_Classic.inf' in build[stamp : stamp + 300]
    assert build.index('Invoke-Sign $fx2_cat') < build.index('"Fx2InfPath=$fx2_inf"')
    start = package.index('<?ifdef Fx2InfPath ?>')
    driver = package[start : package.index('<?endif ?>', start)]
    assert '$(var.Fx2CatPath)' in driver
    assert '[INSTALLFOLDER]lvp_fx2_driver_step.exe&quot; uninstall' in driver
    assert 'NOT UPGRADINGPRODUCTCODE' in driver
    assert 'lvp_fx2_driver_step.exe&quot; install' not in package


def test_the_setup_bundle_runs_the_one_file_step_after_the_app_in_a_signed_build():
    # pin-justified: the spec, build.ps1 and Bundle.wxs run on the Windows
    # build box, not here. The bundle's per-machine package is what runs the
    # step elevated on the person's desktop at install (WiX: "must be
    # executed elevated"), where a deferred MSI action impersonating them
    # runs with user rights; a one-file build, so it runs from the bundle's
    # cache with the INF and catalog beside it. Not vital, so a declined
    # prompt fails nothing and Repair completes it; permanent, so uninstall
    # stays the MSI's; run at every install, so an upgrade takes back what a
    # later package took. Until the signed pair is committed, only a signed
    # build has an INF and catalog to give it.
    spec = (CONFIG / 'lumaviewpro_win_release.spec').read_text(encoding='utf-8')
    build = (REPO_ROOT / 'scripts' / 'appBuild' / 'build.ps1').read_text(encoding='utf-8')
    bundle = (BUILD_EXE / 'wix' / 'Bundle.wxs').read_text(encoding='utf-8-sig')
    min_version = int((REPO_ROOT / 'scripts' / 'appBuild' / 'MIN_BUILD_SCRIPT_VERSION').read_text())

    setup = spec[spec.index('driver_step_setup_exe = EXE(') :]
    assert "name='lvp_fx2_driver_step_setup'" in setup[: setup.index('\n)\n')]
    assert "('version.txt', '.')" in spec[spec.index('driver_step = Analysis(') :]

    lvp = bundle[bundle.index('<MsiPackage Id="LVP"') :]
    lvp = lvp[: lvp.index('</MsiPackage>')]
    assert '<MsiProperty Name="INSTALLFOLDER" Value="[InstallFolder]" />' in lvp
    start = bundle.index('<?ifdef Fx2InfPath ?>', bundle.index('<MsiPackage Id="LVP"'))
    step = bundle[start : bundle.index('<?endif ?>', start)]
    for attribute in (
        'Id="Fx2DriverStep"',
        'SourceFile="$(var.Fx2DriverStepExe)"',
        'PerMachine="yes"',
        'Vital="no"',
        'Permanent="yes"',
        'DetectCondition="0"',
        'InstallArguments=\'install-if-classic "[InstallFolder]logs"\'',
        '<Payload SourceFile="$(var.Fx2InfPath)" />',
        '<Payload SourceFile="$(var.Fx2CatPath)" />',
    ):
        assert attribute in step, attribute

    signed = build[build.index('if ($signing_thumbprint) {\n    $fx2_dir_out') :]
    signed = signed[: signed.index('\n}\n')]
    assert 'Invoke-Sign $fx2_step_setup' in signed
    defines = signed[signed.index('$bundle_args_fx2 = @(') :]
    for define in (
        '"Fx2DriverStepExe=$fx2_step_setup"',
        '"Fx2InfPath=$fx2_inf"',
        '"Fx2CatPath=$fx2_cat"',
    ):
        assert define in defines, define
    assert '$bundle_args += $bundle_args_fx2' in build
    assert 'lvp_fx2_driver_step_setup.exe' in signed
    assert min_version >= 9


def test_a_bundle_variable_that_names_another_is_resolved_when_read():
    # pin-justified: Bundle.wxs runs in Burn on the Windows build box, not
    # here. Burn takes a variable's value literally unless it is declared
    # Type="formatted": beta44's InstallFolder reached the MSI as the text
    # "[ProgramFiles6432Folder]LumaViewPro\", and the install failed with
    # Error 1606 at CostFinalize and rolled back.
    bundle = (BUILD_EXE / 'wix' / 'Bundle.wxs').read_text(encoding='utf-8-sig')
    variables = re.findall(r'<Variable\b[^>]*>', bundle)
    assert variables, 'Bundle.wxs declares no Variable'
    literal = [v for v in variables if '[' in v and 'Type="formatted"' not in v]
    assert not literal, literal
