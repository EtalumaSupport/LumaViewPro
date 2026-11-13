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
now in the repo with its signed catalog, binding both of the FX2's
identities, signed once by ``build.ps1 -SignClassicDriver`` and carried by
every build as committed, after ``fx2_driver_package`` checks that the INF is
the one its catalog was signed over.
"""

from __future__ import annotations

import configparser
import datetime
import importlib.util
import pathlib
import re
import subprocess

import pytest

from _lumascope.drivers import fx2driver

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / 'scripts' / 'appBuild' / 'config'
BUILD_EXE = REPO_ROOT / 'scripts' / 'appBuild' / 'build_exe'
SHIPPED_PACKAGE = BUILD_EXE / 'fx2_driver'
SHIPPED_INF = SHIPPED_PACKAGE / 'Lumascope_Classic.inf'
SHIPPED_CAT = SHIPPED_PACKAGE / 'Lumascope_Classic.cat'


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


def test_repair_runs_the_driver_step_and_package_the_build_ships():
    from _lumascope.modules import fx2_driver_step

    assert fx2_driver_step.DRIVER_STEP_EXE == fx2_bundle.DRIVER_STEP_EXE
    assert fx2_driver_step.PACKAGE_INF == fx2_driver_package.INF_NAME == SHIPPED_INF.name


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


def test_the_committed_package_binds_both_identities_and_is_the_one_its_catalog_signed():
    # The check every build runs before the MSI: an INF edited and not signed
    # again fails here, on any platform, at its own commit.
    driver_ver = fx2_driver_package.check_driver_package(str(SHIPPED_PACKAGE))

    inf = _read_inf(SHIPPED_INF)
    assert inf['Version']['DriverVer'].strip() == driver_ver
    assert inf['Version']['CatalogFile'].strip() == SHIPPED_CAT.name
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


def _package_copy(tmp_path, inf: bytes | None = None, catalog_name: str = SHIPPED_CAT.name):
    """The committed pair copied to its own folder, its INF's bytes optionally replaced."""
    package = tmp_path / 'package'
    package.mkdir()
    (package / SHIPPED_INF.name).write_bytes(SHIPPED_INF.read_bytes() if inf is None else inf)
    (package / catalog_name).write_bytes(SHIPPED_CAT.read_bytes())
    return package


def test_an_inf_whose_line_endings_changed_after_signing_is_refused(tmp_path):
    # A checkout that converts line endings changes every byte the catalog's
    # hash covers, and Windows then refuses the package at install.
    package = _package_copy(tmp_path, inf=SHIPPED_INF.read_bytes().replace(b'\r\n', b'\n'))

    with pytest.raises(SystemExit, match=r'is not the INF Lumascope_Classic\.cat was signed over'):
        fx2_driver_package.check_driver_package(str(package))


def test_a_catalog_named_in_another_case_than_the_infs_catalogfile_is_refused(tmp_path):
    # Inf2Cat writes the catalog's name in lower case; Windows reads the file
    # the INF names, and macOS's file system would match either.
    package = _package_copy(tmp_path, catalog_name='lumascope_classic.cat')

    with pytest.raises(SystemExit, match=r'holds no file named Lumascope_Classic\.cat'):
        fx2_driver_package.check_driver_package(str(package))


def test_a_driver_ver_that_names_no_real_date_is_refused(tmp_path):
    driver_ver = fx2_driver_package.check_driver_package(str(SHIPPED_PACKAGE))
    version = driver_ver.partition(',')[2]
    inf = SHIPPED_INF.read_bytes().replace(
        driver_ver.encode('ascii'), f'13/40/2026,{version}'.encode('ascii')
    )

    with pytest.raises(SystemExit, match='names no real date'):
        fx2_driver_package.check_driver_package(str(_package_copy(tmp_path, inf=inf)))


def _repo_inf_copy(tmp_path) -> pathlib.Path:
    inf = tmp_path / 'checkout' / 'Lumascope_Classic.inf'
    inf.parent.mkdir()
    inf.write_bytes(SHIPPED_INF.read_bytes())
    return inf


def test_a_signing_writes_the_signing_day_and_the_committed_version_raised_by_one(tmp_path):
    # The driver's version is its own, not the app's: a signing raises the
    # committed INF's last part. The catalog is signed over the INF's bytes as
    # committed, so the writer changes the DriverVer value and nothing else.
    inf = _repo_inf_copy(tmp_path)
    before = inf.read_bytes()
    committed = fx2_driver_package.check_driver_package(str(SHIPPED_PACKAGE))
    *head, last = committed.partition(',')[2].split('.')
    raised = '.'.join([*head, str(int(last) + 1)])

    written = fx2_driver_package.write_driver_ver(str(inf), datetime.date(2026, 12, 1))

    assert written == f'12/01/2026,{raised}'
    assert inf.read_bytes() == before.replace(committed.encode('ascii'), written.encode('ascii'))


def test_each_signing_raises_the_drivers_last_version_part_even_on_one_day(tmp_path):
    # The driver step keeps the package with our current DriverVer and
    # removes our others, so no two signings may write one DriverVer: the
    # version makes each new, even a second signing on the same day.
    inf = _repo_inf_copy(tmp_path)

    first = fx2_driver_package.write_driver_ver(str(inf), datetime.date(2026, 12, 1))
    second = fx2_driver_package.write_driver_ver(str(inf), datetime.date(2026, 12, 1))

    assert first != second
    assert first.rpartition('.')[0] == second.rpartition('.')[0]
    assert int(second.rpartition('.')[2]) == int(first.rpartition('.')[2]) + 1
    assert _read_inf(inf)['Version']['DriverVer'].strip() == second


def test_git_keeps_the_signed_pairs_bytes_in_every_checkout():
    # A checkout with line-ending conversion would hand the build an INF its
    # catalog was not signed over; git reads this from .gitattributes.
    result = subprocess.run(
        ['git', 'check-attr', 'text', '--', str(SHIPPED_INF), str(SHIPPED_CAT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert [line.rpartition(': ')[2] for line in result.stdout.splitlines()] == ['unset', 'unset']


def _build_script() -> str:
    return (REPO_ROOT / 'scripts' / 'appBuild' / 'build.ps1').read_text(encoding='utf-8')


def _block(text: str, opener: str) -> str:
    """The top-level block ``opener`` begins, to its closing brace."""
    block = text[text.index(opener) :]
    return block[: block.index('\n}\n')]


def test_sign_classic_driver_signs_in_the_builds_own_clone_and_copies_out_only_a_verified_pair():
    # pin-justified: build.ps1 runs on the Windows build box, not here. The
    # mode signs in the clone the next run deletes, so a failed signature
    # leaves no INF dated and no checkout is needed; the catalog holds the
    # INF's hashes signed or not, so only signtool's verify can pass the pair
    # out; and nothing runs after the log's copy into the package folder.
    build = _build_script()

    assert '[switch]$SignClassicDriver' in build
    mode = _block(build, 'if ($SignClassicDriver) {\n    Write-Phase')
    assert build.index('git clone --depth 1 --branch $Branch $repo_url $clone') < build.index(mode)
    assert build.index('$version = $matches[0]') < build.index(mode)
    assert 'Join-Path $clone "scripts\\appBuild\\config\\fx2_driver_package.py"' in mode
    # The driver's version is its own: the mode passes no app version and
    # names its folder for the version the writer printed.
    assert '"sign", $fx2_package_inf))' in mode and 'Get-InstallerVersion' not in mode
    sign = mode.index('Invoke-Sign $fx2_package_cat')
    verify = mode.index(
        '& $signtool verify /pa /tw /hash SHA256 /c $fx2_package_cat $fx2_package_inf'
    )
    copy = mode.index('Copy-Item $fx2_package_inf')
    assert mode.index('& $inf2cat "/driver:$fx2_package_dir" "/os:10_X64"') < sign < verify < copy
    assert 'Join-Path $artifacts "classic_driver_$($fx2_driver_version)_$build_log_ts"' in mode
    assert '-Destination (Join-Path $fx2_package_out "Lumascope_Classic.cat")' in mode
    assert mode.rstrip().endswith('Join-Path $fx2_package_out "sign.log") -Force\n    Exit 0')
    assert build.count('$wix_ver = Get-InstallerVersion $version') == 1
    assert build.count("-match '^(\\d+\\.\\d+\\.\\d+)-beta(\\d+)$'") == 1


def test_the_signing_mode_asks_the_branch_and_never_assigns_the_package_type():
    # pin-justified: build.ps1 runs on the Windows build box, not here. An
    # empty assignment to the ValidateSet-checked $BuildType throws, which
    # stopped -SignClassicDriver run alone; the mode clones a branch, so it
    # asks for one as a build does.
    build = _build_script()

    assert 'if (-not $Branch) {' in build
    questions = _block(build, 'if (-not $SignClassicDriver) {')
    assert '$BuildType = "Release"' in questions and '$BuildType = "Dev"' in questions
    assignments = re.findall(r'^\s*\$BuildType = .*$|\{ \$BuildType = [^}]*\}', build, re.MULTILINE)
    assert assignments == ['{ $BuildType = "Release" }', '{ $BuildType = "Dev" }']


def test_a_build_reads_the_certificate_only_when_signing_is_asked_for():
    # pin-justified: build.ps1 runs on the Windows build box, not here. Each
    # signed build spends signatures from a monthly allowance, so the
    # certificate's file on the box never signs a build by being there: -Sign
    # or a yes at the interactive question does, default no.
    build = _build_script()

    lookup = _block(build, 'if ($sign_build -or $SignClassicDriver) {')
    assert 'ReadAllText($signing_file)' in lookup
    assert build.count('ReadAllText($signing_file)') == 1
    assert '[switch]$Sign,' in build and '$sign_build = [bool]$Sign' in build
    questions = _block(build, 'if (-not $SignClassicDriver) {')
    ask = 'if (-not $sign_build -and -not $build_type_given -and (Test-Path $signing_file)) {'
    assert ask in questions
    assert 'Read-Host -Prompt "Sign this build? [y/N]"' in questions
    assert "$sign_build = $sign_choice -match '^(y|yes)$'" in questions


def test_every_build_carries_the_committed_pair_checked_and_the_installer_only_uninstalls_it():
    # pin-justified: build.ps1 and the WiX sources run on the Windows build
    # box, not here. The pair is signed once and committed, so every build,
    # signed or not, carries it as committed after its check, and none makes
    # or signs a catalog; and the driver is bound where Windows can ask the
    # person (the setup bundle's step, and the app's Repair), never by an MSI
    # action running as SYSTEM.
    build = _build_script()
    package = (BUILD_EXE / 'wix' / 'Package.wxs').read_text(encoding='utf-8-sig')

    check = build.index('fx2_driver_package.py") check $fx2_package')
    assert (
        build.index('$fx2_package = Join-Path $src "scripts\\appBuild\\build_exe\\fx2_driver"')
        < check
    )
    assert check < build.index('$msi_args += @(\'-d\', "Fx2InfPath=$fx2_inf"')
    assert '$fx2_inf = Join-Path $fx2_package "Lumascope_Classic.inf"' in build
    assert '$fx2_cat = Join-Path $fx2_package "Lumascope_Classic.cat"' in build
    # Only the signing mode makes and signs a catalog.
    mode = _block(build, 'if ($SignClassicDriver) {\n    Write-Phase')
    assert build.count('& $inf2cat ') == mode.count('& $inf2cat ') == 1
    assert 'Invoke-Sign $fx2_cat' not in build
    assert '"ClassicDriver=bundled (DriverVer $fx2_driver_ver)"' in build
    assert 'Fx2InfPath ?>' not in package
    driver = package[package.index('<Directory Id="Fx2DriverDir"') :]
    driver = driver[: driver.index('</InstallExecuteSequence>')]
    assert '$(var.Fx2InfPath)' in driver and '$(var.Fx2CatPath)' in driver
    assert '[INSTALLFOLDER]lvp_fx2_driver_step.exe&quot; uninstall' in driver
    assert 'NOT UPGRADINGPRODUCTCODE' in driver
    assert 'lvp_fx2_driver_step.exe&quot; install' not in package


def test_the_setup_bundle_runs_the_one_file_step_after_the_app_in_every_build():
    # pin-justified: the spec, build.ps1 and Bundle.wxs run on the Windows
    # build box, not here. The bundle's per-machine package is what runs the
    # step elevated on the person's desktop at install (WiX: "must be
    # executed elevated"), where a deferred MSI action impersonating them
    # runs with user rights; a one-file build, so it runs from the bundle's
    # cache with the INF and catalog beside it. Not vital, so a declined
    # prompt fails nothing and Repair completes it; permanent, so uninstall
    # stays the MSI's; run at every install, so an upgrade takes back what a
    # later package took. Every build carries it: the INF and catalog are the
    # repo's signed pair, and only a signed build signs the step's executable.
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
    assert 'Fx2InfPath ?>' not in bundle
    start = bundle.index('<ExePackage Id="Fx2DriverStep"', bundle.index('<MsiPackage Id="LVP"'))
    step = bundle[start : bundle.index('</ExePackage>', start)]
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

    assert '$fx2_step_setup = Join-Path $src "dist\\lvp_fx2_driver_step_setup.exe"' in build
    assert 'if ($signing_thumbprint) { Invoke-Sign $fx2_step_setup }' in build
    defines = build[build.index('$bundle_args_fx2 = @(') :]
    defines = defines[: defines.index('\n)\n')]
    for define in (
        '"Fx2DriverStepExe=$fx2_step_setup"',
        '"Fx2InfPath=$fx2_inf"',
        '"Fx2CatPath=$fx2_cat"',
    ):
        assert define in defines, define
    assert '$bundle_args += $bundle_args_fx2' in build
    assert min_version >= 13


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
