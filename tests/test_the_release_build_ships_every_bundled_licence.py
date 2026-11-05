# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The release build ships the licence of every distribution it bundles.

The spec shipped a distribution's dist-info only when one of the bundle's
top-level names appeared in that distribution's top_level.txt. pywin32's
top_level.txt names neither ``win32`` nor ``pywin32_system32``, the folders
its binaries ship in, so the beta42 build bundled ``win32/win32api.pyd`` and
``pywin32_system32/pywintypes313.dll`` with no pywin32 licence, and the
licence gate passed. ``licence_bundle.bundled_distributions`` matches each
bundled file to the distribution whose RECORD holds it, and refuses a
site-packages file no distribution claims. PyInstaller does not run here,
so the matcher is run on a temporary site-packages.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib

import pytest

CONFIG = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / 'appBuild' / 'config'


def _load_licence_bundle():
    # The spec imports it from its folder by path; the test does the same.
    spec = importlib.util.spec_from_file_location('licence_bundle', CONFIG / 'licence_bundle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


licence_bundle = _load_licence_bundle()


def _install(site_packages: pathlib.Path, name: str, files: list[str], top_level: list[str]):
    """A distribution installed as pip installs a wheel: its files, and a
    dist-info whose RECORD lists them and whose top_level.txt says what it says."""
    dist_info = site_packages / f'{name}-1.0.dist-info'
    dist_info.mkdir(parents=True)
    for rel in files:
        path = site_packages / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x')
    (dist_info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n')
    (dist_info / 'top_level.txt').write_text('\n'.join(top_level) + '\n')
    record = [
        *files,
        f'{dist_info.name}/METADATA',
        f'{dist_info.name}/top_level.txt',
        f'{dist_info.name}/RECORD',
    ]
    (dist_info / 'RECORD').write_text(''.join(f'{rel},,\n' for rel in record))
    return importlib.metadata.PathDistribution(dist_info)


def test_a_distribution_whose_top_level_names_miss_its_folders_is_still_bundled(tmp_path):
    site_packages = tmp_path / 'site-packages'
    # Laid out as pywin32 312 is: its binaries under win32/ and
    # pywin32_system32/, which its top_level.txt does not name.
    pywin32 = _install(
        site_packages,
        'pywin32',
        ['win32/win32api.pyd', 'pywin32_system32/pywintypes313.dll', 'win32/lib/win32con.py'],
        top_level=['win32api', 'pywintypes', 'win32\\lib\\win32con'],
    )

    owners = licence_bundle.bundled_distributions(
        [
            str(site_packages / 'win32' / 'win32api.pyd'),
            str(site_packages / 'pywin32_system32' / 'pywintypes313.dll'),
        ],
        distributions=[pywin32],
        site_packages=[str(site_packages)],
    )

    assert owners == ['pywin32']


def test_a_bundled_site_packages_file_no_distribution_claims_refuses_the_build(tmp_path):
    site_packages = tmp_path / 'site-packages'
    numpy = _install(site_packages, 'numpy', ['numpy/__init__.py'], top_level=['numpy'])
    stray = site_packages / 'stray' / 'stray.pyd'
    stray.parent.mkdir()
    stray.write_bytes(b'x')

    with pytest.raises(
        SystemExit, match=r'1 bundled file\(s\) in site-packages .* first .*stray\.pyd'
    ):
        licence_bundle.bundled_distributions(
            [str(site_packages / 'numpy' / '__init__.py'), str(stray)],
            distributions=[numpy],
            site_packages=[str(site_packages)],
        )


def test_a_bundled_file_outside_site_packages_is_no_distribution_s(tmp_path):
    site_packages = tmp_path / 'site-packages'
    numpy = _install(site_packages, 'numpy', ['numpy/__init__.py'], top_level=['numpy'])
    lvp_module = tmp_path / 'repo' / 'src/_lumascope/modules' / 'scope_session.py'
    lvp_module.parent.mkdir(parents=True)
    lvp_module.write_text('')

    owners = licence_bundle.bundled_distributions(
        [str(site_packages / 'numpy' / '__init__.py'), str(lvp_module), '-'],
        distributions=[numpy],
        site_packages=[str(site_packages)],
    )

    assert owners == ['numpy']
