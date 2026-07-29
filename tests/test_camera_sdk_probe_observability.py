# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: camera-SDK probes report import truth, with reasons.

Bug shape: the startup banner read importlib.metadata to report the
camera bindings -- but frozen (installer) builds bundle the modules with
almost no dist metadata, so the banner said "ids_peak: not installed" /
"pypylon binding: unknown" regardless of whether the binding imports.
Meanwhile the driver import guards and the tech-support probe swallowed
the actual ImportError, so when an installer build genuinely could not
import ids_peak, no log anywhere named the reason. Cost: a full
misdiagnosis cycle on a client machine (2026-07-29 installer incident).

Contract under test: probes answer by IMPORTING, and every import
failure surfaces its exception text -- banner, tech-support system info,
and the driver import guards.
"""

import pathlib
import re
import sys
import types

from modules import app_environment, tech_support_report


def _stub_ids_peak(monkeypatch, version='9.9.9-stub'):
    pkg = types.ModuleType('ids_peak')
    binding = types.ModuleType('ids_peak.ids_peak')
    binding.__version__ = version
    pkg.ids_peak = binding
    monkeypatch.setitem(sys.modules, 'ids_peak', pkg)
    monkeypatch.setitem(sys.modules, 'ids_peak.ids_peak', binding)


def test_probe_reports_importable_binding_despite_missing_metadata(monkeypatch):
    """Frozen-build shape: module imports fine, dist metadata absent."""
    _stub_ids_peak(monkeypatch)
    import importlib.metadata as imeta

    def _no_metadata(name):
        raise imeta.PackageNotFoundError(name)

    monkeypatch.setattr(imeta, 'version', _no_metadata)

    lines = app_environment.camera_sdk_probe()

    ids_lines = [line for line in lines if 'ids_peak' in line]
    assert ids_lines, f'probe must emit an ids_peak line; got: {lines}'
    assert not any('not installed' in line for line in ids_lines), (
        f'metadata absence must not be reported as not-installed when the '
        f'binding imports; got: {ids_lines}'
    )
    assert any('9.9.9-stub' in line for line in ids_lines), (
        f'probe must report the importable binding version; got: {ids_lines}'
    )


def test_probe_names_the_import_failure_reason(monkeypatch):
    """When the binding truly cannot import, the probe names why."""
    monkeypatch.setitem(sys.modules, 'ids_peak', None)

    lines = app_environment.camera_sdk_probe()

    ids_lines = [line for line in lines if 'ids_peak' in line]
    assert ids_lines, f'probe must emit an ids_peak line; got: {lines}'
    assert any(re.search(r'\w+Error', line) for line in ids_lines), (
        f'the probe must carry the import failure reason; got: {ids_lines}'
    )


def test_environment_banner_consumes_the_import_probe():
    """Source-text pin (lvp_logger is conftest-mocked, so its code cannot be
    driven in tests): the banner takes the probe's lines from its caller --
    it must not carry its own metadata-based camera-SDK lines, and the entry
    point must actually pass the probe output (the banner keeps zero
    dependency on modules/, so the glue lives at the call site)."""
    logger_src = pathlib.Path('lvp_logger.py').read_text()
    assert 'camera_sdk_lines' in logger_src, (
        'the banner must take the camera-SDK lines as a required parameter'
    )
    assert "'[LVP Main  ] ids_peak: not installed'" not in logger_src, (
        'the metadata-based ids_peak banner line cannot tell the truth in '
        'frozen builds and must be gone'
    )
    entry_src = pathlib.Path('lumaviewpro.py').read_text()
    assert re.search(r'log_environment_banner\([^)]*camera_sdk_probe\(\)', entry_src), (
        'the entry point must pass camera_sdk_probe() output into the banner'
    )


def test_system_info_records_import_failure_text(monkeypatch):
    monkeypatch.setitem(sys.modules, 'ids_peak', None)

    info = tech_support_report._collect_system_info()

    assert 'ids_peak_import_error' in info, (
        f'tech-support probe must record WHY ids_peak failed to import; keys: {sorted(info)}'
    )
    assert re.search(r'\w+Error', info['ids_peak_import_error'])


def test_driver_import_guards_log_their_reason():
    """Source-text pin: both driver-import guards in _lumascope surface the
    swallowed ImportError instead of silently degrading."""
    import modules.lumascope_api._lumascope as lumascope_mod

    src = pathlib.Path(lumascope_mod.__file__).read_text()

    ids_guard = re.search(
        r'try:\s*\n\s*from drivers\.idscamera import IDSCamera\s*\n'
        r'except ImportError([^\n]*):\n(.*?)(?=\n(?:try:|from |import |#))',
        src,
        re.DOTALL,
    )
    assert ids_guard, 'expected the guarded idscamera import'
    assert 'logger' in ids_guard.group(0), (
        'the IDS driver import guard must log the ImportError reason -- '
        'a silent guard hid an installer bundling failure entirely'
    )

    fx2_guard = re.search(
        r'try:\s*\n\s*import drivers\.fx2driver[^\n]*\n'
        r'except ImportError([^\n]*):\n(.*?)(?=\n(?:try:|from |import |#))',
        src,
        re.DOTALL,
    )
    assert fx2_guard, 'expected the guarded fx2driver import'
    assert 'logger' in fx2_guard.group(0), (
        'the FX2 driver import guard must log the ImportError reason '
        '(same silent-swallow shape as the IDS guard)'
    )
