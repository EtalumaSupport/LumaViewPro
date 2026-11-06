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

import json
import logging
import pathlib
import re
import subprocess
import sys
import types

from _lumascope.modules import app_environment, tech_support_report


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
    """Source-text pin (the banner is written once per process, and a test
    process's first session has written it): the banner takes the probe's lines from its caller --
    it must not carry its own metadata-based camera-SDK lines. That the
    session's bring-up hands it the probe's output, and every probe line
    reaches the log, is driven in a real process by
    ``tests/test_every_session_host_logs_its_environment_once.py``."""
    logger_src = pathlib.Path('src/_lumascope/lvp_logger.py').read_text(encoding='utf-8')
    assert 'camera_sdk_lines' in logger_src, (
        'the banner must take the camera-SDK lines as a required parameter'
    )
    assert "'[LVP Main  ] ids_peak: not installed'" not in logger_src, (
        'the metadata-based ids_peak banner line cannot tell the truth in '
        'frozen builds and must be gone'
    )


def test_system_info_records_import_failure_text(monkeypatch):
    monkeypatch.setitem(sys.modules, 'ids_peak', None)

    info = tech_support_report._collect_system_info()

    assert 'ids_peak_import_error' in info, (
        f'tech-support probe must record WHY ids_peak failed to import; keys: {sorted(info)}'
    )
    assert re.search(r'\w+Error', info['ids_peak_import_error'])


# A host where the FX2 driver cannot import: its module entry is None, so
# the import raises ImportError as a missing pyusb does. Run in a child,
# since the guard runs once, as the scope module is first imported.
_FX2_ABSENT = """
import json, sys
sys.path.insert(0, 'src')
sys.modules['_lumascope.drivers.fx2driver'] = None
from _lumascope.modules.lumascope_api._lumascope import driver_availability_lines
print(json.dumps(driver_availability_lines()))
"""


def test_a_failed_fx2_driver_import_is_stated_as_a_warning_with_its_reason():
    """The FX2 driver-import guard surfaces the swallowed ImportError instead
    of silently degrading: the environment banner's driver lines carry it at
    WARNING, with the exception's words. The IDS guard is pinned in
    test_ids_absence_on_macos_is_not_a_warning.py."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-c', _FX2_ABSENT],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    lines = json.loads(done.stdout.strip().splitlines()[-1])

    fx2 = [(level, line) for level, line in lines if 'FX2 (Classic) drivers unavailable' in line]
    assert fx2 and fx2[0][0] == logging.WARNING, f'no FX2 import warning among {lines}'
    assert 'fx2driver' in fx2[0][1], f'the warning does not carry the reason: {fx2[0][1]}'
