# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Both support bundles carry what every bundle carries, and say what they are in one metadata file.

``TechSupportReport._run_every_bundles_steps`` is the one place the steps both
bundles share are named, so the logs-only bundle holds nothing the full report
lacks: a logs-only bundle from the field once lacked the runtime census, added
to the full report alone. Both write ``report_metadata.json`` through one
writer, with ``report_type`` saying which bundle it is.
"""

import json
import zipfile
from unittest.mock import patch

import pytest

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.protocol_drives import run_identity
from tests.settings_fixtures import complete_settings

# Read by the one writer from every bundle, whatever steps it ran.
_COMMON_KEYS = {
    'report_type',
    'report_version',
    'generated_at',
    'serial_number',
    'lvp_version',
    'generator',
    'contents',
}


@pytest.fixture
def ls850(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850'),
        simulate=True,
    )
    yield s
    s.shutdown()


def _top_level(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return {name.split('/')[0] for name in z.namelist()}


def _metadata(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read('report_metadata.json'))


def _both_bundles(session, tmp_path):
    report = TechSupportReport(session=session)
    # A run holds the scope, so the full report's hardware steps write that they
    # were skipped: the steps under test are the ones every bundle carries.
    run = session._activity_claim.try_claim('protocol', run=run_identity())
    try:
        # a stand-in by design: the full report's own host reads (pip freeze,
        # the serial-port and USB enumeration, a disk write test) reach what the
        # test process does not; neither bundle shares them.
        with (
            patch.object(TechSupportReport, '_step_system_info', lambda self, tmp: None),
            patch.object(TechSupportReport, '_step_usb_devices', lambda self, tmp: None),
            patch.object(TechSupportReport, '_step_disk_speed', lambda self, tmp: None),
        ):
            full = report.generate(output_dir=tmp_path / 'full')
        logs_only = TechSupportReport(session=session).generate_logs_only(
            output_dir=tmp_path / 'logs'
        )
    finally:
        run.release()
    return full, logs_only


def test_the_logs_only_bundle_holds_nothing_the_full_report_lacks(ls850, tmp_path):
    full, logs_only = _both_bundles(ls850, tmp_path)

    missing = _top_level(logs_only) - _top_level(full)

    assert missing == set(), f'in the logs-only bundle, not the full report: {sorted(missing)}'
    assert {'logs', 'data', 'runtime_census', 'live_settings.json'} <= _top_level(logs_only)


def test_both_bundles_write_one_metadata_file_naming_their_kind(ls850, tmp_path):
    full, logs_only = _both_bundles(ls850, tmp_path)

    for zip_path, kind in ((full, 'tsr'), (logs_only, 'logs_only')):
        assert 'metadata.json' not in _top_level(zip_path)
        meta = _metadata(zip_path)
        assert meta.keys() >= _COMMON_KEYS, (kind, sorted(_COMMON_KEYS - meta.keys()))
        assert meta['report_type'] == kind
