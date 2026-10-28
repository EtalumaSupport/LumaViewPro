# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A recording's manifest names the host that recorded it.

Both recording legs write ``gather_host_provenance()`` into the manifest's
``provenance.host``, so a recording can be told apart by machine.
"""

from modules.recording_manifest import gather_host_provenance


def test_gather_host_provenance_keys():
    host = gather_host_provenance()
    assert 'hostname' in host
    assert 'os_platform' in host
    assert 'cpu_model' in host
    assert 'python_version' in host
    # Sanity: all values are non-empty strings.
    for k, v in host.items():
        assert isinstance(v, str)
        assert len(v) > 0, f'{k} should be non-empty'
